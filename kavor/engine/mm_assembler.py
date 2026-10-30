from bisect import bisect_left
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.distributed as dist

from kavor.config import Config
from kavor.engine.mm_cache import MMCache
from kavor.engine.multimodal import MediaItem, ModalityInputs, placeholder_positions
from kavor.engine.sequence import Sequence
from kavor.utils.mm_plan import plan_shards


class VisibleSlice(NamedTuple):
    """本步要拼进 inputs_embeds 的一段可见特征。"""

    seq: Sequence
    item: MediaItem
    begin: int
    count: int


class ItemEmbeds(NamedTuple):
    """一个媒体项的全部特征行:merged + 与缓存池布局一致的多份 deepstack。"""

    merged: torch.Tensor
    deepstack: list[torch.Tensor]


EmbedsByHash = dict[bytes, ItemEmbeds]


@dataclass(frozen=True)
class MMModality:
    """一种模态在 MMInputs 里的字段名,以及它在模型 kwargs 里的名字。"""

    name: str           # "image"/"video":MMInputs 的字段名,也传给模型
    label: str          # 报错信息里的中文名
    pixel_kwarg: str
    grid_kwarg: str
    ranges_kwarg: str   # pixels 路径:可见区间参数名
    embeds_kwarg: str   # embeds 路径:拼好的视觉特征参数名
    deepstack_kwarg: str


MM_MODALITIES = (
    MMModality(
        name="image",
        label="图片",
        pixel_kwarg="pixel_values",
        grid_kwarg="image_grid_thw",
        ranges_kwarg="image_visible_ranges",
        embeds_kwarg="image_embeds",
        deepstack_kwarg="deepstack_image_embeds",
    ),
    MMModality(
        name="video",
        label="视频",
        pixel_kwarg="pixel_values_videos",
        grid_kwarg="video_grid_thw",
        ranges_kwarg="video_visible_ranges",
        embeds_kwarg="video_embeds",
        deepstack_kwarg="deepstack_video_embeds",
    ),
)


def item_spans(
    seq: Sequence,
    inputs: ModalityInputs,
    spec: MMModality,
) -> list[tuple[MediaItem, int, int]]:
    """每个媒体在本步 prefill 窗口里可见的行区间 [begin, begin+count)。

    用占位符在 input_ids 里的真实位置算,而不是假设「一行 grid 对应一段占位符
    run」:视频拆帧后 grid 行数和占位符段数不一定相等。
    """
    if inputs.placeholder_runs is None or inputs.grid_thw is None or not inputs.items:
        raise ValueError(
            f"多模态输入缺少 {spec.label}的占位符区间、grid 或媒体项:"
            "只有 PromptInputs.build_inputs 会填它们,手工构造 MMInputs 时不能只带像素")
    positions = placeholder_positions(inputs.placeholder_runs)
    total = sum(item.embed_rows for item in inputs.items)
    if len(positions) != total:
        raise ValueError(
            f"{spec.label}占位符有 {len(positions)} 个,媒体项推出 {total} 个:"
            "processor 出的 input_ids 和 grid 不是同一份输入")
    grid_rows = sum(item.grid_rows for item in inputs.items)
    if grid_rows != inputs.grid_thw.shape[0]:
        raise ValueError(
            f"{spec.label}媒体项占用 {grid_rows} 行 grid,实际有 {inputs.grid_thw.shape[0]} 行")
    start = seq.num_cached_tokens
    end = start + seq.num_scheduled_tokens
    spans = []
    offset = 0
    for item in inputs.items:
        item_positions = positions[offset:offset + item.embed_rows]
        offset += item.embed_rows
        begin = bisect_left(item_positions, start)
        finish = bisect_left(item_positions, end)
        spans.append((item, begin, finish - begin))
    return spans


class MMInputAssembler:
    """为一次 forward 组装多模态 kwargs,并持有这一步的 pin 账本。"""

    def __init__(
        self,
        model: torch.nn.Module,
        config: Config,
        mm_cache: MMCache | None,
        rank: int,
        world_size: int,
        enable_log: bool,
    ):
        self.model = model
        self.config = config
        self.mm_cache = mm_cache
        self.rank = rank
        self.world_size = world_size
        self.enable_log = enable_log
        self.pins: list[bytes] = []     # 本步 pin 住的媒体,forward 之后统一释放

    @torch.inference_mode()
    def prepare(self, seqs: list[Sequence]) -> dict:
        mm_seqs = [seq for seq in seqs if seq.is_multimodal]
        if not mm_seqs:
            return {}
        if self.mm_cache is None:
            return self.prepare_pixel_inputs(mm_seqs)
        return self.prepare_embed_inputs(mm_seqs)

    def release_pins(self):
        if not self.pins:
            return
        for hash_id in self.pins:
            self.mm_cache.release(hash_id)
        self.pins.clear()

    @staticmethod
    def modality_inputs(seq: Sequence, spec: MMModality) -> ModalityInputs | None:
        mm_inputs = seq.mm_inputs
        return None if mm_inputs is None else getattr(mm_inputs, spec.name)

    # embeds 路径

    def prepare_embed_inputs(self, seqs: list[Sequence]) -> dict:
        kwargs = {}
        has_media = False
        for spec in MM_MODALITIES:
            # Step 1: visible row spans + deduplicated items for this step.
            rows, items = self.collect_visible_rows(seqs, spec)
            if not rows:
                continue
            has_media = True
            # Step 2: pool hits are read in place; misses are computed and written back.
            resolved, hit_ids = self.load_or_compute_embeddings(items, spec)
            # Step 3: per-item hit/miss stats (engine side only).
            for row in rows:
                if row.seq.stats is not None:
                    row.seq.stats.record_mm_item(
                        row.item.hash_id, row.item.pixel_start, row.item.embed_rows,
                        row.item.hash_id in hit_ids)
            # Step 4: concatenate visible rows in placeholder order.
            embeds = self.splice_visible_rows(rows, resolved)
            kwargs[spec.embeds_kwarg] = embeds.merged
            kwargs[spec.deepstack_kwarg] = embeds.deepstack
        # Step 5: log cache counters once per step.
        if has_media and self.enable_log:
            self.mm_cache.log_step()
        return kwargs

    def collect_visible_rows(
        self, seqs: list[Sequence], spec: MMModality
    ) -> tuple[list[VisibleSlice], dict[bytes, tuple[Sequence, MediaItem]]]:
        """按序列、媒体原始顺序收集本步要放进 inputs_embeds 的行,并按 hash 去重媒体项。"""
        rows: list[VisibleSlice] = []
        items: dict[bytes, tuple[Sequence, MediaItem]] = {}
        for seq in seqs:
            inputs = self.modality_inputs(seq, spec)
            if inputs is None or not inputs.items:
                continue
            for item, begin, count in item_spans(seq, inputs, spec):
                if count == 0:      # 整段媒体都在 prefix cache 里,本步不需要
                    continue
                rows.append(VisibleSlice(seq, item, begin, count))
                items.setdefault(item.hash_id, (seq, item))
        return rows, items

    def load_or_compute_embeddings(
        self, items: dict[bytes, tuple[Sequence, MediaItem]], spec: MMModality
    ) -> tuple[EmbedsByHash, set[bytes]]:
        """命中直接读池;未命中的像素只把这一批 miss 送去视觉塔,算完写池。

        返回 (embeds_by_hash, hit_ids):hit_ids 是本步开始时已在池里的项,只用于
        日志区分命中与「本步刚算完」。
        """
        # Step 1: pin pool hits and read their full rows; queue misses.
        resolved: EmbedsByHash = {}
        hit_ids: set[bytes] = set()
        pending: list[tuple[bytes, Sequence, MediaItem]] = []
        for hash_id, (seq, item) in items.items():
            if self.mm_cache.acquire(hash_id) is not None:
                self.pins.append(hash_id)
                resolved[hash_id] = ItemEmbeds(*self.mm_cache.read(hash_id))
                hit_ids.add(hash_id)
            else:
                pending.append((hash_id, seq, item))
        if not pending:
            return resolved, hit_ids

        # Step 2: gather full rows for each miss; pixels live on rank0 only.
        grids, pixels = [], []
        for _, seq, item in pending:
            inputs = self.modality_inputs(seq, spec)
            grids.append(inputs.grid_thw[item.grid_start:item.grid_start + item.grid_rows])
            if inputs.pixel_values is not None:
                pixels.append(
                    inputs.pixel_values[item.pixel_start:item.pixel_start + item.pixel_rows])
        grid = torch.cat(grids, dim=0).to(
            device=self.mm_cache.merged.device, non_blocking=True)
        scattered = self.scatter_pixels_to_ranks(pixels, grid)
        # Step 3: run the vision tower once for all misses.
        merged, deepstack = self.model.get_mm_embeddings(scattered, grid, spec.name)
        assert len(deepstack) == self.mm_cache.spec.num_deepstack, (
            f"模型返回 {len(deepstack)} 份 deepstack 特征,缓存池按 "
            f"{self.mm_cache.spec.num_deepstack} 份规划的:检查 vision_config")

        # Step 4: split the batch output back per item.
        sizes = [item.embed_rows for _, _, item in pending]
        merged_parts = torch.split(merged, sizes)
        deepstack_parts = [torch.split(part, sizes) for part in deepstack]
        # Step 5: write items to the pool; items that don't fit are used directly.
        for index, (hash_id, _, _) in enumerate(pending):
            part = merged_parts[index]
            ds_part = [parts[index] for parts in deepstack_parts]
            if self.mm_cache.put(hash_id, part, ds_part) is None:
                # too big for the pool (or no free space): use this tensor directly
                resolved[hash_id] = ItemEmbeds(part, ds_part)
            else:
                self.pins.append(hash_id)
                resolved[hash_id] = ItemEmbeds(*self.mm_cache.read(hash_id))
        return resolved, hit_ids

    def splice_visible_rows(
        self, rows: list[VisibleSlice], resolved: EmbedsByHash
    ) -> ItemEmbeds:
        """按占位符顺序拼出本步的视觉特征:每段只取自己的可见行。"""
        merged_parts: list[torch.Tensor] = []
        deepstack_parts: list[list[torch.Tensor]] = [
            [] for _ in range(self.mm_cache.spec.num_deepstack)]
        for row in rows:
            embeds = resolved[row.item.hash_id]
            merged_parts.append(embeds.merged[row.begin:row.begin + row.count])
            for parts, part in zip(deepstack_parts, embeds.deepstack):
                parts.append(part[row.begin:row.begin + row.count])
        return ItemEmbeds(
            torch.cat(merged_parts, dim=0),
            [torch.cat(parts, dim=0) for parts in deepstack_parts])

    # pixels 路径

    def prepare_pixel_inputs(self, seqs: list[Sequence]) -> dict:
        kwargs = {}
        for spec in MM_MODALITIES:
            inputs_list: list[ModalityInputs] = []
            for seq in seqs:
                inputs = self.modality_inputs(seq, spec)
                if inputs is not None and inputs.grid_thw is not None:
                    inputs_list.append(inputs)
            if not inputs_list:
                continue
            pixels = [inputs.pixel_values for inputs in inputs_list
                      if inputs.pixel_values is not None]
            grid = torch.cat([inputs.grid_thw for inputs in inputs_list], dim=0).cuda(
                non_blocking=True)
            kwargs[spec.grid_kwarg] = grid
            kwargs[spec.pixel_kwarg] = self.scatter_pixels_to_ranks(pixels, grid)
            kwargs[spec.ranges_kwarg] = self.compute_visible_ranges(seqs, grid, spec)
        return kwargs

    def compute_visible_ranges(self, seqs: list[Sequence], grid: torch.Tensor,
                               spec: MMModality) -> list[tuple[int, int]] | None:
        """每行 grid(图片一张、视频一帧)在本步窗口里的可见区间,按批拍平。

        全部可见时返回 None,让模型跳过行选择。
        """
        merge_unit = self.config.vision_config.spatial_merge_size ** 2
        ranges, lengths = [], []
        for seq in seqs:
            inputs = self.modality_inputs(seq, spec)
            if inputs is None or inputs.grid_thw is None:
                continue
            rows = (inputs.grid_thw.prod(-1) // merge_unit).tolist()
            runs = inputs.placeholder_runs
            if runs is None:
                raise ValueError(
                    f"多模态输入缺少 {spec.label}的占位符区间:只有 PromptInputs.build_inputs "
                    "会填它,手工构造 MMInputs 时不能直接带像素张量")
            if len(runs) != len(rows):
                raise ValueError(
                    f"{spec.label}占位符区间有 {len(runs)} 段,"
                    f"{spec.grid_kwarg} 推出 {len(rows)} 行,对不上")
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            for (placeholder_start, placeholder_len), row in zip(runs, rows):
                if placeholder_len != row:
                    raise ValueError(
                        f"{spec.label}占位符个数 {placeholder_len} 和 grid 推出的特征行数 "
                        f"{row} 对不上:processor 出的 input_ids 和 grid 不是同一份输入")
                begin = min(max(start - placeholder_start, 0), placeholder_len)
                finish = min(max(end - placeholder_start, 0), placeholder_len)
                ranges.append((begin, max(finish - begin, 0)))
                lengths.append(placeholder_len)
        assert len(ranges) == grid.shape[0], (
            f"算出 {len(ranges)} 段{spec.label},和 grid 的 {grid.shape[0]} 行对不上")
        if all(count == length for (_, count), length in zip(ranges, lengths)):
            return None
        return ranges

    def scatter_pixels_to_ranks(self, pixels: list[torch.Tensor],
                                grid: torch.Tensor) -> torch.Tensor:
        world = self.world_size
        rows = grid.prod(-1).tolist()
        counts = [shard.row_end - shard.row_start for shard in plan_shards(rows, world)]
        vision_config = self.config.vision_config
        patch_dim = (vision_config.in_channels * vision_config.temporal_patch_size
                     * vision_config.patch_size ** 2)
        dtype = self.model.mm_pixel_dtype

        if world == 1:
            assert pixels, "有 grid 却没有像素张量"
            return torch.cat(pixels, dim=0).to(device="cuda", dtype=dtype)

        device = torch.device("cuda", self.rank)
        if self.rank == 0:
            assert pixels, "rank0 上有 grid 却没有像素张量"
            # 按视觉塔的 dtype 传:塔里第一件事就是这个转换,提前做结果逐位相同、字节减半
            src = torch.cat(pixels, dim=0).to(device=device, dtype=dtype)
        else:
            src = torch.empty(0, patch_dim, dtype=dtype, device=device)
        mine = torch.empty(counts[self.rank], patch_dim, dtype=dtype, device=device)
        dist.all_to_all_single(
            mine, src,
            output_split_sizes=[counts[self.rank] if i == 0 else 0 for i in range(world)],
            input_split_sizes=counts if self.rank == 0 else [0] * world,
        )
        return mine
