import math
import pickle
from dataclasses import dataclass

import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from kavor.config import Config
from kavor.engine.multimodal import MultiModalInputs
from kavor.engine.sequence import Sequence
from kavor.layers.sampler import Sampler
from kavor.utils.context import set_context, get_context, reset_context
from kavor.utils.logger import logger
from kavor.utils.mm_plan import plan_shards
from kavor.models import AutoModelForCausalLM


@dataclass(frozen=True)
class CacheLayout:
    """页池的"图纸":只描述尺寸和分组,不持有任何显存。"""

    attn_slots: list[tuple[torch.nn.Module, int]]   # [(attention 模块, 第几个 KV 槽)]
    linear_modules: list[torch.nn.Module]           # 所有线性注意力层
    num_pools: int                                  # 池数 = 每组最多几层
    num_blocks: int                                 # 每个池的行数(块 id 的范围)
    block_size: int                                 # 一个 KV 块装多少 token
    page_bytes: int                                 # 每层每页的字节数
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    dtype_size: int
    num_state_groups: int                           # 线性层分成几个状态块组
    max_state_bytes: int                            # 最重的一层线性状态占多少字节


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.enable_log = self.rank == 0

        dist.init_process_group(
            "nccl", f"tcp://localhost:{config.rdzv_port}",
            world_size=self.world_size, rank=rank, device_id=torch.device(f"cuda:{rank}"))
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.text_config.dtype)
        torch.set_default_device("cuda")
        Sequence.block_size = config.kvcache_block_size
        self.model = AutoModelForCausalLM.from_pretrained(hf_config, use_tqdm=self.enable_log)
        self.sampler = Sampler()
        self.linear_padding_slot = None
        self.num_state_groups = 0
        self.linear_state_len = []
        # self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                try:
                    self.shm = SharedMemory(name="kavor")
                    self.shm.close()
                    self.shm.unlink()
                    logger.warning("Removed stale shared memory.")
                except FileNotFoundError:
                    pass
                self.shm = SharedMemory(name="kavor", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="kavor")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        if n + 4 > len(self.shm.buf):
            raise RuntimeError(
                f"要发给 TP worker 的数据有 {n / 2**20:.1f} MB,超过共享内存 "
                f"{len(self.shm.buf) / 2**20:.0f} MB:args 里不要夹带大张量"
                "(多模态的像素张量应该在 MultiModalInputs.for_workers 里就被丢掉,"
                "改走 _scatter_mm 那条 NCCL 数据面)")
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence(MultiModalInputs([0] * seq_len, "")) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    @staticmethod
    def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        strides = []
        acc = 1
        for size in reversed(shape):
            strides.append(acc)
            acc *= size
        return tuple(reversed(strides))

    @staticmethod
    def _linear_state_nbytes(module) -> int:
        """一层线性注意力的状态页要占多少字节(conv_state 在前,recurrent_state 紧随其后)。"""
        conv_shape, rec_shape = module.state_shapes
        conv_dtype, rec_dtype = module.state_dtypes
        conv_bytes = math.prod(conv_shape) * conv_dtype.itemsize
        rec_offset = (conv_bytes + rec_dtype.itemsize - 1) // rec_dtype.itemsize
        return rec_offset * rec_dtype.itemsize + math.prod(rec_shape) * rec_dtype.itemsize

    def _all_reduce_int(self, value: int, op) -> int:
        """各 rank 必须就 block_size/num_blocks 达成一致,否则 block_table 会对不上。"""
        if self.world_size == 1:
            return value
        t = torch.tensor([value], dtype=torch.int64, device="cuda")
        dist.all_reduce(t, op=op)
        return int(t.item())

    def _collect_attention_slots(self) -> list[tuple[torch.nn.Module, int]]:
        """按模型顺序列出每个 attention KV 槽;每个槽将来独占一个层页池。"""
        slots = []
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                slots.extend((module, i) for i in range(module.num_kv_slots))
        assert slots, "纯线性注意力模型还没有页池布局,暂不支持"
        return slots

    def _collect_linear_layers(self) -> list[torch.nn.Module]:
        return [m for m in self.model.modules() if hasattr(m, "state_shapes")]

    def _choose_block_size(self, max_state_bytes: int, attn_bytes_per_token: int) -> int:
        """调大 attention 的 logical block_size,让一层 attention 的一页装得下一层线性状态。

        装不下的差额用状态页 padding 补齐(见 _bind_state_page)。
        FlashAttention 的分页 KV 要求 page_block_size 是 256 的倍数。
        """
        block_size = self.config.kvcache_block_size
        if max_state_bytes == 0:
            return block_size
        need = (max_state_bytes + attn_bytes_per_token - 1) // attn_bytes_per_token
        block_size = max(block_size, ((need + 255) // 256) * 256)
        block_size = self._all_reduce_int(block_size, dist.ReduceOp.MAX)
        self.config.kvcache_block_size = block_size
        return block_size

    def _probe_num_blocks(self, num_pools: int, page_bytes: int) -> int:
        """按剩余显存算页池行数;每块在所有池里各占一页。"""
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        budget = int(total * self.config.gpu_memory_utilization - used - peak + current)
        num_blocks = self._all_reduce_int(
            budget // (num_pools * page_bytes), dist.ReduceOp.MIN)
        assert num_blocks >= 2, (
            f"页池只有 {num_blocks} 块:至少要 1 块给请求、1 块给 CUDA graph 占位。"
            "调大 gpu_memory_utilization 或减小 max_model_len")
        return num_blocks

    def _plan_cache_layout(self) -> CacheLayout:
        text_config = self.config.text_config
        num_kv_heads = text_config.num_key_value_heads // self.world_size
        head_dim = getattr(
            text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)
        dtype_size = text_config.dtype.itemsize

        attn_slots = self._collect_attention_slots()
        linear_modules = self._collect_linear_layers()
        max_state_bytes = max(
            (self._linear_state_nbytes(m) for m in linear_modules), default=0)

        attn_bytes_per_token = 2 * num_kv_heads * head_dim * dtype_size
        block_size = self._choose_block_size(max_state_bytes, attn_bytes_per_token)
        self.block_size = block_size
        Sequence.block_size = block_size

        page_bytes = attn_bytes_per_token * block_size
        assert max_state_bytes <= page_bytes, (
            f"线性状态 {max_state_bytes} B 装不进 attention 的一页 {page_bytes} B")

        num_pools = len(attn_slots)
        num_state_groups = ((len(linear_modules) + num_pools - 1) // num_pools
                            if linear_modules else 0)
        num_blocks = self._probe_num_blocks(num_pools, page_bytes)
        self.config.num_kvcache_blocks = num_blocks

        return CacheLayout(
            attn_slots=attn_slots,
            linear_modules=linear_modules,
            num_pools=num_pools,
            num_blocks=num_blocks,
            block_size=block_size,
            page_bytes=page_bytes,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=text_config.dtype,
            dtype_size=dtype_size,
            num_state_groups=num_state_groups,
            max_state_bytes=max_state_bytes,
        )

    def allocate_kv_cache(self):
        layout = self._plan_cache_layout()

        self.cache_pools = torch.empty(
            layout.num_pools, layout.num_blocks, layout.page_bytes, dtype=torch.uint8)

        self._bind_attention_pages(layout)
        self._bind_linear_state_pages(layout)
        self._activate_linear_state(layout)
        self._log_cache_layout(layout)

    def _bind_attention_pages(self, layout: CacheLayout):
        """第 i 个 attention 槽 ↔ 第 i 个池;K 在页的前半,V 在后半。"""
        block_elems = layout.block_size * layout.num_kv_heads * layout.head_dim
        block_stride = layout.page_bytes // layout.dtype_size
        inner = (layout.num_kv_heads * layout.head_dim, layout.head_dim, 1)
        shape = (layout.num_blocks, layout.block_size, layout.num_kv_heads, layout.head_dim)
        for pool, (module, slot) in enumerate(layout.attn_slots):
            flat = self.cache_pools[pool].view(layout.dtype)
            base = flat.storage_offset()
            module.k_cache[slot] = flat.as_strided(shape, (block_stride, *inner), base)
            module.v_cache[slot] = flat.as_strided(
                shape, (block_stride, *inner), base + block_elems)

    def _bind_linear_state_pages(self, layout: CacheLayout):
        """第 j 个线性层 ↔ 第 j % num_pools 个池;同组 8 层共用同一个状态 block id。"""
        for j, module in enumerate(layout.linear_modules):
            module.state_group = j // layout.num_pools
            self._bind_state_page(module, self.cache_pools[j % layout.num_pools], layout)

    def _bind_state_page(self, module, pool: torch.Tensor, layout: CacheLayout):
        conv_shape, rec_shape = module.state_shapes
        conv_dtype, rec_dtype = module.state_dtypes
        conv_bytes = math.prod(conv_shape) * conv_dtype.itemsize
        rec_offset = (conv_bytes + rec_dtype.itemsize - 1) // rec_dtype.itemsize
        state_bytes = rec_offset * rec_dtype.itemsize + math.prod(rec_shape) * rec_dtype.itemsize
        assert state_bytes <= layout.page_bytes, (
            f"线性状态 {state_bytes} B 装不进一页 {layout.page_bytes} B")
        assert (layout.page_bytes % conv_dtype.itemsize == 0
                and layout.page_bytes % rec_dtype.itemsize == 0)

        conv_flat = pool.view(conv_dtype)
        rec_flat = pool.view(rec_dtype)
        module.conv_pool = conv_flat.as_strided(
            (layout.num_blocks, *conv_shape),
            (layout.page_bytes // conv_dtype.itemsize, *self._contiguous_strides(conv_shape)),
            conv_flat.storage_offset())
        module.recurrent_pool = rec_flat.as_strided(
            (layout.num_blocks, *rec_shape),
            (layout.page_bytes // rec_dtype.itemsize, *self._contiguous_strides(rec_shape)),
            rec_flat.storage_offset() + rec_offset)

    def _activate_linear_state(self, layout: CacheLayout):
        """把状态分组/占位块 id 发布给调度器和模型侧。"""
        config = self.config
        has_linear = bool(layout.linear_modules)
        self.num_state_groups = layout.num_state_groups
        config.has_linear_attention = has_linear
        config.num_linear_state_groups = layout.num_state_groups
        self.linear_padding_slot = layout.num_blocks - 1 if has_linear else None
        self.linear_state_len = [-1] * layout.num_blocks
        if not has_linear:
            return
        self.cache_pools[:, self.linear_padding_slot].zero_()
        self.model.linear_padding_slot = self.linear_padding_slot

    def _log_cache_layout(self, layout: CacheLayout):
        if not self.enable_log:
            return
        logger.info(
            "统一页池: %d 块 × %d 层页 × %.2f MB = %.2f GB (block_size=%d, 每 token %.1f KB)",
            layout.num_blocks, layout.num_pools, layout.page_bytes / 2**20,
            layout.num_blocks * layout.num_pools * layout.page_bytes / 2**30,
            layout.block_size, layout.page_bytes / layout.block_size / 1024,
        )
        if layout.linear_modules:
            logger.info(
                "线性注意力状态: 每层 %.2f MB × %d 层, 分 %d 个块组(每组 %d 层);"
                " 状态页 padding %.2f MB/层",
                layout.max_state_bytes / 2**20, len(layout.linear_modules),
                layout.num_state_groups, layout.num_pools,
                (layout.page_bytes - layout.max_state_bytes) / 2**20,
            )

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_linear_slot_ids(self, seqs: list[Sequence]) -> torch.Tensor | None:
        """这一批序列的状态块 id,形状 [B, num_state_groups](没有线性注意力层时返回 None)。

        和 block_table 一样属于 batch 元数据:模型侧只拿它做索引,不查 dict、不做 GPU->CPU 同步。
        """
        if self.linear_padding_slot is None:
            return None
        rows = []
        for seq in seqs:
            ids = seq.state_block_ids
            if ids is None:
                # warmup 直接造的假序列没有经过调度器,统一指向占位块
                assert not seq.block_table, (
                    f"序列 {seq.seq_id} 没有状态块就要跑模型了:状态块是在 prefill 调度时分配的")
                rows.append([self.linear_padding_slot] * self.num_state_groups)
                continue
            assert len(ids) == self.num_state_groups, (
                f"序列 {seq.seq_id} 有 {len(ids)} 个状态块,模型需要 {self.num_state_groups} 个")
            assert all(0 <= i < self.linear_padding_slot for i in ids), (
                f"状态块 id 越界({ids}):合法范围是 [0, {self.linear_padding_slot}),"
                "最后一块是 CUDA graph 的占位块")
            assert len(set(ids)) == len(ids), (
                f"同一条序列里出现了重复的状态块({ids}):不同块组共用一页会互相覆盖")
            rows.append(ids)
        return torch.tensor(rows, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)

    def check_linear_states(self, seqs: list[Sequence]):
        """decode 前的自检:每条序列的状态块应该正好停在 len(seq) - 1 个 token 上。

        块 id 现在由 BlockManager 统一分配,复用/错位都会在这里当场炸出来,
        而不是悄悄算出胡话。
        """
        if self.linear_padding_slot is None:
            return
        for seq in seqs:
            assert seq.state_block_ids is not None, f"序列 {seq.seq_id} 没有状态块就要 decode 了"
            got, want = self.linear_state_len[seq.state_block_ids[0]], len(seq) - 1
            assert got == want, (
                f"序列 {seq.seq_id} 的状态和它自己错位了:块 {seq.state_block_ids} 停在第 {got} 个 token,"
                f"这一步要处理的却是第 {want} 个 token"
                "(prefill 没跑到?prefix cache 命中跳过了前缀?被抢占后没重新 prefill?)")

    def _record_linear_states(self, seqs: list[Sequence]):
        """记下每条序列的状态块吃到第几个 token。纯 host 记账,不碰 GPU,也不引入同步。"""
        if self.linear_padding_slot is None:
            return
        for seq in seqs:
            if seq.state_block_ids is None:
                continue
            for block_id in seq.state_block_ids:
                self.linear_state_len[block_id] = seq.num_cached_tokens + seq.num_scheduled_tokens

    def position_ids(self, seq: Sequence, start: int, end: int) -> torch.Tensor:
        if seq.mm is None:
            return torch.arange(start, end).unsqueeze(0).expand(3, -1)
        return self.model.mm_position_ids(seq)[:, start:end]

    def prepare_mm_inputs(self, seqs: list[Sequence]) -> dict:
        mm_seqs = [seq for seq in seqs if seq.mm is not None]
        if not mm_seqs:
            return {}
        kwargs = {}
        for pixel_key, grid_key in (("pixel_values", "image_grid_thw"),
                                    ("pixel_values_videos", "video_grid_thw")):
            grids = [getattr(seq.mm, grid_key) for seq in mm_seqs
                     if getattr(seq.mm, grid_key) is not None]
            if not grids:
                continue
            pixels = [getattr(seq.mm, pixel_key) for seq in mm_seqs
                      if getattr(seq.mm, pixel_key) is not None]
            kwargs[grid_key] = torch.cat(grids, dim=0).cuda(non_blocking=True)
            kwargs[pixel_key] = self._scatter_mm(pixels, kwargs[grid_key])
        return kwargs

    def _scatter_mm(self, pixels: list[torch.Tensor], grid: torch.Tensor) -> torch.Tensor:
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

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        has_mm = any(seq.mm is not None for seq in seqs)
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            if has_mm:
                positions.append(self.position_ids(seq, start, end))
            else:
                positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        if has_mm:
            positions = torch.cat(positions, dim=1).cuda(non_blocking=True)     # (3, 总 token 数)
        else:
            positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        linear_slot_ids = self.prepare_linear_slot_ids(seqs)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, linear_slot_ids)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        self.check_linear_states(seqs)
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1 + (seq.mm.rope_delta if seq.mm is not None else 0))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        linear_slot_ids = self.prepare_linear_slot_ids(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens,
                    block_tables=block_tables, linear_slot_ids=linear_slot_ids)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool, mm_kwargs: dict | None = None):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            logits = self.model(input_ids, positions, **(mm_kwargs or {}))
            return self.model.compute_logits(logits)
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            if graph_vars["linear_slot_ids"] is not None:
                graph_vars["linear_slot_ids"].fill_(self.linear_padding_slot)
                graph_vars["linear_slot_ids"][:bs] = context.linear_slot_ids
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        mm_kwargs = self.prepare_mm_inputs(seqs) if is_prefill else {}
        logits = self.run_model(input_ids, positions, is_prefill, mm_kwargs)
        self._record_linear_states(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        text_config = config.text_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, text_config.hidden_size)
        linear_slot_ids = (None if self.linear_padding_slot is None else
                           torch.full((max_bs, self.num_state_groups),
                                      self.linear_padding_slot, dtype=torch.int64))
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs],
                        block_tables=block_tables[:bs],
                        linear_slot_ids=None if linear_slot_ids is None else linear_slot_ids[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            linear_slot_ids=linear_slot_ids,
            outputs=outputs,
        )
