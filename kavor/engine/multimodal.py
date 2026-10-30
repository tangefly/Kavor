from dataclasses import dataclass
import hashlib

import torch

from kavor.engine.media_io import normalize_messages


@dataclass
class MediaItem:
    """一个媒体(图片/视频)在 grid、pixel 张量里的位置,以及内容哈希。"""

    hash_id: bytes
    embed_rows: int     # merge 后的视觉 token 数 = 缓存行数
    grid_start: int     # 在(视频拆帧后的)grid 张量里的起始行
    grid_rows: int      # 占用的 grid 行数:图片 1,视频 = 时域 patch 数
    pixel_start: int    # 在 pixel 张量里的起始行
    pixel_rows: int     # 占用的 pixel 行数:图片 prod(thw),视频 t*h*w


def build_media_items(
    pixel_values: torch.Tensor | None,
    grid_thw: torch.Tensor | None,
    merge_unit: int,
    *,
    split_video: bool = False,
) -> list[MediaItem]:
    if grid_thw is None or pixel_values is None:
        return []
    items = []
    pixel_start = 0
    grid_start = 0
    for row in grid_thw:
        t, h, w = (int(value) for value in row.tolist())
        pixel_rows = t * h * w
        # 图片的 grid 张量一行就是一张图;视频拆帧后一行是一帧
        grid_rows = t if split_video else 1
        assert pixel_rows > 0, f"非法的 grid {row.tolist()}"
        digest = hashlib.sha256()
        digest.update(b"video" if split_video else b"image")
        slice_ = pixel_values[pixel_start:pixel_start + pixel_rows]
        digest.update(slice_.detach().to("cpu").contiguous().view(torch.uint8).numpy())
        digest.update(row.detach().to("cpu").contiguous().view(torch.uint8).numpy())
        items.append(MediaItem(
            hash_id=digest.digest(),
            embed_rows=pixel_rows // merge_unit,
            grid_start=grid_start,
            grid_rows=grid_rows,
            pixel_start=pixel_start,
            pixel_rows=pixel_rows,
        ))
        pixel_start += pixel_rows
        grid_start += grid_rows
    assert pixel_start == pixel_values.shape[0], (
        f"pixel 行数 {pixel_values.shape[0]} 和 grid 推出的 {pixel_start} 对不上")
    return items


def request_media_hash(image_items: list[MediaItem], video_items: list[MediaItem]) -> bytes:
    """请求级媒体摘要(prefix cache 的块哈希用),由 per-media id 依次拼出来。"""
    digest = hashlib.sha256()
    for item in image_items + video_items:
        digest.update(item.hash_id)
    return digest.digest()


def placeholder_positions(placeholder_runs) -> list[int]:
    """把 [(start, len), ...] 展开成占位符在 prompt 里的位置列表(升序)。"""
    positions = []
    for placeholder_start, placeholder_len in placeholder_runs:
        positions.extend(range(placeholder_start, placeholder_start + placeholder_len))
    return positions


@dataclass
class ModalityInputs:
    """一种模态的 processor 输出 + 引擎侧的位置/身份簿记。

    placeholder_runs 与 items 一一对应:第 i 段占位符属于第 i 个媒体项。
    """

    pixel_values: torch.Tensor | None = None
    grid_thw: torch.Tensor | None = None
    placeholder_runs: list[tuple[int, int]] | None = None
    items: list[MediaItem] | None = None

    def for_workers(self, prefill: bool) -> "ModalityInputs":
        if not prefill:
            return ModalityInputs()
        # 像素只留在 rank0;TP worker 需要每媒体 id 才能做出同样的缓存决策
        return ModalityInputs(
            grid_thw=self.grid_thw,
            placeholder_runs=self.placeholder_runs,
            items=self.items,
        )


@dataclass
class MMInputs:
    """一条序列的多模态输入:每种模态的 processor 输出 + 引擎簿记,再加 M-RoPE 状态。"""

    mm_token_type_ids: list[int] | None = None
    image: ModalityInputs | None = None
    video: ModalityInputs | None = None
    mm_hash_id: bytes | int = -1
    rope_delta: int = 0

    def for_workers(self, prefill: bool) -> "MMInputs":
        if not prefill:
            return MMInputs(rope_delta=self.rope_delta)
        return MMInputs(
            mm_token_type_ids=self.mm_token_type_ids,
            image=None if self.image is None else self.image.for_workers(prefill=True),
            video=None if self.video is None else self.video.for_workers(prefill=True),
            rope_delta=self.rope_delta,
        )


@dataclass
class PromptInputs:
    """一次请求的 prompt:token ids + 文本,以及可选的 multimodal 载荷。"""

    token_ids: list[int]
    prompt: str
    mm_inputs: MMInputs | None = None

    @classmethod
    def build_inputs(
        cls,
        tokenizer,
        processor,
        messages: list[dict],
        max_pixels: int | None = None,
        processor_kwargs: dict | None = None,
        **template_kwargs,
    ) -> "PromptInputs":
        messages, images, videos = normalize_messages(messages)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
        if processor:
            mm_kwargs = {"return_tensors": "pt"}
            if max_pixels:
                mm_kwargs["images_kwargs"] = {"max_pixels": max_pixels}
            mm_kwargs.update(processor_kwargs or {})
            inputs = processor(
                text=prompt,
                images=images or None,
                videos=videos or None,
                **mm_kwargs,
            )
            token_ids = inputs["input_ids"][0]
            mm_inputs = None
            if images or videos:
                # build_media_items 拿的是未拆分的 video grid,拆帧发生在 MMInputs 里
                merge_unit = processor.image_processor.merge_size ** 2
                image_items = build_media_items(
                    inputs.get("pixel_values"), inputs.get("image_grid_thw"), merge_unit)
                video_items = build_media_items(
                    inputs.get("pixel_values_videos"), inputs.get("video_grid_thw"),
                    merge_unit, split_video=True)
                mm_inputs = MMInputs(
                    mm_token_type_ids=inputs["mm_token_type_ids"][0].tolist(),
                    image=ModalityInputs(
                        pixel_values=inputs.get("pixel_values"),
                        grid_thw=inputs.get("image_grid_thw"),
                        placeholder_runs=find_placeholder_runs(
                            token_ids, processor.image_token_id),
                        items=image_items,
                    ) if image_items else None,
                    video=ModalityInputs(
                        pixel_values=inputs.get("pixel_values_videos"),
                        grid_thw=split_video_grid(inputs.get("video_grid_thw")),
                        placeholder_runs=find_placeholder_runs(
                            token_ids, processor.video_token_id),
                        items=video_items,
                    ) if video_items else None,
                    mm_hash_id=request_media_hash(image_items, video_items),
                )
            return cls(token_ids=token_ids.tolist(), prompt=prompt, mm_inputs=mm_inputs)
        inputs = tokenizer(prompt, return_tensors="pt")
        return cls(token_ids=inputs["input_ids"][0].tolist(), prompt=prompt)


def split_video_grid(video_grid_thw: torch.Tensor | None) -> torch.Tensor | None:
    """Expand each video row (T, H, W) into T rows of (1, H, W).

    (num_videos, 3) -> (sum(T), 3): one grid row per temporal patch, matching the
    one <|video_pad|> run per patch in the prompt. Pixel rows are already ordered
    t-major, so row order is preserved and the per-frame vision attention sees
    the same layout.
    """
    if video_grid_thw is None:
        return None
    rows = video_grid_thw.repeat_interleave(video_grid_thw[:, 0], dim=0)
    rows[:, 0] = 1
    return rows


def find_placeholder_runs(token_ids, placeholder_token_id):
    tokens = torch.as_tensor(token_ids)
    mask = tokens == placeholder_token_id

    start_mask = mask.clone()
    start_mask[1:] &= ~mask[:-1]

    end_mask = mask.clone()
    end_mask[:-1] &= ~mask[1:]

    starts = torch.where(start_mask)[0].tolist()
    ends = torch.where(end_mask)[0].tolist()

    lengths = [end - start + 1 for start, end in zip(starts, ends)]

    runs = list(zip(starts, lengths))
    return runs
