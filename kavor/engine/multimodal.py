from dataclasses import dataclass
import base64
import io
import urllib.request

import numpy as np
import torch
from PIL import Image


MAX_IMAGE_BYTES = 32 * 2**20

@dataclass
class MultiModalInputs:

    token_ids: list[int]
    prompt: str
    mm_token_type_ids: list[int] | None = None
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    pixel_values_videos: torch.Tensor | None = None
    video_grid_thw: torch.Tensor | None = None
    rope_delta: int = 0                             # prefill 时由模型回填,解码步的位置要加它
    def for_workers(self, prefill: bool) -> "MultiModalInputs":
        if prefill:
            return MultiModalInputs(
                token_ids=[],
                prompt="",
                mm_token_type_ids=self.mm_token_type_ids,
                image_grid_thw=self.image_grid_thw,
                video_grid_thw=self.video_grid_thw,
                rope_delta=self.rope_delta,
            )
        return MultiModalInputs(token_ids=[], prompt="", rope_delta=self.rope_delta)

    @classmethod
    def build_mm_inputs(
        cls,
        tokenizer,
        processor,
        messages: list[dict],
        max_pixels: int | None = None,
        **template_kwargs,
    ) -> "MultiModalInputs":
        """messages 是 OpenAI 风格的原始消息(可以混着 str 和 list 两种 content)。
        template_kwargs 直接转给 apply_chat_template(比如 tools=...)。
        """
        messages, images, videos = normalize_content(messages)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
        inputs = processor(
            text=prompt,
            images=images or None,
            videos=videos or None,
            return_tensors="pt",
            **({"images_kwargs": {"max_pixels": max_pixels}} if max_pixels else {}),
        )
        _check_placeholders(inputs, processor, len(images), len(videos))
        return cls(
            token_ids=inputs["input_ids"][0].tolist(),
            prompt=prompt,
            mm_token_type_ids=inputs["mm_token_type_ids"][0].tolist(),
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_values_videos=inputs.get("pixel_values_videos"),
            video_grid_thw=inputs.get("video_grid_thw"),
        )


def _content_of(message):
    """消息可能是 dict(库内调用),也可能是 pydantic 模型(API 层),两种都要能取到 content。"""
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def has_media(messages: list[dict]) -> bool:
    """请求里有没有图/视频 —— API 层用它决定走不走多模态分支。

    文本消息的 content 是 str,这里必须能安全地跳过(不能直接 for item in content)。
    """
    for message in messages:
        content = _content_of(message)
        if not isinstance(content, (list, tuple)):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if "image" in item or "image_url" in item or item.get("type") in ("image", "image_url", "input_image"):
                return True
            if "video" in item or item.get("type") in ("video", "video_url"):
                return True
    return False


def normalize_content(messages: list[dict]) -> tuple[list[dict], list[Image.Image], list]:
    """洗消息 + 抽媒体,返回 (给模板用的 messages, images, videos)。

    模板的判定逻辑只看 key 存不存在(值被忽略),真正喂给 processor 的是这里抽出来的对象。
    支持这几种写法:
      {"type": "image_url", "image_url": {"url": ...}}    OpenAI / vLLM 标准
      {"type": "image",     "image": ...}                 Qwen 原生
      {"type": "video_url", "video_url": {"url": ...}}    vLLM(模板不认,这里转成 "video")
      {"type": "video",     "video": ...}                 Qwen 原生
    其它 item(type=="text" 或带 "text" 键)原样保留。
    """
    out_messages: list[dict] = []
    images: list[Image.Image] = []
    videos: list = []

    for message in messages:
        content = message.get("content")
        if not isinstance(content, (list, tuple)):
            out_messages.append(dict(message))      # 纯字符串 content,原样过
            continue

        new_content: list[dict] = []
        for item in content:
            if not isinstance(item, dict):
                new_content.append(item)
                continue

            if "image_url" in item or item.get("type") in ("image", "image_url", "input_image"):
                src = item.get("image_url") or item.get("image") or item.get("input_image")
                if isinstance(src, dict):
                    src = src.get("url")
                image = load_image(src)
                images.append(image)
                new_content.append({"type": "image", "image": image})
            elif "video_url" in item or item.get("type") in ("video", "video_url"):
                src = item.get("video_url") or item.get("video")
                if isinstance(src, dict):
                    src = src.get("url")
                frames = load_video(src)
                videos.append(frames)
                new_content.append({"type": "video", "video": frames})
            else:
                new_content.append(item)

        out_messages.append({**message, "content": new_content})

    return out_messages, images, videos


def load_image(src) -> Image.Image:
    """data:image/png;base64,... / http(s):// → PIL.Image(RGB)。已经是 PIL 的直接返回。

    只放行这两种:本地路径和 file:// 一律拒绝 —— 这是对外的 HTTP 服务,放开路径
    等于把服务器变成任意文件读取器。
    """
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    if not isinstance(src, str):
        raise ValueError(f"看不懂的图片输入: {type(src).__name__}")

    if src.startswith("data:"):
        head, _, b64 = src.partition(",")
        if not b64 or "base64" not in head:
            raise ValueError(f"只支持 base64 编码的 data URL: {head[:48]}")
        try:
            raw = base64.b64decode(b64, validate=True)
        except Exception as e:
            raise ValueError(f"data URL 的 base64 解码失败: {e}") from e
        return Image.open(io.BytesIO(raw)).convert("RGB")

    if src.startswith(("http://", "https://")):
        return _fetch_url(src)

    raise ValueError(f"不支持的图片地址(只认 data: 和 http(s)://): {src[:64]}")


def image_from_path(path: str) -> Image.Image:
    """本地文件 → PIL.Image,给可信调用方(命令行/脚本)用。

    写进请求体里的本地路径仍然被 load_image 拒绝:对外服务放开路径等于任意文件读取。
    """
    return Image.open(path).convert("RGB")


def _fetch_url(url: str) -> Image.Image:
    # 走 stdlib,不引额外依赖。注意:这里会去访问任意 URL,只适合内网可信部署。
    request = urllib.request.Request(url, headers={"User-Agent": "kavor"})
    with urllib.request.urlopen(request, timeout=10) as response:      # noqa: S310
        raw = response.read(MAX_IMAGE_BYTES + 1)
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"图片超过 {MAX_IMAGE_BYTES // 2**20} MiB 上限")
    return Image.open(io.BytesIO(raw)).convert("RGB")


def load_video(src):
    """视频 → 帧列表。已经是帧数组的直接返回。

    这个环境里 torchcodec / decord / av / cv2 一个都没有(transformers 的 fetch_videos
    固定走 torchcodec),所以 URL 和路径现在解不了,只能由调用方解好帧再传。
    """
    if isinstance(src, np.ndarray):
        if src.ndim == 4:       # (T, H, W, 3)
            return list(src)
        raise ValueError(f"帧数组应该是 (T, H, W, 3),收到 {src.shape}")
    if isinstance(src, (list, tuple)):
        return list(src)
    raise ValueError(
        "视频解码不可用:环境里没有 torchcodec/decord/av/cv2。"
        "请先把视频解码成帧列表([np.ndarray(H,W,3)] 或 (T,H,W,3) 数组)再传,"
        "或 `pip install torchcodec` 后改用 video_url")


def _check_placeholders(inputs: dict, processor, n_images: int, n_videos: int):
    """文本占位符数 == grid 推出来的 token 数,且张数 == 请求里的媒体数。

    两处不匹配时 processor 都不报错(多给的图被静默丢弃、文本里没占位符的图直接消失),
    模型只会拿到对不上的输入然后答错,所以必须在这里拦。
    """
    merge = processor.image_processor.merge_size ** 2
    for token_id, grid_key, n_given, name in (
        (processor.image_token_id, "image_grid_thw", n_images, "图片"),
        (processor.video_token_id, "video_grid_thw", n_videos, "视频"),
    ):
        grid = inputs.get(grid_key)
        if grid is None:
            if n_given:
                raise ValueError(f"请求里有 {n_given} 个{name},但 processor 没有产出 {grid_key}")
            continue
        n_pad = int((inputs["input_ids"][0] == token_id).sum())
        n_from_grid = int(grid.prod(-1).sum() // merge)
        if n_pad != n_from_grid:
            raise ValueError(
                f"{name} token 对不上: input_ids 里 {n_pad} 个, grid 推出 {n_from_grid} 个"
                "(多半是模板没渲染出占位符,或用了截断)")
        if grid.shape[0] != n_given:
            raise ValueError(f"{name}张数对不上: 请求里 {n_given} 个, processor 只处理了 {grid.shape[0]} 个")
