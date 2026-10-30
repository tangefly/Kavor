"""对外消息内容 → PIL 图片 / 帧列表:URL 下载、data URL 解码、ffmpeg 抽帧。

这里只做 I/O,不认识 MMInputs/MediaItem;multimodal.py 拿到原始媒体后再交给
processor。
"""

import base64
import io
import json
import subprocess
import tempfile
import urllib.request

import numpy as np
from PIL import Image


MAX_IMAGE_BYTES = 32 * 2**20
MAX_VIDEO_BYTES = 64 * 2**20


def normalize_messages(messages: list[dict]) -> tuple[list[dict], list[Image.Image], list]:
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
        return fetch_image_url(src)

    raise ValueError(f"不支持的图片地址(只认 data: 和 http(s)://): {src[:64]}")


def fetch_image_url(url: str) -> Image.Image:
    # 走 stdlib,不引额外依赖。注意:这里会去访问任意 URL,只适合内网可信部署。
    request = urllib.request.Request(url, headers={"User-Agent": "kavor"})
    with urllib.request.urlopen(request, timeout=10) as response:      # noqa: S310
        raw = response.read(MAX_IMAGE_BYTES + 1)
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"图片超过 {MAX_IMAGE_BYTES // 2**20} MiB 上限")
    return Image.open(io.BytesIO(raw)).convert("RGB")


def load_video(src):
    """视频 → 帧列表。已经是帧数组的直接返回。

    data:video/...;base64,... 先落到临时文件,再交给 ffmpeg 解码(见
    video_frames_from_path)。本地路径和 file:// 一律拒绝 —— 这是对外的 HTTP 服务,
    放开路径等于把服务器变成任意文件读取器。
    """
    if isinstance(src, np.ndarray):
        if src.ndim == 4:       # (T, H, W, 3)
            return list(src)
        raise ValueError(f"帧数组应该是 (T, H, W, 3),收到 {src.shape}")
    if isinstance(src, (list, tuple)):
        return list(src)
    if isinstance(src, str) and src.startswith("data:"):
        head, _, b64 = src.partition(",")
        if not b64 or "base64" not in head:
            raise ValueError(f"只支持 base64 编码的 data URL: {head[:48]}")
        try:
            raw = base64.b64decode(b64, validate=True)
        except Exception as e:
            raise ValueError(f"data URL 的 base64 解码失败: {e}") from e
        if len(raw) > MAX_VIDEO_BYTES:
            raise ValueError(f"视频超过 {MAX_VIDEO_BYTES // 2**20} MiB 上限")
        # 容器后缀只影响 ffmpeg 的探测顺序,内容还是靠 probe
        suffix = "." + (head[5:].rpartition("/")[2].partition(";")[0] or "mp4")
        with tempfile.NamedTemporaryFile(suffix=suffix) as f:
            f.write(raw)
            f.flush()
            return video_frames_from_path(f.name)
    raise ValueError(
        "不支持的视频地址(只认 base64 data URL)。"
        "请先把视频解码成帧列表([np.ndarray(H,W,3)] 或 (T,H,W,3) 数组)再传;"
        "本地脚本可以直接用 video_frames_from_path(path) 走 ffmpeg 解码")


def probe_video(path: str) -> dict:
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
             "-of", "json", path],
            capture_output=True, check=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("读本地视频需要 ffprobe/ffmpeg(解码视频帧)") from e
    stream = json.loads(probe.stdout)["streams"][0]
    num, _, den = stream["r_frame_rate"].partition("/")      # "30/1"
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(num) / float(den or 1),
        "num_frames": int(stream.get("nb_frames") or 0),
    }


def video_frames_from_path(path: str, max_frames: int | None = None) -> list[np.ndarray]:
    info = probe_video(path)
    width, height = info["width"], info["height"]

    try:
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True, check=True).stdout
    except FileNotFoundError as e:
        raise RuntimeError("读本地视频需要 ffmpeg(解码视频帧)") from e

    frame_bytes = width * height * 3
    frames = np.frombuffer(raw, dtype=np.uint8)
    frames = frames[: len(frames) // frame_bytes * frame_bytes].reshape(-1, height, width, 3)
    if max_frames is not None:
        frames = frames[:max_frames]
    return [frame.copy() for frame in frames]
