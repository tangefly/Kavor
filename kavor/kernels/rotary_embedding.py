from functools import lru_cache

import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _rope_kernel(
    q_ptr,
    k_ptr,
    pos_ptr,
    cos_sin_ptr,
    q_heads,
    k_heads,
    q_stride_t,
    q_stride_h,
    k_stride_t,
    k_stride_h,
    HALF: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """q、k 一次 launch 做完 RoPE,逐 (token, head) 一格,原地写回。

    数值路径与 eager 参考一致:fp32 计算 (x1·cos − x2·sin, x2·cos + x1·sin),
    转回原 dtype 存储。cos_sin 表布局 (max_pos, 1, head_size),每行前半 cos、
    后半 sin。q/k 是 qkv split 出来的不连续视图,按真实 stride 寻址。
    """
    t = tl.program_id(0)
    h = tl.program_id(1)
    i = tl.arange(0, BLOCK)
    mask = i < HALF
    if h < q_heads:
        base = q_ptr + t * q_stride_t + h * q_stride_h
    else:
        base = k_ptr + t * k_stride_t + (h - q_heads) * k_stride_h
    pos = tl.load(pos_ptr + t)
    cs = cos_sin_ptr + pos * (2 * HALF)
    cos = tl.load(cs + i, mask=mask, other=0.0)
    sin = tl.load(cs + HALF + i, mask=mask, other=0.0)
    if INTERLEAVED:
        off1 = 2 * i
        off2 = 2 * i + 1
    else:
        off1 = i
        off2 = i + HALF
    x1 = tl.load(base + off1, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(base + off2, mask=mask, other=0.0).to(tl.float32)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    tl.store(base + off1, y1.to(q_ptr.dtype.element_ty), mask=mask)
    tl.store(base + off2, y2.to(q_ptr.dtype.element_ty), mask=mask)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # 前后半拆:(x_i, x_{i + d/2}) 这一对用 theta_i 旋转(eager 参考实现)
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def apply_rotary_emb_interleaved(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # 相邻成对:(x_2i, x_2i+1) 这一对用 theta_i 旋转(Ernie4_5/GLM 的写法,eager 参考实现)
    orig_dtype = x.dtype
    x = x.float()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.stack((y1, y2), dim=-1).flatten(-2).to(orig_dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        interleaved: bool = False,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.interleaved = interleaved
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # fused:q + k 一个 kernel 原地做完,替代原先 ~14 个 eager elementwise
        half = self.head_size // 2
        q_heads, k_heads = query.shape[1], key.shape[1]
        BLOCK = max(triton.next_power_of_2(half), 16)
        _rope_kernel[(query.shape[0], q_heads + k_heads)](
            query, key, positions, self.cos_sin_cache,
            q_heads, k_heads,
            query.stride(0), query.stride(1), key.stride(0), key.stride(1),
            HALF=half, INTERLEAVED=self.interleaved, BLOCK=BLOCK,
            num_warps=max(1, min(4, BLOCK // 64)))
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    interleaved: bool = False,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base, interleaved)
    return rotary_emb


@triton.jit
def _mrope_partial_kernel(
    q_ptr,
    k_ptr,
    pos_ptr,            # [3, T] int64;文本时三行同源(stride 0 也合法)
    inv_freq_ptr,       # [HALF] fp32
    freq_row_ptr,       # [HALF] int:第 i 个频率取哪条位置流(0=t,1=h,2=w)
    q_heads,
    k_heads,
    q_stride_t,
    q_stride_h,
    k_stride_t,
    k_stride_h,
    stride_pos_stream,
    HALF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """3 流 mrope + 部分旋转,原地写 q/k 的前 rotary_dim 列(rotate_half 语义,其余列不动)。

    数值路径与 Qwen3_5MRotaryEmbedding eager 一致:freq_i = positions[freq_row[i]] ·
    inv_freq[i] 在 fp32 算,cos/sin fp32,(x1·cos − x2·sin, x2·cos + x1·sin) fp32
    算完转回原 dtype。一个 launch 做完 q+k,替代 eager 的 ~25 个算子。
    """
    t = tl.program_id(0)
    h = tl.program_id(1)
    i = tl.arange(0, BLOCK)
    mask = i < HALF
    if h < q_heads:
        base = q_ptr + t * q_stride_t + h * q_stride_h
    else:
        base = k_ptr + t * k_stride_t + (h - q_heads) * k_stride_h
    row = tl.load(freq_row_ptr + i, mask=mask, other=0)
    pos = tl.load(pos_ptr + row * stride_pos_stream + t, mask=mask, other=0).to(tl.float32)
    freq = tl.load(inv_freq_ptr + i, mask=mask, other=0.0)
    ang = pos * freq
    cos = tl.cos(ang)
    sin = tl.sin(ang)
    x1 = tl.load(base + i, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(base + HALF + i, mask=mask, other=0.0).to(tl.float32)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    tl.store(base + i, y1.to(q_ptr.dtype.element_ty), mask=mask)
    tl.store(base + HALF + i, y2.to(q_ptr.dtype.element_ty), mask=mask)


def apply_mrope_partial(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    inv_freq: torch.Tensor,
    freq_row: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q/k 形状 [T, H, head_dim] 的视图(最后一维连续),只旋转前 rotary_dim 列,原地返回。"""
    assert query.stride(-1) == 1 and key.stride(-1) == 1
    if positions.dim() == 1:      # 纯文本:三条流同源
        positions = positions.unsqueeze(0).expand(3, -1)
    half = rotary_dim // 2
    q_heads, k_heads = query.shape[1], key.shape[1]
    BLOCK = max(triton.next_power_of_2(half), 16)
    _mrope_partial_kernel[(query.shape[0], q_heads + k_heads)](
        query, key, positions, inv_freq, freq_row,
        q_heads, k_heads,
        query.stride(0), query.stride(1), key.stride(0), key.stride(1),
        positions.stride(0),
        HALF=half, BLOCK=BLOCK,
        num_warps=max(1, min(4, BLOCK // 64)))
    return query, key
