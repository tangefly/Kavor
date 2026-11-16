"""Qwen3_5 线性注意力的 causal conv1d 融合算子(Triton)。

decode:`causal_conv1d_update_inplace` 单 kernel 完成「读状态池 → 卷积(+silu)→
状态前移写回」,替代 eager 的 gather + cat + copy_ + F.conv1d + silu + scatter。

prefill:`varlen_causal_conv1d` 对打包的变长 token 流一次算完,序列头部不足
kernel_size-1 的窗口从各序列的 conv 状态(state)取,和 F.conv1d 前缀拼接语义一致。

数值路径与 eager 对齐:F.conv1d 的 bf16 累加在 fp32 做、输出先落回原 dtype,
silu 再按 fp32 opmath 计算后落回原 dtype。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_update_kernel(
    x_ptr,                # [B, C] 当前 token 的 mixed qkv
    state_ptr,            # conv 状态池视图 [num_slots, C, K-1]
    state_indices,        # [B] slot id(任意 stride)
    w_ptr,                # [C, K]
    b_ptr,                # [C],无 bias 时传 dummy
    o_ptr,                # [B, C]
    stride_x_b,
    stride_o_b,
    stride_state_slot,    # 池行 stride(页池跨块不连续)
    stride_idx,
    C,
    BC: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACT: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_c = tl.program_id(1)
    c = i_c * BC + tl.arange(0, BC)
    mask_c = c < C
    # int64:状态池可超 2^31 元素
    idx = tl.load(state_indices + i_n * stride_idx).to(tl.int64)
    p_s = state_ptr + idx * stride_state_slot + c * (K - 1)

    x = tl.load(x_ptr + i_n * stride_x_b + c, mask=mask_c, other=0.0)
    acc = tl.zeros([BC], dtype=tl.float32)
    if HAS_BIAS:
        acc += tl.load(b_ptr + c, mask=mask_c, other=0.0).to(tl.float32)
    # xw = [state | x] 共 K 列,FIR 一趟
    for j in tl.static_range(K):
        wj = tl.load(w_ptr + c * K + j, mask=mask_c, other=0.0).to(tl.float32)
        if j == K - 1:
            v = x.to(tl.float32)
        else:
            v = tl.load(p_s + j, mask=mask_c, other=0.0).to(tl.float32)
        acc += wj * v
    if ACT:
        # conv 输出先落原 dtype(同 F.conv1d),silu 按原 dtype 的 fp32 opmath
        acc = acc.to(o_ptr.dtype.element_ty).to(tl.float32)
        acc = acc * tl.sigmoid(acc)
    tl.store(o_ptr + i_n * stride_o_b + c, acc.to(o_ptr.dtype.element_ty), mask=mask_c)

    # 新状态 = xw[1:],整体前移一格;原地写回池
    for j in tl.static_range(K - 1):
        if j == K - 2:
            v = x
        else:
            v = tl.load(p_s + j + 1, mask=mask_c, other=0.0)
        tl.store(p_s + j, v, mask=mask_c)


def causal_conv1d_update_inplace(
    x: torch.Tensor,
    conv_state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
):
    """decode 单步:输入 [B, C],状态在池内原地前移,返回卷积输出 [B, C]。"""
    B, C = x.shape
    K = weight.shape[-1]
    assert conv_state_pool.stride(1) == K - 1 and conv_state_pool.stride(2) == 1
    has_bias = bias is not None
    if bias is None:
        bias = x.new_zeros(1)
    o = torch.empty_like(x)
    BC = 128
    _causal_conv1d_update_kernel[(B, triton.cdiv(C, BC))](
        x, conv_state_pool, state_indices, weight, bias, o,
        x.stride(0), o.stride(0),
        conv_state_pool.stride(0), state_indices.stride(0),
        C, BC=BC, K=K,
        HAS_BIAS=has_bias,
        ACT=activation is not None,
        num_warps=1, num_stages=2,
    )
    return o


@triton.jit
def _varlen_causal_conv1d_kernel(
    x_ptr,                # [total, C] 打包 token 流
    y_ptr,                # [total, C]
    s_ptr,                # [N, C, K-1] 各序列的头部状态(fresh 序列为 0)
    cu_ptr,               # [N+1] 每序列 token 偏移
    w_ptr,                # [C, K]
    b_ptr,
    stride_s_n,
    stride_x_t,
    stride_y_t,
    C,
    BC: tl.constexpr,
    BT: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACT: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_c = tl.program_id(1)
    i_t = tl.program_id(2)
    start = tl.load(cu_ptr + i_n)
    end = tl.load(cu_ptr + i_n + 1)
    t0 = start + i_t * BT
    if t0 >= end:
        return
    c = i_c * BC + tl.arange(0, BC)
    mask_c = c < C
    t = tl.arange(0, BT)
    mask_t = t0 + t < end

    acc = tl.zeros([BT, BC], dtype=tl.float32)
    if HAS_BIAS:
        acc += tl.load(b_ptr + c, mask=mask_c, other=0.0).to(tl.float32)[None, :]
    for j in tl.static_range(K):
        a = (t0 - (K - 1) + j + t).to(tl.int64)    # 该窗口列对应的绝对 token 下标
        from_x = a >= start
        xv = tl.load(x_ptr + a[:, None] * stride_x_t + c[None, :],
                     mask=(from_x & mask_t)[:, None] & mask_c[None, :],
                     other=0.0).to(tl.float32)
        sc = a - start + (K - 1)                   # 落在状态里的列号
        sv = tl.load(s_ptr + i_n * stride_s_n + c[None, :] * (K - 1) + sc[:, None],
                     mask=((~from_x) & (sc >= 0) & (sc < K - 1))[:, None] & mask_c[None, :],
                     other=0.0).to(tl.float32)
        v = tl.where(from_x[:, None], xv, sv)
        wj = tl.load(w_ptr + c * K + j, mask=mask_c, other=0.0).to(tl.float32)
        acc += v * wj[None, :]
    if ACT:
        acc = acc.to(y_ptr.dtype.element_ty).to(tl.float32)
        acc = acc * tl.sigmoid(acc)
    ta = (t0 + t).to(tl.int64)
    tl.store(y_ptr + ta[:, None] * stride_y_t + c[None, :],
             acc.to(y_ptr.dtype.element_ty),
             mask=mask_t[:, None] & mask_c[None, :])


def varlen_causal_conv1d(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
    max_len: int | None = None,
) -> torch.Tensor:
    """prefill:打包变长 token 流一次卷完,返回 [total, C]。

    states [N, C, K-1] 是本段之前已消费 token 的尾部(fresh 序列传零),等价于
    旧实现里 cat([conv_state, x]) 再 F.conv1d 的语义。max_len 已知时传入可省
    一次 GPU->CPU 同步。
    """
    total, C = x.shape
    N = cu_seqlens.numel() - 1
    K = weight.shape[-1]
    assert states.shape == (N, C, K - 1) and states.is_contiguous()
    has_bias = bias is not None
    if bias is None:
        bias = x.new_zeros(1)
    y = torch.empty_like(x)
    if max_len is None:
        max_len = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    BC, BT = 128, 16
    _varlen_causal_conv1d_kernel[(N, triton.cdiv(C, BC), triton.cdiv(max_len, BT))](
        x, y, states, cu_seqlens, weight, bias,
        states.stride(0), x.stride(0), y.stride(0),
        C, BC=BC, BT=BT, K=K,
        HAS_BIAS=has_bias,
        ACT=activation is not None,
        num_warps=4, num_stages=2,
    )
    return y
