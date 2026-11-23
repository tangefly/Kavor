"""W8A16 GEMM:FP8 块量化权重 × BF16 激活(Phase 1 计算路径)。

权重在显存里保持 fp8,kernel 内按 tile 转成 bf16 再进 tensor core,
HBM 流量只有 fp8 一份。128x128 块对齐让每个 (BLOCK_N, BLOCK_K) tile
恰好对应一个 scale 标量,scale 直接乘在部分和上:

    y[m, n] = Σ_kb  S[n/128, kb] × (x[m, kb] · W_fp8[n, kb])

约定(加载期 Fp8BlockLinearMethod.create_weights 已断言):N、K 均为
128 的倍数,所以 kernel 里只有 M 维需要 mask;BLOCK_N/BLOCK_K 固定 128。
fp32 累加,bf16 输出。无 host 同步、无动态分配,可进 CUDA graph。

decode(小 M)时 N/128 个 CTA 常常喂不满 SM(如 down_proj 只有 40 个),
此时沿 K 再切 SPLIT_K 份、fp32 atomic 累加到工作区后在 host 侧合成;
atomic 的归约顺序不确定,带来末位比特级的非确定性(推理可接受)。
"""

import torch
import triton
import triton.language as tl

sm_count_cache: dict = {}


def get_sm_count(device) -> int:
    if device not in sm_count_cache:
        sm_count_cache[device] = torch.cuda.get_device_properties(device).multi_processor_count
    return sm_count_cache[device]


def pick_split_k(M: int, N: int, K: int, sm_count: int) -> int:
    """小 M 且 N-tile 明显喂不满 SM 时,选 K/128 的一个因子补并发。"""
    if M > 32:
        return 1
    n_tiles = N // 128
    if n_tiles > sm_count // 2:     # tile 数过半就不切,split 对高占用形状反而略亏
        return 1
    k_blocks = K // 128
    target = max(1, 2 * sm_count // max(n_tiles, 1))
    for d in range(min(target, k_blocks), 0, -1):
        if k_blocks % d == 0:
            return d
    return 1


@triton.jit
def w8a16_block_gemm(
    x_ptr, w_ptr, scale_ptr, bias_ptr, y_ptr, acc_ptr,
    M, N, K, kb_per_split,
    stride_xm, stride_wn, stride_sn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,      # 固定 128,与 scale 块对齐
    BLOCK_K: tl.constexpr,      # 固定 128,同上
    HAS_BIAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m[:, None] < M
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    # 权重 [N, K] 行主序,沿 k 连续;tile 按 [BLOCK_K, BLOCK_N] 取转置视图
    w_ptrs = w_ptr + offs_k[:, None] + offs_n[None, :] * stride_wn
    scale_row = scale_ptr + pid_n * stride_sn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_begin = pid_k * kb_per_split * BLOCK_K
    x_ptrs += k_begin     # 指针也要跳到自己那份的 k 段起点
    w_ptrs += k_begin
    # 必须用 for(range) 而不是 while:Triton 只对前者做软件流水(num_stages)
    for k0 in range(k_begin, k_begin + kb_per_split * BLOCK_K, BLOCK_K):
        x_tile = tl.load(x_ptrs, mask=m_mask, other=0.0)
        w_tile = tl.load(w_ptrs).to(tl.bfloat16)     # fp8 -> bf16,寄存器内转换
        s = tl.load(scale_row + k0 // BLOCK_K).to(tl.float32)
        acc += tl.dot(x_tile, w_tile) * s            # 块内一个标量,乘在部分和上
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    if SPLIT_K == 1:
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n).to(tl.float32)[None, :]
        y_ptrs = y_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(y_ptrs, acc.to(tl.bfloat16), mask=m_mask)
    else:
        # split-K:各份额 fp32 原子累加,bias 与 bf16 舍入由调用方统一做
        tl.atomic_add(acc_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=m_mask)


def fp8_blockwise_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """W8A16 Linear:x[..., K] bf16 @ (W_fp8 * scale)^T -> bf16。"""
    assert weight.dtype == torch.float8_e4m3fn and weight.is_contiguous()
    assert weight_scale.is_contiguous()
    M = x.numel() // x.shape[-1]
    K = x.shape[-1]
    N = weight.shape[0]
    x2 = x.reshape(M, K).contiguous()

    split_k = pick_split_k(M, N, K, get_sm_count(x.device))
    # tensor core 要求各维 >= 16;decode 小 M 用 16,prefill 大 M 用 128
    BLOCK_M = 16 if M <= 16 else 32 if M <= 32 else 128
    num_warps = 4 if BLOCK_M <= 32 else 8

    if split_k == 1:
        y = torch.empty(M, N, dtype=x.dtype, device=x.device)
        acc = y   # 占位,SPLIT_K=1 的分支不会碰它
    else:
        acc = torch.zeros(M, N, dtype=torch.float32, device=x.device)
        y = acc   # 占位,同理

    grid = (triton.cdiv(M, BLOCK_M), N // 128, split_k)
    w8a16_block_gemm[grid](
        x2, weight, weight_scale,
        bias if bias is not None else x2,
        y, acc,
        M, N, K, K // 128 // split_k,
        x2.stride(0), weight.stride(0), weight_scale.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=128, BLOCK_K=128,
        HAS_BIAS=bias is not None,
        SPLIT_K=split_k,
        num_warps=num_warps,
        num_stages=3,
    )

    if split_k == 1:
        return y.reshape(*x.shape[:-1], N)
    out = acc.to(torch.bfloat16)
    if bias is not None:
        out = out + bias
    return out.reshape(*x.shape[:-1], N)
