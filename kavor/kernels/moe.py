import os

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:      # CPU-only 环境:回退到 torch 参考实现
    _HAS_TRITON = False


# 对齐粒度:专家数较多(本仓目标是 256)时,每个被命中的专家都会向上取整到
# block_m 行,粒度太大在小 batch 下会白算;粒度太小又会让 prefill 的 GEMM
# M 维过碎。按 slot 总数分档,可用 KAVOR_MOE_BLOCK_M 强制覆盖(调参用)。
def pick_block_m(num_slots: int) -> int:
    override = os.environ.get("KAVOR_MOE_BLOCK_M")
    if override:
        return int(override)
    if num_slots <= 16384:
        return 16
    if num_slots <= 65536:
        return 32
    return 64


def moe_align_block_size(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """slot 级重排:同专家相邻、每专家对齐到 block_m。

    参数
    ----
    topk_ids: (T, top_k) 整型,每个 token 选中的专家下标
    topk_weights: (T, top_k) 浮点,routing weight

    返回
    ----
    sorted_ids: (num_blocks * block_m,) int32
        重排后每个位置对应的 token 下标,空位填哨兵 ``T``。
    expert_ids: (num_blocks,) int32
        每个 block 所属的专家;无效 block 为 -1。
    sorted_weights: (num_blocks * block_m,) fp32
        与 sorted_ids 对应的 routing weight,空位填 0。

    block 数上界取 ``ceil(S/block_m) + E``(S = T*top_k):每个专家的向上取整
    合计不超过 ``S + E*(block_m-1)``,这个上界一定够,且只依赖静态形状。
    """
    if _HAS_TRITON and topk_ids.is_cuda and num_experts <= 2048:
        return _moe_align_block_size_triton(topk_ids, topk_weights, num_experts, block_m)
    return _moe_align_block_size_torch(topk_ids, topk_weights, num_experts, block_m)


def _moe_align_block_size_torch(topk_ids, topk_weights, num_experts, block_m):
    """参考实现(CPU 兜底)。一长串小算子,单层就要 0.17ms,GPU 上走 Triton 版。"""
    num_tokens, top_k = topk_ids.shape
    num_slots = num_tokens * top_k
    device = topk_ids.device

    flat = topk_ids.reshape(-1).to(torch.int64)
    # 稳定排序:同专家内部保持 slot 原来的顺序(可复现,便于对拍)
    sorted_experts, order = torch.sort(flat, stable=True)
    # 不能用 torch.bincount:它的输出长度取决于 max(元素),内部要做 D2H 同步,
    # CUDA graph 捕获会直接失败。scatter_add_ 形状静态、无同步。
    counts = torch.zeros(num_experts, dtype=torch.int64, device=device)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    padded_counts = (counts + block_m - 1) // block_m * block_m
    expert_offsets = torch.cumsum(counts, 0) - counts
    pad_offsets = torch.cumsum(padded_counts, 0) - padded_counts
    num_pad_slots = pad_offsets[-1] + padded_counts[-1]
    # slot 在排序结果里的位置 -> 对齐后的落点(每个落点唯一)
    dst = pad_offsets[sorted_experts] + (
        torch.arange(num_slots, device=device) - expert_offsets[sorted_experts])

    num_blocks = (num_slots + block_m - 1) // block_m + num_experts
    sorted_ids = torch.full(
        (num_blocks * block_m,), num_tokens, dtype=torch.int32, device=device)
    # order 是 slot 下标,除以 top_k 才是 token 下标
    sorted_ids.scatter_(
        0, dst, torch.div(order, top_k, rounding_mode="floor").to(torch.int32))
    sorted_weights = torch.zeros(
        num_blocks * block_m, dtype=torch.float32, device=device)
    sorted_weights.scatter_(
        0, dst, topk_weights.reshape(-1).float()[order])

    block_starts = torch.arange(num_blocks, device=device) * block_m
    # right=True 会跳过"空专家"造成的重复 offset,正好落在该 block 的归属专家上
    expert_ids = torch.searchsorted(pad_offsets, block_starts, right=True) - 1
    expert_ids = torch.where(
        block_starts < num_pad_slots,
        expert_ids.clamp(0, num_experts - 1),
        expert_ids.new_full((), -1),
    ).to(torch.int32)
    return sorted_ids, expert_ids, sorted_weights


@triton.jit
def _moe_count_kernel(ids_ptr, counts_ptr, num_slots, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_slots
    expert = tl.load(ids_ptr + offs, mask=mask, other=0)
    tl.atomic_add(counts_ptr + expert, 1, mask=mask)


@triton.jit
def _moe_offsets_kernel(
    counts_ptr, pad_ptr, eoff_ptr, total_ptr,
    NUM_EXPERTS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_E: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_E)
    mask = offs < NUM_EXPERTS
    counts = tl.load(counts_ptr + offs, mask=mask, other=0)
    padded = (counts + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    tl.store(eoff_ptr + offs, tl.cumsum(counts, 0) - counts, mask=mask)
    tl.store(pad_ptr + offs, tl.cumsum(padded, 0) - padded, mask=mask)
    tl.store(total_ptr, tl.sum(padded, 0))


@triton.jit
def _moe_scatter_kernel(
    sorted_experts_ptr, order_ptr, pad_ptr, eoff_ptr, w_ptr, sid_ptr, sw_ptr,
    num_slots, TOP_K: tl.constexpr, BLOCK: tl.constexpr,
):
    """按"排序位置 - 专家起点"决定落点,无 atomic,结果可复现。"""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < num_slots
    expert = tl.load(sorted_experts_ptr + i, mask=mask, other=0)
    slot = tl.load(order_ptr + i, mask=mask, other=0)
    dst = (tl.load(pad_ptr + expert, mask=mask, other=0) + i
           - tl.load(eoff_ptr + expert, mask=mask, other=0))
    tl.store(sid_ptr + dst, slot // TOP_K, mask=mask)
    weight = tl.load(w_ptr + slot, mask=mask, other=0.0)
    tl.store(sw_ptr + dst, weight.to(tl.float32), mask=mask)


@triton.jit
def _moe_expert_ids_kernel(
    pad_ptr, eid_ptr, total_ptr,
    NUM_EXPERTS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_E: tl.constexpr,
):
    """每个 block 的归属专家 = 最后一个 pad_offset <= block_start 的专家。

    空专家会让相邻 offset 相等,但它的 pad 区间为空,所以"取最后一个不小于"
    正好跳过它们(与非空专家的严格递增一起保证唯一)。
    """
    block_start = tl.program_id(0) * BLOCK_M
    offs = tl.arange(0, BLOCK_E)
    pad = tl.load(pad_ptr + offs, mask=offs < NUM_EXPERTS, other=2**31 - 1)
    expert = tl.sum((pad <= block_start).to(tl.int32), 0) - 1
    expert = tl.where(block_start < tl.load(total_ptr), tl.maximum(expert, 0), -1)
    tl.store(eid_ptr + tl.program_id(0), expert)


def _moe_align_block_size_triton(topk_ids, topk_weights, num_experts, block_m):
    num_tokens, top_k = topk_ids.shape
    num_slots = num_tokens * top_k
    device = topk_ids.device
    num_blocks = (num_slots + block_m - 1) // block_m + num_experts

    # 稳定排序放在 torch 侧:省掉一版 Triton 排序,同时保证同专家内的顺序可复现
    flat = topk_ids.reshape(-1)
    sorted_experts, order = torch.sort(flat, stable=True)

    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    _moe_count_kernel[((num_slots + 1023) // 1024,)](
        flat.to(torch.int32), counts, num_slots, BLOCK=1024)

    pad_offsets = torch.empty(num_experts, dtype=torch.int32, device=device)
    expert_offsets = torch.empty(num_experts, dtype=torch.int32, device=device)
    total = torch.empty(1, dtype=torch.int32, device=device)
    _moe_offsets_kernel[(1,)](
        counts, pad_offsets, expert_offsets, total,
        NUM_EXPERTS=num_experts, BLOCK_M=block_m,
        BLOCK_E=max(triton.next_power_of_2(num_experts), 16))

    sorted_ids = torch.full(
        (num_blocks * block_m,), num_tokens, dtype=torch.int32, device=device)
    sorted_weights = torch.zeros(
        num_blocks * block_m, dtype=torch.float32, device=device)
    _moe_scatter_kernel[((num_slots + 1023) // 1024,)](
        sorted_experts, order.to(torch.int32), pad_offsets, expert_offsets,
        topk_weights.reshape(-1), sorted_ids, sorted_weights,
        num_slots, TOP_K=top_k, BLOCK=1024)

    expert_ids = torch.empty(num_blocks, dtype=torch.int32, device=device)
    _moe_expert_ids_kernel[(num_blocks,)](
        pad_offsets, expert_ids, total,
        NUM_EXPERTS=num_experts, BLOCK_M=block_m,
        BLOCK_E=max(triton.next_power_of_2(num_experts), 16))
    return sorted_ids, expert_ids, sorted_weights


@triton.jit
def _moe_gate_up_silu_kernel(
    x_ptr, sid_ptr, eid_ptr, w_ptr, s_ptr, out_ptr,
    num_tokens, hidden_size, local_inter, inter_blocks,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,      # 中间维上的一块,scale 路径下固定 128
    BLOCK_K: tl.constexpr,      # scale 路径下固定 128
    HAS_SCALE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert = tl.load(eid_ptr + pid_m)
    if expert < 0:      # 本 block 没有有效 token(尾部填充 block)
        return

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    slot_tokens = tl.load(sid_ptr + rows)
    valid = slot_tokens < num_tokens
    tok = tl.where(valid, slot_tokens, 0).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 权重布局:(E, 2L, H),行主序;专家 e 的前 L 行是 gate,后 L 行是 up
    e_off = expert.to(tl.int64)
    gate_w = w_ptr + e_off * (2 * local_inter) * hidden_size
    up_w = gate_w + local_inter * hidden_size
    # tl.dot 要 (BLOCK_K, BLOCK_N):沿 k 连续取,按 n 跨 stride
    wg_ptrs = gate_w + offs_k[:, None] + offs_n[None, :] * hidden_size
    wu_ptrs = up_w + offs_k[:, None] + offs_n[None, :] * hidden_size
    x_ptrs = x_ptr + tok[:, None] * hidden_size + offs_k[None, :]

    if HAS_SCALE:
        n_blk = (pid_n * BLOCK_N) // 128       # 该 tile 落在哪个 128 列的 scale 块
        s_gate = s_ptr + (e_off * inter_blocks + n_blk) * (hidden_size // 128)
        s_up = s_gate + local_inter // 128 * (hidden_size // 128)

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # 必须用 range 让 Triton 做软件流水(同 triton_w8a16 的说明)
    for k0 in range(0, hidden_size, BLOCK_K):
        x_tile = tl.load(x_ptrs, mask=valid[:, None], other=0.0)
        wg = tl.load(wg_ptrs).to(tl.bfloat16)
        wu = tl.load(wu_ptrs).to(tl.bfloat16)
        if HAS_SCALE:
            kb = k0 // BLOCK_K
            acc_g += tl.dot(x_tile, wg) * tl.load(s_gate + kb).to(tl.float32)
            acc_u += tl.dot(x_tile, wu) * tl.load(s_up + kb).to(tl.float32)
        else:
            acc_g += tl.dot(x_tile, wg)
            acc_u += tl.dot(x_tile, wu)
        x_ptrs += BLOCK_K
        wg_ptrs += BLOCK_K
        wu_ptrs += BLOCK_K

    y = acc_g * tl.sigmoid(acc_g) * acc_u       # silu(gate) * up
    out_ptrs = out_ptr + rows[:, None] * local_inter + offs_n[None, :]
    tl.store(out_ptrs, y.to(out_ptr.dtype.element_ty), mask=valid[:, None])


@triton.jit
def _moe_down_kernel(
    x2_ptr, sid_ptr, eid_ptr, sw_ptr, w_ptr, s_ptr, out_ptr,
    num_tokens, hidden_size, local_inter,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,      # hidden 维上的一块
    BLOCK_K: tl.constexpr,      # 中间维上的一块
    HAS_SCALE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert = tl.load(eid_ptr + pid_m)
    if expert < 0:
        return

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    slot_tokens = tl.load(sid_ptr + rows)
    valid = slot_tokens < num_tokens
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    e_off = expert.to(tl.int64)
    w_base = w_ptr + e_off * hidden_size * local_inter
    w_ptrs = w_base + offs_k[:, None] + offs_n[None, :] * local_inter
    x2_ptrs = x2_ptr + rows[:, None] * local_inter + offs_k[None, :]

    if HAS_SCALE:
        n_blk = (pid_n * BLOCK_N) // 128
        s_base = s_ptr + (e_off * (hidden_size // 128) + n_blk) * (local_inter // 128)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, local_inter, BLOCK_K):
        x_tile = tl.load(x2_ptrs, mask=valid[:, None], other=0.0)
        w = tl.load(w_ptrs).to(tl.bfloat16)
        if HAS_SCALE:
            acc += tl.dot(x_tile, w) * tl.load(s_base + k0 // BLOCK_K).to(tl.float32)
        else:
            acc += tl.dot(x_tile, w)
        x2_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    acc *= tl.load(sw_ptr + rows).to(tl.float32)[:, None]
    tok = tl.where(valid, slot_tokens, 0)
    # 一个 token 的 top_k 个 slot 落在不同 block,这里靠 fp32 atomic 合并
    tl.atomic_add(
        out_ptr + tok[:, None] * hidden_size + offs_n[None, :],
        acc, mask=valid[:, None])


def _block_sizes(local_inter: int, hidden_size: int, has_scale: bool) -> tuple[int, int]:
    """选 tile:scale 路径(128x128 块)必须 128;bf16 路径退到能整除的粒度。

    gate/up 两个 W tile 同时在流水线里(fp8 各 16KB/级),BLOCK_K 用 128 时
    bf16 会有 2x32KB/级、超过 100KB 的 shared memory,所以 bf16 走
    BLOCK_K=64 + num_stages=2。
    """
    if has_scale:
        assert local_inter % 128 == 0 and hidden_size % 128 == 0, (
            f"fp8 block scale 要求 local_inter/hidden 都是 128 的倍数,"
            f"当前 {local_inter}/{hidden_size}")
        return 128, 128
    block_n = 128 if local_inter % 128 == 0 else 64 if local_inter % 64 == 0 else 32
    block_k = 64 if hidden_size % 64 == 0 and local_inter % 64 == 0 else 32
    assert local_inter % block_n == 0 and hidden_size % block_k == 0
    return block_n, block_k


def _num_stages(block_k: int) -> int:
    return 2 if block_k >= 128 else 3


def _moe_forward_triton(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_scale: torch.Tensor | None,
    down_proj: torch.Tensor,
    down_scale: torch.Tensor | None,
) -> torch.Tensor:
    num_tokens, hidden_size = x.shape
    num_experts, two_inter, _ = gate_up_proj.shape
    local_inter = two_inter // 2
    has_scale = gate_up_scale is not None
    assert x.dtype == torch.bfloat16, f"MoE kernel 只支持 bf16 激活,当前 {x.dtype}"

    block_m = pick_block_m(num_tokens * topk_ids.shape[1])
    sorted_ids, expert_ids, sorted_w = moe_align_block_size(
        topk_ids, topk_weights, num_experts, block_m)
    block_n, block_k = _block_sizes(local_inter, hidden_size, has_scale)

    num_blocks = expert_ids.numel()
    inter = torch.empty(
        (num_blocks * block_m, local_inter), dtype=x.dtype, device=x.device)
    # 小 BLOCK_M 时 8 个 warp 喂不满 MMA(每颗只摊到很少的 tile),用 4 个更省
    num_warps = 4 if block_m <= 16 else 8

    _moe_gate_up_silu_kernel[(num_blocks, local_inter // block_n)](
        x, sorted_ids, expert_ids, gate_up_proj,
        gate_up_scale if has_scale else x,
        inter,
        num_tokens, hidden_size, local_inter, two_inter // 128,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        HAS_SCALE=has_scale,
        num_warps=num_warps,
        num_stages=_num_stages(block_k),
    )

    # fp32 累加缓冲:bf16 上 atomic 累加会掉精度,也便于和 shared expert 相加
    out = torch.zeros((num_tokens, hidden_size), dtype=torch.float32, device=x.device)
    _moe_down_kernel[(num_blocks, hidden_size // block_n)](
        inter, sorted_ids, expert_ids, sorted_w, down_proj,
        down_scale if has_scale else x,
        out,
        num_tokens, hidden_size, local_inter,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        HAS_SCALE=has_scale,
        num_warps=num_warps,
        num_stages=_num_stages(block_k),
    )
    return out


def _dequant_weight(weight: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    """(E, N, K) 权重按 128x128 block scale 反量化成 fp32(参考实现用)。"""
    weight = weight.float()
    if scale is None:
        return weight
    _, n, k = weight.shape
    expanded = scale.float().repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)
    return weight * expanded[:, :n, :k]


def torch_fused_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_scale: torch.Tensor | None,
    down_proj: torch.Tensor,
    down_scale: torch.Tensor | None,
) -> torch.Tensor:
    """纯 torch 参考/兜底实现(按专家循环)。

    有 host 同步(``nonzero``/``numel``),不能进 CUDA graph:只用于 CPU、
    单测对拍和调试。返回 fp32。
    """
    num_tokens, hidden_size = x.shape
    num_experts = gate_up_proj.shape[0]
    local_inter = gate_up_proj.shape[1] // 2
    w13 = _dequant_weight(gate_up_proj, gate_up_scale)
    w2 = _dequant_weight(down_proj, down_scale)

    xf = x.float()
    out = torch.zeros((num_tokens, hidden_size), dtype=torch.float32, device=x.device)
    for expert in range(num_experts):
        token_idx, slot_idx = torch.where(topk_ids == expert)
        if token_idx.numel() == 0:
            continue
        h = xf[token_idx]
        gate, up = (h @ w13[expert].T).split(local_inter, dim=-1)
        y = (F.silu(gate) * up) @ w2[expert].T
        y = y * topk_weights[token_idx, slot_idx].float()[:, None]
        out.index_add_(0, token_idx, y)
    return out


def fused_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_scale: torch.Tensor | None,
    down_proj: torch.Tensor,
    down_scale: torch.Tensor | None,
) -> torch.Tensor:
    """MoE 专家前向:x (T, H) -> (T, H) fp32。

    ``gate_up_proj`` 是 (E, 2*local_inter, H):前 local_inter 行 gate、后
    local_inter 行 up(与 kavor 其它合并 linear 的排布一致);TP 切分在
    中间维上,各 rank 只算自己那一片,调用方负责 all_reduce。
    """
    assert x.dim() == 2 and gate_up_proj.dim() == 3
    if os.environ.get("KAVOR_MOE_TORCH"):
        # 调试/对拍用:强制走纯 torch 参考实现(有 host 同步,不能进 CUDA graph)
        return torch_fused_moe(
            x, topk_ids, topk_weights, gate_up_proj, gate_up_scale,
            down_proj, down_scale)
    if not (_HAS_TRITON and x.is_cuda):
        return torch_fused_moe(
            x, topk_ids, topk_weights, gate_up_proj, gate_up_scale,
            down_proj, down_scale)
    return _moe_forward_triton(
        x.contiguous(), topk_ids, topk_weights,
        gate_up_proj, gate_up_scale, down_proj, down_scale)
