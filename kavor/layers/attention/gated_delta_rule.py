import torch
from transformers.utils import is_torchdynamo_exporting
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

# NOTE: The FLA package computes `x / torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)` instead, so if we align
# with the GatedRMSNorm, maybe we can make that change as well.
def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


@triton.jit(do_not_specialize=['T'])
def fused_recurrent_gdn_decode_kernel(
    q, k, v, g, beta, o,
    a, b, A_log, dt_bias,   # USE_GATE_IN_KERNEL=True 时的原始投影(a 未 softplus,b 未 sigmoid)
    h0,                     # fp32 state pool viewed as [num_slots, HV, K, V]; base ptr already skips the conv region
    state_indices,          # [B] slot id per sequence
    scale,
    T,
    stride_q_b, stride_k_b, stride_v_b,     # batch strides; != H*K/HV*V because _split_qkv views into packed qkv
    stride_g_b, stride_beta_b, stride_o_b,
    stride_a_b, stride_b_b,
    stride_h_slot,          # pool row stride in elements; != HV*K*V when states share a page with conv states
    stride_idx_b,           # state_indices row stride; != 1 when sliced from [B, num_state_groups]
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
):
    """Single-step (T=1) gated-delta-rule recurrence that updates the state pool in place.

    The recurrence, the T loop scaffold and the tile config mirror fla's
    fused_recurrent_gated_delta_rule_fwd_kernel (STATE_V_FIRST=False, headwise beta), so
    the arithmetic matches the fla reference to ulp level. Differences:
    1. state addressing: instead of a contiguous [N, HV, K, V] h0/ht, each program resolves
       its sequence's row via `state_indices` and `stride_h_slot`, and writes the updated
       tile back in place (removes the gather/scatter around the kernel);
    2. USE_GATE_IN_KERNEL: g/beta computed in-kernel from raw projections, matching
       g = -exp(A_log) * softplus(a + dt_bias) and beta = sigmoid(b) (F.softplus 的
       threshold=20 log1p 形式),替代 host 侧 ~8 个 eager 算子。
    """
    pid = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    i_v, i_nh = pid % NV, pid // NV
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    # q/k/v/g/beta are [B, T, ...] views whose batch stride is the packed conv_dim, so the
    # base offsets go through stride_*_b; the per-step increments below assume a contiguous
    # [T, H, K] packing inside one sequence, which holds for the T=1 decode shapes we run.
    p_q = q + i_n * stride_q_b + i_h * K + o_k
    p_k = k + i_n * stride_k_b + i_h * K + o_k
    p_v = v + i_n * stride_v_b + i_hv * V + o_v
    p_o = o + i_n * stride_o_b + i_hv * V + o_v
    if not USE_GATE_IN_KERNEL:
        p_g = g + i_n * stride_g_b + i_hv
        p_beta = beta + i_n * stride_beta_b + i_hv

    # Resolve the state tile through the slot index; int64 arithmetic because a page pool can
    # exceed 2^31 fp32 elements, which would silently wrap in 32-bit pointer math.
    state_idx = tl.load(state_indices + i_n * stride_idx_b).to(tl.int64)
    p_h = h0 + state_idx * stride_h_slot + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)

    for _ in tl.range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        if USE_GATE_IN_KERNEL:
            b_a = tl.load(a + i_n * stride_a_b + i_hv).to(tl.float32)
            b_b = tl.load(b + i_n * stride_b_b + i_hv).to(tl.float32)
            b_A = tl.load(A_log + i_hv).to(tl.float32)
            b_dt = tl.load(dt_bias + i_hv).to(tl.float32)
            sp_x = b_a + b_dt
            sp = tl.where(sp_x <= 20.0, tldevice.log1p(tl.exp(sp_x)), sp_x)
            b_g = -tl.exp(b_A) * sp
            b_beta = tl.sigmoid(b_b)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
            b_g = tl.load(p_g).to(tl.float32)

        # The recurrence itself, verbatim from the fla kernel loop body.
        b_h *= tl.exp(b_g)
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H * K
        p_k += H * K
        p_v += HV * V
        if not USE_GATE_IN_KERNEL:
            p_g += HV
            p_beta += HV
        p_o += HV * V

    tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)


def fused_recurrent_gdn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    """Decode-step gated delta rule that updates the recurrent state pool in place.

    Gate 输入二选一:
    - g/beta 预计算好(``[B, 1, HV]``,g 为 log 空间、beta 已过 sigmoid);
    - 或 a/b 传原始投影(``[B, 1, HV]``)+ A_log/dt_bias(``[HV]``),kernel 内算
      g = -exp(A_log)·softplus(a + dt_bias)、beta = sigmoid(b)(省 host 侧 ~8 个算子)。

    其余参数:
        q, k: ``[B, 1, H, K]``; v: ``[B, 1, HV, V]`` (batch strides may come from a packed
            qkv view, they are passed through to the kernel; GVA(H < HV)由 kernel 内映射)。
        state_pool: fp32 ``[num_slots, HV, K, V]`` strided view as bound by
            ``CachePool._bind_state_page`` (arbitrary row stride, contiguous within a slot).
        state_indices: ``[B]`` int slot ids with arbitrary stride (e.g. a column slice of
            ``linear_slot_ids``).
    Returns:
        ``o: [B, 1, HV, V]``. On return the pool rows at ``state_indices`` hold the updated
        states; nothing is gathered or scattered.

    There is no autograd wrapper on purpose: kavor only runs this under ``inference_mode``,
    and unlike fla's entry point the strided pool view must not be forced contiguous.
    """
    B, T, H, K = q.shape
    if T != 1:
        raise ValueError(f"decode kernel is single-step, got T={T}")
    HV, V = v.shape[-2], v.shape[-1]
    if HV % H:
        raise ValueError(f"HV ({HV}) must be divisible by H ({H})")
    for name, x, num_heads, head_dim in (("q", q, H, K), ("k", k, H, K), ("v", v, HV, V)):
        if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f"{name} must be a float dtype, got {x.dtype}")
        if x.stride(-1) != 1:
            raise ValueError(f"{name} must be contiguous in the head_dim dim")
        # the kernel addresses heads as `i_h * head_dim`; with a single head the term is 0
        if num_heads > 1 and x.stride(2) != head_dim:
            raise ValueError(
                f"{name} head stride must be {head_dim} (packed), got {x.stride(2)}")
    use_gate_in_kernel = a is not None
    if use_gate_in_kernel:
        if b is None or A_log is None or dt_bias is None:
            raise ValueError("USE_GATE_IN_KERNEL requires b, A_log and dt_bias")
        for name, x in (("a", a), ("b", b)):
            if x.shape != (B, 1, HV) or x.stride(-1) != 1:
                raise ValueError(f"{name} must be [B, 1, HV] contiguous, got {tuple(x.shape)}")
        if A_log.numel() != HV or dt_bias.numel() != HV:
            raise ValueError(f"A_log/dt_bias must have {HV} elements")
        g = g if g is not None else a   # dummy 指针,kernel 不读
        beta = beta if beta is not None else b
    else:
        if g is None or beta is None:
            raise ValueError("either (g, beta) or (a, b, A_log, dt_bias) must be given")
        if g.shape != (B, 1, HV) or beta.shape != (B, 1, HV):
            raise ValueError(
                f"g/beta must be [B, 1, HV]={(B, 1, HV)}, got {tuple(g.shape)}/{tuple(beta.shape)}")
        if g.stride(-1) != 1 or beta.stride(-1) != 1:
            raise ValueError("g/beta must be contiguous in the head dim")
        a, b, A_log, dt_bias = g, g, g, g     # dummy 指针,kernel 不读
    if state_pool.dtype != torch.float32:
        raise ValueError(f"state_pool must be fp32, got {state_pool.dtype}")
    if state_pool.stride()[-3:] != (K * V, V, 1):
        raise ValueError(
            f"state_pool must be contiguous within a slot [HV, K, V], got strides {state_pool.stride()}")
    if state_indices.ndim != 1 or state_indices.shape[0] != B:
        raise ValueError(f"state_indices must be [B]={(B,)}, got {tuple(state_indices.shape)}")
    if scale is None:
        scale = K ** -0.5

    o = torch.empty(B, 1, HV, V, dtype=v.dtype, device=v.device)
    # BV=32/2 warps 是 H100 上扫出来的最优(H=32,K=V=128):状态读写是纯带宽场景,
    # fla 默认的 BV=8 会切出过多小块,大 batch 下慢 ~10%。
    BK = triton.next_power_of_2(K)
    BV = min(32, triton.next_power_of_2(V))
    NV = triton.cdiv(V, BV)
    fused_recurrent_gdn_decode_kernel[(NV * B * HV,)](
        q, k, v, g, beta, o,
        a, b, A_log, dt_bias,
        state_pool, state_indices,
        scale,
        1,      # T: decode is single-step; kept as an arg to mirror fla's kernel structure
        q.stride(0), k.stride(0), v.stride(0), g.stride(0), beta.stride(0), o.stride(0),
        a.stride(0), b.stride(0),
        state_pool.stride(0), state_indices.stride(0),
        H=H, HV=HV, K=K, V=V, BK=BK, BV=BV,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_GATE_IN_KERNEL=use_gate_in_kernel,
        num_warps=2, num_stages=3,
    )
    return o

def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)  # -1 is the padded sequence length
    decay = g  # rename for clarity: argument name must stay "g" to match flash_linear_attention's API

    # Make sure all tensors are fp32 and reshape them to [batch_size, num_*_heads, seqlen, ...]
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    # If enabled, normalize query and key vectors (in fp32 to match the FLA library)
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    # And always normalize queries by the head dimension
    scaling = query.shape[-1] ** -0.5
    query = query * scaling

    # Pad sequence length to be a multiple of chunk_size. Padding is described as (left_pad, right_pad) for each dim.
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))

    total_sequence_length = sequence_length + pad_size
    num_chunks = total_sequence_length // chunk_size

    # Apply beta to K and V, which is the "learning rate" of the recurrent state for a given token, ie. how much the new
    # state influence the old state. Beta is often normalized to (0, 1) where 0 = no update; 1 = overwrite old state.
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Reshape all tensors to chunk the sequence dimension (adds a new dimension of size chunk_size)
    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)

    # Create a chunk-sized strictly upper triangular mask, ie. the mask of what a causal chunk may not attend to
    strictly_upper_mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)

    # Cumulative decay within each chunk (dim 3 is the position inside the chunk). Since decay is in log space,
    # cum_decay[..., t] is the log of a product of decays between the start of the chunk and position t
    cum_decay = decay.cumsum(dim=3)

    # First phase: compute intra-chunk quantities.
    # The pairwise decays: pairwise_decay[..., i, j] = exp(cum_decay_i - cum_decay_j) is the decay accumulated between
    # positions j and i of a chunk. Positive values are masked to -inf before exp to avoid overflow
    pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, float("-inf"))
    pairwise_decay = pairwise_decay.exp()  # with the exp, we exit log space, so we can apply this decay to the states

    # Compute auxiliary tensors: the Upper Triangular (ut) transform system and the intra-chunk attn (QK dot product)
    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    # Gated delta attention uses a UT transform to condense several delta rule updates into a few matmuls. After the UT
    # system is solved, we can then compute the new_values (called "u" in the DeltaNet paper) and the decayed keys
    # reading the old state (k_cumdecay). In the update, the part of new_values that the old state already predicts is
    # subtracted out, so that only the correction is written to the recurrent state: this is the delta rule.

    # Not all export targets support the fast triangular solver, so we build the inverse by forward substitution then
    if not is_torchdynamo_exporting():
        new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
        k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)
    else:
        ut_system = -ut_system.tril(-1)  # ut_system is masked to only keep the strictly lower triangle
        for i in range(1, chunk_size):
            row = ut_system[..., i, :i].clone()
            sub = ut_system[..., :i, :i].clone()
            ut_system[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        ut_system = ut_system + torch.eye(chunk_size, dtype=ut_system.dtype, device=ut_system.device)
        new_values, k_cumdecay = ut_system @ v_beta, ut_system @ decayed_k_beta

    if initial_state is None:
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=new_values.dtype, device=new_values.device)
    else:
        last_recurrent_state = initial_state.to(new_values)
    core_attn_out = torch.zeros_like(new_values)

    # Apply decay once rather than in each chunk
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    # Second phase: the sequential scan over chunks
    for i in range(num_chunks):
        # Compute attention output for the current chunk: add the read of the previous recurrent state
        # (inter_chunk_attn) with the within-chunk attention (intra_chunk_attn)
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
        inter_chunk_attn = query[:, :, i] @ last_recurrent_state
        core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
        # Update the recurrent state: new recurrent state (S_t+1) = decayed old state (S_t * (I-βkk^T)) + update (βvk^T)
        last_recurrent_state = last_recurrent_state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new

    # Discard the final state if not requested
    last_recurrent_state = None if not output_final_state else last_recurrent_state
    # Reshape the output to the orignal shape: flatten the chunk dimension, then drop padding
    core_attn_out = core_attn_out.reshape(padded_output_shape)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    # Convert back to the original shape [batch_size, sequence_length, num_v_heads, v_head_dim] and dtype
    core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    decay = g  # rename for clarity: argument name must stay "g" to match flash_linear_attention's API

    # Make sure all tensors are fp32 and reshape them to [batch_size, num_*_heads, seqlen, ...]
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    # If enabled, normalize query and key vectors (done once in fp32 for better accuracy)
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # And always normalize queries by the head dimension
    query = query / (query.shape[-1] ** 0.5)

    # Create the storage for the last recurrent state, which will be updated in place. If a previous state is provided,
    # it is the starting point, otherwise start with a zeroed buffer.
    if initial_state is None:
        recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=value.dtype, device=value.device)
    else:
        last_recurrent_state = initial_state.to(value)
    core_attn_out = torch.zeros_like(value)

    # Loop over each token and update the recurrent state
    for i in range(sequence_length):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        # Decay the recurrent state
        decay_t = decay[:, :, i].exp()[..., None, None]
        last_recurrent_state = last_recurrent_state * decay_t
        # Update the recurrent state with the current token
        beta_t = beta[:, :, i].unsqueeze(-1)
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # And use it to compute the attention output for the current token
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    # Discard the final state if not requested
    last_recurrent_state = None if not output_final_state else last_recurrent_state
    # Convert back to the original shape [batch_size, sequence_length, num_v_heads, v_head_dim] and dtype
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state