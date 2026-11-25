"""fused kernel(RMSNorm / RoPE / Sampler)与 eager 参考实现的数值对齐测试。

需要 GPU;无 CUDA 时整文件跳过。
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")

from kavor.kernels.attention.gated_delta_rule import fused_recurrent_gdn_decode  # noqa: E402
from kavor.kernels.layernorm import RMSNorm  # noqa: E402
from kavor.kernels.rotary_embedding import (  # noqa: E402
    RotaryEmbedding,
    apply_rotary_emb,
    apply_rotary_emb_interleaved,
)
from kavor.kernels.sampler import Sampler  # noqa: E402

DEV = "cuda"
EPS = 1e-6


def _ref_rms(x, weight, eps):
    orig = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf.mul_(torch.rsqrt(var + eps))
    return xf.to(orig).mul_(weight)


def _ref_add_rms(x, residual, weight, eps):
    orig = x.dtype
    x = x.float().add_(residual.float())
    residual = x.to(orig)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    return x.to(orig).mul_(weight), residual


@pytest.mark.parametrize("shape,view", [
    ((128, 4096), False),
    ((1, 4096), False),
    ((37, 32, 128), True),     # q/k norm:qkv split 出来的不连续视图
    ((5, 8, 80), True),        # 非 2 的幂 head_dim
])
def test_rmsnorm_matches_reference(shape, view):
    torch.manual_seed(0)
    norm = RMSNorm(shape[-1], eps=EPS).to(DEV)
    with torch.no_grad():
        norm.weight.normal_()
    if view:
        buf = torch.randn(shape[0], shape[1] * 3, shape[2], dtype=torch.bfloat16, device=DEV)
        x = buf[:, :shape[1], :]
    else:
        x = torch.randn(shape, dtype=torch.bfloat16, device=DEV)
    ref = _ref_rms(x.clone(), norm.weight, EPS)
    out = norm(x.clone())
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def test_add_rmsnorm_matches_reference():
    torch.manual_seed(0)
    norm = RMSNorm(4096, eps=EPS).to(DEV)
    with torch.no_grad():
        norm.weight.normal_()
    x = torch.randn(256, 4096, dtype=torch.bfloat16, device=DEV)
    residual = torch.randn(256, 4096, dtype=torch.bfloat16, device=DEV)
    ref_out, ref_res = _ref_add_rms(x.clone(), residual.clone(), norm.weight, EPS)
    out, res = norm(x.clone(), residual.clone())
    assert torch.equal(res, ref_res)
    torch.testing.assert_close(out, ref_out, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("interleaved", [False, True])
def test_rope_matches_reference(interleaved):
    torch.manual_seed(0)
    head_dim, max_pos = 128, 4096
    rope = RotaryEmbedding(head_dim, head_dim, max_pos, 1000000.0,
                           interleaved=interleaved).to(DEV)
    T, Hq, Hkv = 64, 32, 8
    qkv = torch.randn(T, Hq * head_dim + 2 * Hkv * head_dim, dtype=torch.bfloat16, device=DEV)
    q = qkv[:, :Hq * head_dim].view(T, Hq, head_dim)          # 不连续视图,同真实路径
    k = qkv[:, Hq * head_dim:Hq * head_dim + Hkv * head_dim].view(T, Hkv, head_dim)
    pos = torch.randint(0, max_pos, (T,), dtype=torch.int64, device=DEV)
    cos_sin = rope.cos_sin_cache[pos]
    cos, sin = cos_sin.chunk(2, dim=-1)
    apply = apply_rotary_emb_interleaved if interleaved else apply_rotary_emb
    q_ref, k_ref = apply(q.clone(), cos, sin), apply(k.clone(), cos, sin)
    q_out, k_out = rope(pos, q.clone(), k.clone())
    torch.testing.assert_close(q_out.float(), q_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(k_out.float(), k_ref.float(), rtol=2e-2, atol=2e-2)


def test_sampler_greedy_matches_argmax():
    torch.manual_seed(0)
    sampler = Sampler()
    logits = (torch.randn(512, 151936, dtype=torch.float32, device=DEV) * 3).to(torch.bfloat16)
    temps = torch.zeros(512, dtype=torch.float32, device=DEV)
    tokens = sampler(logits, temps)
    assert tokens.dtype == torch.int64
    assert torch.equal(tokens, logits.float().argmax(dim=-1))


def test_sampler_distribution_matches_softmax():
    torch.manual_seed(0)
    sampler = Sampler()
    V, N = 50, 20000
    logits = (torch.randn(V, dtype=torch.float32, device=DEV) * 4).to(torch.bfloat16)
    probs = torch.softmax(logits.float(), -1)
    tokens = sampler(logits.unsqueeze(0).expand(N, V).contiguous(),
                     torch.ones(N, dtype=torch.float32, device=DEV))
    freq = torch.bincount(tokens, minlength=V).float() / N
    assert (freq - probs).abs().max().item() < 0.01
    assert freq.argmax().item() == probs.argmax().item()


def test_sampler_reproducible_for_same_seed():
    torch.manual_seed(0)
    sampler = Sampler()
    logits = torch.randn(1000, 4096, dtype=torch.bfloat16, device=DEV)
    temps = torch.ones(1000, dtype=torch.float32, device=DEV)
    sampler.seed.zero_()
    a = sampler(logits, temps)
    sampler.seed.zero_()
    b = sampler(logits, temps)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# GDN decode: 状态池原地更新 kernel vs 旧路径(gather -> fla kernel -> scatter)
#
# 对拍分两层:
# 1. 算术层 vs fla:递推体逐行抄自 fla 的 fused_recurrent kernel。两边编译出的
#    tl.sum 归约/FMA 调度不同,fp32 状态有 <=2 ulp(按全局最大幅值)的合法舍入差异;
#    o 是 bf16,个别元素恰在舍入边界会翻 1 个 bf16 ulp。真实 bug(寻址/头映射错)
#    产生 O(1) 级差异,远超此公差。
# 2. 寻址层 vs kernel 自身:置换对照与 graph replay(同一 cubin),必须严格 bit 级
#    相等,否则就是索引/stride/int64 寻址写错。
# ---------------------------------------------------------------------------

FP32_EPS = torch.finfo(torch.float32).eps


def _assert_ulp_close(a, b, max_ulp=2):
    """|a-b| 全局不超过 max_ulp * ulp(max|b|)。递推含相减抵消,小元素相对误差无意义,
    用全池最大幅值的 ulp 做绝对界。"""
    import math
    d = (a - b).abs().max().item()
    max_b = b.abs().max().item()
    tol = max_ulp * 2 ** (math.floor(math.log2(max_b)) - 23)
    assert d <= tol, f"fp32 状态超出 {max_ulp} ulp(max|ref|): max diff {d:.3e}, tol {tol:.3e}"


def _fake_page_pool(num_slots, hv, k, v, conv_floats=18432, pad_tail=7):
    """复刻 CachePool._bind_state_page 的布局:conv 区在页头,recurrent 区带 float 偏移,
    行 stride != HV*K*V(和真实页池一样跨块不连续)。"""
    page_floats = conv_floats + hv * k * v + pad_tail
    storage = torch.randn(num_slots, page_floats, dtype=torch.float32, device=DEV) * 0.5
    pool = storage.as_strided((num_slots, hv, k, v), (page_floats, k * v, v, 1), conv_floats)
    return storage, pool


def _decode_inputs(B, H, HV, K, V, seed=0):
    """和真实 decode 路径同构的输入:q/k/v 是 packed qkv 的切片视图(batch stride != H*K),
    g/beta 是 [B, 1, HV] 连续张量。"""
    torch.manual_seed(seed)
    conv_dim = 2 * H * K + HV * V + 5          # +5: 保证 batch stride 不是任何 head 维的倍数
    mixed = torch.randn(B, 1, conv_dim, dtype=torch.bfloat16, device=DEV)
    q = mixed[..., :H * K].reshape(B, 1, H, K)
    k = mixed[..., H * K:2 * H * K].reshape(B, 1, H, K)
    v = mixed[..., 2 * H * K:2 * H * K + HV * V].reshape(B, 1, HV, V)
    g = -torch.rand(B, 1, HV, dtype=torch.float32, device=DEV) * 0.2      # log 空间衰减
    beta = torch.rand(B, 1, HV, dtype=torch.float32, device=DEV)          # 已过 sigmoid
    return q, k, v, g, beta


def _reference_fla_decode(q, k, v, g, beta, pristine, pool, slot_ids):
    """旧路径参考:在 pool 快照上 gather -> fla fused_recurrent -> scatter。"""
    try:
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as fla_recurrent
    except ImportError:
        pytest.skip("fla 未安装,无法对拍")
    ref_storage = pristine.clone()
    ref_pool = ref_storage.as_strided(pool.shape, pool.stride(), pool.storage_offset())
    o, ht = fla_recurrent(
        q, k, v, g=g, beta=beta,
        initial_state=ref_pool[slot_ids],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    ref_pool[slot_ids] = ht
    return o, ref_storage


@pytest.mark.parametrize("B,idx_stride", [(1, 1), (3, 1), (64, 1), (64, 3)])
@pytest.mark.parametrize("H,HV,K,V", [
    (32, 32, 128, 128),    # Qwen3.5-4B TP=1 的每 rank 形状
    (16, 32, 128, 128),    # GQA 展开:H < HV,验证 i_h 映射
    (4, 4, 64, 80),        # V 非二次幂,验证 mask 路径
])
def test_gdn_decode_matches_fla(B, idx_stride, H, HV, K, V):
    num_slots = max(300, B + 10)
    storage, pool = _fake_page_pool(num_slots, HV, K, V)
    q, k, v, g, beta = _decode_inputs(B, H, HV, K, V, seed=B * 100 + H)
    ids = torch.randperm(num_slots - 1, device=DEV)[:B].to(torch.int64)
    if idx_stride > 1:
        # 模拟 linear_slot_ids[:, group] 的列切片(stride = num_state_groups)
        ids = torch.stack([ids, ids], dim=1)[:, 0]
        assert ids.stride(0) == 2
    pristine = storage.clone()

    o = fused_recurrent_gdn_decode(q, k, v, state_pool=pool, state_indices=ids, g=g, beta=beta)

    ref_o, ref_storage = _reference_fla_decode(q, k, v, g, beta, pristine, pool, ids)
    torch.testing.assert_close(o, ref_o, rtol=2 ** -7, atol=2 ** -8)   # ~1 个 bf16 ulp
    _assert_ulp_close(storage, ref_storage)


def test_gdn_decode_slot_permutation_bitwise():
    """寻址对照:同一批 (序列输入, slot) 对,按两种 batch 顺序跑,每个序列落在同一物理
    slot 上,结果必须逐位一致——这检验 slot 索引/stride/int64 寻址本身,与舍入无关。"""
    B, H, HV, K, V = 64, 32, 32, 128, 128
    storage, pool = _fake_page_pool(128, HV, K, V)
    q, k, v, g, beta = _decode_inputs(B, H, HV, K, V, seed=11)
    ids = torch.randperm(100, device=DEV)[:B].to(torch.int64)
    perm = torch.randperm(B, device=DEV)
    pristine = storage.clone()

    storage.copy_(pristine)
    o1 = fused_recurrent_gdn_decode(q, k, v, state_pool=pool, state_indices=ids, g=g, beta=beta)
    s1 = storage.clone()

    # 在 mixed/g/beta 层 gather(fancy indexing);宽度补回 +5 的 slack,保持切片视图的
    # batch stride 与 run1 同为奇数,避免触发 Triton 不同的 stride 特化
    slack = torch.zeros(B, 1, 5, dtype=q.dtype, device=DEV)
    mixed_p = torch.cat(
        [q.reshape(B, 1, -1), k.reshape(B, 1, -1), v.reshape(B, 1, -1), slack], -1)[perm]
    q2 = mixed_p[..., :H * K].reshape(B, 1, H, K)
    k2 = mixed_p[..., H * K:2 * H * K].reshape(B, 1, H, K)
    v2 = mixed_p[..., 2 * H * K:2 * H * K + HV * V].reshape(B, 1, HV, V)
    storage.copy_(pristine)
    o2 = fused_recurrent_gdn_decode(q2, k2, v2, state_pool=pool, state_indices=ids[perm], g=g[perm], beta=beta[perm])

    assert torch.equal(o2, o1[perm])
    assert torch.equal(storage, s1)


def test_gdn_decode_padding_slot_rows_do_not_corrupt_others():
    """CUDA graph 填充行共享同一个 padding slot:其余行的输出和状态必须与
    只跑真实行时一致(padding 块本身的写入次序无定义,不比较)。"""
    B, H, HV, K, V = 64, 32, 32, 128, 128
    pad_slot = 255
    storage, pool = _fake_page_pool(256, HV, K, V)
    q, k, v, g, beta = _decode_inputs(B, H, HV, K, V, seed=7)
    n_real = B // 2
    real = torch.randperm(pad_slot, device=DEV)[:n_real].to(torch.int64)
    ids = torch.cat([real, torch.full((B - n_real,), pad_slot, dtype=torch.int64, device=DEV)])
    pristine = storage.clone()

    o = fused_recurrent_gdn_decode(q, k, v, state_pool=pool, state_indices=ids, g=g, beta=beta)

    ref_o, ref_storage = _reference_fla_decode(
        q[:n_real], k[:n_real], v[:n_real], g[:n_real], beta[:n_real],
        pristine, pool, real)
    torch.testing.assert_close(o[:n_real], ref_o, rtol=2 ** -7, atol=2 ** -8)
    # 真实行在 ulp 内;padding 行写入只落在末块,不在这段区域里
    _assert_ulp_close(storage[:pad_slot], ref_storage[:pad_slot])


def test_gdn_decode_cuda_graph_replay_with_new_indices():
    """graph 安全性:capture 后只改写索引 buffer 的内容再 replay,kernel 必须按新索引
    读写状态池,且与同 kernel 的 eager 执行逐位一致(这正是 decode replay 的每步方式)。"""
    B, H, HV, K, V = 8, 32, 32, 128, 128
    storage, pool = _fake_page_pool(64, HV, K, V)
    q, k, v, g, beta = _decode_inputs(B, H, HV, K, V, seed=3)
    ids = torch.zeros(B, dtype=torch.int64, device=DEV)                 # 持久索引 buffer
    ids0 = torch.randperm(32, device=DEV)[:B].to(torch.int64)           # slot 0-31
    ids1 = (32 + torch.randperm(32, device=DEV)[:B]).to(torch.int64)    # slot 32-63,两组不相交
    pristine = storage.clone()

    ids.copy_(ids0)
    _ = fused_recurrent_gdn_decode(q, k, v, state_pool=pool, state_indices=ids, g=g, beta=beta)        # JIT 预热(捕获外)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o = fused_recurrent_gdn_decode(q, k, v, state_pool=pool, state_indices=ids, g=g, beta=beta)

    for step_ids in (ids0, ids1):
        storage.copy_(pristine)
        ids.copy_(step_ids)
        graph.replay()
        o_replay, s_replay = o.clone(), storage.clone()
        storage.copy_(pristine)
        ids.copy_(step_ids)
        o_eager = fused_recurrent_gdn_decode(q, k, v, state_pool=pool, state_indices=ids, g=g, beta=beta)
        assert torch.equal(o, o_eager)
        assert torch.equal(o, o_replay)
        assert torch.equal(storage, s_replay)


# ---------------------------------------------------------------------------
# 根因2/3 的融合算子:gemma RMSNorm、RMSNormGated、partial mrope、conv1d(update
# 原地 + varlen)、GDN gate 核内计算 —— 全部与原 eager 实现对拍。
# ---------------------------------------------------------------------------

from kavor.kernels.attention.causal_conv1d import (  # noqa: E402
    causal_conv1d_update_inplace,
    varlen_causal_conv1d,
)
from kavor.kernels.layernorm import RMSNormGated, rmsnorm_gemma  # noqa: E402
from kavor.kernels.rotary_embedding import apply_mrope_partial  # noqa: E402


def _ref_gemma_rms(x, w, eps):
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return (h * (1.0 + w.float())).type_as(x)


def _ref_gemma_rms_gated(x, w, gate, eps):
    input_dtype = x.dtype
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    h = w * h.to(input_dtype)
    return (h * F.silu(gate.float())).to(input_dtype)


@pytest.mark.parametrize("shape,view", [
    ((128, 2560), False),          # decode/prefill 的 hidden norm
    ((37, 16, 256), True),         # q/k norm:qkv split 出来的不连续视图
    ((1, 16, 256), False),
])
def test_gemma_rmsnorm_matches_eager(shape, view):
    torch.manual_seed(1)
    w = (torch.rand(shape[-1], dtype=torch.float32, device=DEV) - 0.5) * 0.2
    if view:
        buf = torch.randn(shape[0], shape[1] * 3, shape[2], dtype=torch.bfloat16, device=DEV)
        x = buf[:, :shape[1], :]
    else:
        x = torch.randn(shape, dtype=torch.bfloat16, device=DEV)
    torch.testing.assert_close(rmsnorm_gemma(x.clone(), w, EPS),
                               _ref_gemma_rms(x, w, EPS), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("rows", [1, 37, 2048])
def test_rmsnorm_gated_matches_eager(rows):
    torch.manual_seed(2)
    D = 128
    w = (torch.rand(D, dtype=torch.float32, device=DEV) - 0.5) * 0.2
    norm = RMSNormGated(D, eps=EPS).to(DEV)
    with torch.no_grad():
        norm.weight.copy_(w)
    x = torch.randn(rows, D, dtype=torch.bfloat16, device=DEV)
    gate = torch.randn(rows, D, dtype=torch.bfloat16, device=DEV)
    torch.testing.assert_close(norm(x.clone(), gate.clone()),
                               _ref_gemma_rms_gated(x, w, gate, EPS),
                               rtol=2e-2, atol=2e-2)


def _mrope_tables(rotary_dim, base, mrope_section):
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
    freq_row = torch.zeros(inv_freq.numel(), dtype=torch.long)
    for dim, offset in enumerate((1, 2), start=1):
        freq_row[offset: mrope_section[dim] * 3: 3] = dim
    return inv_freq.to(DEV), freq_row.to(DEV)


def _ref_mrope(positions, q, k, inv_freq, freq_row, rotary_dim):
    if positions.dim() == 1:
        positions = positions.unsqueeze(0).expand(3, -1)
    freqs = positions.float()[:, :, None] * inv_freq[None, None, :]
    picked = freqs[0]
    for row in (1, 2):
        picked = torch.where(freq_row == row, freqs[row], picked)
    cos, sin = picked.cos().unsqueeze(1), picked.sin().unsqueeze(1)

    def rot(x):
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        x1, x2 = torch.chunk(x_rot.float(), 2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.cat([torch.cat((y1, y2), -1).to(x.dtype), x_pass], -1)

    return rot(q.clone()), rot(k.clone())


@pytest.mark.parametrize("three_stream", [False, True])
def test_mrope_partial_matches_eager(three_stream):
    torch.manual_seed(3)
    head_dim, rotary_dim = 256, 64          # partial_rotary_factor=0.25
    Hq, Hkv, T = 16, 4, 64
    inv_freq, freq_row = _mrope_tables(rotary_dim, 1e7, [11, 11, 10])
    qkv = torch.randn(T, (2 * Hq + Hkv) * head_dim, dtype=torch.bfloat16, device=DEV)
    q = qkv[:, :Hq * head_dim].view(T, Hq, head_dim)      # 不连续视图,同真实路径
    k = qkv[:, Hq * head_dim:Hq * head_dim + Hkv * head_dim].view(T, Hkv, head_dim)
    if three_stream:
        positions = torch.randint(0, 4096, (3, T), dtype=torch.int64, device=DEV)
    else:
        positions = torch.randint(0, 4096, (T,), dtype=torch.int64, device=DEV)

    q_ref, k_ref = _ref_mrope(positions, q, k, inv_freq, freq_row, rotary_dim)
    apply_mrope_partial(positions, q, k, inv_freq, freq_row, rotary_dim)
    torch.testing.assert_close(q, q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(k, k_ref, rtol=2e-2, atol=2e-2)


def _ref_conv_update(x, conv_state, weight, activation):
    """旧 eager 路径:x [B, C],conv_state 是可变副本 [B, C, K-1],weight [C, K]。"""
    B, C = x.shape
    K = weight.shape[-1]
    new = torch.cat([conv_state, x.unsqueeze(-1)], -1).to(weight.dtype)
    conv_state.copy_(new[:, :, -(K - 1):])
    out = F.conv1d(new, weight.view(C, 1, K), padding=0, groups=C)[:, :, -1:]
    if activation:
        out = F.silu(out)
    return out.squeeze(-1).to(x.dtype)


def test_causal_conv1d_update_inplace_matches_eager():
    torch.manual_seed(4)
    B, C, K = 64, 8192, 4
    num_slots, group = 128, 3
    # 复刻页池布局:行 stride = C*K + 尾部 padding,索引是 [B, G] 的列切片
    page = C * K + 13
    storage = torch.randn(num_slots, page, dtype=torch.bfloat16, device=DEV) * 0.5
    pool = storage.as_strided((num_slots, C, K - 1), (page, K - 1, 1), 5)
    ids = torch.stack([torch.randperm(num_slots - 1, device=DEV)[:B].to(torch.int64)] * group, 1)[:, 0]
    assert ids.stride(0) == group

    x = torch.randn(B, C, dtype=torch.bfloat16, device=DEV)
    weight = (torch.randn(C, K, dtype=torch.bfloat16, device=DEV) * 0.1)
    pristine = storage.clone()

    out = causal_conv1d_update_inplace(x, pool, ids, weight, None, "silu")

    ref_pool = pristine.clone().as_strided(pool.shape, pool.stride(), pool.storage_offset())
    ref_state = ref_pool[ids].clone()          # gather 出副本再让 eager 原地更新
    ref_out = _ref_conv_update(x, ref_state, weight, True)
    ref_pool[ids] = ref_state
    torch.testing.assert_close(out, ref_out, rtol=2e-2, atol=2e-2)
    # 池逐元素一致:被更新的 slot 严格相等,未触碰的 slot 也应原样
    torch.testing.assert_close(pool[ids], ref_pool[ids], rtol=0, atol=0)
    untouched = torch.tensor(
        [i for i in range(num_slots) if i not in set(ids.tolist())],
        dtype=torch.int64, device=DEV)
    torch.testing.assert_close(pool[untouched], ref_pool[untouched], rtol=0, atol=0)


def _ref_varlen_conv(x, cu, states, weight, activation):
    outs, new_states = [], []
    K = weight.shape[-1]
    N = len(cu) - 1
    for i in range(N):
        seg = x[cu[i]:cu[i + 1]].t()                          # [C, T]
        cat = torch.cat([states[i], seg], -1).to(weight.dtype)
        out = F.conv1d(cat.unsqueeze(0), weight.view(-1, 1, K), padding=0,
                       groups=weight.shape[0])[0].t()
        if activation:
            out = F.silu(out)
        outs.append(out)
        new_states.append(cat[:, -(K - 1):])
    return torch.cat(outs, 0), torch.stack(new_states)


@pytest.mark.parametrize("lens", [
    [200, 37, 512],             # 变长混合
    [1, 2],                     # 段比 kernel 还短:状态左补
    [4096],                     # 单条长序列
])
def test_varlen_causal_conv1d_matches_eager(lens):
    torch.manual_seed(5)
    C, K = 1024, 4
    total = sum(lens)
    N = len(lens)
    x = torch.randn(total, C, dtype=torch.bfloat16, device=DEV)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=DEV)
    states = torch.randn(N, C, K - 1, dtype=torch.bfloat16, device=DEV) * 0.3
    weight = torch.randn(C, K, dtype=torch.bfloat16, device=DEV) * 0.1

    out = varlen_causal_conv1d(x, cu, states, weight, None, "silu",
                               max_len=max(lens))
    ref_out, _ = _ref_varlen_conv(x, cu.tolist(), states, weight, True)
    torch.testing.assert_close(out, ref_out, rtol=2e-2, atol=2e-2)


def test_gdn_decode_gate_in_kernel_matches_precomputed():
    """USE_GATE_IN_KERNEL(a/b 原始投影)与 host 预计算 g/beta 两条路应一致。"""
    torch.manual_seed(6)
    B, H, HV, K, V = 64, 16, 32, 128, 128
    storage, pool = _fake_page_pool(96, HV, K, V)
    q, k, v, _, _ = _decode_inputs(B, H, HV, K, V, seed=9)
    a = torch.randn(B, 1, HV, dtype=torch.bfloat16, device=DEV)
    b = torch.randn(B, 1, HV, dtype=torch.bfloat16, device=DEV)
    A_log = (torch.rand(HV, dtype=torch.float32, device=DEV) * 2)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=DEV)
    g = -A_log.exp() * F.softplus(a.float() + dt_bias)
    # kernel 全程 fp32;旧 eager 的 sigmoid(bf16) 会把 beta 舍回 bf16(差 ~2e-3),
    # 这里用 fp32 参考对齐 kernel 的数学定义
    beta = torch.sigmoid(b.float())
    ids = torch.randperm(95, device=DEV)[:B].to(torch.int64)

    snap = storage.clone()
    o_gate = fused_recurrent_gdn_decode(q, k, v, state_pool=pool, state_indices=ids,
                                        a=a, b=b, A_log=A_log, dt_bias=dt_bias)
    s_gate = storage.clone()
    storage.copy_(snap)
    o_ref = fused_recurrent_gdn_decode(q, k, v, state_pool=pool, state_indices=ids,
                                       g=g, beta=beta)
    torch.testing.assert_close(o_gate, o_ref, rtol=2e-2, atol=2e-2)
    _assert_ulp_close(s_gate, storage)
