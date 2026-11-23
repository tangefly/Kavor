"""W8A8 FP8 kernel(vendor 自 vllm 的 fp8_utils)与参考实现的数值对齐测试。

覆盖:激活动态量化(per-token per-128-group)、blockwise FP8×FP8 GEMM、
端到端 w8a8_block_fp8_linear,以及 Fp8BlockLinearMethod 的 w8a8/w8a16 分流。

需要 GPU;无 CUDA 时整文件跳过。
"""

import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")

from kavor.layers.fp8_kernel import fp8_blockwise_linear  # noqa: E402
from kavor.layers.fp8_utils import (  # noqa: E402
    per_token_group_quant_fp8, w8a8_block_fp8_linear, w8a8_block_fp8_matmul)
from kavor.layers.quant import Fp8BlockLinearMethod, Fp8Config  # noqa: E402

DEV = "cuda"
FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max      # 448
EPS = 1e-10


def _assert_quant_noise(out, ref, thresh=5e-2):
    """激活量化噪声的正确度量:相对 Frobenius 误差。

    e4m3 只有 3 bit 尾数,逐元素相对舍入误差 2~6%,点积累计约 2~3% 的
    相对误差,且近零元素会让逐元素 rtol 失真,所以不用 assert_close。
    """
    err = (out.float() - ref.float()).norm() / ref.float().norm()
    assert err < thresh, f"相对 Frobenius 误差 {err:.4f} 超出量化噪声量级"


def _ref_group_quant(x, group_size=128, eps=EPS):
    """per-token per-group 量化参考:scale = max(absmax, eps) / 448。"""
    M, K = x.shape
    xf = x.float().view(M, K // group_size, group_size)
    absmax = xf.abs().amax(dim=-1)
    scale = torch.maximum(absmax, torch.full_like(absmax, eps)) / FP8_MAX
    q = (xf / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q.view(M, K), scale


def _quant_weight_blockwise(w, block=(128, 128), eps=EPS):
    """构造 checkpoint 形态的块量化权重:W_fp8 [N, K] + scale [N/128, K/128]。"""
    N, K = w.shape
    wf = w.float().view(N // block[0], block[0], K // block[1], block[1])
    scale = torch.maximum(wf.abs().amax(dim=(1, 3)),
                          torch.full((1,), eps, device=w.device)) / FP8_MAX
    q = (wf / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q.view(N, K), scale


def _dequant_blockwise(q, scale, block=(128, 128)):
    """W_fp8 * scale_inv -> bf16,块内一个标量。"""
    N, K = q.shape
    out = q.float().view(N // block[0], block[0], K // block[1], block[1])
    return (out * scale[:, None, :, None]).view(N, K).to(torch.bfloat16)


@pytest.mark.parametrize("shape", [(1, 256), (37, 512), (128, 384)])
def test_per_token_group_quant_fp8(shape):
    torch.manual_seed(0)
    M, K = shape
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    x[0, :128] = 0.0        # 全零 group:走 eps 下限,不除零
    q_ref, s_ref = _ref_group_quant(x)

    q, s = per_token_group_quant_fp8(x.contiguous(), 128)

    torch.testing.assert_close(s, s_ref, rtol=0, atol=0)
    assert (q.view(torch.uint8) == q_ref.view(torch.uint8)).all(), (
        "量化值应与逐元素参考比特级一致(同样的 fp32 公式 + RTNE)")


@pytest.mark.parametrize("M,N,K", [
    (1, 256, 512),    # decode
    (7, 384, 256),    # M 非整块:kernel 的 % M 回绕路径
    (64, 256, 384),
    (1024, 256, 256),  # prefill 量级
])
def test_w8a8_block_fp8_matmul_matches_dequant_reference(M, N, K):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=DEV) / 16
    w_q, w_s = _quant_weight_blockwise(w)
    x_q, x_s = per_token_group_quant_fp8(x.contiguous(), 128)

    out = w8a8_block_fp8_matmul(x_q, w_q, x_s, w_s, [128, 128],
                                output_dtype=torch.bfloat16)

    # oracle:量化后的输入/权重各自反量化到 fp32 再 matmul,只检验 GEMM 数值
    ref = (x_q.float() * x_s.repeat_interleave(128, dim=-1)) @ \
        _dequant_blockwise(w_q, w_s).float().T
    torch.testing.assert_close(out, ref.to(torch.bfloat16), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("M,N,K", [
    (1, 256, 512),     # decode
    (7, 256, 384),     # M 非整块
    (256, 512, 256),   # prefill
])
def test_w8a8_block_fp8_linear_end_to_end(M, N, K):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=DEV) / 16
    w_q, w_s = _quant_weight_blockwise(w)
    bias = torch.randn(N, dtype=torch.bfloat16, device=DEV) / 16

    out = w8a8_block_fp8_linear(x, w_q, w_s, (128, 128), bias)

    # oracle:反量化权重 + bf16 F.linear;误差应落在激活量化噪声量级
    ref = F.linear(x, _dequant_blockwise(w_q, w_s), bias)
    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    _assert_quant_noise(out, ref)


def test_w8a8_block_fp8_linear_3d_input():
    """非 2D 激活(qkv/gate_up 前的 [B, T, hidden])走 reshape 路径。"""
    torch.manual_seed(0)
    B, T, K, N = 2, 17, 256, 512
    x = torch.randn(B, T, K, dtype=torch.bfloat16, device=DEV)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=DEV) / 16
    w_q, w_s = _quant_weight_blockwise(w)

    out = w8a8_block_fp8_linear(x, w_q, w_s, (128, 128))
    assert out.shape == (B, T, N) and out.dtype == torch.bfloat16

    ref = F.linear(x, _dequant_blockwise(w_q, w_s))
    _assert_quant_noise(out, ref)


def _make_method(fp8_mode):
    config = Fp8Config((128, 128), modules_to_not_convert=[], fp8_mode=fp8_mode)
    return Fp8BlockLinearMethod(config)


def test_fp8_linear_method_routing():
    """fp8_mode 决定 apply 走 W8A8(vendored)还是 W8A16(kavor kernel)。"""
    torch.manual_seed(0)
    M, N, K = 33, 256, 384
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=DEV) / 16
    w_q, w_s = _quant_weight_blockwise(w)
    layer = SimpleNamespace(weight=w_q, weight_scale_inv=w_s)

    out8 = _make_method("w8a8").apply(layer, x, None)
    out16 = _make_method("w8a16").apply(layer, x, None)

    torch.testing.assert_close(out8, w8a8_block_fp8_linear(x, w_q, w_s, (128, 128)))
    torch.testing.assert_close(
        out16, fp8_blockwise_linear(x, w_q, w_s, None))

    # 两条路径都应落在量化噪声量级内(相对 bf16 oracle)
    ref = F.linear(x, _dequant_blockwise(w_q, w_s))
    _assert_quant_noise(out8, ref)
    _assert_quant_noise(out16, ref)


def test_fp8_config_from_config_parses_mode():
    quant_config = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    cfg = Fp8Config.from_config(quant_config, fp8_mode="w8a16")
    assert cfg.fp8_mode == "w8a16"
    assert cfg.activation_scheme == "dynamic"
    assert cfg.weight_block_size == (128, 128)

    assert Fp8Config.from_config(None) is None          # 非量化模型
    with pytest.raises(AssertionError):
        Fp8Config.from_config(quant_config, fp8_mode="w4a16")
