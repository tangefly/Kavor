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

import kavor.kernels.fp8.flashinfer as fp8_flashinfer  # noqa: E402
from kavor.kernels.fp8.triton_w8a16 import fp8_blockwise_linear  # noqa: E402
from kavor.kernels.fp8.triton_w8a8 import (  # noqa: E402
    per_token_group_quant_fp8, w8a8_block_fp8_linear, w8a8_block_fp8_matmul)
from kavor.layers.quant.fp8 import (  # noqa: E402
    Fp8BlockLinearMethod, Fp8Config)

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


def _make_method(backend):
    config = Fp8Config((128, 128), modules_to_not_convert=[], fp8_backend=backend)
    return Fp8BlockLinearMethod(config)


def _make_layer(w_q, w_s, bias=None):
    return SimpleNamespace(
        weight=w_q, weight_scale_inv=w_s, bias=bias,
        weight_block_size=(128, 128), orig_dtype=torch.bfloat16)


def test_fp8_linear_method_routing():
    """显式 backend 时 apply 走对应 kernel;两条 Triton 路径都要对齐参考实现。"""
    torch.manual_seed(0)
    M, N, K = 33, 256, 384
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=DEV) / 16
    w_q, w_s = _quant_weight_blockwise(w)

    out8 = _make_method("triton_w8a8").apply(_make_layer(w_q, w_s), x, None)
    out16 = _make_method("triton_w8a16").apply(_make_layer(w_q, w_s), x, None)

    torch.testing.assert_close(out8, w8a8_block_fp8_linear(x, w_q, w_s, (128, 128)))
    torch.testing.assert_close(out16, fp8_blockwise_linear(x, w_q, w_s, None))

    # 两条路径都应落在量化噪声量级内(相对 bf16 oracle)
    ref = F.linear(x, _dequant_blockwise(w_q, w_s))
    _assert_quant_noise(out8, ref)
    _assert_quant_noise(out16, ref)


def test_fp8_linear_method_auto_picks_one_kernel():
    """auto 在 prepare 阶段固定一个 kernel,不再按 M 分流。"""
    from kavor.kernels.fp8 import marlin_build

    torch.manual_seed(0)
    N, K = 256, 384
    w = torch.randn(N, K, dtype=torch.bfloat16, device=DEV) / 16
    w_q, w_s = _quant_weight_blockwise(w)
    method = _make_method("auto")
    layer = _make_layer(w_q, w_s)
    method.prepare_kernel(layer)

    if fp8_flashinfer.flashinfer_available():
        expected = "flashinfer"
    elif marlin_build.marlin_available()[0]:
        expected = "marlin"
    else:
        expected = "triton_w8a16"
    assert method.kernel.name == expected

    for M in (1, 32, 64, 256):
        x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
        ref = F.linear(x, _dequant_blockwise(w_q, w_s))
        _assert_quant_noise(method.apply(layer, x, None), ref)


def test_flashinfer_block_fp8_linear_matches_reference():
    if not fp8_flashinfer.flashinfer_available():
        pytest.skip("FlashInfer 不可用")
    torch.manual_seed(0)
    M, N, K = 128, 512, 512
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=DEV) / 16
    w_q, w_s = _quant_weight_blockwise(w)
    bias = torch.randn(N, dtype=torch.bfloat16, device=DEV) / 16

    out = fp8_flashinfer.flashinfer_block_fp8_linear(x, w_q, w_s, bias)
    ref = F.linear(x, _dequant_blockwise(w_q, w_s), bias)

    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    _assert_quant_noise(out, ref)


def test_flashinfer_block_fp8_linear_3d_input():
    """[B, T, hidden] 进 FlashInfer 路径要能 reshape 回 3D。"""
    if not fp8_flashinfer.flashinfer_available():
        pytest.skip("FlashInfer 不可用")
    torch.manual_seed(0)
    B, T, N, K = 2, 17, 512, 512
    x = torch.randn(B, T, K, dtype=torch.bfloat16, device=DEV)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=DEV) / 16
    w_q, w_s = _quant_weight_blockwise(w)

    out = fp8_flashinfer.flashinfer_block_fp8_linear(x, w_q, w_s)
    assert out.shape == (B, T, N) and out.dtype == torch.bfloat16
    _assert_quant_noise(out, F.linear(x, _dequant_blockwise(w_q, w_s)))


def test_flashinfer_requires_fp32_scale():
    """FlashInfer 要求 fp32 scale;bf16 scale 必须被判为不可用(回退 Triton)。"""
    x = torch.empty(8, 512, dtype=torch.bfloat16, device=DEV)
    w = torch.empty(512, 512, dtype=FP8, device=DEV)
    s_bf16 = torch.empty(4, 4, dtype=torch.bfloat16, device=DEV)
    assert not fp8_flashinfer.supports(x, w, s_bf16)


def _no_flashinfer(monkeypatch):
    """模拟"没装 flashinfer":让探测函数抛 ImportError,并清掉探测缓存。"""
    def boom():
        raise ImportError("no flashinfer")

    monkeypatch.setattr(fp8_flashinfer, "_load_gemm", boom)
    monkeypatch.setattr(fp8_flashinfer, "_checked", False)
    monkeypatch.setattr(fp8_flashinfer, "_available", False)
    monkeypatch.setattr(fp8_flashinfer, "_gemm", None)


def test_flashinfer_import_failure_falls_back(monkeypatch):
    """没装 flashinfer 时探测应返回 False,让 auto 回退到 Triton 路径。"""
    _no_flashinfer(monkeypatch)
    assert fp8_flashinfer.flashinfer_available() is False
    assert not fp8_flashinfer.supports(
        torch.empty(8, 512, dtype=torch.bfloat16, device=DEV),
        torch.empty(512, 512, dtype=FP8, device=DEV),
        torch.empty(4, 4, dtype=torch.float32, device=DEV))


def test_flashinfer_backend_requires_availability(monkeypatch):
    """fp8_backend='flashinfer' 但在没有 FlashInfer 的机器上加载要直接报错。"""
    _no_flashinfer(monkeypatch)
    torch.manual_seed(0)
    w = torch.randn(512, 512, dtype=torch.bfloat16, device=DEV) / 16
    w_q, w_s = _quant_weight_blockwise(w)

    with pytest.raises(RuntimeError, match="flashinfer"):
        _make_method("flashinfer").prepare_kernel(_make_layer(w_q, w_s))


def test_fp8_config_from_config_parses_backend():
    quant_config = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    cfg = Fp8Config.from_config(quant_config, fp8_backend="triton_w8a16")
    assert cfg.fp8_backend == "triton_w8a16"
    assert cfg.activation_scheme == "dynamic"
    assert cfg.weight_block_size == (128, 128)
    assert Fp8Config.from_config(quant_config).fp8_backend == "auto"

    assert Fp8Config.from_config(None) is None          # 非量化模型
    with pytest.raises(AssertionError):
        Fp8Config.from_config(quant_config, fp8_backend="w4a16")
