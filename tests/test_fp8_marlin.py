"""Marlin W8A16 的数值验证(需要 GPU)。

用 checkpoint 形态的 128×128 blockwise FP8 权重构造 layer,走完整的
``Fp8BlockLinearMethod.prepare_kernel`` + ``apply`` 路径,再与
dequant oracle 比较 Frobenius 相对误差。

默认会跑;设置 ``KAVOR_MARLIN_DISABLE=1`` 或没有 CUDA 时跳过。
首次运行需要 JIT 编译 Marlin(几十秒),之后命中缓存。
"""

from types import SimpleNamespace

import os

import pytest
import torch

from kavor.layers.quant.fp8 import Fp8BlockLinearMethod

DEV = "cuda"
FP8 = torch.float8_e4m3fn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or os.environ.get("KAVOR_MARLIN_DISABLE", "") not in ("", "0"),
    reason="需要 CUDA 且未禁用 Marlin",
)


def _quant_blockwise(w: torch.Tensor):
    """``[N,K]`` fp32 → ``([N,K] fp8, [N/128,K/128] fp32)`` blockwise 量化。"""
    n, k = w.shape
    wf = w.float().view(n // 128, 128, k // 128, 128)
    scale = (wf.abs().amax(dim=(1, 3)) / 448).clamp_min(1e-10)
    q = (wf / scale[:, None, :, None]).to(FP8).view(n, k)
    return q, scale


def _dequant_blockwise(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    n, k = q.shape
    return (q.float().view(n // 128, 128, k // 128, 128)
            * scale[:, None, :, None]).view(n, k)


def _make_layer(n, k, dtype=torch.bfloat16, bias=True, seed=0):
    """构造一个只有量化 Linear 所需字段的 layer(避开分布式初始化)。"""
    torch.manual_seed(seed)
    w = torch.randn(n, k, device=DEV) / 16
    q, scale = _quant_blockwise(w)
    b = torch.randn(n, device=DEV, dtype=dtype) / 16 if bias else None
    layer = SimpleNamespace(
        weight=torch.nn.Parameter(q, requires_grad=False),
        weight_scale_inv=torch.nn.Parameter(scale, requires_grad=False),
        bias=torch.nn.Parameter(b, requires_grad=False) if b is not None else None,
        weight_block_size=(128, 128),
        orig_dtype=dtype,
    )
    return layer, q, scale, b


def _method(backend="marlin"):
    return Fp8BlockLinearMethod(
        SimpleNamespace(weight_block_size=(128, 128), fp8_backend=backend))


def _oracle(x, q, scale, bias):
    n, k = q.shape
    ref = x.float().reshape(-1, k) @ _dequant_blockwise(q, scale).T
    if bias is not None:
        ref = ref + bias.float()
    return ref.reshape(x.shape[:-1] + (n,))


def _check(out, ref, thresh):
    err = ((out.float() - ref).norm() / ref.norm()).item()
    assert err < thresh, f"相对 Frobenius 误差 {err:.4f} 超过 {thresh}"


@pytest.mark.parametrize("m", [1, 7, 32, 33, 128, 1024])
@pytest.mark.parametrize("n,k", [(256, 512), (512, 5120), (17408, 5120)])
def test_marlin_matches_dequant_oracle(m, n, k):
    layer, q, scale, b = _make_layer(n, k)
    method = _method()
    method.prepare_kernel(layer)
    assert method.kernel.name == "marlin"

    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    out = method.apply(layer, x, layer.bias)
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    _check(out, _oracle(x, q, scale, b), 1.5e-2)


def test_marlin_3d_input():
    layer, q, scale, b = _make_layer(512, 5120)
    method = _method()
    method.prepare_kernel(layer)

    x = torch.randn(2, 7, 5120, device=DEV, dtype=torch.bfloat16)
    out = method.apply(layer, x, layer.bias)
    assert out.shape == (2, 7, 512)
    _check(out, _oracle(x, q, scale, b), 1.5e-2)


def test_marlin_without_bias_and_lazy_prepare():
    """没显式 prepare 时 apply 会懒触发一次;无 bias 路径也要正确。"""
    layer, q, scale, _ = _make_layer(256, 512, bias=False)
    method = _method()

    x = torch.randn(33, 512, device=DEV, dtype=torch.bfloat16)
    out = method.apply(layer, x, None)
    assert method.kernel is not None
    _check(out, _oracle(x, q, scale, None), 1.5e-2)


def test_marlin_fp16():
    layer, q, scale, b = _make_layer(256, 512, dtype=torch.float16)
    method = _method()
    method.prepare_kernel(layer)

    x = torch.randn(7, 512, device=DEV, dtype=torch.float16)
    out = method.apply(layer, x, layer.bias)
    assert out.dtype == torch.float16
    _check(out, _oracle(x, q, scale, b), 5e-3)


def test_marlin_prepare_layout_and_repeated_apply():
    """prepare 后 weight/scale/bias 都换成 Marlin 布局,meta 记录原始形状。"""
    layer, _, _, _ = _make_layer(256, 512)
    method = _method()
    method.prepare_kernel(layer)

    n, k, padded_n, padded_k = layer._marlin_meta
    assert (n, k, padded_n, padded_k) == (256, 512, 256, 512)
    assert layer.weight.dtype == torch.int32
    assert layer.weight.shape == (k // 16, 4 * n)
    assert layer.weight_scale_inv.dtype == torch.bfloat16
    assert layer.weight_scale_inv.shape == (padded_k // 128, padded_n)
    # 再次调用不应破坏已经准备好的布局
    out1 = method.apply(layer, torch.randn(4, k, device=DEV, dtype=torch.bfloat16),
                        layer.bias)
    out2 = method.apply(layer, torch.randn(4, k, device=DEV, dtype=torch.bfloat16),
                        layer.bias)
    assert out1.shape == out2.shape == (4, n)
