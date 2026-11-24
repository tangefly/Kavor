"""把 Kavor checkpoint 形态的 FP8 权重/scale 转成 Marlin 期望的布局。

输入(``Fp8BlockLinearMethod.create_weights`` 建立、weight_loader 填好):

- ``layer.weight``: ``[N, K]`` ``float8_e4m3fn``
- ``layer.weight_scale_inv``: ``[N/128, K/128]`` ``float32``
- ``layer.bias``: 可选 ``[N]``

Marlin 需要:

- weight: ``gptq_marlin_repack`` 后的 int32 ``[K/16, 4N]``(K/N 可零补到
  thread tile 对齐)
- scale: ``[K/128, N]`` 的 group-wise 布局,转成计算 dtype、乘指数偏移
  (bf16 ``2**120`` / fp16 ``2**8``)后再按 Marlin 的 tile 顺序 permute
- bias: 按 N 补零后同样 permute

这里只做布局转换;选哪个 kernel、何时调用由 registry 决定。
"""

import math

import torch
import torch.nn.functional as F

from kavor.kernels.fp8 import marlin_build


def pack_fp8_to_int32(fp8_tensor: torch.Tensor) -> torch.Tensor:
    """``[N, K]`` fp8 → ``[K/4, N]`` int32(K 方向 4 个 fp8 打包成一个 int32)。"""
    assert fp8_tensor.dtype == torch.float8_e4m3fn
    assert fp8_tensor.ndim == 2
    n, k = fp8_tensor.shape
    assert k % 4 == 0
    return fp8_tensor.contiguous().view(torch.int32).T.contiguous()


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def padded_nk(size_n: int, size_k: int, group_size: int) -> tuple[int, int]:
    """满足 Marlin thread tile 的最小 (padded_n, padded_k)。

    Marlin 要求 ``(n%64, k%128)`` 或 ``(n%128, k%64)``;都不满足时补到面积更小
    的那一组。K 保持是 group_size 的整数倍,scale 的 group 数才不会被破坏。
    补出来的权重是 0(fp8 的 0 解码也是 0),对结果没有贡献。
    """
    group = group_size if group_size > 0 else 1
    candidates = (
        (_round_up(size_n, 64), _round_up(size_k, math.lcm(128, group))),
        (_round_up(size_n, 128), _round_up(size_k, math.lcm(64, group))),
    )
    return min(candidates, key=lambda nk: (nk[0] * nk[1], nk[0] + nk[1]))


def pad_dim(x: torch.Tensor, size: int, padded: int) -> torch.Tensor:
    if padded == size:
        return x
    return F.pad(x, (0, padded - size))


def pad_qweight(
    qweight: torch.Tensor, size_n: int, size_k: int, padded_n: int, padded_k: int
) -> torch.Tensor:
    """``[K/4, N]`` int32 零补到 ``[padded_k/4, padded_n]``。"""
    if (padded_n, padded_k) == (size_n, size_k):
        return qweight
    pack_factor = size_k // qweight.size(0)
    return F.pad(
        qweight, (0, padded_n - size_n, 0, (padded_k - size_k) // pack_factor)
    )


def pad_scales(
    scales: torch.Tensor,
    size_n: int,
    size_k: int,
    padded_n: int,
    padded_k: int,
    group_size: int,
) -> torch.Tensor:
    if (padded_n, padded_k) == (size_n, size_k):
        return scales
    pad_rows = padded_k // group_size - scales.size(0) if group_size > 0 else 0
    assert pad_rows >= 0
    return F.pad(scales, (0, padded_n - size_n, 0, pad_rows))


def _scale_perms() -> tuple[list[int], list[int]]:
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int
) -> torch.Tensor:
    scale_perm, scale_perm_single = _scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape((-1, size_n)).contiguous()


def permute_bias(s: torch.Tensor) -> torch.Tensor:
    origin_shape = s.shape
    _, scale_perm_single = _scale_perms()
    s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape(*origin_shape).contiguous()


def fused_exponent_bias_into_scales(
    scales: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """把 FP8 的 exponent bias 折进 scale,省掉 kernel 内每次反量化的一次乘。

    fp8 e4m3 的 exponent bias 是 2**3;bf16 的指数域比它多 2**7 - 2**3 = 120,
    fp16 多 2**4 - 2**3 = 8。所以 bf16 乘 2**120,fp16 乘 2**8。
    """
    fp8_exponent = 4
    target_exponent = 5 if dtype == torch.float16 else 8
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp8_exponent - 1)
    return scales * float(2**exponent_bias)


def prepare_layer(
    layer: torch.nn.Module, weight_block_size: tuple[int, int], orig_dtype: torch.dtype
) -> tuple[int, int, int, int]:
    """原地替换 layer 的 weight / weight_scale_inv / bias,返回尺寸元信息。

    返回 ``(size_n, size_k, padded_n, padded_k)``;调用方把它存到 layer 上,
    forward 时只做 reshape / 补零 / 切片。
    """
    weight = layer.weight
    scale = layer.weight_scale_inv
    assert weight.dtype == torch.float8_e4m3fn
    assert weight.ndim == 2
    size_n, size_k = weight.shape
    block_n, block_k = weight_block_size
    group_size = block_k
    assert scale.shape == (size_n // block_n, size_k // block_k), (
        f"weight_scale_inv {tuple(scale.shape)} 与权重 "
        f"{tuple(weight.shape)}/block {weight_block_size} 不匹配")

    padded_n, padded_k = padded_nk(size_n, size_k, group_size)
    device = weight.device

    # --- weight: fp8 [N,K] -> int32 [K/16, 4N] ---
    qweight = pack_fp8_to_int32(weight)
    qweight = pad_qweight(qweight, size_n, size_k, padded_n, padded_k)
    perm = torch.empty(0, dtype=torch.int32, device=device)
    ops = marlin_build.load_marlin()
    marlin_weight = ops.gptq_marlin_repack(
        qweight, perm, padded_k, padded_n, 8, False
    )
    _replace(layer, "weight", marlin_weight)

    # --- scale: fp32 [N/128, K/128] -> group-wise [K/128, N] (**padded_k/128, padded_n) ---
    scales = scale.to(orig_dtype).T.contiguous()
    scales = scales.repeat_interleave(block_n, dim=1)
    scales = scales[:, :size_n]
    scales = pad_scales(scales, size_n, size_k, padded_n, padded_k, group_size)
    marlin_scales = permute_scales(scales, padded_k, padded_n, group_size)
    marlin_scales = fused_exponent_bias_into_scales(marlin_scales, orig_dtype)
    _replace(layer, "weight_scale_inv", marlin_scales)

    # --- bias: pad + permute ---
    bias = getattr(layer, "bias", None)
    if bias is not None:
        assert bias.shape == (size_n,)
        _replace(layer, "bias", permute_bias(pad_dim(bias, size_n, padded_n)))

    return size_n, size_k, padded_n, padded_k


def _replace(layer: torch.nn.Module, name: str, data: torch.Tensor) -> None:
    """替换参数但保留 ``weight_loader``,以便需要时还能重新灌权重。"""
    old = getattr(layer, name)
    param = torch.nn.Parameter(data, requires_grad=False)
    if hasattr(old, "weight_loader"):
        param.weight_loader = old.weight_loader
    setattr(layer, name, param)
