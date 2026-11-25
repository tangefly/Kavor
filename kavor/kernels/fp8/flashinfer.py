import torch

from kavor.utils.logger import logger

_gemm = None
_checked = False
_available = False


def _load_gemm():
    """Separate function so tests can simulate the "not installed" case."""
    from flashinfer.gemm import fp8_blockscale_gemm_sm90

    return fp8_blockscale_gemm_sm90


def flashinfer_available() -> bool:
    """Lazily import flashinfer and check the platform; probed once, result cached."""
    global _gemm, _checked, _available
    if _checked:
        return _available
    _checked = True
    try:
        gemm = _load_gemm()
        major = torch.cuda.get_device_capability()[0]
        if major != 9:
            logger.info(
                "FlashInfer FP8 block GEMM requires SM90, current SM%d0; falling back to Triton",
                major)
            return False
        _gemm = gemm
        _available = True
    except Exception as exc:    # not installed / this version has no such API
        logger.info(
            "FlashInfer FP8 block GEMM unavailable (%s); falling back to Triton", exc)
    return _available


def supports(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> bool:
    """Whether this call can use FlashInfer: platform + dtype + N/K alignment.

    The N/K alignment must still hold after TP sharding.
    """
    return (flashinfer_available()
            and x.dtype == torch.bfloat16
            and weight_scale.dtype == torch.float32
            and weight.shape[0] % 64 == 0
            and weight.shape[1] % 128 == 0)


def flashinfer_block_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """x[..., K] bf16 @ (W_fp8 * S)^T -> x.dtype; activation quant happens inside the kernel."""
    k = x.shape[-1]
    x_2d = x.reshape(-1, k)
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    out = _gemm(x_2d, weight, weight_scale=weight_scale, out_dtype=x.dtype)
    if bias is not None:
        out = out + bias
    return out.view(*x.shape[:-1], -1)
