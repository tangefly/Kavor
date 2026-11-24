"""FlashInfer 的 block-scale FP8 GEMM(SM90 上走 DeepGEMM 后端)。

本模型 decode 时大量 GEMM 的 N 很小(down_proj 只有 5120/128 = 40 个 CTA),
自写 Triton w8a8 用默认 tile、无 split-K,SM 占用只有 35~70%。FlashInfer 的
``fp8_blockscale_gemm_sm90`` 在 M<32 用 swapAB、M>=32 用 DeepGEMM,实测
(M=1..8192 × 6 个权重形状)比 Triton w8a8 快 1.4~3.7x,小 M 也比 w8a16 的
split-K 路径快 1.2~2.7x;数值误差与 w8a8 同级。

kernel 首次调用要 JIT 编译(数分钟),之后缓存在 ~/.cache/flashinfer。所以模型
构建阶段先 warmup 一次(见 model_runner.capture_cudagraph 的 pre-capture 前向),
CUDA graph 捕获时 kernel 已编译完成。

约束(不满足时调用方回退 Triton):SM90、bf16 输入输出、fp32 weight scale、
N % 64 == 0、K % 128 == 0。
"""

import torch

from kavor.utils.logger import logger

_gemm = None
_checked = False
_available = False


def _load_gemm():
    """单独成函数,便于测试注入"未安装"的情形。"""
    from flashinfer.gemm import fp8_blockscale_gemm_sm90

    return fp8_blockscale_gemm_sm90


def flashinfer_available() -> bool:
    """Lazily import flashinfer and check the platform;结果缓存,只探测一次。"""
    global _gemm, _checked, _available
    if _checked:
        return _available
    _checked = True
    try:
        gemm = _load_gemm()
        major = torch.cuda.get_device_capability()[0]
        if major != 9:
            logger.info("FlashInfer FP8 block GEMM 需要 SM90,当前 SM%d0,回退 Triton", major)
            return False
        _gemm = gemm
        _available = True
    except Exception as exc:    # 未安装 / 版本无此 API
        logger.info("FlashInfer FP8 block GEMM 不可用(%s),回退 Triton", exc)
    return _available


def supports(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> bool:
    """本次调用能否走 FlashInfer:平台 + dtype + N/K 对齐(TP 分片后仍需满足)。"""
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
    """x[..., K] bf16 @ (W_fp8 * S)^T -> x.dtype;激活量化在 kernel 内部完成。"""
    k = x.shape[-1]
    x_2d = x.reshape(-1, k)
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    out = _gemm(x_2d, weight, weight_scale=weight_scale, out_dtype=x.dtype)
    if bias is not None:
        out = out + bias
    return out.view(*x.shape[:-1], -1)
