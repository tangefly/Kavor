import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

import kavor.kernels.fp8.flashinfer as fp8_flashinfer
from kavor.kernels.fp8.marlin import MarlinKernel
from kavor.kernels.fp8.triton_w8a8 import w8a8_block_fp8_linear
from kavor.kernels.fp8.triton_w8a16 import fp8_blockwise_linear
from kavor.utils.logger import logger

BACKENDS = ("auto", "flashinfer", "marlin", "triton_w8a16", "triton_w8a8")


@dataclass(frozen=True)
class Fp8LinearConfig:
    n: int
    k: int
    block_n: int
    block_k: int
    input_dtype: torch.dtype
    has_bias: bool


class Fp8Kernel(ABC):
    name: str = ""
    priority: int = 0
    # W8A8: fp8 weights x fp8 activations, W8A16: fp8 weights x bf16 activations.
    scheme: str = ""

    @classmethod
    @abstractmethod
    def is_available(cls, cfg: Fp8LinearConfig) -> tuple[bool, str]:
        """硬件/安装层面是否可用(可能触发 Marlin 的 JIT 构建)。"""

    @classmethod
    @abstractmethod
    def can_implement(cls, cfg: Fp8LinearConfig) -> tuple[bool, str]:
        """给定形状/数据类型是否支持,不触发构建。"""

    @abstractmethod
    def prepare(self, layer) -> None:
        """加载期一次性重排权重/scale(不需要重排的实现留空)。"""

    @abstractmethod
    def apply(self, layer, x: torch.Tensor, bias) -> torch.Tensor:
        """forward。"""


class FlashInferKernel(Fp8Kernel):
    name = "flashinfer"
    priority = 100
    scheme = "W8A8"

    @classmethod
    def is_available(cls, cfg):
        if not torch.cuda.is_available():
            return False, "需要 CUDA"
        if not fp8_flashinfer.flashinfer_available():
            return False, "FlashInfer 不可用(未安装或非 SM90)"
        return True, ""

    @classmethod
    def can_implement(cls, cfg):
        if cfg.input_dtype != torch.bfloat16:
            return False, f"FlashInfer 需要 bf16 激活,当前 {cfg.input_dtype}"
        if cfg.n % 64 != 0 or cfg.k % 128 != 0:
            return False, f"FlashInfer 要求 N%64==0 且 K%128==0,当前 N={cfg.n} K={cfg.k}"
        return True, ""

    def prepare(self, layer):
        pass

    def apply(self, layer, x, bias):
        return fp8_flashinfer.flashinfer_block_fp8_linear(
            x, layer.weight, layer.weight_scale_inv, bias)


class TritonW8A16Kernel(Fp8Kernel):
    name = "triton_w8a16"
    priority = 50
    scheme = "W8A16"

    @classmethod
    def is_available(cls, cfg):
        return (True, "") if torch.cuda.is_available() else (False, "需要 CUDA")

    @classmethod
    def can_implement(cls, cfg):
        if cfg.input_dtype != torch.bfloat16:
            return False, f"triton_w8a16 需要 bf16 激活,当前 {cfg.input_dtype}"
        if cfg.block_n != 128 or cfg.block_k != 128:
            return False, f"triton_w8a16 需要 128x128 block,当前 {cfg.block_n}x{cfg.block_k}"
        return True, ""

    def prepare(self, layer):
        pass

    def apply(self, layer, x, bias):
        return fp8_blockwise_linear(
            x, layer.weight, layer.weight_scale_inv, bias)


class TritonW8A8Kernel(Fp8Kernel):
    name = "triton_w8a8"
    priority = 40
    scheme = "W8A8"

    @classmethod
    def is_available(cls, cfg):
        if not torch.cuda.is_available():
            return False, "需要 CUDA"
        cc = torch.cuda.get_device_capability()
        if cc[0] * 10 + cc[1] < 89:
            return False, f"triton_w8a8 需要 SM89+,当前 SM{cc[0]}{cc[1]}"
        return True, ""

    @classmethod
    def can_implement(cls, cfg):
        if cfg.input_dtype != torch.bfloat16:
            return False, f"triton_w8a8 需要 bf16 激活,当前 {cfg.input_dtype}"
        if cfg.block_n != 128 or cfg.block_k != 128:
            return False, f"triton_w8a8 需要 128x128 block,当前 {cfg.block_n}x{cfg.block_k}"
        return True, ""

    def prepare(self, layer):
        pass

    def apply(self, layer, x, bias):
        return w8a8_block_fp8_linear(
            x, layer.weight, layer.weight_scale_inv, layer.weight_block_size, bias)


_ORDERED = (FlashInferKernel, MarlinKernel, TritonW8A16Kernel, TritonW8A8Kernel)
_BY_NAME = {kernel.name: kernel for kernel in _ORDERED}


def _disabled() -> set[str]:
    raw = os.environ.get("KAVOR_DISABLED_KERNELS", "")
    return {name.strip() for name in raw.split(",") if name.strip()}


def select_kernel(cfg: Fp8LinearConfig, backend: str, prefix: str = "") -> Fp8Kernel:
    if backend not in BACKENDS:
        raise ValueError(
            f"未知 fp8_backend={backend!r},可选: {', '.join(BACKENDS)}")
    disabled = _disabled()

    if backend == "auto":
        candidates = sorted(_ORDERED, key=lambda k: -k.priority)
    else:
        candidates = [_BY_NAME[backend]]

    reasons: list[str] = []
    for cls in candidates:
        if cls.name in disabled:
            reasons.append(f"{cls.name}: 被 KAVOR_DISABLED_KERNELS 禁用")
            continue
        ok, why = cls.can_implement(cfg)
        if not ok:
            reasons.append(f"{cls.name}: {why}")
            continue
        ok, why = cls.is_available(cfg)
        if not ok:
            reasons.append(f"{cls.name}: {why}")
            continue
        kernel = cls()
        logger.debug(
            "%sSelected fp8 kernel %s (N=%d K=%d dtype=%s)",
            f"{prefix}: " if prefix else "", kernel.name, cfg.n, cfg.k,
            cfg.input_dtype)
        return kernel

    detail = "\n  ".join(reasons) if reasons else "(无候选)"
    if backend == "auto":
        raise RuntimeError(
            "没有可用的 FP8 kernel。\n  " + detail +
            "\n可以用 KAVOR_DISABLED_KERNELS 排除有问题的 kernel 后重试。")
    raise RuntimeError(
        f"fp8_backend={backend!r} 不可用:\n  " + detail +
        "\n只有测试/调试才会指定具体 backend,正常推理请用 'auto' 让 Kavor 自动选择。")
