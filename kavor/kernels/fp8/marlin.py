"""Marlin W8A16 kernel 的 Python 入口。

权重/scale 的布局转换在 ``marlin_prepare.py``(加载期一次性完成);这里只负责
forward:二维化、K 方向补零、取持久 workspace、调用 ``torch.ops.kavor_marlin``。
无 host 同步、无按调用分配的大缓冲,可进 CUDA graph。
"""

import os

import torch

from kavor.kernels.fp8 import marlin_build, marlin_prepare
from kavor.utils.logger import logger

# 每卡一份持久锁数组(kernel 每次调用后自己重置锁);不能每次 forward 新分配。
_workspace_cache: dict[int, torch.Tensor] = {}
_warned_atomic = False


def _workspace(device: torch.device) -> torch.Tensor:
    index = device.index
    ws = _workspace_cache.get(index)
    if ws is None:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        ws = torch.zeros(sms * 4, dtype=torch.int32, device=device)
        _workspace_cache[index] = ws
    return ws


def _should_use_atomic_add(m: int, n: int, k: int, dtype: torch.dtype) -> bool:
    # 只有 m*n 小、k 大时 atomicAdd 的全局归约才比 fp32 reduce 划算。
    if n >= 2048 or k < 2048:
        return False
    if os.environ.get("KAVOR_MARLIN_USE_ATOMIC_ADD", "") in ("", "0"):
        return False
    cc = torch.cuda.get_device_capability()
    if cc[0] < 9 and dtype == torch.bfloat16:
        global _warned_atomic
        if not _warned_atomic:
            logger.warning("SM%d0 不支持 bf16 atomicAdd,Marlin 关闭 atomic reduce", cc[0])
            _warned_atomic = True
        return False
    return True


class MarlinKernel:
    """FP8 weight-only GEMM,128×128 block scale,bf16/fp16 激活。"""

    name = "marlin"
    priority = 90

    @classmethod
    def is_available(cls, cfg) -> tuple[bool, str]:
        if not torch.cuda.is_available():
            return False, "需要 CUDA"
        cc = torch.cuda.get_device_capability()
        cc_int = cc[0] * 10 + cc[1]
        if not 80 <= cc_int <= 90:
            return False, f"Marlin 仅支持 SM80/86/89/90,当前 SM{cc_int}"
        ok, reason = marlin_build.marlin_available()
        if not ok:
            return False, reason
        return True, ""

    @classmethod
    def can_implement(cls, cfg) -> tuple[bool, str]:
        if cfg.block_n != 128 or cfg.block_k != 128:
            return False, f"Marlin 需要 128x128 block,当前 {cfg.block_n}x{cfg.block_k}"
        if cfg.input_dtype not in (torch.bfloat16, torch.float16):
            return False, f"Marlin 需要 bf16/fp16 激活,当前 {cfg.input_dtype}"
        if cfg.n <= 0 or cfg.k <= 0:
            return False, f"非法形状 N={cfg.n} K={cfg.k}"
        return True, ""

    def prepare(self, layer: torch.nn.Module) -> None:
        block_n, block_k = layer.weight_block_size
        orig_dtype = layer.orig_dtype
        n, k, padded_n, padded_k = marlin_prepare.prepare_layer(
            layer, (block_n, block_k), orig_dtype
        )
        layer._marlin_meta = (n, k, padded_n, padded_k)
        if (padded_n, padded_k) != (n, k):
            logger.warning(
                "Marlin 对权重做了 tile 补零: N %d->%d, K %d->%d;"
                "每次 forward 会补零激活并切片输出", n, padded_n, k, padded_k)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias) -> torch.Tensor:
        n, k, padded_n, padded_k = layer._marlin_meta
        out_shape = x.shape[:-1] + (n,)

        x2 = x.reshape(-1, k)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        x2 = marlin_prepare.pad_dim(x2, k, padded_k)
        size_m = x2.shape[0]

        use_atomic = _should_use_atomic_add(size_m, padded_n, padded_k, x.dtype)
        out = marlin_build.load_marlin().marlin_gemm(
            x2, None, layer.weight, bias, layer.weight_scale_inv,
            None, None, None, None, None, _workspace(x.device),
            size_m, padded_n, padded_k,
            True, use_atomic, True, False)

        if padded_n != n:
            out = out[:, :n].contiguous()
        return out.reshape(out_shape)
