"""FP8 kernel registry 的选择逻辑(CPU 可跑)。

真实的硬件探测被替换成假 kernel;另外单独验证各 kernel 的
``can_implement`` / ``is_available`` 对 SM 与数据类型的判断。
"""

from types import SimpleNamespace

import pytest
import torch

from kavor.kernels.fp8 import registry
from kavor.kernels.fp8.registry import (
    Fp8Kernel,
    Fp8LinearConfig,
    MarlinKernel,
    TritonW8A8Kernel,
    TritonW8A16Kernel,
    select_kernel,
)


def _cfg(n=256, k=512, block=(128, 128), dtype=torch.bfloat16):
    return Fp8LinearConfig(n, k, block[0], block[1], dtype, True)


def _fake(name, priority, can=True, avail=True):
    def is_available(cls, cfg):
        return avail, "" if avail else f"{name} unavailable"

    def can_implement(cls, cfg):
        return can, "" if can else f"{name} cannot implement"

    Fake = type(name, (Fp8Kernel,), {
        "name": name,
        "priority": priority,
        "is_available": classmethod(is_available),
        "can_implement": classmethod(can_implement),
        "prepare": lambda self, layer: None,
        "apply": lambda self, layer, x, bias: x,
    })
    return Fake


def _install(monkeypatch, *kernels):
    ordered = tuple(kernels)
    monkeypatch.setattr(registry, "_ORDERED", ordered)
    monkeypatch.setattr(registry, "_BY_NAME", {k.name: k for k in ordered})
    # 让假 kernel 的名字也能通过 backend 合法性校验
    monkeypatch.setattr(
        registry, "BACKENDS", ("auto", *(k.name for k in ordered)))


def test_auto_picks_highest_priority(monkeypatch):
    _install(monkeypatch,
             _fake("low", 10), _fake("high", 100), _fake("mid", 50))
    assert select_kernel(_cfg(), "auto").name == "high"


def test_auto_skips_unsupported_then_picks_next(monkeypatch):
    _install(monkeypatch,
             _fake("high", 100, can=False),
             _fake("mid", 50),
             _fake("low", 10))
    assert select_kernel(_cfg(), "auto").name == "mid"


def test_auto_skips_unavailable_then_picks_next(monkeypatch):
    _install(monkeypatch,
             _fake("high", 100, avail=False),
             _fake("mid", 50))
    assert select_kernel(_cfg(), "auto").name == "mid"


def test_forced_backend_only_tries_that_kernel(monkeypatch):
    _install(monkeypatch, _fake("high", 100), _fake("low", 10))
    assert select_kernel(_cfg(), "low").name == "low"

    # 指定 kernel 不满足条件时直接报错,不会悄悄回退到别的 kernel
    _install(monkeypatch, _fake("high", 100), _fake("low", 10, can=False))
    with pytest.raises(RuntimeError, match="low"):
        select_kernel(_cfg(), "low")


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="未知 fp8_backend"):
        select_kernel(_cfg(), "bogus")


def test_disabled_env_applies_to_auto_and_forced(monkeypatch):
    _install(monkeypatch, _fake("high", 100), _fake("low", 10))
    monkeypatch.setenv("KAVOR_DISABLED_KERNELS", "high")
    assert select_kernel(_cfg(), "auto").name == "low"
    with pytest.raises(RuntimeError, match="high"):
        select_kernel(_cfg(), "high")


def test_auto_reports_all_reasons_when_nothing_works(monkeypatch):
    _install(monkeypatch,
             _fake("a", 100, can=False),
             _fake("b", 50, avail=False))
    with pytest.raises(RuntimeError) as exc:
        select_kernel(_cfg(), "auto")
    msg = str(exc.value)
    assert "a:" in msg and "b:" in msg


def test_sm80_disables_triton_w8a8(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 0))
    ok, reason = TritonW8A8Kernel.is_available(_cfg())
    assert not ok and "SM89" in reason

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 9))
    ok, _ = TritonW8A8Kernel.is_available(_cfg())
    assert ok


def test_triton_w8a16_needs_bf16_and_128_block():
    assert TritonW8A16Kernel.can_implement(_cfg())[0]
    assert not TritonW8A16Kernel.can_implement(_cfg(dtype=torch.float16))[0]
    assert not TritonW8A16Kernel.can_implement(_cfg(block=(64, 128)))[0]


def test_marlin_can_implement_scope():
    assert MarlinKernel.can_implement(_cfg())[0]
    assert MarlinKernel.can_implement(_cfg(dtype=torch.float16))[0]
    assert not MarlinKernel.can_implement(_cfg(dtype=torch.float32))[0]
    assert not MarlinKernel.can_implement(_cfg(block=(64, 128)))[0]


def test_marlin_is_available_rejects_sm100(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 0))
    ok, reason = MarlinKernel.is_available(_cfg())
    assert not ok and "SM100" in reason


def test_auto_falls_back_to_triton_when_marlin_unavailable(monkeypatch):
    """Marlin 构建失败/不可用时,auto 继续尝试低优先级 kernel。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 9))
    monkeypatch.setattr(
        MarlinKernel, "is_available",
        classmethod(lambda cls, cfg: (False, "marlin build failed")))
    monkeypatch.setattr(
        registry.FlashInferKernel, "is_available",
        classmethod(lambda cls, cfg: (False, "no flashinfer")))
    assert select_kernel(_cfg(), "auto").name == "triton_w8a16"
