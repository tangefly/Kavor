"""JIT 编译 vendored 的 Marlin CUDA 扩展(kavor/csrc/marlin)。

首次调用 ``load_marlin()`` 会编译并加载 ``torch.ops.kavor_marlin``;之后进程内
直接复用。产物按「源码 hash + torch/CUDA 版本 + 设备 CC」命名,缓存在 torch
的扩展目录里,源码不变就不会重编。

- ``KAVOR_MARLIN_DISABLE=1``:直接判为不可用(auto 模式会回退到 Triton)。
- ``python -m kavor.kernels.fp8.marlin_build --precompile``:预先编译一次,
  方便在服务启动前把编译耗时挪走。

多 TP worker 会同时 import;编译用 ``fcntl.flock`` 串行化,先到的进程编译,
其余进程等锁后直接用缓存产物。torch.utils.cpp_extension 自身也有 FileBaton,
这里再包一层是为了让「等待」发生在 Kavor 的锁上,方便排查。
"""

import fcntl
import hashlib
import os
import sys
from pathlib import Path

import torch

from kavor.utils.logger import logger

CSRC = Path(__file__).resolve().parents[2] / "csrc" / "marlin"
_EXT_PREFIX = "kavor_marlin"

_ops = None
_failure: Exception | None = None


def is_disabled() -> bool:
    return os.environ.get("KAVOR_MARLIN_DISABLE", "") not in ("", "0")


def _sources() -> list[str]:
    generated = sorted((CSRC / "generated").glob("sm80_kernel_*.cu"))
    if not generated:
        raise RuntimeError(
            f"{CSRC / 'generated'} 下没有 sm80_kernel_*.cu,"
            "请先运行 python csrc/marlin/generate_kernels.py 8.0")
    return [
        str(CSRC / "marlin.cu"),
        str(CSRC / "gptq_marlin_repack.cu"),
        *map(str, generated),
    ]


def _device_cc() -> tuple[int, int]:
    major, minor = torch.cuda.get_device_capability()
    return major, minor


def _arch_flags() -> list[str]:
    """当前设备编 cubin;非 SM80 时再加一份 8.0 PTX 做架构家族内兼容。"""
    major, minor = _device_cc()
    cc = f"{major}{minor}"
    flags = [f"-gencode=arch=compute_{cc},code=sm_{cc}"]
    if cc != "80":
        flags.append("-gencode=arch=compute_80,code=compute_80")
    return flags


def _source_hash() -> str:
    h = hashlib.sha256()
    for path in sorted(CSRC.rglob("*")):
        if path.is_dir() or path.suffix in (".pyc", ".md"):
            continue
        if path.name in ("LICENSE.vllm",):
            continue
        h.update(path.relative_to(CSRC).as_posix().encode())
        h.update(path.read_bytes())
    h.update(torch.__version__.encode())
    h.update((torch.version.cuda or "").encode())
    h.update(f"{_device_cc()}".encode())
    h.update(" ".join(_arch_flags()).encode())
    return h.hexdigest()[:12]


def _extension_name() -> str:
    major, minor = _device_cc()
    return f"{_EXT_PREFIX}_{major}{minor}_{_source_hash()}"


def _build() -> None:
    from torch.utils.cpp_extension import _get_build_directory, load

    name = _extension_name()
    build_dir = Path(_get_build_directory(name, verbose=False))
    build_dir.mkdir(parents=True, exist_ok=True)
    lock_path = build_dir / ".kavor_build.lock"

    cuda_flags = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        # Marlin 的 kernel 实例化在 generated/*.cu 里,主 TU 只引用符号;
        # nvcc 12+ 默认把模板 stub 设为 static,跨 TU 链接会失败。
        "-static-global-template-stub=false",
        *_arch_flags(),
    ]
    verbose = os.environ.get("KAVOR_MARLIN_VERBOSE", "") not in ("", "0")

    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            load(
                name=name,
                sources=_sources(),
                extra_include_paths=[str(CSRC), str(CSRC / "generated")],
                extra_cuda_cflags=cuda_flags,
                extra_cflags=["-O3", "-std=c++17"],
                is_python_module=False,
                verbose=verbose,
            )
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)

    logger.info("Marlin 扩展已就绪: %s", name)


def load_marlin():
    """返回已加载的 ``torch.ops.kavor_marlin``;失败后缓存失败结果。"""
    global _ops, _failure
    if _ops is not None:
        return _ops
    if _failure is not None:
        raise _failure
    if is_disabled():
        _failure = RuntimeError("KAVOR_MARLIN_DISABLE 已设置,Marlin 不可用")
        raise _failure
    if not torch.cuda.is_available():
        _failure = RuntimeError("Marlin 需要 CUDA")
        raise _failure
    try:
        _build()
        _ops = torch.ops.kavor_marlin
    except Exception as exc:  # noqa: BLE001 - 构建失败要转成 kernel 不可用
        _failure = exc
        raise
    return _ops


def marlin_available() -> tuple[bool, str]:
    """供 registry 查询;构建失败只记账、不抛。"""
    try:
        load_marlin()
    except Exception as exc:  # noqa: BLE001
        return False, f"Marlin 扩展不可用: {exc}"
    return True, ""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--precompile" not in argv:
        print("用法: python -m kavor.kernels.fp8.marlin_build --precompile")
        return 2
    try:
        load_marlin()
    except Exception as exc:  # noqa: BLE001
        print(f"预编译失败: {exc}", file=sys.stderr)
        return 1
    print("Marlin 预编译完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
