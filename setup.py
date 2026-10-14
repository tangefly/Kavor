import json
import os
import platform
import re
import subprocess
import sys

from setuptools import setup

FLASH_ATTN_VERSION = "2.8.4"
FLASH_ATTN_PYTHONS = ("cp312",)
WHEEL_BASE_URL = os.environ.get(
    "KAVOR_WHEEL_BASE",
    "https://github.com/tangefly/Kavor/releases/download/wheels-v1",
)
TORCH_VARIANTS = {
    "cu130": {
        "torch": "2.9.1",
        "cuda": "13.0",
        "abi": True,
        "index": "https://download.pytorch.org/whl/cu130",
        "wheel": "torch-2.9.1%2Bcu130-{py}-{py}-manylinux_2_28_x86_64.whl",
        "driver_cuda_min": 13,
    },
}
DEFAULT_VARIANT = "cu130"
TORCH_SPEC = "torch>=2.9,<2.10"
BASE_REQUIREMENTS = [
    "triton>=3.0.0",
    "transformers>=4.51.0",
    "xxhash",
    "fastapi",
    "uvicorn"
]
_PROBE = (
    "import json, torch; print(json.dumps({"
    "'version': torch.__version__,"
    "'cuda': torch.version.cuda,"
    "'abi': bool(torch._C._GLIBCXX_USE_CXX11_ABI)}))"
)

_probe_cache = None


def probe_target_torch():
    """探测目标环境(真正要跑 kavor 的那个环境)里的 torch,返回 dict 或 None。

    pip 默认在隔离的构建环境里跑本文件,那里没有 torch;
    但 CONDA_PREFIX / VIRTUAL_ENV 会原样传进来,所以借目标环境的解释器
    跑一次探测最准,也不用往构建依赖里塞一整个 torch。
    """
    global _probe_cache
    if _probe_cache is not None:
        return _probe_cache

    # 没有构建隔离(pip install -e . --no-build-isolation)时可以直接 import
    try:
        import torch
    except ImportError:
        pass
    else:
        _probe_cache = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        }
        return _probe_cache

    for var in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        prefix = os.environ.get(var)
        if not prefix:
            continue
        python = os.path.join(prefix, "bin", "python")
        if not os.path.exists(python):
            continue
        try:
            done = subprocess.run(
                [python, "-c", _PROBE], capture_output=True, text=True, timeout=300
            )
        except Exception:
            continue
        if done.returncode != 0 or not done.stdout.strip():
            continue
        try:
            _probe_cache = json.loads(done.stdout.strip().splitlines()[-1])
        except ValueError:
            continue
        return _probe_cache

    return None


def detect_driver_cuda():
    """nvidia-smi 报告的驱动支持的 CUDA 上限,读不到返回 None。"""
    try:
        done = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
    except Exception:
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", done.stdout or "")
    return int(m.group(1)) if m else None


def torch_requirement(info):
    """决定 torch 这条依赖怎么写,以及按哪个 torch 选 flash-attn wheel。

    返回 (依赖字符串, 供 flash_attn_requirement 用的 torch 信息)。
    """
    if info is not None:
        return TORCH_SPEC, info

    variant = os.environ.get("KAVOR_TORCH_CUDA", "auto").lower()

    if variant == "auto":
        v = TORCH_VARIANTS[DEFAULT_VARIANT]
        driver = detect_driver_cuda()
        if driver is not None and driver < v["driver_cuda_min"]:
            print(
                f"[kavor] 驱动只支持到 CUDA {driver}.x,装 {DEFAULT_VARIANT} 的 torch 跑不起来。\n"
                "[kavor] 请先自己装好 torch(比如 pip install torch --index-url "
                "https://download.pytorch.org/whl/cu126),再 pip install -e ."
            )
            return TORCH_SPEC, None
        variant = DEFAULT_VARIANT

    if variant not in TORCH_VARIANTS:
        print(f"[kavor] KAVOR_TORCH_CUDA={variant},不替 pip 指定 torch")
        return TORCH_SPEC, None

    if sys.platform != "linux" or platform.machine() != "x86_64":
        print("[kavor] 非 linux/x86_64,不替 pip 指定 torch")
        return TORCH_SPEC, None

    v = TORCH_VARIANTS[variant]
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    url = f"{v['index']}/{v['wheel'].format(py=py)}"
    print(f"[kavor] 目标环境没有 torch,将装 {variant} 官方 wheel:\n        {url}")
    return (
        f"torch @ {url}",
        {"version": v["torch"], "cuda": v["cuda"], "abi": v["abi"]},
    )


def flash_attn_requirement(info):
    """按探测结果拼出预编译 wheel 的直链,没有对应 wheel 时返回 None。"""
    if not info or not info.get("cuda"):
        return None

    py = f"cp{sys.version_info.major}{sys.version_info.minor}"  # cp312
    if py not in FLASH_ATTN_PYTHONS:
        return None
    cu = "cu" + info["cuda"].split(".")[0]  # 13.0 -> cu13
    tv = "torch" + ".".join(info["version"].split("+")[0].split(".")[:2])  # 2.9.1 -> torch2.9
    abi = "TRUE" if info["abi"] else "FALSE"
    plat = f"linux_{platform.machine()}"  # linux_x86_64
    wheel = f"flash_attn-{FLASH_ATTN_VERSION}+{cu}{tv}cxx11abi{abi}-{py}-{py}-{plat}.whl"
    return f"flash-attn @ {WHEEL_BASE_URL}/{wheel}"


def get_requirements():
    info = probe_target_torch()
    if info is not None:
        print(
            f"[kavor] 目标环境已有 torch {info['version']} / CUDA {info['cuda']} "
            f"/ cxx11abi={info['abi']}"
        )

    torch_req, fa_info = torch_requirement(info)
    req = [torch_req] + list(BASE_REQUIREMENTS)

    mode = os.environ.get("KAVOR_FLASH_ATTN", "auto").lower()
    if mode == "skip":
        print("[kavor] KAVOR_FLASH_ATTN=skip,不装 flash-attn")
        return req

    wheel = flash_attn_requirement(fa_info)
    if mode == "build":
        print("[kavor] KAVOR_FLASH_ATTN=build,走源码编译")
        req.append("flash-attn")
    elif wheel is not None:
        print(f"[kavor] flash-attn 使用预编译 wheel: {wheel.split(' @ ')[1]}")
        req.append(wheel)
    else:
        print("[kavor] 没匹配到预编译 wheel,本次不装 flash-attn。")
        print("        自己编一个传上去,或设 KAVOR_FLASH_ATTN=build 走源码编译。")
    return req


setup(install_requires=get_requirements())
