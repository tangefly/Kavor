import json
import os
import platform
import re
import subprocess
import sys

from setuptools import setup

FLASH_ATTN_VERSION = "2.8.3.post2"
FLASH_ATTN_PYTHONS = ("cp312",)
WHEEL_BASE_URL = os.environ.get(
    "KAVOR_WHEEL_BASE",
    "https://github.com/tangefly/Kavor/releases/download/wheels-v1",
)
TORCH_VARIANTS = {
    "cu126": {
        "torch": "2.9.1",
        "cuda": "12.6",
        "abi": True,
        "index": "https://download.pytorch.org/whl/cu126",
        "wheel": "torch-2.9.1%2Bcu126-{py}-{py}-manylinux_2_28_x86_64.whl",
        "driver_cuda_min": 12,
        "flash_attn_version": "2.8.3.post2",
        "flash_attn_local": "cu126torch2.9cxx11abitrue",
    },
    "cu130": {
        "torch": "2.9.1",
        "cuda": "13.0",
        "abi": True,
        "index": "https://download.pytorch.org/whl/cu130",
        "wheel": "torch-2.9.1%2Bcu130-{py}-{py}-manylinux_2_28_x86_64.whl",
        "driver_cuda_min": 13,
        "flash_attn_version": "2.8.3.post2",
        "flash_attn_local": "cu13torch2.9cxx11abitrue",
    },
}
DEFAULT_VARIANT = "cu130"
TORCH_SPEC = "torch>=2.9,<2.10"
BASE_REQUIREMENTS = [
    "triton>=3.0.0",
    "transformers>=4.51.0",
    "xxhash",
    "fastapi",
    "uvicorn",
    "flash-linear-attention[cuda]"
]
_PROBE = (
    "import json, torch; print(json.dumps({"
    "'version': torch.__version__,"
    "'cuda': torch.version.cuda,"
    "'abi': bool(torch._C._GLIBCXX_USE_CXX11_ABI)}))"
)
_FLASH_ATTN_PROBE = (
    "import json\n"
    "try:\n"
    "    import flash_attn\n"
    "except Exception as exc:\n"
    "    print(json.dumps({'error': type(exc).__name__}))\n"
    "else:\n"
    "    print(json.dumps({'version': flash_attn.__version__}))\n"
)

_probe_cache = None
_probe_python = None
_flash_attn_probe_cache = None


def _run_probe(python, code):
    """借目标环境的解释器跑一段探测脚本,返回最后一行 JSON;失败返回 None。"""
    try:
        # pip 的构建隔离会把 PYTHONPATH 指向它自己的 site 目录,
        # 那里的 sitecustomize.py 会把目标环境的 site-packages 从
        # sys.path 里删掉。这里必须清掉,否则连目标环境的 torch 也 import 不到。
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("PYTHONPATH", "PYTHONHOME")
        }
        done = subprocess.run(
            [python, "-c", code],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
    except Exception:
        return None
    if done.returncode != 0 or not done.stdout.strip():
        return None
    try:
        return json.loads(done.stdout.strip().splitlines()[-1])
    except ValueError:
        return None


def probe_target_torch():
    """探测目标环境(真正要跑 kavor 的那个环境)里的 torch,返回 dict 或 None。

    pip 默认在隔离的构建环境里跑本文件,那里没有 torch;
    但 CONDA_PREFIX / VIRTUAL_ENV 会原样传进来,所以借目标环境的解释器
    跑一次探测最准,也不用往构建依赖里塞一整个 torch。
    """
    global _probe_cache, _probe_python
    if _probe_cache is not None:
        return _probe_cache

    # 没有构建隔离(pip install -e . --no-build-isolation)时可以直接 import
    try:
        import torch
    except ImportError:
        pass
    else:
        _probe_python = sys.executable
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
        result = _run_probe(python, _PROBE)
        if result is None:
            continue
        _probe_python = python
        _probe_cache = result
        return _probe_cache

    return None


def probe_local_flash_attn():
    """目标环境里已装且能 import 的 flash-attn 版本,没有返回 None。

    只认能 import 的:装了个 ABI 不匹配的 .so 时 import 会失败,
    这种情况要继续走预编译 wheel 重装。
    """
    global _flash_attn_probe_cache
    if _flash_attn_probe_cache is not None:
        return _flash_attn_probe_cache

    if _probe_python is None:
        return None

    result = _run_probe(_probe_python, _FLASH_ATTN_PROBE)
    if not result or result.get("error") or not result.get("version"):
        return None
    _flash_attn_probe_cache = result["version"]
    return _flash_attn_probe_cache


def detect_nvcc_cuda():
    """nvcc -V 报告的本机 CUDA 版本(实际在用的那个),读不到返回 "13.0" 这样的字符串。"""
    try:
        done = subprocess.run(
            ["nvcc", "-V"], capture_output=True, text=True, timeout=10
        )
    except Exception:
        return None
    out = (done.stdout or "") + (done.stderr or "")
    # Cuda compilation tools, release 13.0, V13.0.88
    m = re.search(r"release\s+(\d+)\.(\d+)", out)
    return f"{m.group(1)}.{m.group(2)}" if m else None


def detect_driver_cuda():
    """nvidia-smi 报告的驱动支持的 CUDA 上限(不是实际在用的版本),读不到返回 None。"""
    try:
        done = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
    except Exception:
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", done.stdout or "")
    return int(m.group(1)) if m else None


def auto_variant():
    """auto 模式下按 nvcc 选变体,驱动版本只作为上限校验。

    nvcc 的版本才是本机实际在用的 CUDA;驱动里的 "CUDA Version" 只是它
    能支持到的最高版本。没装 nvcc 时退回默认变体。
    """
    toolkit = detect_nvcc_cuda()
    if toolkit is None:
        variant = DEFAULT_VARIANT
    else:
        major = int(toolkit.split(".")[0])
        usable = [
            key
            for key, v in TORCH_VARIANTS.items()
            if int(v["cuda"].split(".")[0]) <= major
        ]
        if not usable:
            print(
                f"[kavor] nvcc 显示本机实际是 CUDA {toolkit},没有匹配的 torch 变体。\n"
                "[kavor] 请先自己装好对应版本的 torch(比如 pip install torch "
                "--index-url https://download.pytorch.org/whl/cu126),再 pip install -e ."
            )
            return None
        variant = max(
            usable,
            key=lambda k: tuple(int(x) for x in TORCH_VARIANTS[k]["cuda"].split(".")),
        )

    v = TORCH_VARIANTS[variant]
    driver = detect_driver_cuda()
    if driver is not None and driver < v["driver_cuda_min"]:
        print(
            f"[kavor] 驱动只支持到 CUDA {driver}.x,装 {variant} 的 torch 跑不起来。\n"
            "[kavor] 请先自己装好 torch(比如 pip install torch --index-url "
            "https://download.pytorch.org/whl/cu126),再 pip install -e ."
        )
        return None
    return variant


def resolve_variant():
    """按 nvcc 得到目标变体;KAVOR_TORCH_CUDA 可以显式覆盖。"""
    variant = os.environ.get("KAVOR_TORCH_CUDA", "auto").lower()

    if variant == "auto":
        return auto_variant()

    if variant not in TORCH_VARIANTS:
        print(f"[kavor] KAVOR_TORCH_CUDA={variant},不替 pip 指定 torch")
        return None

    return variant


def variant_info(v):
    """变体表里的 torch 信息,格式和 probe_target_torch 的返回值一致。"""
    return {"version": v["torch"], "cuda": v["cuda"], "abi": v["abi"]}


def same_cuda(info, v):
    """已装 torch 的 CUDA 和变体是否一致(只比较 major.minor)。"""
    installed = (info.get("cuda") or "").split(".")
    wanted = v["cuda"].split(".")
    return installed[:2] == wanted[:2]


def torch_wheel_url(variant):
    v = TORCH_VARIANTS[variant]
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return f"{v['index']}/{v['wheel'].format(py=py)}"


def torch_requirement(info):
    """决定 torch 这条依赖怎么写,以及按哪个 torch 选 flash-attn wheel。

    nvcc 是唯一判据:它选出的变体和已装 torch 不一致时,按 nvcc 重装 torch。
    返回 (依赖字符串, flash-attn 用的 torch 信息, 变体 key)。
    """
    variant = resolve_variant()
    v = TORCH_VARIANTS[variant] if variant is not None else None

    if v is not None and info is not None and same_cuda(info, v):
        print(
            f"[kavor] 目标环境 torch {info['version']} / CUDA {info['cuda']} "
            f"与 nvcc 选出的 {variant} 一致"
        )
        return TORCH_SPEC, info, variant

    if v is not None:
        if sys.platform != "linux" or platform.machine() != "x86_64":
            print("[kavor] 非 linux/x86_64,不替 pip 指定 torch")
            return TORCH_SPEC, info, variant

        if info is not None:
            print(
                f"[kavor] 目标环境 torch 是 CUDA {info['cuda']},"
                f"与 nvcc 选出的 {variant} 不一致,按 nvcc 重装 torch"
            )
        else:
            print(f"[kavor] 目标环境没有 torch,将装 {variant} 官方 wheel:")
        print(f"        {torch_wheel_url(variant)}")
        return f"torch @ {torch_wheel_url(variant)}", variant_info(v), variant

    # nvcc 没有给出变体:沿用已装 torch,没有就不指定具体版本
    if info is not None:
        # flash-attn 的版本是按变体定的,这里按已装 torch 的 CUDA 反查变体
        for key, cv in TORCH_VARIANTS.items():
            if same_cuda(info, cv):
                variant = key
                break
    return TORCH_SPEC, info, variant


def flash_attn_requirement(info, variant=None):
    """按探测结果拼出预编译 wheel 的直链,没有对应 wheel 时返回 None。"""
    if not info or not info.get("cuda"):
        return None

    version = FLASH_ATTN_VERSION
    local = None
    if variant is not None and variant in TORCH_VARIANTS:
        v = TORCH_VARIANTS[variant]
        version = v.get("flash_attn_version", version)
        local = v.get("flash_attn_local")

    py = f"cp{sys.version_info.major}{sys.version_info.minor}"  # cp312
    if py not in FLASH_ATTN_PYTHONS:
        return None
    if local is None:
        cu = "cu" + info["cuda"].split(".")[0]  # 13.0 -> cu13
        tv = "torch" + ".".join(info["version"].split("+")[0].split(".")[:2])  # 2.9.1 -> torch2.9
        abi = "TRUE" if info["abi"] else "FALSE"
        local = f"{cu}{tv}cxx11abi{abi}"
    plat = f"linux_{platform.machine()}"  # linux_x86_64
    wheel = f"flash_attn-{version}+{local}-{py}-{py}-{plat}.whl"
    return f"flash-attn @ {WHEEL_BASE_URL}/{wheel}"


def get_requirements():
    info = probe_target_torch()
    if info is not None:
        print(
            f"[kavor] 目标环境已有 torch {info['version']} / CUDA {info['cuda']} "
            f"/ cxx11abi={info['abi']}"
        )

    torch_req, fa_info, variant = torch_requirement(info)
    req = [torch_req] + list(BASE_REQUIREMENTS)

    mode = os.environ.get("KAVOR_FLASH_ATTN", "auto").lower()
    if mode == "skip":
        print("[kavor] KAVOR_FLASH_ATTN=skip,不装 flash-attn")
        return req

    wheel = flash_attn_requirement(fa_info, variant)
    if mode == "build":
        print("[kavor] KAVOR_FLASH_ATTN=build,走源码编译")
        req.append("flash-attn")
    elif mode == "wheel":
        if wheel is not None:
            print(
                "[kavor] KAVOR_FLASH_ATTN=wheel,"
                f"使用预编译 wheel: {wheel.split(' @ ')[1]}"
            )
            req.append(wheel)
        else:
            print("[kavor] KAVOR_FLASH_ATTN=wheel,但没有匹配到预编译 wheel。")
    elif (local := probe_local_flash_attn()) is not None:
        print(f"[kavor] 目标环境已有 flash-attn {local},跳过预编译 wheel 下载")
        if local.split("+")[0] != FLASH_ATTN_VERSION:
            print(
                f"        预编译 wheel 的版本是 {FLASH_ATTN_VERSION},"
                "如需换成它请设 KAVOR_FLASH_ATTN=wheel"
            )
        req.append("flash-attn")
    elif wheel is not None:
        print(f"[kavor] flash-attn 使用预编译 wheel: {wheel.split(' @ ')[1]}")
        req.append(wheel)
    else:
        print("[kavor] 没匹配到预编译 wheel,本次不装 flash-attn。")
        print("        自己编一个传上去,或设 KAVOR_FLASH_ATTN=build 走源码编译。")
    return req


setup(
    install_requires=get_requirements(),
    # Marlin 的 CUDA 源码随包分发,首次使用时 JIT 编译(见
    # kavor/kernels/fp8/marlin_build.py)。kavor/csrc 不是 Python 包,
    # 所以要在这里显式声明为 kavor 包的 data files。
    package_data={
        "kavor": [
            "csrc/quantization/marlin/*.cu",
            "csrc/quantization/marlin/*.cuh",
            "csrc/quantization/marlin/*.h",
            "csrc/quantization/marlin/*.py",
            "csrc/quantization/marlin/*.md",
            "csrc/quantization/marlin/LICENSE.vllm",
            "csrc/quantization/marlin/NOTICE",
        ],
    },
)
