import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from setuptools import setup

TORCH_VERSION = "2.14.1"
TORCH_MINOR = ".".join(TORCH_VERSION.split(".")[:2])
TORCHVISION_VERSION = "0.29.1"
PYTORCH_INDEX = "https://download.pytorch.org/whl"

FLASH_ATTN_VERSION = "2.8.3.post2"
FLASH_ATTN_REPO = "https://github.com/Dao-AILab/flash-attention"
KAVOR_RELEASES = "https://api.github.com/repos/tangefly/Kavor/releases"
REPO_ROOT = Path(__file__).resolve().parent
GIT_TIMEOUT = 60  # a clone still running after 1 minute is a dead connection

# Set KAVOR_INSTALL_FLASH_ATTN=0 to leave flash-attn out of the install.
INSTALL_FLASH_ATTN = os.environ.get("KAVOR_INSTALL_FLASH_ATTN", "1").strip().lower() not in (
    "0", "false", "no", "off"
)

# torch CUDA build -> CUDA major, which is also the lowest one the driver must support.
CUDA_VARIANTS = {"cu126": 12, "cu130": 13}
DEFAULT_VARIANT = "cu126"  # runs on the oldest drivers

BASE_REQUIREMENTS = [
    "triton>=3.3",
    "transformers==5.17.0",
    "numpy>=1.26,<3",
    "pillow>=10.0.1",
    "safetensors>=0.8.0",
    "tqdm>=4.66",
    "pydantic>=2.6",
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "einops>=0.8",
    "flash-linear-attention[cuda]==0.6.0",
]

ARCH = platform.machine()
CP = f"cp{sys.version_info.major}{sys.version_info.minor}"
WHEEL_SUFFIX = f"-{CP}-{CP}-manylinux_2_28_{ARCH}.whl"
PREBUILT_TORCH = sys.platform.startswith("linux") and ARCH in ("x86_64", "aarch64")

# One JSON line per step, so a later crash (e.g. a broken flash-attn .so) keeps the earlier results.
TARGET_PROBE = """
import json
try:
    import torch
except BaseException:
    pass
else:
    print(json.dumps({"torch": {"version": torch.__version__, "cuda": torch.version.cuda,
                               "abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI)}}), flush=True)
try:
    import flash_attn
except BaseException:
    pass
else:
    print(json.dumps({"flash_attn": flash_attn.__version__}), flush=True)
"""
GPU_PROBE = (
    "import json, torch; "
    "print(json.dumps({'capability': '%d.%d' % torch.cuda.get_device_capability()}))"
)


def log(message):
    """Print straight to the terminal, because pip hides most of the build output."""
    line = f"[kavor] {message}"
    try:
        with open("/dev/tty", "w") as tty:
            print(line, file=tty, flush=True)
    except OSError:
        print(line, file=sys.stderr, flush=True)


def target_env(extra=None):
    """Environment for subprocesses that must see the target environment's packages.

    Build isolation points PYTHONPATH at pip's own overlay, which hides them.
    """
    blocked = ("PYTHONPATH", "PYTHONHOME", "PYTHONNOUSERSITE", "PYTHONSTARTUP")
    env = {key: value for key, value in os.environ.items() if key not in blocked}
    env.update(extra or {})
    return env


def run(cmd, env=None, cwd=None, timeout=None):
    """Run a command quietly and return its stdout, or '' when it cannot run."""
    try:
        done = subprocess.run(
            cmd, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        return done.stdout
    except Exception:
        return ""


def show(cmd, env=None, cwd=None, timeout=None):
    """Run a command with its output sent to the terminal; returns the exit code."""
    try:
        tty = open("/dev/tty", "w")
    except OSError:  # no terminal to write to
        tty = None
    redirections = {"stdout": tty, "stderr": subprocess.STDOUT} if tty else {}
    try:
        done = subprocess.run(cmd, env=env, cwd=cwd, timeout=timeout, **redirections)
        return done.returncode
    except subprocess.TimeoutExpired:
        log(f"timed out after {timeout}s: {shlex.join(cmd)}")
        return -1
    finally:
        if tty:
            tty.close()


def git_fetch(args, env, what):
    """Run a `git` network command once and return; raises when it fails."""
    if show(["git", *args], env=env, timeout=GIT_TIMEOUT) != 0:
        raise RuntimeError(f"{what} failed")


def pip_install(*args):
    """Install into the target environment; returns True on success."""
    return show([sys.executable, "-m", "pip", "install", *args], env=target_env()) == 0


def probe_target(code=TARGET_PROBE):
    """Run `code` with the target interpreter and merge the JSON lines it prints."""
    found = {}
    output = run([sys.executable, "-c", code], env=target_env(), timeout=120)
    for line in output.splitlines():
        try:
            found.update(json.loads(line))
        except ValueError:
            pass
    return found


def nvcc_major():
    """CUDA major version reported by nvcc, or None."""
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    candidates = [os.path.join(home, "bin", "nvcc")] if home else []
    if shutil.which("nvcc"):
        candidates.append(shutil.which("nvcc"))
    for nvcc in candidates:
        match = re.search(r"release\s+(\d+)\.(\d+)", run([nvcc, "-V"], timeout=15))
        if match:
            return int(match.group(1))
    return None


def driver_major():
    """Highest CUDA major the NVIDIA driver supports, or None."""
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", run(["nvidia-smi"], timeout=15))
    return int(match.group(1)) if match else None


def resolve_variant(nvcc, driver, forced):
    """Pick the torch CUDA build; returns (variant or None, notes to log).

    KAVOR_TORCH_CUDA wins over nvcc, and the driver is a hard upper bound.
    """
    forced = forced.strip().lower()
    if forced == "none":
        return None, ["KAVOR_TORCH_CUDA=none: using the PyPI build of torch"]
    if forced in CUDA_VARIANTS:
        return forced, [f"KAVOR_TORCH_CUDA={forced}"]

    notes = []
    if forced != "auto":
        notes.append(f"KAVOR_TORCH_CUDA={forced} is not a known variant, selecting automatically")
    if nvcc is None and driver is None:
        return DEFAULT_VARIANT, notes + [
            f"neither nvcc nor a driver was detected, using {DEFAULT_VARIANT}"
        ]

    allowed = [key for key, cuda in CUDA_VARIANTS.items() if driver is None or cuda <= driver]
    if not allowed:
        return None, notes + [f"the driver only supports CUDA {driver}.x, below every torch build"]

    highest = max(allowed, key=lambda key: CUDA_VARIANTS[key])
    chosen = next((key for key in allowed if CUDA_VARIANTS[key] == nvcc), highest)
    if nvcc is None:
        notes.append(f"nvcc not found, selecting {chosen} by the driver limit CUDA {driver}.x")
    elif CUDA_VARIANTS[chosen] != nvcc:
        notes.append(
            f"nvcc is CUDA {nvcc}.x but {chosen} was selected; JIT kernels such as marlin "
            "may fail to build and fall back to triton"
        )
    return chosen, notes


def cuda_major(info):
    """CUDA major of a probed torch, or None."""
    return int(str((info or {}).get("cuda") or 0).split(".")[0]) or None


def flash_attn_matches(version, cu_major):
    """Whether an installed flash-attn was built for the pinned torch line and CUDA major.

    A version without local tags cannot be told apart, so it counts as a match.
    """
    torch = re.search(r"torch(\d+\.\d+)", version)
    cuda = re.search(r"cu(\d+)", version)
    if torch and torch.group(1) != TORCH_MINOR:
        return False
    return cu_major is None or not cuda or int(cuda.group(1)[:2]) == cu_major


def wheel_cache_dir():
    """Where built wheels and this run's requirements memo live.

    Defaults to the checkout's own .cache (git-ignored), so a wheel stays next to the
    source it was built from instead of in a global cache the user cannot find.
    """
    if os.environ.get("KAVOR_WHEEL_CACHE"):
        return Path(os.environ["KAVOR_WHEEL_CACHE"]).expanduser()
    return REPO_ROOT / ".cache" / "wheels"


def source_dir():
    """The flash-attention checkout the build compiles from; reused across runs."""
    return wheel_cache_dir() / "src" / f"flash-attention-v{FLASH_ATTN_VERSION}"


def wheel_pattern(cu_major, abi):
    """Matches a flash-attn wheel for this interpreter, the pinned torch line and the CUDA major,
    whatever spelling the local tag uses (cu12 / cu126, cxx11abitrue / cxx11abiTRUE)."""
    return re.compile(
        rf"flash_attn-{re.escape(FLASH_ATTN_VERSION)}\+cu{cu_major}\d*torch{TORCH_MINOR}"
        rf"cxx11abi{'true' if abi else 'false'}-{CP}-{CP}-linux_{ARCH}\.whl$",
        re.IGNORECASE,
    )


def release_wheel(cu_major, abi):
    """URL of a usable flash-attn wheel attached to the Kavor releases, or None."""
    try:
        with urllib.request.urlopen(KAVOR_RELEASES, timeout=30) as response:
            releases = json.load(response)
        urls = [
            asset["browser_download_url"]
            for release in releases
            for asset in release["assets"]
        ]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log(f"could not list the wheels on the Kavor release ({exc})")
        return None
    return next((url for url in urls if wheel_pattern(cu_major, abi).search(url)), None)


def cached_wheel(cu_major, abi):
    """URL of a flash-attn wheel built earlier on this machine, or None."""
    pattern = wheel_pattern(cu_major, abi)
    wheels = wheel_cache_dir().glob("*.whl")
    return next((path.as_uri() for path in wheels if pattern.search(path.name)), None)


def max_jobs():
    """Parallel compile jobs: each nvcc process peaks at roughly 6-9 GB of memory."""
    if os.environ.get("MAX_JOBS"):
        return max(1, int(os.environ["MAX_JOBS"]))
    available_gb = 32.0
    try:
        meminfo = Path("/proc/meminfo").read_text()
        available_gb = int(re.search(r"MemAvailable:\s+(\d+)", meminfo).group(1)) / 1024**2
    except (OSError, AttributeError):
        pass
    return max(1, min(os.cpu_count() or 4, int(available_gb // 9) or 1, 8))


# flash-attn reads its own arch knob and ignores TORCH_CUDA_ARCH_LIST; it only knows
# sm_80/90/100/120, and an sm_80 build runs on every 8.x card.
CUDA_ARCHS = {"8": "80", "9": "90", "10": "100", "12": "120"}


def flash_attn_archs(capability):
    """The flash-attn arch name for this GPU, or None to keep the upstream default."""
    return CUDA_ARCHS.get((capability or "").split(".")[0])


def patch_cxx_std(src):
    """flash-attn 2.8.3.post2 hardcodes -std=c++17, which the pinned torch headers reject."""
    setup_py = src / "setup.py"
    text = setup_py.read_text()
    if '"-std=c++17"' not in text:
        return
    setup_py.write_text(text.replace('"-std=c++17"', '"-std=c++20"'))
    log("building flash-attn with -std=c++20 (torch 2.14 requires C++20)")


def compile_flash_attn(torch_req):
    """Compile flash-attn from its repo and return the URL of the wheel, kept in the wheel cache."""
    # pip installs torch only after this metadata step, but the build needs it right now.
    if not pip_install(torch_req):
        raise RuntimeError(f"could not install {torch_req}")
    info = probe_target()["torch"]
    cu_major, abi = cuda_major(info), bool(info.get("abi", True))

    nvcc = nvcc_major()
    if nvcc is None:
        raise RuntimeError("nvcc not found, install the CUDA toolkit first")
    if nvcc != cu_major:
        log(
            f"warning: nvcc is CUDA {nvcc}.x but torch uses CUDA {info['cuda']}; "
            "the build may fail"
        )

    capability = probe_target(GPU_PROBE).get("capability")
    env = target_env(
        {
            "MAX_JOBS": str(max_jobs()),
            "NVCC_THREADS": os.environ.get("NVCC_THREADS", "1"),
            # Never swap in a wheel from the official flash-attn release.
            "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
            "FLASH_ATTN_LOCAL_VERSION": (
                f"cu{cu_major}torch{TORCH_MINOR}cxx11abi{'TRUE' if abi else 'FALSE'}"
            ),
        }
    )
    archs = flash_attn_archs(capability)
    if archs:
        env["FLASH_ATTN_CUDA_ARCHS"] = archs

    dest = wheel_cache_dir()
    dest.mkdir(parents=True, exist_ok=True)
    log(
        f"building flash-attn {FLASH_ATTN_VERSION} ({CP}, CUDA {cu_major}); "
        "the first build usually takes 10-30 minutes"
    )
    log(
        f"MAX_JOBS={env['MAX_JOBS']} NVCC_THREADS={env['NVCC_THREADS']} "
        f"FLASH_ATTN_CUDA_ARCHS={archs or 'upstream default'}"
    )

    if not pip_install("setuptools", "wheel", "ninja", "packaging", "psutil"):
        raise RuntimeError("could not install the flash-attn build requirements")

    src = source_dir()
    repo = str(src)
    if (src / ".git").is_dir() and (src / "setup.py").is_file():
        log(f"reusing the flash-attention checkout in {src}")
    else:
        # A clone that never finished is not a checkout; start over from a clean path.
        shutil.rmtree(src, ignore_errors=True)
        src.parent.mkdir(parents=True, exist_ok=True)
        git_fetch(
            [
                "clone",
                "--depth",
                "1",
                "--branch",
                f"v{FLASH_ATTN_VERSION}",
                FLASH_ATTN_REPO,
                repo,
            ],
            env,
            "cloning flash-attention",
        )
    # The kernels are compiled against the cutlass checkout pinned by the tag.
    if not (src / "csrc" / "cutlass" / "include").is_dir():
        git_fetch(
            ["-C", repo, "submodule", "update", "--init", "csrc/cutlass"],
            env,
            "fetching csrc/cutlass",
        )
    patch_cxx_std(src)

    # --no-build-isolation builds against the target environment's torch.
    show(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            repo,
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(dest),
        ],
        env=env,
    )

    wheel = cached_wheel(cu_major, abi)
    if wheel is None:
        raise RuntimeError("no wheel was produced, see the compiler output above")
    return wheel


def build_flash_attn(torch_req):
    """Build flash-attn and return the wheel's URL.

    Every model module imports flash_attn at import time, so ending an install without it
    only moves the failure to the first model load; fail here instead of silently
    shipping a broken environment (KAVOR_INSTALL_FLASH_ATTN=0 is the way to skip it).
    """
    try:
        wheel = compile_flash_attn(torch_req)
    except Exception as exc:
        raise RuntimeError(
            f"flash-attn could not be built: {exc}. Fix the cause and run pip install "
            "-e . again, or set KAVOR_INSTALL_FLASH_ATTN=0 to install without it"
        ) from exc
    log(f"flash-attn built: {wheel}")
    return wheel


def flash_attn_requirement(variant, torch_req):
    """Resolve flash-attn: already usable -> wheel on the Kavor release -> wheel built earlier
    here -> build it now. Returns the requirement string, or None when flash-attn is left out.
    """
    probe = probe_target()
    if variant is None:
        # Without a variant the CUDA major only shows up once the pinned torch exists,
        # and pip installs torch after this step.
        pip_install(torch_req)
        probe = probe_target()
    info = probe.get("torch") or {}
    cu_major, abi = CUDA_VARIANTS.get(variant) or cuda_major(info), bool(info.get("abi", True))

    installed = probe.get("flash_attn")
    if installed and flash_attn_matches(installed, cu_major):
        log(f"flash-attn {installed} is already installed and usable, keeping it")
        return None

    if cu_major is None:
        log("no CUDA torch to build flash-attn against, it is left out")
        return None
    url = release_wheel(cu_major, abi) or cached_wheel(cu_major, abi)
    if url is None:
        url = build_flash_attn(torch_req)
    return f"flash-attn @ {url}"


# Only a running pip resolves requirements, and no single run lasts this long.
REQUIREMENTS_MEMO_AGE = 6 * 3600


def requirements_memo():
    """File that carries this pip run's requirements to its later invocations."""
    return wheel_cache_dir() / f"requirements-{os.getppid()}.json"


def get_requirements():
    """install_requires, resolved once per pip run."""
    memo = requirements_memo()
    try:
        if time.time() - memo.stat().st_mtime < REQUIREMENTS_MEMO_AGE:
            return json.loads(memo.read_text())
    except (OSError, ValueError):
        pass

    requirements = resolve_requirements()
    try:
        memo.parent.mkdir(parents=True, exist_ok=True)
        memo.write_text(json.dumps(requirements))
    except OSError:
        pass
    return requirements


def resolve_requirements():
    """Pick the torch build and the flash-attn source this machine can use."""
    variant, notes = resolve_variant(
        nvcc_major(), driver_major(), os.environ.get("KAVOR_TORCH_CUDA", "auto")
    )
    for note in notes:
        log(note)

    requirements = [f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}"]
    if variant is not None:
        if PREBUILT_TORCH:
            index = f"{PYTORCH_INDEX}/{variant}"
            requirements = [
                f"torch @ {index}/torch-{TORCH_VERSION}%2B{variant}{WHEEL_SUFFIX}",
                f"torchvision @ {index}/torchvision-{TORCHVISION_VERSION}%2B{variant}"
                f"{WHEEL_SUFFIX}",
            ]
        else:
            log("no prebuilt CUDA torch for this platform, falling back to the PyPI build")
            variant = None
    requirements += BASE_REQUIREMENTS

    if INSTALL_FLASH_ATTN:
        flash_attn = flash_attn_requirement(variant, requirements[0])
        if flash_attn:
            requirements.append(flash_attn)
    return requirements


setup(install_requires=get_requirements())
