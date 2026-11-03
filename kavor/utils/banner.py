import os
import platform
import sys

_DISABLE_ENV = "KAVOR_NO_BANNER"
_TAGLINE = "a lightweight vLLM implementation"

# figlet "ANSI Shadow" 字体的 KAVOR,六行一一对齐。
_ART = """\
██╗  ██╗ █████╗ ██╗   ██╗ ██████╗ ██████╗
██║ ██╔╝██╔══██╗██║   ██║██╔═══██╗██╔══██╗
█████╔╝ ███████║██║   ██║██║   ██║██████╔╝
██╔═██╗ ██╔══██║╚██╗ ██╔╝██║   ██║██╔══██╗
██║  ██║██║  ██║ ╚████╔╝ ╚██████╔╝██║  ██║
╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝   ╚═════╝ ╚═╝  ╚═╝"""

_RESET = "\033[0m"
_LOGO_COLOR = "\033[1;36m"  # bold cyan
_DIM = "\033[90m"


def get_version() -> str:
    """kavor 包版本;源码目录里没装元数据时返回空串。"""
    try:
        from importlib.metadata import version

        return version("kavor")
    except Exception:
        return ""


def runtime_info() -> dict[str, str]:
    """运行环境信息(Python / PyTorch / CUDA),读不到的项直接省略。"""
    info = {"Python": platform.python_version()}
    try:
        import torch
    except ImportError:
        return info
    info["PyTorch"] = torch.__version__
    if torch.version.cuda:
        info["CUDA"] = torch.version.cuda
    return info


def _compose(version: str | None, info: dict[str, str] | None):
    """把横幅拆成 (文本, 颜色) 列表,渲染和染色共用同一份布局。"""
    version = get_version() if version is None else version
    art = [line.rstrip() for line in _ART.splitlines()]
    width = max(len(line) for line in art)
    head = f"Kavor v{version}" if version else "Kavor"
    tagline = f"{head} · {_TAGLINE}".center(width).rstrip()

    lines: list[tuple[str, str | None]] = [(line, _LOGO_COLOR) for line in art]
    lines.append((tagline, _DIM))

    rows = list((info or {}).items())
    if rows:
        key_width = max(len(key) for key, _ in rows)
        lines.append(("", None))
        lines.extend(
            (f"{key.ljust(key_width)}  {value}", None) for key, value in rows
        )
    return lines


def render_banner(version: str | None = None,
                  info: dict[str, str] | None = None) -> str:
    """返回不带颜色、不带结尾换行的横幅文本。"""
    return "\n".join(text for text, _ in _compose(version, info))


def _supports_color(stream) -> bool:
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def print_banner(info: dict[str, str] | None = None,
                 version: str | None = None,
                 stream=None) -> None:
    """打印横幅;KAVOR_NO_BANNER=1 时静默跳过。"""
    if os.environ.get(_DISABLE_ENV) == "1":
        return

    stream = sys.stderr if stream is None else stream
    color = _supports_color(stream)
    lines = _compose(version, info)
    if not color:
        print("\n".join(text for text, _ in lines), file=stream, flush=True)
        return

    out = []
    for text, code in lines:
        out.append(f"{code}{text}{_RESET}" if code and text else text)
    print("\n".join(out), file=stream, flush=True)
