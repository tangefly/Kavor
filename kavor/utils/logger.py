import copy
import logging
import os
import sys

logger = logging.getLogger("kavor")

_LEVEL_ENV = "KAVOR_LOGGING_LEVEL"
_ENABLE_ENV = "KAVOR_CONFIGURE_LOGGING"
_DEFAULT_LEVEL = logging.INFO

_DATE_FORMAT = "%m-%d %H:%M:%S"
_COLORS = {
    logging.DEBUG: "\033[90m",      # gray
    logging.INFO: "\033[32m",       # green
    logging.WARNING: "\033[33m",    # yellow
    logging.ERROR: "\033[31m",      # red
    logging.CRITICAL: "\033[1;31m", # blod red
}
_RESET = "\033[0m"
_GRAY = "\033[90m"


class _ColorFormatter(logging.Formatter):

    def __init__(self, fmt: str, datefmt: str, *, use_color: bool):
        super().__init__(fmt, datefmt)
        self._use_color = use_color

    def formatTime(self, record, datefmt=None):
        # asctime 由基类 format() 内部生成，只能在 formatTime 里染色
        ts = super().formatTime(record, datefmt)
        return f"{_GRAY}{ts}{_RESET}" if self._use_color else ts

    def format(self, record):
        record = copy.copy(record)

        # location
        loc = f"[{record.filename}:{record.lineno}]"
        record.location = f"{_GRAY}{loc}{_RESET}" if self._use_color else loc

        if self._use_color:
            # level
            color = _COLORS.get(record.levelno, "")
            record.levelname = (
                f"{color}{record.levelname:<5}{_RESET}"
            )

        return super().format(record)


_LEVEL_NAMES = getattr(logging, "getLevelNamesMapping", lambda: logging._nameToLevel)


def _resolve_level(level: int | str | None) -> int | None:
    if level is None:
        level = os.environ.get(_LEVEL_ENV)
    if level is None:
        return None
    if isinstance(level, str):
        return _LEVEL_NAMES().get(level.upper())
    return level if isinstance(level, int) else None


def setup_logging(level: int | str | None = None, *, rank: int | None = None) -> None:
    if os.environ.get(_ENABLE_ENV) == "0":
        return

    resolved = _resolve_level(level)
    if resolved is None:
        resolved = _DEFAULT_LEVEL

    own = [h for h in logger.handlers if getattr(h, "_kavor", False)]
    if not own:
        handler = logging.StreamHandler()
        handler._kavor = True
        logger.addHandler(handler)
        own = [handler]

    prefix = f"[rank {rank}] " if rank is not None else ""
    fmt = f"%(levelname)s %(asctime)s {prefix}%(location)s %(message)s"
    use_color = sys.stderr.isatty() and not os.environ.get("NO_COLOR")
    own[0].setFormatter(_ColorFormatter(fmt, _DATE_FORMAT, use_color=use_color))

    logger.setLevel(resolved)

    logger.propagate = False
