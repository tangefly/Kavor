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


class _ColorFormatter(logging.Formatter):

    def __init__(self, fmt: str, datefmt: str, *, use_color: bool):
        super().__init__(fmt, datefmt)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if not self._use_color:
            return super().format(record)

        record = copy.copy(record)
        color = _COLORS.get(record.levelno, "")
        levelname = f"{record.levelname}:"
        padded = f"{levelname:<9}"
        record.levelname = (
            f"{color}{record.levelname}{_RESET}"
            f"{padded[len(record.levelname):]}"
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
    fmt = f"%(levelname)s %(asctime)s {prefix}[%(filename)s:%(lineno)d] %(message)s"
    use_color = sys.stderr.isatty() and not os.environ.get("NO_COLOR")
    own[0].setFormatter(_ColorFormatter(fmt, _DATE_FORMAT, use_color=use_color))

    logger.setLevel(resolved)

    logger.propagate = False
