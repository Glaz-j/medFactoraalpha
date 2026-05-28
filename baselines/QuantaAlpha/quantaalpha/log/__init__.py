"""
AlphaAgent logging module.

The original QuantaAlpha workflow uses RD-Agent's logger. For PyHealth
experiments we want to import lightweight modules without installing the full
RD-Agent/Qlib stack, so this module falls back to Python logging when RD-Agent
is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from contextlib import contextmanager


try:
    from rdagent.log import rdagent_logger as _rdagent_logger
    from rdagent.log.utils import LogColors
except ModuleNotFoundError:
    _rdagent_logger = None

    class LogColors:
        """No-op color constants used when RD-Agent is not installed."""

        BOLD = ""
        CYAN = ""
        END = ""
        GREEN = ""
        MAGENTA = ""
        RED = ""
        YELLOW = ""


class _FallbackLogger:
    """Small logger with the subset of the RD-Agent logger API we use."""

    def __init__(self):
        self._logger = logging.getLogger("quantaalpha")
        self._trace_path = Path.cwd() / "git_ignore_folder" / "log"

    @property
    def log_trace_path(self) -> Path:
        return self._trace_path

    def set_trace_path(self, path) -> None:
        self._trace_path = Path(path)

    @contextmanager
    def tag(self, tag: str):
        yield

    def log_object(self, obj, tag: str | None = None) -> None:
        label = f"{tag}: " if tag else ""
        self.debug("%s%s", label, obj)

    def _log(self, level: int, message, *args, **kwargs) -> None:
        kwargs.pop("tag", None)
        self._logger.log(level, message, *args, **kwargs)

    def debug(self, message, *args, **kwargs) -> None:
        self._log(logging.DEBUG, message, *args, **kwargs)

    def info(self, message, *args, **kwargs) -> None:
        self._log(logging.INFO, message, *args, **kwargs)

    def warning(self, message, *args, **kwargs) -> None:
        self._log(logging.WARNING, message, *args, **kwargs)

    def error(self, message, *args, **kwargs) -> None:
        self._log(logging.ERROR, message, *args, **kwargs)

    def exception(self, message, *args, **kwargs) -> None:
        kwargs.pop("tag", None)
        self._logger.exception(message, *args, **kwargs)


class _AlphaAgentLoggerWrapper:
    """
    Wraps rdagent_logger and adds log_trace_path / set_trace_path.

    Other attributes/methods delegate to rdagent_logger.
    """

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    @property
    def log_trace_path(self) -> Path:
        """Return current log trace path."""
        return self._inner.storage.path

    def set_trace_path(self, path) -> None:
        """Set new log trace path."""
        from rdagent.log.storage import FileStorage

        self._inner.storage = FileStorage(Path(path))

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        if name in ("_inner",):
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)


logger = (
    _AlphaAgentLoggerWrapper(_rdagent_logger)
    if _rdagent_logger is not None
    else _FallbackLogger()
)

__all__ = ["logger", "LogColors"]
