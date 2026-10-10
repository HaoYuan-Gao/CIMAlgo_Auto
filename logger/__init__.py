"""Expose a process-wide logger that is created on first use."""

from threading import Lock
from typing import Any, cast

from .logger import Logger


_logger: Logger | None = None
_logger_lock = Lock()


def get_logger() -> Logger:
    """Return the shared logger, creating it exactly once.

    The first check keeps normal log calls lock-free after initialization. The
    second check, made while holding the lock, prevents concurrent callers from
    constructing multiple ``Logger`` instances.
    """
    global _logger

    instance = _logger
    if instance is None:
        with _logger_lock:
            instance = _logger
            if instance is None:
                instance = Logger()
                _logger = instance

    return instance


class LoggerProxy:
    """Forward attribute access to the lazily created shared logger."""

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(get_logger(), name)


# The cast gives callers and IDEs the Logger API while the runtime object stays
# a lightweight proxy, so importing this package still has no logging side
# effects.
logger = cast(Logger, LoggerProxy())


__all__ = ["logger", "get_logger", "Logger"]
