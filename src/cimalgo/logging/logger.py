"""基于标准库 logging 的缩进 logger; 每个模块一个命名 logger, 共享 handler 和线程局部的缩进状态"""

from __future__ import annotations

import functools
import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

ROOT_NAME = "cimalgo"

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[41m",
}
_RESET = "\033[0m"

# 缩进和模块标签按线程隔离, 所有 CimLogger 实例共享
_state = threading.local()
_configure_lock = threading.Lock()
_configured = False

P = ParamSpec("P")
R = TypeVar("R")


def _indent() -> int:
    return getattr(_state, "indent", 0)


def _set_indent(value: int) -> None:
    _state.indent = max(0, value)


def _tag_stack() -> list[str]:
    stack = getattr(_state, "tags", None)
    if stack is None:
        stack = _state.tags = []
    return stack


class _IndentFormatter(logging.Formatter):
    def __init__(self, *, color: bool):
        super().__init__()
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        indent = " " * getattr(record, "indent", 0)
        tag = getattr(record, "tag", "-")
        # 去掉公共前缀, 日志里只显示 cimalgo 之下的模块名
        name = record.name.removeprefix(f"{ROOT_NAME}.")
        line = f"{timestamp} [{record.threadName}] [{record.levelname}] [{name}] [{tag}]: {indent}{record.getMessage()}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        if self.color and (color := _COLORS.get(record.levelno)):
            return f"{color}{line}{_RESET}"
        return line


def configure_logging(
    *,
    level: str | int | None = None,
    log_file: str | Path | None = None,
    color: bool | None = None,
    force: bool = False,
) -> None:
    """
    配置 cimalgo 根 logger; 只生效一次, 除非 force=True

    参数为 None 时读取环境变量 CIM_LOG_LEVEL / CIM_LOG_FILE / CIM_LOG_COLOR
    """
    global _configured
    with _configure_lock:
        if _configured and not force:
            return

        if level is None:
            level = os.environ.get("CIM_LOG_LEVEL", "INFO")
        if isinstance(level, str):
            level = _LEVELS.get(level.upper(), logging.INFO)
        if log_file is None:
            log_file = os.environ.get("CIM_LOG_FILE") or None
        if color is None:
            color = os.environ.get("CIM_LOG_COLOR", "1") != "0" and sys.stderr.isatty()

        root = logging.getLogger(ROOT_NAME)
        root.setLevel(level)
        # 保持 propagate=True, 让 pytest 的 caplog / live-log (挂在根 logger 上) 能收到记录;
        # 自己的 handler 只负责终端格式。若外部通过 logging.getLogger() 获得根 logger, 并且配了
        # StreamHandler 会重复输出一次, 这是有意接受的代价, 比在测试里收不到日志要好。
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(_IndentFormatter(color=color))
        root.addHandler(console)

        if log_file:
            path = Path(log_file).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(_IndentFormatter(color=False))
            root.addHandler(file_handler)

        _configured = True


class CimLogger(logging.LoggerAdapter):
    """给标准 Logger 增加缩进、模块标签和耗时追踪"""

    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        extra = dict(kwargs.get("extra") or {})
        extra.setdefault("indent", _indent())
        tags = _tag_stack()
        extra.setdefault("tag", tags[-1] if tags else "-")
        kwargs["extra"] = extra
        return msg, kwargs

    @contextmanager
    def indent(self, step: int = 4) -> Generator[None, None, None]:
        """缩进上下文; 退出时恢复, 线程之间互不影响"""
        _set_indent(_indent() + step)
        try:
            yield
        finally:
            _set_indent(_indent() - step)

    @contextmanager
    def tagged(self, tag: str) -> Generator[None, None, None]:
        """模块标签上下文; 用于把一段流程的日志归到同一个标签下"""
        stack = _tag_stack()
        stack.append(tag)
        try:
            yield
        finally:
            stack.pop()

    def trace(self, tag: str | None = None, *, step: int = 4) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """记录函数开始、结束和耗时; 异常时记录 traceback 并重新抛出"""

        def decorator(function: Callable[P, R]) -> Callable[P, R]:
            name = function.__qualname__
            label = tag or function.__module__.removeprefix(f"{ROOT_NAME}.")

            @functools.wraps(function)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                start = time.perf_counter()
                with self.tagged(label):
                    self.info("Start: %s", name)
                    with self.indent(step):
                        try:
                            result = function(*args, **kwargs)
                        except Exception:
                            elapsed = (time.perf_counter() - start) * 1000
                            self.exception("Error: %s (%.2f ms)", name, elapsed)
                            raise
                    elapsed = (time.perf_counter() - start) * 1000
                    self.info("End: %s (%.2f ms)", name, elapsed)
                return result

            return wrapper

        return decorator


def get_logger(name: str | None = None) -> CimLogger:
    """返回命名 logger; 首次调用时按环境变量完成配置"""
    configure_logging()
    if name is None:
        qualified = ROOT_NAME
    elif name == ROOT_NAME or name.startswith(f"{ROOT_NAME}."):
        qualified = name
    else:
        qualified = f"{ROOT_NAME}.{name}"
    return CimLogger(logging.getLogger(qualified), {})
