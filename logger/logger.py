import os
import time
import logging
import threading
import functools
from pathlib import Path


class _IndentContext:
    def __init__(self, logger, step=2):
        self.logger = logger
        self.step = step

    def __enter__(self):
        self.logger._increase_indent(self.step)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger._decrease_indent(self.step)


class _IndentFormatter(logging.Formatter):
    COLOR = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
        "RESET": "\033[0m",
    }

    def __init__(self, use_color=True, include_thread=True):
        super().__init__()
        self.use_color = use_color
        self.include_thread = include_thread

    def format(self, record):
        asctime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        level = record.levelname
        name = record.name
        msg = record.getMessage()

        indent = getattr(record, "indent", 0)
        indent_str = " " * indent

        module_tag = getattr(record, "module_tag", "-")

        thread_part = f" [{record.threadName}]" if self.include_thread else ""
        base = f"{asctime}{thread_part} [{level}] [{name}] [{module_tag}]: {indent_str}{msg}"

        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            base = f"{base}\n{exc_text}"

        if self.use_color:
            color = self.COLOR.get(level, "")
            reset = self.COLOR["RESET"]
            return f"{color}{base}{reset}"
        return base


class Logger:
    LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARNING,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    def __init__(self, name="CIM"):
        self.name = name
        self._local = threading.local()
        self._local.indent = 0

        level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        level = self.LEVELS.get(level_str, logging.INFO)

        self.to_file = os.getenv("LOG_TO_FILE", "0") == "1"
        self.log_file = os.getenv("LOG_FILE", "app.log")
        self.indent_step = int(os.getenv("LOG_INDENT_STEP", "2"))

        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.propagate = False

        if not self._logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self):
        # Console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self._logger.level)
        console_handler.setFormatter(_IndentFormatter(use_color=True, include_thread=True))
        self._logger.addHandler(console_handler)

        # File
        if self.to_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(self._logger.level)
            file_handler.setFormatter(_IndentFormatter(use_color=False, include_thread=True))
            self._logger.addHandler(file_handler)

    def _get_indent(self):
        return getattr(self._local, "indent", 0)

    def _increase_indent(self, step=None):
        if not hasattr(self._local, "indent"):
            self._local.indent = 0
        self._local.indent += step if step is not None else self.indent_step

    def _decrease_indent(self, step=None):
        if not hasattr(self._local, "indent"):
            self._local.indent = 0
        self._local.indent = max(0, self._local.indent - (step if step is not None else self.indent_step))

    def indent(self, step=None):
        return _IndentContext(self, step if step is not None else self.indent_step)

    def _push_module_tag(self, tag):
        if not hasattr(self._local, "module_stack"):
            self._local.module_stack = []
        self._local.module_stack.append(tag)

    def _pop_module_tag(self):
        if hasattr(self._local, "module_stack") and self._local.module_stack:
            self._local.module_stack.pop()

    def _get_module_tag(self):
        if hasattr(self._local, "module_stack") and self._local.module_stack:
            return self._local.module_stack[-1]
        return "-"

    def _log(self, level, msg, *args, exc_info=False, extra=None, **kwargs):
        base_extra = {"indent": self._get_indent(), "module_tag": self._get_module_tag()}

        if extra:
            base_extra.update(extra)

        self._logger.log(level, msg, *args, extra=base_extra, exc_info=exc_info, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._log(logging.INFO, msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self._log(logging.WARNING, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self._log(logging.ERROR, msg, *args, exc_info=True, **kwargs)

    def trace(self, module=None):
        """
        Usage:
            @logger.trace()
            def foo(...): ...

            @logger.trace("workflow")
            def bar(...): ...
        """

        def decorator(func):
            trace_name = module or func.__module__
            func_name = func.__qualname__

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()

                # 设置上下文
                self._push_module_tag(trace_name)

                self.info(f"Start: {func_name}")
                self._increase_indent()

                elapsed = None
                try:
                    result = func(*args, **kwargs)
                    elapsed = (time.perf_counter() - start) * 1000
                    return result
                except Exception:
                    elapsed = (time.perf_counter() - start) * 1000
                    self.exception(f"Error: {func_name} ({elapsed:.2f} ms)")
                    raise
                finally:
                    self._decrease_indent()
                    if elapsed is None:
                        elapsed = (time.perf_counter() - start) * 1000

                    self.info(f"End: {func_name} ({elapsed:.2f} ms)")

                    # 清理上下文
                    self._pop_module_tag()

            return wrapper

        return decorator
