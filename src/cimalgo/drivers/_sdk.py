"""
对 wtm SDK 的薄封装: 延迟导入、统一错误类型

SDK 模块只在函数内 import, 这样没有安装 SDK 的环境 (如单元测试、CI) 也能导入 cimalgo.drivers;
测试通过向 sys.modules 注入假模块来桩掉 SDK。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib import import_module
from types import ModuleType
from typing import Any

from .errors import DeviceError


def sdk(name: str) -> ModuleType:
    """import wtm.<name>; 失败时给出可操作的错误"""
    try:
        return import_module(f"wtm.{name}")
    except ImportError as error:
        raise DeviceError(f"无法导入 SDK 模块 wtm.{name}; 请先运行 pytest --download 安装 SDK: {error}") from error


def proto(name: str) -> ModuleType:
    try:
        return import_module(name)
    except ImportError as error:
        raise DeviceError(f"无法导入 SDK protobuf 模块 {name}: {error}") from error


def require_ok(response: Any, operation: str) -> Any:
    """gRPC 响应 status_code == 0 视为成功; 否则抛 DeviceError 并带上 SDK 的错误文本"""
    status = getattr(response, "status_code", None)
    if status == 0:
        return response
    detail = getattr(response, "error", None) or getattr(response, "message", None) or f"status_code={status}"
    raise DeviceError(f"{operation}失败: {detail}")


def require_success(response: Any, operation: str, *, nested: bool = False) -> Any:
    """infer 子模块的响应用 .success / .error 表示状态"""
    status = response.status if nested else response
    if not getattr(status, "success", False):
        raise DeviceError(f"{operation}失败: {getattr(status, 'error', None) or 'SDK 返回失败状态'}")
    return response


@contextmanager
def wrap_errors(operation: str) -> Iterator[None]:
    """把 SDK 抛出的任意异常统一成 DeviceError; DeviceError 原样透传"""
    try:
        yield
    except DeviceError:
        raise
    except Exception as error:
        raise DeviceError(f"{operation}失败: {error}") from error


def call(operation: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    with wrap_errors(operation):
        return require_ok(function(*args, **kwargs), operation)
