"""
芯片驱动: 抽象基类 base(芯片操作抽象) + 各芯片的 SDK 适配 (每颗芯片一个模块, 如 bw01d.py)

SDK 导入统一经 _sdk 延迟进行。
"""

# 导入即注册 driver
from . import bw01d as _bw01d
from .base import DeviceOps, DeviceTarget, InitState
from .errors import DeviceError
from .infer import InferSession, NodeResult

__all__ = ["DeviceError", "DeviceOps", "DeviceTarget", "InferSession", "InitState", "NodeResult"]
