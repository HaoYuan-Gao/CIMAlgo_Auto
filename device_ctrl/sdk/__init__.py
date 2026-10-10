"""SDK 设备操作契约及各版本实现"""

from .ops import DeviceOps, DeviceTarget
from .wtm_233 import Wtm233DeviceOps

__all__ = [
    "DeviceOps",
    "DeviceTarget",
    "Wtm233DeviceOps",
]
