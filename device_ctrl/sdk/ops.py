"""测试项目使用的稳定设备操作契约"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from ..errors import DeviceError


@dataclass(frozen=True)
class DeviceTarget:
    """连接和控制一台设备所需的显式配置"""

    ip: str
    grpc_port: int
    power_address: str
    control_address: str
    power_backend: str
    driver: str = "wtm_233"


class DeviceOps(ABC):
    """所有 SDK 版本向测试提供的共同原子操作契约"""

    _implementations: ClassVar[dict[str, type[DeviceOps]]] = {}

    def __init_subclass__(cls, *, driver: str | None = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if driver is None:
            return

        key = driver.casefold()
        if not key or key in cls._implementations:
            raise TypeError(f"设备驱动注册无效或重复: {driver!r}")
        cls._implementations[key] = cls

    def __init__(self, target: DeviceTarget):
        self.target = target

    @classmethod
    def for_target(cls, target: DeviceTarget, **kwargs) -> DeviceOps:
        """根据 target.driver 创建已注册的设备操作实现"""
        implementation = cls._implementations.get(target.driver.casefold())
        if implementation is None:
            supported = ", ".join(sorted(cls._implementations)) or "无"
            raise DeviceError(f"设备驱动 {target.driver!r} 没有实现; 已支持: {supported}")
        return implementation(target, **kwargs)

    @abstractmethod
    def enable_power_channels(self) -> None:
        """打开全部设备电源通道，不代表完整上电流程"""

    @abstractmethod
    def disable_power_channels(self) -> None:
        """关闭全部设备电源通道，不代表完整下电流程"""

    @abstractmethod
    def set_phy_l2(self, ids: Sequence[int]) -> None:
        """配置 PHY L2"""

    @abstractmethod
    def select_layer(self, hb_id: int, tsv_sel0: int, tsv_sel1: int) -> None:
        """选择目标层"""

    @abstractmethod
    def get_n45_version(self) -> str:
        """查询当前 N45 固件版本"""

    @abstractmethod
    def switch_n45_version(self, version: str) -> None:
        """切换 N45 固件版本"""

    @abstractmethod
    def close(self) -> None:
        """关闭底层连接并释放本实例持有的资源"""
