"""由稳定原子操作组合而成的复用流程"""

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from .errors import DeviceError
from .sdk import DeviceOps


@dataclass(frozen=True)
class LayerSelection:
    """一次 HB 选层参数"""

    hb_id: int
    tsv_sel0: int
    tsv_sel1: int


class DeviceWorkflows:
    """只包含多个测试共同需要且步骤顺序明确的流程"""

    def __init__(
        self,
        device: DeviceOps,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.device = device
        self._sleep = sleep

    def power_cycle_channels(self, off_seconds: float = 5) -> None:
        """关闭全部电源通道，等待后再打开；不代表 SDK 的完整上下电流程"""
        if off_seconds < 0:
            raise ValueError("off_seconds 不能小于 0")

        self.device.disable_power_channels()
        self._sleep(off_seconds)
        self.device.enable_power_channels()

    def configure_layers(
        self,
        physical_l2_ids: Sequence[int],
        selections: Iterable[LayerSelection],
    ) -> None:
        """先下发可用物理 L2，再依次完成 HB 选层"""
        self.device.set_phy_l2(physical_l2_ids)
        for selection in selections:
            self.device.select_layer(
                selection.hb_id,
                selection.tsv_sel0,
                selection.tsv_sel1,
            )

    def ensure_n45_version(self, target_version: str) -> bool:
        """确保固件版本，并返回本次是否执行了切换"""
        if self.device.get_n45_version() == target_version:
            return False

        self.device.switch_n45_version(target_version)

        # 等待、重试和验证语义确认后，再替换这里的立即验证。
        actual_version = self.device.get_n45_version()
        if actual_version != target_version:
            raise DeviceError(f"N45 固件版本不符：期望 {target_version}，实际 {actual_version}")
        return True
