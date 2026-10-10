"""wtm 2.3.3 设备操作实现"""

import subprocess
from collections.abc import Sequence
from typing import Any

from ..errors import DeviceError
from .ops import DeviceOps, DeviceTarget


class Wtm233DeviceOps(DeviceOps, driver="wtm_233"):
    """将 DeviceOps 映射到 wtm 2.3.3，并统一为失败抛 DeviceError。"""

    _POWER_BACKEND = "evb"
    _SITE_BY_GRPC_PORT = {
        60000: 0,
        60010: 1,
    }

    def __init__(
        self,
        target: DeviceTarget,
        *,
        grpc_timeout: float = 30,
        power_timeout: float = 2,
        control_timeout: float = 60,
        control_executable: str = "bw01d_ctrl",
    ):
        super().__init__(target)
        self._grpc_timeout = grpc_timeout
        self._power_timeout = power_timeout
        self._control_timeout = control_timeout
        self._control_executable = control_executable
        self._channel: Any | None = None

    @staticmethod
    def _require_success(response: Any, operation: str) -> None:
        status_code = getattr(response, "status_code", None)
        if status_code == 0:
            return

        detail = getattr(response, "error", None) or getattr(response, "message", None) or f"status_code={status_code}"
        raise DeviceError(f"{operation}失败：{detail}")

    def _get_channel(self) -> Any:
        if self._channel is not None:
            return self._channel

        try:
            from wtm.channel_handle import ChannelHandle

            channel = ChannelHandle({"ip": self.target.ip, "port": self.target.grpc_port})
            if not channel.get_channel():
                raise DeviceError(f"创建 gRPC 连接失败：{self.target.ip}:{self.target.grpc_port}")
        except DeviceError:
            raise
        except Exception as error:
            raise DeviceError(f"创建 gRPC 连接失败：{self.target.ip}:{self.target.grpc_port}: {error}") from error

        self._channel = channel
        return channel

    def _set_all_power_channels(self, enabled: bool) -> None:
        if self.target.power_backend != self._POWER_BACKEND:
            raise DeviceError(
                f"wtm 2.3.3 初版仅支持 power_backend={self._POWER_BACKEND!r}，实际为 {self.target.power_backend!r}"
            )

        action = "打开全部电源通道" if enabled else "关闭全部电源通道"
        handle = None
        operation_failed = False
        try:
            from wtm.bw01d import evb

            handle = evb.init(
                self.target.power_address,
                timeout=self._power_timeout,
            )
            response = evb.power_control.set_all_channel_enabled(handle, enabled)
            self._require_success(response, action)
        except DeviceError:
            operation_failed = True
            raise
        except Exception as error:
            operation_failed = True
            raise DeviceError(f"{action}失败：{error}") from error
        finally:
            if handle is not None:
                try:
                    evb.deinit(handle)
                except Exception as error:
                    if not operation_failed:
                        raise DeviceError(f"关闭 EVB 连接失败：{error}") from error

    def enable_power_channels(self) -> None:
        self._set_all_power_channels(True)

    def disable_power_channels(self) -> None:
        self._set_all_power_channels(False)

    def set_phy_l2(self, ids: Sequence[int]) -> None:
        l2_ids = list(ids)
        try:
            from wtm import bw01d

            response = bw01d.set_phy_l2(
                self._get_channel(),
                l2_ids=l2_ids,
                timeout=self._grpc_timeout,
            )
            self._require_success(response, f"配置物理 L2 {l2_ids}")
        except DeviceError:
            raise
        except Exception as error:
            raise DeviceError(f"配置物理 L2 {l2_ids} 失败：{error}") from error

    def select_layer(self, hb_id: int, tsv_sel0: int, tsv_sel1: int) -> None:
        try:
            from wtm import bw01d

            response = bw01d.hb_ctrl.fc_select_layer(
                self._get_channel(),
                hb_id=hb_id,
                tsv_sel0=tsv_sel0,
                tsv_sel1=tsv_sel1,
                timeout=self._grpc_timeout,
            )
            self._require_success(response, f"HB {hb_id} 选层")
        except DeviceError:
            raise
        except Exception as error:
            raise DeviceError(f"HB {hb_id} 选层失败：{error}") from error

    def _get_site(self) -> int:
        try:
            return self._SITE_BY_GRPC_PORT[self.target.grpc_port]
        except KeyError:
            raise DeviceError(f"无法由 gRPC 端口 {self.target.grpc_port} 确定 bw01d_ctrl site") from None

    def _run_control(self, command: str, *args: str, timeout: float) -> str:
        control_command = [
            self._control_executable,
            command,
            "--server",
            self.target.control_address,
            *args,
            "--site",
            str(self._get_site()),
        ]
        try:
            result = subprocess.run(
                control_command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as error:
            raise DeviceError(f"未找到 N45 控制程序：{self._control_executable}") from error
        except subprocess.TimeoutExpired as error:
            raise DeviceError(f"N45 控制命令超时：{command}") from error
        except OSError as error:
            raise DeviceError(f"执行 N45 控制命令失败：{command}：{error}") from error

        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "无错误输出"
            raise DeviceError(f"N45 控制命令失败：{command}：{detail}")
        return result.stdout.strip()

    def get_n45_version(self) -> str:
        version = self._run_control(
            "get-n45-version",
            timeout=self._control_timeout,
        )
        if not version:
            raise DeviceError("查询 N45 固件版本成功，但返回内容为空")
        return version

    def switch_n45_version(self, version: str) -> None:
        self._run_control(
            "switch-n45-version",
            version,
            timeout=self._control_timeout,
        )

    def close(self) -> None:
        if self._channel is None:
            return

        channel, self._channel = self._channel, None
        try:
            channel.close()
        except Exception as error:
            raise DeviceError(f"关闭 gRPC 连接失败：{error}") from error
