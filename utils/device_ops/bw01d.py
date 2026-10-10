"""BW01D 设备操作实现。"""

import subprocess
import time
from collections.abc import Sequence
from typing import Any

from .base import DeviceError, DeviceOps, DeviceTarget


class Bw01dDeviceOps(DeviceOps, driver="bw01d"):
    """将 BW01D 操作映射到 SDK API, 失败时抛出 DeviceError"""

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
        raise DeviceError(f"{operation} 失败：{detail}")

    def init_connection(self) -> Any:
        """通过 SDK 初始化 gRPC 连接, 同一实例复用通道"""
        if self._channel is not None:
            return self._channel

        try:
            from wtm import bw01d

            grpc_config = {
                "ip": self.target.ip,
                "port": self.target.grpc_port,
                "is_cluster": self.target.is_cluster,
                "test_stage": self.target.test_stage,
            }

            channel = bw01d.init(grpc_config)
            if channel is None:
                raise DeviceError(f"创建 gRPC 连接失败：{self.target.ip}:{self.target.grpc_port}")
        except DeviceError:
            raise
        except Exception as error:
            raise DeviceError(f"创建 gRPC 连接失败：{self.target.ip}:{self.target.grpc_port}: {error}") from error

        self._channel = channel
        return channel

    def _set_all_power_channels(self, enabled: bool, *, timeout: float | None = None) -> Any:
        action = "打开全部电源通道" if enabled else "关闭全部电源通道"
        handle = None
        operation_failed = False
        try:
            from wtm.bw01d import evb

            handle = evb.init(
                self.target.evb_address,
                timeout=self._power_timeout if timeout is None else timeout,
            )
            response = evb.power_control.set_all_channel_enabled(handle, enabled)
            self._require_success(response, action)
            return response
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

    def evb_power_set_all_enabled(self, *, timeout: float | None = None) -> Any:
        return self._set_all_power_channels(True, timeout=timeout)

    def evb_power_set_all_disabled(self, *, timeout: float | None = None) -> Any:
        return self._set_all_power_channels(False, timeout=timeout)

    def power_cycle(self, *, off_seconds: float = 5) -> None:
        self.evb_power_set_all_disabled()
        time.sleep(off_seconds)
        self.evb_power_set_all_enabled()

    def set_phy_l2(self, ids: Sequence[int], *, timeout: float | None = None) -> Any:
        l2_ids = list(ids)
        try:
            from wtm import bw01d

            response = bw01d.set_phy_l2(
                self.init_connection(),
                l2_ids,
                timeout=self._grpc_timeout if timeout is None else timeout,
            )
            self._require_success(response, f"配置物理 L2 {l2_ids}")
            return response
        except DeviceError:
            raise
        except Exception as error:
            raise DeviceError(f"配置物理 L2 {l2_ids} 失败：{error}") from error

    def select_layer(
        self,
        hb_id: int,
        tsv_sel0: int = 0,
        tsv_sel1: int = 255,
        *,
        timeout: float | None = None,
        cim_trim0: bytes = b"",
        cim_trim1: bytes = b"",
    ) -> Any:
        try:
            from wtm import bw01d

            response = bw01d.hb_ctrl.fc_select_layer(
                self.init_connection(),
                hb_id=hb_id,
                tsv_sel0=tsv_sel0,
                tsv_sel1=tsv_sel1,
                timeout=self._grpc_timeout if timeout is None else timeout,
                cim_trim0=cim_trim0,
                cim_trim1=cim_trim1,
            )
            self._require_success(response, f"HB {hb_id} 选层")
            return response
        except DeviceError:
            raise
        except Exception as error:
            raise DeviceError(f"HB {hb_id} 选层失败：{error}") from error

    def _call_l2(self, operation: str, function_name: str, *args: Any, **kwargs: Any) -> Any:
        """在当前 gRPC 通道上执行 L2 API, 保留成功响应中的数据"""
        try:
            from wtm.bw01d import l2_ctrl

            response = getattr(l2_ctrl, function_name)(self.init_connection(), *args, **kwargs)
            self._require_success(response, operation)
            return response
        except DeviceError:
            raise
        except Exception as error:
            raise DeviceError(f"{operation}失败：{error}") from error

    def upload_program(self, l2_id: int, l2_tar: str, *, chip_id: int = 0, max_wait_seconds: int = 60) -> Any:
        return self._call_l2(
            "上传 L2 程序",
            "upload_program",
            l2_id,
            l2_tar,
            max_wait_seconds=max_wait_seconds,
            chip_id=chip_id,
        )

    def run_program(self, l2_id: int, k: int, d: int, *, max_wait_seconds: int = 7200) -> Any:
        return self._call_l2("执行 L2 烧录", "run_program", l2_id, k, d, max_wait_seconds=max_wait_seconds)

    def get_program_result(self, l2_id: int, output_dir: str = "./output", *, max_wait_seconds: int = 30) -> Any:
        return self._call_l2("获取 L2 结果", "get_program_result", l2_id, output_dir, max_wait_seconds=max_wait_seconds)

    def get_program_progress(self, l2_id: int, *, max_wait_seconds: int = 5) -> Any:
        return self._call_l2("获取 L2 进度", "get_program_progress", l2_id, max_wait_seconds=max_wait_seconds)

    def map_program(
        self,
        l2_id: int,
        l2_tar: str,
        k: int,
        d: int,
        output_dir: str = "./output",
        *,
        max_wait_seconds: int = 7200,
        chip_id: int = 0,
    ) -> Any:
        return self._call_l2(
            "执行 L2 map 烧录",
            "map_program",
            l2_id,
            l2_tar,
            k,
            d,
            output_dir=output_dir,
            max_wait_seconds=max_wait_seconds,
            chip_id=chip_id,
        )

    def infer(self, l2_id: int, output_dir: str = "./output", *, max_wait_seconds: int = 90) -> Any:
        return self._call_l2("执行 L2 推理", "infer", l2_id, output_dir=output_dir, max_wait_seconds=max_wait_seconds)

    def _run_control(self, command: str, *args: str, timeout: float, chip_add_args: bool = False) -> str:
        control_command = [
            self._control_executable,
            command,
            "--server",
            self.target.server_address,
            *args,
        ]
        if chip_add_args:
            control_command.extend(("--site", str(self.target.site)))
        try:
            result = subprocess.run(
                control_command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as error:
            raise DeviceError(f"未找到设备控制程序：{self._control_executable}") from error
        except subprocess.TimeoutExpired as error:
            raise DeviceError(f"设备控制命令超时：{command}") from error
        except OSError as error:
            raise DeviceError(f"执行设备控制命令失败：{command}：{error}") from error

        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "无错误输出"
            raise DeviceError(f"设备控制命令失败：{command}：{detail}")
        return result.stdout.strip()

    def get_sdk_version(self, *, timeout: float | None = None) -> str:
        version = self._run_control(
            "get-sdk-version",
            timeout=self._control_timeout if timeout is None else timeout,
            chip_add_args=False,
        )
        if not version:
            raise DeviceError("查询 SDK 版本成功, 但返回内容为空")
        return version

    def get_firmware_version(self, *, timeout: float | None = None) -> str:
        version = self._run_control(
            "get-n45-version",
            timeout=self._control_timeout if timeout is None else timeout,
            chip_add_args=True,
        )
        if not version:
            raise DeviceError("查询 N45 固件版本成功, 但返回内容为空")
        return version

    def switch_firmware_version(self, version: str, *, timeout: float | None = None) -> None:
        self._run_control(
            "switch-n45-version",
            version,
            timeout=self._control_timeout if timeout is None else timeout,
            chip_add_args=True,
        )
        time.sleep(3)

    def close(self) -> None:
        if self._channel is None:
            return

        channel, self._channel = self._channel, None
        try:
            channel.close()
        except Exception as error:
            raise DeviceError(f"关闭 gRPC 连接失败：{error}") from error
