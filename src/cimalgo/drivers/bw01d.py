"""BW01D 芯片驱动: DeviceOps 基类的 SDK 适配"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cimalgo.drivers import _sdk
from cimalgo.drivers.base import DeviceError, DeviceOps, DeviceTarget
from cimalgo.logging import get_logger

log = get_logger(__name__)

# gRPC 端口到 bw01d_ctrl --site 的映射
SITE_BY_GRPC_PORT = {60000: 0, 60010: 1}


class Bw01dDeviceOps(DeviceOps, driver="bw01d"):
    def __init__(
        self,
        target: DeviceTarget,
        *,
        grpc_timeout: float = 30,
        power_timeout: float = 2,
        control_timeout: float = 60,
        control_executable: str = "bw01d_ctrl",
        firmware_settle_seconds: float = 3,
    ):
        super().__init__(target)
        self._grpc_timeout = grpc_timeout
        self._power_timeout = power_timeout
        self._control_timeout = control_timeout
        self._control_executable = control_executable
        self._firmware_settle_seconds = firmware_settle_seconds
        self._channel: Any | None = None

    # ---- 连接 ----

    def channel(self) -> Any:
        if self._channel is not None:
            return self._channel
        bw01d = _sdk.sdk("bw01d")
        config = {
            "ip": self.target.ip,
            "port": self.target.grpc_port,
            "is_cluster": self.target.is_cluster,
            "test_stage": self.target.test_stage,
        }
        with _sdk.wrap_errors(f"创建 gRPC 连接 {self.target.ip}:{self.target.grpc_port}"):
            channel = bw01d.init(config)
        if channel is None:
            raise DeviceError(f"创建 gRPC 连接失败: {self.target.ip}:{self.target.grpc_port}")
        self._channel = channel
        log.debug("已建立 gRPC 连接 %s:%d", self.target.ip, self.target.grpc_port)
        return channel

    def close(self) -> None:
        channel, self._channel = self._channel, None
        if channel is None:
            return
        with _sdk.wrap_errors("关闭 gRPC 连接"):
            channel.close()
        log.debug("已关闭 gRPC 连接 %s:%d", self.target.ip, self.target.grpc_port)

    # ---- 电源 ----

    def _set_power(self, enabled: bool) -> None:
        action = "打开全部电源通道" if enabled else "关闭全部电源通道"
        evb = _sdk.sdk("bw01d.evb")
        handle = None
        failed = False
        try:
            with _sdk.wrap_errors(action):
                handle = evb.init(self.target.evb_address, timeout=self._power_timeout)
                _sdk.require_ok(evb.power_control.set_all_channel_enabled(handle, enabled), action)
        except BaseException:
            failed = True
            raise
        finally:
            if handle is not None:
                try:
                    evb.deinit(handle)
                except Exception as error:
                    if not failed:
                        raise DeviceError(f"关闭 EVB 连接失败: {error}") from error
                    log.warning("关闭 EVB 连接失败 (已有其他错误): %s", error)

    def _power_cycle(self, *, off_seconds: float) -> None:
        log.info("[%s] 下电 %.1fs 后重新上电", self.target.ip, off_seconds)
        self._set_power(False)
        time.sleep(off_seconds)
        self._set_power(True)

    # ---- 控制程序 (bw01d_ctrl) ----

    def _site(self) -> int:
        try:
            return SITE_BY_GRPC_PORT[self.target.grpc_port]
        except KeyError:
            known = ", ".join(str(port) for port in SITE_BY_GRPC_PORT)
            raise DeviceError(f"gRPC 端口 {self.target.grpc_port} 没有对应的 site; 已知端口: {known}") from None

    def _control(self, command: str, *args: str, with_site: bool) -> str:
        argv = [self._control_executable, command, "--server", self.target.server_address, *args]
        if with_site:
            argv += ["--site", str(self._site())]
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=self._control_timeout, check=False)
        except FileNotFoundError:
            raise DeviceError(f"未找到设备控制程序: {self._control_executable}") from None
        except subprocess.TimeoutExpired:
            raise DeviceError(f"设备控制命令超时: {command}") from None
        except OSError as error:
            raise DeviceError(f"执行设备控制命令失败: {command}: {error}") from error
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "无错误输出"
            raise DeviceError(f"设备控制命令失败: {command}: {detail}")
        return result.stdout.strip()

    def get_sdk_version(self) -> str:
        version = self._control("get-sdk-version", with_site=False)
        if not version:
            raise DeviceError("查询 SDK 版本成功, 但返回内容为空")
        return version

    def get_firmware_version(self) -> str:
        version = self._control("get-n45-version", with_site=True)
        if not version:
            raise DeviceError("查询 N45 固件版本成功, 但返回内容为空")
        return version

    def _switch_firmware_version(self, version: str) -> None:
        log.info("[%s] 切换 N45 固件到 %s", self.target.ip, version)
        self._control("switch-n45-version", version, with_site=True)
        time.sleep(self._firmware_settle_seconds)

    # ---- 配置 ----

    def _set_phy_l2(self, ids: Sequence[int]) -> None:
        l2_ids = list(ids)
        bw01d = _sdk.sdk("bw01d")
        _sdk.call(f"配置物理 L2 {l2_ids}", bw01d.set_phy_l2, self.channel(), l2_ids, timeout=self._grpc_timeout)

    def _select_layer(self, hb_id: int, *, tsv_sel0: int, tsv_sel1: int) -> None:
        hb_ctrl = _sdk.sdk("bw01d.hb_ctrl")
        _sdk.call(
            f"HB {hb_id} 选层",
            hb_ctrl.fc_select_layer,
            self.channel(),
            hb_id=hb_id,
            tsv_sel0=tsv_sel0,
            tsv_sel1=tsv_sel1,
            timeout=self._grpc_timeout,
        )

    # ---- 烧录 ----

    def _l2(self, operation: str, function: str, *args: Any, **kwargs: Any) -> Any:
        l2_ctrl = _sdk.sdk("bw01d.l2_ctrl")
        return _sdk.call(operation, getattr(l2_ctrl, function), self.channel(), *args, **kwargs)

    def _program_progress(self, l2_id: int) -> int:
        response = self._l2("获取 L2 进度", "get_program_progress", l2_id, max_wait_seconds=5)
        return int(response.content["data"]["progress"])

    def _program_one(
        self, l2_id: int, archive: str | Path, k: int, d: int, output_dir: Path, *, max_wait_seconds: int
    ) -> Any:
        return self._l2(
            f"L2_{l2_id} map 烧录", "map_program", l2_id, str(archive), k, d,
            output_dir=str(output_dir), max_wait_seconds=max_wait_seconds, chip_id=0,
        )  # fmt: skip
