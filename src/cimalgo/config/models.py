"""配置数据模型; 全部是不可变 dataclass, 由 loader 组装"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DownloadConfig:
    url: str | None
    archive: Path
    auth: str = "basic"


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    wrapper_dir: str | None = None
    exclude_dirs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SdkConfig:
    name: str
    chip: str
    version: str
    prefix: str
    toolchain_name: str
    install_dir: Path
    download: DownloadConfig
    extract: ExtractionConfig = ExtractionConfig()

    @property
    def scripts_dir(self) -> Path:
        return self.install_dir / "Host" / "scripts"

    @property
    def wheel_dir(self) -> Path:
        return self.install_dir / "Host" / "dist"


@dataclass(frozen=True, slots=True)
class ToolPackageConfig:
    prefix: str
    suffix: str
    release_tag: str
    release_url: str | None
    install_dir: Path
    download: DownloadConfig
    extract: ExtractionConfig = ExtractionConfig()


@dataclass(frozen=True, slots=True)
class ToolchainConfig:
    name: str
    version: str
    map_tools: ToolPackageConfig
    compiler_tools: ToolPackageConfig


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    """
    一台测试设备的静态连接信息; 密码等凭据不放在这里, 由运行时注入

    gRPC 端口选择的是板上哪颗芯片 (site), 属于用例的初始化参数, 由 DeviceSpec 声明, 不在这里。
    evb_port 和 bmc_ip 都是电源控制入口, 分别对应不同芯片, 一台设备只能配置其中一个。
    """

    name: str
    ip: str
    ssh_port: int = 22
    ssh_username: str = "root"
    server_port: int = 60001
    evb_port: int | None = None
    bmc_ip: str | None = None
    is_cluster: bool = False
    test_stage: str = "CP3"


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    sdk: SdkConfig
    toolchain: ToolchainConfig
    # 全局共用一份 netrc: 下载认证和设备 SSH 凭据都从这里读
    netrc: Path
    devices: tuple[DeviceConfig, ...] = ()
    trusted_ips: tuple[str, ...] = ()
    project_root: Path = field(default_factory=Path.cwd)

    def device(self, name: str) -> DeviceConfig:
        for device in self.devices:
            if device.name == name:
                return device
        known = ", ".join(device.name for device in self.devices) or "无"
        raise KeyError(f"设备 {name!r} 未在配置中定义; 已定义: {known}")
