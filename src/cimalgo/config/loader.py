"""从 cim_config.yaml 组装 ProjectConfig; 只读 YAML, 不读凭据, 不下载"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

from . import validators as v
from .models import (
    DeviceConfig,
    DownloadConfig,
    ProjectConfig,
    SdkConfig,
    ToolchainConfig,
    ToolPackageConfig,
)
from .validators import ProjectConfigError

DEFAULT_CONFIG_NAME = "cim_config.yaml"
SDK_ENV_VAR = "CIM_SDK"


def _load_yaml(config_path: Path) -> Mapping[str, Any]:
    try:
        content = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ProjectConfigError(f"cannot read config: {config_path}") from error
    try:
        return v.mapping(yaml.safe_load(content), "config")
    except yaml.YAMLError as error:
        raise ProjectConfigError(f"invalid YAML in {config_path}: {error}") from error


def _select_sdk(raw: Mapping[str, Any], sdk_name: str | None) -> str:
    """优先级: 显式参数 > 环境变量 CIM_SDK > active_sdk"""
    selected = sdk_name if sdk_name is not None else os.environ.get(SDK_ENV_VAR, raw.get("active_sdk"))
    selected = v.string(selected, "sdk_name")
    if selected not in v.mapping(raw.get("sdks"), "sdks"):
        raise ProjectConfigError(f"SDK {selected!r} is not configured")
    return selected


@dataclass(frozen=True, slots=True)
class _ToolchainContext:
    """一个工具链下两个工具包共享的信息"""

    field: str
    version: str
    base_url: str | None
    auth: str
    install_root: Path
    download_dir: Path


def _tool_package(toolchain_raw: Mapping[str, Any], tool: str, ctx: _ToolchainContext) -> ToolPackageConfig:
    field = f"{ctx.field}.{tool}"
    raw = v.mapping(toolchain_raw.get(tool), field)
    prefix = v.component(raw.get("prefix"), f"{field}.prefix")
    suffix = v.package_suffix(raw.get("suffix", ""), f"{field}.suffix")
    release_tag = f"v{ctx.version}{suffix}"
    filename = f"{prefix}_v{ctx.version}{suffix}.tar.gz"

    release_url = package_url = None
    if ctx.base_url is not None:
        # GitLab: <API项目地址>/releases/<tag>/downloads/<文件名>
        release_url = f"{ctx.base_url}/releases/{quote(release_tag, safe='')}"
        package_url = f"{release_url}/downloads/{quote(filename, safe='')}"

    return ToolPackageConfig(
        prefix=prefix,
        suffix=suffix,
        release_tag=release_tag,
        release_url=release_url,
        install_dir=ctx.install_root / release_tag / tool,
        download=DownloadConfig(url=package_url, archive=ctx.download_dir / filename, auth=ctx.auth),
        extract=v.extraction(raw.get("extract", {}), f"{field}.extract"),
    )


def _devices(raw: Any) -> tuple[DeviceConfig, ...]:
    if raw is None:
        return ()
    entries = v.mapping(raw, "devices")
    devices = []
    for name, item in entries.items():
        field = f"devices.{name}"
        name = v.component(name, field)
        item = v.mapping(item, field)
        # set(item) 取出 dict 键名, 并减去指定键名保留差集
        unexpected = sorted(
            set(item)
            - {
                "ip",
                "ssh_port",
                "ssh_username",
                "server_port",
                "evb_port",
                "bmc_ip",
                "is_cluster",
                "test_stage",
            }
        )
        if unexpected:
            raise ProjectConfigError(f"{field}.{unexpected[0]} is not a supported setting")
        # evb_port / bmc_ip 都是电源控制入口, 对应不同芯片, 只能二选一
        if item.get("evb_port") is not None and item.get("bmc_ip") is not None:
            raise ProjectConfigError(f"{field}: evb_port and bmc_ip are mutually exclusive")
        devices.append(
            DeviceConfig(
                name=name,
                ip=v.ip_address(item.get("ip"), f"{field}.ip"),
                ssh_port=v.port(item.get("ssh_port", 22), f"{field}.ssh_port"),
                ssh_username=v.string(item.get("ssh_username", "root"), f"{field}.ssh_username"),
                server_port=v.port(item.get("server_port", 60001), f"{field}.server_port"),
                evb_port=None if item.get("evb_port") is None else v.port(item["evb_port"], f"{field}.evb_port"),
                bmc_ip=None if item.get("bmc_ip") is None else v.ip_address(item["bmc_ip"], f"{field}.bmc_ip"),
                is_cluster=v.boolean(item.get("is_cluster", False), f"{field}.is_cluster"),
                test_stage=v.string(item.get("test_stage", "CP3"), f"{field}.test_stage"),
            )
        )
    return tuple(devices)


def load_project_config(config_path: str | Path, *, sdk_name: str | None = None) -> ProjectConfig:
    """
    加载并校验项目配置

    相对路径以 YAML 文件所在目录为根。SDK 选择优先级: sdk_name > 环境变量 CIM_SDK > active_sdk。
    url 省略或为 null 表示本地已安装, 不触发下载。
    """
    config_path = Path(config_path).expanduser().resolve()
    root = config_path.parent
    raw = _load_yaml(config_path)

    selected_sdk = _select_sdk(raw, sdk_name)
    sdks = v.mapping(raw.get("sdks"), "sdks")
    toolchains = v.mapping(raw.get("toolchains"), "toolchains")

    netrc = v.project_path(raw.get("netrc", ".netrc"), "netrc", root=root)
    download_dir = v.project_path(raw.get("download_dir"), "download_dir", root=root)
    sdk_root = v.project_path(raw.get("sdk_install_root"), "sdk_install_root", root=root)
    toolchain_root = v.project_path(raw.get("toolchain_install_root"), "toolchain_install_root", root=root)

    sdk_field = f"sdks.{selected_sdk}"
    sdk_raw = v.mapping(sdks[selected_sdk], sdk_field)
    chip = v.component(sdk_raw.get("chip"), f"{sdk_field}.chip")
    version = v.component(sdk_raw.get("version"), f"{sdk_field}.version")
    prefix = v.component(sdk_raw.get("sdk_prefix"), f"{sdk_field}.sdk_prefix")
    toolchain_name = v.string(sdk_raw.get("toolchain"), f"{sdk_field}.toolchain")
    if toolchain_name not in toolchains:
        raise ProjectConfigError(f"{sdk_field}.toolchain refers to unconfigured toolchain {toolchain_name!r}")

    # Nexus: <仓库>/<prefix>_v<版本>/<prefix>_v<版本>.tar.gz
    sdk_stem = f"{prefix}_v{version}"
    sdk_archive = f"{sdk_stem}.tar.gz"
    sdk_url = None
    if sdk_raw.get("url") is not None:
        sdk_url = (
            f"{v.base_url(sdk_raw['url'], f'{sdk_field}.url')}/{quote(sdk_stem, safe='')}/{quote(sdk_archive, safe='')}"
        )
    sdk = SdkConfig(
        name=selected_sdk,
        chip=chip,
        version=version,
        prefix=prefix,
        toolchain_name=toolchain_name,
        install_dir=sdk_root / f"{chip}_{version}",
        download=DownloadConfig(
            url=sdk_url,
            archive=download_dir / sdk_archive,
            auth=v.download_auth(sdk_raw.get("auth", "basic"), f"{sdk_field}.auth"),
        ),
        extract=v.extraction(sdk_raw.get("extract", {}), f"{sdk_field}.extract"),
    )

    toolchain_field = f"toolchains.{toolchain_name}"
    toolchain_raw = v.mapping(toolchains[toolchain_name], toolchain_field)
    toolchain_version = v.component(toolchain_raw.get("version"), f"{toolchain_field}.version")
    toolchain_base_url = None
    if toolchain_raw.get("url") is not None:
        toolchain_base_url = v.gitlab_project_url(toolchain_raw["url"], f"{toolchain_field}.url")
    ctx = _ToolchainContext(
        field=toolchain_field,
        version=toolchain_version,
        base_url=toolchain_base_url,
        auth=v.download_auth(toolchain_raw.get("auth", "basic"), f"{toolchain_field}.auth"),
        install_root=toolchain_root,
        download_dir=download_dir,
    )
    toolchain = ToolchainConfig(
        name=toolchain_name,
        version=toolchain_version,
        map_tools=_tool_package(toolchain_raw, "map_tools", ctx),
        compiler_tools=_tool_package(toolchain_raw, "compiler_tools", ctx),
    )

    trusted_raw = raw.get("trusted_ips", [])
    if not isinstance(trusted_raw, list):
        raise ProjectConfigError("trusted_ips must be a list")
    trusted_ips = tuple(v.ip_network(item, f"trusted_ips[{index}]") for index, item in enumerate(trusted_raw))

    return ProjectConfig(
        sdk=sdk,
        toolchain=toolchain,
        netrc=netrc,
        devices=_devices(raw.get("devices")),
        trusted_ips=trusted_ips,
        project_root=root,
    )
