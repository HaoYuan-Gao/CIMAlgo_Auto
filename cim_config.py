from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

from cim_config_validators import (
    ExtractionConfig,
    ProjectConfigError,
    config_mapping,
    config_string,
    base_url,
    component,
    download_auth,
    extraction_config,
    gitlab_project_url,
    package_suffix,
    project_path,
)
from cim_constants import DEFAULT_CONFIG_PATH

__all__ = [
    "DownloadConfig",
    "ExtractionConfig",
    "ProjectConfig",
    "ProjectConfigError",
    "SdkConfig",
    "ToolPackageConfig",
    "ToolchainConfig",
    "load_project_config",
]


@dataclass(frozen=True, slots=True)
class DownloadConfig:
    url: str | None
    archive: Path
    netrc: Path
    auth: str = "basic"


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
class ProjectConfig:
    sdk: SdkConfig
    toolchain: ToolchainConfig


def _load_yaml(config_path: Path) -> Mapping[str, Any]:
    try:
        content = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ProjectConfigError(f"cannot read config: {config_path}") from error

    try:
        return config_mapping(yaml.safe_load(content), "config")
    except yaml.YAMLError as error:
        raise ProjectConfigError(f"invalid YAML in {config_path}: {error}") from error


def load_project_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    sdk_name: str | None = None,
) -> ProjectConfig:
    """
    Build package URLs and paths from project.yaml's raw configuration.

    SDK 选择优先级: sdk_name > BW_SDK > active_sdk, 三者均为 sdks 下的键

    选中的 SDK 通过名称引用 toolchains 中的共享工具链配置。包名前缀中的芯片和平台信息按原样使用, 不进行推断

    对于无需下载、已在本地安装的 SDK 或工具链, 可以省略 url 或将其设为 null

    SDK 和工具链的 auth 指定 basic 或 bearer 认证, 省略时默认为 basic

    extract.wrapper_dir 配置需要去掉的最外层目录;
    extract.exclude_dirs 配置排除目录

    工具链中的 map_tools 和 compiler_tools 分别配置 prefix、suffix 与 extract;
    suffix 分别添加到对应工具包名和 Release tag 中

    本函数只读取 YAML, 不读取凭据, 也不下载文件
    """
    config_path = Path(config_path)
    raw = _load_yaml(config_path)
    sdks = config_mapping(raw.get("sdks"), "sdks")
    toolchains = config_mapping(raw.get("toolchains"), "toolchains")

    selected_sdk = sdk_name
    if selected_sdk is None:
        selected_sdk = os.environ.get("BW_SDK", raw.get("active_sdk"))
    selected_sdk = config_string(selected_sdk, "sdk_name")
    if selected_sdk not in sdks:
        raise ProjectConfigError(f"SDK {selected_sdk!r} is not configured")

    netrc = project_path(raw.get("netrc"), ".netrc")
    download_dir = project_path(raw.get("download_dir"), "download_dir")
    sdk_root = project_path(raw.get("sdk_install_root"), "sdk_install_root")
    toolchain_root = project_path(raw.get("toolchain_install_root"), "toolchain_install_root")

    sdk_field = f"sdks.{selected_sdk}"
    sdk_raw = config_mapping(sdks[selected_sdk], sdk_field)
    sdk_chip = component(sdk_raw.get("chip"), f"{sdk_field}.chip")
    sdk_version = component(sdk_raw.get("version"), f"{sdk_field}.version")
    sdk_prefix = component(sdk_raw.get("sdk_prefix"), f"{sdk_field}.sdk_prefix")
    sdk_auth = download_auth(sdk_raw.get("auth", "basic"), f"{sdk_field}.auth")
    toolchain_name = config_string(sdk_raw.get("toolchain"), f"{sdk_field}.toolchain")
    if toolchain_name not in toolchains:
        raise ProjectConfigError(f"{sdk_field}.toolchain refers to unconfigured toolchain {toolchain_name!r}")
    toolchain_field = f"toolchains.{toolchain_name}"
    toolchain_raw = config_mapping(toolchains[toolchain_name], toolchain_field)
    toolchain_version = component(toolchain_raw.get("version"), f"{toolchain_field}.version")
    toolchain_auth = download_auth(toolchain_raw.get("auth", "basic"), f"{toolchain_field}.auth")
    toolchain_base_url = None
    if toolchain_raw.get("url") is not None:
        toolchain_base_url = gitlab_project_url(toolchain_raw["url"], f"{toolchain_field}.url")

    # Nexus: <SDK仓库>/<sdk_prefix>_v<版本>/<sdk_prefix>_v<版本>.tar.gz
    sdk_stem = f"{sdk_prefix}_v{sdk_version}"
    sdk_archive = f"{sdk_stem}.tar.gz"
    sdk_url = None
    if sdk_raw.get("url") is not None:
        sdk_base_url = base_url(sdk_raw["url"], f"{sdk_field}.url")
        sdk_url = f"{sdk_base_url}/{quote(sdk_stem, safe='')}/{quote(sdk_archive, safe='')}"
    sdk = SdkConfig(
        name=selected_sdk,
        chip=sdk_chip,
        version=sdk_version,
        prefix=sdk_prefix,
        toolchain_name=toolchain_name,
        install_dir=sdk_root / f"{sdk_chip}_{sdk_version}",
        download=DownloadConfig(
            url=sdk_url,
            archive=download_dir / sdk_archive,
            netrc=netrc,
            auth=sdk_auth,
        ),
        extract=extraction_config(sdk_raw.get("extract", {}), f"{sdk_field}.extract"),
    )

    # GitLab: <API项目地址>/releases/v<版本><后缀>/downloads/<prefix>_v<版本><后缀>.tar.gz
    packages = {}
    for tool in ("map_tools", "compiler_tools"):
        package_field = f"{toolchain_field}.{tool}"
        package_raw = config_mapping(toolchain_raw.get(tool), package_field)
        prefix = component(package_raw.get("prefix"), f"{package_field}.prefix")
        suffix = package_suffix(package_raw.get("suffix", ""), f"{package_field}.suffix")
        release_tag = f"v{toolchain_version}{suffix}"
        release_url = None
        if toolchain_base_url is not None:
            release_url = f"{toolchain_base_url}/releases/{quote(release_tag, safe='')}"
        stem = f"{prefix}_v{toolchain_version}{suffix}"
        filename = f"{stem}.tar.gz"
        package_url = None
        if release_url is not None:
            package_url = f"{release_url}/downloads/{quote(filename, safe='')}"
        packages[tool] = ToolPackageConfig(
            prefix=prefix,
            suffix=suffix,
            release_tag=release_tag,
            release_url=release_url,
            install_dir=toolchain_root / release_tag / tool,
            download=DownloadConfig(
                url=package_url,
                archive=download_dir / filename,
                netrc=netrc,
                auth=toolchain_auth,
            ),
            extract=extraction_config(package_raw.get("extract", {}), f"{package_field}.extract"),
        )

    toolchain = ToolchainConfig(
        name=toolchain_name,
        version=toolchain_version,
        map_tools=packages["map_tools"],
        compiler_tools=packages["compiler_tools"],
    )
    return ProjectConfig(
        sdk=sdk,
        toolchain=toolchain,
    )
