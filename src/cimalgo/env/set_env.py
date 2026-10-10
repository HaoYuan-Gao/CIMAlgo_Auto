"""
按芯片准备测试进程的环境: 下载解压、工具链权限、Python 导入路径和环境变量

SetEnv 固定流程, 子类通过覆盖少量 hook 描述增量; 新增芯片时定义一个带 chip= 的子类即可。
不依赖 pytest: 错误统一抛 SetEnvError, 由调用方决定如何呈现。
"""

from __future__ import annotations

import netrc
import os
import stat
import subprocess
import sys
import tempfile
from base64 import b64encode
from collections.abc import Mapping, Sequence
from importlib import invalidate_caches, metadata
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlsplit

from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from cimalgo.config import DownloadConfig, ProjectConfig, SdkConfig, ToolPackageConfig
from cimalgo.logging import get_logger

from . import archive

log = get_logger(__name__)


class SetEnvError(RuntimeError):
    """环境准备失败"""


# ---------------------------------------------------------------------------
# 下载
# ---------------------------------------------------------------------------


def _authorization_header(download: DownloadConfig, netrc_path: Path) -> str:
    hostname = urlsplit(download.url or "").hostname
    try:
        credentials = netrc.netrc(str(netrc_path)).authenticators(hostname or "")
    except (OSError, netrc.NetrcParseError, UnicodeError):
        # 解析异常可能含凭据原文, 不展示
        raise SetEnvError(f"无法解析 netrc 文件: {netrc_path}") from None
    if credentials is None:
        raise SetEnvError(f"netrc 中缺少 {hostname} 的凭据")

    login, _, password = credentials
    if download.auth == "basic":
        if not login or ":" in login:
            raise SetEnvError(f"netrc 中 {hostname} 的 login 不能为空或包含冒号")
        encoded = b64encode(f"{login}:{password or ''}".encode()).decode("ascii")
        return f"Authorization: Basic {encoded}\n"
    if not password or "\r" in password or "\n" in password:
        raise SetEnvError(f"netrc 中缺少 {hostname} 的有效 Bearer 令牌 (password 字段)")
    return f"Authorization: Bearer {password}\n"


def download_archive(label: str, download: DownloadConfig, netrc_path: Path) -> None:
    """用 curl 下载; 已存在或无 URL 时跳过。认证头经 stdin 传入, 不出现在进程参数里"""
    if download.url is None:
        return
    if download.archive.is_file():
        log.info("%s 已存在: %s", label, download.archive)
        return
    if not netrc_path.is_file():
        raise SetEnvError(f"netrc 文件不存在: {netrc_path}")
    log.info("下载 %s: %s", label, download.archive.name)

    authorization = _authorization_header(download, netrc_path)
    download.archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=f".{download.archive.name}-", dir=download.archive.parent) as tmp:
            temporary = Path(tmp) / "download.part"
            command = [
                "curl", "--fail", "--location", "--show-error", "--progress-bar",
                "--header", "@-", "--output", str(temporary), download.url,
            ]  # fmt: skip
            try:
                subprocess.run(command, check=True, input=authorization, text=True)
            except FileNotFoundError:
                raise SetEnvError("未找到 curl 命令") from None
            temporary.replace(download.archive)
    except (OSError, subprocess.CalledProcessError) as error:
        raise SetEnvError(f"下载 {label} 失败: {download.url}") from error


# ---------------------------------------------------------------------------
# 解压、权限、wheel
# ---------------------------------------------------------------------------


def install_archive(label: str, package: SdkConfig | ToolPackageConfig) -> None:
    """在临时目录解压, 成功后原子移动到 install_dir; 已有非空目录不覆盖"""
    install_dir = package.install_dir
    log.info("安装 %s: %s", label, install_dir)
    try:
        install_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{install_dir.name}-", dir=install_dir.parent) as tmp:
            content = Path(tmp) / "content"
            content.mkdir()
            source = archive.extract(
                package.download.archive,
                content,
                wrapper_dir=package.extract.wrapper_dir,
                exclude_dirs=package.extract.exclude_dirs,
            )
            if not any(source.iterdir()):
                raise SetEnvError(f"{label} 压缩包没有可安装的内容: {package.download.archive}")
            source.rename(install_dir)
    except (OSError, archive.ArchiveError) as error:
        raise SetEnvError(f"安装 {label} 失败: {error}") from error


def ensure_executable(targets: Sequence[tuple[Path, Sequence[str]]]) -> None:
    """为 (安装目录, 相对路径) 列表递归补充所有者执行权限; 不跟随越出安装目录的链接"""

    def fix(path: Path, root: Path) -> None:
        if path.is_symlink():
            if path.is_dir() and not path.resolve().is_relative_to(root):
                raise SetEnvError(f"执行权限路径越出安装目录: {path}")
            return
        if not path.resolve().is_relative_to(root):
            raise SetEnvError(f"执行权限路径越出安装目录: {path}")
        mode = path.stat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            return
        if not mode & stat.S_IXUSR:
            path.chmod(stat.S_IMODE(mode) | stat.S_IXUSR)
        if stat.S_ISDIR(mode):
            for child in path.iterdir():
                fix(child, root)

    for install_dir, relative_paths in targets:
        for relative in relative_paths:
            path = install_dir / relative
            if not path.exists():
                continue
            try:
                fix(path, install_dir.resolve())
            except OSError as error:
                raise SetEnvError(f"设置执行权限失败: {path}: {error}") from error
            log.debug("已补充执行权限: %s", path)


def sync_wheels(sdk: SdkConfig) -> None:
    """把当前 Python 环境中的包版本对齐 SDK 自带的 wheel; 同版本跳过, 支持升级和降级"""
    wheels = sorted(sdk.wheel_dir.glob("*.whl"))
    if not wheels:
        raise SetEnvError(f"SDK wheel 目录不存在或没有 whl 文件: {sdk.wheel_dir}")

    expected: dict[str, Version] = {}
    pending: list[str] = []
    try:
        for wheel in wheels:
            name, target, _, _ = parse_wheel_filename(wheel.name)
            if name in expected:
                raise SetEnvError(f"SDK wheel 目录中有多个同名包: {name}")
            expected[name] = target
            try:
                installed: str | None = metadata.version(name)
            except metadata.PackageNotFoundError:
                installed = None
            if installed is not None and Version(installed) == target:
                continue
            log.info("wheel %s: %s -> %s", name, installed or "未安装", target)
            pending.append(str(wheel))

        if not pending:
            log.info("SDK Python 包版本已匹配, 跳过安装")
            return

        log.info("同步 Python 环境: %s", sys.executable)
        pip_install = [sys.executable, "-m", "pip", "install"]
        pip_flags = ["--no-index", "--no-deps", "--no-user", "--disable-pip-version-check", "--no-input"]
        subprocess.run([*pip_install, *pip_flags, *pending], check=True)
        invalidate_caches()
        for package_name, target in expected.items():
            actual = metadata.version(package_name)
            if Version(actual) != target:
                raise SetEnvError(f"SDK wheel 同步后版本不匹配: {package_name}, 期望 {target}, 实际 {actual}")
    except (
        OSError,
        subprocess.CalledProcessError,
        InvalidWheelFilename,
        InvalidVersion,
        metadata.PackageNotFoundError,
    ) as error:
        raise SetEnvError(f"同步 SDK wheel 失败 ({sys.executable}): {error}") from error


# ---------------------------------------------------------------------------
# SetEnv
# ---------------------------------------------------------------------------


class SetEnv:
    """
    环境准备流程; 子类覆盖 packages / executable_targets / sync_wheels / import_paths / chip_environment

    prepare_packages 在一次运行中只需执行一次 (xdist 主进程); environment 和 import_paths 每个进程都要应用。
    """

    _implementations: ClassVar[dict[str, type[SetEnv]]] = {}

    def __init_subclass__(cls, *, chip: str | None = None, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if chip is None:
            return
        key = chip.casefold()
        if not key or key in cls._implementations:
            raise TypeError(f"环境准备芯片注册无效或重复: {chip!r}")
        cls._implementations[key] = cls

    def __init__(self, config: ProjectConfig):
        self.config = config

    @classmethod
    def for_config(cls, config: ProjectConfig) -> SetEnv:
        implementation = cls._implementations.get(config.sdk.chip.casefold())
        if implementation is None:
            supported = ", ".join(sorted(cls._implementations)) or "无"
            raise SetEnvError(f"芯片 {config.sdk.chip!r} 没有环境准备实现; 已支持: {supported}")
        return implementation(config)

    # ---- hooks ----

    def packages(self) -> tuple[tuple[str, SdkConfig | ToolPackageConfig], ...]:
        return (("SDK", self.config.sdk),)

    def executable_targets(self) -> tuple[tuple[Path, tuple[str, ...]], ...]:
        return ()

    def sync_wheels(self) -> None:
        """默认不需要同步 wheel"""

    def import_paths(self) -> tuple[Path, ...]:
        """需要加到 sys.path 最前面的目录"""
        return ()

    def chip_environment(self) -> Mapping[str, str]:
        return {}

    # ---- 流程 ----

    @log.trace()
    def prepare_packages(self) -> None:
        for label, package in self.packages():
            install_dir = package.install_dir
            if install_dir.is_dir() and any(install_dir.iterdir()):
                log.info("%s 已安装: %s", label, install_dir)
                continue
            if install_dir.is_symlink() or (install_dir.exists() and not install_dir.is_dir()):
                raise SetEnvError(f"安装路径不是可用目录: {install_dir}")
            download = package.download
            if download.url is None and not download.archive.is_file():
                log.info("%s 未配置下载地址且无本地压缩包, 跳过自动安装", label)
                continue
            download_archive(label, download, self.config.netrc)
            install_archive(label, package)
        ensure_executable(self.executable_targets())

    def apply_import_paths(self) -> None:
        paths = [str(path) for path in self.import_paths()]
        sys.path[:] = [path for path in sys.path if path not in paths]
        sys.path[:0] = paths

    def environment(self) -> dict[str, str]:
        sdk = self.config.sdk
        result = {
            "CIM_SDK": sdk.name,
            "CIM_CHIP": sdk.chip,
            "CIM_SDK_VERSION": sdk.version,
            "CIM_SDK_HOME": str(sdk.install_dir),
        }
        result.update(self.chip_environment())
        for name, value in result.items():
            if not name or not isinstance(value, str):
                raise SetEnvError(f"环境变量 {name!r} 的值必须是字符串: {value!r}")
        return result

    @staticmethod
    def prepend_paths(name: str, *paths: str | Path) -> str:
        current = os.environ.get(name, "").split(os.pathsep)
        candidates = [*(str(path) for path in paths), *current]
        return os.pathsep.join(dict.fromkeys(path for path in candidates if path))


class _BwSetEnv(SetEnv):
    """BW01 系列共用流程"""

    def packages(self) -> tuple[tuple[str, SdkConfig | ToolPackageConfig], ...]:
        toolchain = self.config.toolchain
        return (
            ("SDK", self.config.sdk),
            ("map tools", toolchain.map_tools),
            ("compiler tools", toolchain.compiler_tools),
        )

    def executable_targets(self) -> tuple[tuple[Path, tuple[str, ...]], ...]:
        toolchain = self.config.toolchain
        return (
            (toolchain.compiler_tools.install_dir, ("bin", "script")),
            (toolchain.map_tools.install_dir, ("map_tool", "script")),
        )

    def sync_wheels(self) -> None:
        sync_wheels(self.config.sdk)

    def import_paths(self) -> tuple[Path, ...]:
        scripts = self.config.sdk.scripts_dir
        return (scripts / "proto_gen", scripts / "src_py")

    def chip_environment(self) -> Mapping[str, str]:
        configured = os.environ.get("CIM_TC_HOME")
        toolchain_home = (
            Path(configured).expanduser() if configured else self.config.toolchain.compiler_tools.install_dir
        )
        return {
            "CIM_TC_HOME": str(toolchain_home),
            "CIM_TC_SETTINGS": os.environ.get(
                "CIM_TC_SETTINGS", str(toolchain_home / "config" / "cim_tc_settings.json")
            ),
            "LOGIC_TO_PHYSICS_WITH_REDUNDANCY": os.environ.get("LOGIC_TO_PHYSICS_WITH_REDUNDANCY", "1"),
            "LD_LIBRARY_PATH": self.prepend_paths("LD_LIBRARY_PATH", Path(sys.prefix) / "lib", toolchain_home / "lib"),
        }


class Bw01dSetEnv(_BwSetEnv, chip="bw01d"):
    """BW01D"""


class Bw01SetEnv(_BwSetEnv, chip="bw01"):
    """BW01; 当前与 BW01D 共用流程"""
