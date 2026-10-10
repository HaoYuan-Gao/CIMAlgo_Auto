"""
按芯片部署测试进程需要的环境: 下载解压、工具链权限、Python 导入路径和环境变量。

SetEnv 固定部署流程本身 (下载、安装、权限修复、wheel 同步、导入路径)，并通过
build_environment 返回待应用的环境变量字典; 实际应用与恢复交给调用方 (conftest.py)
通过 pytest 的 monkeypatch 完成。各芯片实现只通过覆盖少量 hook 描述自己的增量:
需要准备哪些包 (packages)、哪些目录需要补执行权限 (executable_targets)、是否需要
同步 SDK wheel (sync_wheels)、是否需要调整导入路径 (add_import_paths)、以及自己
的环境变量 (chip_environment)。

新增芯片时定义一个带 chip= 的子类即可，调用方 (conftest.py)不需要增加分支或改动。
"""

from __future__ import annotations

import netrc
import os
import stat
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from base64 import b64encode
from collections.abc import Mapping, Sequence
from importlib import invalidate_caches, metadata
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar
from urllib.parse import urlsplit

import pytest
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from utils import archive_tool

if TYPE_CHECKING:
    from cim_config import DownloadConfig, ProjectConfig, SdkConfig


class SetEnvError(ValueError):
    """环境部署实现缺失或生成了无效配置。"""


def _download_authorization(download: DownloadConfig) -> str:
    """统一读取 netrc, 按配置的认证方式生成 Authorization 请求头"""
    hostname = urlsplit(download.url).hostname
    try:
        credentials = netrc.netrc(str(download.netrc)).authenticators(hostname)
    except (OSError, netrc.NetrcParseError, UnicodeError):
        # netrc 的解析异常可能包含凭据原文, 不展示原始异常
        raise pytest.UsageError(f"无法解析 netrc 文件: {download.netrc}") from None
    if credentials is None:
        raise pytest.UsageError(f"netrc 中缺少 {hostname} 的凭据")

    login, _, password = credentials
    if download.auth == "basic":
        if not login or ":" in login:
            raise pytest.UsageError(f"netrc 中 {hostname} 的 login 不能为空或包含冒号")
        encoded = b64encode(f"{login}:{password or ''}".encode("utf-8")).decode("ascii")
        return f"Authorization: Basic {encoded}\n"
    if download.auth == "bearer":
        if not password or "\r" in password or "\n" in password:
            raise pytest.UsageError(f"netrc 中缺少 {hostname} 的有效 Bearer 令牌 (password 字段)")
        return f"Authorization: Bearer {password}\n"
    raise pytest.UsageError("下载认证方式必须为 basic 或 bearer")


def _download_archive(label: str, download: DownloadConfig) -> None:
    """使用 curl 和 netrc 下载单个压缩包, 已存在或无 URL 时跳过"""
    if download.url is None:
        return
    if download.archive.is_file():
        print(f"[download] {label} 已存在: {download.archive}")
        return
    if not download.netrc.is_file():
        raise pytest.UsageError(f"netrc 文件不存在: {download.netrc}")
    print(f"[download] 正在下载 {label}: {download.archive.name}")

    try:
        authorization = _download_authorization(download)
        download.archive.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{download.archive.name}-",
            dir=download.archive.parent,
        ) as temporary_dir:
            temporary = Path(temporary_dir) / "download.part"
            command = [
                "curl",
                "--fail",
                "--location",
                "--show-error",
                "--progress-bar",
                # 从标准输入读取请求头, 避免令牌出现在进程参数或异常信息中
                "--header",
                "@-",
                "--output",
                str(temporary),
                download.url,
            ]
            try:
                # input=authorization: 把认证头送给 curl 的标准输入
                # curl 的 --header @-: 从标准输入读取内容，作为 HTTP 请求头
                # 效果: python ... | curl --header @- ...
                subprocess.run(command, check=True, input=authorization, text=True)
            except FileNotFoundError as error:
                raise pytest.UsageError("未找到 curl 命令") from error
            temporary.replace(download.archive)
    except (OSError, subprocess.CalledProcessError) as error:
        raise pytest.UsageError(f"下载 {label} 失败: {download.url}") from error


def _extract_archive(label, archive, install_dir, *, wrapper_dir=None, exclude_dirs=()):
    """在临时目录内安全解压，成功后再移动到最终安装目录"""
    print(f"[install] 正在安装 {label}: {install_dir}")
    try:
        install_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{install_dir.name}-",
            dir=install_dir.parent,
        ) as temporary_dir:
            content = Path(temporary_dir) / "content"
            content.mkdir()
            source = archive_tool.extract(
                archive,
                content,
                wrapper_dir=wrapper_dir,
                exclude_dirs=exclude_dirs,
            )

            if not any(source.iterdir()):
                raise pytest.UsageError(f"{label} 压缩包没有可安装的内容: {archive}")

            # 同一文件系统内原子移动; 不会覆盖已有的非空安装目录
            source.rename(install_dir)
    except (OSError, archive_tool.ArchiveError) as error:
        raise pytest.UsageError(f"解压 {label} 失败: {archive}: {error}") from error


def _ensure_executable_permissions(targets: Sequence[tuple[Path, Sequence[str]]]) -> None:
    """为给定的 (安装目录, 相对路径) 列表递归增加所有者执行权限，保留其他权限"""

    def add_execute_permission(path, install_root):
        # 外部目录链接直接报错; 文件链接及安装目录内的链接不跟随
        if path.is_symlink():
            if path.is_dir() and not path.resolve().is_relative_to(install_root):
                raise pytest.UsageError(f"执行权限路径越出工具链安装目录: {path}")
            return
        if not path.resolve().is_relative_to(install_root):
            raise pytest.UsageError(f"执行权限路径越出工具链安装目录: {path}")
        mode = path.stat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            return
        if not mode & stat.S_IXUSR:
            path.chmod(stat.S_IMODE(mode) | stat.S_IXUSR)
        if stat.S_ISDIR(mode):
            # 先给目录补充进入权限, 再遍历子项
            for child in path.iterdir():
                add_execute_permission(child, install_root)

    for install_dir, relative_paths in targets:
        for relative in relative_paths:
            path = install_dir / relative
            try:
                if not path.exists():
                    continue
                add_execute_permission(path, install_dir.resolve())
                print(f"[install] 已补充执行权限: {path}")
            except OSError as error:
                raise pytest.UsageError(f"设置工具链执行权限失败: {path}: {error}") from error


def _sync_sdk_wheels(sdk: SdkConfig) -> None:
    """将当前 Python 环境中的包版本对齐所选 SDK, 同版本跳过, 支持升级和降级"""
    wheel_dir = sdk.install_dir / "Host/dist"
    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        raise pytest.UsageError(f"SDK wheel 目录不存在或没有 whl 文件: {wheel_dir}")

    expected_versions = {}
    pending = []
    try:
        for wheel in wheels:
            # wheel 文件名包含发行包名和版本; 查询安装元数据不会导入旧版模块
            name, target_version, _, _ = parse_wheel_filename(wheel.name)
            if name in expected_versions:
                raise pytest.UsageError(f"SDK wheel 目录中有多个同名包: {name}")
            expected_versions[name] = target_version
            try:
                installed_version = metadata.version(name)
            except metadata.PackageNotFoundError:
                installed_version = None

            if installed_version is not None and Version(installed_version) == target_version:
                continue
            print(f"[wheel] {name}: {installed_version or '未安装'} -> {target_version}")
            pending.append(str(wheel))

        if not pending:
            print("[wheel] SDK Python 包版本已匹配, 跳过安装")
            return

        print(f"[wheel] 正在同步 Python 环境: {sys.executable}")
        # 只安装 SDK 自带的 wheel, 不访问索引或修改其他依赖
        # --no-user 确保写入当前环境, 无权限时直接报错, 不回退到用户级目录
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--no-user",
                "--disable-pip-version-check",
                "--no-input",
                *pending,
            ],
            check=True,
        )
        # 安装发生在测试模块导入前; 刷新查找缓存, 随后校验当前环境可见的版本
        invalidate_caches()
        for name, target_version in expected_versions.items():
            installed_version = metadata.version(name)
            if Version(installed_version) != target_version:
                raise pytest.UsageError(
                    f"SDK wheel 同步后版本不匹配: {name}, 期望 {target_version}, 实际 {installed_version}"
                )
    except (
        OSError,
        subprocess.CalledProcessError,
        InvalidWheelFilename,
        InvalidVersion,
        metadata.PackageNotFoundError,
    ) as error:
        raise pytest.UsageError(f"同步 SDK wheel 失败 ({sys.executable}): {error}") from error


class SetEnv(ABC):
    """
    测试环境部署抽象类

    prepare_packages 统一完成下载解压和权限修复;
    build_environment 返回待应用的环境变量字典；

    子类通过覆盖 packages、executable_targets、sync_wheels、add_import_paths
    和 chip_environment 描述芯片特有的增量；没有相应需求时保留默认实现即可。
    """

    _implementations: ClassVar[dict[str, type[SetEnv]]] = {}

    def __init_subclass__(cls, *, chip: str | None = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if chip is None:
            return

        key = chip.casefold()
        if not key or key in cls._implementations:
            raise TypeError(f"环境部署芯片注册无效或重复: {chip!r}")
        cls._implementations[key] = cls

    def __init__(self, config: ProjectConfig):
        self.config = config

    @classmethod
    def for_config(cls, config: ProjectConfig) -> SetEnv:
        """根据 config.sdk.chip 创建已注册的芯片部署器"""
        chip = config.sdk.chip
        implementation = cls._implementations.get(chip.casefold())
        if implementation is None:
            supported = ", ".join(sorted(cls._implementations)) or "无"
            raise SetEnvError(f"芯片 {chip!r} 没有环境部署实现; 已支持: {supported}")
        return implementation(config)

    # ------------------------------------------------------------------
    # 下载、安装、权限: pytest_sessionstart 中按需调用一次 (本机 xdist 仅主进程)
    # ------------------------------------------------------------------

    def packages(self) -> tuple[tuple[str, object], ...]:
        """返回需要下载解压的 (标签, 包配置) 列表；子类可覆盖以增减包"""
        return (("SDK", self.config.sdk),)

    def executable_targets(self) -> tuple[tuple[Path, tuple[str, ...]], ...]:
        """返回需要补充所有者执行权限的 (安装目录, 相对路径) 列表；默认不需要"""
        return ()

    def prepare_packages(self) -> None:
        """下载、安装 packages() 中的每个包, 复用已有安装和缓存压缩包"""
        for label, package in self.packages():
            install_dir = package.install_dir
            if install_dir.is_dir() and any(install_dir.iterdir()):
                print(f"[install] {label} 已安装: {install_dir}")
                continue
            if install_dir.is_symlink() or (install_dir.exists() and not install_dir.is_dir()):
                raise pytest.UsageError(f"安装路径不是可用目录: {install_dir}")

            download = package.download

            # 1. 用户没有配置下载地址和 archive, 只能期望用户已安装
            # 2. 用户没有配置下载地址，但配置了 archive 文件，尝试从 archive 解压
            if download.url is None and not download.archive.is_file():
                print(f"[install] {label} 使用本地安装, 未配置下载地址, 跳过自动安装")
                continue
            _download_archive(label, download)
            _extract_archive(
                label,
                download.archive,
                install_dir,
                wrapper_dir=package.extract.wrapper_dir,
                exclude_dirs=package.extract.exclude_dirs,
            )
        # 解压完成后补齐执行权限; 复用已有安装时也执行, 不需要重新下载
        self.ensure_executable()

    def ensure_executable(self) -> None:
        """为 executable_targets() 中的目录补充所有者执行权限"""
        _ensure_executable_permissions(self.executable_targets())

    def sync_wheels(self) -> None:
        """将当前 Python 环境对齐已安装的 SDK; 默认不需要同步 wheel"""

    def add_import_paths(self) -> None:
        """将 SDK 的 Python 模块目录加入导入路径；默认不需要调整"""

    # ------------------------------------------------------------------
    # 环境变量: 每个测试进程 (包括每个 xdist worker) 的 session fixture 中部署
    # ------------------------------------------------------------------

    def common_environment(self) -> Mapping[str, str]:
        """所有芯片共享的 SDK 身份和安装路径"""
        sdk = self.config.sdk
        return {
            "BW_SDK": sdk.name,
            "BW_CHIP": sdk.chip,
            "BW_SDK_VERSION": sdk.version,
            "BW_SDK_HOME": str(sdk.install_dir),
        }

    @abstractmethod
    def chip_environment(self) -> Mapping[str, str]:
        """返回当前芯片特有的环境变量"""

    def build_environment(self) -> dict[str, str]:
        """合并并校验本次需要部署的完整环境"""
        environment = dict(self.common_environment())
        environment.update(self.chip_environment())
        for name, value in environment.items():
            if not isinstance(name, str) or not name or not isinstance(value, str):
                raise SetEnvError(f"环境变量名称和值必须分别是非空字符串和字符串: {name!r}={value!r}")
        return environment

    @staticmethod
    def prepend_paths(name: str, *paths: str | Path) -> str:
        """生成去空、去重且保留现有内容的路径型环境变量"""
        current_paths = os.environ.get(name, "").split(os.pathsep)
        candidates = [*(str(path) for path in paths), *current_paths]
        return os.pathsep.join(dict.fromkeys(path for path in candidates if path))


class _BwSetEnv(SetEnv):
    """当前 BW01 系列共用的下载、工具链和环境变量部署流程"""

    def chip_environment(self) -> Mapping[str, str]:
        configured_home = os.environ.get("CIM_TC_HOME")
        toolchain_home = (
            Path(configured_home).expanduser() if configured_home else self.config.toolchain.compiler_tools.install_dir
        )
        toolchain_settings = os.environ.get(
            "CIM_TC_SETTINGS",
            str(toolchain_home / "config" / "cim_tc_settings.json"),
        )
        return {
            "CIM_TC_HOME": str(toolchain_home),
            "CIM_TC_SETTINGS": toolchain_settings,
            "LOGIC_TO_PHYSICS_WITH_REDUNDANCY": os.environ.get("LOGIC_TO_PHYSICS_WITH_REDUNDANCY", "1"),
            "LD_LIBRARY_PATH": self.prepend_paths(
                "LD_LIBRARY_PATH",
                Path(sys.prefix) / "lib",
                toolchain_home / "lib",
            ),
        }

    def packages(self) -> tuple[tuple[str, object], ...]:
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
        _sync_sdk_wheels(self.config.sdk)

    def add_import_paths(self) -> None:
        """将选中 SDK 的 Python 模块放在导入路径最前面"""
        scripts = self.config.sdk.install_dir / "Host" / "scripts"
        paths = [str(scripts / name) for name in ("proto_gen", "src_py")]
        sys.path[:] = [path for path in sys.path if path not in paths]
        sys.path[:0] = paths


class Bw01dSetEnv(
    _BwSetEnv,
    chip="bw01d",
):
    """BW01D 环境部署实现。"""


class Bw01SetEnv(
    _BwSetEnv,
    chip="bw01",
):
    """BW01 环境部署实现；当前与 BW01D 共用部署流程。"""
