"""tar.gz 压缩包的创建与解压工具。"""

import subprocess
import tarfile
from pathlib import Path, PurePosixPath

from utils.deprecated import deprecated


class ArchiveError(RuntimeError):
    """压缩包操作失败"""


def extract(archive, destination, *, wrapper_dir=None, exclude_dirs=()):
    """安全解压并返回已去掉可选包装层的内容路径"""
    archive = Path(archive)
    destination = Path(destination)
    if wrapper_dir is not None:
        wrapper_path = PurePosixPath(wrapper_dir)
        if wrapper_path.is_absolute() or len(wrapper_path.parts) != 1 or ".." in wrapper_path.parts:
            raise ArchiveError(f"包装目录无效: {wrapper_dir}")
        wrapper_dir = wrapper_path.parts[0]
    exclude_dirs = tuple(exclude_dirs)
    destination.mkdir(parents=True, exist_ok=True)

    try:
        excluded = tuple(PurePosixPath(path) for path in exclude_dirs)
        skipped = 0

        def filter_member(member, target):
            nonlocal skipped
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ArchiveError(f"{archive} 包含不安全路径: {member.name}")
            if wrapper_dir is not None:
                if relative.parts[:1] != (wrapper_dir,):
                    raise ArchiveError(f"{archive} 包含包装目录外的路径: {member.name}")
                relative = PurePosixPath(*relative.parts[1:])
            if any(relative.is_relative_to(path) for path in excluded):
                skipped += 1
                return None
            return tarfile.data_filter(member, target)

        with tarfile.open(archive, mode="r:gz") as package:
            package.extractall(destination, filter=filter_member)
        if skipped:
            print(f"[install] {archive.name} 跳过目录: {', '.join(exclude_dirs)}")

        source = destination / wrapper_dir if wrapper_dir else destination
        if wrapper_dir and (not source.is_dir() or source.is_symlink()):
            raise ArchiveError(f"包装目录无效: {wrapper_dir}")

        source_root = source.resolve()
        for entry in source.rglob("*"):
            if entry.is_symlink() and not entry.resolve().is_relative_to(source_root):
                raise ArchiveError(f"{archive.name} 包含越出安装目录的链接: {entry.name}")
        return source
    except ArchiveError:
        raise
    except (OSError, tarfile.TarError, EOFError, RuntimeError) as error:
        raise ArchiveError(str(error)) from error


def compress(archive, source, *, include_root=True):
    """将单个目录写入 tar.gz"""
    try:
        archive = Path(archive)
        source = Path(source)
        if not source.is_dir() or source.is_symlink():
            raise ArchiveError(f"压缩源不是普通目录: {source}")

        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, mode="w:gz") as package:
            entries = (source,) if include_root else sorted(source.iterdir())
            for entry in entries:
                package.add(entry, arcname=entry.name)
        return archive
    except ArchiveError:
        raise
    except (OSError, tarfile.TarError, TypeError) as error:
        raise ArchiveError(str(error)) from error


@deprecated(
    replacement="extract",
    reason="tar 指令在 Ubuntu 和 macOS 上行为不一致",
)
def extract_with_tar(archive, destination, wrapper_dir, *, exclude_dirs=()):
    """使用系统 tar 解压；仅为兼容旧调用而保留"""
    command = [
        "tar",
        "-xzf",
        str(archive),
        "-C",
        str(destination),
        "--no-same-owner",
        "--no-same-permissions",
    ]
    try:
        if exclude_dirs:
            # --anchored: 从完整路径开头匹配
            # --no-wildcards: *、?、[...] 不会被解释成通配符, 仅按字面值匹配路径
            command.extend(["--anchored", "--no-wildcards"])
            for path in exclude_dirs:
                relative = (
                    PurePosixPath(wrapper_dir) / path if wrapper_dir is not None else PurePosixPath(path)
                ).as_posix()
                command.extend([f"--exclude={relative}", f"--exclude=./{relative}"])
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise ArchiveError("未找到 tar 命令，请使用 extract()") from error
