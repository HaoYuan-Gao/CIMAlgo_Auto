"""tar.gz 的安全解压与创建; 纯 Python 实现, 行为与平台无关"""

from __future__ import annotations

import tarfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from cimalgo.logging import get_logger

log = get_logger(__name__)


class ArchiveError(RuntimeError):
    """压缩包操作失败"""


def extract(
    archive: str | Path,
    destination: str | Path,
    *,
    wrapper_dir: str | None = None,
    exclude_dirs: Iterable[str] = (),
) -> Path:
    """
    解压到 destination, 返回内容根目录

    wrapper_dir 指定包内最外层目录名, 解压后返回该目录 (调用方通常再把它移到最终位置);
    exclude_dirs 是相对于内容根目录的路径, 命中的条目整体跳过。
    拒绝绝对路径、.. 和指向安装目录之外的符号链接。
    """
    archive = Path(archive)
    destination = Path(destination)
    excluded = tuple(PurePosixPath(path) for path in exclude_dirs)
    if wrapper_dir is not None:
        parts = PurePosixPath(wrapper_dir).parts
        if len(parts) != 1 or parts[0] in ("..", "/"):
            raise ArchiveError(f"包装目录无效: {wrapper_dir}")
    destination.mkdir(parents=True, exist_ok=True)
    skipped = 0

    def filter_member(member: tarfile.TarInfo, target: str) -> tarfile.TarInfo | None:
        nonlocal skipped
        relative = PurePosixPath(member.name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ArchiveError(f"{archive.name} 包含不安全路径: {member.name}")
        if wrapper_dir is not None:
            if relative.parts[:1] != (wrapper_dir,):
                raise ArchiveError(f"{archive.name} 包含包装目录外的路径: {member.name}")
            relative = PurePosixPath(*relative.parts[1:])
        if any(relative.is_relative_to(path) for path in excluded):
            skipped += 1
            return None
        return tarfile.data_filter(member, target)

    try:
        with tarfile.open(archive, mode="r:gz") as package:
            package.extractall(destination, filter=filter_member)
    except (OSError, tarfile.TarError, EOFError) as error:
        raise ArchiveError(f"解压 {archive.name} 失败: {error}") from error
    if skipped:
        log.info("%s 跳过目录 %s (%d 项)", archive.name, ", ".join(exclude_dirs), skipped)

    source = destination / wrapper_dir if wrapper_dir else destination
    if wrapper_dir and (not source.is_dir() or source.is_symlink()):
        raise ArchiveError(f"{archive.name} 中没有包装目录 {wrapper_dir}")
    source_root = source.resolve()
    for entry in source.rglob("*"):
        if entry.is_symlink() and not entry.resolve().is_relative_to(source_root):
            raise ArchiveError(f"{archive.name} 包含越出安装目录的链接: {entry}")
    return source


def compress(archive: str | Path, source: str | Path, *, include_root: bool = True) -> Path:
    """把目录写入 tar.gz; include_root=False 时只打包目录内容"""
    archive = Path(archive)
    source = Path(source)
    if not source.is_dir() or source.is_symlink():
        raise ArchiveError(f"压缩源不是普通目录: {source}")
    try:
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, mode="w:gz") as package:
            entries = (source,) if include_root else sorted(source.iterdir())
            for entry in entries:
                package.add(entry, arcname=entry.name)
    except (OSError, tarfile.TarError) as error:
        raise ArchiveError(f"压缩 {source} 失败: {error}") from error
    return archive
