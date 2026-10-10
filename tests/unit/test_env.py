import io
import logging
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cimalgo.config import ExtractionConfig
from cimalgo.env import ArchiveError, SetEnv, SetEnvError, compress, extract
from cimalgo.env import set_env as set_env_module
from cimalgo.env.set_env import Bw01dSetEnv, ensure_executable, install_archive


def build_archive(path: Path, files: list[str], *, links: dict[str, str] | None = None) -> Path:
    """以 / 结尾的名字写成目录条目, 其余写成 2 字节文件"""
    with tarfile.open(path, "w:gz") as output:
        for name in files:
            member = tarfile.TarInfo(name.rstrip("/"))
            if name.endswith("/"):
                member.type = tarfile.DIRTYPE
                member.mode = 0o750
                output.addfile(member)
            else:
                member.size = 2
                member.mode = 0o640
                output.addfile(member, io.BytesIO(b"ok"))
        for name, target in (links or {}).items():
            member = tarfile.TarInfo(name)
            member.type = tarfile.SYMTYPE
            member.linkname = target
            output.addfile(member)
    return path


# ---------------------------------------------------------------------------
# archive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wrapper", [None, "SDK"])
@pytest.mark.parametrize("exclude", [(), ("Host/cache",)])
def test_extract_layout_and_exclusions(tmp_path, wrapper, exclude):
    prefix = f"{wrapper}/" if wrapper else ""
    archive = build_archive(
        tmp_path / "a.tar.gz",
        [f"{prefix}Host/dist/module.whl", f"{prefix}Host/cache/data", f"{prefix}Other/Host/cache/data"],
    )
    root = extract(archive, tmp_path / "out", wrapper_dir=wrapper, exclude_dirs=exclude)

    assert root == (tmp_path / "out" / wrapper if wrapper else tmp_path / "out")
    assert (root / "Host/dist/module.whl").read_bytes() == b"ok"
    assert (root / "Host/cache/data").exists() == (not exclude)
    # 排除是锚定的, 不误伤其他位置的同名目录
    assert (root / "Other/Host/cache/data").is_file()


def test_extract_rejects_paths_outside_wrapper(tmp_path):
    archive = build_archive(tmp_path / "a.tar.gz", ["SDK/a", "stray"])
    with pytest.raises(ArchiveError, match="包装目录外"):
        extract(archive, tmp_path / "out", wrapper_dir="SDK")


@pytest.mark.parametrize("name", ["../escape", "/abs/path"])
def test_extract_rejects_unsafe_paths(tmp_path, name):
    archive = build_archive(tmp_path / "a.tar.gz", [name])
    with pytest.raises(ArchiveError):
        extract(archive, tmp_path / "out")


def test_extract_rejects_symlink_escaping_root(tmp_path):
    archive = build_archive(tmp_path / "a.tar.gz", ["Host/a"], links={"Host/python": "/usr/bin/python3"})
    with pytest.raises(ArchiveError):
        extract(archive, tmp_path / "out")


def test_compress_roundtrip(tmp_path):
    source = tmp_path / "src"
    (source / "sub").mkdir(parents=True)
    (source / "sub" / "f").write_text("x")
    archive = compress(tmp_path / "out.tar.gz", source, include_root=False)
    root = extract(archive, tmp_path / "back")
    assert (root / "sub" / "f").read_text() == "x"


def test_compress_rejects_non_directory(tmp_path):
    file = tmp_path / "f"
    file.write_text("x")
    with pytest.raises(ArchiveError):
        compress(tmp_path / "a.tar.gz", file)


# ---------------------------------------------------------------------------
# install / permissions
# ---------------------------------------------------------------------------


def _package(tmp_path: Path, name: str, files: list[str], wrapper: str | None = None) -> SimpleNamespace:
    archive = build_archive(tmp_path / f"{name}.tar.gz", files)
    return SimpleNamespace(
        install_dir=tmp_path / "installed" / name,
        download=SimpleNamespace(url=None, archive=archive),
        extract=ExtractionConfig(wrapper_dir=wrapper),
    )


def test_install_archive_moves_content_atomically(tmp_path):
    package = _package(tmp_path, "SDK", ["Host/dist/keep.whl"])
    install_archive("SDK", package)
    assert (package.install_dir / "Host/dist/keep.whl").is_file()
    assert not list(package.install_dir.parent.glob(".SDK-*"))


def test_install_archive_rejects_empty_package(tmp_path):
    package = _package(tmp_path, "SDK", ["SDK/"], wrapper="SDK")
    with pytest.raises(SetEnvError, match="没有可安装的内容"):
        install_archive("SDK", package)
    assert not package.install_dir.exists()


def test_ensure_executable_only_touches_listed_dirs(tmp_path):
    root = tmp_path / "tc"
    for rel in ("bin/compiler", "script/nested/run.py", "lib/keep.so"):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
        path.chmod(0o640)
    ensure_executable([(root, ("bin", "script"))])
    assert (root / "bin/compiler").stat().st_mode & 0o100
    assert (root / "script/nested/run.py").stat().st_mode & 0o100
    assert not (root / "lib/keep.so").stat().st_mode & 0o100


def test_ensure_executable_rejects_symlink_dir_outside_root(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    (external / "run.py").write_text("x")
    root = tmp_path / "tc"
    root.mkdir()
    (root / "script").symlink_to(external, target_is_directory=True)
    with pytest.raises(SetEnvError, match="越出"):
        ensure_executable([(root, ("script",))])


# ---------------------------------------------------------------------------
# SetEnv
# ---------------------------------------------------------------------------


def _project(tmp_path: Path, chip: str = "bw01d") -> SimpleNamespace:
    sdk = SimpleNamespace(
        name="x", chip=chip, version="1", install_dir=tmp_path / "sdk",
        scripts_dir=tmp_path / "sdk/Host/scripts", wheel_dir=tmp_path / "sdk/Host/dist",
    )  # fmt: skip
    toolchain = SimpleNamespace(
        compiler_tools=SimpleNamespace(install_dir=tmp_path / "tc/compiler"),
        map_tools=SimpleNamespace(install_dir=tmp_path / "tc/map"),
    )
    return SimpleNamespace(sdk=sdk, toolchain=toolchain)


def test_for_config_dispatches_on_chip(tmp_path):
    assert isinstance(SetEnv.for_config(_project(tmp_path, "bw01d")), Bw01dSetEnv)
    assert isinstance(SetEnv.for_config(_project(tmp_path, "BW01D")), Bw01dSetEnv)
    with pytest.raises(SetEnvError, match="没有环境准备实现"):
        SetEnv.for_config(_project(tmp_path, "unknown"))


def test_environment_and_import_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("CIM_TC_HOME", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")
    env = SetEnv.for_config(_project(tmp_path))
    variables = env.environment()

    assert variables["CIM_SDK"] == "x"
    assert variables["CIM_CHIP"] == "bw01d"
    assert variables["CIM_TC_HOME"] == str(tmp_path / "tc/compiler")
    assert variables["LD_LIBRARY_PATH"].endswith("/existing")
    assert str(tmp_path / "tc/compiler/lib") in variables["LD_LIBRARY_PATH"]

    import sys

    original = list(sys.path)
    try:
        sys.path[:] = ["keep", str(tmp_path / "sdk/Host/scripts/src_py")]
        env.apply_import_paths()
        assert sys.path[:2] == [str(tmp_path / "sdk/Host/scripts/proto_gen"), str(tmp_path / "sdk/Host/scripts/src_py")]
        assert sys.path.count(str(tmp_path / "sdk/Host/scripts/src_py")) == 1
    finally:
        sys.path[:] = original


def test_prepare_packages_skips_installed_and_local_only(tmp_path, monkeypatch, caplog):
    project = _project(tmp_path)
    project.sdk.install_dir.mkdir(parents=True)
    (project.sdk.install_dir / "x").write_text("x")
    for package in (project.toolchain.compiler_tools, project.toolchain.map_tools):
        package.download = SimpleNamespace(url=None, archive=tmp_path / "missing.tar.gz")
        package.extract = ExtractionConfig()
    project.sdk.download = SimpleNamespace(url=None, archive=tmp_path / "missing.tar.gz")
    project.sdk.extract = ExtractionConfig()
    download = Mock()
    monkeypatch.setattr(set_env_module, "download_archive", download)

    with caplog.at_level(logging.INFO, logger="cimalgo"):
        SetEnv.for_config(project).prepare_packages()

    download.assert_not_called()
    assert "SDK 已安装" in caplog.text
    assert "map tools 未配置下载地址" in caplog.text
