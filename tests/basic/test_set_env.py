import io
import os
import stat
import sys
import tarfile
from abc import ABC
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils.set_env import (
    Bw01dSetEnv,
    Bw01SetEnv,
    SetEnv,
    SetEnvError,
    _ensure_executable_permissions,
    _extract_archive,
)


def make_config(tmp_path, chip="bw01d"):
    sdk = SimpleNamespace(
        name=f"{chip}_v2.3.3",
        chip=chip,
        version="2.3.3",
        install_dir=tmp_path / "SDK" / f"{chip}_2.3.3",
    )
    compiler_tools = SimpleNamespace(install_dir=tmp_path / "toolchain/compiler")
    toolchain = SimpleNamespace(compiler_tools=compiler_tools)
    return SimpleNamespace(sdk=sdk, toolchain=toolchain)


def test_abstract_set_env_requires_chip_environment(tmp_path):
    assert issubclass(SetEnv, ABC)
    with pytest.raises(TypeError):
        SetEnv(make_config(tmp_path))


@pytest.mark.parametrize(
    "chip,expected_type",
    [("bw01", Bw01SetEnv), ("bw01d", Bw01dSetEnv)],
)
def test_selects_implementation_by_config_chip(tmp_path, chip, expected_type):
    deployer = SetEnv.for_config(make_config(tmp_path, chip))

    assert isinstance(deployer, expected_type)


def test_rejects_chip_without_implementation(tmp_path):
    with pytest.raises(SetEnvError, match="new_chip.*没有环境部署实现"):
        SetEnv.for_config(make_config(tmp_path, "new_chip"))


def test_build_environment_returns_complete_environment(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    toolchain_home = config.toolchain.compiler_tools.install_dir
    monkeypatch.delenv("CIM_TC_HOME", raising=False)
    monkeypatch.delenv("CIM_TC_SETTINGS", raising=False)
    monkeypatch.delenv("LOGIC_TO_PHYSICS_WITH_REDUNDANCY", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing:/existing")
    deployer = SetEnv.for_config(config)

    environment = deployer.build_environment()

    assert environment["BW_SDK"] == "bw01d_v2.3.3"
    assert environment["BW_CHIP"] == "bw01d"
    assert environment["BW_SDK_VERSION"] == "2.3.3"
    assert environment["BW_SDK_HOME"] == str(config.sdk.install_dir)
    assert environment["CIM_TC_HOME"] == str(toolchain_home)
    assert environment["CIM_TC_SETTINGS"] == str(toolchain_home / "config/cim_tc_settings.json")
    assert environment["LOGIC_TO_PHYSICS_WITH_REDUNDANCY"] == "1"
    assert environment["LD_LIBRARY_PATH"].split(os.pathsep) == [
        str(Path(sys.prefix) / "lib"),
        str(toolchain_home / "lib"),
        "/existing",
    ]


def test_explicit_toolchain_environment_is_preserved(tmp_path, monkeypatch):
    custom_home = tmp_path / "custom-toolchain"
    monkeypatch.setenv("CIM_TC_HOME", str(custom_home))
    monkeypatch.setenv("CIM_TC_SETTINGS", "/custom/settings.json")

    environment = SetEnv.for_config(make_config(tmp_path)).build_environment()

    assert environment["CIM_TC_HOME"] == str(custom_home)
    assert environment["CIM_TC_SETTINGS"] == "/custom/settings.json"
    assert str(custom_home / "lib") in environment["LD_LIBRARY_PATH"].split(os.pathsep)


def test_add_import_paths_dedupes_and_orders_sdk_script_dirs(tmp_path, monkeypatch):
    sdk = SimpleNamespace(install_dir=tmp_path / "SDK")
    scripts = sdk.install_dir / "Host" / "scripts"
    proto_gen, src_py = str(scripts / "proto_gen"), str(scripts / "src_py")
    monkeypatch.setattr(sys, "path", ["existing", src_py, "other", proto_gen, src_py])
    deployer = Bw01dSetEnv(SimpleNamespace(sdk=sdk))

    deployer.add_import_paths()

    assert sys.path == [proto_gen, src_py, "existing", "other"]

    # 重复配置时保持顺序, 不增加相同目录的条目
    deployer.add_import_paths()
    assert sys.path == [proto_gen, src_py, "existing", "other"]


@pytest.fixture
def extraction_package(tmp_path):
    archive = tmp_path / "SDK_v1.tar.gz"
    install_dir = tmp_path / "installed"

    def build(files, *, links=None):
        with tarfile.open(archive, "w:gz") as output:
            for name in files:
                member = tarfile.TarInfo(name)
                member.size = 2
                output.addfile(member, io.BytesIO(b"ok"))
            for name, target in (links or {}).items():
                member = tarfile.TarInfo(name)
                member.type = tarfile.SYMTYPE
                member.linkname = target
                output.addfile(member)
        return archive, install_dir

    return build


@pytest.mark.parametrize("wrapper", ["", "SDK/", "SDK_v1/", "installed/"])
@pytest.mark.parametrize("exclude_dirs", [(), ("Host/cache",)])
def test_extraction_layout_and_configured_exclusions(extraction_package, wrapper, exclude_dirs):
    archive, install_dir = extraction_package(
        [
            f"./{wrapper}Host/dist/module.whl",
            f"./{wrapper}Host/cache/data",
            f"./{wrapper}Other/Host/cache/data",
        ]
    )

    _extract_archive(
        "SDK",
        archive,
        install_dir,
        wrapper_dir=wrapper.rstrip("/") or None,
        exclude_dirs=exclude_dirs,
    )

    assert (install_dir / "Host/dist/module.whl").read_bytes() == b"ok"
    assert (install_dir / "Host/cache/data").exists() == (not exclude_dirs)
    assert (install_dir / "Other/Host/cache/data").is_file()
    if wrapper:
        assert not (install_dir / wrapper).exists()


@pytest.mark.parametrize("target", ["/usr/bin/python3.12", "../../outside"])
def test_python_rejects_symlink_outside_wrapper(extraction_package, target):
    archive, install_dir = extraction_package(
        ["Host/dist/module.whl"],
        links={"Host/bin/python": target},
    )

    with pytest.raises(pytest.UsageError, match="解压 SDK 失败"):
        _extract_archive(
            "SDK",
            archive,
            install_dir,
            wrapper_dir="Host",
        )
    assert not install_dir.exists()


@pytest.fixture
def toolchain(tmp_path):
    return SimpleNamespace(
        compiler_tools=SimpleNamespace(install_dir=tmp_path / "compiler_tools"),
        map_tools=SimpleNamespace(install_dir=tmp_path / "map_tools"),
    )


def targets_of(toolchain):
    """按 _BwSetEnv.executable_targets() 的规则, 把 toolchain 转成权限目标列表。"""
    return (
        (toolchain.compiler_tools.install_dir, ("bin", "script")),
        (toolchain.map_tools.install_dir, ("map_tool", "script")),
    )


def make_file(path, mode=0o640):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test")
    path.chmod(mode)
    return path


@pytest.mark.parametrize("already_installed", [False, True])
def test_package_preparation_adds_execution_permissions(tmp_path, already_installed):
    def package(name, files):
        archive = tmp_path / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz") as output:
            for name_in_archive in files:
                member = tarfile.TarInfo(name_in_archive)
                member.mode = 0o640
                member.size = 2
                output.addfile(member, io.BytesIO(b"ok"))
        result = SimpleNamespace(
            install_dir=tmp_path / "installed" / name,
            prefix=name,
            download=SimpleNamespace(url=None, archive=archive),
            extract=SimpleNamespace(wrapper_dir=None, exclude_dirs=()),
        )
        if already_installed:
            for name_in_archive in files:
                make_file(result.install_dir / name_in_archive)
        return result

    sdk = package("SDK", ["Host/dist/keep.whl", "Agent/keep"])
    compiler = package("compiler_tools", ["bin/compiler", "script/nested/run.py", "lib/keep.so"])
    mapper = package("map_tools", ["map_tool", "script/algorithm_script/nested/run.py", "script/keep.py"])
    config = SimpleNamespace(
        sdk=sdk,
        toolchain=SimpleNamespace(compiler_tools=compiler, map_tools=mapper),
    )

    Bw01dSetEnv(config).prepare_packages()

    for path in (
        compiler.install_dir / "bin/compiler",
        compiler.install_dir / "script/nested/run.py",
        mapper.install_dir / "map_tool",
        mapper.install_dir / "script/algorithm_script/nested/run.py",
        mapper.install_dir / "script/keep.py",
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o740
    for path in (
        sdk.install_dir / "Host/dist/keep.whl",
        compiler.install_dir / "lib/keep.so",
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_directories_are_fixed_before_traversal_and_repeated_calls_are_safe(toolchain):
    directory = toolchain.compiler_tools.install_dir / "bin"
    program = make_file(directory / "nested/program")
    existing = make_file(directory / "already_executable", mode=0o751)
    nested = program.parent
    nested.chmod(0o640)
    directory.chmod(0o640)

    _ensure_executable_permissions(targets_of(toolchain))
    _ensure_executable_permissions(targets_of(toolchain))

    for path in (directory, nested, program):
        assert stat.S_IMODE(path.stat().st_mode) == 0o740
    assert stat.S_IMODE(existing.stat().st_mode) == 0o751


def test_symlinks_do_not_change_external_files(toolchain, tmp_path):
    external = make_file(tmp_path / "external/program")
    directory = toolchain.compiler_tools.install_dir / "bin"
    directory.mkdir(parents=True)
    (directory / "program_link").symlink_to(external)
    (directory / "broken_link").symlink_to(tmp_path / "missing")
    map_dir = toolchain.map_tools.install_dir
    map_dir.mkdir()
    (map_dir / "map_tool").symlink_to(external)

    _ensure_executable_permissions(targets_of(toolchain))

    assert stat.S_IMODE(external.stat().st_mode) == 0o640
    assert (directory / "program_link").is_symlink()


@pytest.mark.parametrize("relative_link", ["script", "script/algorithm_script"])
def test_symlinked_external_script_directory_is_rejected(toolchain, tmp_path, relative_link):
    external = make_file(tmp_path / "external/run.py")
    map_dir = toolchain.map_tools.install_dir
    link = map_dir / relative_link
    link.parent.mkdir(parents=True)
    link.symlink_to(external.parent, target_is_directory=True)
    with pytest.raises(pytest.UsageError, match="越出工具链安装目录") as error:
        _ensure_executable_permissions(targets_of(toolchain))

    assert str(link) in str(error.value)
    assert link.is_symlink()
    assert stat.S_IMODE(external.stat().st_mode) == 0o640


def test_missing_optional_paths_are_skipped(toolchain):
    _ensure_executable_permissions(targets_of(toolchain))
    assert not toolchain.compiler_tools.install_dir.exists()
    assert not toolchain.map_tools.install_dir.exists()


def test_chmod_failure_has_path_context(toolchain, monkeypatch):
    program = make_file(toolchain.compiler_tools.install_dir / "bin/program")

    def denied(path, mode):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "chmod", denied)
    with pytest.raises(pytest.UsageError, match="设置工具链执行权限失败") as error:
        _ensure_executable_permissions(targets_of(toolchain))
    assert str(program.parent) in str(error.value)
