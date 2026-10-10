import os
import sys
from abc import ABC
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils.set_env import (
    Bw01SetEnv,
    Bw01dSetEnv,
    SetEnv,
    SetEnvError,
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
