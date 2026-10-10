import os
import re
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from cim_config import ExtractionConfig, ProjectConfigError, load_project_config
from cim_constants import PROJECT_CONFIG_KEY, PROJECT_ROOT


TOOLCHAIN_URL = "https://git.witmem.ad/api/v4/projects/project/bw01/bw01/bw01d/bw01d_cim_toolchain"
TOOLCHAIN_API_URL = "https://git.witmem.ad/api/v4/projects/project%2Fbw01%2Fbw01%2Fbw01d%2Fbw01d_cim_toolchain"
SDK_URL = "https://nexus.witmem.ad/repository/bw01d-beta"


@pytest.fixture
def config_data():
    return {
        "active_sdk": "bw01d_v2.3.3",
        "netrc": ".netrc",
        "download_dir": "third_party/downloads",
        "sdk_install_root": "third_party/SDK",
        "toolchain_install_root": "third_party/toolchain",
        "toolchains": {
            "v1.6.17.2": {
                "version": "1.6.17.2",
                "map_tools": {
                    "prefix": "bw01d_cim_map_tools_ubuntu2404",
                    "suffix": "",
                },
                "compiler_tools": {
                    "prefix": "bw01d_cim_compiler_tools_ubuntu2404",
                    "suffix": "",
                },
                "url": f"{TOOLCHAIN_URL}/",
                "auth": "bearer",
            },
            "v1.6.18.1_tmp": {
                "version": "1.6.18.1",
                "map_tools": {
                    "prefix": "bw01d_cim_map_tools_ubuntu2404",
                    "suffix": "_map",
                },
                "compiler_tools": {
                    "prefix": "bw01d_cim_compiler_tools_ubuntu2404",
                    "suffix": "_compiler",
                },
                "url": TOOLCHAIN_URL,
                "auth": "bearer",
            },
        },
        "sdks": {
            "bw01d_v2.3.3": {
                "chip": "bw01d",
                "version": "2.3.3",
                "sdk_prefix": "BW01D_SDK",
                "url": f"{SDK_URL}/",
                "auth": "basic",
                "toolchain": "v1.6.17.2",
            },
            # bw01 SDK 可以复用同一套 bw01d 工具链
            "bw01_v2.3.3": {
                "chip": "bw01",
                "version": "2.3.3",
                "sdk_prefix": "BW01_SDK",
                "url": None,
                "toolchain": "v1.6.17.2",
            },
            "bw01d_v2.4.0": {
                "chip": "bw01d",
                "version": "2.4.0",
                "sdk_prefix": "BW01D_SDK",
                "url": SDK_URL,
                "toolchain": "v1.6.18.1_tmp",
            },
        },
    }


@pytest.fixture
def selected_sdk(config_data):
    return config_data["sdks"]["bw01d_v2.3.3"]


@pytest.fixture
def selected_toolchain(config_data):
    return config_data["toolchains"]["v1.6.17.2"]


@pytest.fixture
def config_file(tmp_path, config_data, monkeypatch):
    monkeypatch.delenv("BW_SDK", raising=False)

    def write_config():
        path = tmp_path / "project.yaml"
        path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
        return path

    return write_config


def test_loads_active_sdk_and_builds_downloads(config_file):
    config = load_project_config(config_file())
    sdk = config.sdk
    toolchain = config.toolchain

    assert sdk.name == "bw01d_v2.3.3"
    assert sdk.chip == "bw01d"
    assert sdk.version == "2.3.3"
    assert sdk.prefix == "BW01D_SDK"
    assert sdk.toolchain_name == toolchain.name == "v1.6.17.2"
    assert sdk.install_dir == PROJECT_ROOT / "third_party/SDK/bw01d_2.3.3"
    assert sdk.download.url == (f"{SDK_URL}/BW01D_SDK_v2.3.3/BW01D_SDK_v2.3.3.tar.gz")
    assert sdk.download.archive == (PROJECT_ROOT / "third_party/downloads/BW01D_SDK_v2.3.3.tar.gz")
    assert sdk.download.netrc == PROJECT_ROOT / ".netrc"
    assert sdk.download.auth == "basic"
    assert sdk.extract == ExtractionConfig()

    assert toolchain.version == "1.6.17.2"
    expected = {
        "map_tools": "bw01d_cim_map_tools_ubuntu2404",
        "compiler_tools": "bw01d_cim_compiler_tools_ubuntu2404",
    }
    for tool, prefix in expected.items():
        package = getattr(toolchain, tool)
        filename = f"{prefix}_v1.6.17.2.tar.gz"
        assert package.prefix == prefix
        assert package.suffix == ""
        assert package.release_tag == "v1.6.17.2"
        assert package.release_url == f"{TOOLCHAIN_API_URL}/releases/v1.6.17.2"
        assert package.install_dir == (PROJECT_ROOT / "third_party/toolchain/v1.6.17.2" / tool)
        assert package.download.url == (f"{package.release_url}/downloads/{filename}")
        assert package.download.archive == (PROJECT_ROOT / "third_party/downloads" / filename)
        assert package.download.netrc == sdk.download.netrc
        assert package.download.auth == "bearer"
        assert package.extract == ExtractionConfig()


def test_extraction_settings_are_independent_for_sdk_and_tool_packages(config_file, selected_sdk, selected_toolchain):
    selected_sdk["extract"] = {
        "wrapper_dir": "SDK",
        "exclude_dirs": ["./Host/cache/"],
    }
    selected_toolchain["map_tools"]["extract"] = {
        "wrapper_dir": "maps",
        "exclude_dirs": ["examples"],
    }
    selected_toolchain["compiler_tools"]["extract"] = {
        "wrapper_dir": "compiler",
    }

    config = load_project_config(config_file())

    assert config.sdk.extract == ExtractionConfig("SDK", ("Host/cache",))
    assert config.toolchain.map_tools.extract == ExtractionConfig(
        "maps",
        ("examples",),
    )
    assert config.toolchain.compiler_tools.extract == ExtractionConfig(
        "compiler",
    )


@pytest.mark.parametrize(
    "section,name,package",
    [
        ("sdks", "bw01d_v2.3.3", None),
        ("toolchains", "v1.6.17.2", "map_tools"),
        ("toolchains", "v1.6.17.2", "compiler_tools"),
    ],
)
@pytest.mark.parametrize(
    "extract",
    [
        None,
        {"method": "tar"},
        {"method": "unzip"},
        {"method": None},
        {"wrapper_dir": ""},
        {"wrapper_dir": "."},
        {"wrapper_dir": "/SDK"},
        {"wrapper_dir": "Root/SDK"},
        {"wrapper_dir": ".."},
        {"wrapper_dir": "SDK*"},
        {"wrapper_dir": 123},
        {"exclude_dirs": "Host/cache"},
        {"exclude_dirs": None},
        {"exclude_dirs": [None]},
        {"exclude_dirs": [""]},
        {"exclude_dirs": ["."]},
        {"exclude_dirs": ["/tmp/cache"]},
        {"exclude_dirs": ["Host/../cache"]},
        {"exclude_dirs": ["Host/*"]},
    ],
)
def test_rejects_invalid_extraction_settings(
    config_file,
    config_data,
    section,
    name,
    package,
    extract,
):
    selected = config_data[section][name]
    if package is not None:
        selected = selected[package]
    selected["extract"] = extract

    field = ".".join(part for part in (section, name, package, "extract") if part)
    with pytest.raises(ProjectConfigError, match=re.escape(field)):
        load_project_config(config_file())


def test_sdk_selection_priority(config_file, monkeypatch):
    path = config_file()

    monkeypatch.setenv("BW_SDK", "bw01_v2.3.3")
    assert load_project_config(path).sdk.name == "bw01_v2.3.3"
    assert load_project_config(path, sdk_name="bw01d_v2.4.0").sdk.name == "bw01d_v2.4.0"

    monkeypatch.delenv("BW_SDK")
    assert load_project_config(path).sdk.name == "bw01d_v2.3.3"


def test_bw01_sdk_reuses_bw01d_toolchain(config_file):
    path = config_file()
    bw01d = load_project_config(path, sdk_name="bw01d_v2.3.3")
    bw01 = load_project_config(path, sdk_name="bw01_v2.3.3")

    assert bw01.sdk.chip == "bw01"
    assert bw01.sdk.download.url is None
    assert bw01.sdk.install_dir.name == "bw01_2.3.3"
    assert bw01.sdk.download.archive.name == "BW01_SDK_v2.3.3.tar.gz"
    assert bw01.toolchain == bw01d.toolchain


def test_package_suffixes_independently_change_releases_and_install_roots(config_file):
    config = load_project_config(config_file(), sdk_name="bw01d_v2.4.0")
    toolchain = config.toolchain

    assert config.sdk.download.archive.name == "BW01D_SDK_v2.4.0.tar.gz"
    assert config.sdk.download.auth == "basic"  # 省略 auth 时使用默认认证方式
    for tool, suffix in (("map_tools", "_map"), ("compiler_tools", "_compiler")):
        package = getattr(toolchain, tool)
        release_tag = f"v1.6.18.1{suffix}"
        filename = f"{package.prefix}_v1.6.18.1{suffix}.tar.gz"
        assert package.suffix == suffix
        assert package.release_tag == release_tag
        assert package.release_url == (f"{TOOLCHAIN_API_URL}/releases/{release_tag}")
        assert package.install_dir == (PROJECT_ROOT / "third_party/toolchain" / release_tag / tool)
        assert package.download.archive.name == filename
        assert package.download.url == (f"{package.release_url}/downloads/{filename}")


def test_prefixes_are_used_verbatim(config_file, selected_sdk, selected_toolchain):
    selected_sdk["sdk_prefix"] = "CUSTOM_IMAGE"
    selected_toolchain["map_tools"]["prefix"] = "other_chip_mapper_linux"
    selected_toolchain["compiler_tools"]["prefix"] = "vendor_compiler"

    config = load_project_config(config_file())

    assert config.sdk.download.archive.name == "CUSTOM_IMAGE_v2.3.3.tar.gz"
    for tool, prefix in (
        ("map_tools", "other_chip_mapper_linux"),
        ("compiler_tools", "vendor_compiler"),
    ):
        package = getattr(config.toolchain, tool)
        filename = f"{prefix}_v1.6.17.2.tar.gz"
        assert package.prefix == prefix
        assert package.install_dir == (PROJECT_ROOT / "third_party/toolchain/v1.6.17.2" / tool)
        assert package.download.archive.name == filename
        assert package.download.url.endswith(f"/downloads/{filename}")


@pytest.mark.parametrize("omit_url", [True, False], ids=["omitted", "null"])
def test_sdk_can_be_local_only(config_file, selected_sdk, omit_url):
    if omit_url:
        del selected_sdk["url"]
    else:
        selected_sdk["url"] = None

    config = load_project_config(config_file())

    assert config.sdk.download.url is None
    assert config.sdk.download.archive.name == "BW01D_SDK_v2.3.3.tar.gz"
    assert config.sdk.install_dir == PROJECT_ROOT / "third_party/SDK/bw01d_2.3.3"
    assert config.toolchain.compiler_tools.download.url is not None


@pytest.mark.parametrize("omit_url", [True, False], ids=["omitted", "null"])
def test_toolchain_can_be_local_only(config_file, selected_toolchain, omit_url):
    if omit_url:
        del selected_toolchain["url"]
    else:
        selected_toolchain["url"] = None

    config = load_project_config(config_file())

    for tool in ("map_tools", "compiler_tools"):
        package = getattr(config.toolchain, tool)
        assert package.release_url is None
        assert package.download.url is None
        assert package.install_dir == (PROJECT_ROOT / "third_party/toolchain/v1.6.17.2" / tool)
        assert package.download.archive.parent == (PROJECT_ROOT / "third_party/downloads")
    assert config.sdk.download.url is not None


@pytest.mark.parametrize("tool", ["map_tools", "compiler_tools"])
def test_package_suffix_defaults_to_empty(config_file, selected_toolchain, tool):
    del selected_toolchain[tool]["suffix"]

    config = load_project_config(config_file())

    package = getattr(config.toolchain, tool)
    assert package.suffix == ""
    assert package.release_tag == "v1.6.17.2"


def test_identifiers_do_not_determine_package_names(config_file, config_data):
    config_data["sdks"]["sdk_alias"] = config_data["sdks"].pop("bw01d_v2.3.3")
    config_data["toolchains"]["toolchain_alias"] = config_data["toolchains"].pop("v1.6.17.2")
    config_data["sdks"]["sdk_alias"]["toolchain"] = "toolchain_alias"
    config_data["active_sdk"] = "sdk_alias"

    config = load_project_config(config_file())

    assert config.sdk.name == "sdk_alias"
    assert config.sdk.download.archive.name == "BW01D_SDK_v2.3.3.tar.gz"
    assert config.toolchain.name == "toolchain_alias"
    assert config.toolchain.compiler_tools.download.archive.name == (
        "bw01d_cim_compiler_tools_ubuntu2404_v1.6.17.2.tar.gz"
    )


def test_only_selected_entries_are_validated(config_file, config_data):
    config_data["sdks"]["unfinished"] = None
    config_data["toolchains"]["unfinished"] = None

    assert load_project_config(config_file()).sdk.name == "bw01d_v2.3.3"


def test_absolute_roots_and_loading_has_no_side_effects(config_file, config_data, tmp_path):
    fields = (
        "download_dir",
        "sdk_install_root",
        "toolchain_install_root",
        "netrc",
    )
    for field in fields:
        config_data[field] = str(tmp_path / field)
    before = deepcopy(config_data)

    config = load_project_config(config_file())

    assert config_data == before
    assert config.sdk.install_dir.parent == tmp_path / "sdk_install_root"
    assert config.sdk.download.archive.parent == tmp_path / "download_dir"
    assert config.sdk.download.netrc == tmp_path / "netrc"
    for tool in ("map_tools", "compiler_tools"):
        package = getattr(config.toolchain, tool)
        assert package.install_dir.parent.parent == (tmp_path / "toolchain_install_root")
    assert all(not Path(config_data[field]).exists() for field in fields)


def test_global_config_drives_test_environment(request):
    global_config = request.session.stash[PROJECT_CONFIG_KEY]

    assert os.environ["BW_SDK"] == global_config.sdk.name
    assert os.environ["BW_CHIP"] == global_config.sdk.chip
    assert os.environ["BW_SDK_VERSION"] == global_config.sdk.version
    assert os.environ["BW_SDK_HOME"] == str(global_config.sdk.install_dir)
    assert os.environ["CIM_TC_HOME"] == str(global_config.toolchain.compiler_tools.install_dir)
    assert os.environ["CIM_TC_SETTINGS"] == str(
        global_config.toolchain.compiler_tools.install_dir / "config/cim_tc_settings.json"
    )


@pytest.mark.parametrize("value", ["unknown", "", 123])
def test_rejects_invalid_sdk_selection(config_file, value):
    with pytest.raises(ProjectConfigError):
        load_project_config(config_file(), sdk_name=value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("chip", "../outside"),
        ("version", None),
        ("sdk_prefix", "../package"),
        ("toolchain", None),
        ("toolchain", "missing"),
        ("url", "https:///missing-host"),
        ("url", ""),
        ("auth", "unknown"),
        ("auth", None),
    ],
)
def test_rejects_invalid_sdk_configuration(config_file, selected_sdk, field, value):
    selected_sdk[field] = value

    with pytest.raises(
        ProjectConfigError,
        match=re.escape(f"sdks.bw01d_v2.3.3.{field}"),
    ):
        load_project_config(config_file())


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", ""),
        ("version", None),
        ("version", "../invalid"),
        ("url", ""),
        ("url", "https://git.example/repo?token=value"),
        ("auth", "unknown"),
        ("auth", None),
    ],
)
def test_rejects_invalid_toolchain_configuration(config_file, selected_toolchain, field, value):
    selected_toolchain[field] = value

    with pytest.raises(
        ProjectConfigError,
        match=re.escape(f"toolchains.v1.6.17.2.{field}"),
    ):
        load_project_config(config_file())


@pytest.mark.parametrize("tool", ["map_tools", "compiler_tools"])
@pytest.mark.parametrize("value", ["", None, "../outside"])
def test_rejects_invalid_tool_package_prefix(
    config_file,
    selected_toolchain,
    tool,
    value,
):
    selected_toolchain[tool]["prefix"] = value

    with pytest.raises(
        ProjectConfigError,
        match=re.escape(f"toolchains.v1.6.17.2.{tool}.prefix"),
    ):
        load_project_config(config_file())


@pytest.mark.parametrize("tool", ["map_tools", "compiler_tools"])
@pytest.mark.parametrize("value", ["tmp", "_tmp/outside", "../tmp", None, True])
def test_rejects_invalid_tool_package_suffix(
    config_file,
    selected_toolchain,
    tool,
    value,
):
    selected_toolchain[tool]["suffix"] = value

    with pytest.raises(
        ProjectConfigError,
        match=re.escape(f"toolchains.v1.6.17.2.{tool}.suffix"),
    ):
        load_project_config(config_file())


@pytest.mark.parametrize(
    "field,value",
    [
        ("download_dir", ""),
        ("sdk_install_root", None),
        ("toolchain_install_root", ""),
        ("netrc", None),
        ("sdks", None),
        ("toolchains", None),
    ],
)
def test_rejects_invalid_common_configuration(config_file, config_data, field, value):
    config_data[field] = value

    with pytest.raises(ProjectConfigError, match=re.escape(field)):
        load_project_config(config_file())


@pytest.mark.parametrize(
    "section,name",
    [
        ("sdks", "bw01d_v2.3.3"),
        ("toolchains", "v1.6.17.2"),
    ],
)
def test_selected_entry_must_be_mapping(config_file, config_data, section, name):
    config_data[section][name] = None

    with pytest.raises(
        ProjectConfigError,
        match=re.escape(f"{section}.{name} must be a mapping"),
    ):
        load_project_config(config_file())


@pytest.mark.parametrize(
    "key",
    [
        "active_sdk",
        "sdks",
        "toolchains",
        "netrc",
        "download_dir",
        "sdk_install_root",
        "toolchain_install_root",
    ],
)
def test_required_configuration_cannot_be_omitted(config_file, config_data, key):
    del config_data[key]

    with pytest.raises(ProjectConfigError):
        load_project_config(config_file())


@pytest.mark.parametrize("content", ["[", "[]", "null"])
def test_rejects_invalid_yaml(tmp_path, content):
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ProjectConfigError):
        load_project_config(path)


def test_reports_missing_config_file(tmp_path):
    with pytest.raises(ProjectConfigError, match="cannot read config"):
        load_project_config(tmp_path / "missing.yaml")
