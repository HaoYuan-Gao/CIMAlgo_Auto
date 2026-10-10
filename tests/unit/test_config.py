import re

import pytest

from cimalgo.config import DeviceConfig, ExtractionConfig, ProjectConfigError, load_project_config

TOOLCHAIN_API_URL = "https://git.witmem.ad/api/v4/projects/project%2Fbw01%2Fbw01%2Fbw01d%2Fbw01d_cim_toolchain"


def test_loads_active_sdk_and_builds_download_urls(config_file):
    path = config_file()
    config = load_project_config(path)
    root = path.parent

    sdk = config.sdk
    assert sdk.name == "bw01d_v2.3.3"
    assert sdk.chip == "bw01d"
    assert sdk.install_dir == root / "third_party/SDK/bw01d_2.3.3"
    assert sdk.download.url == "https://nexus.witmem.ad/repository/bw01d-beta/BW01D_SDK_v2.3.3/BW01D_SDK_v2.3.3.tar.gz"
    assert sdk.download.archive == root / "third_party/downloads/BW01D_SDK_v2.3.3.tar.gz"
    assert config.netrc == root / ".netrc"
    assert sdk.extract == ExtractionConfig()
    assert sdk.scripts_dir == sdk.install_dir / "Host/scripts"
    assert sdk.wheel_dir == sdk.install_dir / "Host/dist"

    toolchain = config.toolchain
    assert toolchain.name == "v1.6.17.2"
    for tool in ("map_tools", "compiler_tools"):
        package = getattr(toolchain, tool)
        assert package.release_tag == "v1.6.17.2"
        assert package.release_url == f"{TOOLCHAIN_API_URL}/releases/v1.6.17.2"
        assert package.install_dir == root / "third_party/toolchain/v1.6.17.2" / tool
        assert package.download.url == f"{package.release_url}/downloads/{package.prefix}_v1.6.17.2.tar.gz"
        assert package.download.auth == "bearer"

    assert config.trusted_ips == ("10.27.17.0/24", "2001:db8::1")
    assert config.project_root == root


def test_devices_are_loaded_with_defaults(config_file):
    config = load_project_config(config_file())

    assert config.device("evb-01") == DeviceConfig(name="evb-01", ip="10.27.17.45", driver="bw01d", evb_port=61006)
    second = config.device("evb-02")
    assert (second.ssh_port, second.ssh_username, second.evb_port, second.bmc_ip) == (2222, "tester", None, "10.0.0.9")
    with pytest.raises(KeyError, match="evb-99"):
        config.device("evb-99")


def test_devices_section_is_optional(config_file, config_data):
    del config_data["devices"]
    assert load_project_config(config_file()).devices == ()


def test_power_control_without_either_entry_is_allowed(config_file, config_data):
    config_data["devices"]["evb-01"] = {"ip": "10.27.17.45", "driver": "bw01d"}
    device = load_project_config(config_file()).device("evb-01")
    assert (device.evb_port, device.bmc_ip) == (None, None)


def test_device_driver_is_required_and_independent_of_sdk_chip(config_file, config_data):
    # bw01 板子用 bw01d 的接口: sdk.chip 与 device.driver 可以不同, 互不推导
    config_data["devices"]["evb-01"]["driver"] = "bw01"
    config = load_project_config(config_file())
    assert config.sdk.chip == "bw01d" and config.device("evb-01").driver == "bw01"

    del config_data["devices"]["evb-01"]["driver"]
    with pytest.raises(ProjectConfigError, match=re.escape("devices.evb-01.driver")):
        load_project_config(config_file())


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"ip": "not-an-ip"}, "devices.evb-01.ip"),
        ({"driver": ""}, "devices.evb-01.driver"),
        ({"driver": "has space"}, "devices.evb-01.driver"),
        ({"ssh_port": 0}, "devices.evb-01.ssh_port"),
        ({"evb_port": "61006"}, "devices.evb-01.evb_port"),
        ({"bmc_ip": "10.0.0.9"}, "devices.evb-01: evb_port and bmc_ip are mutually exclusive"),
        # gRPC 端口是用例参数, 不再属于设备配置
        ({"grpc_port": 60000}, "devices.evb-01.grpc_port"),
        ({"is_cluster": "yes"}, "devices.evb-01.is_cluster"),
        ({"unknown": 1}, "devices.evb-01.unknown"),
    ],
)
def test_rejects_invalid_device(config_file, config_data, patch, message):
    config_data["devices"]["evb-01"].update(patch)
    with pytest.raises(ProjectConfigError, match=re.escape(message)):
        load_project_config(config_file())


def test_sdk_selection_priority(config_file, monkeypatch):
    path = config_file()
    monkeypatch.setenv("CIM_SDK", "bw01_local")
    assert load_project_config(path).sdk.name == "bw01_local"
    assert load_project_config(path, sdk_name="bw01d_v2.3.3").sdk.name == "bw01d_v2.3.3"
    monkeypatch.delenv("CIM_SDK")
    assert load_project_config(path).sdk.name == "bw01d_v2.3.3"


def test_local_sdk_has_no_download_url(config_file):
    config = load_project_config(config_file(), sdk_name="bw01_local")
    assert config.sdk.download.url is None
    assert config.sdk.install_dir.name == "bw01_2.3.3"
    assert config.toolchain.compiler_tools.download.url is not None


def test_toolchain_without_url_is_local(config_file, config_data):
    config_data["toolchains"]["v1.6.17.2"]["url"] = None
    toolchain = load_project_config(config_file()).toolchain
    assert toolchain.map_tools.release_url is None
    assert toolchain.map_tools.download.url is None


def test_suffix_changes_release_tag_and_filename(config_file, config_data):
    config_data["toolchains"]["v1.6.17.2"]["map_tools"]["suffix"] = "_tmp"
    package = load_project_config(config_file()).toolchain.map_tools
    assert package.release_tag == "v1.6.17.2_tmp"
    assert package.download.archive.name == "bw01d_cim_map_tools_ubuntu2404_v1.6.17.2_tmp.tar.gz"
    assert package.install_dir.parent.name == "v1.6.17.2_tmp"


def test_extraction_settings(config_file, config_data):
    config_data["sdks"]["bw01d_v2.3.3"]["extract"] = {"wrapper_dir": "BW01D_SDK", "exclude_dirs": ["./Host/cache/"]}
    assert load_project_config(config_file()).sdk.extract == ExtractionConfig("BW01D_SDK", ("Host/cache",))


@pytest.mark.parametrize(
    "extract",
    [
        {"method": "tar"},
        {"wrapper_dir": "a/b"},
        {"wrapper_dir": ".."},
        {"exclude_dirs": "Host"},
        {"exclude_dirs": ["/abs"]},
        {"exclude_dirs": ["Host/../x"]},
        {"exclude_dirs": ["Host/*"]},
    ],
)
def test_rejects_invalid_extraction(config_file, config_data, extract):
    config_data["sdks"]["bw01d_v2.3.3"]["extract"] = extract
    with pytest.raises(ProjectConfigError, match=r"sdks\.bw01d_v2\.3\.3\.extract"):
        load_project_config(config_file())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chip", "../x"),
        ("version", None),
        ("toolchain", "missing"),
        ("url", "https://host/path?token=x"),
        ("url", "ftp://host/x"),
        ("auth", "digest"),
    ],
)
def test_rejects_invalid_sdk_fields(config_file, config_data, field, value):
    config_data["sdks"]["bw01d_v2.3.3"][field] = value
    with pytest.raises(ProjectConfigError, match=re.escape(f"sdks.bw01d_v2.3.3.{field}")):
        load_project_config(config_file())


def test_toolchain_url_must_be_gitlab_api(config_file, config_data):
    config_data["toolchains"]["v1.6.17.2"]["url"] = "https://git.witmem.ad/group/project"
    with pytest.raises(ProjectConfigError, match="GitLab API project URL"):
        load_project_config(config_file())


@pytest.mark.parametrize("value", ["10.27.17.0/24", ["bad"], ["10.27.17.45/24"], [123]])
def test_rejects_invalid_trusted_ips(config_file, config_data, value):
    config_data["trusted_ips"] = value
    with pytest.raises(ProjectConfigError, match="trusted_ips"):
        load_project_config(config_file())


@pytest.mark.parametrize(
    "key", ["active_sdk", "sdks", "toolchains", "download_dir", "sdk_install_root", "toolchain_install_root"]
)
def test_required_keys(config_file, config_data, key):
    del config_data[key]
    with pytest.raises(ProjectConfigError):
        load_project_config(config_file())


def test_netrc_defaults_to_project_root(config_file, config_data):
    del config_data["netrc"]
    path = config_file()
    assert load_project_config(path).netrc == path.parent / ".netrc"


@pytest.mark.parametrize("content", ["[", "[]", "null"])
def test_rejects_invalid_yaml(tmp_path, content):
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ProjectConfigError):
        load_project_config(path)


def test_missing_file(tmp_path):
    with pytest.raises(ProjectConfigError, match="cannot read config"):
        load_project_config(tmp_path / "missing.yaml")


def test_config_is_immutable(config_file):
    config = load_project_config(config_file())
    with pytest.raises(AttributeError):
        config.sdk.chip = "other"  # type: ignore[misc]
