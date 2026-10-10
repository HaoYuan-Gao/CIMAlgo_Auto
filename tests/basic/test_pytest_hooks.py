import ipaddress
import os
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import conftest as project_hooks
from cim_constants import (
    DEVICE_OPS_KEY,
    LOCKED_DEVICE_KEY,
    PROJECT_CONFIG_KEY,
    SSH_CONNECTOR_KEY,
    TEST_IP_WHITELIST,
)
from utils.device_ops import DeviceTarget
from utils.set_env import SetEnv


@pytest.mark.parametrize(
    "download,worker",
    [
        (False, False),
        (True, False),
        (True, True),
    ],
)
def test_session_start_prepares_packages_only_on_primary_download_run(
    monkeypatch,
    download,
    worker,
):
    project_config = SimpleNamespace()
    load = Mock(return_value=project_config)
    deployer = Mock()
    deployer.build_environment.return_value = {}
    ssh_connector = object()
    connector_factory = Mock(return_value=ssh_connector)
    monkeypatch.setattr(project_hooks, "load_project_config", load)
    monkeypatch.setattr(project_hooks, "SSHConnector", connector_factory)
    monkeypatch.setattr(SetEnv, "for_config", Mock(return_value=deployer))

    config = SimpleNamespace(
        getoption=lambda name: {"--sdk": None, "--download": download}[name],
        add_cleanup=Mock(),
    )
    if worker:
        config.workerinput = {"workerid": "gw0"}
    session = SimpleNamespace(config=config, stash={})

    project_hooks.pytest_sessionstart(session)

    assert session.stash[PROJECT_CONFIG_KEY] is project_config
    assert session.stash[SSH_CONNECTOR_KEY] is ssh_connector
    load.assert_called_once_with(sdk_name=None)
    connector_factory.assert_called_once_with(trusted_ips=TEST_IP_WHITELIST)
    deployer.add_import_paths.assert_called_once()
    deployer.build_environment.assert_called_once()
    config.add_cleanup.assert_called_once()
    config.add_cleanup.call_args.args[0]()
    if download and not worker:
        deployer.prepare_packages.assert_called_once()
        deployer.sync_wheels.assert_called_once()
    else:
        deployer.prepare_packages.assert_not_called()
        deployer.sync_wheels.assert_not_called()


def test_session_config_drives_test_environment(request):
    global_config = request.session.stash[PROJECT_CONFIG_KEY]

    assert os.environ["BW_SDK"] == global_config.sdk.name
    assert os.environ["BW_CHIP"] == global_config.sdk.chip
    assert os.environ["BW_SDK_VERSION"] == global_config.sdk.version
    assert os.environ["BW_SDK_HOME"] == str(global_config.sdk.install_dir)
    assert os.environ["CIM_TC_HOME"] == str(global_config.toolchain.compiler_tools.install_dir)
    assert os.environ["CIM_TC_SETTINGS"] == str(
        global_config.toolchain.compiler_tools.install_dir / "config/cim_tc_settings.json"
    )


def test_session_connector_trusted_ips_are_sourced_from_constants(request):
    ssh_connector = request.session.stash[SSH_CONNECTOR_KEY]

    assert ssh_connector.trusted_networks == {ipaddress.ip_network(value) for value in TEST_IP_WHITELIST}


def test_case_hooks_open_and_close_locked_device(monkeypatch):
    connector = object()
    locked_device = Mock()
    device_ops = Mock()
    open_locked_device = Mock(return_value=locked_device)
    monkeypatch.setattr(project_hooks.LockedDeviceConnection, "open", open_locked_device)
    for_target = Mock(return_value=device_ops)
    monkeypatch.setattr(project_hooks.DeviceOps, "for_target", for_target)

    item = SimpleNamespace(
        callspec=SimpleNamespace(
            params={
                "case_context": {
                    "case": "device-case",
                    "tag": "external",
                    "device": {
                        "type": "zynq",
                        "ip": "10.27.17.58",
                        "port": 2222,
                        "username": "root",
                        "password": "secret",
                        "hard_block": [0, 1],
                        "sel_layer": [0, 1, 2],
                        "tsv_sel0": 3,
                        "tsv_sel1": 254,
                    },
                }
            }
        ),
        session=SimpleNamespace(
            stash={
                SSH_CONNECTOR_KEY: connector,
                PROJECT_CONFIG_KEY: SimpleNamespace(sdk=SimpleNamespace(chip="bw01d", version="2.3.3")),
            }
        ),
        stash=pytest.Stash(),
        config=SimpleNamespace(pluginmanager=SimpleNamespace(get_plugin=Mock(return_value=None))),
    )

    project_hooks.pytest_runtest_setup(item)

    assert item.stash[LOCKED_DEVICE_KEY] is locked_device
    assert item.stash[DEVICE_OPS_KEY] is device_ops
    open_locked_device.assert_called_once_with(
        connector,
        username="root",
        host="10.27.17.58",
        port=2222,
        password="secret",
        case="device-case",
    )
    locked_device.run.assert_not_called()
    for_target.assert_called_once_with(DeviceTarget(ip="10.27.17.58", bmc_ip=None, driver="bw01d"))
    assert device_ops.mock_calls == [
        call.power_cycle(),
        call.switch_firmware_version("2.3.3"),
        call.set_phy_l2([0, 1]),
        call.select_layer(0, tsv_sel0=3, tsv_sel1=254),
        call.select_layer(1, tsv_sel0=3, tsv_sel1=254),
        call.select_layer(2, tsv_sel0=3, tsv_sel1=254),
    ]
    teardown = project_hooks.pytest_runtest_teardown(item, None)
    next(teardown)
    with pytest.raises(StopIteration):
        next(teardown)
    locked_device.close.assert_called_once_with()
    device_ops.close.assert_called_once_with()


@pytest.mark.parametrize("deploy_error", [None, RuntimeError("deployment failed")])
def test_case_hook_updates_sdk_using_locked_connection(monkeypatch, deploy_error):
    connection = object()
    locked_device = Mock(connection=connection)
    if deploy_error is not None:
        locked_device.run.side_effect = deploy_error
    monkeypatch.setattr(project_hooks.LockedDeviceConnection, "open", Mock(return_value=locked_device))
    device_ops = Mock()
    for_target = Mock(return_value=device_ops)
    monkeypatch.setattr(project_hooks.DeviceOps, "for_target", for_target)

    sdk = SimpleNamespace(chip="bw01d", version="2.3.3")
    deployer = Mock()
    deployer.deploy.return_value = object()
    for_sdk = Mock(return_value=deployer)
    monkeypatch.setattr(project_hooks.SDKDeployer, "for_sdk", for_sdk)

    item = SimpleNamespace(
        callspec=SimpleNamespace(
            params={
                "case_context": {
                    "case": "device-case",
                    "tag": "external",
                    "device": {
                        "ip": "192.0.2.1",
                        "port": 22,
                        "username": "root",
                        "update_sdk": True,
                        "hard_block": [0],
                        "sel_layer": [2],
                    },
                }
            }
        ),
        session=SimpleNamespace(stash={SSH_CONNECTOR_KEY: object(), PROJECT_CONFIG_KEY: SimpleNamespace(sdk=sdk)}),
        stash=pytest.Stash(),
        config=SimpleNamespace(pluginmanager=SimpleNamespace(get_plugin=Mock(return_value=None))),
    )

    if deploy_error is None:
        project_hooks.pytest_runtest_setup(item)
        assert item.stash[LOCKED_DEVICE_KEY] is locked_device
        assert item.stash[DEVICE_OPS_KEY] is device_ops
        locked_device.close.assert_not_called()
        for_target.assert_called_once()
    else:
        with pytest.raises(RuntimeError, match="deployment failed"):
            project_hooks.pytest_runtest_setup(item)
        assert LOCKED_DEVICE_KEY not in item.stash
        locked_device.close.assert_called_once_with()
        for_target.assert_not_called()

    for_sdk.assert_called_once_with(sdk)
    deployer.deploy.assert_called_once_with(connection)
    locked_device.run.assert_called_once_with(deployer.deploy.return_value)


def test_case_hook_releases_device_and_lock_when_initialization_fails(monkeypatch):
    locked_device = Mock()
    monkeypatch.setattr(project_hooks.LockedDeviceConnection, "open", Mock(return_value=locked_device))
    device_ops = Mock()
    device_ops.set_phy_l2.side_effect = RuntimeError("physical L2 failed")
    monkeypatch.setattr(project_hooks.DeviceOps, "for_target", Mock(return_value=device_ops))
    item = SimpleNamespace(
        callspec=SimpleNamespace(
            params={
                "case_context": {
                    "case": "device-case",
                    "tag": "external",
                    "device": {
                        "ip": "192.0.2.1",
                        "port": 22,
                        "username": "root",
                        "hard_block": [0],
                        "sel_layer": [1],
                    },
                }
            }
        ),
        session=SimpleNamespace(
            stash={
                SSH_CONNECTOR_KEY: object(),
                PROJECT_CONFIG_KEY: SimpleNamespace(sdk=SimpleNamespace(chip="bw01d", version="2.3.3")),
            }
        ),
        stash=pytest.Stash(),
        config=SimpleNamespace(pluginmanager=SimpleNamespace(get_plugin=Mock(return_value=None))),
    )

    with pytest.raises(RuntimeError, match="physical L2 failed"):
        project_hooks.pytest_runtest_setup(item)

    assert LOCKED_DEVICE_KEY not in item.stash
    assert DEVICE_OPS_KEY not in item.stash
    device_ops.close.assert_called_once_with()
    locked_device.close.assert_called_once_with()
