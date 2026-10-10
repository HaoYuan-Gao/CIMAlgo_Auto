"""request 读取函数只返回对应 stash 中的对象。"""

from types import SimpleNamespace

import pytest

from cim_constants import (
    DEVICE_OPS_KEY,
    LOCKED_DEVICE_KEY,
    PROJECT_CONFIG_KEY,
    SSH_CONNECTOR_KEY,
    get_device_ops,
    get_locked_device,
    get_project_config,
    get_sdk_config,
    get_ssh_connector,
    get_toolchain_config,
)


def test_request_getters_return_the_original_objects():
    sdk = object()
    toolchain = object()
    project = SimpleNamespace(sdk=sdk, toolchain=toolchain)
    connector = object()
    locked_device = object()
    device_ops = object()
    session = SimpleNamespace(stash=pytest.Stash())
    node = SimpleNamespace(stash=pytest.Stash())
    session.stash[PROJECT_CONFIG_KEY] = project
    session.stash[SSH_CONNECTOR_KEY] = connector
    node.stash[LOCKED_DEVICE_KEY] = locked_device
    node.stash[DEVICE_OPS_KEY] = device_ops
    request = SimpleNamespace(session=session, node=node)

    assert get_project_config(request) is project
    assert get_sdk_config(request) is sdk
    assert get_toolchain_config(request) is toolchain
    assert get_ssh_connector(request) is connector
    assert get_locked_device(request) is locked_device
    assert get_device_ops(request) is device_ops
