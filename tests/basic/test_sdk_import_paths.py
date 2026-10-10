import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import conftest as project_hooks
from cim_constants import PROJECT_CONFIG_KEY
from utils.set_env import Bw01dSetEnv, SetEnv


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
    monkeypatch.setattr(project_hooks, "load_project_config", load)
    monkeypatch.setattr(SetEnv, "for_config", Mock(return_value=deployer))

    config = SimpleNamespace(getoption=lambda name: {"--sdk": None, "--download": download}[name])
    if worker:
        config.workerinput = {"workerid": "gw0"}
    session = SimpleNamespace(config=config, stash={})

    project_hooks.pytest_sessionstart(session)

    assert session.stash[PROJECT_CONFIG_KEY] is project_config
    load.assert_called_once_with(sdk_name=None)
    deployer.add_import_paths.assert_called_once()
    if download and not worker:
        deployer.prepare_packages.assert_called_once()
        deployer.sync_wheels.assert_called_once()
    else:
        deployer.prepare_packages.assert_not_called()
        deployer.sync_wheels.assert_not_called()


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
