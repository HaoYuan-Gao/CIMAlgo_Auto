import argparse
import asyncio
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import asyncssh
import pytest

import utils.sdk_deploy as sdk_deploy
from utils.sdk_deploy import SDKDeployer
from utils.ssh_connection import SSHConnector


class FakeConnection:
    """记录远程命令和上传, 模拟 AsyncSSH connection 的最小接口"""

    def __init__(self, outputs=(), peer=("10.27.17.45", 22), username="root", delay=0, failure=None):
        self.outputs = list(outputs)
        self.peer = peer
        self.username = username
        self.delay = delay
        # 第一条远程命令抛出的异常, 用于模拟远程失败
        self.failure = failure
        self.commands = []
        self.uploads = []
        self.closed = False

    def get_extra_info(self, name):
        return self.peer if name == "peername" else None

    async def run(self, command, check=True):
        self.commands.append(command)
        if self.failure is not None:
            raise self.failure
        if self.delay:
            await asyncio.sleep(self.delay)
        return SimpleNamespace(stdout=self.outputs.pop(0), stderr="")

    def start_sftp_client(self):
        connection = self

        class Sftp:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc_info):
                return False

            async def put(self, local, remote):
                connection.uploads.append((local, remote))

        return Sftp()

    def close(self):
        self.closed = True

    async def wait_closed(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        self.close()


@pytest.fixture
def deployer(tmp_path):
    deb = tmp_path / "sdk.deb"
    deb.write_bytes(b"test")
    return SDKDeployer(deb)


def successful_outputs(deployer):
    return ["arm64\n", f"{deployer.sha256}  -\n", "arm64\n", "", "installed\n"]


# ---------- SDKDeployer ----------


@pytest.mark.parametrize("kwargs", [{"timeout": 0}, {"timeout": -1}, {"timeout": True}, {"timeout": "600"}])
def test_rejects_invalid_timeout(deployer, kwargs):
    with pytest.raises(ValueError, match="timeout"):
        SDKDeployer(deployer.local_deb, **kwargs)


def test_rejects_missing_deb(tmp_path):
    with pytest.raises(FileNotFoundError):
        SDKDeployer(tmp_path / "missing.deb")


def test_module_exposes_only_single_connection_deploy(deployer):
    # 一个 case 只对应一台设备, 批量部署、建连和交互都不属于这个模块
    assert not hasattr(deployer, "deploy_many")
    assert not hasattr(deployer, "ssh")
    assert not hasattr(deployer, "jobs")
    assert not hasattr(sdk_deploy, "run_deployment")


# ---------- deploy: 使用调用方提供的连接 ----------


def test_deploy_uses_given_connection_and_leaves_it_open(deployer, capsys):
    connection = FakeConnection(successful_outputs(deployer), peer=("2001:db8::1", 2222, 0, 0))

    output = asyncio.run(deployer.deploy(connection))

    assert output == "installed\n"
    assert connection.uploads == [(str(deployer.local_deb), "/opt/sdk.deb")]
    assert connection.commands[-2:] == [
        "rm -rf --one-file-system -- /opt/bw01d /opt/bw01",
        "dpkg -i -- /opt/sdk.deb",
    ]
    assert "sha256sum < /opt/sdk.deb" in connection.commands
    # 不检查登录账号, 权限问题交给远程命令自身报错
    assert not any(command.startswith("id ") for command in connection.commands)
    assert not connection.closed
    assert "[[2001:db8::1]:2222] 上传 sdk.deb" in capsys.readouterr().out


def test_remote_path_is_quoted_for_shell(tmp_path):
    deb = tmp_path / "sdk v1 (beta).deb"
    deb.write_bytes(b"test")
    deployer = SDKDeployer(deb)
    connection = FakeConnection(successful_outputs(deployer))

    asyncio.run(deployer.deploy(connection))

    assert connection.uploads == [(str(deb), "/opt/sdk v1 (beta).deb")]
    assert connection.commands[-1] == "dpkg -i -- '/opt/sdk v1 (beta).deb'"


@pytest.mark.parametrize(
    "outputs,error,message",
    [
        (["amd64\n"], ValueError, "不是 arm64"),
        (["arm64\n", "0" * 64 + "  -\n"], ValueError, "SHA-256"),
        (["arm64\n", None, "amd64\n"], ValueError, "arm64 架构"),
    ],
)
def test_deploy_stops_before_removing_old_install(deployer, outputs, error, message):
    outputs = [deployer.sha256 + "  -\n" if item is None else item for item in outputs]
    connection = FakeConnection(outputs)

    with pytest.raises(error, match=message):
        asyncio.run(deployer.deploy(connection))

    assert not any(command.startswith("rm -rf") for command in connection.commands)
    assert not any(command.startswith("dpkg -i") for command in connection.commands)
    assert not connection.closed


def test_deploy_times_out_and_leaves_connection_to_caller(deployer):
    deployer.timeout = 0.01
    connection = FakeConnection(successful_outputs(deployer), delay=1)

    with pytest.raises(TimeoutError):
        asyncio.run(deployer.deploy(connection))

    assert connection.commands == ["dpkg --print-architecture"]
    assert not connection.closed


def test_deploy_can_be_repeated_on_different_connections(deployer):
    first = FakeConnection(successful_outputs(deployer), peer=("10.27.17.45", 22))
    second = FakeConnection(successful_outputs(deployer), peer=("10.27.17.46", 22))

    assert asyncio.run(deployer.deploy(first)) == "installed\n"
    assert asyncio.run(deployer.deploy(second)) == "installed\n"
    assert first.uploads == second.uploads == [(str(deployer.local_deb), "/opt/sdk.deb")]


# ---------- 命令行入口 ----------


@pytest.fixture
def connect(monkeypatch):
    """用 FakeConnection 替代真实建连; 测试可通过 outputs / failure 属性预置远程行为"""
    state = SimpleNamespace(outputs=[], failure=None, created=[])

    async def fake_connect(username, host, *, port=22, password=None):
        connection = FakeConnection(
            list(state.outputs),
            peer=(host, port),
            username=username,
            failure=state.failure,
        )
        state.created.append(connection)
        return connection

    mock = AsyncMock(side_effect=fake_connect)
    # runpy 重新执行模块时会重新 import SSHConnector, 因此必须替换类属性而不是模块属性
    monkeypatch.setattr(SSHConnector, "connect", mock)
    state.mock = mock
    return state


def run_main(monkeypatch, *argv):
    """以 __main__ 方式运行模块, 返回退出码"""
    monkeypatch.setattr(sys, "argv", ["sdk_deploy.py", *argv])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(sdk_deploy.__file__, run_name="__main__")
    return error.value.code


def test_import_does_not_parse_arguments(monkeypatch):
    parse = Mock(side_effect=AssertionError("导入时不应解析参数"))
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse)

    namespace = runpy.run_path(sdk_deploy.__file__)

    assert "SDKDeployer" in namespace
    assert "run_deployment" not in namespace
    assert "main" not in namespace
    parse.assert_not_called()


def test_script_entry_parses_help(monkeypatch, capsys):
    assert run_main(monkeypatch, "--help") == 0

    output = capsys.readouterr().out
    for option in ("--ip", "--port", "--username", "--password", "--deb", "--trusted-ips", "--timeout"):
        assert option in output
    for removed in ("--jobs", "--password-per-host", "--identity"):
        assert removed not in output


@pytest.mark.parametrize("missing", ["--ip", "--password", "--deb"])
def test_script_entry_requires_ip_password_and_deb(deployer, connect, monkeypatch, capsys, missing):
    argv = {
        "--ip": "10.27.17.45",
        "--password": "secret",
        "--deb": str(deployer.local_deb),
    }
    del argv[missing]
    flattened = [item for pair in argv.items() for item in pair]

    assert run_main(monkeypatch, *flattened) == 2

    assert missing in capsys.readouterr().err
    connect.mock.assert_not_awaited()


def test_script_entry_connects_with_all_arguments_and_closes_connection(deployer, connect, monkeypatch, capsys):
    connect.outputs = successful_outputs(deployer)
    monkeypatch.setattr("builtins.input", Mock(side_effect=AssertionError("不应交互输入")))

    code = run_main(
        monkeypatch,
        "--ip",
        "2001:0db8::1",
        "--port",
        "2222",
        "--username",
        "tester",
        "--password",
        "secret",
        "--deb",
        str(deployer.local_deb),
        "--timeout",
        "120",
    )

    assert code == 0
    connect.mock.assert_awaited_once_with("tester", "2001:0db8::1", port=2222, password="secret")
    assert connect.created[0].closed
    output = capsys.readouterr().out
    assert "tester@[2001:0db8::1]:2222" in output
    assert "installed" in output
    assert "成功" in output


def test_script_entry_defaults_port_and_username(deployer, connect, monkeypatch):
    connect.outputs = successful_outputs(deployer)

    code = run_main(
        monkeypatch,
        "--ip",
        "10.27.17.45",
        "--password",
        "secret",
        "--deb",
        str(deployer.local_deb),
    )

    assert code == 0
    connect.mock.assert_awaited_once_with("root", "10.27.17.45", port=22, password="secret")


def test_script_entry_passes_trusted_ips_to_connector(deployer, monkeypatch):
    host_key = asyncssh.import_public_key(asyncssh.generate_private_key("ssh-ed25519").export_public_key())
    fetch = AsyncMock(return_value=host_key)
    monkeypatch.setattr(asyncssh, "get_server_host_key", fetch)
    asyncssh_connect = Mock(side_effect=RuntimeError("stop before connection"))
    monkeypatch.setattr(asyncssh, "connect", asyncssh_connect)
    # known_hosts 路径由 ssh_connection 里的 Path.home() 决定; 指到临时目录, 不写用户真实文件
    import utils.ssh_connection as ssh_connection

    monkeypatch.setattr(ssh_connection.Path, "home", classmethod(lambda cls: deployer.local_deb.parent))

    with pytest.raises(RuntimeError, match="stop before connection"):
        run_main(
            monkeypatch,
            "--ip",
            "10.27.17.45",
            "--password",
            "secret",
            "--deb",
            str(deployer.local_deb),
            "--trusted-ips",
            "10.27.17.0/24",
            "2001:db8::/64",
        )

    # 白名单生效: 首次连接前取回并记录了主机公钥, 并把 known_hosts 交给 asyncssh.connect
    fetch.assert_awaited_once_with("10.27.17.45", port=22)
    options = asyncssh_connect.call_args.kwargs
    assert options["username"] == "root"
    assert options["port"] == 22
    assert options["connect_timeout"] == 15
    assert host_key in options["known_hosts"].match("10.27.17.45", "10.27.17.45", None)[0]


@pytest.mark.parametrize(
    "argv",
    [
        ["--ip", "invalid"],
        ["--ip", "10.27.17.45:2222"],
        ["--ip", "[::1]"],
        ["--ip", "10.27.17.45", "--port", "0"],
        ["--ip", "10.27.17.45", "--port", "65536"],
        ["--ip", "10.27.17.45", "--username", ""],
        ["--ip", "10.27.17.45", "--trusted-ips", "10.27.17.*"],
    ],
)
def test_script_entry_reports_invalid_connection_arguments(deployer, monkeypatch, capsys, argv):
    # 参数取值的校验交给 SSHConnector, 这里只确认错误被转成退出码 1 且未真正连接
    asyncssh_connect = AsyncMock()
    monkeypatch.setattr(asyncssh, "connect", asyncssh_connect)

    code = run_main(monkeypatch, *argv, "--password", "secret", "--deb", str(deployer.local_deb))

    assert code == 1
    assert "部署失败" in capsys.readouterr().err
    asyncssh_connect.assert_not_awaited()


def test_script_entry_reports_missing_deb(tmp_path, connect, monkeypatch, capsys):
    code = run_main(
        monkeypatch,
        "--ip",
        "10.27.17.45",
        "--password",
        "secret",
        "--deb",
        str(tmp_path / "missing.deb"),
    )

    assert code == 1
    assert "部署失败" in capsys.readouterr().err
    connect.mock.assert_not_awaited()


def test_script_entry_reports_connection_failure(deployer, monkeypatch, capsys):
    monkeypatch.setattr(SSHConnector, "connect", AsyncMock(side_effect=OSError("无法连接")))

    code = run_main(monkeypatch, "--ip", "10.27.17.45", "--password", "secret", "--deb", str(deployer.local_deb))

    assert code == 1
    assert "部署失败: 无法连接" in capsys.readouterr().err


def test_script_entry_reports_remote_command_stderr(deployer, connect, monkeypatch, capsys):
    class RemoteCommandError(asyncssh.Error):
        """模拟 asyncssh.ProcessError: 异常上带有远程命令的 stderr"""

        def __init__(self):
            super().__init__(1, "Process exited with non-zero exit status 1")
            self.stderr = "dpkg: error processing package\n"

    # runpy 会重新定义 SDKDeployer, 不能 patch 类方法, 改为让远程命令本身失败
    connect.failure = RemoteCommandError()

    code = run_main(monkeypatch, "--ip", "10.27.17.45", "--password", "secret", "--deb", str(deployer.local_deb))

    assert code == 1
    assert "部署失败: dpkg: error processing package" in capsys.readouterr().err
    assert connect.created[0].closed


def test_script_entry_reports_timeout(deployer, connect, monkeypatch, capsys):
    connect.outputs = successful_outputs(deployer)

    async def slow_connect(username, host, *, port=22, password=None):
        # 每条远程命令耗时 5 秒, 远超下面传入的 --timeout 1 秒
        connection = FakeConnection(list(connect.outputs), peer=(host, port), username=username, delay=5)
        connect.created.append(connection)
        return connection

    connect.mock.side_effect = slow_connect

    code = run_main(
        monkeypatch,
        "--ip",
        "10.27.17.45",
        "--password",
        "secret",
        "--deb",
        str(deployer.local_deb),
        "--timeout",
        "1",
    )

    assert code == 1
    assert "部署超时" in capsys.readouterr().err
    assert connect.created[0].closed
