"""remote 层测试: 在 paramiko.SSHClient 这一层打桩, 不开真实连接"""

from __future__ import annotations

import ipaddress
from types import SimpleNamespace
from unittest.mock import Mock

import paramiko
import pytest

from cimalgo.remote import (
    CommandError,
    DeviceBusyError,
    DeviceLockError,
    LockedDevice,
    RemoteDeviceLock,
    SDKDeployer,
    SSHConnection,
    SSHConnector,
)
from cimalgo.remote import device_lock as device_lock_module
from cimalgo.remote import ssh as ssh_module

# ---------------------------------------------------------------------------
# 假 paramiko 客户端
# ---------------------------------------------------------------------------


class FakeChannel:
    """模拟 transport.open_session() 返回的 channel: 预置 stdout 行、stderr、退出码"""

    def __init__(self, lines: list[str] = (), stderr: str = "", exit_status: int = 0):
        self._out = "".join(lines).encode()
        self._err = stderr.encode()
        self.exit_status = exit_status
        self.closed = False
        self.eof_received = True
        self.eof_sent = False
        self.timeout: float | None = None
        self.command: str | None = None

    def exec_command(self, command):
        self.command = command

    def settimeout(self, value):
        self.timeout = value

    def recv(self, n):
        chunk, self._out = self._out[:n], self._out[n:]
        return chunk

    def recv_ready(self):
        return bool(self._out)

    def recv_stderr_ready(self):
        return bool(self._err)

    def exit_status_ready(self):
        return True

    def recv_stderr(self, n):
        chunk, self._err = self._err[:n], self._err[n:]
        return chunk

    def recv_exit_status(self):
        return self.exit_status

    def shutdown_write(self):
        self.eof_sent = True

    def close(self):
        self.closed = True


class FakeClient:
    def __init__(self):
        self.commands: list[str] = []
        self.responses: dict[str, tuple[str, str, int]] = {}
        self.uploads: list[tuple[str, str]] = []
        self.process_channel = FakeChannel(["LOCKED\n"])
        self.closed = False
        self.connect_kwargs: dict = {}
        self.policy = None
        self.host_keys = paramiko.HostKeys()

    # SSHClient 接口
    def load_host_keys(self, path):
        pass

    def set_missing_host_key_policy(self, policy):
        self.policy = policy

    def get_host_keys(self):
        return self.host_keys

    def connect(self, host, **kwargs):
        self.connect_kwargs = {"host": host, **kwargs}

    def exec_command(self, command, timeout=None):
        self.commands.append(command)
        stdout, stderr, status = self.responses.get(command, ("", "", 0))
        out_channel = FakeChannel([stdout], stderr=stderr, exit_status=status)
        stdout_file = SimpleNamespace(channel=out_channel, read=lambda: stdout.encode())
        stderr_file = SimpleNamespace(read=lambda: stderr.encode())
        return None, stdout_file, stderr_file

    def get_transport(self):
        channel = self.process_channel
        return SimpleNamespace(is_active=lambda: True, open_session=lambda: channel)

    def open_sftp(self):
        client = self

        class Sftp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def put(self, local, remote):
                client.uploads.append((local, remote))

        return Sftp()

    def close(self):
        self.closed = True


@pytest.fixture
def fake_client(monkeypatch) -> FakeClient:
    client = FakeClient()
    monkeypatch.setattr(ssh_module.paramiko, "SSHClient", lambda: client)
    return client


@pytest.fixture
def connector(tmp_path) -> SSHConnector:
    return SSHConnector(trusted_ips=["10.27.17.0/24"], known_hosts=tmp_path / "known_hosts")


# ---------------------------------------------------------------------------
# SSHConnector
# ---------------------------------------------------------------------------


def test_connect_passes_username_and_auth_options(connector, fake_client):
    connection = connector.connect(" tester ", " 10.27.17.45 ", port=2222, password="secret")

    assert isinstance(connection, SSHConnection)
    assert connection.username == "tester"
    assert connection.endpoint == "10.27.17.45:2222"
    kwargs = fake_client.connect_kwargs
    assert kwargs["host"] == "10.27.17.45"
    assert kwargs["port"] == 2222
    assert kwargs["username"] == "tester"
    assert kwargs["password"] == "secret"
    assert kwargs["allow_agent"] is False and kwargs["look_for_keys"] is False
    assert kwargs["timeout"] == 15


def test_connect_with_identity_enables_key_auth(tmp_path, fake_client):
    identity = tmp_path / "id"
    identity.write_text("key")
    SSHConnector(identity=identity).connect("root", "10.27.17.45")
    kwargs = fake_client.connect_kwargs
    assert kwargs["key_filename"] == str(identity)
    assert kwargs["password"] is None
    assert kwargs["allow_agent"] is True


def test_connect_normalizes_ipv6(connector, fake_client):
    connection = connector.connect("root", "2001:0db8::1")
    assert fake_client.connect_kwargs["host"] == "2001:db8::1"
    assert connection.endpoint == "[2001:db8::1]:22"


def test_connect_wraps_paramiko_errors(connector, fake_client):
    fake_client.connect = Mock(side_effect=paramiko.AuthenticationException("bad password"))
    with pytest.raises(ConnectionError, match="bad password"):
        connector.connect("root", "10.27.17.45")
    assert fake_client.closed


@pytest.mark.parametrize(
    "kwargs",
    [
        {"username": "", "host": "10.27.17.45"},
        {"username": "root", "host": "device.local"},
        {"username": "root", "host": "10.27.17.45", "port": 0},
        {"username": "root", "host": "10.27.17.45", "port": True},
        {"username": "root", "host": "10.27.17.45", "password": 123},
    ],
)
def test_connect_rejects_invalid_arguments(connector, fake_client, kwargs):
    with pytest.raises(ValueError):
        connector.connect(**kwargs)
    assert not fake_client.connect_kwargs


@pytest.mark.parametrize(
    "bad",
    [{"connect_timeout": 0}, {"connect_timeout": True}, {"identity": "missing"}, {"trusted_ips": ["10.0.0.1/24"]}],
)
def test_connector_rejects_invalid_configuration(bad):
    with pytest.raises((ValueError, FileNotFoundError)):
        SSHConnector(**bad)


@pytest.mark.parametrize(
    ("rule", "ip", "expected"),
    [
        ("10.27.17.0/24", "10.27.17.45", True),
        ("10.27.17.0/24", "10.27.18.1", False),
        ("10.27.17.45", "10.27.17.45", True),
        ("2001:db8::/64", "2001:db8::1", True),
        ("2001:db8::/64", "2001:db9::1", False),
        ("fe80::1%eth0", "fe80::1%eth0", True),
        ("fe80::1%eth0", "fe80::1%eth1", False),
        ("10.27.17.0/24", "device.local", False),
    ],
)
def test_is_trusted_ip(rule, ip, expected):
    assert SSHConnector(trusted_ips=[rule]).is_trusted_ip(ip) is expected


def test_single_ip_stored_as_host_network():
    connector = SSHConnector(trusted_ips=["10.27.17.45", "2001:db8::1"])
    assert connector.trusted_networks == {
        ipaddress.ip_network("10.27.17.45/32"),
        ipaddress.ip_network("2001:db8::1/128"),
    }


# ---------------------------------------------------------------------------
# 首次信任
# ---------------------------------------------------------------------------


@pytest.fixture
def host_key() -> paramiko.PKey:
    return (
        paramiko.Ed25519Key.generate() if hasattr(paramiko.Ed25519Key, "generate") else paramiko.RSAKey.generate(1024)
    )


def test_trusted_host_policy_records_key_once(connector, fake_client, host_key):
    connector.connect("root", "10.27.17.45", port=2222)
    policy = fake_client.policy
    policy.missing_host_key(fake_client, "[10.27.17.45]:2222", host_key)
    policy.missing_host_key(fake_client, "[10.27.17.45]:2222", host_key)

    lines = connector.known_hosts.read_text().strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("[10.27.17.45]:2222 ")
    assert connector.known_hosts.stat().st_mode & 0o777 == 0o600
    assert fake_client.host_keys.lookup("[10.27.17.45]:2222")


def test_untrusted_host_is_rejected(connector, fake_client, host_key):
    connector.connect("root", "10.27.18.99")
    with pytest.raises(paramiko.SSHException, match="不在首次信任白名单"):
        fake_client.policy.missing_host_key(fake_client, "10.27.18.99", host_key)
    assert not connector.known_hosts.exists()


def test_existing_record_is_never_replaced(connector, fake_client, host_key):
    connector.known_hosts.parent.mkdir(exist_ok=True)
    original = f"10.27.17.45 {host_key.get_name()} {host_key.get_base64()}\n"
    connector.known_hosts.write_text(original)
    other = paramiko.RSAKey.generate(1024)
    with pytest.raises(paramiko.BadHostKeyException):
        connector.remember_host_key("10.27.17.45", 22, other)
    assert connector.known_hosts.read_text() == original


@pytest.mark.parametrize("port", [22, 2222])
@pytest.mark.parametrize("different_algorithm", [False, True])
def test_first_trust_rechecks_key_for_client_with_stale_cache(connector, host_key, port, different_algorithm):
    first, stale = FakeClient(), FakeClient()
    hostname = "10.27.17.45" if port == 22 else f"[10.27.17.45]:{port}"
    policy = ssh_module._TrustedHostPolicy(connector, "10.27.17.45", port)
    other_key = paramiko.ECDSAKey.generate() if different_algorithm else paramiko.RSAKey.generate(1024)

    policy.missing_host_key(first, hostname, host_key)
    with pytest.raises(paramiko.BadHostKeyException):
        policy.missing_host_key(stale, hostname, other_key)
    assert not stale.host_keys.lookup(hostname)
    assert paramiko.HostKeys(str(connector.known_hosts)).check(hostname, host_key)
    # 相同公钥的并发首次连接仍可以正常建立。
    policy.missing_host_key(stale, hostname, host_key)
    assert stale.host_keys.check(hostname, host_key)


# ---------------------------------------------------------------------------
# SSHConnection
# ---------------------------------------------------------------------------


def test_run_returns_result_and_checks_exit_status(connector, fake_client):
    fake_client.responses["echo hi"] = ("hi\n", "", 0)
    fake_client.responses["false"] = ("", "boom\n", 1)
    with connector.connect("root", "10.27.17.45") as connection:
        assert connection.run("echo hi").stdout == "hi\n"
        assert connection.run("false", check=False).exit_status == 1
        with pytest.raises(CommandError, match="boom") as raised:
            connection.run("false")
        assert raised.value.result.exit_status == 1
    assert fake_client.closed
    with pytest.raises(RuntimeError, match="已关闭"):
        connection.run("echo")


def test_upload_uses_sftp(connector, fake_client, tmp_path):
    local = tmp_path / "f"
    local.write_text("x")
    with connector.connect("root", "10.27.17.45") as connection:
        connection.upload(local, "/opt/f")
    assert fake_client.uploads == [(str(local), "/opt/f")]


def test_run_drains_large_stdout_and_stderr_before_waiting(connector, fake_client):
    events = []

    class FlowControlledChannel(FakeChannel):
        def recv(self, n):
            events.append("stdout")
            return super().recv(n)

        def recv_stderr(self, n):
            events.append("stderr")
            return super().recv_stderr(n)

        def exit_status_ready(self):
            return not self._out and not self._err

        def recv_exit_status(self):
            assert self.exit_status_ready(), "必须先读取输出, 否则远端写满窗口后无法退出"
            return self.exit_status

    out, err = "x" * (4 * 1024 * 1024), "y" * (4 * 1024 * 1024)
    channel = FlowControlledChannel([out], stderr=err)
    fake_client.exec_command = Mock(return_value=(None, SimpleNamespace(channel=channel), None))
    connection = connector.connect("root", "10.27.17.45")
    result = connection.run("verbose", timeout=2)
    assert result.stdout == out and result.stderr == err
    assert events[:4] == ["stdout", "stderr", "stdout", "stderr"]
    assert channel.closed and not fake_client.closed


def test_run_timeout_closes_real_paramiko_channel(connector, fake_client):
    channel = paramiko.Channel(0)
    channel.active = True
    channel.transport = Mock()
    channel.recv_exit_status = Mock(side_effect=AssertionError("未就绪时不能调用阻塞等待"))
    fake_client.exec_command = Mock(return_value=(None, SimpleNamespace(channel=channel), None))
    connection = connector.connect("root", "10.27.17.45")
    with pytest.raises(TimeoutError, match="超时"):
        connection.run("never exits", timeout=0.02)
    assert channel.closed
    channel.recv_exit_status.assert_not_called()
    assert not fake_client.closed


def test_run_waits_for_output_eof_even_if_exit_status_arrives_first(connector, fake_client, monkeypatch):
    channel = FakeChannel()
    channel.eof_received = False

    def receive_late_output(_):
        channel._out = b"last output"
        channel.eof_received = True

    monkeypatch.setattr(ssh_module.time, "sleep", receive_late_output)
    fake_client.exec_command = Mock(return_value=(None, SimpleNamespace(channel=channel), None))
    connection = connector.connect("root", "10.27.17.45")
    assert connection.run("late output").stdout == "last output"
    assert channel.closed


@pytest.mark.parametrize("timeout", [0, -1, True, "1", float("inf"), float("nan")])
def test_run_rejects_invalid_timeout(connector, fake_client, timeout):
    connection = connector.connect("root", "10.27.17.45")
    with pytest.raises(ValueError, match="timeout"):
        connection.run("echo", timeout=timeout)
    assert not fake_client.commands


# ---------------------------------------------------------------------------
# 设备锁
# ---------------------------------------------------------------------------


def test_lock_acquire_and_release_keep_connection_open(connector, fake_client):
    connection = connector.connect("root", "10.27.17.45")
    lock = RemoteDeviceLock(connection, case="test_conv", owner="alice")

    with lock:
        assert lock.acquired
    assert not lock.acquired
    channel = fake_client.process_channel
    assert channel.eof_sent and channel.closed
    assert "flock -n 9" in channel.command and '"case":"test_conv"' in channel.command
    assert not fake_client.closed
    assert '"owner":"alice"' in lock.lock_info
    assert '"case":"test_conv"' in lock.lock_info


def test_lock_busy_reports_owner(connector, fake_client):
    info = '{"owner":"other","case":"running"}'
    fake_client.process_channel = FakeChannel(["BUSY\n", info + "\n"])
    connection = connector.connect("root", "10.27.17.45")
    with pytest.raises(DeviceBusyError) as raised:
        RemoteDeviceLock(connection, case="new").acquire()
    assert raised.value.lock_info == info
    assert '"owner": "other"' in str(raised.value)
    assert fake_client.process_channel.closed
    assert not fake_client.closed


def test_lock_error_is_not_busy(connector, fake_client):
    fake_client.process_channel = FakeChannel(["ERROR\n"], stderr="远程设备未安装 flock\n")
    connection = connector.connect("root", "10.27.17.45")
    with pytest.raises(DeviceLockError, match="未安装 flock"):
        RemoteDeviceLock(connection, case="x").acquire()


def test_lock_acquire_timeout(connector, fake_client):
    class Hanging(FakeChannel):
        def recv(self, n):
            raise TimeoutError

    fake_client.process_channel = Hanging()
    connection = connector.connect("root", "10.27.17.45")
    with pytest.raises(DeviceLockError, match="超时"):
        RemoteDeviceLock(connection, case="x", acquire_timeout=0.1).acquire()
    assert fake_client.process_channel.closed


def test_lock_double_acquire_rejected(connector, fake_client):
    connection = connector.connect("root", "10.27.17.45")
    lock = RemoteDeviceLock(connection, case="x").acquire()
    with pytest.raises(DeviceLockError, match="已由当前实例持有"):
        lock.acquire()
    lock.release()
    lock.release()  # 幂等


def test_lock_release_enforces_deadline_and_closes_channel(connector, fake_client, monkeypatch):
    connection = connector.connect("root", "10.27.17.45")
    lock = RemoteDeviceLock(connection, case="x").acquire()
    channel = paramiko.Channel(0)
    channel.active = True
    channel.transport = Mock()
    channel.recv_exit_status = Mock(side_effect=AssertionError("未就绪时不能调用阻塞等待"))
    lock._channel = channel
    monkeypatch.setattr(ssh_module.time, "monotonic", Mock(side_effect=[0, 0, 6]))
    monkeypatch.setattr(ssh_module.time, "sleep", lambda _: None)

    lock.release()
    assert channel.closed and not lock.acquired
    assert not fake_client.closed
    channel.recv_exit_status.assert_not_called()
    lock.release()


@pytest.mark.parametrize("kwargs", [{"case": ""}, {"case": "x", "owner": ""}, {"case": "x", "acquire_timeout": 0}])
def test_lock_rejects_invalid_arguments(connector, fake_client, kwargs):
    connection = connector.connect("root", "10.27.17.45")
    with pytest.raises(ValueError):
        RemoteDeviceLock(connection, **kwargs)


def test_locked_device_releases_lock_then_closes_connection(connector, fake_client):
    with LockedDevice.open(connector, username="root", host="10.27.17.45", case="x") as device:
        assert device.lock.acquired
        assert not fake_client.closed
    assert not device.lock.acquired
    assert fake_client.closed


def test_locked_device_closes_connection_when_lock_fails(connector, fake_client):
    fake_client.process_channel = FakeChannel(["BUSY\n", "{}\n"])
    with pytest.raises(DeviceBusyError):
        LockedDevice.open(connector, username="root", host="10.27.17.45", case="x")
    assert fake_client.closed


def test_device_lock_cli_busy_exit_code(connector, fake_client, monkeypatch):
    fake_client.process_channel = FakeChannel(["BUSY\n", "{}\n"])
    monkeypatch.setattr(device_lock_module, "SSHConnector", lambda **kwargs: connector)
    monkeypatch.setattr(device_lock_module.getpass, "getpass", lambda prompt: "pw")
    assert device_lock_module.main(["--ip", "10.27.17.45", "--case", "x"]) == 75


# ---------------------------------------------------------------------------
# SDK 部署
# ---------------------------------------------------------------------------


@pytest.fixture
def deployer(tmp_path) -> SDKDeployer:
    deb = tmp_path / "sdk.deb"
    deb.write_bytes(b"test")
    return SDKDeployer(deb)


def _happy_responses(client: FakeClient, deployer: SDKDeployer) -> None:
    client.responses["dpkg --print-architecture"] = ("arm64\n", "", 0)
    client.responses[f"sha256sum < {deployer.remote_deb}"] = (f"{deployer.sha256}  -\n", "", 0)
    client.responses[f"dpkg-deb --show --showformat='${{Architecture}}' {deployer.remote_deb}"] = ("arm64\n", "", 0)
    client.responses[f"dpkg -i -- {deployer.remote_deb}"] = ("installed\n", "", 0)


def test_deploy_happy_path(connector, fake_client, deployer):
    _happy_responses(fake_client, deployer)
    with connector.connect("root", "10.27.17.45") as connection:
        assert deployer.deploy(connection) == "installed\n"
    assert fake_client.uploads == [(str(deployer.local_deb), "/opt/sdk.deb")]
    assert fake_client.commands[-2:] == ["rm -rf --one-file-system -- /opt/bw01d /opt/bw01", "dpkg -i -- /opt/sdk.deb"]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"dpkg --print-architecture": ("amd64\n", "", 0)}, "不是 arm64"),
        ({"sha256sum < /opt/sdk.deb": ("0" * 64 + "  -\n", "", 0)}, "SHA-256"),
        ({"dpkg-deb --show --showformat='${Architecture}' /opt/sdk.deb": ("amd64\n", "", 0)}, "arm64 架构"),
    ],
)
def test_deploy_stops_before_removing_old_install(connector, fake_client, deployer, override, message):
    _happy_responses(fake_client, deployer)
    fake_client.responses.update(override)
    with connector.connect("root", "10.27.17.45") as connection, pytest.raises(ValueError, match=message):
        deployer.deploy(connection)
    assert not any(command.startswith("rm -rf") for command in fake_client.commands)


def test_deploy_surfaces_remote_command_failure(connector, fake_client, deployer):
    _happy_responses(fake_client, deployer)
    fake_client.responses["dpkg -i -- /opt/sdk.deb"] = ("", "dpkg: error\n", 1)
    with connector.connect("root", "10.27.17.45") as connection, pytest.raises(CommandError, match="dpkg: error"):
        deployer.deploy(connection)


def test_for_sdk_finds_unique_deb(tmp_path):
    sdk = SimpleNamespace(chip="bw01d", version="2.3.3", install_dir=tmp_path)
    deb = tmp_path / "Agent/EVB/debs/bw01d-sdk_2.3.3_arm64.deb"
    deb.parent.mkdir(parents=True)
    deb.write_bytes(b"x")
    assert SDKDeployer.for_sdk(sdk).local_deb == deb
    (tmp_path / "Agent/OTHER/debs").mkdir(parents=True)
    (tmp_path / "Agent/OTHER/debs/bw01d-sdk_2.3.3_arm64.deb").write_bytes(b"y")
    with pytest.raises(ValueError, match="多个"):
        SDKDeployer.for_sdk(sdk)


@pytest.mark.parametrize("timeout", [0, -1, True, "600"])
def test_deployer_rejects_invalid_timeout(deployer, timeout):
    with pytest.raises(ValueError):
        SDKDeployer(deployer.local_deb, timeout=timeout)
