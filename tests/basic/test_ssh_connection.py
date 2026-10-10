import asyncio
import ipaddress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import asyncssh
import pytest

import utils.ssh_connection as ssh_connection
from utils.ssh_connection import SSHConnector, endpoint_of


@pytest.fixture
def connector(tmp_path):
    # 所有信任记录只写入 pytest 临时目录, 不访问用户的真实 known_hosts
    return SSHConnector(
        trusted_ips=["10.27.17.45", "2001:db8::1"],
        known_hosts=tmp_path / "ssh" / "known_hosts",
    )


@pytest.fixture
def host_key():
    key = asyncssh.generate_private_key("ssh-ed25519")
    return asyncssh.import_public_key(key.export_public_key())


# ---------- 公共参数与连接创建 ----------


def test_connect_returns_connection_with_common_and_call_options(monkeypatch):
    connection = object()
    connect = AsyncMock(return_value=connection)
    monkeypatch.setattr(ssh_connection.asyncssh, "connect", connect)
    connector = SSHConnector(connect_timeout=20)

    result = asyncio.run(connector.connect(" tester ", " 10.27.17.45 ", port=2222, password="secret"))

    assert result is connection
    connect.assert_awaited_once_with(
        "10.27.17.45",
        port=2222,
        password="secret",
        username="tester",
        connect_timeout=20,
    )


def test_each_connection_can_use_a_different_username(monkeypatch):
    connect = AsyncMock(return_value=object())
    monkeypatch.setattr(ssh_connection.asyncssh, "connect", connect)
    connector = SSHConnector()

    asyncio.run(connector.connect("root", "10.27.17.45"))
    asyncio.run(connector.connect("tester", "10.27.17.45"))

    assert [args.kwargs["username"] for args in connect.await_args_list] == ["root", "tester"]
    assert not hasattr(connector, "username")


def test_default_known_hosts_behavior_is_not_overridden(monkeypatch):
    connect = AsyncMock(return_value=object())
    monkeypatch.setattr(ssh_connection.asyncssh, "connect", connect)
    connector = SSHConnector()

    asyncio.run(connector.connect("root", "10.27.17.45"))

    assert "known_hosts" not in connect.await_args.kwargs
    assert connector.known_hosts == Path.home() / ".ssh" / "known_hosts"


def test_identity_is_validated_and_passed_to_asyncssh(tmp_path, monkeypatch):
    identity = tmp_path / "id_ed25519"
    identity.write_text("not a real key")
    connect = AsyncMock(return_value=object())
    monkeypatch.setattr(ssh_connection.asyncssh, "connect", connect)
    connector = SSHConnector(identity=identity)

    asyncio.run(connector.connect("root", "10.27.17.45"))

    assert connect.await_args.kwargs["client_keys"] == [str(identity)]
    assert connector.make_options("root")["client_keys"] == [str(identity)]


def test_make_options_returns_a_copy_with_given_username():
    connector = SSHConnector()
    options = connector.make_options("tester")
    options["port"] = 2222

    assert options["username"] == "tester"
    assert "port" not in connector.make_options("tester")
    assert "known_hosts" not in connector.make_options("tester")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"connect_timeout": 0},
        {"connect_timeout": True},
        {"identity": "missing-key"},
        {"trusted_ips": ["10.27.17.*"]},
    ],
)
def test_rejects_invalid_common_configuration(kwargs):
    with pytest.raises((ValueError, FileNotFoundError)):
        SSHConnector(**kwargs)


def test_username_is_not_a_constructor_parameter():
    with pytest.raises(TypeError):
        SSHConnector(username="root")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"username": "", "host": "10.27.17.45"},
        {"username": None, "host": "10.27.17.45"},
        {"username": "root", "host": ""},
        {"username": "root", "host": "device.local"},
        {"username": "root", "host": "10.27.17.45", "port": 0},
        {"username": "root", "host": "10.27.17.45", "port": True},
        {"username": "root", "host": "10.27.17.45", "password": 123},
    ],
)
def test_rejects_invalid_per_connection_configuration(monkeypatch, kwargs):
    connect = AsyncMock()
    monkeypatch.setattr(ssh_connection.asyncssh, "connect", connect)
    fetch = AsyncMock()
    monkeypatch.setattr(asyncssh, "get_server_host_key", fetch)
    connector = SSHConnector(trusted_ips=["10.27.17.45"])

    with pytest.raises(ValueError):
        asyncio.run(connector.connect(**kwargs))

    connect.assert_not_awaited()
    fetch.assert_not_awaited()


# ---------- 首次信任主机公钥 ----------


@pytest.mark.parametrize(
    "ip,port",
    [
        ("10.27.17.45", 22),
        ("10.27.17.45", 2222),
        ("2001:db8::1", 22),
        ("2001:db8::1", 2222),
    ],
)
def test_trusted_new_host_is_saved_once(connector, host_key, monkeypatch, ip, port):
    fetch = AsyncMock(return_value=host_key)
    monkeypatch.setattr(asyncssh, "get_server_host_key", fetch)
    records = asyncio.run(connector.prepare_known_hosts(ip, port))
    asyncio.run(connector.prepare_known_hosts(ip, port))

    fetch.assert_awaited_once_with(ip, port=port)
    assert host_key in records.match(ip, ip, None if port == 22 else port)[0]
    address = ip if port == 22 else f"[{ip}]:{port}"
    assert connector.known_hosts.read_text().strip() == (f"{address} {host_key.export_public_key().decode().strip()}")
    assert connector.known_hosts.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("host", ["10.27.17.99", "device.local", "localhost"])
def test_unknown_host_is_not_automatically_trusted(connector, monkeypatch, host):
    fetch = AsyncMock()
    monkeypatch.setattr(asyncssh, "get_server_host_key", fetch)

    assert connector.is_trusted_ip(host) is False
    records = asyncio.run(connector.prepare_known_hosts(host))

    # asyncssh 的 addr 参数只接受 IP, 主机名时留空
    assert not any(records.match(host, "", None))
    fetch.assert_not_awaited()
    assert not connector.known_hosts.exists()


@pytest.mark.parametrize("marker", ["", "@revoked ", "@cert-authority "])
@pytest.mark.parametrize(
    "rule,ip",
    [
        ("10.27.17.45", "10.27.17.45"),
        ("10.27.17.0/24", "10.27.17.45"),
        ("2001:db8::/64", "2001:db8::1"),
    ],
)
def test_existing_records_are_never_replaced(connector, host_key, monkeypatch, marker, rule, ip):
    connector.trusted_networks = {ipaddress.ip_network(rule)}
    connector.known_hosts.parent.mkdir()
    original = f"{marker}{ip} {host_key.export_public_key().decode()}"
    connector.known_hosts.write_text(original)
    replacement = asyncssh.generate_private_key("ssh-ed25519")
    fetch = AsyncMock(return_value=replacement)
    monkeypatch.setattr(asyncssh, "get_server_host_key", fetch)

    records = asyncio.run(connector.prepare_known_hosts(ip))

    fetch.assert_not_awaited()
    assert connector.known_hosts.read_text() == original
    assert replacement not in records.match(ip, ip, None)[0]


def test_new_entry_preserves_existing_content(connector, host_key, monkeypatch):
    connector.known_hosts.parent.mkdir()
    original = f"10.27.17.99 {host_key.export_public_key().decode().strip()}"
    connector.known_hosts.write_text(original)  # 原文件没有末尾换行
    monkeypatch.setattr(asyncssh, "get_server_host_key", AsyncMock(return_value=host_key))

    records = asyncio.run(connector.prepare_known_hosts("10.27.17.45"))

    assert connector.known_hosts.read_text().startswith(original + "\n")
    assert host_key in records.match("10.27.17.99", "10.27.17.99", None)[0]
    assert host_key in records.match("10.27.17.45", "10.27.17.45", None)[0]


def test_rechecks_records_after_fetching_key(connector, host_key, monkeypatch):
    replacement = asyncssh.generate_private_key("ssh-ed25519")
    keys = iter([host_key, replacement])

    async def fetch(*args, **kwargs):
        key = next(keys)
        await asyncio.sleep(0)
        return key

    monkeypatch.setattr(asyncssh, "get_server_host_key", fetch)

    async def run():
        return await asyncio.gather(
            connector.prepare_known_hosts("10.27.17.45"),
            connector.prepare_known_hosts("10.27.17.45"),
        )

    results = asyncio.run(run())

    for records in results:
        matches = records.match("10.27.17.45", "10.27.17.45", None)[0]
        assert host_key in matches and replacement not in matches
    assert len(connector.known_hosts.read_text().strip().splitlines()) == 1


def test_missing_host_key_stops_before_login(connector, monkeypatch):
    monkeypatch.setattr(asyncssh, "get_server_host_key", AsyncMock(return_value=None))
    connect = AsyncMock()
    monkeypatch.setattr(asyncssh, "connect", connect)

    with pytest.raises(ValueError, match="无法获取"):
        asyncio.run(connector.connect("root", "10.27.17.45", password="test-password"))

    connect.assert_not_called()
    assert not connector.known_hosts.exists()


@pytest.mark.parametrize("ip", ["10.27.17.45", "10.27.17.99"])
@pytest.mark.parametrize("port", [22, 2222])
def test_connect_only_overrides_known_hosts_for_allowlist(connector, host_key, monkeypatch, ip, port):
    fetch = AsyncMock(return_value=host_key)
    monkeypatch.setattr(asyncssh, "get_server_host_key", fetch)
    # 在连接入口停止, 不建立真实 SSH 连接
    connect = Mock(side_effect=RuntimeError("stop before connection"))
    monkeypatch.setattr(asyncssh, "connect", connect)

    with pytest.raises(RuntimeError, match="stop before connection"):
        asyncio.run(connector.connect("root", ip, port=port))

    options = connect.call_args.kwargs
    assert options["port"] == port
    assert options["username"] == "root"
    if connector.is_trusted_ip(ip):
        fetch.assert_awaited_once_with(ip, port=port)
        assert options["known_hosts"] is not None
        assert host_key in options["known_hosts"].match(ip, ip, None if port == 22 else port)[0]
    else:
        assert "known_hosts" not in options
        fetch.assert_not_awaited()
        assert not connector.known_hosts.exists()
    assert "known_hosts" not in connector.make_options("root")


def test_trusted_ipv6_is_normalized_for_connection_and_record(connector, host_key, monkeypatch):
    fetch = AsyncMock(return_value=host_key)
    monkeypatch.setattr(asyncssh, "get_server_host_key", fetch)
    connect = Mock(side_effect=RuntimeError("stop before connection"))
    monkeypatch.setattr(asyncssh, "connect", connect)

    with pytest.raises(RuntimeError, match="stop before connection"):
        asyncio.run(connector.connect("root", "2001:0db8::1", port=2222))

    fetch.assert_awaited_once_with("2001:db8::1", port=2222)
    assert connect.call_args.args == ("2001:db8::1",)
    assert "[2001:db8::1]:2222 " in connector.known_hosts.read_text()


def test_same_ip_different_ports_keep_separate_keys(connector, host_key, monkeypatch):
    other_key = asyncssh.generate_private_key("ssh-ed25519").convert_to_public()
    fetch = AsyncMock(side_effect=[host_key, other_key])
    monkeypatch.setattr(asyncssh, "get_server_host_key", fetch)
    ip = "10.27.17.45"

    asyncio.run(connector.prepare_known_hosts(ip, 22))
    records = asyncio.run(connector.prepare_known_hosts(ip, 2222))

    assert fetch.await_count == 2
    assert set(records.match(ip, ip, None)[0]) == {host_key}
    assert set(records.match(ip, ip, 2222)[0]) == {other_key}
    asyncio.run(connector.prepare_known_hosts(ip, 22))
    asyncio.run(connector.prepare_known_hosts(ip, 2222))
    assert fetch.await_count == 2


@pytest.mark.parametrize(
    "rule,ip,expected",
    [
        ("10.27.17.0/24", "10.27.17.0", True),
        ("10.27.17.0/24", "10.27.17.45", True),
        ("10.27.17.0/24", "10.27.17.255", True),
        ("10.27.17.0/24", "10.27.18.0", False),
        ("10.27.17.0/24", "10.27.16.255", False),
        ("10.27.17.0/24", "110.27.17.45", False),
        ("10.27.17.0/25", "10.27.17.127", True),
        ("10.27.17.0/25", "10.27.17.128", False),
        ("10.27.0.0/16", "10.27.200.45", True),
        ("10.27.0.0/16", "10.28.17.45", False),
        ("10.27.17.45", "10.27.17.45", True),
        ("10.27.17.45", "10.27.17.46", False),
        ("10.27.17.45/32", "10.27.17.45", True),
        ("2001:0db8::1", "2001:db8::1", True),
        ("2001:db8::1", "2001:db8::2", False),
        ("2001:db8::1/128", "2001:0db8:0:0:0:0:0:1", True),
        ("2001:db8:1234:5678::/64", "2001:db8:1234:5678::", True),
        ("2001:db8:1234:5678::/64", "2001:db8:1234:5678:ffff:ffff:ffff:ffff", True),
        ("2001:db8:1234:5678::/64", "2001:db8:1234:5679::", False),
        ("2001:db8:1234:5678::/64", "2001:db8:1234:5677:ffff:ffff:ffff:ffff", False),
        ("10.27.17.0/24", "::ffff:10.27.17.45", False),
        ("2001:db8::/32", "10.27.17.45", False),
        ("fe80::1%eth0", "fe80::1%eth0", True),
        ("fe80::1%eth0", "fe80::1%eth1", False),
        ("fe80::1%eth0", "fe80::1", False),
        ("fe80::%eth0/64", "fe80::2%eth0", True),
        ("fe80::%eth0/64", "fe80::2%eth1", False),
        ("fe80::/64", "fe80::1%eth0", True),
        ("10.27.17.0/24", "device.local", False),
    ],
)
def test_trusted_networks_match_only_expected_addresses(rule, ip, expected):
    assert SSHConnector(trusted_ips=[rule]).is_trusted_ip(ip) is expected


@pytest.mark.parametrize(
    "rule",
    [
        "*",
        "10.27.17.*",
        "10.27.17.*0",
        "fe80::*",
        "",
        "invalid",
        123,
        None,
        "10.27.17.256",
        "010.27.17.0/24",
        "10.27.17.0/33",
        "10.27.17.0/-1",
        "2001:db8::/129",
        "10.27.17.45/24",
        "2001:db8::1/64",
    ],
)
def test_invalid_trusted_networks_are_rejected(rule):
    with pytest.raises(ValueError):
        SSHConnector(trusted_ips=[rule])


@pytest.mark.parametrize(
    "rule,ip,expected",
    [
        ("10.27.17.0/24", "10.27.17.45", True),
        ("10.27.17.0/24", "10.27.18.45", False),
        ("2001:db8::/64", "2001:db8::1", True),
        ("2001:db8::/64", "2001:db8:0:1::1", False),
    ],
)
def test_subnet_controls_first_host_key_acceptance(connector, host_key, monkeypatch, rule, ip, expected):
    connector.trusted_networks = {ipaddress.ip_network(rule)}
    fetch = AsyncMock(return_value=host_key)
    monkeypatch.setattr(asyncssh, "get_server_host_key", fetch)
    connect = Mock(side_effect=RuntimeError("stop before connection"))
    monkeypatch.setattr(asyncssh, "connect", connect)

    with pytest.raises(RuntimeError, match="stop before connection"):
        asyncio.run(connector.connect("root", ip, port=2222))

    if expected:
        fetch.assert_awaited_once_with(ip, port=2222)
        records = connect.call_args.kwargs["known_hosts"]
        assert host_key in records.match(ip, ip, 2222)[0]
        assert f"[{ip}]:2222" in connector.known_hosts.read_text()
    else:
        fetch.assert_not_awaited()
        assert "known_hosts" not in connect.call_args.kwargs
        assert not connector.known_hosts.exists()


def test_single_ips_are_stored_as_host_networks(connector):
    assert connector.trusted_networks == {
        ipaddress.ip_network("10.27.17.45/32"),
        ipaddress.ip_network("2001:db8::1/128"),
    }


def test_empty_allowlist_trusts_no_hosts():
    connector = SSHConnector()
    assert not connector.is_trusted_ip("10.27.17.45")
    assert not connector.is_trusted_ip("2001:db8::1")


def test_mixed_allowlist():
    connector = SSHConnector(
        trusted_ips={
            "10.27.17.0/24",
            "10.27.18.45",
            "2001:db8::/64",
            "2001:db8:1::1",
        }
    )
    for ip in ("10.27.17.100", "10.27.18.45", "2001:db8::abcd", "2001:db8:1::1"):
        assert connector.is_trusted_ip(ip)
    for ip in ("10.27.18.46", "2001:db8:1::2"):
        assert not connector.is_trusted_ip(ip)


# ---------- 连接描述 ----------


@pytest.mark.parametrize(
    "peername,expected",
    [
        (("10.27.17.45", 22), "10.27.17.45:22"),
        (("2001:db8::1", 2222, 0, 0), "[2001:db8::1]:2222"),
        (None, "unknown"),
        ((), "unknown"),
    ],
)
def test_endpoint_of_describes_peer_address(peername, expected):
    connection = SimpleNamespace(get_extra_info=Mock(return_value=peername))

    assert endpoint_of(connection) == expected
    connection.get_extra_info.assert_called_once_with("peername")
