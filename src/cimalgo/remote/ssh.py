"""
基于 paramiko 的同步 SSH 连接, 并处理白名单设备的首次主机公钥信任

SSHConnector 保存与账号无关的公共参数 (超时、私钥、白名单、known_hosts 路径);
每次 connect 以指定用户名返回一个 SSHConnection, 其生命周期由调用方用 with 管理。
"""

from __future__ import annotations

import fcntl
import ipaddress
import math
import shlex
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import paramiko

from cimalgo.logging import get_logger

log = get_logger(__name__)


class CommandError(RuntimeError):
    """远程命令退出码非 0"""

    def __init__(self, command: str, result: CommandResult):
        self.command = command
        self.result = result
        detail = result.stderr.strip() or result.stdout.strip() or "无输出"
        super().__init__(f"远程命令失败 (exit {result.exit_status}): {command}: {detail}")


@dataclass(frozen=True, slots=True)
class CommandResult:
    exit_status: int
    stdout: str
    stderr: str


def _collect_output(channel: paramiko.Channel, *, timeout: float | None = None) -> CommandResult:
    """交替读取两路输出, 等待 EOF 和退出状态; channel 的 socket timeout 不限制退出等待"""
    deadline = None if timeout is None else time.monotonic() + timeout
    stdout, stderr = bytearray(), bytearray()
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("等待远程命令退出超时")
        received = False
        if channel.recv_ready():
            stdout.extend(channel.recv(65536))
            received = True
        if channel.recv_stderr_ready():
            stderr.extend(channel.recv_stderr(65536))
            received = True
        if (
            channel.exit_status_ready()
            and (channel.eof_received or channel.closed)
            and not channel.recv_ready()
            and not channel.recv_stderr_ready()
        ):
            return CommandResult(
                exit_status=channel.recv_exit_status(),
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
            )
        if not received:
            time.sleep(0.01)


def endpoint_of(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


class _TrustedHostPolicy(paramiko.MissingHostKeyPolicy):
    """首次连接白名单设备时自动写入 known_hosts; 其他主机沿用拒绝策略。已有记录不匹配时 paramiko 仍会拒绝"""

    def __init__(self, connector: SSHConnector, host: str, port: int):
        self._connector = connector
        self._host = host
        self._port = port

    def missing_host_key(self, client: paramiko.SSHClient, hostname: str, key: paramiko.PKey) -> None:
        if not self._connector.is_trusted_ip(self._host):
            raise paramiko.SSHException(
                f"未知主机 {hostname}, 且不在首次信任白名单内; 请先手动确认指纹并加入 known_hosts"
            )
        self._connector.remember_host_key(self._host, self._port, key)
        client.get_host_keys().add(hostname, key.get_name(), key)


class SSHConnector:
    """保存公共 SSH 参数和首次信任白名单; 用户名在 connect 时按设备指定"""

    def __init__(
        self,
        *,
        connect_timeout: float = 15,
        trusted_ips: Iterable[str] = (),
        identity: str | Path | None = None,
        known_hosts: str | Path | None = None,
    ):
        if isinstance(connect_timeout, bool) or not isinstance(connect_timeout, (int, float)) or connect_timeout <= 0:
            raise ValueError("connect_timeout 必须大于 0")
        self.connect_timeout = float(connect_timeout)

        self.identity: Path | None = None
        if identity is not None:
            self.identity = Path(identity).expanduser().resolve()
            if not self.identity.is_file():
                raise FileNotFoundError(self.identity)

        # 单个 IP 自动转为 /32 或 /128; CIDR 必须填网络地址, 不自动扩大信任范围
        self.trusted_networks: set[ipaddress.IPv4Network | ipaddress.IPv6Network] = set()
        for value in trusted_ips:
            if not isinstance(value, str):
                raise ValueError("trusted_ips 中每项必须是 IP 或 CIDR 字符串")
            self.trusted_networks.add(ipaddress.ip_network(value, strict=True))

        self.known_hosts = (
            Path(known_hosts).expanduser() if known_hosts is not None else Path.home() / ".ssh" / "known_hosts"
        )

    def is_trusted_ip(self, host: str) -> bool:
        """只匹配当前设备, 不展开网段; 主机名一律不自动信任; 带 scope 的 IPv6 只匹配同一接口"""
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return False

        # scope_id 是 IPv6 地址的接口标识
        # ipaddress.ip_address("fe80::1%eth0").scope_id ---> "eth0"
        scope = getattr(address, "scope_id", None)
        return any(
            address in network and getattr(network.network_address, "scope_id", None) in (None, scope)
            for network in self.trusted_networks
        )

    def remember_host_key(self, host: str, port: int, key: paramiko.PKey) -> None:
        """追加到 known_hosts; 文件加锁后重新检查, 绝不替换已有记录"""
        entry_host = host if port == 22 else f"[{host}]:{port}"
        # 0o700: 目录只对当前用户可读写执行
        self.known_hosts.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # 0o600: 文件只对当前用户可读写
        self.known_hosts.touch(mode=0o600, exist_ok=True)
        with self.known_hosts.open("a+", encoding="utf-8") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            file.seek(0)
            existing = paramiko.HostKeys()
            existing.load(str(self.known_hosts))
            # 根据主机名查找所有类型公钥
            known = existing.lookup(entry_host)
            if known:
                # key.get_name(): 公钥算法类型
                expected = known.get(key.get_name())
                # 如果公钥不同, 或者根本没有保存过这个算法类型, 就拒绝连接
                if expected != key:
                    raise paramiko.BadHostKeyException(entry_host, key, expected or next(iter(known.values())))
                return
            file.write(f"{entry_host} {key.get_name()} {key.get_base64()}\n")
        log.warning("首次信任 %s 的主机公钥 %s, 已写入 %s", entry_host, key.get_fingerprint().hex(), self.known_hosts)

    def connect(self, username: str, host: str, *, port: int = 22, password: str | None = None) -> SSHConnection:
        """以指定用户名建立连接; host 必须是 IP 地址"""
        if not isinstance(username, str) or not username.strip():
            raise ValueError("username 必须是非空字符串")
        if not isinstance(host, str) or not host.strip():
            raise ValueError("host 必须是非空字符串")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("port 必须是 1 到 65535 之间的整数")
        if password is not None and not isinstance(password, str):
            raise ValueError("password 必须是字符串或 None")
        try:
            host = str(ipaddress.ip_address(host.strip()))
        except ValueError:
            raise ValueError(f"host 必须是 IPv4/IPv6 地址: {host!r}") from None

        client = paramiko.SSHClient()
        if self.known_hosts.is_file():
            client.load_host_keys(str(self.known_hosts))
        client.set_missing_host_key_policy(_TrustedHostPolicy(self, host, port))
        try:
            client.connect(
                host,
                port=port,
                username=username.strip(),
                password=password,
                key_filename=str(self.identity) if self.identity else None,
                timeout=self.connect_timeout,
                banner_timeout=self.connect_timeout,
                auth_timeout=self.connect_timeout,
                allow_agent=password is None,
                look_for_keys=password is None,
            )
        except (TimeoutError, OSError, paramiko.SSHException) as error:
            client.close()
            raise ConnectionError(f"连接 {username}@{endpoint_of(host, port)} 失败: {error}") from error
        log.debug("已连接 %s@%s", username, endpoint_of(host, port))
        return SSHConnection(client, host=host, port=port, username=username.strip())


class SSHConnection:
    """一条已建立的 SSH 连接; 提供 run / upload, 用 with 关闭"""

    def __init__(self, client: paramiko.SSHClient, *, host: str, port: int, username: str):
        self._client: paramiko.SSHClient | None = client
        self.host = host
        self.port = port
        self.username = username

    @property
    def endpoint(self) -> str:
        return endpoint_of(self.host, self.port)

    def __repr__(self) -> str:
        return f"<SSHConnection {self.username}@{self.endpoint}>"

    def _require(self) -> paramiko.SSHClient:
        if self._client is None:
            raise RuntimeError("SSH 连接已关闭")
        return self._client

    def run(self, command: str, *, check: bool = True, timeout: float | None = None) -> CommandResult:
        """执行命令并等待退出; check=True 时非 0 退出码抛 CommandError"""
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout 必须是大于 0 的有限数值")
        _stdin, stdout, _stderr = self._require().exec_command(command, timeout=timeout)
        channel = stdout.channel
        try:
            channel.shutdown_write()
            result = _collect_output(channel, timeout=timeout)
        finally:
            channel.close()
        if check and result.exit_status != 0:
            raise CommandError(command, result)
        return result

    def open_process(self, command: str) -> paramiko.Channel:
        """启动长期运行的远程进程, 返回 channel; 调用方负责读写和关闭"""
        transport = self._require().get_transport()
        if transport is None or not transport.is_active():
            raise RuntimeError(f"SSH 传输层已断开: {self.username}@{self.endpoint}")
        channel = transport.open_session()
        channel.exec_command(command)
        return channel

    def upload(self, local: str | Path, remote: str) -> None:
        with self._require().open_sftp() as sftp:
            sftp.put(str(local), remote)

    def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            client.close()
            log.debug("已关闭 %s@%s", self.username, self.endpoint)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()


def quote(value: str) -> str:
    return shlex.quote(value)
