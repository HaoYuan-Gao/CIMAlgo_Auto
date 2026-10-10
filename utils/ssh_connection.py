"""统一创建 AsyncSSH connection, 并处理白名单设备的首次主机公钥信任"""

from __future__ import annotations

import asyncio
import fcntl
import ipaddress
from pathlib import Path

import asyncssh
from utils.deprecated import deprecated


def endpoint_of(connection) -> str:
    """用对端地址描述连接, IPv6 写作 [IPv6]:port; 取不到地址时返回 unknown"""
    peer = connection.get_extra_info("peername")
    if not peer or len(peer) < 2:
        return "unknown"
    host, port = str(peer[0]), peer[1]
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


@deprecated(reason="废弃 ip:port 格式作为函数参数的方案")
def parse_endpoint(endpoint):
    """解析 IP、IPv4:port 或 [IPv6]:port; 裸 IPv6 整体视为地址, 端口默认 22"""
    if not isinstance(endpoint, str):
        raise TypeError("设备地址必须是字符串: IP、IPv4:port 或 [IPv6]:port")
    port_text = "22"
    if endpoint.startswith("["):
        ip, closing, suffix = endpoint[1:].partition("]")
        if not closing or (suffix and not suffix.startswith(":")):
            raise ValueError(f"IPv6 地址格式应为 [IPv6]:port: {endpoint}")
        if ipaddress.ip_address(ip).version != 6:
            raise ValueError(f"方括号仅用于 IPv6 地址: {endpoint}")
        if suffix:
            port_text = suffix[1:]
    elif endpoint.count(":") == 1:
        ip, port_text = endpoint.split(":")
    else:
        ip = endpoint
    if not port_text.isascii() or not port_text.isdecimal():
        raise ValueError(f"端口必须是 1 到 65535 之间的整数: {endpoint}")
    port = int(port_text)
    if not 1 <= port <= 65535:
        raise ValueError(f"端口必须是 1 到 65535 之间的整数: {endpoint}")
    return str(ipaddress.ip_address(ip)), port


class SSHConnector:
    """保存与账号无关的公共 SSH 参数和首次信任白名单; 用户名在 connect 时按设备指定"""

    def __init__(
        self,
        *,
        connect_timeout: int | float = 15,
        trusted_ips=(),
        identity: str | Path | None = None,
        known_hosts: str | Path | None = None,
    ):
        if not isinstance(connect_timeout, (int, float)) or isinstance(connect_timeout, bool) or connect_timeout <= 0:
            raise ValueError("connect_timeout 必须大于 0")

        self.connect_timeout = connect_timeout
        self.identity = None
        if identity is not None:
            self.identity = Path(identity).expanduser().resolve()
            if not self.identity.is_file():
                raise FileNotFoundError(self.identity)

        # 单个 IP 自动转为 /32 或 /128; CIDR (无类别域间路由) 必须填写网络地址, 不自动扩大信任范围
        # CIDR:
        #   1. /xx 表示前 xx 位为网络地址, 以 ipv4 为例, /24 表示前 24 位为网络地址, 后 8 位为主机地址(自主分配)
        #   2. ipv4 一共 32 位, 所以 /32 表示整个 ipv4 地址, 即单个 ip 地址
        #
        # 示例：
        #   trusted_ips={
        #       "10.27.17.45",             # 单个 IPv4
        #       "10.27.17.45/32",          # 单个 IPv4
        #       "10.27.18.0/24",           # IPv4 网段
        #       "2001:db8::1",             # 单个 IPv6
        #       "2001:db8:1234::/64",      # IPv6 网段
        #   }
        self.trusted_networks = set()
        for value in trusted_ips or ():
            if not isinstance(value, str):
                raise ValueError("trusted_ips 中每项必须是 IP 或 CIDR 字符串")
            self.trusted_networks.add(ipaddress.ip_network(value, strict=True))
        # 白名单设备使用此文件读取和保存主机公钥, 不关闭主机密钥校验
        self.known_hosts = (
            Path(known_hosts).expanduser() if known_hosts is not None else Path.home() / ".ssh" / "known_hosts"
        )

    def make_options(self, username: str) -> dict:
        """返回不会影响内部状态的 AsyncSSH 参数副本, 用户名由本次连接指定"""
        if not isinstance(username, str) or not username.strip():
            raise ValueError("username 必须是非空字符串")
        options = {
            "username": username.strip(),
            "connect_timeout": self.connect_timeout,
        }
        if self.identity is not None:
            options["client_keys"] = [str(self.identity)]
        return options

    def is_trusted_ip(self, host) -> bool:
        """只匹配当前设备, 不展开网段, 也不扫描网络; 主机名一律不自动信任"""
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return False
        # IPv6 规则若指定了 %eth0 等网络接口, 不扩展信任到其他网络接口上的同名地址, 例如本机有两个网卡:
        #   fe80::1%eth0    通过本机 eth0 接口访问的设备
        #   fe80::1%eth1    通过本机 eth1 接口访问的设备
        # 若 trusted_ips = {"fe80::1%eth0"}, 就只允许自动信任 eth0 上的这个地址
        scope = getattr(address, "scope_id", None)
        return any(
            address in network and getattr(network.network_address, "scope_id", None) in (None, scope)
            for network in self.trusted_networks
        )

    async def prepare_known_hosts(self, host, port: int = 22):
        """对白名单设备首次连接时自动回答 yes, 绝不替换已有记录; 其他主机原样返回已有记录"""
        try:
            known_hosts = asyncssh.read_known_hosts(str(self.known_hosts))
        except FileNotFoundError:
            known_hosts = asyncssh.import_known_hosts("")
        if not self.is_trusted_ip(host):
            return known_hosts

        ip = str(ipaddress.ip_address(host))
        address = ip if port == 22 else f"[{ip}]:{port}"
        # 将完整地址作为 host 匹配, addr 留空, 避免回退到 22 端口的记录
        # 包括普通公钥、CA 和撤销记录; 有记录时交给正常校验, 不自动接受新密钥
        if any(known_hosts.match(address, "", None)):
            return known_hosts

        # 只获取主机公钥, 不发送登录密码; 正式连接仍要与保存的公钥匹配
        key = await asyncio.wait_for(asyncssh.get_server_host_key(ip, port=port), timeout=15)
        if key is None:
            raise ValueError(f"无法获取 {address} 的 SSH 主机公钥")
        self.known_hosts.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.known_hosts.touch(mode=0o600, exist_ok=True)
        with self.known_hosts.open("a+", encoding="utf-8") as file:
            # 获取公钥期间文件可能被其他任务更新; 加锁后重新检查, 避免覆盖或重复添加
            fcntl.flock(file, fcntl.LOCK_EX)
            file.seek(0)
            known_hosts = asyncssh.import_known_hosts(file.read())
            if any(known_hosts.match(address, "", None)):
                return known_hosts
            entry = f"\n{address} {key.export_public_key('openssh').decode('ascii')}"
            file.write(entry)
            known_hosts.load(entry)
        print(f"[{address}] 首次信任主机公钥 {key.get_fingerprint()}, 已保存到 {self.known_hosts}", flush=True)
        return known_hosts

    async def connect(self, username: str, host: str, *, port: int = 22, password: str | None = None):
        """以指定用户名创建并返回 connection, 关闭及生命周期由业务调用方负责"""
        if not isinstance(host, str) or not host.strip():
            raise ValueError("host 必须是非空字符串")
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            raise ValueError("port 必须是 1 到 65535 之间的整数")
        if password is not None and not isinstance(password, str):
            raise ValueError("password 必须是字符串或 None")

        options = self.make_options(username)
        try:
            host = str(ipaddress.ip_address(host.strip()))
        except ValueError:
            raise ValueError(f"host 必须是 IPv4/IPv6 地址: {host!r}") from None

        # 白名单 IP 先准备 known_hosts 再连接; 非白名单主机保留 AsyncSSH 的默认主机密钥校验
        if self.is_trusted_ip(host):
            options["known_hosts"] = await self.prepare_known_hosts(host, port)
        return await asyncssh.connect(host, port=port, password=password, **options)
