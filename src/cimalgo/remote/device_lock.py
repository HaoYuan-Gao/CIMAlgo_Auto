"""
通过远程 flock 为测试设备提供协作式独占锁

RemoteDeviceLock 在一条已建立的 SSHConnection 上启动持锁进程; LockedDevice 把建连和加锁合在一起,
是 pytest fixture 和命令行共用的入口。

命令行: cim-device-lock --ip 10.27.17.45 --case test_conv [--trusted-ips 10.27.17.0/24]
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shlex
import socket
import sys
import threading
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Self

import paramiko

from cimalgo.logging import get_logger

from .ssh import SSHConnection, SSHConnector, _collect_output

log = get_logger(__name__)

LOCK_FILE = "/run/lock/cim-test.lock"
OWNER_FILE = "/run/lock/cim-test.owner"
BUSY_EXIT_STATUS = 75

# 持锁脚本: 拿到 flock 后输出 LOCKED 并阻塞在 stdin 上; 本地关闭 stdin 即释放
_REMOTE_LOCK_SCRIPT = f"""
set -eu
lock_file={shlex.quote(LOCK_FILE)}
owner_file={shlex.quote(OWNER_FILE)}
umask 007
if ! command -v flock >/dev/null 2>&1; then
    echo ERROR
    echo '远程设备未安装 flock' >&2
    exit 69
fi
exec 9<>"$lock_file"
if ! flock -n 9; then
    echo BUSY
    cat "$owner_file" 2>/dev/null || true
    exit {BUSY_EXIT_STATUS}
fi
printf '%s\\n' "$1" >"$owner_file"
echo LOCKED
trap ': >"$owner_file"' EXIT
cat >/dev/null
""".strip()


class DeviceLockError(RuntimeError):
    """设备锁无法建立或维持"""


class DeviceBusyError(DeviceLockError):
    """设备已被其他测试持有"""

    def __init__(self, endpoint: str, lock_info: str):
        self.endpoint = endpoint
        self.lock_info = lock_info
        try:
            detail = json.dumps(json.loads(lock_info), ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            detail = lock_info or "占用者信息不可用"
        super().__init__(f"设备 {endpoint} 已被占用, 当前占用信息:\n{detail}")


def make_owner(case: str, owner: str | None = None) -> str:
    return json.dumps(
        {
            "owner": owner if owner is not None else getpass.getuser(),
            "case": case,
            "runner_host": socket.gethostname(),
            "pid": os.getpid(),
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _read_line(channel: paramiko.Channel, timeout: float) -> tuple[str, bytes]:
    """读到第一个换行为止; 返回 (该行, 多读出来的字节), 后者要交给后续读取者, 否则会丢数据"""
    channel.settimeout(timeout)
    buffer = b""
    while b"\n" not in buffer:
        # 每次最多读取 256 字节
        chunk = channel.recv(256)
        if not chunk:
            break
        buffer += chunk
    line, _, rest = buffer.partition(b"\n")
    # 输出包含非法字符不报错，而是替换为 U+FFFD, 并删除结尾的 \r
    return line.decode("utf-8", errors="replace").rstrip("\r"), rest


def _drain(channel: paramiko.Channel, *, stdout_prefix: bytes = b"") -> tuple[str, str]:
    stdout, stderr = stdout_prefix, b""
    while True:
        chunk = channel.recv(4096)
        if not chunk:
            break
        stdout += chunk
    while True:
        chunk = channel.recv_stderr(4096)
        if not chunk:
            break
        stderr += chunk
    return stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")


class RemoteDeviceLock:
    """在调用方提供的连接上持有设备锁; 不关闭连接"""

    def __init__(self, connection: SSHConnection, *, case: str, owner: str | None = None, acquire_timeout: float = 15):
        if not isinstance(case, str) or not case.strip():
            raise ValueError("case 必须是非空字符串")
        if owner is not None and (not isinstance(owner, str) or not owner.strip()):
            raise ValueError("owner 必须是非空字符串")
        if acquire_timeout <= 0:
            raise ValueError("acquire_timeout 必须大于 0")
        self.connection = connection
        self.case = case.strip()
        self.owner = owner.strip() if owner is not None else getpass.getuser()
        self.lock_info = make_owner(self.case, self.owner)
        self.acquire_timeout = acquire_timeout
        self._channel: paramiko.Channel | None = None
        self._guard = threading.Lock()

    @property
    def acquired(self) -> bool:
        return self._channel is not None

    def acquire(self) -> Self:
        # 防止当前进程的多个线程同时调用同一个 RemoteDeviceLock 实例
        with self._guard:
            if self.acquired:
                raise DeviceLockError(f"设备 {self.connection.endpoint} 的锁已由当前实例持有")
            command = f"sh -c {shlex.quote(_REMOTE_LOCK_SCRIPT)} device-lock {shlex.quote(self.lock_info)}"
            channel = self.connection.open_process(command)
            try:
                status, rest = _read_line(channel, self.acquire_timeout)
                if status == "LOCKED":
                    self._channel = channel
                    log.info("已锁定设备 %s (case=%s)", self.connection.endpoint, self.case)
                    return self
                stdout, stderr = _drain(channel, stdout_prefix=rest)
                if status == "BUSY":
                    raise DeviceBusyError(self.connection.endpoint, stdout.strip())
                detail = stderr.strip() or stdout.strip() or "远程持锁进程意外退出"
                raise DeviceLockError(f"设备 {self.connection.endpoint} 加锁失败: {detail}")
            except TimeoutError:
                channel.close()
                raise DeviceLockError(f"设备 {self.connection.endpoint} 加锁超时") from None
            except BaseException:
                channel.close()
                raise

    def release(self) -> None:
        with self._guard:
            channel, self._channel = self._channel, None
        if channel is None:
            return
        try:
            channel.shutdown_write()
            try:
                _collect_output(channel, timeout=5)
            except TimeoutError:
                log.warning("持锁进程未在 5 秒内退出, 强制关闭通道")
        finally:
            channel.close()
        log.info("已释放设备 %s", self.connection.endpoint)

    def __enter__(self) -> Self:
        return self.acquire()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.release()


class LockedDevice:
    """连接 + 锁的组合; 退出时先释放锁再关闭连接"""

    def __init__(self, connection: SSHConnection, lock: RemoteDeviceLock):
        self.connection = connection
        self.lock = lock

    @classmethod
    def open(
        cls,
        connector: SSHConnector,
        *,
        username: str,
        host: str,
        port: int = 22,
        password: str | None = None,
        case: str,
        owner: str | None = None,
        acquire_timeout: float = 15,
    ) -> LockedDevice:
        connection = connector.connect(username, host, port=port, password=password)
        try:
            lock = RemoteDeviceLock(connection, case=case, owner=owner, acquire_timeout=acquire_timeout).acquire()
        except BaseException:
            connection.close()
            raise
        return cls(connection, lock)

    def close(self) -> None:
        try:
            self.lock.release()
        finally:
            self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="通过远程 flock 独占一台测试设备, 按 Ctrl+C 释放")
    parser.add_argument("--ip", required=True, help="设备 IPv4/IPv6 地址")
    parser.add_argument("--case", required=True, help="当前测试 case 名称")
    parser.add_argument("--owner", help="占用者名称, 默认本机登录用户")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--username", default="root")
    parser.add_argument("--identity", type=Path, help="SSH 私钥; 不指定则提示输入密码")
    parser.add_argument("--trusted-ips", nargs="+", default=[], metavar="IP/CIDR")
    args = parser.parse_args(argv)

    password = getpass.getpass("SSH 密码: ") if args.identity is None else None
    connector = SSHConnector(identity=args.identity, trusted_ips=args.trusted_ips)
    try:
        with LockedDevice.open(
            connector,
            username=args.username,
            host=args.ip,
            port=args.port,
            password=password,
            case=args.case,
            owner=args.owner,
        ) as device:
            print(f"设备 {device.connection.endpoint} 已加锁: {device.lock.lock_info}", flush=True)
            print("保持本进程运行以持有锁, 按 Ctrl+C 释放", flush=True)
            threading.Event().wait()
    except DeviceBusyError as error:
        print(error, file=sys.stderr)
        return BUSY_EXIT_STATUS
    except (OSError, DeviceLockError, ValueError) as error:
        print(f"设备加锁失败: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n设备锁已释放", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
