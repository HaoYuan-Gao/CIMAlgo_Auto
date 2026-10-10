"""
通过远程 ``flock`` 为测试设备提供协作式独占锁

命令行用法：
    python -m utils.device_lock 10.27.17.45 --case test_conv --user root --trusted-ips 10.27.17.0/24

代码中使用：
    connection = await SSHConnector(...).connect("root", ip, password=password)
    async with connection:
        async with RemoteDeviceLock(connection, case="test_conv"):
            ...

保持命令运行即可持有锁，按 Ctrl+C 释放。其他测试必须使用同一把锁, 否则该机制无法阻止它们直接操作设备
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import shlex
import socket
import sys
from datetime import datetime
from pathlib import Path

import asyncssh
from utils.ssh_connection import SSHConnector, endpoint_of


LOCK_FILE = "/run/lock/bw01-cim-test.lock"
OWNER_FILE = "/run/lock/bw01-cim-test.owner"
BUSY_EXIT_STATUS = 75

_REMOTE_LOCK_SCRIPT = f"""
set -eu

lock_file={shlex.quote(LOCK_FILE)}
owner_file={shlex.quote(OWNER_FILE)}

# 注意: umask 只影响新建文件，不修改已有文件权限，也不负责设置文件所属组
# 007: 同组用户可读写, 默认登录用户为 root 
umask 007

# command -v flock: 查找命令 flock 是否存在
if ! command -v flock >/dev/null 2>&1; then
    echo ERROR
    echo '远程设备未安装 flock' >&2
    exit 69
fi

# 9: 文件描述符编号
# <>：以 "可读且可写" 方式打开文件, 若文件不存在则创建
exec 9<>"$lock_file"

# flock: 文件锁命令, 加锁成功, flock 返回 0
# -n: 非阻塞, 锁被占用时立即失败
if ! flock -n 9; then
    echo BUSY

    # 2>/dev/null: 丢弃标准错误 (0: 标准输入, 1: 标准输出, 2: 标准错误)
    # || true: 当 cat 命令失败时, 执行 true
    cat "$owner_file" 2>/dev/null || true
    exit {BUSY_EXIT_STATUS}
fi

# 第一个 Shell 参数写入 owner 文件
printf '%s\n' "$1" >"$owner_file"
echo LOCKED

# trap: 注册 "退出或收到信号时执行的操作"，类似清理回调
# 
# EXIT: Shell 退出时触发
# : ：什么都不做的空命令
# > "$owner_file"：清空 owner 文件
# 单引号保证命令在退出时才执行
trap ': >"$owner_file"' EXIT

# cat 不带文件参数时，不读取文件，而是持续读取标准输入
# stdin 保持打开期间，当前 shell 和文件描述符 9 会一直持有锁
# Python 发送 EOF 后, cat 退出, Shell 随后退出并释放锁
cat >/dev/null
""".strip()


class DeviceLockError(RuntimeError):
    """远程设备锁无法建立或维持"""


class DeviceBusyError(DeviceLockError):
    """设备锁已被其他测试持有"""

    def __init__(self, host: str, lock_info: str):
        self.host = host
        self.lock_info = lock_info
        try:
            detail = json.dumps(json.loads(lock_info), ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            detail = lock_info or "占用者信息不可用"
        super().__init__(f"设备 {host} 已被占用，当前占用信息:\n{detail}")


def make_owner(case: str, owner: str | None = None) -> str:
    """生成写入远程设备的单行 JSON 锁信息"""
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


class RemoteDeviceLock:
    """在调用方提供的 SSH 连接上, 用一条专用持锁进程占有设备端文件锁; 连接的关闭由调用方负责"""

    def __init__(
        self,
        connection,
        *,
        case: str,
        owner: str | None = None,
        acquire_timeout: float = 15,
    ):
        if connection is None:
            raise ValueError("connection 必须是 SSHConnector.connect 返回的连接")
        if not isinstance(case, str) or not case.strip():
            raise ValueError("case 必须是非空字符串")
        if owner is not None and (not isinstance(owner, str) or not owner.strip()):
            raise ValueError("owner 必须是非空字符串")
        if acquire_timeout <= 0:
            raise ValueError("acquire_timeout 必须大于 0")

        self.connection = connection
        self.host = endpoint_of(connection)
        self.case = case.strip()
        self.owner = owner.strip() if owner is not None else getpass.getuser()
        self.lock_info = make_owner(self.case, self.owner)
        self.acquire_timeout = acquire_timeout
        self._process = None

    @property
    def acquired(self) -> bool:
        return self._process is not None

    async def acquire(self):
        """原子获取设备锁；设备繁忙时抛出 DeviceBusyError"""
        if self.acquired:
            raise DeviceLockError(f"设备 {self.host} 的锁已由当前实例持有")

        command = f"sh -c {shlex.quote(_REMOTE_LOCK_SCRIPT)} device-lock {shlex.quote(self.lock_info)}"
        process = await self.connection.create_process(command)
        try:
            status = await asyncio.wait_for(process.stdout.readline(), timeout=self.acquire_timeout)
            if status.rstrip("\r\n") == "LOCKED":
                self._process = process
                return self

            stdout, stderr = await asyncio.gather(process.stdout.read(), process.stderr.read())
            await process.wait()
            if status.rstrip("\r\n") == "BUSY":
                raise DeviceBusyError(self.host, stdout.strip())
            detail = stderr.strip() or stdout.strip() or "远程持锁进程意外退出"
            raise DeviceLockError(f"设备 {self.host} 加锁失败: {detail}")
        except BaseException:
            # 只结束未成功的持锁进程, 连接继续归调用方使用
            process.close()
            await process.wait_closed()
            raise

    async def release(self):
        """结束持锁进程并释放锁; 重复调用是安全的, 不关闭连接"""
        process, self._process = self._process, None
        if process is None:
            return

        process.stdin.write_eof()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            process.terminate()
            await process.wait()

    async def __aenter__(self):
        return await self.acquire()

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="通过远程 flock 独占一台测试设备，按 Ctrl+C 释放")
    parser.add_argument("host", help="设备 IPv4/IPv6 地址")
    parser.add_argument("--case", required=True, help="当前测试 case 名称")
    parser.add_argument("--owner", help="设备占用者名称，默认使用本机登录用户名")
    parser.add_argument("--port", type=int, default=22, help="SSH 端口，默认 22")
    parser.add_argument("--user", default="root", help="SSH 用户名，默认 root")
    parser.add_argument("--identity", type=Path, help="SSH 私钥路径")
    parser.add_argument(
        "--trusted-ips",
        nargs="+",
        default=[],
        metavar="IP/CIDR",
        help="首次主机公钥白名单, 支持 IPv4/IPv6 地址或 CIDR; 已有公钥不匹配时仍拒绝",
    )

    args = parser.parse_args()
    password = getpass.getpass("SSH 密码: ") if args.identity is None else None

    async def hold_until_interrupted():
        ssh = SSHConnector(
            identity=args.identity,
            trusted_ips=args.trusted_ips,
        )
        connection = await ssh.connect(args.user, args.host, port=args.port, password=password)
        async with connection:
            async with RemoteDeviceLock(
                connection,
                case=args.case,
                owner=args.owner,
            ) as lock:
                print(f"设备 {lock.host} 已加锁: {lock.lock_info}", flush=True)
                print("保持本进程运行以持有锁，按 Ctrl+C 释放", flush=True)
                await asyncio.Event().wait()

    try:
        # 执行且阻塞，直到收到 KeyboardInterrupt 或 DeviceBusyError
        asyncio.run(hold_until_interrupted())
    except DeviceBusyError as error:
        print(error, file=sys.stderr)
        sys.exit(BUSY_EXIT_STATUS)
    except (OSError, asyncssh.Error, DeviceLockError, ValueError) as error:
        print(f"设备加锁失败: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n设备锁已释放", file=sys.stderr)
        sys.exit(130)
