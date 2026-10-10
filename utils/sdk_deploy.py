"""
SDKDeployer 在一条已建立的 SSH 连接上部署一个 deb 包

命令行 (仅用于功能验证, 连接参数显式给出):
    python -m utils.sdk_deploy --ip 10.27.17.45 --password xxx --deb /path/to/sdk.deb --trusted-ips 10.27.17.0/24

代码中部署:
    connection = await SSHConnector(trusted_ips=[...]).connect("root", ip, password=password)
    async with connection:
        output = await SDKDeployer(deb_path).deploy(connection)

一个 case 只对应一台设备和一份 SDK, 因此这里不提供批量部署; 多台设备由调用方各自建连后分别调用 deploy

执行方式区别：
| 命令               | 查找方式           | 包上下文 | 相对导入 |
|--------------------|------------------ |----------|----------|
| python xx.py       | 直接执行文件       | 无       | 通常失败 |
| python -m utils.xx | 从模块搜索路径查找 | utils    | 正常工作 |


TODO: 如果 bw01d_ctrl 或者 bw01_ctrl 功能稳定, 可以直接使用它们的方法部署 SDK
"""

import argparse
import asyncio
import hashlib
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import asyncssh

from utils.ssh_connection import SSHConnector, endpoint_of

if TYPE_CHECKING:
    from cim_config import SdkConfig


class SDKDeployer:
    """一个实例对应一个 deb 包; deploy 使用调用方提供的连接, 不自己建连也不关闭连接"""

    def __init__(self, deb_path, *, timeout=600):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout 必须大于 0")

        self.local_deb = Path(deb_path).expanduser().resolve()
        if not self.local_deb.is_file():
            raise FileNotFoundError(self.local_deb)
        self.remote_deb = f"/opt/{self.local_deb.name}"
        self.timeout = timeout

        # 初始化时计算本地 deb 文件的 SHA-256 校验值, 多次部署共用
        with self.local_deb.open("rb") as package:
            self.sha256 = hashlib.file_digest(package, "sha256").hexdigest()

    @classmethod
    def for_sdk(cls, sdk: "SdkConfig") -> "SDKDeployer":
        """从当前 SDK 安装目录中选择唯一的设备安装包。"""
        # TODO: 如果命名发生变化, 这里需要修改
        package_name = f"{sdk.chip}-sdk_{sdk.version}_arm64.deb"
        packages = sorted((sdk.install_dir / "Agent").glob(f"**/debs/{package_name}"))
        if not packages:
            raise FileNotFoundError(f"SDK 设备安装包不存在: {sdk.install_dir}/Agent/**/debs/{package_name}")
        if len(packages) > 1:
            raise ValueError(f"找到多个 SDK 设备安装包 {package_name}: {packages}")
        return cls(packages[0])

    async def _verify_checksum(self, connection):
        """上传后比对整个文件的 SHA-256; 不一致时阻止删除和安装"""
        result = await connection.run(
            f"sha256sum < {shlex.quote(self.remote_deb)}",
            check=True,
        )
        checksum = result.stdout.split(maxsplit=1)
        if not checksum or checksum[0] != self.sha256:
            raise ValueError("SHA-256 校验失败: 远程 deb 与本地文件不一致")

    async def deploy(self, connection):
        """
        在已建立的连接上部署一台设备, 返回远程 dpkg -i 的标准输出

        整个流程受 self.timeout 约束, 超时抛出 TimeoutError, 此时远程命令可能仍在运行
        任一步失败直接抛出异常, 不继续删除或安装; 连接的关闭由调用方负责
        """
        endpoint = endpoint_of(connection)
        quoted_deb = shlex.quote(self.remote_deb)

        async with asyncio.timeout(self.timeout):
            # 1. 检查设备架构; 删除和安装需要的权限由远程命令自身报错
            result = await connection.run("dpkg --print-architecture", check=True)
            if result.stdout.strip() != "arm64":
                raise ValueError(f"目标设备 {endpoint} 不是 arm64")

            # 2. 上传后校验完整性和包类型; 失败时不会删除旧目录
            print(f"[{endpoint}] 上传 {self.local_deb.name}", flush=True)
            async with connection.start_sftp_client() as sftp:
                await sftp.put(str(self.local_deb), self.remote_deb)
            await self._verify_checksum(connection)
            result = await connection.run(
                "dpkg-deb --show --showformat='${Architecture}' " + quoted_deb,
                check=True,
            )
            if result.stdout.strip() != "arm64":
                raise ValueError("安装包必须是 arm64 架构")

            # 3. 删除旧目录, 然后安装; 只使用固定删除目标和远程 deb 路径
            print(f"[{endpoint}] 删除旧目录并安装", flush=True)
            await connection.run("rm -rf --one-file-system -- /opt/bw01d /opt/bw01", check=True)
            result = await connection.run(f"dpkg -i -- {quoted_deb}", check=True)
        return result.stdout


if __name__ == "__main__":
    ### just for function test ###
    parser = argparse.ArgumentParser(description="通过 AsyncSSH 向一台 arm64 设备部署 SDK")
    parser.add_argument("--ip", required=True, help="设备 IPv4/IPv6 地址")
    parser.add_argument("--port", type=int, default=22, help="SSH 端口, 默认 22")
    parser.add_argument("--username", type=str, default="root", help="SSH 用户名, 默认 root")
    parser.add_argument("--password", required=True, help="SSH 密码")
    parser.add_argument("--deb", type=Path, required=True, help="本地 deb 文件")
    parser.add_argument("--timeout", type=int, default=600, help="部署超时秒数, 默认 600")
    parser.add_argument(
        "--trusted-ips",
        nargs="+",
        default=[],
        metavar="IP/CIDR",
        help="首次主机公钥白名单, 支持 IPv4/IPv6 地址或 CIDR, 不支持 *; 已有公钥不匹配时仍拒绝",
    )
    args = parser.parse_args()

    target = f"{args.username}@[{args.ip}]:{args.port}" if ":" in args.ip else f"{args.username}@{args.ip}:{args.port}"

    async def connect_and_deploy():
        ssh = SSHConnector(connect_timeout=15, trusted_ips=args.trusted_ips)
        deployer = SDKDeployer(args.deb, timeout=args.timeout)
        connection = await ssh.connect(args.username, args.ip, port=args.port, password=args.password)
        async with connection:
            return await deployer.deploy(connection)

    try:
        print(f"目标设备: {target}")
        output = asyncio.run(connect_and_deploy())
        # 打印远程 dpkg -i 安装命令的标准输出
        for line in output.splitlines():
            print(f"[{target}] {line}", flush=True)
        print(f"\n部署结果: {target}: 成功")
        sys.exit(0)
    except TimeoutError:
        print("部署失败: 部署超时, 远程命令可能仍在运行, 请检查设备状态", file=sys.stderr)
        sys.exit(1)
    except (OSError, ValueError, asyncssh.Error) as error:
        # ProcessError 的 stderr 比异常文本更能说明远程命令为何失败
        message = (getattr(error, "stderr", "") or str(error)).strip() or "部署失败"
        print(f"部署失败: {message}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n部署已中断, 请检查设备的远程命令是否仍在运行", file=sys.stderr)
        sys.exit(130)
