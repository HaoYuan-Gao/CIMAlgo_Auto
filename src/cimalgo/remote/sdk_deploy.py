"""
在一条已建立的 SSH 连接上向设备部署 SDK deb 包

命令行: cim-deploy-sdk --ip 10.27.17.45 --password xxx --deb /path/to/sdk.deb [--trusted-ips 10.27.17.0/24]
"""

from __future__ import annotations

import argparse
import hashlib
import shlex
import sys
from pathlib import Path

from cimalgo.config import SdkConfig
from cimalgo.logging import get_logger

from .ssh import CommandError, SSHConnection, SSHConnector

log = get_logger(__name__)

REMOTE_INSTALL_DIRS = ("/opt/bw01d", "/opt/bw01")


class SDKDeployer:
    """一个实例对应一个 deb 包; deploy 使用调用方提供的连接"""

    def __init__(self, deb_path: str | Path, *, timeout: float = 600):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout 必须大于 0")
        self.local_deb = Path(deb_path).expanduser().resolve()
        if not self.local_deb.is_file():
            raise FileNotFoundError(self.local_deb)
        self.remote_deb = f"/opt/{self.local_deb.name}"
        self.timeout = float(timeout)
        with self.local_deb.open("rb") as package:
            self.sha256 = hashlib.file_digest(package, "sha256").hexdigest()

    @classmethod
    def for_sdk(cls, sdk: SdkConfig) -> SDKDeployer:
        """从 SDK 安装目录中找到唯一的设备端安装包"""
        package_name = f"{sdk.chip}-sdk_{sdk.version}_arm64.deb"
        packages = sorted((sdk.install_dir / "Agent").glob(f"**/debs/{package_name}"))
        if not packages:
            raise FileNotFoundError(f"SDK 设备安装包不存在: {sdk.install_dir}/Agent/**/debs/{package_name}")
        if len(packages) > 1:
            raise ValueError(f"找到多个 SDK 设备安装包 {package_name}: {packages}")
        return cls(packages[0])

    @log.trace()
    def deploy(self, connection: SSHConnection) -> str:
        """
        检查架构 -> 上传 -> 校验 SHA-256 和包架构 -> 删除旧目录 -> dpkg -i; 返回 dpkg 输出

        任一步失败直接抛异常, 不继续删除或安装; 连接由调用方关闭
        """
        quoted = shlex.quote(self.remote_deb)
        timeout = self.timeout

        result = connection.run("dpkg --print-architecture", timeout=timeout)
        if result.stdout.strip() != "arm64":
            raise ValueError(f"目标设备 {connection.endpoint} 不是 arm64")

        log.info("[%s] 上传 %s", connection.endpoint, self.local_deb.name)
        connection.upload(self.local_deb, self.remote_deb)
        checksum = connection.run(f"sha256sum < {quoted}", timeout=timeout).stdout.split(maxsplit=1)
        if not checksum or checksum[0] != self.sha256:
            raise ValueError("SHA-256 校验失败: 远程 deb 与本地文件不一致")
        architecture = connection.run(f"dpkg-deb --show --showformat='${{Architecture}}' {quoted}", timeout=timeout)
        if architecture.stdout.strip() != "arm64":
            raise ValueError("安装包必须是 arm64 架构")

        log.info("[%s] 删除旧目录并安装", connection.endpoint)
        connection.run(f"rm -rf --one-file-system -- {' '.join(REMOTE_INSTALL_DIRS)}", timeout=timeout)
        return connection.run(f"dpkg -i -- {quoted}", timeout=timeout).stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="向一台 arm64 设备部署 SDK")
    parser.add_argument("--ip", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--username", default="root")
    parser.add_argument("--password", required=True)
    parser.add_argument("--deb", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--trusted-ips", nargs="+", default=[], metavar="IP/CIDR")
    args = parser.parse_args(argv)

    try:
        deployer = SDKDeployer(args.deb, timeout=args.timeout)
        connector = SSHConnector(connect_timeout=15, trusted_ips=args.trusted_ips)
        with connector.connect(args.username, args.ip, port=args.port, password=args.password) as connection:
            output = deployer.deploy(connection)
        for line in output.splitlines():
            print(line)
        print("部署成功")
        return 0
    except (OSError, ValueError, CommandError) as error:
        print(f"部署失败: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n部署已中断, 请检查设备的远程命令是否仍在运行", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
