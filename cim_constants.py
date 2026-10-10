"""项目共享的基础路径、测试默认值和会话数据键"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    # 仅供类型检查, 没有实际导入, 避免与导入本模块的 cim_config 形成循环依赖
    from cim_config import ProjectConfig


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "cim_config.yaml"

LOG_ENV_DEFAULTS = {
    "LOG_LEVEL": "DEBUG",
    "LOG_TO_FILE": "0",
    "LOG_INDENT_STEP": "4",
}

# 首次连接时允许自动记录 SSH 主机公钥的 IP 白名单；支持单个 IPv4/IPv6 或 CIDR
TEST_IP_WHITELIST: tuple[str, ...] = ("10.27.17.0/24",)

# stash 的写入方和读取方必须共用同一个键对象, 因此只在这里创建一次
PROJECT_CONFIG_KEY: pytest.StashKey[ProjectConfig] = pytest.StashKey()
