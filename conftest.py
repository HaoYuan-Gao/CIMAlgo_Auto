import os
from collections.abc import Mapping

import pytest

from cim_config import ProjectConfigError, load_project_config
from cim_constants import (
    DEVICE_OPS_KEY,
    LOCKED_DEVICE_KEY,
    LOG_ENV_DEFAULTS,
    PROJECT_CONFIG_KEY,
    PROJECT_ROOT,
    SSH_CONNECTOR_KEY,
    TEST_IP_WHITELIST,
)
from utils.device_lock import LockedDeviceConnection
from utils.device_ops import DeviceOps, DeviceTarget
from utils.sdk_deploy import SDKDeployer
from utils.set_env import SetEnv
from utils.ssh_connection import SSHConnector


def pytest_addoption(parser):
    group = parser.getgroup("project", "项目配置")
    group.addoption(
        "--sdk",
        action="store",
        default=None,
        metavar="NAME",
        help="指定 project.yaml 中的 SDK 配置名称",
    )
    group.addoption(
        "--download",
        action="store_true",
        help="在收集测试前下载、安装 SDK/工具链并同步 SDK wheel (本机 xdist 仅主进程执行)",
    )


def pytest_configure(config):
    """在测试模块导入前设置日志默认值"""
    for name, value in LOG_ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    """
    缓存项目配置。pytest_sessionstart 是 pytest 自动调用的钩子, 早于测试用例的收集和执行。xdist (pytest-xdist) 是并行测试插件。
    pytest -n 4 会启动 4 个 worker(进程) 执行测试, 主进程按需在收集测试、启动 xdist worker 前准备依赖, 各进程拥有独立的 session。
    tryfirst=True 让本实现优先于同一钩子的普通实现执行; xdist 启动的 worker 默认使用 trylast=True
    """
    config = session.config
    try:
        project_config = load_project_config(sdk_name=config.getoption("--sdk"))
    except ProjectConfigError as error:
        raise pytest.UsageError(str(error)) from error
    # 不论是否下载、是否为 worker，每个进程都缓存各自的会话数据
    session.stash[PROJECT_CONFIG_KEY] = project_config
    session.stash[SSH_CONNECTOR_KEY] = SSHConnector(trusted_ips=TEST_IP_WHITELIST)

    deployer = SetEnv.for_config(project_config)
    # xdist 只在 worker 的 config 上添加 workerinput 字典, 记录 workerid 等信息, 主进程没有此属性; 通过它识别并跳过 worker, 避免重复下载、解压
    if config.getoption("--download") and not hasattr(config, "workerinput"):
        # 本机和 xdist worker 共用安装目录和 Python 环境, 只在主进程执行准备工作
        deployer.prepare_packages()
        deployer.sync_wheels()

    # 每个进程都要设置导入路径; 普通测试和 xdist worker 也需要导入 SDK 模块
    deployer.add_import_paths()

    # 在测试收集前部署环境变量，并在 config 卸载时自动恢复
    monkeypatch = pytest.MonkeyPatch()
    config.add_cleanup(monkeypatch.undo)
    for name, value in deployer.build_environment().items():
        monkeypatch.setenv(name, value)


def _case_metadata(value):
    """从扁平化的 case 定义中读取显示信息"""
    if not isinstance(value, Mapping):
        return None

    case_name = value.get("case")
    tag = value.get("tag")
    if not case_name or not tag:
        return None
    return str(case_name), str(tag)


# pytest 内置钩子, 用于生成参数化测试的 ID, 晚于 pytest_sessionstart
def pytest_make_parametrize_id(config, val, argname):
    """使用 case 字段作为 case_context 的 pytest 参数 ID"""
    if argname == "case_context" and (metadata := _case_metadata(val)):
        return metadata[0]
    return None


def pytest_runtest_setup(item):
    """每个 case 执行前, 提前处理相关流程"""
    callspec = getattr(item, "callspec", None)
    if callspec is None or "case_context" not in callspec.params:
        return

    case_context = callspec.params["case_context"]
    metadata = _case_metadata(case_context)
    if metadata is None:
        return

    case_name, tag = metadata
    title = f"CASE: {case_name} | tag={tag}"
    # pytest 负责往终端打印测试进度/结果的插件
    terminal_reporter = item.config.pluginmanager.get_plugin("terminalreporter")
    if terminal_reporter is not None:
        terminal_reporter.write_sep("-", title, purple=True)
    else:
        print(title)

    # ----------------------------- setup -----------------------------
    # 1. 处理 fake case
    if tag == "fake":
        from utils.fake_data_gen import resolve_fake_case

        case_context = resolve_fake_case(case_context, base_dir=PROJECT_ROOT)

    device = case_context.get("device")
    if device is not None:
        ssh_connector = item.session.stash[SSH_CONNECTOR_KEY]
        # 2. 打开设备连接, 并获取锁
        locked_device = LockedDeviceConnection.open(
            ssh_connector,
            username=device["username"],
            host=device["ip"],
            port=device["port"],
            password=device.get("password"),
            case=case_context["case"],
        )

        device_ops = None
        try:
            sdk = item.session.stash[PROJECT_CONFIG_KEY].sdk
            # 3. 更新 SDK
            if device.get("update_sdk") is True:
                deployer = SDKDeployer.for_sdk(sdk)
                locked_device.run(deployer.deploy(locked_device.connection))

            # 4. 初始化设备: 下电上电、切换固件、配置物理 L2、逐个 HB 选层
            hard_blocks = device["hard_block"]
            sel_layers = device["sel_layer"]
            tsv_sel0 = device.get("tsv_sel0", 0)
            tsv_sel1 = device.get("tsv_sel1", 255)
            target = DeviceTarget(
                ip=device["ip"],
                bmc_ip=device.get("bmc_ip", None),
                grpc_port=device.get("grpc_port", 60000),
                evb_port=device.get("evb_port", 61006),
                server_port=device.get("server_port", 60001),
                is_cluster=device.get("is_cluster", False),
                test_stage=device.get("test_stage", "CP3"),
                driver=sdk.chip,
            )
            device_ops = DeviceOps.for_target(target)
            device_ops.power_cycle()
            device_ops.switch_firmware_version(sdk.version)
            device_ops.set_phy_l2(hard_blocks)

            for hb_id in sel_layers:
                device_ops.select_layer(
                    hb_id,
                    tsv_sel0=tsv_sel0,
                    tsv_sel1=tsv_sel1,
                )
        except BaseException:
            try:
                if device_ops is not None:
                    device_ops.close()
            finally:
                locked_device.close()
            raise
        item.stash[LOCKED_DEVICE_KEY] = locked_device
        item.stash[DEVICE_OPS_KEY] = device_ops


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """在测试和 fixture teardown 完成后关闭设备操作连接并释放 SSH 锁"""
    try:
        yield
    finally:
        try:
            device_ops = item.stash.get(DEVICE_OPS_KEY, None)
            if device_ops is not None:
                print("\n释放 GRPC 链接资源:")
                device_ops.close()
        finally:
            locked_device = item.stash.get(LOCKED_DEVICE_KEY, None)
            if locked_device is not None:
                locked_device.close()
