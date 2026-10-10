import os
from collections.abc import Mapping

import pytest

from cim_config import ProjectConfigError, load_project_config
from cim_constants import (
    LOG_ENV_DEFAULTS,
    PROJECT_CONFIG_KEY,
    PROJECT_ROOT,
    TEST_IP_WHITELIST,
)
from utils.set_env import SetEnv
from utils.ssh_connection import SSHConnector


def pytest_configure(config):
    """在测试模块导入前设置日志默认值"""
    for name, value in LOG_ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    """
    缓存项目配置, 主进程按需在收集测试、启动 xdist worker 前准备依赖

    pytest_sessionstart 是 pytest 自动调用的钩子, 在每个进程的测试会话
    开始时执行, 早于测试用例的收集和执行; session.config 保存运行配置

    xdist (pytest-xdist) 是并行测试插件。pytest -n 4 会启动 4 个 worker 执行测试, 主进程负责调度和汇总。主进程和 worker 都会调用此钩子

    各进程拥有独立的 session, 配置在每个进程中加载一次并存入 session.stash 中

    tryfirst=True 让本实现优先于同一钩子的普通实现执行; xdist 启动 worker 的实现使用 trylast=True, 因此这里的下载、安装会先完成
    """
    config = session.config
    try:
        project_config = load_project_config(sdk_name=config.getoption("--sdk"))
    except ProjectConfigError as error:
        raise pytest.UsageError(str(error)) from error
    # 不论是否下载、是否为 worker, 都先缓存配置, 确保 fixture 可以直接读取
    session.stash[PROJECT_CONFIG_KEY] = project_config

    deployer = SetEnv.for_config(project_config)
    # xdist 只在 worker 的 config 上添加 workerinput 字典, 记录 workerid 等信息,
    # 主进程没有此属性; 通过它识别并跳过 worker, 避免重复下载、解压
    if config.getoption("--download") and not hasattr(config, "workerinput"):
        # 本机 xdist worker 共用安装目录和 Python 环境, 只在主进程执行准备工作
        deployer.prepare_packages()
        deployer.sync_wheels()

    # 每个进程都要设置导入路径; 普通测试和 xdist worker 也需要导入 SDK 模块
    deployer.add_import_paths()


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


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment(request):
    """按项目配置选择芯片实现，并在整次测试期间部署环境变量"""
    global_config = request.session.stash[PROJECT_CONFIG_KEY]
    environment = SetEnv.for_config(global_config).build_environment()
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name, value in environment.items():
            monkeypatch.setenv(name, value)

        # session 结束后退出 context, 自动恢复上述环境变量
        yield


@pytest.fixture(scope="session")
def ssh_connector() -> SSHConnector:
    """整个测试会话共用一个 SSHConnector 单例, 复用首次信任白名单状态"""
    return SSHConnector(trusted_ips=TEST_IP_WHITELIST)


@pytest.fixture
def case_context(request):
    """
    解析通过 pytest 间接参数化传入的 case

    Usage::

        @pytest.mark.parametrize("case_context", CASES, indirect=True)
        def test_model(case_context): ...

    如果不写 indirect=True, pytest 会跳过 case_context fixture, 直接把原始字典传给测试函数
    """
    if not hasattr(request, "param"):
        raise pytest.UsageError("case_context must be parametrized with indirect=True")

    from utils.case_resolver import resolve_case

    return resolve_case(request.param, base_dir=PROJECT_ROOT)


def _case_metadata(value):
    """从扁平化的 case 定义中读取显示信息"""
    if not isinstance(value, Mapping):
        return None

    case_name = value.get("case")
    tag = value.get("tag")
    if not case_name or not tag:
        return None
    return str(case_name), str(tag)


def pytest_make_parametrize_id(config, val, argname):
    """使用 case 字段作为 case_context 的 pytest 参数 ID"""
    if argname == "case_context" and (metadata := _case_metadata(val)):
        return metadata[0]
    return None


def pytest_runtest_setup(item):
    """仅为参数化硬件 case 输出名称和 tag"""
    callspec = getattr(item, "callspec", None)
    if callspec is None or "case_context" not in callspec.params:
        return

    metadata = _case_metadata(callspec.params["case_context"])
    if metadata is None:
        return

    case_name, tag = metadata
    title = f"CASE: {case_name} | tag={tag}"

    terminal_reporter = item.config.pluginmanager.get_plugin("terminalreporter")
    if terminal_reporter is not None:
        terminal_reporter.write_sep("-", title, purple=True)
    else:
        print(title)
