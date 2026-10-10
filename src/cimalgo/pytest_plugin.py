"""
pytest 插件: 命令行选项、会话级配置、设备 fixture、YAML case 收集

在 conftest.py 里 ``pytest_plugins = ["cimalgo.pytest_plugin"]`` 即可启用。

两种写 case 的方式共用同一套 fixture::

    # 1. YAML: 由 --cases 目录加载, 每个文件参数化出一个 test_case
    def test_case(case, device_ops, ...): ...

    # 2. Python: 用 device 标记声明设备和初始化参数, 流程自己写
    @pytest.mark.hardware
    @pytest.mark.device("evb-01", hard_blocks=[0, 1], select_layers=[0])
    def test_sweep(device_ops, workdir): ...

设备连接、锁和初始化状态按设备名缓存在 session 级的 DevicePool 中:
相邻测试要求的 InitState 相同就跳过重复初始化; 测试失败后该设备的状态作废, 下一个测试强制重新初始化。

InferSession 不是 fixture: 推理运行时的 init 依赖芯片上已烧录的程序, 必须在测试体内 program() 之后
用 with 打开。


pytest 启动流程：
    加载插件 / conftest.py
            ↓
    解析命令行和全局配置
            ↓
    创建 Session
            ↓
    收集测试与参数化 case
            ↓
    修改、过滤、排序测试项
            ↓
    逐个执行测试
            ↓
    结束 Session、释放资源
"""

from __future__ import annotations

import netrc
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from cimalgo.config import DeviceConfig, ProjectConfig, ProjectConfigError, load_project_config
from cimalgo.env import SetEnv, SetEnvError
from cimalgo.logging import configure_logging, get_logger
from cimalgo.pipeline.case import (
    CaseDefinitionError,
    CaseSpec,
    DeviceSpec,
    ModelSpec,
    load_cases,
    parse_device,
    parse_model,
)

if TYPE_CHECKING:
    from cimalgo.drivers import DeviceOps
    from cimalgo.remote import LockedDevice, SSHConnector

log = get_logger(__name__)

PROJECT_CONFIG_KEY: pytest.StashKey[ProjectConfig] = pytest.StashKey()
SSH_CONNECTOR_KEY: pytest.StashKey[SSHConnector] = pytest.StashKey()
CASES_KEY: pytest.StashKey[list[CaseSpec]] = pytest.StashKey()
DEVICE_POOL_KEY: pytest.StashKey[DevicePool] = pytest.StashKey()

# NOTE: 通过环境获取设备密码, 要求多个设备使用同一个密码
DEVICE_PASSWORD_ENV = "CIM_DEVICE_PASSWORD"

# ---------------------------------------------------------------------------
# 选项与会话
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("cimalgo", "CIM 测试框架")
    group.addoption(
        "--cim-config", default="cim_config.yaml", metavar="PATH", help="项目配置文件, 默认 ./cim_config.yaml"
    )
    group.addoption("--sdk", default=None, metavar="NAME", help="覆盖 active_sdk")
    group.addoption("--download", action="store_true", help="收集前下载安装 SDK/工具链并同步 wheel (xdist 仅主进程)")
    group.addoption("--cases", default=None, metavar="DIR", help="YAML case 目录; 不指定则不收集 YAML case")
    group.addoption("--device", default=None, metavar="NAME", help="只运行指定设备的测试, 其余跳过")
    group.addoption(
        "--device-password",
        default=None,
        help=f"所有设备共用的 SSH 密码 (兜底); 优先级低于 .netrc 中按 IP 配置的凭据; 也可用环境变量 {DEVICE_PASSWORD_ENV}",
    )
    group.addoption("--workdir", default="workspace", metavar="DIR", help="编译产物和推理结果目录")


def pytest_configure(config: pytest.Config) -> None:
    configure_logging()
    # 为列表型配置 markers 增加三个自定义标记
    config.addinivalue_line("markers", "hardware: 需要真实设备, 默认不运行; 用 -m hardware 显式选择")
    config.addinivalue_line(
        "markers",
        "device(name, *, grpc_port=60000, hard_blocks=(), select_layers=(), tsv_sel0=0, tsv_sel1=255, "
        "power_cycle=True, update_sdk=False): Python 测试声明所用设备、gRPC 端口 (site) 和初始化参数",
    )
    config.addinivalue_line(
        "markers",
        "model(kind, *, path=None, seed=42, layers=()): Python 测试声明模型来源; 字段与 YAML 的 model 段相同",
    )


def _wants_project(config: pytest.Config) -> bool:
    """
    只在明确请求时加载项目配置: 指定了 --cases / --download / --sdk / --device, 或给了 -m

    -m 不管具体表达式是什么都算, 跟 pytest_collection_modifyitems 里"默认跳过 hardware"的判断
    用同一个条件: 只要用户给了 -m, 就认为可能选中 hardware, 交给 pytest 自己的 -m 过滤去判断
    具体有没有选中, 这里不重新解析表达式。
    """
    return bool(
        config.getoption("--cases")
        or config.getoption("--download")
        or config.getoption("--sdk")
        or config.getoption("--device")
        or config.getoption("-m")
    )


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session: pytest.Session) -> None:
    config = session.config
    session.stash[CASES_KEY] = []
    if not _wants_project(config):
        return

    try:
        project = load_project_config(config.getoption("--cim-config"), sdk_name=config.getoption("--sdk"))
    except ProjectConfigError as error:
        raise pytest.UsageError(str(error)) from error
    session.stash[PROJECT_CONFIG_KEY] = project

    from cimalgo.remote import SSHConnector

    session.stash[SSH_CONNECTOR_KEY] = SSHConnector(trusted_ips=project.trusted_ips)
    session.stash[DEVICE_POOL_KEY] = DevicePool(
        project, session.stash[SSH_CONNECTOR_KEY], _credentials_lookup(config, project)
    )

    set_env = SetEnv.for_config(project)
    try:
        if config.getoption("--download") and not hasattr(config, "workerinput"):
            set_env.prepare_packages()
            set_env.sync_wheels()
        set_env.apply_import_paths()
    except SetEnvError as error:
        raise pytest.UsageError(str(error)) from error

    monkeypatch = pytest.MonkeyPatch()
    config.add_cleanup(monkeypatch.undo)
    for name, value in set_env.environment().items():
        monkeypatch.setenv(name, value)

    cases_dir = config.getoption("--cases")
    if cases_dir:
        try:
            session.stash[CASES_KEY] = list(load_cases(cases_dir))
        except CaseDefinitionError as error:
            raise pytest.UsageError(str(error)) from error


def pytest_sessionfinish(session: pytest.Session) -> None:
    pool = session.stash.get(DEVICE_POOL_KEY, None)
    if pool is not None:
        pool.close_all()


# ---------------------------------------------------------------------------
# 凭据
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Credentials:
    username: str
    password: str | None


def _credentials_lookup(config: pytest.Config, project: ProjectConfig) -> Any:
    """返回 DeviceConfig -> Credentials; .netrc 按设备 IP 匹配优先, 其次 --device-password / 环境变量"""
    fallback = config.getoption("--device-password") or os.environ.get(DEVICE_PASSWORD_ENV) or None
    netrc_path = project.netrc
    entries: netrc.netrc | None = None
    if netrc_path.is_file():
        try:
            entries = netrc.netrc(str(netrc_path))
        except (OSError, netrc.NetrcParseError):
            log.warning("无法解析 %s, 设备凭据回退到命令行/环境变量", netrc_path)

    def lookup(device: DeviceConfig) -> Credentials:
        if entries is not None:
            found = entries.authenticators(device.ip)
            if found is not None:
                login, _, password = found
                return Credentials(username=login or device.ssh_username, password=password or None)
        return Credentials(username=device.ssh_username, password=fallback)

    return lookup


# ---------------------------------------------------------------------------
# 设备池
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    locked: LockedDevice
    ops: DeviceOps  # 芯片当前状态由 ops.known_state 跟踪


class DevicePool:
    """按设备名缓存连接、锁和设备操作对象; 同一设备的相邻测试之间复用"""

    def __init__(self, project: ProjectConfig, connector: SSHConnector, credentials: Any):
        self._project = project
        self._connector = connector
        self._credentials = credentials
        self._entries: dict[str, _Entry] = {}

    def acquire(self, spec: DeviceSpec, *, case_name: str) -> DeviceOps:
        """返回已初始化到 spec 所要求状态的 DeviceOps; 状态与上次相同则跳过初始化"""
        from cimalgo.drivers import DeviceOps, DeviceTarget
        from cimalgo.remote import LockedDevice, SDKDeployer

        device = self._project.device(spec.name)
        entry = self._entries.get(spec.name)
        if entry is None:
            credentials = self._credentials(device)
            locked = LockedDevice.open(
                self._connector,
                username=credentials.username,
                host=device.ip,
                port=device.ssh_port,
                password=credentials.password,
                case=case_name,
            )
            try:
                ops = DeviceOps.for_target(
                    DeviceTarget.from_config(device, driver=self._project.sdk.chip, grpc_port=spec.grpc_port)
                )
            except BaseException:
                locked.close()
                raise
            entry = self._entries[spec.name] = _Entry(locked=locked, ops=ops)
        elif entry.ops.use_grpc_port(spec.grpc_port):
            log.info("[%s] 切换 gRPC 端口到 %d, 芯片状态作废", spec.name, spec.grpc_port)

        if spec.update_sdk:
            SDKDeployer.for_sdk(self._project.sdk).deploy(entry.locked.connection)
            entry.ops.invalidate()  # 部署后芯片状态不可信

        # TODO: 如果 sdk version 未来不是 firmware 版本, 则需要特殊处理
        wanted = spec.init_state(self._project.sdk.version)
        with log.tagged(case_name):
            if not entry.ops.initialize(wanted, power_cycle=spec.power_cycle):
                log.info("[%s] 初始化状态与上一个测试相同, 跳过", spec.name)
        return entry.ops

    def invalidate(self, name: str) -> None:
        """测试失败后调用: 芯片状态不可信, 下一个测试强制重新初始化"""
        entry = self._entries.get(name)
        if entry is not None:
            entry.ops.invalidate()

    def locked_device(self, name: str) -> LockedDevice:
        """已 acquire 过的设备的 SSH 连接和锁"""
        try:
            return self._entries[name].locked
        except KeyError:
            raise KeyError(f"设备 {name!r} 尚未通过 acquire 建连") from None

    def release(self, name: str) -> None:
        """关闭并移除指定设备的操作对象、SSH 连接和锁"""
        entry = self._entries.pop(name, None)
        if entry is None:
            return

        errors: list[Exception] = []
        for closer in (entry.ops.close, entry.locked.close):
            try:
                closer()
            except Exception as error:
                log.error("释放设备 %s 失败: %s", name, error)
                errors.append(error)
        if errors:
            raise errors[0]

    def close_all(self) -> None:
        """释放池中所有设备; 作为会话异常结束等情况的兜底"""
        errors: list[Exception] = []
        for name in list(self._entries):
            try:
                self.release(name)
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]


# ---------------------------------------------------------------------------
# 收集
# ---------------------------------------------------------------------------


def _device_spec_of(item: pytest.Item) -> DeviceSpec | None:
    """YAML case 从 callspec 取; Python 测试从 device 标记取"""
    callspec = getattr(item, "callspec", None)
    spec = callspec.params.get("case") if callspec else None
    if isinstance(spec, CaseSpec):
        return spec.device

    # 列表配置项 markers 添加的 device 定义
    marker = item.get_closest_marker("device")
    if marker is None:
        return None
    if not marker.args or not isinstance(marker.args[0], str):
        raise pytest.UsageError(f"{item.nodeid}: device 标记需要设备名作为第一个参数")
    try:
        return parse_device({"name": marker.args[0], **marker.kwargs})
    except CaseDefinitionError as error:
        raise pytest.UsageError(f"{item.nodeid}: device 标记无效: {error}") from None


def _model_spec_of(item: pytest.Item) -> ModelSpec | None:
    """YAML case 从 callspec 取; Python 测试从 model 标记取, 相对路径以测试文件所在目录为根"""
    callspec = getattr(item, "callspec", None)
    spec = callspec.params.get("case") if callspec else None
    if isinstance(spec, CaseSpec):
        return spec.model

    # 列表配置项 markers 添加的 model 定义
    marker = item.get_closest_marker("model")
    if marker is None:
        return None
    if not marker.args or not isinstance(marker.args[0], str):
        raise pytest.UsageError(f"{item.nodeid}: model 标记需要 kind ('fake' 或 'external') 作为第一个参数")
    try:
        # item.path: 测试函数所属测试文件的路径
        return parse_model({"kind": marker.args[0], **marker.kwargs}, base_dir=item.path.parent)
    except CaseDefinitionError as error:
        raise pytest.UsageError(f"{item.nodeid}: model 标记无效: {error}") from None


def _case_name_of(item: pytest.Item) -> str:
    callspec = getattr(item, "callspec", None)
    spec = callspec.params.get("case") if callspec else None
    # item.name: 测试项的名称
    return spec.name if isinstance(spec, CaseSpec) else item.name


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    """默认排除 hardware; --device 过滤其他设备; 按 (设备, 初始化状态) 排序让相同状态的测试相邻"""
    # pytest 只运行 -m 指定的 marker 标记的测试
    if not config.getoption("-m"):
        skip = pytest.mark.skip(reason="需要 -m hardware 显式选择")
        for item in items:
            if "hardware" in item.keywords:
                item.add_marker(skip)

    only_device = config.getoption("--device")
    firmware = session.stash[PROJECT_CONFIG_KEY].sdk.version if PROJECT_CONFIG_KEY in session.stash else None

    # 利用 Python 元组的 "字典序比较" 来进行多级排序, 本意是让 device 相同的测试相邻, 同一 gRPC 端口 (site) 的相邻,
    # init_state 相同的测试也相邻
    def sort_key(item: pytest.Item) -> tuple:
        spec = _device_spec_of(item)
        if spec is None:
            return (1, item.nodeid)
        if only_device and spec.name != only_device:
            item.add_marker(pytest.mark.skip(reason=f"--device={only_device} 排除了 {spec.name}"))
        return (0, spec.name, spec.grpc_port, spec.init_state(firmware).sort_key(), _case_name_of(item))

    items.sort(key=sort_key)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """
    请求 case 参数的测试按已加载的 YAML case 动态参数化; 没有 case 时产生空参数集, 其行为如下:

        pytest.ini 中开启 empty_parameter_set_mark, 可选值及效果
            skip: 跳过, 默认值
            xfail: 预期失败
            fail_at_collect: 直接报错
        例如:
            [pytest]
            empty_parameter_set_mark = fail_at_collect
    """
    # metafunc: 用来描述测试项函数
    # fixturenames: 测试项函数全部参数的名字(包括 fixture, 参数化名称), 例如: test_pipeline(case, ...)
    # definition: item 类型的 “函数定义节点”
    #
    # 参考 tests/hardware/test_pipeline.py::test_case, 目的是使用 yaml 自动生成测试 case
    if "case" in metafunc.fixturenames:
        cases = metafunc.definition.session.stash[CASES_KEY]
        metafunc.parametrize("case", cases, ids=[spec.id for spec in cases])


# 每个测试项的 setup、call、teardown 每个阶段结束后都会执行。hookwrapper 将该函数作为 "包装器" 执行
#
#       进入 hookwrapper
#           ↓
#       yield
#           ↓
#       执行 pytest 核心的 pytest_runtest_makereport
#           ↓
#       核心返回 TestReport
#           ↓
#       回到 hookwrapper
#           ↓
#       outcome.get_result()
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> Iterator[None]:
    """测试体或 setup 失败后, 该设备的初始化状态作废"""
    outcome = yield
    report = outcome.get_result()  # type: ignore[attr-defined]
    if report.failed and call.when in ("setup", "call"):
        spec = _device_spec_of(item)
        pool = item.session.stash.get(DEVICE_POOL_KEY, None)
        if spec is not None and pool is not None:
            pool.invalidate(spec.name)

# pytest_runtest_protocol
#   - 被 -k/-m deselect 的测试：不会调用。
#   - 被 skip 的测试: 通常仍会调用，只是在 setup 阶段发现标记 skip, 跳过函数体。
#   - nextitem: pytest 实际准备处理的下一个测试项；最后一项时为 None。
#
#   进入 hookwrapper
#       ↓
#   yield
#       ↓
#   pytest 核心的 pytest_runtest_protocol
#       ├── setup
#       ├── call
#       └── teardown
#       ↓
#   回到 hookwrapper
#       ↓
#   判断 nextitem，必要时 release
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Iterator[None]:
    """同一设备的最后一个测试完成后立即释放连接和设备锁"""
    yield

    current = _device_spec_of(item)
    following = _device_spec_of(nextitem) if nextitem is not None else None
    if current is not None and (following is None or following.name != current.name):
        pool = item.session.stash.get(DEVICE_POOL_KEY, None)
        if pool is not None:
            pool.release(current.name)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def project_config(request: pytest.FixtureRequest) -> ProjectConfig:
    try:
        return request.session.stash[PROJECT_CONFIG_KEY]
    except KeyError:
        pytest.fail("项目配置未加载; 需要 --cases、--download、--sdk、--device 之一, 或 -m hardware")


@pytest.fixture(scope="session")
def ssh_connector(request: pytest.FixtureRequest) -> SSHConnector:
    return request.session.stash[SSH_CONNECTOR_KEY]


@pytest.fixture(scope="session")
def device_pool(request: pytest.FixtureRequest) -> DevicePool:
    try:
        return request.session.stash[DEVICE_POOL_KEY]
    except KeyError:
        pytest.fail("设备池未创建; 需要 -m hardware 或 --cases / --device")


@pytest.fixture(scope="session")
def workdir(request: pytest.FixtureRequest) -> Path:
    path = Path(request.config.getoption("--workdir")).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def case_workdir(workdir: Path, request: pytest.FixtureRequest) -> Path:
    path = workdir / _case_name_of(request.node)
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def model_spec(request: pytest.FixtureRequest) -> ModelSpec:
    """当前测试的模型来源: 来自 YAML case 的 model 段或 @pytest.mark.model"""
    spec = _model_spec_of(request.node)
    if spec is None:
        pytest.fail("测试没有指定模型: 请通过 YAML case 的 model 字段或 @pytest.mark.model(...) 声明")
    return spec


@pytest.fixture
def model_dir(model_spec: ModelSpec, case_workdir: Path) -> Path:
    """external 直接返回配置的目录; fake 在 case_workdir/model 下生成"""
    if model_spec.kind == "external":
        assert model_spec.path is not None
        if not model_spec.path.is_dir():
            pytest.fail(f"模型目录不存在: {model_spec.path}")
        return model_spec.path
    from cimalgo.pipeline.fake_data import make_fake_model

    target = model_spec.path or case_workdir / "model"
    return make_fake_model(target, model_spec.layers, seed=model_spec.seed)


@pytest.fixture
def device_spec(request: pytest.FixtureRequest) -> DeviceSpec:
    """当前测试的设备和初始化参数: 来自 YAML case 或 @pytest.mark.device"""
    spec = _device_spec_of(request.node)
    if spec is None:
        pytest.fail("测试没有指定设备: 请通过 YAML case 的 device 字段或 @pytest.mark.device(...) 声明")
    return spec


@pytest.fixture
def device_ops(device_spec: DeviceSpec, device_pool: DevicePool, request: pytest.FixtureRequest) -> DeviceOps:
    """已初始化到 device_spec 所要求状态的设备操作对象; 连接和锁由 DevicePool 跨测试持有"""
    return device_pool.acquire(device_spec, case_name=_case_name_of(request.node))


@pytest.fixture
def locked_device(device_ops: DeviceOps, device_spec: DeviceSpec, device_pool: DevicePool) -> LockedDevice:
    """当前设备的 SSH 连接 (已持锁); device_ops 用于保证 fixture 的执行顺序, 同一测试只部署和初始化一次"""
    return device_pool.locked_device(device_spec.name)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    spec = _device_spec_of(item)
    if spec is not None:
        reporter = item.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_sep("-", f"CASE: {_case_name_of(item)} | device={spec.name}", purple=True)
