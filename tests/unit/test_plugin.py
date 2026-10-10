"""pytest 插件集成测试: 用 pytester 跑一个子 pytest, 验证收集、参数化、fixture、设备池和排序"""

from __future__ import annotations

import json
import textwrap

import pytest
import yaml

pytest_plugins = ["pytester"]

# 子 pytest 的 conftest: 启用插件, 并把 SSH 和 SDK 换成记录调用的假对象 (写入 device_log.json)
FAKE_DEVICE_CONFTEST = textwrap.dedent(
    """
    import json
    from pathlib import Path
    import pytest
    from cimalgo.drivers import DeviceOps, InitState
    from cimalgo.remote import device_lock, SDKDeployer

    pytest_plugins = ["cimalgo.pytest_plugin"]
    LOG = Path(__file__).parent / "device_log.json"

    def record(event, **data):
        entries = json.loads(LOG.read_text()) if LOG.exists() else []
        entries.append({"event": event, **data})
        LOG.write_text(json.dumps(entries))

    class FakeLocked:
        def __init__(self, host, username, password, case):
            record("connect", host=host, username=username, password=password, case=case)
            self.connection = object()
        def close(self):
            record("disconnect")

    class FakeOps(DeviceOps):
        def __init__(self, target):
            super().__init__(target)
            record("ops_created", ip=target.ip, port=target.grpc_port)
        def channel(self): return object()
        def close(self): record("ops_closed", ip=self.target.ip, port=self.target.grpc_port)
        def _power_cycle(self, *, off_seconds): record("power_cycle", ip=self.target.ip)
        def get_firmware_version(self): return "2.3.3"
        def _switch_firmware_version(self, version): record("switch_firmware", version=version)
        def _set_phy_l2(self, ids): record("set_phy_l2", ip=self.target.ip, port=self.target.grpc_port, ids=list(ids))
        def _select_layer(self, hb_id, *, tsv_sel0, tsv_sel1):
            record("select_layer", ip=self.target.ip, hb=hb_id, tsv0=tsv_sel0, tsv1=tsv_sel1)
        def _program_one(self, l2_id, archive, k, d, output_dir, *, max_wait_seconds): return None
        def _program_progress(self, l2_id): return 100
        def get_sdk_version(self): return "2.3.3"

    @pytest.fixture(autouse=True, scope="session")
    def _fake_remote():
        original_open = device_lock.LockedDevice.open
        original_for_target = DeviceOps.for_target
        original_for_sdk = SDKDeployer.for_sdk
        class FakeDeployer:
            def deploy(self, connection): record("deploy_sdk")
        device_lock.LockedDevice.open = classmethod(
            lambda cls, connector, *, username, host, port, password, case, **kw: FakeLocked(host, username, password, case)
        )
        DeviceOps.for_target = classmethod(lambda cls, target, **kw: FakeOps(target))
        SDKDeployer.for_sdk = classmethod(lambda cls, sdk: FakeDeployer())
        yield
        device_lock.LockedDevice.open = original_open
        DeviceOps.for_target = original_for_target
        SDKDeployer.for_sdk = original_for_sdk
    """
)


def _case_yaml(name: str, device: str, blocks: list[int], layers: list[int] | None = None) -> str:
    return yaml.safe_dump(
        {
            "case": name,
            "device": {"name": device, "hard_blocks": blocks, "select_layers": layers if layers is not None else [0]},
            "model": {"kind": "fake", "layers": [{"name": "n", "batch": 1, "in_features": 1, "out_features": 1}]},
        }
    )


@pytest.fixture
def project(pytester: pytest.Pytester, config_data: dict) -> pytest.Pytester:
    pytester.makefile(".yaml", cim_config=yaml.safe_dump(config_data))
    pytester.makeconftest(FAKE_DEVICE_CONFTEST)
    cases = pytester.mkdir("cases")
    (cases / "b_case.yaml").write_text(_case_yaml("b_case", "evb-01", [0, 1]))
    (cases / "a_case.yaml").write_text(_case_yaml("a_case", "evb-01", [0, 1]))
    (cases / "c_case.yaml").write_text(_case_yaml("c_case", "evb-02", [2]))
    return pytester


def _events(project: pytest.Pytester) -> list[dict]:
    log = project.path / "device_log.json"
    return json.loads(log.read_text()) if log.exists() else []


def _run(project: pytest.Pytester, *args: str) -> pytest.RunResult:
    return project.runpytest(*args, "-p", "no:cacheprovider", "-W", "ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# 收集
# ---------------------------------------------------------------------------


def test_unit_run_does_not_load_project_config(pytester: pytest.Pytester):
    """没有 --cases/--download/--sdk/--device 时不读配置文件, 即使它不存在"""
    pytester.makeconftest('pytest_plugins = ["cimalgo.pytest_plugin"]')
    pytester.makepyfile("def test_ok(): pass")
    result = pytester.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)


def test_cases_are_collected_sorted_by_device_init(project: pytest.Pytester):
    project.makepyfile(
        textwrap.dedent(
            """
            import pytest
            @pytest.mark.hardware
            def test_case(case, model_dir):
                assert (model_dir / "n" / "input_data.npy").is_file()
            """
        )
    )
    result = _run(project, "--cases", "cases", "-m", "hardware", "-v")
    result.assert_outcomes(passed=3)
    order = [line.split("::")[1].split()[0] for line in result.outlines if "::test_case[" in line and "PASSED" in line]
    assert order == ["test_case[a_case]", "test_case[b_case]", "test_case[c_case]"]


def test_hardware_tests_skipped_by_default(project: pytest.Pytester):
    project.makepyfile(
        textwrap.dedent(
            """
            import pytest
            @pytest.mark.hardware
            def test_hw(): pass
            def test_unit(): pass
            """
        )
    )
    result = _run(project, "--cases", "cases")
    result.assert_outcomes(passed=1, skipped=1)


def test_device_option_skips_other_devices(project: pytest.Pytester):
    project.makepyfile(
        textwrap.dedent(
            """
            import pytest
            @pytest.mark.hardware
            def test_case(case): pass
            """
        )
    )
    result = _run(project, "--cases", "cases", "-m", "hardware", "--device", "evb-02", "-rs")
    result.assert_outcomes(passed=1, skipped=2)
    result.stdout.fnmatch_lines(["*--device=evb-02 排除了 evb-01*"])


def test_invalid_case_file_is_usage_error(project: pytest.Pytester):
    (project.path / "cases" / "bad.yaml").write_text("case: ok\ndevice: evb-01\nmodel: {kind: nope}\n")
    result = _run(project, "--cases", "cases")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*bad.yaml*model.kind*"])


@pytest.mark.parametrize("kind", ["missing", "file", "empty"])
def test_invalid_cases_directory_is_usage_error(project: pytest.Pytester, kind):
    path = project.path / "invalid_cases"
    if kind == "file":
        path.write_text("not a directory")
    elif kind == "empty":
        path.mkdir()
    project.makepyfile("def test_case(case): pass")
    result = _run(project, "--cases", str(path), "-m", "hardware")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*case 目录*"])
    assert not _events(project)


# ---------------------------------------------------------------------------
# 设备池: 初始化复用、失败作废、跨设备、按设备组释放
# ---------------------------------------------------------------------------


def test_pool_reuses_init_and_releases_after_device_group(project: pytest.Pytester, config_data: dict):
    project.makepyfile(
        textwrap.dedent(
            """
            import pytest
            @pytest.mark.hardware
            def test_case(case, device_ops):
                assert device_ops.target.ip in ("10.27.17.45", "10.27.17.46")
            """
        )
    )
    result = _run(project, "--cases", "cases", "-m", "hardware")
    result.assert_outcomes(passed=3)

    events = _events(project)
    kinds = [event["event"] for event in events]
    # a_case 和 b_case 同设备同状态: 只建连一次、初始化一次; c_case 另一台设备: 各一次
    assert kinds.count("connect") == 2
    assert kinds.count("set_phy_l2") == 2
    assert [e["ids"] for e in events if e["event"] == "set_phy_l2"] == [[0, 1], [2]]
    # 每个设备的最后一个测试结束后释放, 且 sessionfinish 不会重复释放
    assert kinds.count("ops_closed") == 2 and kinds.count("disconnect") == 2
    evb_01_closed = next(
        index
        for index, event in enumerate(events)
        if event["event"] == "ops_closed" and event["ip"] == "10.27.17.45"
    )
    evb_02_connected = next(
        index
        for index, event in enumerate(events)
        if event["event"] == "connect" and event["host"] == "10.27.17.46"
    )
    assert evb_01_closed < evb_02_connected
    assert kinds.index("disconnect") < evb_02_connected


def test_pool_reinitializes_after_failure(project: pytest.Pytester, config_data: dict):
    project.makepyfile(
        textwrap.dedent(
            """
            import pytest
            @pytest.mark.hardware
            def test_case(case, device_ops):
                if case.name == "a_case":
                    raise RuntimeError("芯片炸了")
            """
        )
    )
    result = _run(project, "--cases", "cases", "-m", "hardware", "-k", "a_case or b_case")
    result.assert_outcomes(passed=1, failed=1)
    kinds = [event["event"] for event in _events(project)]
    # a_case 失败后 b_case 虽然状态相同也要重新初始化
    assert kinds.count("set_phy_l2") == 2
    assert kinds.count("connect") == 1


def test_manual_state_change_forces_reinit_in_next_test(project: pytest.Pytester):
    project.makepyfile(
        textwrap.dedent(
            """
            import pytest
            @pytest.mark.hardware
            def test_case(case, device_ops):
                if case.name == "a_case":
                    # 测试体里手动改了选层, 芯片状态和池记录的不再一致
                    device_ops.select_layer(0, tsv_sel0=99)
            """
        )
    )
    result = _run(project, "--cases", "cases", "-m", "hardware", "-k", "a_case or b_case")
    result.assert_outcomes(passed=2)
    selects = [e for e in _events(project) if e["event"] == "select_layer"]
    # a_case 初始化 (tsv 0) -> 手动改 (tsv 99) -> b_case 虽然要求状态相同也必须重新初始化 (tsv 0)
    assert [s["tsv0"] for s in selects] == [0, 99, 0]


def test_python_test_with_device_marker_shares_pool(project: pytest.Pytester, config_data: dict):
    project.makepyfile(
        textwrap.dedent(
            """
            import pytest

            @pytest.mark.hardware
            def test_case(case, device_ops):
                pass

            @pytest.mark.hardware
            @pytest.mark.device("evb-01", hard_blocks=[0, 1], select_layers=[0])
            def test_same_as_yaml(device_ops, locked_device, case_workdir):
                assert device_ops.target.ip == "10.27.17.45"
                assert case_workdir.name == "test_same_as_yaml"

            @pytest.mark.hardware
            @pytest.mark.device("evb-01", hard_blocks=[0, 1], select_layers=[0], tsv_sel0=7, power_cycle=False)
            def test_different_tsv(device_ops):
                pass

            @pytest.mark.hardware
            def test_without_device(device_ops):
                pass
            """
        )
    )
    result = _run(project, "--cases", "cases", "-m", "hardware", "-v")
    result.assert_outcomes(passed=5, errors=1)
    result.stdout.fnmatch_lines(["*测试没有指定设备*"])
    # YAML case 和 marker 测试混排: 同设备同状态相邻 (a, b, test_same_as_yaml), 然后状态不同的, 最后另一台设备
    ran = [line.split("::")[1].split()[0] for line in result.outlines if "::test_" in line and "%]" in line]
    assert ran == [
        "test_case[a_case]",
        "test_case[b_case]",
        "test_same_as_yaml",
        "test_different_tsv",
        "test_case[c_case]",
        "test_without_device",
    ]

    events = _events(project)
    # evb-01 默认状态初始化一次 (a_case), b_case 和 test_same_as_yaml 复用; tsv=7 一次; evb-02 一次
    selects = [e for e in events if e["event"] == "select_layer"]
    assert [(s["ip"], s["tsv0"]) for s in selects] == [("10.27.17.45", 0), ("10.27.17.45", 7), ("10.27.17.46", 0)]
    kinds = [e["event"] for e in events]
    # tsv=7 那次 power_cycle=False, 其余两次初始化各下电上电一次
    assert kinds.count("power_cycle") == 2
    # 每台设备只建连一次
    assert kinds.count("connect") == 2


def test_grpc_port_switch_reuses_ssh_but_reinitializes(project: pytest.Pytester, config_data: dict):
    project.makepyfile(
        textwrap.dedent(
            """
            import pytest

            # 定义顺序故意交错: 排序后同一端口的应相邻
            @pytest.mark.hardware
            @pytest.mark.device("evb-01", grpc_port=60010, hard_blocks=[0], power_cycle=False)
            def test_site1_a(device_ops):
                assert device_ops.target.grpc_port == 60010

            @pytest.mark.hardware
            @pytest.mark.device("evb-01", hard_blocks=[0], power_cycle=False)
            def test_site0(device_ops):
                assert device_ops.target.grpc_port == 60000

            @pytest.mark.hardware
            @pytest.mark.device("evb-01", grpc_port=60010, hard_blocks=[0], power_cycle=False)
            def test_site1_b(device_ops):
                assert device_ops.target.grpc_port == 60010
            """
        )
    )
    result = _run(project, "-m", "hardware", "-v")
    result.assert_outcomes(passed=3)
    ran = [line.split("::")[1].split()[0] for line in result.outlines if "::test_" in line and "%]" in line]
    assert ran == ["test_site0", "test_site1_a", "test_site1_b"]

    events = _events(project)
    kinds = [e["event"] for e in events]
    # 同一块板: SSH 只连一次, DeviceOps 只建一个
    assert kinds.count("connect") == 1 and kinds.count("ops_created") == 1
    # 切端口时关掉旧 gRPC 通道 (会话结束时再关一次)
    closed = [e["port"] for e in events if e["event"] == "ops_closed"]
    assert closed == [60000, 60010]
    # site0 初始化一次, 切到 site1 后重新初始化一次, site1_b 复用
    assert [e["port"] for e in events if e["event"] == "set_phy_l2"] == [60000, 60010]


def test_credentials_come_from_netrc_by_ip(project: pytest.Pytester, config_data: dict):
    netrc = project.path / ".netrc"
    netrc.write_text("machine 10.27.17.45\nlogin admin\npassword pw-45\n")
    netrc.chmod(0o600)
    project.makepyfile(
        textwrap.dedent(
            """
            import pytest
            @pytest.mark.hardware
            def test_case(case, device_ops): pass
            """
        )
    )
    result = _run(project, "--cases", "cases", "-m", "hardware", "--device-password", "fallback")
    result.assert_outcomes(passed=3)
    connects = {e["host"]: e for e in _events(project) if e["event"] == "connect"}
    # evb-01 在 .netrc 里: 用 netrc 的账号密码; evb-02 不在: 用配置的用户名 + 兜底密码
    assert connects["10.27.17.45"]["username"] == "admin" and connects["10.27.17.45"]["password"] == "pw-45"
    assert connects["10.27.17.46"]["username"] == "tester" and connects["10.27.17.46"]["password"] == "fallback"


@pytest.mark.parametrize("absolute", [False, True])
def test_device_credentials_use_configured_netrc(project: pytest.Pytester, config_data: dict, absolute):
    path = project.path / "credentials" / "devices.netrc"
    path.parent.mkdir()
    path.write_text("machine 10.27.17.45\nlogin configured-user\npassword configured-password\n")
    path.chmod(0o600)
    # 根目录故意放另一份凭据, 确认不会悄悄使用它。
    default = project.path / ".netrc"
    default.write_text("machine 10.27.17.45\nlogin wrong-user\npassword wrong-password\n")
    default.chmod(0o600)
    config_data["netrc"] = str(path if absolute else path.relative_to(project.path))
    project.makefile(".yaml", cim_config=yaml.safe_dump(config_data))
    project.makepyfile(
        """
        import pytest
        @pytest.mark.hardware
        @pytest.mark.device("evb-01")
        def test_device(device_ops): pass
        """
    )
    result = _run(project, "-m", "hardware", "--device-password", "fallback")
    result.assert_outcomes(passed=1)
    (connect,) = [event for event in _events(project) if event["event"] == "connect"]
    assert connect["username"] == "configured-user" and connect["password"] == "configured-password"


@pytest.mark.parametrize("fixtures", ["device_ops, locked_device", "locked_device, device_ops", "locked_device"])
def test_device_fixtures_deploy_sdk_once_per_case(project: pytest.Pytester, fixtures):
    project.makepyfile(
        f"""
        import pytest
        @pytest.mark.hardware
        @pytest.mark.device("evb-01", update_sdk=True)
        @pytest.mark.parametrize("index", [0, 1])
        def test_device(index, {fixtures}): pass
        """
    )
    result = _run(project, "-m", "hardware")
    result.assert_outcomes(passed=2)
    kinds = [event["event"] for event in _events(project)]
    # 每个 case 一次, 不是每个 fixture 一次, 也不是整个 session 只部署一次。
    assert kinds.count("deploy_sdk") == 2
    assert kinds.count("power_cycle") == 2
    assert kinds.count("set_phy_l2") == 2
    assert kinds.count("connect") == 1


def test_missing_device_in_config_fails_clearly(project: pytest.Pytester, config_data: dict):
    del config_data["devices"]["evb-02"]
    project.makefile(".yaml", cim_config=yaml.safe_dump(config_data))
    project.makepyfile(
        textwrap.dedent(
            """
            import pytest
            @pytest.mark.hardware
            def test_case(case, device_ops): pass
            """
        )
    )
    result = _run(project, "--cases", "cases", "-m", "hardware", "-k", "c_case")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*evb-02*未在配置中定义*"])


def test_model_marker_provides_model_dir_without_yaml_case(project: pytest.Pytester):
    project.makepyfile(
        textwrap.dedent(
            """
            import numpy as np
            import pytest

            LAYER = {"name": "n", "batch": 2, "in_features": 3, "out_features": 4}

            @pytest.mark.hardware
            @pytest.mark.device("evb-01", hard_blocks=[0], select_layers=[0])
            @pytest.mark.model("fake", seed=3, layers=[LAYER])
            def test_fake(model_dir, model_spec, case_workdir):
                assert model_spec.kind == "fake" and model_spec.seed == 3
                assert model_dir == case_workdir / "model"
                assert np.load(model_dir / "n" / "input_data.npy").shape == (2, 3)

            @pytest.mark.hardware
            @pytest.mark.device("evb-01", hard_blocks=[0], select_layers=[0])
            @pytest.mark.model("external", path="models/x")
            def test_external_missing(model_dir):
                pass

            @pytest.mark.hardware
            @pytest.mark.device("evb-01", hard_blocks=[0], select_layers=[0])
            def test_no_model(model_dir):
                pass
            """
        )
    )
    result = _run(project, "-m", "hardware", "-v")
    result.assert_outcomes(passed=1, errors=2)
    result.stdout.fnmatch_lines(["*模型目录不存在*models/x*", "*测试没有指定模型*"])


def test_invalid_marker_is_usage_error(project: pytest.Pytester):
    project.makepyfile(
        textwrap.dedent(
            """
            import pytest
            @pytest.mark.hardware
            @pytest.mark.device("evb-01", tsv_sel0=999)
            def test_bad(device_ops): pass
            """
        )
    )
    result = _run(project, "-m", "hardware")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*device 标记无效*tsv_sel0*"])
