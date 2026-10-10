"""device 层测试: SDK 通过 fake_sdk fixture 注入"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cimalgo.config import DeviceConfig
from cimalgo.drivers import DeviceError, DeviceOps, DeviceTarget, InferSession, InitState
from cimalgo.drivers.bw01d import Bw01dDeviceOps


@pytest.fixture
def target() -> DeviceTarget:
    return DeviceTarget.from_config(DeviceConfig(name="evb", ip="10.27.17.45", evb_port=61006), driver="bw01d")


@pytest.fixture
def ops(fake_sdk, target) -> Bw01dDeviceOps:
    return Bw01dDeviceOps(target, control_executable="/nonexistent/bw01d_ctrl", firmware_settle_seconds=0)


# ---------------------------------------------------------------------------
# DeviceTarget / 注册
# ---------------------------------------------------------------------------


def test_target_from_config_copies_addresses():
    device = DeviceConfig(name="evb", ip="10.27.17.45", evb_port=1, server_port=2)
    target = DeviceTarget.from_config(device, driver="BW01D", grpc_port=60010)
    assert (target.ip, target.grpc_port, target.bmc_ip, target.driver) == ("10.27.17.45", 60010, None, "BW01D")
    assert target.evb_address == "10.27.17.45:1"
    assert target.server_address == "10.27.17.45:2"
    # 不传 grpc_port 时使用默认 site
    assert DeviceTarget.from_config(device, driver="bw01d").grpc_port == 60000


def test_evb_address_requires_evb_port():
    target = DeviceTarget.from_config(DeviceConfig(name="evb", ip="10.27.17.45", bmc_ip="10.0.0.9"), driver="bw01d")
    assert target.bmc_ip == "10.0.0.9"
    with pytest.raises(DeviceError, match="未配置 evb_port"):
        target.evb_address


def test_power_cycle_without_evb_port_fails_clearly(fake_sdk):
    ops = Bw01dDeviceOps(DeviceTarget(ip="10.27.17.45"))
    with pytest.raises(DeviceError, match="未配置 evb_port"):
        ops.power_cycle()


def test_use_grpc_port_switches_site_and_invalidates(fake_sdk, ops):
    ops.known_state = InitState(firmware_version="2.3.3")
    ops.channel()
    assert ops.use_grpc_port(60000) is False  # 端口相同: 不动
    assert ops.known_state is not None and ops._channel is not None

    assert ops.use_grpc_port(60010) is True
    assert ops.target.grpc_port == 60010 and ops._site() == 1
    assert ops.known_state is None and ops._channel is None  # 旧通道已关, 状态作废
    assert fake_sdk.channel.close.called


def test_for_target_dispatches_case_insensitively(fake_sdk, target):
    assert isinstance(DeviceOps.for_target(target), Bw01dDeviceOps)
    assert isinstance(DeviceOps.for_target(DeviceTarget(ip="1.1.1.1", driver="BW01D")), Bw01dDeviceOps)
    with pytest.raises(DeviceError, match="没有实现"):
        DeviceOps.for_target(DeviceTarget(ip="1.1.1.1", driver="nope"))


def test_duplicate_driver_registration_rejected():
    with pytest.raises(TypeError, match="重复"):

        class Again(Bw01dDeviceOps, driver="bw01d"):
            pass


# ---------------------------------------------------------------------------
# 连接与通道复用
# ---------------------------------------------------------------------------


def test_channel_is_created_once_and_closed(fake_sdk, ops):
    with ops:
        assert ops.channel() is fake_sdk.channel
        assert ops.channel() is fake_sdk.channel
        fake_sdk.bw01d.init.assert_called_once_with(
            {"ip": "10.27.17.45", "port": 60000, "is_cluster": False, "test_stage": "CP3"}
        )
    fake_sdk.channel.close.assert_called_once()
    ops.close()  # 幂等


def test_channel_init_failure_is_device_error(fake_sdk, ops):
    fake_sdk.bw01d.init.return_value = None
    with pytest.raises(DeviceError, match="创建 gRPC 连接失败"):
        ops.channel()


def test_sdk_import_failure_is_actionable(monkeypatch, target):
    import sys

    monkeypatch.setitem(sys.modules, "wtm", None)
    with pytest.raises(DeviceError, match="pytest --download"):
        Bw01dDeviceOps(target).channel()


# ---------------------------------------------------------------------------
# 初始化流程
# ---------------------------------------------------------------------------


def test_power_cycle_opens_and_closes_evb_handle(fake_sdk, ops, monkeypatch):
    monkeypatch.setattr("cimalgo.drivers.bw01d.time.sleep", Mock())
    ops.power_cycle(off_seconds=0)
    calls = fake_sdk.evb.power_control.set_all_channel_enabled.call_args_list
    assert [call.args[1] for call in calls] == [False, True]
    assert fake_sdk.evb.deinit.call_count == 2


def test_power_failure_still_deinits(fake_sdk, ops):
    fake_sdk.evb.power_control.set_all_channel_enabled.return_value = SimpleNamespace(status_code=5, error="no power")
    with pytest.raises(DeviceError, match="no power"):
        ops._set_power(True)
    fake_sdk.evb.deinit.assert_called_once_with("evb-handle")


def test_set_phy_l2_and_select_layer_use_shared_channel(fake_sdk, ops):
    ops.set_phy_l2([0, 1])
    ops.select_layer(3, tsv_sel0=1, tsv_sel1=2)
    fake_sdk.bw01d.set_phy_l2.assert_called_once_with(fake_sdk.channel, [0, 1], timeout=30)
    fake_sdk.hb_ctrl.fc_select_layer.assert_called_once_with(
        fake_sdk.channel, hb_id=3, tsv_sel0=1, tsv_sel1=2, timeout=30
    )


def test_sdk_failure_status_becomes_device_error(fake_sdk, ops):
    fake_sdk.bw01d.set_phy_l2.return_value = SimpleNamespace(status_code=3, error="bad l2")
    with pytest.raises(DeviceError, match=r"配置物理 L2 .* bad l2"):
        ops.set_phy_l2([9])


def test_initialize_runs_standard_sequence(fake_sdk, ops, monkeypatch):
    monkeypatch.setattr("cimalgo.drivers.bw01d.time.sleep", Mock())
    ops.get_firmware_version = Mock(side_effect=["1.0", "2.0"])
    ops._switch_firmware_version = Mock()
    ops.initialize(InitState("2.0", hard_blocks=(0, 1), select_layers=(0, 2)), power_cycle=True)

    ops._switch_firmware_version.assert_called_once_with("2.0")
    fake_sdk.bw01d.set_phy_l2.assert_called_once()
    assert [call.kwargs["hb_id"] for call in fake_sdk.hb_ctrl.fc_select_layer.call_args_list] == [0, 2]


def test_initialize_skips_firmware_switch_when_matching(fake_sdk, ops):
    ops.get_firmware_version = Mock(return_value="2.0")
    ops._switch_firmware_version = Mock()
    ops.initialize(InitState("2.0"), power_cycle=False)
    ops._switch_firmware_version.assert_not_called()
    fake_sdk.evb.init.assert_not_called()


def test_initialize_reports_failed_firmware_switch(fake_sdk, ops):
    ops.get_firmware_version = Mock(side_effect=["1.0", "1.0"])
    ops._switch_firmware_version = Mock()
    with pytest.raises(DeviceError, match="固件版本切换失败"):
        ops.initialize(InitState("2.0"), power_cycle=False)


def test_control_program_missing_is_device_error(ops):
    with pytest.raises(DeviceError, match="未找到设备控制程序"):
        ops.get_sdk_version()


def test_site_lookup_error_lists_known_ports(fake_sdk):
    ops = Bw01dDeviceOps(DeviceTarget(ip="1.1.1.1", grpc_port=12345))
    with pytest.raises(DeviceError, match="12345 没有对应的 site; 已知端口: 60000, 60010"):
        ops._site()


# ---------------------------------------------------------------------------
# 烧录
# ---------------------------------------------------------------------------


def test_program_runs_each_job_and_returns_by_l2(fake_sdk, ops, tmp_path):
    results = ops.program([(0, "a.tar.gz"), (1, "b.tar.gz")], k=1, d=2, output_dir=tmp_path, poll_interval=0.01)
    assert set(results) == {0, 1}
    calls = {call.args[1]: call for call in fake_sdk.l2_ctrl.map_program.call_args_list}
    assert calls[0].args[2:] == ("a.tar.gz", 1, 2)
    assert calls[0].kwargs["output_dir"] == str(tmp_path / "L2_0")
    assert (tmp_path / "L2_1").is_dir()


def test_program_rejects_duplicate_l2(ops, tmp_path):
    with pytest.raises(DeviceError, match="不能重复"):
        ops.program([(0, "a"), (0, "b")], k=1, d=1, output_dir=tmp_path)


def test_program_collects_failures(fake_sdk, ops, tmp_path):
    fake_sdk.l2_ctrl.map_program.side_effect = [
        SimpleNamespace(status_code=0),
        SimpleNamespace(status_code=9, error="burn failed"),
    ]
    with pytest.raises(DeviceError, match=r"L2_1: .*burn failed"):
        ops.program([(0, "a"), (1, "b")], k=1, d=1, output_dir=tmp_path, max_workers=1, poll_interval=0.01)


def test_program_empty_jobs(ops, tmp_path):
    assert ops.program([], k=1, d=1, output_dir=tmp_path) == {}


# ---------------------------------------------------------------------------
# InferSession
# ---------------------------------------------------------------------------


@pytest.fixture
def model_dir(tmp_path):
    root = tmp_path / "model"
    root.mkdir()

    def add(l2, l0, node, *, input_data=None, output=None):
        path = root / f"model.layers.0.test_L2_{l2}_L0_{l0}_node_{node}"
        path.mkdir()
        np.save(
            path / "input_data.npy", np.array([[1, 2], [3, 4]], dtype=np.int8) if input_data is None else input_data
        )
        np.save(path / "output_data.npy", np.array([[10], [20]], dtype=np.int8) if output is None else output)
        return path

    return root, add


def test_infer_session_shares_channel_and_deinits(fake_sdk, ops, tmp_path):
    with InferSession(ops, tmp_path / "res", model_name="demo", mode="vp_erpc") as session:
        fake_sdk.infer.init.assert_called_once_with([fake_sdk.channel], model_name="demo", mode="vp_erpc")
        fake_sdk.infer.deinit.assert_not_called()
        assert session._handles == [fake_sdk.channel]
    fake_sdk.infer.deinit.assert_called_once_with([fake_sdk.channel])
    # device 的通道不受推理会话影响
    fake_sdk.channel.close.assert_not_called()


def test_infer_model_runs_nodes_in_order_and_trims_dummy(fake_sdk, ops, model_dir, tmp_path):
    root, add = model_dir
    add(0, 3, 4)
    add(0, 1, 2)
    fake_sdk.infer.L2Infer.return_value = [
        SimpleNamespace(success=True, error=None, data=np.array([[99], [10], [20]], dtype=np.int8))
    ]

    with InferSession(ops, tmp_path / "res") as session:
        results = session.infer_model(root, dummy=1, prefix="case")

    assert [r.name for r in results] == ["case_L2_0_L0_1_Node_2", "case_L2_0_L0_3_Node_4"]
    np.testing.assert_array_equal(results[0].output, [[10], [20]])
    np.testing.assert_array_equal(results[0].golden, [[10], [20]])
    np.testing.assert_array_equal(np.load(results[0].output_path), [[10], [20]])
    padded = fake_sdk.infer.L2Infer.call_args_list[0].args[2][0]
    np.testing.assert_array_equal(padded, [[1, 2], [1, 2], [3, 4]])
    (mode,) = fake_sdk.infer.L2Infer.call_args.args[3]
    assert mode.options["repeat"] == 1 and mode.options["sub_batch_size"] == 20


def test_infer_model_mode_override(fake_sdk, ops, model_dir, tmp_path):
    root, add = model_dir
    add(0, 0, 0)
    with InferSession(ops, tmp_path / "res") as session:
        session.infer_model(root, infer_mode={"repeat": 3})
    (mode,) = fake_sdk.infer.L2Infer.call_args.args[3]
    assert mode.options["repeat"] == 3 and mode.options["enable_debug_RAW"] is False


def test_infer_model_accepts_model_name_prefix(fake_sdk, ops, model_dir, tmp_path):
    root, add = model_dir
    node = add(5, 6, 7)
    node.rename(root / "model.layers.0.self_attn.q_proj.ic_0_oc_0_L2_5_L0_6_node_7")
    with InferSession(ops, tmp_path / "res") as session:
        results = session.infer_model(root)
    assert fake_sdk.infer.L2Infer.call_args.args[:2] == (5, [(6, 7)])
    assert [(result.l2_id, result.l0_id, result.node_id) for result in results] == [(5, 6, 7)]
    assert results[0].output_path.is_file()


@pytest.mark.parametrize(
    "name", ["L2_0_L0_1", "node_1", "L2_5_L0_6_Node_7", "extra", "prefix_L2_0_L0_0_node_0_extra", "xL2_0_L0_0_node_0"]
)
def test_infer_model_rejects_unexpected_dir(fake_sdk, ops, model_dir, tmp_path, name):
    root, add = model_dir
    add(0, 1, 2)
    (root / name).mkdir()
    with InferSession(ops, tmp_path / "res") as session, pytest.raises(ValueError, match="节点目录名"):
        session.infer_model(root)


@pytest.mark.parametrize(
    ("data", "dummy"),
    [(None, 0), (np.zeros((0, 1), dtype=np.int8), 0), (np.array([[1]], dtype=np.int8), 1)],
)
def test_infer_model_rejects_insufficient_frames(fake_sdk, ops, model_dir, tmp_path, data, dummy):
    root, add = model_dir
    add(0, 1, 2)
    fake_sdk.infer.L2Infer.return_value = [SimpleNamespace(success=True, error=None, data=data)]
    with InferSession(ops, tmp_path / "res") as session, pytest.raises(DeviceError, match="不足以去掉"):
        session.infer_model(root, dummy=dummy)
    assert not list((tmp_path / "res").iterdir())


def test_infer_model_reports_sdk_failure(fake_sdk, ops, model_dir, tmp_path):
    root, add = model_dir
    add(0, 1, 2)
    fake_sdk.infer.L2Infer.return_value = [SimpleNamespace(success=False, error="device busy", data=None)]
    with InferSession(ops, tmp_path / "res") as session, pytest.raises(DeviceError, match="device busy"):
        session.infer_model(root)


def test_infer_session_methods_require_active_context(fake_sdk, ops, tmp_path):
    session = InferSession(ops, tmp_path / "res")
    with pytest.raises(RuntimeError, match="请在 with 内调用"):
        session.calib(0, 1)


def test_infer_session_deinit_failure_preserves_body_error(fake_sdk, ops, tmp_path):
    fake_sdk.infer.deinit.return_value = False
    with pytest.raises(ValueError, match="body") as raised, InferSession(ops, tmp_path / "res"):
        raise ValueError("body")
    assert any("deinit 返回 False" in note for note in raised.value.__notes__)


def test_infer_session_passthrough_methods(fake_sdk, ops, tmp_path):
    with InferSession(ops, tmp_path / "res") as session:
        session.calib(0, 1)
        session.get_oc_info(0, 1, 2, True)
        session.set_runtime_log_level(0, 4)
    fake_sdk.infer.Calib.assert_called_once_with(0, 1, channel_handles=[fake_sdk.channel])
    config = fake_sdk.infer.UpdateLogConfig.call_args.args[1]
    assert config.options["filter_lvl"] == 4
