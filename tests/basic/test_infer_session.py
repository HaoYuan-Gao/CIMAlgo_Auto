"""BW01D 推理会话的生命周期测试，不连接真实设备。"""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from utils.infer_session import InferSession

INPUT_DATA = np.array([[1, 2], [3, 4]], dtype=np.int8)
OUTPUT_DATA = np.array([[10], [20]], dtype=np.int8)


@pytest.fixture
def result_dir(tmp_path):
    return tmp_path / "nested" / "results"


@pytest.fixture
def model_dir(tmp_path):
    """按 L2Infer 约定的目录结构生成编译产物; 返回 (目录, 添加节点的函数)"""
    root = tmp_path / "model"
    root.mkdir()

    def add_node(l2_id, l0_id, node_id, *, input_data=INPUT_DATA, output_data=OUTPUT_DATA):
        node = root / f"L2_{l2_id}_L0_{l0_id}_node_{node_id}"
        node.mkdir()
        np.save(node / "input_data.npy", input_data)
        np.save(node / "output_data.npy", output_data)
        return node

    return root, add_node


@pytest.fixture
def analyzer(monkeypatch):
    """替换绘图模块, 只记录调用, 不真的画图"""
    module = ModuleType("analyzer")
    module.draw_basic_metrics = Mock()
    monkeypatch.setitem(sys.modules, "analyzer", module)
    return module.draw_basic_metrics


@pytest.fixture
def sdk_infer(monkeypatch):
    wtm = ModuleType("wtm")
    bw01d = ModuleType("wtm.bw01d")
    infer = ModuleType("wtm.bw01d.infer")
    pb = ModuleType("infer_service_pb2")

    class InferMode:
        def __init__(self, **options):
            self.options = options

    pb.InferMode = InferMode
    infer.init = Mock(return_value=True)
    infer.deinit = Mock(return_value=True)
    infer.L2Infer = Mock(return_value=[SimpleNamespace(success=True, data=OUTPUT_DATA.copy(), error=None)])
    for name in ("Calib", "SetOcState", "ReplaceOcParams", "UpdateLogConfig"):
        setattr(infer, name, Mock(return_value=SimpleNamespace(success=True, error="")))
    infer.GetOcInfo = Mock(return_value=SimpleNamespace(status=SimpleNamespace(success=True, error="")))

    class LogConfig:
        def __init__(self, **options):
            self.options = options

    infer.LogConfig = LogConfig
    bw01d.infer = infer
    wtm.bw01d = bw01d
    monkeypatch.setitem(sys.modules, "wtm", wtm)
    monkeypatch.setitem(sys.modules, "wtm.bw01d", bw01d)
    monkeypatch.setitem(sys.modules, "wtm.bw01d.infer", infer)
    monkeypatch.setitem(sys.modules, "infer_service_pb2", pb)
    return infer


def test_infer_session_initializes_and_deinitializes_only_its_handles(sdk_infer, analyzer, result_dir, model_dir):
    root, add_node = model_dir
    add_node(0, 1, 2)
    handles = [object(), object()]
    session = InferSession(handles, result_dir, model_name="demo", mode="vp_erpc")
    assert result_dir.is_dir()

    with session as infer:
        results = infer.L2Infer(root)
        sdk_infer.init.assert_called_once_with(handles, model_name="demo", mode="vp_erpc")
        sdk_infer.deinit.assert_not_called()

    assert list(results) == ["L2_0_L0_1_Node_2"]
    assert results["L2_0_L0_1_Node_2"] is sdk_infer.L2Infer.return_value[0]
    sdk_infer.L2Infer.assert_called_once()
    args = sdk_infer.L2Infer.call_args
    assert args.args[0] == 0
    assert args.args[1] == [(1, 2)]
    np.testing.assert_array_equal(args.args[2][0], INPUT_DATA)
    (infer_mode,) = args.args[3]
    assert infer_mode.options == {
        "enable_debug_LORA": False,
        "enable_debug_RAS": False,
        "enable_debug_Temp": False,
        "enable_debug_PN": False,
        "enable_debug_PN_diff_only": False,
        "enable_debug_RAW": False,
        "enable_profile": False,
        "enable_debug_calib": False,
        "repeat": 1,
        "big_small_mode": 0,
        "sub_batch_size": 20,
    }
    assert args.kwargs == {"channel_handles": handles}
    np.testing.assert_array_equal(np.load(result_dir / "L2_0_L0_1_Node_2.npy"), OUTPUT_DATA)
    sdk_infer.deinit.assert_called_once_with(handles)


def test_l2_infer_dict_overrides_default_mode_without_mutating_input(sdk_infer, analyzer, result_dir, model_dir):
    root, add_node = model_dir
    add_node(1, 0, 2)
    updates = {"enable_debug_RAW": True, "repeat": 3, "sub_batch_size": 8}

    with InferSession(object(), result_dir) as session:
        session.L2Infer(root, infer_modes=updates)

    (infer_mode,) = sdk_infer.L2Infer.call_args.args[3]
    assert infer_mode.options["enable_debug_RAW"] is True
    assert infer_mode.options["repeat"] == 3
    assert infer_mode.options["sub_batch_size"] == 8
    assert infer_mode.options["enable_debug_LORA"] is False
    assert updates == {"enable_debug_RAW": True, "repeat": 3, "sub_batch_size": 8}


def test_l2_infer_rejects_non_dict_modes(sdk_infer, analyzer, result_dir, model_dir):
    root, add_node = model_dir
    add_node(0, 0, 1)

    with InferSession(object(), result_dir) as session:
        with pytest.raises(TypeError, match="infer_modes 必须是 dict"):
            session.L2Infer(root, infer_modes=[object()])

    sdk_infer.L2Infer.assert_not_called()


@pytest.mark.parametrize("dummy", [-1, 1.5, True, "1"])
def test_l2_infer_rejects_invalid_dummy(sdk_infer, analyzer, result_dir, model_dir, dummy):
    root, add_node = model_dir
    add_node(0, 0, 1)

    with InferSession(object(), result_dir) as session:
        with pytest.raises(ValueError, match="dummy 必须是非负整数"):
            session.L2Infer(root, dummy=dummy)

    sdk_infer.L2Infer.assert_not_called()


def test_l2_infer_rejects_missing_model_dir(sdk_infer, analyzer, result_dir, tmp_path):
    with InferSession(object(), result_dir) as session:
        with pytest.raises(NotADirectoryError, match="模型目录不存在"):
            session.L2Infer(tmp_path / "missing")

    sdk_infer.L2Infer.assert_not_called()


@pytest.mark.parametrize("name", ["L2_0_L0_1", "node_1", "L2_a_L0_1_node_2", "L2_5_L0_6_Node_7", "extra"])
def test_l2_infer_rejects_unexpected_node_dir(sdk_infer, analyzer, result_dir, model_dir, name):
    root, add_node = model_dir
    add_node(0, 1, 2)
    (root / name).mkdir()

    with InferSession(object(), result_dir) as session:
        with pytest.raises(ValueError, match="节点目录名必须形如") as raised:
            session.L2Infer(root)

    assert name in str(raised.value)


def test_l2_infer_ignores_plain_files_in_model_dir(sdk_infer, analyzer, result_dir, model_dir):
    root, add_node = model_dir
    add_node(0, 1, 2)
    (root / "README.txt").write_text("not a node", encoding="utf-8")

    with InferSession(object(), result_dir) as session:
        results = session.L2Infer(root)

    assert list(results) == ["L2_0_L0_1_Node_2"]
    sdk_infer.L2Infer.assert_called_once()


def test_l2_infer_trims_dummy_frames_and_saves_prefixed_result(sdk_infer, analyzer, result_dir, model_dir):
    root, add_node = model_dir
    add_node(0, 1, 2)
    sdk_infer.L2Infer.return_value = [
        SimpleNamespace(success=True, data=np.array([[99], [10], [20]], dtype=np.int8), error=None)
    ]

    with InferSession(object(), result_dir) as session:
        results = session.L2Infer(root, dummy=1, prefix_name="case")

    np.testing.assert_array_equal(results["case_L2_0_L0_1_Node_2"].data, [[10], [20]])
    np.testing.assert_array_equal(np.load(result_dir / "case_L2_0_L0_1_Node_2.npy"), [[10], [20]])
    assert analyzer.call_args.kwargs["save_path"] == str(result_dir / "case_L2_0_L0_1_Node_2.png")


def test_l2_infer_reports_sdk_failure_before_saving(sdk_infer, analyzer, result_dir, model_dir):
    root, add_node = model_dir
    add_node(0, 1, 2)
    sdk_infer.L2Infer.return_value = [SimpleNamespace(success=False, data=None, error="设备推理失败")]

    with InferSession(object(), result_dir) as session:
        with pytest.raises(RuntimeError, match="设备推理失败"):
            session.L2Infer(root)

    assert list(result_dir.iterdir()) == []
    analyzer.assert_not_called()


def test_l2_infer_rejects_empty_results(sdk_infer, analyzer, result_dir, model_dir):
    root, add_node = model_dir
    add_node(0, 1, 2)
    sdk_infer.L2Infer.return_value = []

    with InferSession(object(), result_dir) as session:
        with pytest.raises(RuntimeError, match="未返回结果"):
            session.L2Infer(root)

    assert list(result_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("data", "dummy", "frames"),
    [
        (None, 0, 0),
        (np.zeros((0, 1), dtype=np.int8), 0, 0),
        (np.array([[99]], dtype=np.int8), 1, 1),
        (np.array([[99], [98]], dtype=np.int8), 3, 2),
    ],
)
def test_l2_infer_rejects_results_without_enough_frames(sdk_infer, analyzer, result_dir, model_dir, data, dummy, frames):
    root, add_node = model_dir
    add_node(0, 1, 2)
    sdk_infer.L2Infer.return_value = [SimpleNamespace(success=True, data=data, error=None)]

    with InferSession(object(), result_dir) as session:
        with pytest.raises(RuntimeError, match=f"L2_0_L0_1_Node_2 推理返回 {frames} 帧, 不足以去掉 {dummy} 个 dummy 帧"):
            session.L2Infer(root, dummy=dummy)

    assert list(result_dir.iterdir()) == []
    analyzer.assert_not_called()


def test_l2_infer_runs_each_node_dir_in_sorted_order(sdk_infer, analyzer, result_dir, model_dir):
    root, add_node = model_dir
    second_input = np.array([[5, 6], [7, 8]], dtype=np.int8)
    add_node(0, 3, 4, input_data=second_input)
    add_node(0, 1, 2)
    add_node(1, 0, 0)

    with InferSession(object(), result_dir) as session:
        results = session.L2Infer(root)

    assert list(results) == ["L2_0_L0_1_Node_2", "L2_0_L0_3_Node_4", "L2_1_L0_0_Node_0"]
    assert sdk_infer.L2Infer.call_count == 3
    calls = sdk_infer.L2Infer.call_args_list
    assert [(call.args[0], call.args[1]) for call in calls] == [(0, [(1, 2)]), (0, [(3, 4)]), (1, [(0, 0)])]
    np.testing.assert_array_equal(calls[1].args[2][0], second_input)
    for name in results:
        assert (result_dir / f"{name}.npy").is_file()
    assert analyzer.call_count == 3


def test_l2_infer_draws_metrics_against_node_golden(sdk_infer, analyzer, result_dir, model_dir):
    root, add_node = model_dir
    golden = np.array([[7], [8]], dtype=np.int8)
    add_node(0, 1, 2, output_data=golden)
    sdk_infer.L2Infer.return_value = [
        SimpleNamespace(success=True, data=np.array([[99], [10], [20]], dtype=np.int8), error=None)
    ]

    with InferSession(object(), result_dir) as session:
        session.L2Infer(root, dummy=1)

    analyzer.assert_called_once()
    kwargs = analyzer.call_args.kwargs
    np.testing.assert_array_equal(kwargs["gold"], golden)
    np.testing.assert_array_equal(kwargs["chip"], [[10], [20]])
    assert kwargs["save_path"] == str(result_dir / "L2_0_L0_1_Node_2.png")


@pytest.mark.parametrize("dummy", [0, 1, 3])
def test_l2_infer_repeats_first_frame_then_removes_dummy_frames(sdk_infer, analyzer, result_dir, model_dir, dummy):
    root, add_node = model_dir
    add_node(0, 1, 2)

    def echo_input(_l2_id, _node_ids, padded_inputs, _modes, *, channel_handles):
        return [SimpleNamespace(success=True, data=padded_inputs[0].copy(), error=None)]

    sdk_infer.L2Infer.side_effect = echo_input

    with InferSession(object(), result_dir) as session:
        results = session.L2Infer(root, dummy=dummy)

    padded_input = sdk_infer.L2Infer.call_args.args[2][0]
    np.testing.assert_array_equal(padded_input, np.array([[1, 2]] * dummy + [[1, 2], [3, 4]], dtype=np.int8))
    np.testing.assert_array_equal(results["L2_0_L0_1_Node_2"].data, INPUT_DATA)
    np.testing.assert_array_equal(np.load(result_dir / "L2_0_L0_1_Node_2.npy"), INPUT_DATA)


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("Calib", (0, 1)),
        ("SetOcState", (0, 1, 2, 3, 4, True)),
        ("ReplaceOcParams", (0, 1, 2, b"kb", b"bb", b"ks", b"bs", False)),
        ("GetOcInfo", (0, 1, 2, True)),
    ],
)
def test_infer_session_methods_use_its_handles(sdk_infer, name, args, result_dir):
    handle = object()

    with InferSession(handle, result_dir) as session:
        assert getattr(session, name)(*args) is getattr(sdk_infer, name).return_value

    getattr(sdk_infer, name).assert_called_once_with(*args, channel_handles=[handle])


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("Calib", (0, 1)),
        ("SetOcState", (0, 1, 2, 3, 4, True)),
        ("ReplaceOcParams", (0, 1, 2, b"kb", b"bb", b"ks", b"bs", False)),
        ("GetOcInfo", (0, 1, 2, True)),
        ("UpdateLogConfig", (0, 4)),
    ],
)
def test_infer_session_rejects_failed_sdk_status(sdk_infer, name, args, result_dir):
    status = SimpleNamespace(success=False, error="设备返回失败")
    response = SimpleNamespace(status=status) if name == "GetOcInfo" else status
    getattr(sdk_infer, name).return_value = response

    with InferSession(object(), result_dir) as session:
        with pytest.raises(RuntimeError, match=f"{name} 失败: 设备返回失败"):
            getattr(session, name)(*args)

    sdk_infer.deinit.assert_called_once()


def test_update_log_config_builds_sdk_config_from_log_level(sdk_infer, result_dir):
    handle = object()

    with InferSession(handle, result_dir) as session:
        assert session.UpdateLogConfig(2, 4) is sdk_infer.UpdateLogConfig.return_value

    sdk_infer.UpdateLogConfig.assert_called_once()
    assert sdk_infer.UpdateLogConfig.call_args.args[0] == 2
    config = sdk_infer.UpdateLogConfig.call_args.args[1]
    assert isinstance(config, sdk_infer.LogConfig)
    assert config.options == {
        "log_open": True,
        "log_color": True,
        "filter_lvl": 4,
        "filter_tag": [],
        "filter_keyword": [],
        "log_path": "/var/log/bw01d/wit/runtime.log",
        "max_size": 10 * 1024 * 1024,
        "max_rotate": 5,
    }
    assert sdk_infer.UpdateLogConfig.call_args.kwargs == {"channel_handles": [handle]}


def test_infer_session_exposes_log_config_type(sdk_infer, result_dir):
    with InferSession(object(), result_dir) as session:
        config = session.LogConfig(log_open=False)

    assert isinstance(config, sdk_infer.LogConfig)
    assert config.options == {"log_open": False}


def test_infer_session_methods_require_active_context(result_dir):
    session = InferSession(object(), result_dir)

    with pytest.raises(RuntimeError, match="请在 with 内调用"):
        session.Calib(0, 1)


def test_infer_session_deinitializes_when_body_fails(sdk_infer, result_dir):
    handle = object()

    with pytest.raises(ValueError, match="test failed"):
        with InferSession(handle, result_dir):
            raise ValueError("test failed")

    sdk_infer.deinit.assert_called_once_with([handle])


def test_infer_session_does_not_enter_when_init_fails(sdk_infer, result_dir):
    sdk_infer.init.return_value = False

    with pytest.raises(RuntimeError, match="初始化失败"):
        with InferSession(object(), result_dir):
            pytest.fail("初始化失败后不应进入 with")

    sdk_infer.deinit.assert_not_called()


def test_infer_session_reports_deinit_failure(sdk_infer, result_dir):
    sdk_infer.deinit.return_value = False

    with pytest.raises(RuntimeError, match="deinit 返回 False"):
        with InferSession(object(), result_dir):
            pass


def test_infer_session_preserves_body_error_on_deinit_failure(sdk_infer, result_dir):
    sdk_infer.deinit.return_value = False

    with pytest.raises(ValueError, match="test failed") as raised:
        with InferSession(object(), result_dir):
            raise ValueError("test failed")

    assert any("deinit 返回 False" in note for note in raised.value.__notes__)


def test_infer_session_does_not_replace_file_at_result_dir(tmp_path):
    result_file = tmp_path / "results"
    result_file.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        InferSession(object(), result_file)

    assert result_file.read_text(encoding="utf-8") == "keep"
