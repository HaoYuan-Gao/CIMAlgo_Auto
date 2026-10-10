import logging
import threading

import numpy as np
import pytest

from cimalgo.analyzer import LayerAnalyzer, draw_basic_metrics, draw_heatmap, draw_line_chart
from cimalgo.analyzer.plot_config import AxConfig
from cimalgo.logging import configure_logging, get_logger

# ---------------------------------------------------------------------------
# analyzer
# ---------------------------------------------------------------------------


def test_error_is_computed_in_float_not_int8():
    golden = np.array([[127]], dtype=np.int8)
    chip = np.array([[-128]], dtype=np.int8)
    analyzer = LayerAnalyzer(golden, chip)
    # int8 直接相减会绕回成 -1; 正确值是 -255
    assert analyzer.error_2d[0, 0] == -255.0


def test_shapes_and_channel_dim():
    golden = np.zeros((2, 3, 4, 4))
    chip = np.ones((2, 3, 4, 4))
    analyzer = LayerAnalyzer(golden, chip, channel_dim=1)
    assert (analyzer.frame_num, analyzer.channel_num) == (32, 3)
    assert analyzer.error_std0.shape == (3,)
    assert analyzer.cs_frame.shape == (32,)
    with pytest.raises(ValueError, match="mismatch"):
        LayerAnalyzer(np.zeros((2, 3)), np.zeros((3, 2)))


def test_cosine_handles_zero_vectors_without_nan():
    analyzer = LayerAnalyzer(np.zeros((3, 4)), np.ones((3, 4)))
    assert not np.isnan(analyzer.cs_frame).any()
    assert analyzer.cs_overall == 0.0


def test_perfect_match_summary():
    data = np.arange(24, dtype=np.int8).reshape(6, 4)
    summary = LayerAnalyzer(data, data).summary()
    assert summary["cs_overall"] == pytest.approx(1.0)
    assert summary["error_std"] == 0.0
    assert summary["exact_match_ratio"] == 1.0
    assert summary["fit_slope"] == pytest.approx(1.0)


def test_channel_offset_removal():
    rng = np.random.default_rng(0)
    golden = rng.integers(-100, 100, size=(50, 3)).astype(np.float64)
    chip = golden + np.array([5, -3, 0])  # 每通道固定偏移
    analyzer = LayerAnalyzer(golden, chip)
    np.testing.assert_array_equal(analyzer.channel_offset, [-5, 3, 0])
    assert analyzer.add_error_2d.std() == pytest.approx(0.0)
    assert analyzer.error_std0.max() == pytest.approx(0.0)


def test_constant_column_fit_is_zero():
    golden = np.zeros((5, 2))
    golden[:, 1] = np.arange(5)
    analyzer = LayerAnalyzer(golden, golden)
    assert tuple(analyzer.coeff[0]) == (0.0, 0.0)
    assert analyzer.coeff[1][0] == pytest.approx(1.0)


def test_cached_property_computes_once(monkeypatch):
    analyzer = LayerAnalyzer(np.zeros((2, 2)), np.ones((2, 2)))
    first = analyzer.error_2d
    assert analyzer.error_2d is first


def test_to_dict_serializes_arrays():
    analyzer = LayerAnalyzer(np.zeros((2, 2)), np.ones((2, 2)))
    result = analyzer.to_dict(["error_mean0", "cs_overall"])
    assert result["error_mean0"] == [1.0, 1.0]
    assert isinstance(result["cs_overall"], float)


def test_draw_basic_metrics_writes_png_and_returns_summary(tmp_path):
    rng = np.random.default_rng(0)
    golden = rng.integers(-128, 128, size=(20, 8)).astype(np.float32)
    chip = golden + rng.normal(0, 2, size=golden.shape).astype(np.float32)
    path = tmp_path / "sub" / "m.png"
    summary = draw_basic_metrics(golden, chip, path)
    assert path.is_file() and path.stat().st_size > 0
    assert set(summary) >= {"cs_overall", "error_std", "fit_slope"}


def test_draw_heatmap_and_line_chart(tmp_path):
    golden = np.zeros((4, 4))
    chip = golden.copy()
    chip[0, 0] = 1
    error = draw_heatmap(golden, chip, tmp_path / "h.png")
    assert error[0, 0] == 1 and (tmp_path / "h.png").is_file()

    charts = [
        AxConfig(title=f"c{i}", plot_data=[{"args": (np.arange(5),), "kwargs": {"label": "l"}}]) for i in range(3)
    ]
    assert draw_line_chart(charts, tmp_path / "l.png", ncols=2) == tmp_path / "l.png"
    with pytest.raises(ValueError):
        draw_line_chart([], tmp_path / "x.png")


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_logging(monkeypatch):
    monkeypatch.delenv("CIM_LOG_FILE", raising=False)
    configure_logging(level="DEBUG", color=False, force=True)
    yield
    configure_logging(force=True)


def test_logger_names_are_prefixed(fresh_logging):
    assert get_logger("cimalgo.env").logger.name == "cimalgo.env"
    assert get_logger("env").logger.name == "cimalgo.env"
    assert get_logger().logger.name == "cimalgo"


def test_indent_and_tag_appear_in_output(fresh_logging, caplog):
    log = get_logger("test")
    with caplog.at_level(logging.DEBUG, logger="cimalgo"), log.tagged("compile"), log.indent(4):
        log.info("hello")
    record = caplog.records[-1]
    assert record.indent == 4 and record.tag == "compile"


def test_indent_is_thread_local(fresh_logging):
    log = get_logger("test")
    seen = {}

    def worker():
        from cimalgo.logging.logger import _indent

        seen["worker"] = _indent()

    with log.indent(8):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
    assert seen["worker"] == 0


def test_trace_logs_start_end_and_error(fresh_logging, caplog):
    log = get_logger("test")

    @log.trace("step")
    def ok():
        return 1

    @log.trace()
    def bad():
        raise ValueError("x")

    with caplog.at_level(logging.INFO, logger="cimalgo"):
        assert ok() == 1
        with pytest.raises(ValueError):
            bad()
    messages = [record.getMessage() for record in caplog.records]
    assert messages[0] == "Start: test_trace_logs_start_end_and_error.<locals>.ok"
    assert messages[1].startswith("End: test_trace_logs_start_end_and_error.<locals>.ok (")
    assert any(message.startswith("Error:") for message in messages)
    assert caplog.records[0].tag == "step"


def test_configure_logging_writes_file(tmp_path):
    log_file = tmp_path / "log" / "cim.log"
    configure_logging(level="INFO", log_file=log_file, color=False, force=True)
    try:
        get_logger("test").info("to file")
        for handler in logging.getLogger("cimalgo").handlers:
            handler.flush()
        assert "to file" in log_file.read_text()
    finally:
        configure_logging(force=True)
