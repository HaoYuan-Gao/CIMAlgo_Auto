import numpy as np

from analyzer.canvas import draw_basic_metrics, draw_canvas, draw_line_chart
from analyzer.layer_analyzer import LayerAnalyzer
from analyzer.plot_config import AxConfig, CanvasConfig, PlotSpec


def _make_outputs(shape, seed=0):
    rng = np.random.default_rng(seed)
    gold = rng.integers(-128, 128, size=shape).astype(np.float32)
    noise = rng.normal(0, 3, size=shape).astype(np.float32)
    return gold, gold + noise


def _assert_image_created(path):
    assert path.is_file()
    assert path.stat().st_size > 0


def test_draw_2darray_basic_metrics(tmp_path):
    gold, chip = _make_outputs((20, 8))
    save_path = tmp_path / "2darray.png"

    metrics = draw_basic_metrics(gold, chip, save_path=str(save_path))

    _assert_image_created(save_path)
    assert set(metrics) == {"error_std0", "error_mean0", "cs_frame", "cs_output"}
    assert np.asarray(metrics["error_std0"]).shape == (8,)
    assert np.asarray(metrics["cs_frame"]).shape == (20,)


def test_draw_nchw_basic_metrics(tmp_path):
    gold, chip = _make_outputs((2, 3, 4, 4))
    save_path = tmp_path / "nchw.png"

    metrics = draw_basic_metrics(
        gold,
        chip,
        channel_dim=1,
        save_path=str(save_path),
    )

    _assert_image_created(save_path)
    assert np.asarray(metrics["error_std0"]).shape == (3,)
    assert np.asarray(metrics["cs_frame"]).shape == (32,)
    assert np.asarray(metrics["cs_output"]).shape == (3,)


def test_get_selected_analysis_metrics(tmp_path):
    gold, chip = _make_outputs((20, 8))
    save_path = tmp_path / "metrics.png"
    fields = ["error_2d", "error_std0", "error_std1", "cs_frame", "cs_output"]

    metrics = draw_basic_metrics(
        gold,
        chip,
        save_path=str(save_path),
        fields=fields,
    )

    _assert_image_created(save_path)
    assert set(metrics) == set(fields)
    assert np.asarray(metrics["error_2d"]).shape == (20, 8)
    assert np.asarray(metrics["error_std0"]).shape == (8,)
    assert np.asarray(metrics["error_std1"]).shape == (20,)


def test_draw_custom_metrics(tmp_path):
    gold, chip = _make_outputs((20, 8))
    save_path = tmp_path / "custom_metrics.png"
    analyzer = LayerAnalyzer(
        gold,
        chip,
        channel_dim=-1,
        diff_clip_val=(-128, 127),
    )
    canvas_config = CanvasConfig(
        nrows=1,
        ncols=2,
        figsize=(16, 5),
        save_path=str(save_path),
    )
    layout = [
        PlotSpec(analyzer, "plot_single_frame_out", row=0, col=0, args=(0,)),
        PlotSpec(analyzer, "plot_single_frame_err", row=0, col=1, args=(0,)),
    ]

    draw_canvas(canvas_config, layout)

    _assert_image_created(save_path)


def test_draw_lines(tmp_path):
    rng = np.random.default_rng(0)
    save_path = tmp_path / "lines.png"
    data = []

    for index in range(4):
        values = rng.integers(-128, 128, size=rng.integers(10, 30)).astype(np.float32)
        data.append(
            AxConfig(
                title=f"Chart {index}",
                xlabel="id",
                ylabel="value",
                plot_data=[
                    {
                        "args": (values,),
                        "kwargs": {"label": f"line-{index}"},
                    }
                ],
            )
        )

    draw_line_chart(data, save_path=str(save_path), ncols=2)

    _assert_image_created(save_path)
