import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from . import charts
from .charts import plot_scatter_of_linear
from .layer_analyzer import LayerAnalyzer
from .plot_config import CanvasConfig, AxConfig, PlotSpec


def draw_canvas(canvas_config: CanvasConfig, layout: list[PlotSpec]):
    """
    绘制 canvas.

    参数:
        canvas_config (CanvasConfig): 画布配置
        layout (list[PlotSpec]): 子图布局
    """
    fig, axes = plt.subplots(
        canvas_config.nrows,
        canvas_config.ncols,
        figsize=canvas_config.figsize,
        dpi=canvas_config.dpi,
        facecolor=canvas_config.facecolor,
    )

    # 保证 axes 永远是二维结构
    if canvas_config.nrows == 1 and canvas_config.ncols == 1:
        axes = [[axes]]
    elif canvas_config.nrows == 1:
        axes = [axes]
    elif canvas_config.ncols == 1:
        axes = [[ax] for ax in axes]

    used_positions = set()

    for spec in layout:
        if not (0 <= spec.row < canvas_config.nrows):
            raise ValueError(f"Row {spec.row} is out of range [0, {canvas_config.nrows - 1}]")

        if not (0 <= spec.col < canvas_config.ncols):
            raise ValueError(f"Col {spec.col} is out of range [0, {canvas_config.ncols - 1}]")

        key = (spec.row, spec.col)
        if key in used_positions:
            raise ValueError(f"Position {key} is already used.")
        used_positions.add(key)

        if spec.kwargs is None:
            spec.kwargs = {}

        analyzer = spec.analyzer
        plot_func = getattr(analyzer, spec.plot_method)

        ax = axes[spec.row][spec.col]
        ax_config = spec.ax_config
        plot_func(*spec.args, ax=ax, ax_config=ax_config, **spec.kwargs)

        if ax_config.title is not None:
            ax.set_title(ax_config.title)

    # 没用到的位置隐藏掉, 不显示 grid
    for r in range(canvas_config.nrows):
        for c in range(canvas_config.ncols):
            if (r, c) not in used_positions:
                axes[r][c].axis("off")

    if canvas_config.super_title is not None:
        fig.suptitle(canvas_config.super_title, fontsize=canvas_config.super_title_size)
        fig.tight_layout(rect=[0, 0, 1, canvas_config.super_title_pad])
    else:
        fig.tight_layout()

    if canvas_config.save_path is not None:
        fig.savefig(canvas_config.save_path, bbox_inches="tight")

    # 关闭 fig, 释放内存
    plt.close(fig)


def draw_basic_metrics(
    gold: np.ndarray,
    chip: np.ndarray,
    save_path: str,
    channel_dim: int = -1,
    diff_clip_val: tuple = (-128, 127),
    fields: list = None,
):
    """
    绘制 Precision Analysis Metrics 看板.

    参数:
        gold (np.ndarray):  golden 输出, shape 任意
        chip (np.ndarray):  chip 输出, shape 任意
        channel_dim (int):  通道维度, 默认为 -1 (最后一个维度)
        diff_clip_val (tuple): clip 值, 默认为 (-128, 127)
        save_path (str):  保存路径, 默认为 None
    """
    # 构造分析工具类
    analyzer = LayerAnalyzer(gold, chip, channel_dim=channel_dim, diff_clip_val=diff_clip_val)

    ROWS = 3
    COLS = 3

    if save_path is None:
        raise ValueError("save_path must be specified.")

    # 构造 Canvas 配置
    canvas_config = CanvasConfig(
        nrows=ROWS,
        ncols=COLS,
        figsize=(COLS * 8, ROWS * 5),
        super_title="Precision Analysis Metrics",
        save_path=save_path,
    )

    analysis_single_frame = gold.shape[0] - 1

    # 构造 layout 布局
    layout = [
        PlotSpec(analyzer, "plot_single_frame_out", row=0, col=0, args=(analysis_single_frame,)),
        PlotSpec(analyzer, "plot_error_statistics", row=0, col=1),
        PlotSpec(analyzer, "plot_cosine_similarity", row=0, col=2),
        PlotSpec(analyzer, "plot_single_frame_err", row=1, col=0, args=(analysis_single_frame,)),
        PlotSpec(analyzer, "plot_std0_2d", row=1, col=1),
        PlotSpec(analyzer, "plot_cosine_similarity_output", row=1, col=2),
        PlotSpec(analyzer, "plot_fitting", row=2, col=0),
        PlotSpec(analyzer, "plot_slope_of_linear", row=2, col=1),
        PlotSpec(analyzer, "plot_fitting_of_add_diff", row=2, col=2),
    ]

    draw_canvas(canvas_config, layout)

    if fields is None:
        fields = ["error_std0", "error_mean0", "cs_frame", "cs_output"]

    return analyzer.to_dict(fields=fields)


def draw_line_chart(
    data: List[AxConfig], save_path: str, ncols: int = 1, figsize: tuple = None, super_title: str = None
):
    """
    绘制折线图, 根据 ncols 自定义子图布局.

    Args:
        data: 折线数据
        save_path: 保存路径
        ncols: 子图列数, 默认为 1
        super_title: 超标题, 默认为 None
    """
    # 计算子图行数
    nrows = (len(data) + ncols - 1) // ncols

    if save_path is None:
        raise ValueError("save_path must be specified.")

    # 构造 Canvas 配置
    canvas_config = CanvasConfig(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize if figsize else (ncols * 8, nrows * 5),
        save_path=save_path,
    )

    if super_title is not None:
        canvas_config.super_title = super_title

    # 构造 layout 布局
    layout = []
    for i, ax_data in enumerate(data):
        layout.append(PlotSpec(charts, "plot_lines", row=i // ncols, col=i % ncols, ax_config=ax_data))

    draw_canvas(canvas_config, layout)


def draw_single_heatmap(gold: np.ndarray, chip: np.ndarray, save_path: str, channel_dim: int = -1):
    """
    绘制单个 heatmap 子图.

    Args:
        gold (np.ndarray):  golden 输出, shape 任意
        chip (np.ndarray):  chip 输出, shape 任意
        channel_dim (int):  通道维度, 默认为 -1 (最后一个维度)
        save_path (str):  保存路径, 默认为 None

    Returns:
        dict:  heatmap 数据
    """
    # 构造分析工具类
    analyzer = LayerAnalyzer(gold, chip, channel_dim=channel_dim)

    if save_path is None:
        raise ValueError("save_path must be specified.")

    # 构造 Canvas 配置
    canvas_config = CanvasConfig(
        nrows=1,
        ncols=1,
        figsize=(20, 13),
        save_path=save_path,
    )

    # 构造 layout 布局
    layout = [
        PlotSpec(analyzer, "plot_diff_heatmap", row=0, col=0),
    ]

    draw_canvas(canvas_config, layout)

    return analyzer.to_dict(fields=["error_2d"])


## sepcial for infer-order
def draw_multi_metrics(
    res: List[tuple], save_path: str, super_title: str, channel_dim: int = -1, diff_clip_val: tuple = (-128, 127)
):
    """
    绘制 Precision Analysis Metrics 看板.
    """
    # 构造分析工具类
    analyzers = [None] * len(res)
    for idx, (gold, chip) in enumerate(res):
        analyzers[idx] = LayerAnalyzer(
            gold,
            chip,
            channel_dim=channel_dim,
            diff_clip_val=diff_clip_val,
        )

    ROWS = 9
    COLS = len(analyzers)

    if save_path is None:
        raise ValueError("save_path must be specified.")

    # 构造 Canvas 配置
    canvas_config = CanvasConfig(
        nrows=ROWS,
        ncols=COLS,
        figsize=(COLS * 8, ROWS * 5),
        super_title=super_title,
        save_path=save_path,
    )

    analysis_single_frame = gold.shape[0] - 1

    # 构造 layout 布局
    layout = []
    for i in range(len(analyzers)):
        layout.extend(
            [
                PlotSpec(analyzers[i], "plot_single_frame_out", row=0, col=i, args=(analysis_single_frame,)),
                PlotSpec(analyzers[i], "plot_error_statistics", row=2, col=i),
                PlotSpec(analyzers[i], "plot_cosine_similarity", row=4, col=i),
                PlotSpec(analyzers[i], "plot_single_frame_err", row=1, col=i, args=(analysis_single_frame,)),
                PlotSpec(analyzers[i], "plot_std0_2d", row=3, col=i),
                PlotSpec(analyzers[i], "plot_cosine_similarity_output", row=5, col=i),
                PlotSpec(analyzers[i], "plot_fitting", row=6, col=i),
                PlotSpec(analyzers[i], "plot_slope_of_linear", row=7, col=i),
                PlotSpec(analyzers[i], "plot_fitting_of_add_diff", row=8, col=i),
            ]
        )

    draw_canvas(canvas_config, layout)

    # sub = analyzers[1].to_dict(fields=["coff"])
    # shf = analyzers[2].to_dict(fields=["coff"])

    # sub = [row[0] for row in sub['coff']]
    # shf = [row[0] for row in shf['coff']]

    # sub = np.asarray(sub)
    # shf = np.asarray(shf)

    # plot_scatter_of_linear(sub, shf, "sub_vs_shf.png")
