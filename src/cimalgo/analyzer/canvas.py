"""把多个子图组合成一张画布并保存; matplotlib 只在这里导入, 后端在首次使用时设置"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .layer_analyzer import LayerAnalyzer
from .plot_config import AxConfig, CanvasConfig, PlotSpec

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _pyplot() -> Any:
    import matplotlib

    # 测试和 CI 没有显示器; 只在没人显式设置后端时才改
    if matplotlib.get_backend().lower() not in {"agg", "module://matplotlib_inline.backend_inline"}:
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def plot_lines(*, ax: Axes, ax_config: AxConfig) -> None:
    """按 ax_config.plot_data 绘制若干折线"""
    for line in ax_config.plot_data:
        ax.plot(*line["args"], **line.get("kwargs", {}))
    if ax_config.xlabel is not None:
        ax.set_xlabel(ax_config.xlabel, fontsize=ax_config.label_size)
    if ax_config.ylabel is not None:
        ax.set_ylabel(ax_config.ylabel, fontsize=ax_config.label_size)
    ax.tick_params(labelsize=ax_config.font_size)
    if ax_config.grid:
        ax.grid(True, alpha=ax_config.grid_alpha)
    if any("label" in line.get("kwargs", {}) for line in ax_config.plot_data):
        ax.legend(loc=ax_config.legend_loc, fontsize=ax_config.font_size)


class _Lines:
    plot_lines = staticmethod(plot_lines)


def draw_canvas(canvas: CanvasConfig, layout: Sequence[PlotSpec]) -> Path | None:
    """按 layout 画图; 未使用的位置隐藏; 保存后关闭 figure"""
    plt = _pyplot()
    fig, axes = plt.subplots(
        canvas.nrows, canvas.ncols, figsize=canvas.figsize, dpi=canvas.dpi, facecolor=canvas.facecolor, squeeze=False
    )
    try:
        used: set[tuple[int, int]] = set()
        for spec in layout:
            if not (0 <= spec.row < canvas.nrows and 0 <= spec.col < canvas.ncols):
                raise ValueError(f"PlotSpec ({spec.row}, {spec.col}) 超出 {canvas.nrows}x{canvas.ncols} 画布")
            if (spec.row, spec.col) in used:
                raise ValueError(f"位置 ({spec.row}, {spec.col}) 已被占用")
            used.add((spec.row, spec.col))
            ax = axes[spec.row][spec.col]
            getattr(spec.target, spec.method)(*spec.args, ax=ax, ax_config=spec.ax_config, **spec.kwargs)
            if spec.ax_config.title is not None:
                ax.set_title(spec.ax_config.title, fontsize=spec.ax_config.title_size)

        for row in range(canvas.nrows):
            for col in range(canvas.ncols):
                if (row, col) not in used:
                    axes[row][col].axis("off")

        if canvas.super_title is not None:
            fig.suptitle(canvas.super_title, fontsize=canvas.super_title_size)
            fig.tight_layout(rect=(0, 0, 1, canvas.super_title_pad))
        else:
            fig.tight_layout()

        if canvas.save_path is not None:
            path = Path(canvas.save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches="tight")
            return path
        return None
    finally:
        plt.close(fig)


def draw_basic_metrics(
    golden: Any,
    chip: Any,
    save_path: str | Path,
    *,
    channel_dim: int = -1,
    clip: tuple[int, int] = (-128, 127),
    frame_index: int | None = None,
) -> dict[str, float]:
    """3x3 精度看板; 返回 LayerAnalyzer.summary()"""
    analyzer = LayerAnalyzer(golden, chip, channel_dim=channel_dim, clip=clip)
    index = analyzer.frame_num - 1 if frame_index is None else frame_index
    canvas = CanvasConfig(
        nrows=3, ncols=3, figsize=(24, 15), super_title="Precision Analysis Metrics", save_path=str(save_path)
    )
    layout = [
        PlotSpec(analyzer, "plot_single_frame_out", 0, 0, args=(index,)),
        PlotSpec(analyzer, "plot_error_statistics", 0, 1),
        PlotSpec(analyzer, "plot_cosine_similarity", 0, 2),
        PlotSpec(analyzer, "plot_single_frame_err", 1, 0, args=(index,)),
        PlotSpec(analyzer, "plot_std0_2d", 1, 1),
        PlotSpec(analyzer, "plot_cosine_similarity_output", 1, 2),
        PlotSpec(analyzer, "plot_fitting", 2, 0),
        PlotSpec(analyzer, "plot_slope_of_linear", 2, 1),
        PlotSpec(analyzer, "plot_fitting_of_add_diff", 2, 2),
    ]
    draw_canvas(canvas, layout)
    return analyzer.summary()


def draw_heatmap(golden: Any, chip: Any, save_path: str | Path, *, channel_dim: int = -1) -> np.ndarray:
    """二值匹配热力图; 返回 error_2d"""
    analyzer = LayerAnalyzer(golden, chip, channel_dim=channel_dim)
    draw_canvas(
        CanvasConfig(nrows=1, ncols=1, figsize=(20, 13), save_path=str(save_path)),
        [PlotSpec(analyzer, "plot_diff_heatmap", 0, 0)],
    )
    return analyzer.error_2d


def draw_line_chart(
    charts: Sequence[AxConfig],
    save_path: str | Path,
    *,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    super_title: str | None = None,
) -> Path | None:
    if not charts:
        raise ValueError("charts 不能为空")
    nrows = (len(charts) + ncols - 1) // ncols
    canvas = CanvasConfig(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize or (ncols * 8, nrows * 5),
        super_title=super_title,
        save_path=str(save_path),
    )
    layout = [PlotSpec(_Lines, "plot_lines", i // ncols, i % ncols, ax_config=cfg) for i, cfg in enumerate(charts)]
    return draw_canvas(canvas, layout)
