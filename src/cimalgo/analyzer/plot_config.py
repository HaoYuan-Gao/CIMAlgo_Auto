from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CanvasConfig:
    nrows: int
    ncols: int
    figsize: tuple[float, float] = (18, 12)
    dpi: int = 150
    facecolor: str = "white"
    super_title: str | None = None
    super_title_size: int = 18
    super_title_pad: float = 0.98
    save_path: str | None = None


@dataclass
class AxConfig:
    font_size: int = 12
    title_size: int = 16
    label_size: int = 12
    grid: bool = True
    grid_alpha: float = 0.7
    legend_loc: str = "best"
    xlabel: str | None = None
    ylabel: str | None = None
    title: str | None = None
    # 每项是 {"args": (x, y, fmt), "kwargs": {"label": ...}}, 对应一次 ax.plot 调用
    plot_data: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PlotSpec:
    """把 target 上名为 method 的绘图函数放到 (row, col)"""

    target: Any
    method: str
    row: int
    col: int
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    ax_config: AxConfig = field(default_factory=AxConfig)
