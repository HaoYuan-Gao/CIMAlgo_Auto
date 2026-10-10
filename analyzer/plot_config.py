from dataclasses import dataclass, field
from typing import Optional, Tuple, Any


@dataclass
class CanvasConfig:
    # 画布配置
    nrows: int
    ncols: int

    figsize: Tuple[int, int] = (18, 12)
    dpi: int = 150
    facecolor: str = "white"

    super_title: Optional[str] = None
    super_title_size: int = 18
    # 超标题 Top Padding
    super_title_pad: float = 0.98

    save_path: str = "canvas.png"


@dataclass
class AxConfig:
    font_size: int = 12
    title_size: int = 16
    label_size: int = 12

    grid: bool = True
    grid_alpha: float = 0.7

    legend_loc: str = "best"

    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    title: Optional[str] = None

    # plot 绘图数据配置
    # 列表的每一项是一个字典，其元素和 ax.plot 函数的参数相同, 每一项表示一条折线
    # 例如: [{"args": (x, y, "b-*"), "kwargs": {"label": "line1"}}, ...]
    plot_data: list = field(default_factory=list)


@dataclass
class PlotSpec:
    # 分析器实例
    analyzer: Any
    plot_method: str

    row: int
    col: int

    # 函数参数配置
    args: tuple = ()
    kwargs: Optional[dict] = None

    # 子图配置
    ax_config: AxConfig = field(default_factory=AxConfig)
