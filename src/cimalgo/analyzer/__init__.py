"""推理结果分析与绘图: golden 与 chip 输出的误差统计、余弦相似度、线性拟合"""

from .canvas import draw_basic_metrics, draw_heatmap, draw_line_chart
from .layer_analyzer import LayerAnalyzer
from .plot_config import AxConfig, CanvasConfig, PlotSpec

__all__ = [
    "AxConfig",
    "CanvasConfig",
    "LayerAnalyzer",
    "PlotSpec",
    "draw_basic_metrics",
    "draw_heatmap",
    "draw_line_chart",
]
