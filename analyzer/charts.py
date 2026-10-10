import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_config import AxConfig


def plot_lines(ax: plt.Axes, ax_config: AxConfig):
    """
    通用的折线图绘制函数

    Args:
        ax: 要绘制的 Axes 对象
        ax_config: 包含 Axes 配置的 AxConfig 对象
    """
    data = ax_config.plot_data
    if data is None:
        raise ValueError(f"ax_config.plot_data must be specified.")

    # 遍历数据字典, 绘制每个折线
    for line_data in data:
        ax.plot(*line_data["args"], **(line_data.get("kwargs") or {}))

    if ax_config.title is not None:
        ax.set_title(ax_config.title, fontsize=ax_config.title_size)

    if ax_config.xlabel is not None:
        ax.set_xlabel(ax_config.xlabel, fontsize=ax_config.label_size)

    if ax_config.ylabel is not None:
        ax.set_ylabel(ax_config.ylabel, fontsize=ax_config.label_size)

    ax.tick_params(labelsize=ax_config.font_size)
    if ax_config.grid:
        ax.grid(True, alpha=ax_config.grid_alpha)
    ax.legend(loc=ax_config.legend_loc, fontsize=ax_config.font_size)


def plot_scatter_of_linear(slop1: np.ndarray, slop2: np.ndarray, title: str):
    """
    九宫格散点图对比
    """
    # 残差
    residual = slop2 - slop1

    # 3 sigma
    mu = residual.mean()
    sigma = residual.std()

    outlier_idx = np.where(np.abs(residual - mu) > 3 * sigma)[0]

    # print("outlier num =", len(outlier_idx))
    # print("outlier idx =", outlier_idx)

    # -------------------------
    # 画图
    # -------------------------
    plt.figure(figsize=(8, 8))

    # 所有点
    plt.scatter(slop1, slop2, c="red", s=8, alpha=0.5)

    # y=x
    min_v = min(slop1.min(), slop2.min())
    max_v = max(slop1.max(), slop2.max())

    plt.plot([min_v, max_v], [min_v, max_v], "b-", linewidth=2, label="y=x")

    # 离群点高亮
    plt.scatter(slop1[outlier_idx], slop2[outlier_idx], c="black", s=30, marker="x", label="outlier")

    # 标 index
    for idx in outlier_idx:
        plt.annotate(str(idx), (slop1[idx], slop2[idx]), fontsize=8, xytext=(3, 3), textcoords="offset points")

    plt.xlabel("slop1")
    plt.ylabel("slop2")
    plt.title("slop1 vs slop2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(title, dpi=300)
