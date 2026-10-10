"""
单层输出的精度分析

所有指标按 (frame, channel) 的二维视图计算; 误差 = chip - golden, 以 float64 计算避免 int8 溢出。
"add" 系列指标是在 chip 上加回每通道的平均偏移后再算一遍, 用来区分系统偏移和随机噪声。
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from .plot_config import AxConfig

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def to_2d(array: Any, channel_dim: int = -1) -> np.ndarray:
    """任意形状 -> (N, C), 通道维移到最后"""
    array = np.asarray(array)
    if array.ndim == 0:
        raise ValueError("array must have at least one dimension")
    if array.ndim == 1:
        if channel_dim not in (-1, 0):
            raise ValueError(f"Invalid channel_dim={channel_dim} for 1D array")
        return array.reshape(1, -1)
    if channel_dim < 0:
        channel_dim += array.ndim
    if not 0 <= channel_dim < array.ndim:
        raise ValueError(f"Invalid channel_dim={channel_dim} for array with ndim={array.ndim}")
    moved = np.moveaxis(array, channel_dim, -1)
    return moved.reshape(-1, moved.shape[-1])


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐行余弦相似度; 零向量对应位置为 0 (不是 NaN)"""
    dot = np.einsum("ij,ij->i", a, b)
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm > 0, dot / norm, 0.0)


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """最小二乘 y = k x + b; x 常量时返回 (0, 0)"""
    if x.size < 2 or np.all(x == x[0]):
        return 0.0, 0.0
    try:
        slope, intercept = np.polyfit(x, y, deg=1)
    except np.linalg.LinAlgError:
        return 0.0, 0.0
    return float(slope), float(intercept)


class LayerAnalyzer:
    def __init__(self, golden: Any, chip: Any, *, channel_dim: int = -1, clip: tuple[int, int] = (-128, 127)):
        self.golden_2d = to_2d(golden, channel_dim).astype(np.float64)
        self.chip_2d = to_2d(chip, channel_dim).astype(np.float64)
        if self.golden_2d.shape != self.chip_2d.shape:
            raise ValueError(
                f"golden and chip shape mismatch after reshape: {self.golden_2d.shape} vs {self.chip_2d.shape}"
            )
        self.clip = clip
        self.frame_num, self.channel_num = self.golden_2d.shape

    # ---- 基础误差 ----

    @cached_property
    def error_2d(self) -> np.ndarray:
        return self.chip_2d - self.golden_2d

    @cached_property
    def error_mean0(self) -> np.ndarray:
        """每通道误差均值"""
        return self.error_2d.mean(axis=0)

    @cached_property
    def error_std0(self) -> np.ndarray:
        return self.error_2d.std(axis=0)

    @cached_property
    def error_mean1(self) -> np.ndarray:
        """每帧误差均值"""
        return self.error_2d.mean(axis=1)

    @cached_property
    def error_std1(self) -> np.ndarray:
        return self.error_2d.std(axis=1)

    # ---- 去掉每通道系统偏移后的误差 ----

    @cached_property
    def channel_offset(self) -> np.ndarray:
        """每通道 golden - chip 的平均值, 取整"""
        return np.rint((self.golden_2d - self.chip_2d).mean(axis=0))

    @cached_property
    def chip_add_2d(self) -> np.ndarray:
        return np.clip(self.chip_2d + self.channel_offset, *self.clip)

    @cached_property
    def add_error_2d(self) -> np.ndarray:
        return self.chip_add_2d - self.golden_2d

    @cached_property
    def add_error_mean0(self) -> np.ndarray:
        return self.add_error_2d.mean(axis=0)

    @cached_property
    def add_error_std0(self) -> np.ndarray:
        return self.add_error_2d.std(axis=0)

    @cached_property
    def add_error_mean1(self) -> np.ndarray:
        return self.add_error_2d.mean(axis=1)

    @cached_property
    def add_error_std1(self) -> np.ndarray:
        return self.add_error_2d.std(axis=1)

    # ---- 余弦相似度 ----

    @cached_property
    def cs_frame(self) -> np.ndarray:
        return _cosine_rows(self.chip_2d, self.golden_2d)

    @cached_property
    def cs_output(self) -> np.ndarray:
        return _cosine_rows(self.chip_2d.T, self.golden_2d.T)

    @cached_property
    def cs_overall(self) -> float:
        return float(_cosine_rows(self.chip_2d.reshape(1, -1), self.golden_2d.reshape(1, -1))[0])

    @cached_property
    def cs_frame_add(self) -> np.ndarray:
        return _cosine_rows(self.chip_add_2d, self.golden_2d)

    @cached_property
    def cs_output_add(self) -> np.ndarray:
        return _cosine_rows(self.chip_add_2d.T, self.golden_2d.T)

    @cached_property
    def cs_overall_add(self) -> float:
        return float(_cosine_rows(self.chip_add_2d.reshape(1, -1), self.golden_2d.reshape(1, -1))[0])

    # ---- 线性拟合 ----

    @cached_property
    def fit(self) -> tuple[float, float]:
        """全局 (slope, intercept)"""
        return _linear_fit(self.golden_2d.ravel(), self.chip_2d.ravel())

    @cached_property
    def fit_add(self) -> tuple[float, float]:
        return _linear_fit(self.golden_2d.ravel(), self.chip_add_2d.ravel())

    @cached_property
    def coeff(self) -> np.ndarray:
        """每通道 (slope, intercept), shape (C, 2)"""
        return np.array([_linear_fit(self.golden_2d[:, c], self.chip_2d[:, c]) for c in range(self.channel_num)])

    @cached_property
    def coeff_add(self) -> np.ndarray:
        return np.array([_linear_fit(self.golden_2d[:, c], self.chip_add_2d[:, c]) for c in range(self.channel_num)])

    # ---- 汇总 ----

    def summary(self) -> dict[str, float]:
        """标量指标, 用于阈值判定和报告"""
        return {
            "cs_overall": self.cs_overall,
            "cs_overall_add": self.cs_overall_add,
            "cs_frame_min": float(self.cs_frame.min()),
            "error_mean": float(self.error_2d.mean()),
            "error_std": float(self.error_2d.std()),
            "error_std_max": float(self.error_std0.max()),
            "error_abs_max": float(np.abs(self.error_2d).max()),
            "add_error_std": float(self.add_error_2d.std()),
            "fit_slope": self.fit[0],
            "fit_intercept": self.fit[1],
            "exact_match_ratio": float((self.error_2d == 0).mean()),
        }

    def to_dict(self, fields: list[str]) -> dict[str, Any]:
        result = {}
        for name in fields:
            value = getattr(self, name)
            result[name] = value.tolist() if isinstance(value, np.ndarray) else value
        return result

    # ---- 绘图 ----

    @staticmethod
    def _style(ax: Axes, cfg: AxConfig, title: str, xlabel: str, ylabel: str, *, legend: bool = True) -> None:
        ax.set_title(title, fontsize=cfg.title_size)
        ax.set_xlabel(xlabel, fontsize=cfg.label_size)
        ax.set_ylabel(ylabel, fontsize=cfg.label_size)
        ax.tick_params(labelsize=cfg.font_size)
        if cfg.grid:
            ax.grid(True, alpha=cfg.grid_alpha)
        if legend:
            ax.legend(loc=cfg.legend_loc, fontsize=cfg.font_size)

    def plot_single_frame_out(self, index: int, *, ax: Axes, ax_config: AxConfig) -> None:
        ax.plot(self.chip_2d[index], "r.", label="chip")
        ax.plot(self.golden_2d[index], "b", label="golden")
        self._style(ax, ax_config, f"Single Frame Output (Index={index})", "Channel Index", "Output Value")

    def plot_single_frame_err(self, index: int, *, ax: Axes, ax_config: AxConfig) -> None:
        error = self.error_2d[index]
        add_error = self.add_error_2d[index]
        ax.plot(error, ".", label=f"error std = {error.std():.2f} / {add_error.std():.2f} (add diff)")
        self._style(ax, ax_config, f"Single Frame Error (Index={index})", "Channel Index", "Error Value")

    def plot_error_statistics(self, *, ax: Axes, ax_config: AxConfig) -> None:
        ax.plot(self.error_std1, "b", label=f"std {self.error_std1.mean():.2f}")
        ax.plot(self.add_error_std1, "b.", label=f"std {self.add_error_std1.mean():.2f} (add diff)")
        ax.plot(self.error_mean1, "r", label="mean")
        ax.plot(self.add_error_mean1, "r.", label="mean (add diff)")
        self._style(ax, ax_config, "Error Statistics per Frame", "Frame Index", "Error")

    def plot_std0_2d(self, *, ax: Axes, ax_config: AxConfig) -> None:
        ax.plot(self.error_std0, "b", label=f"std {self.error_std0.mean():.2f}")
        ax.plot(self.error_mean0, "r", label=f"mean {self.error_mean0.mean():.2f} / {self.error_mean0.std():.2f}")
        self._style(ax, ax_config, "Mean & Std of Error per Channel", "Channel Index", "Error")

    def plot_cosine_similarity(self, *, ax: Axes, ax_config: AxConfig) -> None:
        ax.plot(self.cs_frame, "b", label=f"{self.cs_overall:.3f}")
        ax.plot(self.cs_frame_add, "r", label=f"{self.cs_overall_add:.3f} (add diff)")
        self._style(ax, ax_config, "Cosine Similarity of Each Frame", "Frame Index", "Cosine Similarity")

    def plot_cosine_similarity_output(self, *, ax: Axes, ax_config: AxConfig) -> None:
        ax.plot(self.cs_output, "b", label=f"{self.cs_overall:.3f}")
        ax.plot(self.cs_output_add, "r", label=f"{self.cs_overall_add:.3f} (add diff)")
        self._style(ax, ax_config, "Cosine Similarity of Each Channel", "Channel Index", "Cosine Similarity")

    def _plot_fit(
        self, ax: Axes, cfg: AxConfig, chip: np.ndarray, error: np.ndarray, fit: tuple[float, float], title: str
    ) -> None:
        x = np.arange(self.clip[0], self.clip[1] + 1)
        slope, intercept = fit
        ax.plot(self.golden_2d.ravel(), chip.ravel(), "r.", label=f"std = {error.std():.3f}")
        ax.plot(x, slope * x + intercept, "b", label=f"y = {slope:.3f}x + {intercept:.2f}")
        ax.plot(x, x, "k")
        self._style(ax, cfg, title, "Golden Output", "Chip Output")

    def plot_fitting(self, *, ax: Axes, ax_config: AxConfig) -> None:
        self._plot_fit(ax, ax_config, self.chip_2d, self.error_2d, self.fit, "Curve Fitting of Chip Output")

    def plot_fitting_of_add_diff(self, *, ax: Axes, ax_config: AxConfig) -> None:
        self._plot_fit(ax, ax_config, self.chip_add_2d, self.add_error_2d, self.fit_add, "Curve Fitting (Add Diff)")

    def plot_slope_of_linear(self, *, ax: Axes, ax_config: AxConfig) -> None:
        ax.plot(self.coeff[:, 0])
        self._style(ax, ax_config, "Slope of Linear Fitting", "Channel Index", "Slope", legend=False)

    def plot_intercept_of_linear(self, *, ax: Axes, ax_config: AxConfig) -> None:
        ax.plot(self.coeff[:, 1])
        self._style(ax, ax_config, "Intercept of Linear Fitting", "Channel Index", "Intercept", legend=False)

    def plot_diff_heatmap(self, *, ax: Axes, ax_config: AxConfig) -> None:
        from matplotlib.colors import ListedColormap

        image = ax.imshow(self.error_2d == 0, cmap=ListedColormap(["red", "green"]), aspect="auto", vmin=0, vmax=1)
        self._style(ax, ax_config, "Binary Difference Heatmap", "Channel Index", "Frame Index", legend=False)
        colorbar = ax.figure.colorbar(image, ax=ax, ticks=[0, 1])
        colorbar.ax.set_yticklabels(["Mismatch", "Match"])
        colorbar.set_label("Match Status", fontsize=ax_config.label_size)
        colorbar.ax.tick_params(labelsize=ax_config.font_size)
