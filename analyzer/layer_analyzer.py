import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.metrics.pairwise import cosine_similarity

from .plot_config import AxConfig


class LayerAnalyzer:
    def __init__(
        self,
        golden: np.ndarray,
        chip: np.ndarray,
        channel_dim: int = -1,
        diff_clip_val: tuple = (-128, 127),
    ):
        """
        初始化 Analyzer 类, 计算并缓存基本指标.

        函数内将输入 array 转换为 2D 数组, 并将通道维度移动到最后一个维度. Error = chip - golden

        参数:
            golden (np.ndarray):  golden 输出, shape 任意
            chip (np.ndarray):  chip 输出, shape 任意
            channel_dim (int):  通道维度, 默认为 -1 (最后一个维度)
            diff_clip_val (tuple): clip 值, 默认为 (-128, 127)
        """
        self.golden = golden
        self.chip = chip
        self.channel_dim = channel_dim
        self.diff_clip_val = diff_clip_val

        self.golden_2d = self.to_2d_array(self.golden, channel_dim=self.channel_dim)
        self.chip_2d = self.to_2d_array(self.chip, channel_dim=self.channel_dim)

        if self.golden_2d.shape != self.chip_2d.shape:
            raise ValueError(
                f"golden and chip shape mismatch after reshape: "
                f"golden={self.golden_2d.shape}, chip={self.chip_2d.shape}"
            )

        self.frame_num, self.channel_num = self.golden_2d.shape

        # ===== Basic cache =====
        self.error_2d = self.chip_2d - self.golden_2d

        # ===== Lazy cache: computed only when corresponding plot needs it =====
        self._add_delta_2d = None
        self._chip_add_2d = None
        self._add_error_2d = None

        self._error_mean0 = None
        self._error_std0 = None
        self._error_mean1 = None
        self._error_std1 = None
        self._add_error_mean0 = None
        self._add_error_std0 = None
        self._add_error_mean1 = None
        self._add_error_std1 = None

        self._cs_frame = None
        self._cs_output = None
        self._cs_overall = None
        self._cs_frame_add = None
        self._cs_output_add = None
        self._cs_overall_add = None

        self._coff = None
        self._coff_add = None
        self._fit = None
        self._pfun = None
        self._fit_add = None
        self._pfun_add = None

    def to_dict(self, fields: list = None):
        """返回指标数据的字典, 可用于打印或保存"""
        if not fields:
            return {}

        def normalize(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            return v

        return {k: normalize(v) for k in fields if (v := getattr(self, k, None)) is not None}

    @staticmethod
    def to_2d_array(a: np.ndarray, channel_dim: int = -1) -> np.ndarray:
        """Convert multi-dimensional array to 2D array with shape (N, C)."""
        a = np.asarray(a)

        if a.ndim == 1:
            if channel_dim in (-1, 0):
                return a.reshape(1, -1)
            raise ValueError(f"Invalid channel_dim={channel_dim} for 1D array")

        if channel_dim < 0:
            channel_dim = a.ndim + channel_dim

        if not (0 <= channel_dim < a.ndim):
            raise ValueError(f"Invalid channel_dim={channel_dim} for array with ndim={a.ndim}")

        a = np.moveaxis(a, channel_dim, -1)
        return a.reshape(-1, a.shape[-1])

    # ===== Lazy properties =====
    @property
    def add_delta_2d(self) -> np.ndarray:
        """golden 和 chip 做差, 然后对每个 channel 取均值"""
        if self._add_delta_2d is None:
            self._add_delta_2d = np.mean(self.golden_2d - self.chip_2d, axis=0).astype(np.int32)
        return self._add_delta_2d

    @property
    def chip_add_2d(self) -> np.ndarray:
        """chip 输出：在 golden 基础上加 diff"""
        if self._chip_add_2d is None:
            self._chip_add_2d = np.clip(
                self.chip_2d + self.add_delta_2d,
                self.diff_clip_val[0],
                self.diff_clip_val[1],
            )
        return self._chip_add_2d

    @property
    def add_error_2d(self) -> np.ndarray:
        """chip 加 diff 后与 golden 的误差"""
        if self._add_error_2d is None:
            self._add_error_2d = self.chip_add_2d - self.golden_2d
        return self._add_error_2d

    @property
    def error_mean0(self) -> np.ndarray:
        """误差的均值, 对每个 channel 取均值"""
        if self._error_mean0 is None:
            self._error_mean0 = np.mean(self.error_2d, axis=0)
        return self._error_mean0

    @property
    def error_std0(self) -> np.ndarray:
        """误差的标准差, 对每个 channel 取标准差"""
        if self._error_std0 is None:
            self._error_std0 = np.std(self.error_2d, axis=0)
        return self._error_std0

    @property
    def error_mean1(self) -> np.ndarray:
        """误差的均值, 对每个 frame 取均值"""
        if self._error_mean1 is None:
            self._error_mean1 = np.mean(self.error_2d, axis=1)
        return self._error_mean1

    @property
    def error_std1(self) -> np.ndarray:
        """误差的标准差, 对每个 frame 取标准差"""
        if self._error_std1 is None:
            self._error_std1 = np.std(self.error_2d, axis=1)
        return self._error_std1

    @property
    def add_error_mean0(self) -> np.ndarray:
        """chip 加 diff 后与 golden 的误差的均值, 对每个 channel 取均值"""
        if self._add_error_mean0 is None:
            self._add_error_mean0 = np.mean(self.add_error_2d, axis=0)
        return self._add_error_mean0

    @property
    def add_error_std0(self) -> np.ndarray:
        """chip 加 diff 后与 golden 的误差的标准差, 对每个 channel 取标准差"""
        if self._add_error_std0 is None:
            self._add_error_std0 = np.std(self.add_error_2d, axis=0)
        return self._add_error_std0

    @property
    def add_error_mean1(self) -> np.ndarray:
        """chip 加 diff 后与 golden 的误差的均值, 对每个 frame 取均值"""
        if self._add_error_mean1 is None:
            self._add_error_mean1 = np.mean(self.add_error_2d, axis=1)
        return self._add_error_mean1

    @property
    def add_error_std1(self) -> np.ndarray:
        """chip 加 diff 后与 golden 的误差的标准差, 对每个 frame 取标准差"""
        if self._add_error_std1 is None:
            self._add_error_std1 = np.std(self.add_error_2d, axis=1)
        return self._add_error_std1

    def _compute_cosine(self, chip_2d: np.ndarray):
        """计算 cosine similarity"""
        cs_frame = np.diag(cosine_similarity(chip_2d, self.golden_2d))
        cs_output = np.diag(cosine_similarity(chip_2d.T, self.golden_2d.T))
        cs_overall = cosine_similarity(
            chip_2d.reshape(1, -1),
            self.golden_2d.reshape(1, -1),
        )[0, 0]
        return cs_frame, cs_output, cs_overall

    def _ensure_cosine(self):
        if self._cs_frame is None:
            self._cs_frame, self._cs_output, self._cs_overall = self._compute_cosine(self.chip_2d)

    def _ensure_cosine_add(self):
        if self._cs_frame_add is None:
            self._cs_frame_add, self._cs_output_add, self._cs_overall_add = self._compute_cosine(self.chip_add_2d)

    @property
    def cs_frame(self) -> np.ndarray:
        """chip 输出与 golden 的 cosine similarity, 对每个 frame 取相似度"""
        self._ensure_cosine()
        return self._cs_frame

    @property
    def cs_output(self) -> np.ndarray:
        """chip 输出与 golden 的 cosine similarity, 对每个 channel 取相似度"""
        self._ensure_cosine()
        return self._cs_output

    @property
    def cs_overall(self) -> float:
        """chip 输出与 golden 的 cosine similarity, 对所有数据取相似度"""
        self._ensure_cosine()
        return self._cs_overall

    @property
    def cs_frame_add(self) -> np.ndarray:
        """chip 加 diff 后与 golden 的 cosine similarity, 对每个 frame 取相似度"""
        self._ensure_cosine_add()
        return self._cs_frame_add

    @property
    def cs_output_add(self) -> np.ndarray:
        """chip 加 diff 后与 golden 的 cosine similarity, 对每个 channel 取相似度"""
        self._ensure_cosine_add()
        return self._cs_output_add

    @property
    def cs_overall_add(self) -> float:
        """chip 加 diff 后与 golden 的 cosine similarity, 对所有数据取相似度"""
        self._ensure_cosine_add()
        return self._cs_overall_add

    def _compute_global_fit(self, chip_2d: np.ndarray):
        """计算全局线性拟合系数"""
        fit = np.polyfit(self.golden_2d.ravel(), chip_2d.ravel(), deg=1)
        return fit, np.poly1d(fit)

    def _compute_channel_linear_coeff(self, chip_2d: np.ndarray) -> np.ndarray:
        """计算每个 channel 的线性拟合系数"""
        coeff = np.zeros((self.channel_num, 2), dtype=np.float64)

        for col in range(self.channel_num):
            x = self.golden_2d[:, col]
            y = chip_2d[:, col]

            if np.all(x == self.golden_2d[0, col]) or np.all(y == chip_2d[0, col]):
                coeff[col, :] = 0
            else:
                try:
                    fit = np.polyfit(x, y, deg=1)
                    coeff[col, :] = np.poly1d(fit)
                except np.linalg.LinAlgError:
                    coeff[col, :] = 0

        return coeff

    @property
    def coff(self) -> np.ndarray:
        """每个 channel 的线性拟合系数, 第一列是斜率, 第二列是截距"""
        if self._coff is None:
            self._coff = self._compute_channel_linear_coeff(self.chip_2d)
        return self._coff

    @property
    def coff_add(self) -> np.ndarray:
        """chip 加 diff 后每个 channel 的线性拟合系数, 第一列是斜率, 第二列是截距"""
        if self._coff_add is None:
            self._coff_add = self._compute_channel_linear_coeff(self.chip_add_2d)
        return self._coff_add

    @property
    def fit(self) -> np.ndarray:
        """全局线性拟合系数"""
        if self._fit is None:
            self._fit, self._pfun = self._compute_global_fit(self.chip_2d)
        return self._fit

    @property
    def pfun(self):
        """全局线性拟合函数"""
        if self._pfun is None:
            self._fit, self._pfun = self._compute_global_fit(self.chip_2d)
        return self._pfun

    @property
    def fit_add(self) -> np.ndarray:
        """chip 加 diff 后全局线性拟合系数"""
        if self._fit_add is None:
            self._fit_add, self._pfun_add = self._compute_global_fit(self.chip_add_2d)
        return self._fit_add

    @property
    def pfun_add(self):
        """chip 加 diff 后全局线性拟合函数"""
        if self._pfun_add is None:
            self._fit_add, self._pfun_add = self._compute_global_fit(self.chip_add_2d)
        return self._pfun_add

    @staticmethod
    def _apply_common_axis_style(ax: plt.Axes, ax_config: AxConfig):
        """应用通用的 axis 样式"""
        ax.tick_params(labelsize=ax_config.font_size)
        if ax_config.grid:
            ax.grid(True, alpha=ax_config.grid_alpha)

    @staticmethod
    def _set_labels(ax: plt.Axes, ax_config: AxConfig, title: str, xlabel: str, ylabel: str):
        """应用通用的 axis 样式"""
        ax.set_title(title, fontsize=ax_config.title_size)
        ax.set_xlabel(xlabel, fontsize=ax_config.label_size)
        ax.set_ylabel(ylabel, fontsize=ax_config.label_size)

    def plot_single_frame_out(self, index: int, ax: plt.Axes, ax_config: AxConfig):
        """绘制单个 frame 的 chip 输出与 golden 输出"""
        y_chip = self.chip_2d[index]
        y_gold = self.golden_2d[index]

        ax.plot(y_chip, "r.", label="chip")
        ax.plot(y_gold, "b", label="golden")

        self._set_labels(ax, ax_config, f"Single Frame Output (Index={index})", "Channel Index", "Output Value")
        self._apply_common_axis_style(ax, ax_config)
        ax.legend(loc=ax_config.legend_loc, fontsize=ax_config.font_size)

    def plot_single_frame_err(self, index: int, ax: plt.Axes, ax_config: AxConfig):
        """绘制单个 frame 的 error 与 add error"""
        error = self.error_2d[index]
        add_error = self.add_error_2d[index]

        ax.plot(error, ".", label=f"error std = {np.std(error):.2f} / {np.std(add_error):.2f} (add diff)")

        self._set_labels(ax, ax_config, f"Single Frame Error (Index={index})", "Channel Index", "Error Value")
        self._apply_common_axis_style(ax, ax_config)
        ax.legend(loc=ax_config.legend_loc, fontsize=ax_config.font_size)

    def plot_error_statistics(self, ax: plt.Axes, ax_config: AxConfig):
        """绘制 error 与 add error 的统计信息"""
        ax.plot(self.error_std1, "b", label=f"{np.mean(self.error_std1):.2f}")
        ax.plot(self.add_error_std1, "b.", label=f"{np.mean(self.add_error_std1):.2f}")

        ax.plot(self.error_mean1, "r")
        ax.plot(self.add_error_mean1, "r.")

        self._set_labels(ax, ax_config, "Error Statistics per Frame", "Frame Index", "Error Statistics")
        self._apply_common_axis_style(ax, ax_config)
        ax.legend(loc=ax_config.legend_loc, fontsize=ax_config.font_size)

    def plot_std0_2d(self, ax: plt.Axes, ax_config: AxConfig):
        """绘制每个 channel 的 error 与 add error 的统计信息"""
        ax.plot(self.error_std0, "b", label=f"{np.mean(self.error_std0):.2f}")
        ax.plot(self.error_mean0, "r", label=f"{np.mean(self.error_mean0):.2f} / {np.std(self.error_mean0):.2f}")

        self._set_labels(ax, ax_config, "Mean & Std of Error per Channel", "Channel Index", "Error Statistics")
        self._apply_common_axis_style(ax, ax_config)
        ax.legend(loc=ax_config.legend_loc, fontsize=ax_config.font_size)

    def plot_cosine_similarity(self, ax: plt.Axes, ax_config: AxConfig):
        """绘制每个 frame 的 cosine similarity 与 add cosine similarity"""
        ax.plot(self.cs_frame, "b", label=f"{self.cs_overall:.3f}")
        ax.plot(self.cs_frame_add, "r", label=f"{self.cs_overall_add:.3f} (add diff)")

        self._set_labels(ax, ax_config, "Cosine Similarity of Each Frame", "Frame Index", "Cosine Similarity")
        self._apply_common_axis_style(ax, ax_config)
        ax.legend(loc=ax_config.legend_loc, fontsize=ax_config.font_size)

    def plot_cosine_similarity_output(self, ax: plt.Axes, ax_config: AxConfig):
        """绘制每个 channel 的 cosine similarity 与 add cosine similarity"""
        ax.plot(self.cs_output, "b", label=f"{self.cs_overall:.3f}")
        ax.plot(self.cs_output_add, "r", label=f"{self.cs_overall_add:.3f} (add diff)")

        self._set_labels(ax, ax_config, "Cosine Similarity of Each Channel", "Channel Index", "Cosine Similarity")
        self._apply_common_axis_style(ax, ax_config)
        ax.legend(loc=ax_config.legend_loc, fontsize=ax_config.font_size)

    def plot_fitting(self, ax: plt.Axes, ax_config: AxConfig):
        """绘制 chip 输出与 golden 输出的曲线拟合"""
        x = np.arange(self.diff_clip_val[0], self.diff_clip_val[1] + 1)
        fit_chip = self.chip_2d.ravel()
        fit_golden = self.golden_2d.ravel()

        ax.plot(fit_golden, fit_chip, "r.", label=f"std = {np.std(self.error_2d):.3f}")
        ax.plot(x, self.pfun(x), "b", label=f"y = {str(self.pfun)}")
        ax.plot(x, x, "k")

        self._set_labels(ax, ax_config, "Curve Fitting of Chip Output", "Golden Output", "Chip Output")
        self._apply_common_axis_style(ax, ax_config)
        ax.legend(loc=ax_config.legend_loc, fontsize=ax_config.font_size)

    def plot_fitting_of_add_diff(self, ax: plt.Axes, ax_config: AxConfig):
        """绘制 chip 输出与 golden 输出（添加 diff）的曲线拟合"""
        x = np.arange(self.diff_clip_val[0], self.diff_clip_val[1] + 1)
        fit_chip = self.chip_add_2d.ravel()
        fit_golden = self.golden_2d.ravel()

        ax.plot(fit_golden, fit_chip, "r.", label=f"std = {np.std(self.add_error_2d):.3f}")
        ax.plot(x, self.pfun_add(x), "b", label=f"y = {str(self.pfun_add)} (add diff)")
        ax.plot(x, x, "k")

        self._set_labels(
            ax, ax_config, "Curve Fitting of Chip Output (Add Diff)", "Golden Output", "Chip Output (Add Diff)"
        )
        self._apply_common_axis_style(ax, ax_config)
        ax.legend(loc=ax_config.legend_loc, fontsize=ax_config.font_size)

    def plot_slope_of_linear(self, ax: plt.Axes, ax_config: AxConfig):
        """绘制每个 channel 的 linear fitting 斜率"""
        ax.plot(self.coff[:, 0])

        self._set_labels(ax, ax_config, "Slope of Linear Fitting", "Channel Index", "Slope")
        self._apply_common_axis_style(ax, ax_config)

    def plot_intercept_of_linear(self, ax: plt.Axes, ax_config: AxConfig):
        """绘制每个 channel 的 linear fitting 截距"""
        ax.plot(self.coff[:, 1])

        self._set_labels(ax, ax_config, "Intercept of Linear Fitting", "Channel Index", "Intercept")
        self._apply_common_axis_style(ax, ax_config)

    def plot_std1_2d(self, ax: plt.Axes, ax_config: AxConfig):
        """绘制每个 frame 的 error 与 add error 的 std1 信息"""
        ax.plot(self.error_std1, "b", label="std1")
        ax.plot(self.error_mean1, "r", label="mean1")

        self._set_labels(ax, ax_config, "Mean & Std of Error per Frame", "Frame Index", "Value")
        self._apply_common_axis_style(ax, ax_config)
        ax.legend(loc=ax_config.legend_loc, fontsize=ax_config.font_size)

    def plot_diff_heatmap(self, ax: plt.Axes, ax_config: AxConfig):
        """绘制二值热力图 (True / False)"""
        # True 表示误差为 0
        diff = np.abs(self.error_2d) == 0

        # 两种颜色：
        # False -> 红色
        # True  -> 绿色
        cmap = ListedColormap(["red", "green"])

        im = ax.imshow(diff, cmap=cmap, aspect="auto", vmin=0, vmax=1)

        self._set_labels(ax, ax_config, "Binary Difference Heatmap", "Channel Index", "Frame Index")

        self._apply_common_axis_style(ax, ax_config)

        # colorbar
        cbar = ax.figure.colorbar(im, ax=ax, ticks=[0, 1])

        cbar.ax.set_yticklabels(["False", "True"])

        cbar.set_label("Match Status", fontsize=ax_config.label_size)

        cbar.ax.tick_params(labelsize=ax_config.font_size)
