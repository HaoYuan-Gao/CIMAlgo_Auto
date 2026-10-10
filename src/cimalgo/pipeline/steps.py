"""
标准管线的可复用步骤: 编译 / 烧录 / 分析

每个步骤是一个普通函数, 输入输出都是明确的对象; YAML 管线 (tests/hardware/test_pipeline.py) 和
手写的 Python 测试调用同一套函数, 不需要各自抄一遍。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cimalgo.analyzer import draw_basic_metrics
from cimalgo.config import ToolchainConfig
from cimalgo.drivers import DeviceOps, NodeResult
from cimalgo.drivers.infer import parse_node_name
from cimalgo.logging import get_logger

from .case import Threshold
from .toolchain import Bw01dToolchain, MapOptions, map_l2_id

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CompiledModel:
    """编译 + 映射的产物; jobs 是与 layer 绑定的 (L2 编号, 压缩包) 列表"""

    model_dir: Path
    compile_dir: Path
    jobs: tuple[tuple[int, Path], ...]

    def __post_init__(self) -> None:
        if not self.jobs:
            raise ValueError("jobs 不能为空")
        ids = [l2 for l2, _ in self.jobs]
        if any(isinstance(l2, bool) or not isinstance(l2, int) or l2 < 0 for l2 in ids):
            raise ValueError("L2 编号必须是非负整数")
        if len(set(ids)) != len(ids):
            raise ValueError(f"L2 编号不得重复: {ids}")

    @classmethod
    def from_archives(cls, model_dir: Path, compile_dir: Path, archives: Sequence[Path]) -> CompiledModel:
        """从已有压缩包构造, L2 编号取自文件名 Die_<n>_L2_<id>.tar.gz"""
        return cls(model_dir, compile_dir, tuple((map_l2_id(path), Path(path)) for path in archives))

    @property
    def archives(self) -> tuple[Path, ...]:
        return tuple(path for _, path in self.jobs)

    @property
    def l2_ids(self) -> tuple[int, ...]:
        return tuple(l2 for l2, _ in self.jobs)


@dataclass(frozen=True, slots=True)
class NodeAnalysis:
    node: NodeResult
    metrics: dict[str, float]
    plot_path: Path
    failures: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    nodes: tuple[NodeAnalysis, ...] = field(default_factory=tuple)

    @property
    def failures(self) -> list[str]:
        return [failure for node in self.nodes for failure in node.failures]

    @property
    def passed(self) -> bool:
        return not self.failures

    def flat_metrics(self) -> dict[str, float]:
        """{节点名.指标名: 值}, 用于 record_property 或报告"""
        return {f"{node.node.name}.{key}": value for node in self.nodes for key, value in node.metrics.items()}


@log.trace()
def compile_model(
    toolchain: ToolchainConfig,
    model_dir: str | Path,
    compile_dir: str | Path,
    *,
    oc_comp: bool = False,
    # NOTE: 外部提供 redundancy info
    map_options: Mapping[int, MapOptions] | None = None,
    timeout: float = 600,
) -> CompiledModel:
    """
    生成节点配置 -> 编译 -> 映射打包; 返回可烧录的 CompiledModel

    map_options 按 L2 编号配置每个 Map 的映射参数 (stage、冗余信息), 见 Bw01dToolchain.remap_maps;
    map_options 只接受 {L2 编号: MapOptions}。layer、编译产物、映射、烧录和推理使用同一个 L2 编号。
    """
    tc = Bw01dToolchain.from_config(toolchain, model_dir, compile_dir)
    model_l2_ids = {
        parse_node_name(node.name)[0]
        for node in tc.model_path.iterdir()
        if node.is_dir() and not node.is_symlink()
    }
    if not model_l2_ids:
        raise ValueError(f"模型中没有 layer 目录: {tc.model_path}")
    tc.generate_node_config(timeout=timeout)
    tc.compile_model(oc_comp=oc_comp, timeout=timeout)
    compiled_l2_ids = {map_l2_id(path) for path in tc.map_directories()}
    if compiled_l2_ids != model_l2_ids:
        missing = sorted(model_l2_ids - compiled_l2_ids)
        extra = sorted(compiled_l2_ids - model_l2_ids)
        raise ValueError(f"layer 与编译产物的 L2 不一致: 编译产物缺少 {missing}, 多出 {extra}")
    jobs = tc.remap_maps(map_options, timeout=timeout)
    return CompiledModel(model_dir=tc.model_path, compile_dir=tc.compile_path, jobs=tuple(jobs))


@log.trace()
def program_model(
    device: DeviceOps,
    compiled: CompiledModel,
    output_dir: str | Path,
    *,
    k: int = 1,
    d: int = 1,
) -> dict[int, Any]:
    """按编译产物绑定的 L2 烧录到设备, 不允许在 steps 层覆盖目标 L2"""
    return device.program(compiled.jobs, k, d, output_dir)


def analyze_results(
    results: Sequence[NodeResult],
    plot_dir: str | Path,
    *,
    thresholds: Sequence[Threshold] = (),
) -> AnalysisReport:
    """逐节点绘图、算指标、按阈值判定; 不抛异常, 结果在 report.failures 里"""
    plot_dir = Path(plot_dir)
    nodes = []
    for node in results:
        plot_path = plot_dir / f"{node.name}.png"
        metrics = draw_basic_metrics(node.golden, node.output, plot_path)
        log.info("%s: %s", node.name, ", ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
        failures = []
        for threshold in thresholds:
            if threshold.metric not in metrics:
                failures.append(f"{node.name}: 未知指标 {threshold.metric}; 可用: {', '.join(sorted(metrics))}")
            elif not threshold.check(metrics[threshold.metric]):
                failures.append(f"{node.name}: {threshold.metric}={metrics[threshold.metric]:.4f} 不满足 {threshold}")
        nodes.append(NodeAnalysis(node=node, metrics=metrics, plot_path=plot_path, failures=tuple(failures)))
    return AnalysisReport(nodes=tuple(nodes))
