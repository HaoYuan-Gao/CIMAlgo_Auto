"""
标准管线: 编译 -> 映射 -> 烧录 -> 推理 -> 分析 -> 阈值判定

按 YAML case 参数化的硬件测试; 差异全在 YAML 里。运行::

    pytest tests/hardware -m hardware --cases cases

每一步都是 cimalgo.pipeline 里的普通函数, 手写的 Python 测试可以复用它们, 只改顺序或中间插入自己的逻辑。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cimalgo.config import ProjectConfig
from cimalgo.drivers import DeviceOps, InferSession
from cimalgo.pipeline import CaseSpec, analyze_results, compile_model, program_model

# 该文件中的所有测试函数都加上 hardware 标记
pytestmark = pytest.mark.hardware


def test_case(
    case: CaseSpec,
    project_config: ProjectConfig,
    model_dir: Path,
    case_workdir: Path,
    device_ops: DeviceOps,
    record_property,
) -> None:
    compiled = compile_model(project_config.toolchain, model_dir, case_workdir / "compile", oc_comp=case.infer.oc_comp)
    program_model(device_ops, compiled, case_workdir / "program")

    # 推理运行时依赖芯片上已烧录的程序, 必须在 program 之后才能 init, 因此不做成 fixture
    with InferSession(device_ops, case_workdir / "results") as session:
        results = session.infer_model(model_dir, dummy=case.infer.dummy, infer_mode=dict(case.infer.mode))

    report = analyze_results(results, case_workdir / "results", thresholds=case.thresholds)
    for key, value in report.flat_metrics().items():
        record_property(key, value)
    assert report.passed, "\n".join(report.failures)
