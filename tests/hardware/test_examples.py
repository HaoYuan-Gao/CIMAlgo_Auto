"""
手写 Python 硬件测试示例: 流程不是固定管线时, 用 device / model 标记声明依赖, 其余自己写

这些测试和 YAML case 共用设备池、初始化复用和排序; 运行::

    pytest tests/hardware/test_examples.py -m hardware
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cimalgo.config import ProjectConfig
from cimalgo.drivers import DeviceOps, InferSession
from cimalgo.pipeline import Threshold, analyze_results, compile_model, program_model

# 该文件中的所有测试函数都加上 hardware 标记
pytestmark = pytest.mark.hardware

DENSE = {
    "name": "model.layers.0.dense_L2_0_L0_0_node_0",
    "batch": 32,
    "in_features": 128,
    "out_features": 256,
}


@pytest.mark.device("evb-01", hard_blocks=[0, 1, 2, 3], select_layers=[0])
def test_firmware_matches_sdk(device_ops: DeviceOps, project_config: ProjectConfig) -> None:
    """只验证一个功能点, 不编译不烧录"""
    assert device_ops.get_firmware_version() == project_config.sdk.version


@pytest.mark.device("evb-01", hard_blocks=[0, 1, 2, 3], select_layers=[0])
@pytest.mark.model("fake", seed=7, layers=[DENSE])
def test_inference_is_repeatable(
    device_ops: DeviceOps, project_config: ProjectConfig, model_dir: Path, case_workdir: Path
) -> None:
    """同一份程序循环推理 N 次, 输出必须完全一致"""
    compiled = compile_model(project_config.toolchain, model_dir, case_workdir / "compile")
    program_model(device_ops, compiled, case_workdir / "program")

    runs = []
    with InferSession(device_ops, case_workdir / "results") as session:
        for index in range(10):
            runs.append(session.infer_model(model_dir, prefix=f"run{index}"))

    first = runs[0]
    for index, run in enumerate(runs[1:], start=1):
        for baseline, current in zip(first, run, strict=True):
            np.testing.assert_array_equal(current.output, baseline.output, err_msg=f"第 {index} 次推理与第 0 次不一致")


@pytest.mark.device("evb-01", hard_blocks=[0, 1, 2, 3], select_layers=[0], power_cycle=False)
@pytest.mark.model("fake", seed=7, layers=[DENSE])
def test_tsv_sweep(
    device_ops: DeviceOps, project_config: ProjectConfig, model_dir: Path, case_workdir: Path, record_property
) -> None:
    """扫 TSV: 中途手动改选层, 每个点各烧录推理一次, 汇总精度"""
    compiled = compile_model(project_config.toolchain, model_dir, case_workdir / "compile")
    thresholds = (Threshold("cs_overall", ">=", 0.99),)

    summary = {}
    for tsv in (0, 64, 128, 255):
        # 直接调用 select_layer 会让设备池把芯片状态标为未知, 下一个测试会重新初始化
        device_ops.select_layer(0, tsv_sel0=tsv)
        point_dir = case_workdir / f"tsv_{tsv}"
        program_model(device_ops, compiled, point_dir / "program")
        with InferSession(device_ops, point_dir / "results") as session:
            results = session.infer_model(model_dir, dummy=2)
        report = analyze_results(results, point_dir / "results", thresholds=thresholds)
        summary[tsv] = report.nodes[0].metrics["cs_overall"]
        record_property(f"tsv_{tsv}.cs_overall", summary[tsv])

    best = max(summary, key=summary.get)  # type: ignore[arg-type]
    record_property("best_tsv", best)
    assert summary[best] >= 0.99, f"所有 TSV 下余弦相似度都低于阈值: {summary}"
