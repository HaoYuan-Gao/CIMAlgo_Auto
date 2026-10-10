"""测试 isum 与去共模的关系"""

from __future__ import annotations

import pytest

from cimalgo.config import ProjectConfig
from cimalgo.pipeline import DeviceSpec
from cimalgo.drivers import DeviceOps, InferSession
from cimalgo.pipeline import ModelSpec, analyze_results, compile_model, program_model

# 该文件中的所有测试函数都加上 hardware 标记
pytestmark = pytest.mark.hardware

@pytest.mark.device("evb-01", firmware_version="2.3.3", hard_blocks=[0, 1, 2], select_layers=[0], update_sdk=False)
@pytest.mark.model("external", path="./workspace/test_cursur_en/")
def test_cursur_en(project_config: ProjectConfig, device_ops: DeviceOps, model_spec: ModelSpec):
    model_dir = model_spec.path
    compile_dir = model_spec.path.parent / "CompileResult"
    program_dir = model_spec.path.parent / "ProgramResult"

    # 编译模型
    compiled = compile_model(
        project_config.toolchain, 
        model_dir, 
        compile_dir,
        updates={
            "optimization": {
                "dacfifo28_y_in_th_cfg": 10,
                "dacfifo28_y_out_th_cfg": 10,
                "tia_mode": 0,
            },
            "burn": {
                "cursur_en": [0, 0],
            },
        },
    )

    # 烧录模型
    program_model(device_ops, compiled, program_dir)

