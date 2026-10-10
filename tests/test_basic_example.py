"""最简单的通路测试: 确认 pytest 能正常收集、执行, conftest.py 的环境部署没有卡住"""

import pytest
from pathlib import Path

from utils.toolchain import Bw01dToolchain
from utils.infer_session import InferSession
from cim_constants import get_toolchain_config, get_device_ops


@pytest.mark.parametrize(
    "case_context",
    [
        {
            "case": "test_workflow",
            "tag": "external",
            "dummy": 50,
            "compile_model": {
                "model_path": "workspace/test_workflow/model",
                "save_path": "workspace/test_workflow/binary",
                "program_dir": "workspace/test_workflow/program",
            },
            "device": {
                "type": "zyqn",
                "ip": "10.27.17.204",
                "port": 22,
                "username": "root",
                "password": "Witmem*123",
                "update_sdk": False,
                "hard_block": [0, 1, 2],
                "sel_layer": [0, 1, 2],
                "tsv_sel0": 0,
                "tsv_sel1": 255,
            },
        },
    ],
)
def test_smoke(case_context, request):
    # 获取 device_ops
    device_ops = get_device_ops(request)
    # 获取 toolchain_config
    toolchain_config = get_toolchain_config(request)

    # 初始化 Bw01dToolchain
    bw01d_toolchain = Bw01dToolchain(
        toolchain_config.compiler_tools.install_dir,
        toolchain_config.map_tools.install_dir,
        case_context["compile_model"]["model_path"],
        case_context["compile_model"]["save_path"],
    )

    bw01d_toolchain.generate_node_config()
    bw01d_toolchain.update_node_configs(
        optimization={
            "dacfifo28_y_in_th_cfg": 16,
            "dacfifo28_y_out_th_cfg": 1,
        },
    )
    bw01d_toolchain.compile_model()
    bw01d_toolchain.remap_die(l2_id=0)

    program_job = [
        (0, str(Path(case_context["compile_model"]["save_path"]) / "Die_0_L2_0.tar.gz")),
        # (1, str(Path(case_context["compile_model"]["save_path"]) / "Die_0_L2_0.tar.gz")),
    ]
    device_ops.program(jobs=program_job, k=2, d=3, program_dir=str(Path(case_context["compile_model"]["program_dir"])))

    result_dir = Path(case_context["compile_model"]["model_path"]).parent / "result"
    with InferSession(device_ops.init_connection(), result_dir, chip="bw01d") as infer:
        infer.L2Infer(
            model_path=case_context["compile_model"]["model_path"],
            dummy=50,
            prefix_name="test",
            infer_modes={
                "sub_batch_size": 20,
            },
        )
