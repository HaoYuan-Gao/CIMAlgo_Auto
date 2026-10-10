"""基础功能测试, 可用作模板, 快速执行某项功能"""

from __future__ import annotations

import pytest

from cimalgo.config import ProjectConfig
from cimalgo.pipeline import DeviceSpec
from cimalgo.drivers import DeviceOps, InferSession
from cimalgo.pipeline import ModelSpec, analyze_results, compile_model, program_model

# 该文件中的所有测试函数都加上 hardware 标记
pytestmark = pytest.mark.hardware


def test_download_sdk(project_config: ProjectConfig) -> None:
    """验证 --download 已将当前 SDK 下载并解压到本地"""
    sdk = project_config.sdk
    assert sdk.install_dir.is_dir(), f"SDK 未安装: {sdk.install_dir}; 请先使用 --download"
    assert any(sdk.install_dir.iterdir()), f"SDK 安装目录为空: {sdk.install_dir}"
    assert sdk.wheel_dir.is_dir(), f"SDK wheel 目录不存在: {sdk.wheel_dir}"
    assert any(sdk.wheel_dir.glob("*.whl")), f"SDK wheel 目录中没有 whl 文件: {sdk.wheel_dir}"


@pytest.mark.device("evb-01", firmware_version="2.3.3", hard_blocks=[0, 1, 2], select_layers=[0], update_sdk=True)
def test_update_sdk(project_config: ProjectConfig, device_spec: DeviceSpec, device_ops: DeviceOps) -> None:
    """部署当前 SDK 后，验证设备服务和固件状态"""
    actual_sdk_version = device_ops.get_sdk_version()
    actual_firmware_version = device_ops.get_firmware_version()

    assert actual_sdk_version == project_config.sdk.version, (
        f"设备 SDK 版本不匹配: 期望 {project_config.sdk.version}, 实际 {actual_sdk_version}"
    )
    assert actual_firmware_version == device_spec.firmware_version, (
        f"设备固件版本不匹配: 期望 {device_spec.firmware_version}, 实际 {actual_firmware_version}"
    )

@pytest.mark.device("evb-01", firmware_version="2.3.3", hard_blocks=[0, 1, 2], select_layers=[0])
@pytest.mark.model("external", path="./workspace/test_map_program/")
def test_map_program(project_config: ProjectConfig, device_ops: DeviceOps, model_spec: ModelSpec) -> None:
    model_dir = model_spec.path
    compile_dir = model_spec.path.parent / "Compile"
    program_dir = model_spec.path.parent / "Program"
    # 编译模型
    compiled = compile_model(project_config.toolchain, model_dir, compile_dir)
    # 烧录模型
    program_model(device_ops, compiled, program_dir)

@pytest.mark.device("evb-01", firmware_version="2.3.3", hard_blocks=[0, 1, 2], select_layers=[0])
@pytest.mark.model("external", path="./workspace/test_map_program/")
def test_infer_model(project_config: ProjectConfig, device_ops: DeviceOps, model_spec: ModelSpec) -> None:
    model_dir = model_spec.path
    result_dir = model_spec.path.parent / "results"

    with InferSession(device_ops, result_dir) as session:
        results = session.infer_model(model_dir, dummy=10)

    report = analyze_results(results, result_dir)
