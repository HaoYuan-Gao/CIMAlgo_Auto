"""测试管线: case 定义、假数据生成、编译/映射、可复用的管线步骤"""

from .case import CaseDefinitionError, CaseSpec, DeviceSpec, InferSpec, ModelSpec, Threshold, load_case, load_cases
from .fake_data import make_fake_model
from .steps import AnalysisReport, CompiledModel, NodeAnalysis, analyze_results, compile_model, program_model
from .toolchain import Bw01dToolchain, MapOptions, ToolchainError, map_l2_id

__all__ = [
    "AnalysisReport",
    "Bw01dToolchain",
    "CaseDefinitionError",
    "CaseSpec",
    "CompiledModel",
    "DeviceSpec",
    "InferSpec",
    "MapOptions",
    "ModelSpec",
    "NodeAnalysis",
    "Threshold",
    "ToolchainError",
    "analyze_results",
    "compile_model",
    "map_l2_id",
    "load_case",
    "load_cases",
    "make_fake_model",
    "program_model",
]
