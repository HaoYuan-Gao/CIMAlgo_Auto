"""
推理会话: 在 with 作用域内初始化芯片推理运行时, 退出时反初始化

通道从 DeviceOps 复用, 不再单独建连。推理结果只保存 .npy, 分析和绘图由 pipeline 层负责,
这样 device 层不依赖 matplotlib / sklearn。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Any, Self

import numpy as np

from cimalgo.logging import get_logger

from . import _sdk
from .base import DeviceError, DeviceOps

log = get_logger(__name__)

NODE_DIR_PATTERN = re.compile(
    r"model\.layers\.(?P<layer_index>\d+)\."
    r"(?P<operation>.+)"
    r"_L2_(?P<l2_id>\d+)"
    r"_L0_(?P<l0_id>\d+)"
    r"_node_(?P<node_id>\d+)\Z"
)


def parse_node_name(name: str) -> tuple[int, int, int]:
    """解析 layer 目录名中与硬件绑定的 (L2, L0, node) 地址"""
    match = NODE_DIR_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"节点目录名必须符合 model.layers.<index>.<operation>_L2_<l2>_L0_<l0>_node_<node>: {name}")
    return int(match["l2_id"]), int(match["l0_id"]), int(match["node_id"])


DEFAULT_INFER_MODE: dict[str, Any] = {
    "enable_debug_LORA": False,
    "enable_debug_RAS": False,
    "enable_debug_Temp": False,
    "enable_debug_PN": False,
    "enable_debug_PN_diff_only": False,
    "enable_debug_RAW": False,
    "enable_profile": False,
    "enable_debug_calib": False,
    "repeat": 1,
    "big_small_mode": 0,
    "sub_batch_size": 20,
}


@dataclass(frozen=True, slots=True)
class NodeResult:
    """单个节点的推理结果"""

    name: str
    l2_id: int
    l0_id: int
    node_id: int
    output: np.ndarray
    golden: np.ndarray
    output_path: Path


class InferSession:
    """进入时 init, 退出时 deinit; 通道由 device 持有, 这里不关闭"""

    def __init__(
        self,
        device: DeviceOps,
        result_dir: str | Path,
        *,
        model_name: str = "test_model",
        mode: str = "vp",
    ):
        self._device = device
        self._result_dir = Path(result_dir).expanduser().resolve()
        self._result_dir.mkdir(parents=True, exist_ok=True)
        self._model_name = model_name
        self._mode = mode
        self._infer: ModuleType | None = None
        self._handles: list[Any] = []

    def __enter__(self) -> Self:
        if self._infer is not None:
            raise RuntimeError("推理会话已经启动")
        # 推理子包跟随设备声明的 driver (cim_config.yaml devices.*.driver), 与 SDK 的 chip 无关
        infer = _sdk.sdk(f"{self._device.target.driver}.infer")
        self._handles = [self._device.channel()]
        with _sdk.wrap_errors("推理运行时初始化"):
            ok = infer.init(self._handles, model_name=self._model_name, mode=self._mode)
        if not ok:
            raise DeviceError("推理运行时初始化失败")
        self._infer = infer
        log.debug("推理会话已启动 (model=%s, mode=%s)", self._model_name, self._mode)
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        infer, self._infer = self._infer, None
        if infer is None:
            return
        try:
            ok = infer.deinit(self._handles)
        except Exception as error:
            if exc is not None:
                exc.add_note(f"推理运行时清理失败: {error}")
                return
            raise DeviceError("推理运行时清理失败") from error
        if not ok:
            if exc is not None:
                exc.add_note("推理运行时清理失败: deinit 返回 False")
            else:
                raise DeviceError("推理运行时清理失败: deinit 返回 False")

    def _require(self) -> ModuleType:
        if self._infer is None:
            raise RuntimeError("推理会话尚未启动; 请在 with 内调用")
        return self._infer

    # ---- 推理 ----

    def infer_model(
        self,
        model_path: str | Path,
        *,
        dummy: int = 0,
        prefix: str | None = None,
        infer_mode: dict[str, Any] | None = None,
    ) -> list[NodeResult]:
        """
        遍历模型节点目录 (model.layers.<index>.<operation>_L2_<l2>_L0_<l0>_node_<node>), 逐节点推理

        dummy 帧用首帧复制填充, 结果中再去掉; 每个节点的输出保存为 <result_dir>/<name>.npy
        """
        infer = self._require()
        pb = _sdk.proto("infer_service_pb2")
        if isinstance(dummy, bool) or not isinstance(dummy, int) or dummy < 0:
            raise ValueError("dummy 必须是非负整数")
        if infer_mode is not None and not isinstance(infer_mode, dict):
            raise TypeError("infer_mode 必须是 dict")
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise NotADirectoryError(f"模型目录不存在: {model_path}")

        mode = pb.InferMode(**{**DEFAULT_INFER_MODE, **(infer_mode or {})})
        results: list[NodeResult] = []
        for node in sorted(path for path in model_path.iterdir() if path.is_dir()):
            l2_id, l0_id, node_id = parse_node_name(node.name)
            name = f"L2_{l2_id}_L0_{l0_id}_Node_{node_id}"
            if prefix is not None:
                name = f"{prefix}_{name}"

            input_data = np.load(node / "input_data.npy")
            if dummy:
                input_data = np.concatenate((np.repeat(input_data[:1], dummy, axis=0), input_data), axis=0)

            with _sdk.wrap_errors(f"{name} 推理"):
                responses = infer.L2Infer(
                    l2_id, [(l0_id, node_id)], [input_data], [mode], channel_handles=self._handles
                )
            if not responses:
                raise DeviceError(f"{name} 推理未返回结果")
            response = _sdk.require_success(responses[0], f"{name} 推理")
            data = response.data
            frames = 0 if data is None else data.shape[0]
            if frames <= dummy:
                raise DeviceError(f"{name} 推理返回 {frames} 帧, 不足以去掉 {dummy} 个 dummy 帧")
            output = np.asarray(data[dummy:])

            output_path = self._result_dir / f"{name}.npy"
            np.save(output_path, output)
            log.info("%s 推理完成, %d 帧 -> %s", name, output.shape[0], output_path.name)
            results.append(
                NodeResult(
                    name=name,
                    l2_id=l2_id,
                    l0_id=l0_id,
                    node_id=node_id,
                    output=output,
                    golden=np.load(node / "output_data.npy"),
                    output_path=output_path,
                )
            )
        return results

    # ---- 校准 / 日志: 直接透传 SDK, 统一错误处理 ----

    def calib(self, l2_id: int, calib_type: int) -> Any:
        with _sdk.wrap_errors("Calib"):
            return _sdk.require_success(
                self._require().Calib(l2_id, calib_type, channel_handles=self._handles), "Calib"
            )

    def set_oc_state(
        self, l2_id: int, l0_id: int, node_id: int, big_state: int, small_state: int, big_small_en: bool
    ) -> Any:
        with _sdk.wrap_errors("SetOcState"):
            response = self._require().SetOcState(
                l2_id, l0_id, node_id, big_state, small_state, big_small_en, channel_handles=self._handles
            )
            return _sdk.require_success(response, "SetOcState")

    def replace_oc_params(
        self,
        l2_id: int,
        l0_id: int,
        node_id: int,
        *,
        k_big: bytes,
        b_big: bytes,
        k_small: bytes,
        b_small: bytes,
        big_small_en: bool,
    ) -> Any:
        with _sdk.wrap_errors("ReplaceOcParams"):
            response = self._require().ReplaceOcParams(
                l2_id, l0_id, node_id, k_big, b_big, k_small, b_small, big_small_en, channel_handles=self._handles
            )
            return _sdk.require_success(response, "ReplaceOcParams")

    def get_oc_info(self, l2_id: int, l0_id: int, node_id: int, big_small_en: bool) -> Any:
        with _sdk.wrap_errors("GetOcInfo"):
            response = self._require().GetOcInfo(l2_id, l0_id, node_id, big_small_en, channel_handles=self._handles)
            return _sdk.require_success(response, "GetOcInfo", nested=True)

    def set_runtime_log_level(self, l2_id: int, level: int, *, log_path: str = "/var/log/bw01d/wit/runtime.log") -> Any:
        infer = self._require()
        config = infer.LogConfig(
            log_open=True,
            log_color=True,
            filter_lvl=level,
            filter_tag=[],
            filter_keyword=[],
            log_path=log_path,
            max_size=10 * 1024 * 1024,
            max_rotate=5,
        )
        with _sdk.wrap_errors("UpdateLogConfig"):
            return _sdk.require_success(
                infer.UpdateLogConfig(l2_id, config, channel_handles=self._handles), "UpdateLogConfig"
            )
