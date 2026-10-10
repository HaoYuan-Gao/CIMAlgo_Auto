"""在 with 作用域内管理芯片推理运行时"""

from __future__ import annotations

import re
import numpy as np
from pathlib import Path
from importlib import import_module
from typing import TYPE_CHECKING, Any
from types import ModuleType, TracebackType

if TYPE_CHECKING:
    from wtm.channel_handle import ChannelHandle

_NODE_DIR_PATTERN = re.compile(r"L2_(\d+)_L0_(\d+)_node_(\d+)$")

class InferSession:
    """进入时 init, 退出时只对本次通道执行 deinit; 不关闭通道"""

    def __init__(
        self,
        channel_handles: ChannelHandle | list[ChannelHandle],
        result_dir: str | Path,
        *,
        model_name: str = "test_model",
        mode: str = "vp",
        chip: str = "bw01d",
    ) -> None:
        self._handles = list(channel_handles) if isinstance(channel_handles, list) else [channel_handles]
        if not self._handles:
            raise ValueError("channel_handles 不能为空")
        self._result_dir = Path(result_dir).expanduser().resolve()
        self._result_dir.mkdir(parents=True, exist_ok=True)
        self._model_name = model_name
        self._mode = mode
        self._infer: ModuleType | None = None
        self._chip = chip

    def __enter__(self) -> InferSession:
        if self._infer is not None:
            raise RuntimeError("推理会话已经启动")
        infer = import_module(f"wtm.{self._chip}.infer")
        if not infer.init(self._handles, model_name=self._model_name, mode=self._mode):
            raise RuntimeError(f"{self._chip} 推理运行时初始化失败")
        self._infer = infer
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        infer, self._infer = self._infer, None
        if infer is None:
            raise RuntimeError("推理会话尚未启动")
        try:
            success = infer.deinit(self._handles)
        except Exception as error:
            if exc is not None:
                exc.add_note(f"{self._chip} 推理运行时清理失败: {error}")
                return False
            raise RuntimeError(f"{self._chip} 推理运行时清理失败") from error
        if not success:
            if exc is not None:
                exc.add_note(f"{self._chip} 推理运行时清理失败: deinit 返回 False")
            else:
                raise RuntimeError(f"{self._chip} 推理运行时清理失败: deinit 返回 False")
        return False

    def _require_infer(self) -> ModuleType:
        if self._infer is None:
            raise RuntimeError("推理会话尚未启动；请在 with 内调用")
        return self._infer

    @staticmethod
    def _check_status(response: Any, operation: str, *, nested: bool = False) -> Any:
        """检查 SDK 返回状态；成功时保留原始返回值"""
        status = response.status if nested else response
        if not status.success:
            raise RuntimeError(f"{operation} 失败: {status.error or 'SDK 返回失败状态'}")
        return response

    def L2Infer(
        self,
        model_path: str | Path,
        dummy: int = 0,
        prefix_name: str | None = None,
        infer_modes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        遍历编译产物目录下的每个节点目录, 逐节点同步推理、丢弃 dummy 帧、保存结果并绘制指标

        节点目录名形如 L2_<l2>_L0_<l0>_node_<node>, 内含 input_data.npy 和 output_data.npy;
        不匹配的子目录直接报错, 不会静默跳过

        返回 {推理结果名: SDK 结果对象}, 结果对象的 data 已去掉 dummy 帧
        """
        infer = self._require_infer()
        import infer_service_pb2 as pb
        from analyzer import draw_basic_metrics

        if isinstance(dummy, bool) or not isinstance(dummy, int) or dummy < 0:
            raise ValueError("dummy 必须是非负整数")
        if infer_modes is not None and not isinstance(infer_modes, dict):
            raise TypeError("infer_modes 必须是 dict")
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise NotADirectoryError(f"模型目录不存在: {model_path}")

        mode_options = {
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
        if infer_modes:
            mode_options.update(infer_modes)
        infer_mode = pb.InferMode(**mode_options)

        results: dict[str, Any] = {}
        # 固定顺序, 让结果文件和报告顺序可复现
        for node in sorted(path for path in model_path.iterdir() if path.is_dir()):
            match = _NODE_DIR_PATTERN.search(node.name)
            if match is None:
                raise ValueError(f"节点目录名必须形如 L2_<l2>_L0_<l0>_node_<node>: {node}")
            l2_id, l0_id, node_id = (int(group) for group in match.groups())

            input_data = np.load(node / "input_data.npy")
            if dummy > 0:
                input_data = np.concatenate((np.repeat(input_data[:1], dummy, axis=0), input_data), axis=0)

            responses = infer.L2Infer(l2_id, [(l0_id, node_id)], [input_data], [infer_mode], channel_handles=self._handles)
            if not responses:
                raise RuntimeError(f"L2_{l2_id} 推理未返回结果: {node.name}")
            result = self._check_status(responses[0], f"L2_{l2_id} 推理结果")

            infer_name = f"L2_{l2_id}_L0_{l0_id}_Node_{node_id}"
            if prefix_name is not None:
                infer_name = f"{prefix_name}_{infer_name}"

            # 帧数沿第 0 维; 不足以去掉 dummy 帧时切片会得到空数组, 这里提前报错
            if result.data is None or result.data.shape[0] <= dummy:
                frames = 0 if result.data is None else result.data.shape[0]
                raise RuntimeError(f"{infer_name} 推理返回 {frames} 帧, 不足以去掉 {dummy} 个 dummy 帧")
            result.data = result.data[dummy:]
            np.save(self._result_dir / f"{infer_name}.npy", result.data)

            golden = np.load(node / "output_data.npy")
            draw_basic_metrics(
                gold=golden,
                chip=result.data,
                save_path=str(self._result_dir / f"{infer_name}.png"),
            )
            results[infer_name] = result
        return results

    def Calib(self, logical_L2_id: int, calib_type: int) -> Any:
        """触发校准"""
        response = self._require_infer().Calib(logical_L2_id, calib_type, channel_handles=self._handles)
        return self._check_status(response, "Calib")

    def SetOcState(
        self,
        logical_L2_id: int,
        l0_id: int,
        node_id: int,
        big_state: int,
        small_state: int,
        big_small_en: bool,
    ) -> Any:
        """设置输出校准状态"""
        response = self._require_infer().SetOcState(
            logical_L2_id,
            l0_id,
            node_id,
            big_state,
            small_state,
            big_small_en,
            channel_handles=self._handles,
        )
        return self._check_status(response, "SetOcState")

    def ReplaceOcParams(
        self,
        logical_L2_id: int,
        l0_id: int,
        node_id: int,
        input_k_big: bytes,
        input_b_big: bytes,
        input_k_small: bytes,
        input_b_small: bytes,
        big_small_en: bool,
    ) -> Any:
        """替换输出校准参数"""
        response = self._require_infer().ReplaceOcParams(
            logical_L2_id,
            l0_id,
            node_id,
            input_k_big,
            input_b_big,
            input_k_small,
            input_b_small,
            big_small_en,
            channel_handles=self._handles,
        )
        return self._check_status(response, "ReplaceOcParams")

    def GetOcInfo(self, logical_L2_id: int, l0_id: int, node_id: int, big_small_en: bool) -> Any:
        """读取输出校准信息"""
        response = self._require_infer().GetOcInfo(
            logical_L2_id, l0_id, node_id, big_small_en, channel_handles=self._handles
        )
        return self._check_status(response, "GetOcInfo", nested=True)

    def UpdateLogConfig(self, logical_L2_id: int, log_lvl: int) -> Any:
        """更新运行时日志配置, 仅修改日志级别"""
        log_config = self.LogConfig(
            log_open=True,
            log_color=True,
            filter_lvl=log_lvl,
            filter_tag=[],
            filter_keyword=[],
            log_path="/var/log/bw01d/wit/runtime.log",
            max_size=10 * 1024 * 1024,
            max_rotate=5,
        )

        response = self._require_infer().UpdateLogConfig(logical_L2_id, log_config, channel_handles=self._handles)
        return self._check_status(response, "UpdateLogConfig")

    @property
    def LogConfig(self) -> type:
        """返回 SDK 的 LogConfig 类型，用于构造日志配置"""
        return self._require_infer().LogConfig
