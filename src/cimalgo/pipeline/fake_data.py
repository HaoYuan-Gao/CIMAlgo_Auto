"""生成假模型的输入、权重和期望输出, 用于不依赖真实模型的回归 case"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cimalgo.config import validators as v

# 补偿缓冲区: 文件名, 长度, 默认值, dtype
BUFFER_SPECS: dict[str, tuple[str, int, int, type]] = {
    "pre_comp_k": ("pre_comp_scale.npy", 128, 128, np.uint8),
    "post_comp_k": ("post_comp_scale.npy", 4224, 128, np.uint32),
    "post_comp_b": ("post_comp_bias.npy", 4224, 0, np.int8),
}

GeneratorFn = Callable[[np.random.Generator, int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


class FakeDataError(ValueError):
    """假数据定义不完整或不一致"""


def generate_dense(rng: np.random.Generator, batch: int, in_features: int, out_features: int):
    """默认生成器: 均匀分布的 int8 输入和权重, 零偏置, 固定 G scale"""
    input_data = rng.integers(-128, 128, size=(batch, in_features), dtype=np.int8)
    weight = rng.integers(-128, 128, size=(in_features, out_features), dtype=np.int8)
    bias = np.zeros(out_features, dtype=np.int32)
    g_scale = np.full(out_features, 1024, dtype=np.float32)
    return input_data, weight, bias, g_scale


def _segment_buffer(segments: Any, *, size: int, default: int, dtype: type, field: str) -> np.ndarray | None:
    if segments is None:
        return None
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        raise FakeDataError(f"{field} must be a sequence of [start, end, value]")
    if not segments:
        return None
    limits = np.iinfo(dtype)
    buffer: np.ndarray = np.full(size, default, dtype=dtype)
    for index, segment in enumerate(segments):
        if not isinstance(segment, Sequence) or len(segment) != 3:
            raise FakeDataError(f"{field}[{index}] must contain start, end and value")
        try:
            start, end, value = (v.integer(item, f"{field}[{index}]") for item in segment)
        except v.ProjectConfigError as error:
            raise FakeDataError(str(error)) from None
        if not 0 <= start < end <= size:
            raise FakeDataError(f"{field}[{index}] range [{start}, {end}) is outside [0, {size})")
        if not limits.min <= value <= limits.max:
            raise FakeDataError(f"{field}[{index}] value {value} does not fit {np.dtype(dtype)}")
        buffer[start:end] = value
    return buffer


def _generate_layer(root: Path, layer: Mapping[str, Any], rng: np.random.Generator, index: int) -> Path:
    field = f"layers[{index}]"
    try:
        name = v.string(layer.get("name"), f"{field}.name")
        batch, in_features, out_features = (
            v.integer(layer.get(key), f"{field}.{key}", minimum=1) for key in ("batch", "in_features", "out_features")
        )
    except v.ProjectConfigError as error:
        raise FakeDataError(str(error)) from None
    layer_path = Path(name)
    if layer_path.is_absolute() or ".." in layer_path.parts:
        raise FakeDataError(f"{field}.name must stay inside the model directory")

    generator: GeneratorFn = layer.get("generator", generate_dense)
    if not callable(generator):
        raise FakeDataError(f"{field}.generator must be callable")
    generated = generator(rng, batch, in_features, out_features)
    if not isinstance(generated, (tuple, list)) or len(generated) != 4:
        raise FakeDataError(f"{field}.generator must return input, weight, bias and G scale")
    expected_shapes = ((batch, in_features), (in_features, out_features), (out_features,), (out_features,))
    arrays = tuple(np.asarray(value) for value in generated)
    for label, array, shape in zip(("input", "weight", "bias", "G_scale"), arrays, expected_shapes, strict=True):
        if array.shape != shape:
            raise FakeDataError(f"{field}.{label} shape must be {shape}, got {array.shape}")
    if np.any(arrays[3] == 0):
        raise FakeDataError(f"{field}.G_scale must not contain zero")

    input_data = arrays[0].astype(np.int8)
    weight = arrays[1].astype(np.int8)
    bias = arrays[2].astype(np.int32)
    g_scale = arrays[3].astype(np.float32)
    # 用 int32 累加避免溢出, 再按 G scale 量化回 int8
    output = (input_data.astype(np.int32) @ weight.astype(np.int32) + bias) / g_scale
    output = np.clip(np.rint(output), -128, 127).astype(np.int8)

    layer_dir = root / layer_path
    layer_dir.mkdir(parents=True, exist_ok=True)
    for filename, array in (
        ("input_data.npy", input_data),
        ("weight.npy", weight),
        ("bias.npy", bias),
        ("G_scale.npy", g_scale),
        ("output_data.npy", output),
    ):
        np.save(layer_dir / filename, array)

    for key, (filename, size, default, dtype) in BUFFER_SPECS.items():
        buffer = _segment_buffer(layer.get(key), size=size, default=default, dtype=dtype, field=f"{field}.{key}")
        path = layer_dir / filename
        if buffer is None:
            path.unlink(missing_ok=True)
        else:
            np.save(path, buffer)
    return layer_dir


def make_fake_model(model_path: str | Path, layers: Sequence[Mapping[str, Any]], *, seed: int = 42) -> Path:
    """按 layers 定义生成假模型目录; 同一 seed 产生相同数据, 重复调用会覆盖"""
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)) or not layers:
        raise FakeDataError("layers must be a non-empty list")
    try:
        seed = v.integer(seed, "seed", minimum=0)
    except v.ProjectConfigError as error:
        raise FakeDataError(str(error)) from None
    root = Path(model_path)
    rng = np.random.default_rng(seed)
    for index, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            raise FakeDataError(f"layers[{index}] must be a mapping")
        _generate_layer(root, layer, rng, index)
    return root
