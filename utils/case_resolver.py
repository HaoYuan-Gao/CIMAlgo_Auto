from __future__ import annotations

import copy
import os
import numpy as np
from collections.abc import Sequence
from functools import partial
from pathlib import Path

from utils.type_checks import require_integer, require_mapping, require_string

BUFFER_SPECS = {
    "pre_comp_k": ("pre_comp_scale.npy", 128, 128, np.uint8),
    "post_comp_k": ("post_comp_scale.npy", 4224, 128, np.uint32),
    "post_comp_b": ("post_comp_bias.npy", 4224, 0, np.int8),
}


class CaseDefinitionError(ValueError):
    """Raised when a case definition is incomplete or inconsistent."""


_case_mapping = partial(require_mapping, error_type=CaseDefinitionError)
_case_integer = partial(require_integer, error_type=CaseDefinitionError)
_case_string = partial(require_string, error_type=CaseDefinitionError)


def generate_normal_case(rng, batch, in_features, out_features):
    """Generate deterministic int8 inputs and weights for a dense layer."""
    input_data = rng.integers(-128, 128, size=(batch, in_features), dtype=np.int8)
    weight = rng.integers(-128, 128, size=(in_features, out_features), dtype=np.int8)
    bias = np.zeros(out_features, dtype=np.int32)
    g_scale = np.full(out_features, 1024, dtype=np.float32)
    return input_data, weight, bias, g_scale


def _generated_arrays(generated, dimensions, field):
    if not isinstance(generated, (tuple, list)) or len(generated) != 4:
        raise CaseDefinitionError(f"{field}.case_method must return input, weight, bias and G scale")

    batch, in_features, out_features = dimensions
    names = ("input", "weight", "bias", "G_scale")
    shapes = (
        (batch, in_features),
        (in_features, out_features),
        (out_features,),
        (out_features,),
    )
    arrays = tuple(np.asarray(value) for value in generated)
    for name, array, shape in zip(names, arrays, shapes):
        if array.shape != shape:
            raise CaseDefinitionError(f"{field}.{name} shape must be {shape}, got {array.shape}")
    if np.any(arrays[3] == 0):
        raise CaseDefinitionError(f"{field}.G_scale must not contain zero")
    return arrays


def _segment_buffer(segments, *, size, default, dtype, field):
    if segments is None:
        return None
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        raise CaseDefinitionError(f"{field} must be a sequence of ranges")
    if not segments:
        return None

    dtype = np.dtype(dtype)
    limits = np.iinfo(dtype)
    buffer = np.full(size, default, dtype=dtype)
    for index, segment in enumerate(segments):
        if not isinstance(segment, Sequence) or len(segment) != 3:
            raise CaseDefinitionError(f"{field}[{index}] must contain start, end and value")
        start, end, value = (_case_integer(item, f"{field}[{index}]") for item in segment)
        if not 0 <= start < end <= size:
            raise CaseDefinitionError(f"{field}[{index}] range [{start}, {end}) is outside [0, {size})")
        if not limits.min <= value <= limits.max:
            raise CaseDefinitionError(f"{field}[{index}] value {value} does not fit {dtype}")
        buffer[start:end] = value
    return buffer


def _generate_layer(root_dir, layer, rng, index):
    field = f"layers[{index}]"
    layer = _case_mapping(layer, field)
    layer_name = _case_string(layer.get("name"), f"{field}.name")
    layer_path = Path(layer_name)
    if layer_path.is_absolute() or ".." in layer_path.parts:
        raise CaseDefinitionError(f"{field}.name must stay inside the model directory")

    dimensions = tuple(
        _case_integer(layer.get(name), f"{field}.{name}", minimum=1)
        for name in ("batch", "in_features", "out_features")
    )
    case_method = layer.get("case_method", generate_normal_case)
    if not callable(case_method):
        raise CaseDefinitionError(f"{field}.case_method must be callable")

    arrays = _generated_arrays(case_method(rng, *dimensions), dimensions, field)
    input_data, weight, bias, g_scale = (
        array.astype(dtype)
        for array, dtype in zip(
            arrays,
            (np.int8, np.int8, np.int32, np.float32),
        )
    )
    output = (input_data.astype(np.int32) @ weight.astype(np.int32) + bias) / g_scale
    output = np.clip(np.rint(output), -128, 127).astype(np.int8)

    buffers = {
        name: _segment_buffer(
            layer.get(name),
            size=size,
            default=default,
            dtype=dtype,
            field=f"{field}.{name}",
        )
        for name, (_, size, default, dtype) in BUFFER_SPECS.items()
    }

    case_dir = root_dir / layer_path
    case_dir.mkdir(parents=True, exist_ok=True)
    for filename, array in {
        "input_data.npy": input_data,
        "weight.npy": weight,
        "bias.npy": bias,
        "G_scale.npy": g_scale,
        "output_data.npy": output,
    }.items():
        np.save(case_dir / filename, array)

    for name, (filename, *_rest) in BUFFER_SPECS.items():
        path = case_dir / filename
        if buffers[name] is None:
            path.unlink(missing_ok=True)
        else:
            np.save(path, buffers[name])


def make_fake_case(*, layers, model_path, seed=42):
    """Generate fake layer data and return the model directory."""
    if not isinstance(layers, list) or not layers:
        raise CaseDefinitionError("case_def.layers must be a non-empty list")
    seed = _case_integer(seed, "case_def.seed", minimum=0)
    root_dir = Path(model_path)
    rng = np.random.default_rng(seed)
    for index, layer in enumerate(layers):
        _generate_layer(root_dir, layer, rng, index)
    return root_dir


def _normalized_case(case_def):
    case = copy.deepcopy(dict(_case_mapping(case_def, "case_def")))
    for field in ("case", "tag"):
        case[field] = _case_string(case.get(field), f"case_def.{field}")

    if case["tag"] != "fake":
        return case

    compile_config = dict(_case_mapping(case.get("compile_model"), "case_def.compile_model"))
    if not isinstance(compile_config.get("model_path"), (str, os.PathLike)):
        raise CaseDefinitionError("case_def.compile_model.model_path must be path-like")

    case["compile_model"] = compile_config
    return case


def resolve_case(case_def, *, base_dir=None):
    """Validate a case, normalize its paths and prepare generated data."""
    case = _normalized_case(case_def)
    if case["tag"] != "fake":
        return case

    compile_config = case["compile_model"]
    model_path = Path(compile_config["model_path"]).expanduser()
    if not model_path.is_absolute():
        model_path = Path(base_dir or Path.cwd()) / model_path
    model_path = model_path.resolve()
    compile_config["model_path"] = str(model_path)

    make_fake_case(
        layers=case.get("layers"),
        seed=case.get("seed", 42),
        model_path=model_path,
    )
    return case
