from copy import deepcopy
from pathlib import Path
import re

import numpy as np
import pytest

from utils.case_resolver import CaseDefinitionError, resolve_case


def _fake_case(model_path="generated/model"):
    return {
        "case": "dense-fake",
        "tag": "fake",
        "seed": 7,
        "compile_model": {
            "model_path": model_path,
        },
        "layers": [
            {
                "name": "layer_0",
                "batch": 2,
                "in_features": 3,
                "out_features": 4,
                "pre_comp_k": [(0, 2, 64)],
                "post_comp_k": [(1, 3, 256)],
                "post_comp_b": [(2, 4, -3)],
            }
        ],
    }


def test_resolve_fake_case_generates_valid_layer_data(tmp_path):
    case_def = _fake_case()
    original = deepcopy(case_def)

    resolved = resolve_case(case_def, base_dir=tmp_path)

    assert case_def == original
    model_path = Path(resolved["compile_model"]["model_path"])
    assert model_path == (tmp_path / "generated" / "model").resolve()

    layer_path = model_path / "layer_0"
    input_data = np.load(layer_path / "input_data.npy")
    weight = np.load(layer_path / "weight.npy")
    bias = np.load(layer_path / "bias.npy")
    g_scale = np.load(layer_path / "G_scale.npy")
    output = np.load(layer_path / "output_data.npy")

    assert input_data.shape == (2, 3)
    assert input_data.dtype == np.int8
    assert weight.shape == (3, 4)
    assert weight.dtype == np.int8
    assert bias.shape == (4,)
    assert bias.dtype == np.int32
    assert g_scale.shape == (4,)
    assert g_scale.dtype == np.float32

    expected_output = (input_data.astype(np.int32) @ weight.astype(np.int32) + bias) / g_scale
    expected_output = np.clip(np.rint(expected_output), -128, 127).astype(np.int8)
    np.testing.assert_array_equal(output, expected_output)

    pre_comp = np.load(layer_path / "pre_comp_scale.npy")
    post_comp = np.load(layer_path / "post_comp_scale.npy")
    post_bias = np.load(layer_path / "post_comp_bias.npy")
    np.testing.assert_array_equal(pre_comp[:3], [64, 64, 128])
    np.testing.assert_array_equal(post_comp[:4], [128, 256, 256, 128])
    np.testing.assert_array_equal(post_bias[:5], [0, 0, -3, -3, 0])


def test_resolve_fake_case_refreshes_existing_data(tmp_path):
    case_def = _fake_case(model_path=tmp_path / "model")
    resolved = resolve_case(case_def)
    layer_path = Path(resolved["compile_model"]["model_path"]) / "layer_0"
    expected_input = np.load(layer_path / "input_data.npy").copy()

    np.save(layer_path / "input_data.npy", np.zeros((2, 3), dtype=np.int8))
    case_def["layers"][0].pop("pre_comp_k")
    resolve_case(case_def)

    np.testing.assert_array_equal(
        np.load(layer_path / "input_data.npy"),
        expected_input,
    )
    assert not (layer_path / "pre_comp_scale.npy").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda case: case.pop("case"), "case_def.case"),
        (
            lambda case: case.pop("layers"),
            "case_def.layers",
        ),
        (
            lambda case: case["layers"][0].update({"name": "../escape"}),
            "must stay inside",
        ),
        (
            lambda case: case["layers"][0].update({"pre_comp_k": [(0, 129, 1)]}),
            "outside [0, 128)",
        ),
    ],
    ids=(
        "missing-case",
        "missing-layers",
        "unsafe-layer-directory",
        "invalid-pre-comp-range",
    ),
)
def test_invalid_fake_case_has_contextual_error(tmp_path, change, message):
    case_def = _fake_case()
    change(case_def)

    with pytest.raises(CaseDefinitionError, match=r".*" + re.escape(message)):
        resolve_case(case_def, base_dir=tmp_path)


@pytest.mark.parametrize(
    "case_context",
    [
        {
            "case": "external-case",
            "tag": "external",
            "model": "existing",
        }
    ],
    indirect=True,
)
def test_case_context_uses_indirect_parametrization(case_context):
    assert case_context == {
        "case": "external-case",
        "tag": "external",
        "model": "existing",
    }
