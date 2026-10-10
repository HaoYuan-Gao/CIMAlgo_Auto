"""pipeline 层测试: case 解析、假数据、工具链 (subprocess 打桩)"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from cimalgo.pipeline import (
    Bw01dToolchain,
    CaseDefinitionError,
    MapOptions,
    ToolchainError,
    load_case,
    load_cases,
    make_fake_model,
    map_l2_id,
)
from cimalgo.pipeline import toolchain as toolchain_module
from cimalgo.pipeline.case import Threshold, parse_case
from cimalgo.pipeline.fake_data import FakeDataError

# ---------------------------------------------------------------------------
# case
# ---------------------------------------------------------------------------

CASE = {
    "case": "dense_fake",
    "tags": ["smoke"],
    "device": {
        "name": "evb-01",
        "firmware_version": "2.3.3",
        "hard_blocks": [0, 1],
        "select_layers": [0],
        "tsv_sel1": 200,
    },
    "model": {
        "kind": "fake",
        "seed": 7,
        "layers": [
            {
                "name": "model.layers.0.dense_L2_0_L0_0_node_0",
                "batch": 2,
                "in_features": 3,
                "out_features": 4,
            }
        ],
    },
    "infer": {"dummy": 2, "mode": {"repeat": 3}},
    "thresholds": {"cs_overall": ">= 0.99", "error_std_max": "<= 2"},
}


def test_parse_full_case(tmp_path):
    spec = parse_case(CASE, base_dir=tmp_path)
    assert spec.name == "dense_fake"
    assert spec.tags == ("smoke",)
    assert spec.device.name == "evb-01"
    assert spec.device.hard_blocks == (0, 1)
    assert spec.device.tsv_sel1 == 200 and spec.device.tsv_sel0 == 0
    assert spec.device.power_cycle is True
    assert spec.model.kind == "fake" and spec.model.seed == 7 and spec.model.path is None
    assert spec.infer.dummy == 2 and spec.infer.mode == {"repeat": 3}
    assert spec.thresholds == (Threshold("cs_overall", ">=", 0.99), Threshold("error_std_max", "<=", 2.0))


def test_device_defaults_other_than_required_fields(tmp_path):
    spec = parse_case({**CASE, "device": {"name": "evb-02", "firmware_version": "2.3.3"}}, base_dir=tmp_path)
    assert spec.device.name == "evb-02" and spec.device.hard_blocks == ()
    assert spec.device.grpc_port == 60000 and spec.device.power_cycle is True


def test_device_grpc_port_selects_site(tmp_path):
    spec = parse_case({**CASE, "device": {**CASE["device"], "grpc_port": 60010}}, base_dir=tmp_path)
    assert spec.device.grpc_port == 60010
    # 端口不属于芯片状态: 两个只有端口不同的 case 的 InitState 相同
    assert spec.device.init_state() == parse_case(CASE, base_dir=tmp_path).device.init_state()


def test_device_trim_lists_align_with_select_layers(tmp_path):
    # CASE 的 select_layers 是 [0]; 给一个 HB 配 trim0, trim1 不适用
    device = {**CASE["device"], "cim_trim0": ["trim/t0.csv"], "cim_trim1": [None]}
    spec = parse_case({**CASE, "device": device}, base_dir=tmp_path / "cases")
    assert spec.device.cim_trim0 == ((tmp_path / "cases/trim/t0.csv").resolve(),)
    assert spec.device.cim_trim1 == (None,)
    # 绝对路径原样保留; trim 进入 InitState 参与状态比较
    device = {**CASE["device"], "select_layers": [0, 4], "cim_trim1": [None, "/abs/t1.csv"]}
    spec = parse_case({**CASE, "device": device}, base_dir=tmp_path)
    assert spec.device.cim_trim1 == (None, Path("/abs/t1.csv").resolve())
    state = spec.device.init_state()
    assert state.trims_for(0) == (None, None) and state.trims_for(1) == (None, Path("/abs/t1.csv").resolve())
    assert state != parse_case(CASE, base_dir=tmp_path).device.init_state()
    # 省略时为空元组
    assert parse_case(CASE, base_dir=tmp_path).device.cim_trim0 == ()


@pytest.mark.parametrize(
    ("trim", "message"),
    [
        # 单个字符串会被广播给所有 HB, 正是要避免的错误
        ("trim/t0.csv", r"device\.cim_trim0 must be a list aligned with select_layers"),
        (["a.csv", "b.csv"], r"device\.cim_trim0 has 2 entries but select_layers has 1"),
        ([], r"device\.cim_trim0 has 0 entries but select_layers has 1"),
        ([3], r"device\.cim_trim0\[0\]"),
    ],
)
def test_device_trim_list_shape_is_validated(tmp_path, trim, message):
    with pytest.raises(CaseDefinitionError, match=message):
        parse_case({**CASE, "device": {**CASE["device"], "cim_trim0": trim}}, base_dir=tmp_path)


def test_device_firmware_version_is_required_and_case_level(tmp_path):
    spec = parse_case({**CASE, "device": {**CASE["device"], "firmware_version": "1.9"}}, base_dir=tmp_path)
    assert spec.device.firmware_version == "1.9"
    assert spec.device.init_state().firmware_version == "1.9"
    # 省略直接报错: 固件是设备状态, 不能从 SDK 版本推断
    without = {k: val for k, val in CASE["device"].items() if k != "firmware_version"}
    with pytest.raises(CaseDefinitionError, match=r"device\.firmware_version is required"):
        parse_case({**CASE, "device": without}, base_dir=tmp_path)
    # 字符串简写没有地方放固件版本, 不再支持
    with pytest.raises(CaseDefinitionError, match="device must be a mapping"):
        parse_case({**CASE, "device": "evb-02"}, base_dir=tmp_path)


def test_external_model_path_resolved_relative_to_case_file(tmp_path):
    spec = parse_case({**CASE, "model": {"kind": "external", "path": "models/x"}}, base_dir=tmp_path / "cases")
    assert spec.model.path == (tmp_path / "cases/models/x").resolve()


def test_threshold_check():
    assert Threshold("x", ">=", 1.0).check(1.0)
    assert not Threshold("x", "<", 1.0).check(1.0)
    assert str(Threshold("x", "<=", 2.5)) == "x <= 2.5"


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"case": "bad name"}, "case"),
        ({"unknown": 1}, "case.unknown is not a supported"),
        ({"device": {"name": "x", "firmware_version": "1", "bogus": 1}}, "device.bogus"),
        ({"device": {"name": "x", "firmware_version": "1", "grpc_port": 0}}, "device.grpc_port"),
        ({"device": {"name": "x", "firmware_version": "1", "grpc_port": "60000"}}, "device.grpc_port"),
        ({"device": {"name": "x", "firmware_version": 2}}, "device.firmware_version"),
        ({"device": {"name": "x"}}, "device.firmware_version is required"),
        ({"device": "x"}, "device must be a mapping"),
        ({"device": {"name": "x", "firmware_version": "1", "tsv_sel0": 300}}, "device.tsv_sel0"),
        ({"device": {"name": "x", "firmware_version": "1", "hard_blocks": "0,1"}}, "device.hard_blocks"),
        ({"model": {"kind": "other"}}, "model.kind"),
        ({"model": {"kind": "fake", "seed": 1, "layers": [{}], "bogus": 1}}, "model.bogus is not a supported"),
        ({"model": {"kind": "external", "path": "m", "typo": 1}}, "model.typo is not a supported"),
        ({"infer": {"dummy": 1, "bogus": 1}}, "infer.bogus is not a supported"),
        ({"model": {"kind": "external"}}, "model.path is required"),
        ({"model": {"kind": "fake"}}, "model.layers"),
        ({"infer": {"dummy": -1}}, "infer.dummy"),
        ({"thresholds": {"x": "0.9"}}, "thresholds.x"),
        ({"thresholds": {"x": ">= abc"}}, "thresholds.x"),
        ({"thresholds": {"x": "~= 1"}}, "thresholds.x"),
    ],
)
def test_parse_rejects_invalid(tmp_path, patch, message):
    with pytest.raises(CaseDefinitionError, match=message):
        parse_case({**CASE, **patch}, base_dir=tmp_path)


def test_load_cases_from_directory_sorted_and_unique(tmp_path):
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({**CASE, "case": "b"}))
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/a.yaml").write_text(yaml.safe_dump({**CASE, "case": "a"}))
    assert [spec.name for spec in load_cases(tmp_path)] == ["b", "a"]
    (tmp_path / "dup.yaml").write_text(yaml.safe_dump({**CASE, "case": "a"}))
    with pytest.raises(CaseDefinitionError, match="名称重复"):
        list(load_cases(tmp_path))


def test_load_case_error_includes_path(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("case: [")
    with pytest.raises(CaseDefinitionError, match=str(path)):
        load_case(path)


@pytest.mark.parametrize("kind", ["missing", "file", "empty"])
def test_load_cases_rejects_invalid_or_empty_directory(tmp_path, kind):
    path = tmp_path / "cases"
    if kind == "file":
        path.write_text("not a directory")
    elif kind == "empty":
        path.mkdir()
    with pytest.raises(CaseDefinitionError, match="case 目录"):
        list(load_cases(path))


# ---------------------------------------------------------------------------
# fake data
# ---------------------------------------------------------------------------


def test_make_fake_model_is_deterministic_and_consistent(tmp_path):
    name = "model.layers.0.dense_L2_0_L0_0_node_0"
    layers = [{"name": name, "batch": 2, "in_features": 3, "out_features": 4, "pre_comp_k": [[0, 2, 64]]}]
    root = make_fake_model(tmp_path / "m1", layers, seed=7)
    again = make_fake_model(tmp_path / "m2", layers, seed=7)
    layer = root / name

    input_data = np.load(layer / "input_data.npy")
    weight = np.load(layer / "weight.npy")
    output = np.load(layer / "output_data.npy")
    assert input_data.dtype == np.int8 and input_data.shape == (2, 3)
    assert weight.shape == (3, 4)
    expected = np.clip(np.rint((input_data.astype(np.int32) @ weight.astype(np.int32)) / 1024), -128, 127).astype(
        np.int8
    )
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(np.load(again / name / "input_data.npy"), input_data)
    np.testing.assert_array_equal(np.load(layer / "pre_comp_scale.npy")[:3], [64, 64, 128])
    assert not (layer / "post_comp_scale.npy").exists()


def test_make_fake_model_removes_stale_buffers(tmp_path):
    layers = [{"name": "n", "batch": 1, "in_features": 1, "out_features": 1, "pre_comp_k": [[0, 1, 1]]}]
    root = make_fake_model(tmp_path / "m", layers)
    assert (root / "n/pre_comp_scale.npy").exists()
    make_fake_model(tmp_path / "m", [{**layers[0], "pre_comp_k": None}])
    assert not (root / "n/pre_comp_scale.npy").exists()


@pytest.mark.parametrize(
    ("layer", "message"),
    [
        ({"name": "../x", "batch": 1, "in_features": 1, "out_features": 1}, "stay inside"),
        ({"name": "n", "batch": 0, "in_features": 1, "out_features": 1}, "batch"),
        ({"name": "n", "batch": 1, "in_features": 1, "out_features": 1, "pre_comp_k": [[0, 200, 1]]}, "outside"),
        ({"name": "n", "batch": 1, "in_features": 1, "out_features": 1, "post_comp_b": [[0, 1, 999]]}, "does not fit"),
        ({"name": "n", "batch": 1, "in_features": 1, "out_features": 1, "generator": "x"}, "callable"),
    ],
)
def test_make_fake_model_rejects_invalid_layer(tmp_path, layer, message):
    with pytest.raises(FakeDataError, match=message):
        make_fake_model(tmp_path / "m", [layer])


def test_custom_generator_shape_is_validated(tmp_path):
    def bad(rng, batch, i, o):
        return np.zeros((batch, i)), np.zeros((i, o)), np.zeros(o), np.zeros(o + 1)

    with pytest.raises(FakeDataError, match="G_scale shape"):
        make_fake_model(
            tmp_path / "m", [{"name": "n", "batch": 1, "in_features": 1, "out_features": 1, "generator": bad}]
        )


# ---------------------------------------------------------------------------
# toolchain
# ---------------------------------------------------------------------------


@pytest.fixture
def toolchain(tmp_path, monkeypatch) -> tuple[Bw01dToolchain, Mock]:
    compiler = tmp_path / "compiler"
    for rel in (
        "script/gen_node_cfg.py",
        "script/gen_model_cfg.py",
        "script/append_oc_to_calib.py",
        "bin/bw01d_compiler",
    ):
        (compiler / rel).parent.mkdir(parents=True, exist_ok=True)
        (compiler / rel).write_text("")
    mapper = tmp_path / "map"
    mapper.mkdir()
    model = tmp_path / "model"
    (model / "model.layers.0.dense_L2_2_L0_0_node_0").mkdir(parents=True)
    run = Mock()
    monkeypatch.setattr(toolchain_module.subprocess, "run", run)
    return Bw01dToolchain(compiler, mapper, model, tmp_path / "compile"), run


def test_generate_node_config_clears_old_files_and_runs_script(toolchain):
    tc, run = toolchain
    stale = tc.model_path / "model.layers.0.dense_L2_2_L0_0_node_0/node_cfg.json"
    stale.write_text("{}")
    (tc.model_path / "keep.txt").write_text("x")
    tc.generate_node_config()
    assert not stale.exists() and (tc.model_path / "keep.txt").exists()
    assert run.call_args.args[0][1].endswith("gen_node_cfg.py")
    assert run.call_args.kwargs["cwd"] == tc.compiler_tools


def test_update_node_configs_deep_merges(toolchain):
    tc, _ = toolchain
    cfg = tc.model_path / "model.layers.0.dense_L2_2_L0_0_node_0/node_cfg.json"
    cfg.write_text(json.dumps({"a": {"b": 1, "c": 2}, "d": 3}))
    tc.update_node_configs(a={"b": 9}, e=5)
    assert json.loads(cfg.read_text()) == {"a": {"b": 9, "c": 2}, "d": 3, "e": 5}


def test_compile_model_requires_generated_model_cfg(toolchain):
    tc, _ = toolchain
    with pytest.raises(ToolchainError, match=r"model_cfg\.json"):
        tc.compile_model()


def test_compile_model_invokes_compiler_and_optional_oc(toolchain):
    tc, run = toolchain
    maps = [tc.compile_path / "output" / f"Die_0_L2_{i}" for i in (2, 5)]

    def fake_run(command, **kwargs):
        if command[1].endswith("gen_model_cfg.py"):
            (tc.model_path / "model_cfg.json").write_text("{}")
        if command[0].endswith("bw01d_compiler"):
            for map in maps:
                map.mkdir(parents=True)
                (map / "config.bin").write_bytes(b"config")
        if command[0].endswith("gen_tempcomp"):
            assert Path(command[1]) in maps
            (Path(command[1]) / "calib.bin").write_bytes(b"ic")
        if command[0].endswith("PostCompBinMaker"):
            assert Path(command[2]).is_file()
            assert Path(command[2]).parent in maps
            Path(command[4]).write_bytes(b"oc")
        if command[1].endswith("append_oc_to_calib.py"):
            assert Path(command[3]).is_file() and Path(command[5]).is_file()
            assert Path(command[3]).parent == Path(command[5]).parent

    run.side_effect = fake_run
    assert tc.compile_model(oc_comp=True) == tc.compile_path
    executables = [Path(call.args[0][0]).name for call in run.call_args_list]
    assert executables[1] == "bw01d_compiler"
    assert executables.count("gen_tempcomp") == 2 and executables.count("PostCompBinMaker") == 2
    assert not (tc.compile_path / "config.bin").exists()


def _make_maps(tc, *names):
    for name in names:
        map = tc.compile_path / "output" / name
        map.mkdir(parents=True)
        (map / "config.bin").write_bytes(b"x")


def _map_commands(run):
    return [call.args[0] for call in run.call_args_list if call.args[0][0].endswith("map_tool")]


def _flag(command, flag):
    return command[command.index(flag) + 1] if flag in command else None


def test_remap_maps_packages_each_map_and_returns_jobs(toolchain):
    tc, run = toolchain
    _make_maps(tc, "Die_0_L2_1")
    jobs = tc.remap_maps({1: MapOptions(stage=2)})
    assert jobs == [(1, tc.compile_path / "Die_0_L2_1.tar.gz")]
    assert jobs[0][1].is_file()
    (command,) = _map_commands(run)
    assert _flag(command, "-l2") == "1" and _flag(command, "-stg") == "2" and "-r" not in command


def test_remap_maps_preserves_non_contiguous_l2_ids(toolchain):
    tc, run = toolchain
    _make_maps(tc, "Die_0_L2_2", "Die_0_L2_10")
    jobs = tc.remap_maps()
    # 字典序是 10、2, 目标编号必须仍然是 10、2, 不能变成 0、1
    assert [l2 for l2, _ in jobs] == [10, 2]
    assert [_flag(c, "-l2") for c in _map_commands(run)] == ["10", "2"]
    assert [path.name for _, path in jobs] == ["Die_0_L2_10.tar.gz", "Die_0_L2_2.tar.gz"]


def test_remap_maps_applies_per_l2_options(toolchain, tmp_path):
    tc, run = toolchain
    _make_maps(tc, "Die_0_L2_2", "Die_1_L2_5", "Die_2_L2_7")
    redundancy = tmp_path / "r5.bin"
    redundancy.write_bytes(b"r")
    jobs = tc.remap_maps(
        {
            2: MapOptions(stage=1),
            5: MapOptions(stage=4, redundancy_info=redundancy),
            # L2 7 没有配置: 默认参数
        }
    )
    assert [l2 for l2, _ in jobs] == [2, 5, 7]
    by_source = {c[c.index("-i") + 1].rsplit("_", 1)[-1]: c for c in _map_commands(run)}
    assert _flag(by_source["2"], "-stg") == "1" and "-r" not in by_source["2"]
    assert _flag(by_source["5"], "-stg") == "4"
    assert _flag(by_source["5"], "-r") == str(redundancy.resolve())
    assert _flag(by_source["5"], "-l2") == "5"
    assert _flag(by_source["7"], "-stg") is None and _flag(by_source["7"], "-l2") == "7"


@pytest.mark.parametrize(
    ("names", "options", "message"),
    [
        (["Die_0_L2_2", "Die_1_L2_2"], None, "重复"),
        (["unknown"], None, "解析 L2"),
        (["Die_0_L2_2"], {3: MapOptions()}, "没有对应的 Map"),
    ],
)
def test_remap_maps_rejects_invalid_targets_before_mapping(toolchain, names, options, message):
    tc, run = toolchain
    _make_maps(tc, *names)
    with pytest.raises(ToolchainError, match=message):
        tc.remap_maps(options)
    run.assert_not_called()


@pytest.mark.parametrize("options", [[], {"0": MapOptions()}, {True: MapOptions()}, {0: object()}])
def test_remap_maps_validates_options_mapping(toolchain, options):
    tc, run = toolchain
    _make_maps(tc, "Die_0_L2_0")
    with pytest.raises(TypeError, match=r"Mapping\[int, MapOptions\]"):
        tc.remap_maps(options)
    run.assert_not_called()


def test_remap_maps_overwrites_archive_but_rejects_symlink(toolchain):
    tc, _ = toolchain
    map = tc.compile_path / "output/Die_0_L2_0"
    map.mkdir(parents=True)
    (map / "config.bin").write_bytes(b"new")
    archive = tc.compile_path / "Die_0_L2_0.tar.gz"
    archive.write_bytes(b"old")
    assert tc.remap_maps() == [(0, archive)]
    with tarfile.open(archive) as tar:
        assert tar.extractfile("config.bin").read() == b"new"
    target = archive.with_name("saved.tar.gz")
    archive.rename(target)
    archive.symlink_to(target)
    with pytest.raises(FileExistsError, match="符号链接"):
        tc.remap_maps()
    assert target.is_file()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"stage": 7}, ValueError),
        ({"stage": 0}, ValueError),
        ({"stage": True}, ValueError),
        ({"redundancy_info": Path("/nonexistent/r.bin")}, FileNotFoundError),
    ],
)
def test_map_options_validates_fields(kwargs, error):
    with pytest.raises(error):
        MapOptions(**kwargs)


def test_map_l2_id_parses_dirs_and_archives():
    assert map_l2_id(Path("Die_0_L2_7")) == 7
    assert map_l2_id(Path("die_3_l2_12.tar.gz")) == 12
    with pytest.raises(ToolchainError, match="解析 L2"):
        map_l2_id(Path("Die_0"))


def test_toolchain_command_failure_surfaces_stderr(toolchain):
    import subprocess

    tc, run = toolchain
    run.side_effect = subprocess.CalledProcessError(1, "x", stderr="compile error")
    with pytest.raises(ToolchainError, match="compile error"):
        tc.generate_node_config()


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_options", [False, True])
def test_compile_model_chains_toolchain_and_returns_jobs(toolchain, with_options):
    from types import SimpleNamespace

    from cimalgo.pipeline import compile_model, program_model

    tc, run = toolchain
    config = SimpleNamespace(
        compiler_tools=SimpleNamespace(install_dir=tc.compiler_tools),
        map_tools=SimpleNamespace(install_dir=tc.map_tools),
    )

    def fake_run(command, **kwargs):
        if command[1].endswith("gen_model_cfg.py"):
            (tc.model_path / "model_cfg.json").write_text("{}")
        if command[0].endswith("bw01d_compiler"):
            map = tc.compile_path / "output/Die_0_L2_2"
            map.mkdir(parents=True, exist_ok=True)
            (map / "f").write_text("x")

    run.side_effect = fake_run
    options = {2: MapOptions(stage=3)} if with_options else None
    compiled = compile_model(config, tc.model_path, tc.compile_path, oc_comp=False, map_options=options)
    archive = tc.compile_path / "Die_0_L2_2.tar.gz"
    assert compiled.jobs == ((2, archive),)
    assert compiled.archives == (archive,)
    assert compiled.l2_ids == (2,)
    assert compiled.model_dir == tc.model_path
    mapping = run.call_args.args[0]
    assert mapping[mapping.index("-l2") + 1] == "2"
    assert ("-stg" in mapping) is with_options
    device = Mock()
    program_model(device, compiled, tc.compile_path / "program")
    assert device.program.call_args.args[0] == compiled.jobs


def test_compile_model_fails_without_maps(toolchain):
    from types import SimpleNamespace

    from cimalgo.pipeline import compile_model

    tc, run = toolchain
    config = SimpleNamespace(
        compiler_tools=SimpleNamespace(install_dir=tc.compiler_tools),
        map_tools=SimpleNamespace(install_dir=tc.map_tools),
    )

    def fake_run(command, **kwargs):
        if command[1].endswith("gen_model_cfg.py"):
            (tc.model_path / "model_cfg.json").write_text("{}")
        if command[0].endswith("bw01d_compiler"):
            (tc.compile_path / "output").mkdir(parents=True, exist_ok=True)

    run.side_effect = fake_run
    with pytest.raises(ToolchainError, match="没有生成 Map"):
        compile_model(config, tc.model_path, tc.compile_path)


def test_compile_model_rejects_l2_mismatch_between_layer_and_map(toolchain):
    from types import SimpleNamespace

    from cimalgo.pipeline import compile_model

    tc, run = toolchain
    config = SimpleNamespace(
        compiler_tools=SimpleNamespace(install_dir=tc.compiler_tools),
        map_tools=SimpleNamespace(install_dir=tc.map_tools),
    )

    def fake_run(command, **kwargs):
        if command[1].endswith("gen_model_cfg.py"):
            (tc.model_path / "model_cfg.json").write_text("{}")
        if command[0].endswith("bw01d_compiler"):
            (tc.compile_path / "output/Die_0_L2_3").mkdir(parents=True)

    run.side_effect = fake_run
    with pytest.raises(ValueError, match=r"L2 不一致: 编译产物缺少 \[2\], 多出 \[3\]"):
        compile_model(config, tc.model_path, tc.compile_path)


def test_program_model_uses_compiled_jobs_by_default(tmp_path):
    from cimalgo.pipeline import CompiledModel, program_model

    device = Mock()
    device.program.return_value = {0: "ok"}
    compiled = CompiledModel(model_dir=tmp_path, compile_dir=tmp_path, jobs=((0, tmp_path / "a.tar.gz"),))

    assert program_model(device, compiled, tmp_path / "out", k=2, d=3) == {0: "ok"}
    device.program.assert_called_once_with(((0, tmp_path / "a.tar.gz"),), 2, 3, tmp_path / "out")

    with pytest.raises(TypeError):
        program_model(device, compiled, tmp_path / "out", jobs=[(5, "custom.tar.gz")])


def test_compiled_model_from_archives_reads_l2_from_names(tmp_path):
    from cimalgo.pipeline import CompiledModel

    archives = (tmp_path / "Die_0_L2_10.tar.gz", tmp_path / "Die_0_L2_2.tar.gz")
    compiled = CompiledModel.from_archives(tmp_path, tmp_path, archives)
    assert compiled.jobs == ((10, archives[0]), (2, archives[1]))
    assert compiled.l2_ids == (10, 2)


@pytest.mark.parametrize(
    ("jobs", "message"),
    [
        ((), "不能为空"),
        (((0, "a"), (0, "b")), "不得重复"),
        (((-1, "a"),), "非负整数"),
        (((True, "a"),), "非负整数"),
    ],
)
def test_compiled_model_validates_jobs(tmp_path, jobs, message):
    from cimalgo.pipeline import CompiledModel

    with pytest.raises(ValueError, match=message):
        CompiledModel(tmp_path, tmp_path, tuple((l2, tmp_path / name) for l2, name in jobs))


def test_analyze_results_applies_thresholds_per_node(tmp_path, monkeypatch):
    from cimalgo.drivers import NodeResult
    from cimalgo.pipeline import Threshold, analyze_results
    from cimalgo.pipeline import steps as steps_module

    monkeypatch.setattr(steps_module, "draw_basic_metrics", lambda g, c, p: {"cs_overall": float(c[0, 0]) / 100})
    golden = np.zeros((1, 1))
    nodes = [
        NodeResult("good", 0, 0, 0, np.array([[99.0]]), golden, tmp_path / "good.npy"),
        NodeResult("bad", 0, 0, 1, np.array([[50.0]]), golden, tmp_path / "bad.npy"),
    ]
    thresholds = (Threshold("cs_overall", ">=", 0.9), Threshold("missing", "<=", 1))

    report = analyze_results(nodes, tmp_path / "plots", thresholds=thresholds)

    assert not report.passed
    assert report.nodes[0].metrics == {"cs_overall": 0.99}
    assert report.nodes[0].plot_path == tmp_path / "plots" / "good.png"
    assert [f for f in report.failures if f.startswith("good:")] == ["good: 未知指标 missing; 可用: cs_overall"]
    assert any("bad: cs_overall=0.5000 不满足 cs_overall >= 0.9" in f for f in report.failures)
    assert report.flat_metrics() == {"good.cs_overall": 0.99, "bad.cs_overall": 0.5}


def test_analyze_results_without_thresholds_always_passes(tmp_path, monkeypatch):
    from cimalgo.drivers import NodeResult
    from cimalgo.pipeline import analyze_results
    from cimalgo.pipeline import steps as steps_module

    monkeypatch.setattr(steps_module, "draw_basic_metrics", lambda g, c, p: {"x": 0.0})
    node = NodeResult("n", 0, 0, 0, np.zeros((1, 1)), np.zeros((1, 1)), tmp_path / "n.npy")
    assert analyze_results([node], tmp_path).passed
