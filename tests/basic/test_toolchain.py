import json
import sys
import tarfile
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from utils.toolchain import Bw01dToolchain


def test_init_uses_explicit_tool_model_and_compile_paths(tmp_path):
    compiler = tmp_path / "compiler"
    mapper = tmp_path / "mapper"
    model_path = tmp_path / "nodes"
    model_path.mkdir()
    compile_path = tmp_path / "compiled" / "Die_0_L2_0"

    toolchain = Bw01dToolchain(compiler, mapper, model_path, compile_path)

    assert toolchain.compiler_tools == compiler
    assert toolchain.map_tools == mapper
    assert toolchain.model_path == model_path
    assert toolchain.compile_path == compile_path


def test_init_rejects_missing_model_directory(tmp_path):
    model_path = tmp_path / "missing_nodes"

    with pytest.raises(NotADirectoryError) as error:
        Bw01dToolchain(tmp_path, tmp_path, model_path, tmp_path / "compiled" / "Die_0_L2_0")

    assert error.value.args == (model_path,)


def test_update_node_configs_deep_merges_only_requested_values(tmp_path):
    node = tmp_path / "L2_0_L0_0_node_0"
    node.mkdir()
    config_path = node / "node_cfg.json"
    config_path.write_text(
        json.dumps({"optimization": {"keep": 7, "tia_mode": 9}, "data": {"input": "old"}}),
        encoding="utf-8",
    )
    updates = {"optimization": {"tia_mode": 1}, "data": {"g": "G.npy"}}

    updated = Bw01dToolchain(tmp_path, tmp_path, tmp_path, tmp_path / "compiled" / "Die_0_L2_0").update_node_configs(
        **updates
    )

    assert updated == [config_path]
    content = json.loads(config_path.read_text(encoding="utf-8"))
    assert content["optimization"] == {"keep": 7, "tia_mode": 1}
    assert "burn" not in content
    assert content["data"] == {"input": "old", "g": "G.npy"}
    assert updates == {"optimization": {"tia_mode": 1}, "data": {"g": "G.npy"}}


@pytest.mark.parametrize("use_directory", [False, True])
def test_update_node_config_updates_one_file(tmp_path, use_directory):
    config_path = tmp_path / "node_cfg.json"
    config_path.write_text(json.dumps({"data": {"input": "old"}}), encoding="utf-8")

    target = tmp_path if use_directory else config_path
    result = Bw01dToolchain.update_node_config(target, data={"g": "G.npy"})

    assert result == config_path
    assert json.loads(config_path.read_text(encoding="utf-8")) == {"data": {"input": "old", "g": "G.npy"}}


def test_update_node_configs_requires_config_in_each_direct_node_directory(tmp_path):
    (tmp_path / "node").mkdir()

    with pytest.raises(FileNotFoundError, match="node_cfg.json"):
        Bw01dToolchain(tmp_path, tmp_path, tmp_path, tmp_path / "compiled" / "Die_0_L2_0").update_node_configs(
            burn={"cursur_en": [255, 255]}
        )


def test_clear_generated_configs_keeps_other_json_files(tmp_path):
    node = tmp_path / "node"
    node.mkdir()
    generated = [node / "node_cfg.json", tmp_path / "model_cfg.json", node / ".update"]
    for path in generated:
        path.write_text("old", encoding="utf-8")
    keep = node / "data.json"
    keep.write_text("keep", encoding="utf-8")
    other_update = node / "state.update"
    other_update.write_text("keep", encoding="utf-8")

    removed = Bw01dToolchain(
        tmp_path, tmp_path, tmp_path, tmp_path / "compiled" / "Die_0_L2_0"
    ).clear_generated_configs()

    assert removed == sorted(generated)
    assert all(not path.exists() for path in generated)
    assert keep.read_text(encoding="utf-8") == "keep"
    assert other_update.read_text(encoding="utf-8") == "keep"


def test_remap_die_overwrites_regular_archive_and_accepts_mapping_options(tmp_path, monkeypatch):
    compiler = tmp_path / "compiler"
    mapper = tmp_path / "mapper"
    compile_path = tmp_path / "compiled"
    dies = [compile_path / "output" / name for name in ("Die_0_L2_0", "Die_1_L2_0")]
    for die in dies:
        die.mkdir(parents=True)
        (die / "config.bin").write_bytes(die.name.encode())
    old_archive = compile_path / "Die_0_L2_0.tar.gz"
    old_archive.write_bytes(b"old archive")
    mapper.mkdir()
    redundancy = tmp_path / "redundancy.json"
    redundancy.write_text("{}", encoding="utf-8")
    commands = []
    monkeypatch.setattr("subprocess.run", lambda command, **kwargs: commands.append(command))

    result = Bw01dToolchain(compiler, mapper, tmp_path, compile_path).remap_die(
        l2_id=1, redundancy_info=redundancy, stage=4
    )

    assert result is None
    assert sorted(commands, key=lambda command: command[2]) == [
        [str(mapper / "map_tool"), "-i", str(die), "-l2", "1", "-r", str(redundancy), "-stg", "4"] for die in dies
    ]
    for die in dies:
        archive_path = compile_path / f"{die.name}.tar.gz"
        with tarfile.open(archive_path, "r:gz") as archive:
            assert archive.getnames() == ["config.bin"]
            assert archive.extractfile("config.bin").read() == die.name.encode()


@pytest.mark.parametrize("target_exists", [False, True])
def test_remap_die_rejects_symlink_archive(tmp_path, monkeypatch, target_exists):
    compile_path = tmp_path / "compiled"
    die = compile_path / "output" / "Die_0_L2_0"
    die.mkdir(parents=True)
    target = tmp_path / "outside.tar.gz"
    if target_exists:
        target.write_bytes(b"do not overwrite")
    archive = compile_path / "Die_0_L2_0.tar.gz"
    archive.symlink_to(target)
    commands = []
    toolchain = Bw01dToolchain(tmp_path, tmp_path, tmp_path, compile_path)
    monkeypatch.setattr(toolchain, "_run", lambda *args, **kwargs: commands.append(args))

    with pytest.raises(FileExistsError, match="符号链接"):
        toolchain.remap_die()

    assert archive.is_symlink()
    assert commands == []
    if target_exists:
        assert target.read_bytes() == b"do not overwrite"
    else:
        assert not target.exists()


def test_remap_die_keeps_old_archive_if_compression_fails(tmp_path, monkeypatch):
    compile_path = tmp_path / "compiled"
    die = compile_path / "output" / "Die_0_L2_0"
    die.mkdir(parents=True)
    (die / "config.bin").write_bytes(b"new")
    archive = compile_path / "Die_0_L2_0.tar.gz"
    archive.write_bytes(b"old archive")
    toolchain = Bw01dToolchain(tmp_path, tmp_path, tmp_path, compile_path)
    monkeypatch.setattr(toolchain, "_run", lambda *args, **kwargs: None)

    def fail_compress(*args, **kwargs):
        raise RuntimeError("compression failed")

    monkeypatch.setattr("utils.toolchain.compress", fail_compress)

    with pytest.raises(RuntimeError, match="compression failed"):
        toolchain.remap_die()

    assert archive.read_bytes() == b"old archive"
    assert not list(compile_path.glob(".Die_0_L2_0-*"))


@pytest.mark.parametrize("node_config_updates", [None, {}, {"data": {"g": "G.npy"}}])
@pytest.mark.parametrize("compile_inside_tool", [False, True])
def test_staged_node_compile_map_and_package_flow(tmp_path, monkeypatch, node_config_updates, compile_inside_tool):
    compiler = tmp_path / "compiler"
    mapper = tmp_path / "mapper"
    node_root = tmp_path / "nodes"
    node = node_root / "L2_2_L0_0_node_0"
    node.mkdir(parents=True)
    mapper.mkdir()
    (compiler / "script").mkdir(parents=True)
    for script in ("gen_node_cfg.py", "gen_model_cfg.py"):
        (compiler / "script" / script).write_text("", encoding="utf-8")
    (node / "node_cfg.json").write_text("stale", encoding="utf-8")
    (node / ".update").write_text("stale", encoding="utf-8")
    keep = node / "data.json"
    keep.write_text("keep", encoding="utf-8")
    commands = []
    executed_commands = []
    run_options = []
    compile_root = compiler / "output" if compile_inside_tool else tmp_path / "compiled"
    compile_path = compile_root / "Die_0_L2_2"
    die = compile_path
    mapped_die = compile_path / "output" / "Die_0_L2_2"

    def fake_run(command, **kwargs):
        name = Path(command[1] if command[0] == sys.executable else command[0]).name
        commands.append(name)
        executed_commands.append(command)
        run_options.append(kwargs)
        if name == "gen_node_cfg.py":
            (node / "node_cfg.json").write_text(json.dumps({"data": {"input": "old"}}), encoding="utf-8")
        elif name == "gen_model_cfg.py":
            (node_root / "model_cfg.json").write_text("{}", encoding="utf-8")
        elif name == "bw01d_compiler":
            die.mkdir(parents=True)
            (die / "config.bin").write_bytes(b"config")
            (die / "calib.bin").write_bytes(b"calib")
            mapped_die.mkdir(parents=True)
            (mapped_die / "config.bin").write_bytes(b"mapped config")
        elif name == "PostCompBinMaker":
            (die / "oc.bin").write_bytes(b"oc")
        elif name == "map_tool":
            (mapped_die / "map.bin").write_bytes(b"mapped")
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("subprocess.run", fake_run)
    toolchain = Bw01dToolchain(compiler, mapper, node_root, compile_path)

    toolchain.generate_node_config()
    if node_config_updates:
        toolchain.update_node_configs(**node_config_updates)
    compiled_die = toolchain.compile_model(oc_comp=True)
    toolchain.remap_die(l2_id=2)
    archive_path = compile_path / "Die_0_L2_2.tar.gz"

    assert commands == [
        "gen_node_cfg.py",
        "gen_model_cfg.py",
        "bw01d_compiler",
        "gen_tempcomp",
        "PostCompBinMaker",
        "append_oc_to_calib.py",
        "map_tool",
    ]
    compiler_command = executed_commands[commands.index("bw01d_compiler")]
    assert compiler_command[-2:] == ["-o", str(compile_path)]
    assert compiled_die == die
    assert all("env" not in options for options in run_options)
    node_config = json.loads((node / "node_cfg.json").read_text(encoding="utf-8"))
    expected_data = {"input": "old"}
    if node_config_updates:
        expected_data["g"] = "G.npy"
    assert node_config["data"] == expected_data
    assert "optimization" not in node_config
    assert "burn" not in node_config
    assert not (node / ".update").exists()
    assert keep.read_text(encoding="utf-8") == "keep"
    assert (mapped_die / "map.bin").read_bytes() == b"mapped"
    assert (die / "oc.bin").is_file()
    with tarfile.open(archive_path, "r:gz") as archive:
        assert "map.bin" in archive.getnames()
        assert "config.bin" in archive.getnames()

    toolchain.remap_die(l2_id=2)
    assert commands.count("map_tool") == 2
    with tarfile.open(archive_path, "r:gz") as archive:
        assert archive.extractfile("map.bin").read() == b"mapped"
