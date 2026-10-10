"""BW01D 节点配置生成、编译、物理映射与打包"""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .archive_tool import compress


class ToolchainError(RuntimeError):
    """编译或映射失败"""


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)


class Bw01dToolchain:
    """使用指定的 compiler_tools 和 map_tools 完成节点到 Die 产物的流程"""

    def __init__(
        self,
        compiler_tools: str | Path,
        map_tools: str | Path,
        model_path: str | Path,
        compile_path: str | Path,
    ):
        self.compiler_tools = Path(compiler_tools).expanduser().resolve()
        self.map_tools = Path(map_tools).expanduser().resolve()
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_dir():
            raise NotADirectoryError(self.model_path)
        self.compile_path = Path(compile_path).expanduser().resolve()

    def _run(self, command: list[str], *, cwd: Path, timeout: float) -> None:
        try:
            subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            detail = getattr(error, "stderr", None) or str(error)
            raise ToolchainError(f"工具链命令执行失败: {command[0]}: {detail}") from error

    def clear_generated_configs(self) -> list[Path]:
        """只清理节点树中的 node_cfg.json、model_cfg.json 和 .update 文件"""
        root = self.model_path
        removed = []
        for path in sorted(root.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            if path.name in {"node_cfg.json", "model_cfg.json", ".update"}:
                path.unlink()
                removed.append(path)
        return removed

    def generate_node_config(self, *, timeout: float = 600, reset_generated: bool = True) -> Path:
        """清理旧配置并运行 gen_node_cfg.py, 返回节点目录"""
        root = self.model_path
        if reset_generated:
            self.clear_generated_configs()
        script = self.compiler_tools / "script/gen_node_cfg.py"
        if not script.is_file():
            raise FileNotFoundError(script)
        self._run([sys.executable, str(script), str(root)], cwd=self.compiler_tools, timeout=timeout)
        return root

    @staticmethod
    def update_node_config(cfg_file: str | Path, **updates: Any) -> Path:
        """接受配置文件或节点目录, 递归更新其中的 node_cfg.json"""
        path = Path(cfg_file)
        if path.is_dir():
            path = path / "node_cfg.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        content = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(content, dict):
            raise ToolchainError(f"node_cfg.json 必须是对象: {path}")
        updated = deepcopy(content)
        _deep_merge(updated, updates)
        path.write_text(json.dumps(updated, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
        return path

    def update_node_configs(self, **updates: Any) -> list[Path]:
        """更新 model_path 直属节点目录内的 node_cfg.json"""
        root = self.model_path
        updated = []
        for node_path in sorted(root.iterdir()):
            if not node_path.is_dir() or node_path.is_symlink():
                continue
            updated.append(self.update_node_config(node_path, **updates))
        if not updated:
            raise ToolchainError(f"没有生成 node_cfg.json: {root}")
        return updated

    def compile_model(self, *, oc_comp: bool = False, timeout: float = 600) -> Path:
        """生成 model_cfg.json 并编译, 返回 compile_path 指向的 Die 目录"""
        node_root = self.model_path
        model_script = self.compiler_tools / "script/gen_model_cfg.py"
        if not model_script.is_file():
            raise FileNotFoundError(model_script)
        self._run([sys.executable, str(model_script), str(node_root)], cwd=self.compiler_tools, timeout=timeout)
        model_config = node_root / "model_cfg.json"
        if not model_config.is_file():
            raise ToolchainError(f"没有生成 model_cfg.json: {model_config}")
        self._run(
            [
                str(self.compiler_tools / "bin/bw01d_compiler"),
                "-f",
                str(model_config),
                "-j",
                "-o",
                str(self.compile_path),
            ],
            cwd=self.compiler_tools,
            timeout=timeout,
        )
        die = self.compile_path
        if not die.is_dir():
            raise ToolchainError(f"编译器没有生成 Die 目录: {die}")
        if oc_comp:
            self._run(
                [str(self.compiler_tools / "bin/gen_tempcomp"), str(die)], cwd=self.compiler_tools, timeout=timeout
            )
            self._run(
                [
                    str(self.compiler_tools / "bin/PostCompBinMaker"),
                    "-i",
                    str(die / "config.bin"),
                    "-o",
                    str(die / "oc.bin"),
                    "-m",
                    str(model_config),
                ],
                cwd=self.compiler_tools,
                timeout=timeout,
            )
            self._run(
                [
                    sys.executable,
                    str(self.compiler_tools / "script/append_oc_to_calib.py"),
                    "-oc",
                    str(die / "oc.bin"),
                    "-calib",
                    str(die / "calib.bin"),
                ],
                cwd=self.compiler_tools,
                timeout=timeout,
            )
        return die

    def remap_die(
        self,
        *,
        l2_id: int = 0,
        timeout: float = 600,
        redundancy_info: str | Path | None = None,
        stage: int | None = None,
    ) -> None:
        """映射 compile_path/output 下的 Die 目录，并覆盖 compile_path 中对应的 tar.gz"""
        dies_path = self.compile_path / "output"
        for die in dies_path.iterdir():
            if not die.is_dir():
                raise NotADirectoryError(die)
            archive = self.compile_path / f"{die.name}.tar.gz"
            if archive.is_symlink():
                raise FileExistsError(f"Die 压缩包路径是符号链接: {archive}")
            if stage is not None and stage not in range(1, 7):
                raise ValueError("map stage 必须为 1 到 6")
            command = [str(self.map_tools / "map_tool"), "-i", str(die), "-l2", str(l2_id)]
            if redundancy_info is not None:
                info = Path(redundancy_info).expanduser().resolve()
                if not info.is_file():
                    raise FileNotFoundError(info)
                command.extend(["-r", str(info)])
            if stage is not None:
                command.extend(["-stg", str(stage)])
            self._run(
                command,
                cwd=self.map_tools,
                timeout=timeout,
            )
            with TemporaryDirectory(prefix=f".{die.name}-", dir=self.compile_path) as temporary_dir:
                temporary_archive = Path(temporary_dir) / archive.name
                compress(temporary_archive, die, include_root=False)
                temporary_archive.replace(archive)
