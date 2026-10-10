"""BW01D 工具链: 节点配置生成、编译、物理映射与打包"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from cimalgo.config import ToolchainConfig
from cimalgo.env.archive import compress
from cimalgo.logging import get_logger

log = get_logger(__name__)

_MAP_NAME = re.compile(r"Die_\d+_L2_(\d+)(?:\.tar\.gz)?", flags=re.IGNORECASE)


class ToolchainError(RuntimeError):
    """编译或映射失败"""


@dataclass(frozen=True, slots=True)
class MapOptions:
    """
    remap 时的参数; 每个 L2 各不相同

    目标 L2 编号固定等于 Map 名称里的源 L2 编号, 不提供覆盖: 逻辑 L2 和物理 L2 永远同号,
    模型 layer 名字里的 L2/L0/node 就是硬件上真正烧录/推理的地址, 不需要额外的映射关系追踪。

    redundancy_info: map_tool -r 的冗余信息文件
    stage: map_tool -stg, 1 到 6
    """

    redundancy_info: Path | None = None
    stage: int | None = None

    def __post_init__(self) -> None:
        if self.stage is not None and (isinstance(self.stage, bool) or self.stage not in range(1, 7)):
            raise ValueError("map stage 必须为 1 到 6")
        if self.redundancy_info is not None:
            resolved = Path(self.redundancy_info).expanduser().resolve()
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            object.__setattr__(self, "redundancy_info", resolved)


def map_l2_id(path: Path) -> int:
    """从 Die_<n>_L2_<id> 形式的目录或压缩包名解析 L2 编号"""
    match = _MAP_NAME.fullmatch(path.name)
    if match is None:
        raise ToolchainError(f"无法从 Map 名称解析 L2 编号: {path}")
    return int(match[1])


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)


class Bw01dToolchain:
    """用 compiler_tools 和 map_tools 把节点目录变成可烧录的 Map 压缩包"""

    GENERATED_FILES = frozenset({"node_cfg.json", "model_cfg.json", ".update"})

    def __init__(
        self, compiler_tools: str | Path, map_tools: str | Path, model_path: str | Path, compile_path: str | Path
    ):
        self.compiler_tools = Path(compiler_tools).expanduser().resolve()
        self.map_tools = Path(map_tools).expanduser().resolve()
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_dir():
            raise NotADirectoryError(self.model_path)
        self.compile_path = Path(compile_path).expanduser().resolve()

    @classmethod
    def from_config(
        cls, toolchain: ToolchainConfig, model_path: str | Path, compile_path: str | Path
    ) -> Bw01dToolchain:
        return cls(toolchain.compiler_tools.install_dir, toolchain.map_tools.install_dir, model_path, compile_path)

    def _run(self, command: list[str], *, cwd: Path, timeout: float) -> None:
        log.debug("运行: %s", " ".join(command))
        try:
            subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=True)
        except (OSError, subprocess.SubprocessError) as error:
            detail = getattr(error, "stderr", None) or str(error)
            raise ToolchainError(f"工具链命令执行失败: {command[0]}: {detail}") from error

    def _script(self, name: str) -> Path:
        script = self.compiler_tools / "script" / name
        if not script.is_file():
            raise FileNotFoundError(script)
        return script

    # ---- 节点配置 ----

    def clear_generated_configs(self) -> list[Path]:
        removed = []
        for path in sorted(self.model_path.rglob("*")):
            if path.is_file() and not path.is_symlink() and path.name in self.GENERATED_FILES:
                path.unlink()
                removed.append(path)
        return removed

    @log.trace()
    def generate_node_config(self, *, timeout: float = 600, reset_generated: bool = True) -> Path:
        if reset_generated:
            self.clear_generated_configs()
        self._run(
            [sys.executable, str(self._script("gen_node_cfg.py")), str(self.model_path)],
            cwd=self.compiler_tools,
            timeout=timeout,
        )
        return self.model_path

    @staticmethod
    def update_node_config(cfg_file: str | Path, **updates: Any) -> Path:
        path = Path(cfg_file)
        if path.is_dir():
            path = path / "node_cfg.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        content = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(content, dict):
            raise ToolchainError(f"node_cfg.json 必须是对象: {path}")
        merged = deepcopy(content)
        _deep_merge(merged, updates)
        path.write_text(json.dumps(merged, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
        return path

    def update_node_configs(self, **updates: Any) -> list[Path]:
        updated = [
            self.update_node_config(node, **updates)
            for node in sorted(self.model_path.iterdir())
            if node.is_dir() and not node.is_symlink()
        ]
        if not updated:
            raise ToolchainError(f"没有生成 node_cfg.json: {self.model_path}")
        return updated

    # ---- 编译 ----

    def map_directories(self) -> list[Path]:
        """返回编译器生成的 Map 目录"""
        output = self.compile_path / "output"
        if not output.is_dir():
            raise ToolchainError(f"编译器没有生成 output 目录: {output}")
        maps = sorted(output.iterdir())
        if not maps:
            raise ToolchainError(f"没有生成 Map 目录: {output}")
        for map in maps:
            if not map.is_dir():
                raise NotADirectoryError(map)
        return maps

    @log.trace()
    def compile_model(self, *, oc_comp: bool = False, timeout: float = 600) -> Path:
        """生成 model_cfg.json 并编译; 可选逐 Map 后补偿, 返回编译产物根目录"""
        self._run(
            [sys.executable, str(self._script("gen_model_cfg.py")), str(self.model_path)],
            cwd=self.compiler_tools,
            timeout=timeout,
        )
        model_config = self.model_path / "model_cfg.json"
        if not model_config.is_file():
            raise ToolchainError(f"没有生成 model_cfg.json: {model_config}")

        compiler = self.compiler_tools / "bin" / "bw01d_compiler"
        self._run(
            [str(compiler), "-f", str(model_config), "-j", "-o", str(self.compile_path)],
            cwd=self.compiler_tools,
            timeout=timeout,
        )
        maps = self.map_directories()

        if oc_comp:
            binaries = self.compiler_tools / "bin"
            for map in maps:
                self._run([str(binaries / "gen_tempcomp"), str(map)], cwd=self.compiler_tools, timeout=timeout)
                self._run(
                    [
                        str(binaries / "PostCompBinMaker"),
                        "-i",
                        str(map / "config.bin"),
                        "-o",
                        str(map / "oc.bin"),
                        "-m",
                        str(model_config),
                    ],
                    cwd=self.compiler_tools,
                    timeout=timeout,
                )
                self._run(
                    [
                        sys.executable,
                        str(self._script("append_oc_to_calib.py")),
                        "-oc",
                        str(map / "oc.bin"),
                        "-calib",
                        str(map / "calib.bin"),
                    ],
                    cwd=self.compiler_tools,
                    timeout=timeout,
                )
        return self.compile_path

    # ---- 映射 ----

    @log.trace()
    def remap_maps(
        self,
        options: Mapping[int, MapOptions] | None = None,
        *,
        timeout: float = 600,
    ) -> list[tuple[int, Path]]:
        """
        逐 Map 映射并打包, 返回 [(L2 编号, 压缩包), ...], 按 Map 名称排序

        每个 Map 的 L2 编号来自目录名 Die_<n>_L2_<id>, 映射前后不变 (逻辑 L2 和物理 L2 永远同号,
        不提供覆盖); 映射参数 (stage、冗余信息) 按这个 L2 编号从 options 里取:

        - None: 全部用默认参数
        - {l2_id: MapOptions}: 按 L2 编号各取各的, 没出现的用默认参数; 键必须是已存在的 L2 编号
        """
        maps = self.map_directories()
        source_ids = [map_l2_id(map) for map in maps]
        if len(set(source_ids)) != len(source_ids):
            raise ToolchainError(f"多个 Map 的 L2 编号重复: {source_ids}")

        if options is None:
            per_map = {l2: MapOptions() for l2 in source_ids}
        else:
            if not isinstance(options, Mapping) or any(
                isinstance(l2, bool) or not isinstance(l2, int) or not isinstance(option, MapOptions)
                for l2, option in options.items()
            ):
                raise TypeError("options 必须是 Mapping[int, MapOptions] 或 None")

            # 集合差集运算: options 中存在, 但 source_ids 中不存在的元素
            unknown = sorted(set(options) - set(source_ids))
            if unknown:
                raise ToolchainError(f"options 中的 L2 编号 {unknown} 没有对应的 Map; 现有: {sorted(source_ids)}")
            per_map = {l2: options.get(l2, MapOptions()) for l2 in source_ids}

        archives: list[tuple[int, Path]] = []
        for map, l2 in zip(maps, source_ids, strict=True):
            option = per_map[l2]
            archive = self.compile_path / f"{map.name}.tar.gz"
            if archive.is_symlink():
                raise FileExistsError(f"Map 压缩包路径是符号链接: {archive}")
            command = [str(self.map_tools / "map_tool"), "-i", str(map), "-l2", str(l2)]
            if option.redundancy_info is not None:
                command += ["-r", str(option.redundancy_info)]
            if option.stage is not None:
                command += ["-stg", str(option.stage)]
            self._run(command, cwd=self.map_tools, timeout=timeout)
            with TemporaryDirectory(prefix=f".{map.name}-", dir=self.compile_path) as tmp:
                temporary = Path(tmp) / archive.name
                compress(temporary, map, include_root=False)
                temporary.replace(archive)
            archives.append((l2, archive))
        return archives
