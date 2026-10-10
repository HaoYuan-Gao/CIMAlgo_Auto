"""
测试 case 的 YAML 定义与校验

一个 case 文件描述: 用哪台设备、怎么初始化、模型从哪来 (假数据或外部目录)、推理参数、判定阈值。
加载后得到不可变的 CaseSpec; 执行由 pytest 插件里的 fixture 负责, 这里不碰设备。

示例::

    case: dense_128x256
    device: evb-01                 # cim_config.yaml 的 devices 里的名字
    init:
      power_cycle: true
      hard_blocks: [0, 1, 2, 3]
      select_layers: [0]
      tsv_sel0: 0
      tsv_sel1: 255
    model:
      kind: fake                   # fake | external
      seed: 7
      layers:
        - {name: model.layers.0.dense_L2_0_L0_0_node_0, batch: 32, in_features: 128, out_features: 256}
    infer:
      dummy: 2
      mode: {repeat: 1}
    thresholds:
      TODO: 可能需要和 analyzer 联动, 或者单独使用 承霖 的指标重新定义
      cs_overall: ">= 0.995"
      error_std_max: "<= 2.0"
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from cimalgo.config import validators as v
from cimalgo.drivers.base import InitState


class CaseDefinitionError(ValueError):
    """case 定义不完整或不一致"""


_COMPARATORS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
}


@dataclass(frozen=True, slots=True)
class Threshold:
    metric: str
    comparator: str
    limit: float

    def check(self, value: float) -> bool:
        return _COMPARATORS[self.comparator](value, self.limit)

    def __str__(self) -> str:
        return f"{self.metric} {self.comparator} {self.limit}"


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    """
    设备初始化参数; 通过 init_state() 转成可比较的芯片状态, 相同状态的相邻 case 复用初始化

    grpc_port 选择板上哪颗芯片 (site): 同一设备在不同用例间切换端口时, 设备池会重建 gRPC 通道并重新初始化。
    它不属于 InitState, 因为 InitState 描述的是被选中那颗芯片的状态。
    """

    name: str
    grpc_port: int = 60000
    power_cycle: bool = True
    hard_blocks: tuple[int, ...] = ()
    select_layers: tuple[int, ...] = ()
    tsv_sel0: int = 0
    tsv_sel1: int = 255
    update_sdk: bool = False

    def init_state(self, firmware_version: str | None) -> InitState:
        return InitState(
            firmware_version=firmware_version,
            hard_blocks=self.hard_blocks,
            select_layers=self.select_layers,
            tsv_sel0=self.tsv_sel0,
            tsv_sel1=self.tsv_sel1,
        )


@dataclass(frozen=True, slots=True)
class ModelSpec:
    kind: str  # fake | external
    path: Path | None = None  # external: 模型目录; fake: 生成目录 (None 时由 fixture 放到 workdir)
    seed: int = 42
    layers: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class InferSpec:
    dummy: int = 0
    mode: Mapping[str, Any] = field(default_factory=dict)
    oc_comp: bool = False


@dataclass(frozen=True, slots=True)
class CaseSpec:
    name: str
    device: DeviceSpec
    model: ModelSpec
    infer: InferSpec = InferSpec()
    thresholds: tuple[Threshold, ...] = ()
    tags: tuple[str, ...] = ()
    source: Path | None = None  # 定义 case 的 yaml 文件路径

    @property
    def id(self) -> str:
        return self.name


def _wrap(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except v.ProjectConfigError as error:
        raise CaseDefinitionError(str(error)) from None


def _int_list(value: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise CaseDefinitionError(f"{field_name} must be a list of integers")
    return tuple(_wrap(v.integer, item, f"{field_name}[{index}]", minimum=0) for index, item in enumerate(value))


def parse_device(raw: Any) -> DeviceSpec:
    if isinstance(raw, str):
        return DeviceSpec(name=_wrap(v.string, raw, "device"))
    raw = _wrap(v.mapping, raw, "device")
    unexpected = sorted(
        set(raw)
        - {"name", "grpc_port", "power_cycle", "hard_blocks", "select_layers", "tsv_sel0", "tsv_sel1", "update_sdk"}
    )
    if unexpected:
        raise CaseDefinitionError(f"device.{unexpected[0]} is not a supported setting")
    return DeviceSpec(
        name=_wrap(v.string, raw.get("name"), "device.name"),
        grpc_port=_wrap(v.port, raw.get("grpc_port", 60000), "device.grpc_port"),
        power_cycle=_wrap(v.boolean, raw.get("power_cycle", True), "device.power_cycle"),
        hard_blocks=_int_list(raw.get("hard_blocks", []), "device.hard_blocks"),
        select_layers=_int_list(raw.get("select_layers", []), "device.select_layers"),
        tsv_sel0=_wrap(v.integer, raw.get("tsv_sel0", 0), "device.tsv_sel0", minimum=0, maximum=255),
        tsv_sel1=_wrap(v.integer, raw.get("tsv_sel1", 255), "device.tsv_sel1", minimum=0, maximum=255),
        update_sdk=_wrap(v.boolean, raw.get("update_sdk", False), "device.update_sdk"),
    )


def parse_model(raw: Any, *, base_dir: Path) -> ModelSpec:
    # TODO: 考虑 model、infer 等项目配置拒绝未知字段
    raw = _wrap(v.mapping, raw, "model")
    kind = _wrap(v.string, raw.get("kind"), "model.kind")
    if kind not in {"fake", "external"}:
        raise CaseDefinitionError("model.kind must be 'fake' or 'external'")
    path = None
    if raw.get("path") is not None:
        path = Path(_wrap(v.string, raw["path"], "model.path")).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        path = path.resolve()
    if kind == "external":
        if path is None:
            raise CaseDefinitionError("model.path is required when model.kind is 'external'")
        return ModelSpec(kind=kind, path=path)

    layers = raw.get("layers")
    if not isinstance(layers, list) or not layers:
        raise CaseDefinitionError("model.layers must be a non-empty list for fake models")
    for index, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            raise CaseDefinitionError(f"model.layers[{index}] must be a mapping")
    return ModelSpec(
        kind=kind,
        path=path,
        seed=_wrap(v.integer, raw.get("seed", 42), "model.seed", minimum=0),
        layers=tuple(dict(layer) for layer in layers),
    )


def _infer(raw: Any) -> InferSpec:
    if raw is None:
        return InferSpec()
    raw = _wrap(v.mapping, raw, "infer")
    mode = raw.get("mode", {})
    if not isinstance(mode, Mapping):
        raise CaseDefinitionError("infer.mode must be a mapping")
    return InferSpec(
        dummy=_wrap(v.integer, raw.get("dummy", 0), "infer.dummy", minimum=0),
        mode=dict(mode),
        oc_comp=_wrap(v.boolean, raw.get("oc_comp", False), "infer.oc_comp"),
    )


def _thresholds(raw: Any) -> tuple[Threshold, ...]:
    if raw is None:
        return ()
    raw = _wrap(v.mapping, raw, "thresholds")
    result = []
    for metric, rule in raw.items():
        text = _wrap(v.string, rule, f"thresholds.{metric}")
        parts = text.split()
        if len(parts) != 2 or parts[0] not in _COMPARATORS:
            raise CaseDefinitionError(f"thresholds.{metric} must look like '>= 0.99'; got {text!r}")
        try:
            limit = float(parts[1])
        except ValueError:
            raise CaseDefinitionError(f"thresholds.{metric} limit must be a number; got {parts[1]!r}") from None
        result.append(Threshold(metric=str(metric), comparator=parts[0], limit=limit))
    return tuple(result)


def parse_case(raw: Mapping[str, Any], *, base_dir: Path, source: Path | None = None) -> CaseSpec:
    raw = _wrap(v.mapping, raw, "case")
    unexpected = sorted(set(raw) - {"case", "device", "model", "infer", "thresholds", "tags"})
    if unexpected:
        raise CaseDefinitionError(f"{unexpected[0]} is not a supported case setting")
    tags = raw.get("tags", [])
    if not isinstance(tags, list):
        raise CaseDefinitionError("tags must be a list")
    return CaseSpec(
        name=_wrap(v.component, raw.get("case"), "case"),
        device=parse_device(raw.get("device")),
        model=parse_model(raw.get("model"), base_dir=base_dir),
        infer=_infer(raw.get("infer")),
        thresholds=_thresholds(raw.get("thresholds")),
        tags=tuple(_wrap(v.string, tag, f"tags[{index}]") for index, tag in enumerate(tags)),
        source=source,
    )


def load_case(path: str | Path) -> CaseSpec:
    path = Path(path).resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as error:
        raise CaseDefinitionError(f"cannot read case: {path}") from error
    except yaml.YAMLError as error:
        raise CaseDefinitionError(f"invalid YAML in {path}: {error}") from error
    try:
        # NOTE: base_dir: 以 YAML 文件所在目录为基准
        return parse_case(raw, base_dir=path.parent, source=path)
    except CaseDefinitionError as error:
        raise CaseDefinitionError(f"{path}: {error}") from None


def load_cases(directory: str | Path, *, pattern: str = "*.yaml") -> Iterator[CaseSpec]:
    """递归加载目录下所有 case 文件, 按路径排序"""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise CaseDefinitionError(f"case 目录不存在或不是目录: {directory}")
    paths = sorted(directory.rglob(pattern))
    if not paths:
        raise CaseDefinitionError(f"case 目录中没有匹配 {pattern} 的文件: {directory}")
    seen: dict[str, Path] = {}
    for path in paths:
        spec = load_case(path)
        if spec.name in seen:
            raise CaseDefinitionError(f"case 名称重复: {spec.name!r} 出现在 {seen[spec.name]} 和 {path}")
        seen[spec.name] = path
        # yield 将函数的返回值改变成 Iterator, 把生产者和消费者解耦, 可以节省内存、提前终止(剩下的 case 不会解析),
        # 把异常推迟到执行时触发
        yield spec
