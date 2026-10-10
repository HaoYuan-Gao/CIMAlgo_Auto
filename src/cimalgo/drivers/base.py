"""
设备操作契约

DeviceOps 只定义测试流程真正需要的操作 (初始化、烧录、推理、版本、电源)。
芯片 SDK 的其他 API 属于具体实现的内部细节, 不进入契约。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from types import TracebackType
from typing import Any, ClassVar, Self

from tqdm import tqdm

from cimalgo.config import DeviceConfig
from cimalgo.logging import get_logger

from .errors import DeviceError

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DeviceTarget:
    """连接一台设备所需的地址信息; 由 DeviceConfig、用例选择的 gRPC 端口和当前 SDK 的 chip 组合而成"""

    ip: str
    grpc_port: int = 60000
    server_port: int = 60001
    evb_port: int | None = None
    # TODO: bw01 下上电 ip
    bmc_ip: str | None = None
    is_cluster: bool = False
    test_stage: str = "CP3"
    driver: str = "bw01d"

    @classmethod
    def from_config(cls, device: DeviceConfig, *, driver: str, grpc_port: int = 60000) -> DeviceTarget:
        return cls(
            ip=device.ip,
            grpc_port=grpc_port,
            server_port=device.server_port,
            evb_port=device.evb_port,
            bmc_ip=device.bmc_ip,
            is_cluster=device.is_cluster,
            test_stage=device.test_stage,
            driver=driver,
        )

    @property
    def evb_address(self) -> str:
        # bw01d 下上电 ip
        if self.evb_port is None:
            raise DeviceError(f"设备 {self.ip} 未配置 evb_port, 无法通过 EVB 控制电源")
        return f"{self.ip}:{self.evb_port}"

    @property
    def server_address(self) -> str:
        return f"{self.ip}:{self.server_port}"


@dataclass(frozen=True, slots=True)
class InitState:
    """
    芯片初始化后应处于的状态; 两个 case 的 InitState 相等时, 第二个可以跳过初始化

    只包含描述"状态"的字段: 固件版本、物理 L2、选层。power_cycle 这类动作参数不在这里。
    """

    firmware_version: str | None
    hard_blocks: tuple[int, ...] = ()
    select_layers: tuple[int, ...] = ()
    tsv_sel0: int = 0
    tsv_sel1: int = 255

    def sort_key(self) -> tuple:
        return (self.firmware_version or "", self.hard_blocks, self.select_layers, self.tsv_sel0, self.tsv_sel1)


class DeviceOps(ABC):
    """
    所有芯片向测试提供的共同操作; 用 with 管理底层连接

    状态跟踪: known_state 记录芯片当前处于的 InitState。只有 initialize() 会把它设为确定值;
    任何直接改变芯片状态的调用 (power_cycle / switch_firmware_version / set_phy_l2 / select_layer)
    都会把它置回 None, 这样设备池不会因为上一个测试手动改过选层而错误地跳过初始化。
    子类实现带下划线的 _xxx 方法, 公开方法由基类提供并负责标脏。
    """

    _implementations: ClassVar[dict[str, type[DeviceOps]]] = {}

    def __init_subclass__(cls, *, driver: str | None = None, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if driver is None:
            return
        key = driver.casefold()
        if not key or key in cls._implementations:
            raise TypeError(f"设备驱动注册无效或重复: {driver!r}")
        cls._implementations[key] = cls

    def __init__(self, target: DeviceTarget):
        self.target = target
        self.known_state: InitState | None = None

    @classmethod
    def for_target(cls, target: DeviceTarget, **kwargs: Any) -> DeviceOps:
        implementation = cls._implementations.get(target.driver.casefold())
        if implementation is None:
            supported = ", ".join(sorted(cls._implementations)) or "无"
            raise DeviceError(f"设备驱动 {target.driver!r} 没有实现; 已支持: {supported}")
        return implementation(target, **kwargs)

    # ---- 连接 ----

    @abstractmethod
    def channel(self) -> Any:
        """返回 SDK 通道句柄, 首次调用时建立; 供同一芯片的其他组件 (如 InferSession) 复用"""

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    # ---- 子类实现的原语 ----

    @abstractmethod
    def _power_cycle(self, *, off_seconds: float) -> None: ...

    @abstractmethod
    def get_firmware_version(self) -> str: ...

    @abstractmethod
    def _switch_firmware_version(self, version: str) -> None: ...

    @abstractmethod
    def _set_phy_l2(self, ids: Sequence[int]) -> None: ...

    # TODO: 未来需要增加 trim 参数
    @abstractmethod
    def _select_layer(self, hb_id: int, *, tsv_sel0: int, tsv_sel1: int) -> None: ...


    @abstractmethod
    def _program_one(
        self, l2_id: int, archive: str | Path, k: int, d: int, output_dir: Path, *, max_wait_seconds: int
    ) -> Any:
        """烧录单个 L2, 阻塞到完成或失败并返回 SDK 响应"""

    @abstractmethod
    def _program_progress(self, l2_id: int) -> int:
        """查询单个 L2 当前烧录进度 (0-100); 暂时不可读时应抛异常, 由 program() 决定如何处理"""


    # ---- 公开的状态变更操作: 调用后芯片状态视为未知 ----

    def invalidate(self) -> None:
        """芯片状态不可信 (测试失败、部署 SDK、手动改过配置); 下次 initialize 不会被跳过"""
        self.known_state = None

    def use_grpc_port(self, grpc_port: int) -> bool:
        """
        切换到同一块板上的另一个 gRPC 端口 (site); 端口相同时什么都不做并返回 False

        切换会关闭当前 SDK 通道并把状态标为未知: 新端口对应另一颗芯片, 上一个状态对它没有意义。
        """
        if grpc_port == self.target.grpc_port:
            return False
        self.close()
        self.target = replace(self.target, grpc_port=grpc_port)
        self.invalidate()
        return True

    def power_cycle(self, *, off_seconds: float = 5) -> None:
        self.invalidate()
        self._power_cycle(off_seconds=off_seconds)

    def switch_firmware_version(self, version: str) -> None:
        self.invalidate()
        self._switch_firmware_version(version)

    def set_phy_l2(self, ids: Sequence[int]) -> None:
        self.invalidate()
        self._set_phy_l2(ids)

    def select_layer(self, hb_id: int, *, tsv_sel0: int = 0, tsv_sel1: int = 255) -> None:
        self.invalidate()
        self._select_layer(hb_id, tsv_sel0=tsv_sel0, tsv_sel1=tsv_sel1)

    def initialize(self, state: InitState, *, power_cycle: bool = True, force: bool = False) -> bool:
        """
        把芯片初始化到 state: 可选下电上电 -> 固件版本 -> 物理 L2 -> 逐个 HB 选层

        known_state 已等于 state 且 force=False 时跳过并返回 False; 否则执行并返回 True。
        power_cycle 是动作参数, 不属于 InitState: 它决定这次初始化前要不要断电, 不描述芯片最终状态。
        中途任何一步失败, known_state 保持 None。
        """
        if not force and self.known_state == state:
            return False
        self.known_state = None
        if power_cycle:
            self._power_cycle(off_seconds=5)
        if state.firmware_version is not None and self.get_firmware_version() != state.firmware_version:
            self._switch_firmware_version(state.firmware_version)
            actual = self.get_firmware_version()
            if actual != state.firmware_version:
                raise DeviceError(f"固件版本切换失败: 期望 {state.firmware_version}, 实际 {actual}")
        self._set_phy_l2(state.hard_blocks)
        for hb_id in state.select_layers:
            self._select_layer(hb_id, tsv_sel0=state.tsv_sel0, tsv_sel1=state.tsv_sel1)
        self.known_state = state
        return True

    # ---- 烧录 / 推理 ----

    def program(
        self,
        jobs: Sequence[tuple[int, str | Path]],
        k: int,
        d: int,
        output_dir: str | Path,
        *,
        max_wait_seconds: int = 7200,
        max_workers: int = 8,
        poll_interval: float = 2.0,
    ) -> dict[int, Any]:
        """
        并发烧录多个 L2, 带进度条; 任一失败时在全部结束后汇总抛出

        并发调度、进度条、失败汇总在这里统一处理; 子类只需实现 _program_one (烧录单个 L2)
        和 _program_progress (查询进度), 不需要每个驱动各自重复这套编排逻辑。
        """
        jobs = list(jobs)
        if not jobs:
            return {}
        if len({l2_id for l2_id, _ in jobs}) != len(jobs):
            raise DeviceError("烧录任务中的 L2 ID 不能重复")
        if max_workers < 1 or poll_interval <= 0:
            raise ValueError("max_workers 必须大于 0, poll_interval 必须大于 0")

        # 提前建好连接: worker 会在多个线程里并发调用 self.channel(), channel() 的惰性初始化
        # 没有加锁, 必须在开线程池之前在当前线程里把连接建好, 否则并发首次调用会竞态
        self.channel()
        root = Path(output_dir)

        def worker(l2_id: int, archive: str | Path, started: Event) -> Any:
            target = root / f"L2_{l2_id}"
            target.mkdir(parents=True, exist_ok=True)
            started.set()
            return self._program_one(l2_id, archive, k, d, target, max_wait_seconds=max_wait_seconds)

        results: dict[int, Any] = {}
        failures: list[tuple[int, Exception]] = []
        bars: dict[int, tqdm] = {}
        warned: set[int] = set()
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                tasks = {}
                for position, (l2_id, archive) in enumerate(jobs):
                    started = Event()
                    tasks[pool.submit(worker, l2_id, archive, started)] = (l2_id, started, position)

                pending = set(tasks)
                while pending:
                    done, pending = wait(pending, timeout=poll_interval, return_when=FIRST_COMPLETED)
                    for future in done:
                        l2_id, _, position = tasks[future]
                        try:
                            results[l2_id] = future.result()
                        except Exception as error:
                            failures.append((l2_id, error))
                        else:
                            self._update_bar(bars, l2_id, position, 100)
                    for future in pending:
                        l2_id, started, position = tasks[future]
                        if not started.is_set():
                            continue
                        try:
                            progress = self._program_progress(l2_id)
                        except Exception as error:
                            if l2_id not in warned:
                                log.warning("L2_%d 暂时无法读取烧录进度: %s", l2_id, error)
                                warned.add(l2_id)
                            continue
                        warned.discard(l2_id)
                        self._update_bar(bars, l2_id, position, min(99, max(0, progress)))
        finally:
            for bar in bars.values():
                bar.close()

        if failures:
            details = "; ".join(f"L2_{l2_id}: {error}" for l2_id, error in failures)
            raise DeviceError(f"烧录失败: {details}") from failures[0][1]
        return {l2_id: results[l2_id] for l2_id, _ in jobs}

    @staticmethod
    def _update_bar(bars: dict[int, tqdm], l2_id: int, position: int, target: int) -> None:
        bar = bars.get(l2_id)
        if bar is None:
            if target == 0:
                return
            bar = bars[l2_id] = tqdm(
                total=100,
                initial=target,
                desc=f"L2_{l2_id}",
                bar_format="{desc}: {percentage:.0f}%|{bar}| [{elapsed}]",
                position=position,
                leave=True,
                disable=None,
            )
        else:
            bar.update(max(0, target - bar.n))

    @abstractmethod
    def get_sdk_version(self) -> str: ...
