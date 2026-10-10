"""测试项目使用的设备操作契约。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, ClassVar

from tqdm import tqdm


class DeviceError(RuntimeError):
    """设备操作失败"""


GRPC_PORT_BY_SITE = {
    60000: 0,
    60010: 1,
}


@dataclass(frozen=True)
class DeviceTarget:
    """连接和控制一台设备所需的显式配置"""

    ip: str
    bmc_ip: str  # bw01 电源控制地址
    grpc_port: int = 60000  # grpc 连接端口
    evb_port: int = 61006
    server_port: int = 60001

    is_cluster: bool = False
    test_stage: str = "CP3"
    driver: str = "bw01d"

    @property
    def evb_address(self) -> str:
        return f"{self.ip}:{self.evb_port}"

    @property
    def server_address(self) -> str:
        return f"{self.ip}:{self.server_port}"

    @property
    def site(self) -> int:
        return GRPC_PORT_BY_SITE[self.grpc_port]


class DeviceOps(ABC):
    """所有 SDK 版本向测试提供的共同原子操作契约"""

    _implementations: ClassVar[dict[str, type[DeviceOps]]] = {}

    def __init_subclass__(cls, *, driver: str | None = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if driver is None:
            return

        key = driver.casefold()
        if not key or key in cls._implementations:
            raise TypeError(f"设备驱动注册无效或重复: {driver!r}")
        cls._implementations[key] = cls

    def __init__(self, target: DeviceTarget):
        self.target = target

    @classmethod
    def for_target(cls, target: DeviceTarget, **kwargs) -> DeviceOps:
        """根据 target.driver 创建已注册的设备操作实现"""
        implementation = cls._implementations.get(target.driver.casefold())
        if implementation is None:
            supported = ", ".join(sorted(cls._implementations)) or "无"
            raise DeviceError(f"设备驱动 {target.driver!r} 没有实现; 已支持: {supported}")
        return implementation(target, **kwargs)

    @abstractmethod
    def init_connection(self) -> Any:
        """初始化并返回 gRPC 通道；重复调用复用同一通道"""

    @abstractmethod
    def select_layer(
        self,
        hb_id: int,
        tsv_sel0: int = 0,
        tsv_sel1: int = 255,
        *,
        timeout: float | None = None,
        cim_trim0: bytes = b"",
        cim_trim1: bytes = b"",
    ) -> Any:
        """选择 HB 层，并可传入两个 N45 的 trim 数据"""

    @abstractmethod
    def set_phy_l2(self, ids: Sequence[int], *, timeout: float | None = None) -> Any:
        """配置可用的物理 L2"""

    @abstractmethod
    def upload_program(self, l2_id: int, l2_tar: str, *, chip_id: int = 0, max_wait_seconds: int = 60) -> Any:
        """上传 L2 程序包"""

    @abstractmethod
    def run_program(self, l2_id: int, k: int, d: int, *, max_wait_seconds: int = 7200) -> Any:
        """执行 L2 烧录"""

    @abstractmethod
    def get_program_result(self, l2_id: int, output_dir: str = "./output", *, max_wait_seconds: int = 30) -> Any:
        """获取烧录或推理结果"""

    @abstractmethod
    def get_program_progress(self, l2_id: int, *, max_wait_seconds: int = 5) -> Any:
        """获取 L2 烧录进度"""

    @abstractmethod
    def map_program(
        self,
        l2_id: int,
        l2_tar: str,
        k: int,
        d: int,
        output_dir: str = "./output",
        *,
        max_wait_seconds: int = 7200,
        chip_id: int = 0,
    ) -> Any:
        """执行 SDK 的完整 L2 烧录流程"""

    def program(
        self,
        jobs: Sequence[tuple[int, str | Path]],
        k: int,
        d: int,
        program_dir: str | Path,
        *,
        max_wait_seconds: int = 7200,
        max_workers: int = 8,
        poll_interval: float = 2.0,
    ) -> dict[int, Any]:
        """并发执行完整 map 烧录、显示进度，返回按 L2 ID 索引的结果响应"""
        jobs = list(jobs)
        if not jobs:
            return {}
        if len({l2_id for l2_id, _ in jobs}) != len(jobs):
            raise DeviceError("烧录任务中的 L2 ID 不能重复")
        if max_workers < 1:
            raise ValueError("max_workers 必须大于 0")
        if poll_interval <= 0:
            raise ValueError("poll_interval 必须大于 0")

        self.init_connection()
        root = Path(program_dir)

        def worker(job: tuple[int, str | Path], running: Event) -> Any:
            l2_id, archive = job
            output_dir = root / f"L2_{l2_id}"
            output_dir.mkdir(parents=True, exist_ok=True)
            # 烧录任务执行时, 才主动查询进度, 否则 continue
            running.set()
            return self.map_program(
                l2_id,
                str(archive),
                k,
                d,
                output_dir=str(output_dir),
                max_wait_seconds=max_wait_seconds,
            )

        results: dict[int, Any] = {}
        failures: list[tuple[int, Exception]] = []
        with ExitStack() as stack, ThreadPoolExecutor(max_workers=max_workers) as pool:
            bars = [None] * len(jobs)

            def update_bar(index: int, l2_id: int, target: int) -> None:
                bar = bars[index]
                if bar is None:
                    if target == 0:
                        return
                    # 首次有实际进度时才创建，避免 tqdm 立即画出空的 0% 长条。
                    bars[index] = stack.enter_context(
                        tqdm(
                            total=100,
                            initial=target,
                            desc=f"L2_{l2_id}",
                            bar_format="{desc}: {percentage:.0f}%|{bar}| [{elapsed}]",
                            position=sum(other is not None for other in bars),
                            leave=True,
                            disable=None,
                        )
                    )
                else:
                    bar.update(max(0, target - bar.n))

            tasks = {}
            for index, job in enumerate(jobs):
                running = Event()
                future = pool.submit(worker, job, running)
                tasks[future] = (index, job[0], running)

            pending = set(tasks)
            progress_warnings: set[int] = set()
            while pending:
                completed, pending = wait(pending, timeout=poll_interval, return_when=FIRST_COMPLETED)
                for future in completed:
                    index, l2_id, _ = tasks[future]
                    try:
                        results[l2_id] = future.result()
                    except Exception as error:
                        failures.append((l2_id, error))
                    else:
                        update_bar(index, l2_id, 100)

                for future in pending:
                    if future.done():
                        continue
                    index, l2_id, running = tasks[future]
                    if not running.is_set():
                        continue
                    try:
                        response = self.get_program_progress(l2_id)
                        progress = int(response.content["data"]["progress"])
                    except Exception as error:
                        # 进度查询失败不应中断烧录，最终结果由 map_program 决定。
                        if l2_id not in progress_warnings:
                            tqdm.write(f"L2_{l2_id} 暂时无法读取烧录进度：{error}")
                            progress_warnings.add(l2_id)
                        continue
                    progress_warnings.discard(l2_id)
                    target = min(99, max(0, progress))
                    update_bar(index, l2_id, target)

        if failures:
            details = "; ".join(f"L2_{l2_id}: {error}" for l2_id, error in failures)
            raise DeviceError(f"烧录失败：{details}") from failures[0][1]
        return {l2_id: results[l2_id] for l2_id, _ in jobs}

    @abstractmethod
    def infer(self, l2_id: int, output_dir: str = "./output", *, max_wait_seconds: int = 90) -> Any:
        """执行 L2 推理并获取结果"""

    @abstractmethod
    def get_sdk_version(self, *, timeout: float | None = None) -> str:
        """查询设备 SDK 版本"""

    @abstractmethod
    def get_firmware_version(self, *, timeout: float | None = None) -> str:
        """查询当前 N45 固件版本"""

    @abstractmethod
    def switch_firmware_version(self, version: str, *, timeout: float | None = None) -> None:
        """切换 N45 固件版本"""

    @abstractmethod
    def power_cycle(self, *, off_seconds: float = 5) -> None:
        """关闭设备电源，等待指定时间后重新上电"""

    @abstractmethod
    def evb_power_set_all_enabled(self, *, timeout: float | None = None) -> Any:
        """打开全部 EVB 电源通道"""

    @abstractmethod
    def evb_power_set_all_disabled(self, *, timeout: float | None = None) -> Any:
        """关闭全部 EVB 电源通道"""

    @abstractmethod
    def close(self) -> None:
        """关闭底层连接并释放本实例持有的资源"""
