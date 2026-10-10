"""单元测试共用 fixture"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml


@pytest.fixture
def config_data() -> dict:
    return {
        "active_sdk": "bw01d_v2.3.3",
        "netrc": ".netrc",
        "download_dir": "third_party/downloads",
        "sdk_install_root": "third_party/SDK",
        "toolchain_install_root": "third_party/toolchain",
        "trusted_ips": ["10.27.17.0/24", "2001:db8::1"],
        "devices": {
            "evb-01": {"ip": "10.27.17.45", "driver": "bw01d", "evb_port": 61006},
            "evb-02": {
                "ip": "10.27.17.46",
                "driver": "bw01d",
                "ssh_port": 2222,
                "ssh_username": "tester",
                "bmc_ip": "10.0.0.9",
            },
        },
        "toolchains": {
            "v1.6.17.2": {
                "version": "1.6.17.2",
                "url": "https://git.witmem.ad/api/v4/projects/project/bw01/bw01/bw01d/bw01d_cim_toolchain",
                "auth": "bearer",
                "map_tools": {"prefix": "bw01d_cim_map_tools_ubuntu2404", "suffix": ""},
                "compiler_tools": {"prefix": "bw01d_cim_compiler_tools_ubuntu2404", "suffix": ""},
            },
        },
        "sdks": {
            "bw01d_v2.3.3": {
                "chip": "bw01d",
                "version": "2.3.3",
                "sdk_prefix": "BW01D_SDK",
                "url": "https://nexus.witmem.ad/repository/bw01d-beta",
                "auth": "basic",
                "toolchain": "v1.6.17.2",
            },
            "bw01_local": {
                "chip": "bw01",
                "version": "2.3.3",
                "sdk_prefix": "BW01_SDK",
                "url": None,
                "toolchain": "v1.6.17.2",
            },
        },
    }


@pytest.fixture
def config_file(tmp_path: Path, config_data: dict, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CIM_SDK", raising=False)

    def write(data: dict | None = None) -> Path:
        path = tmp_path / "cim_config.yaml"
        path.write_text(yaml.safe_dump(data if data is not None else config_data), encoding="utf-8")
        return path

    return write


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """向 sys.modules 注入假的 wtm SDK; 返回各子模块方便断言"""
    wtm = ModuleType("wtm")
    bw01d = ModuleType("wtm.bw01d")
    evb = ModuleType("wtm.bw01d.evb")
    hb_ctrl = ModuleType("wtm.bw01d.hb_ctrl")
    passthrough = ModuleType("wtm.bw01d.passthrough")
    l2_ctrl = ModuleType("wtm.bw01d.l2_ctrl")
    infer = ModuleType("wtm.bw01d.infer")
    pb = ModuleType("infer_service_pb2")

    ok = SimpleNamespace(status_code=0)
    channel = Mock(name="channel")
    bw01d.init = Mock(return_value=channel)
    bw01d.set_phy_l2 = Mock(return_value=ok)
    hb_ctrl.fc_select_layer = Mock(return_value=ok)
    # trim: load_cim_trim(path) 返回带 to_format_bytes(fmt) 的对象; 假实现把文件内容原样当作 payload
    hb_ctrl.CIM_TRIM_DEFAULT_FORMAT = "fake-fmt"
    hb_ctrl.load_cim_trim = Mock(
        side_effect=lambda path: SimpleNamespace(to_format_bytes=lambda fmt: Path(path).read_bytes() + fmt.encode())
    )
    hb_ctrl.print_cim_trim = Mock()
    passthrough.send_to_hd = Mock(return_value=ok)
    evb.init = Mock(return_value="evb-handle")
    evb.deinit = Mock()
    evb.power_control = SimpleNamespace(set_all_channel_enabled=Mock(return_value=ok))
    l2_ctrl.map_program = Mock(return_value=ok)
    l2_ctrl.get_program_progress = Mock(return_value=SimpleNamespace(status_code=0, content={"data": {"progress": 50}}))

    class InferMode:
        def __init__(self, **options):
            self.options = options

    class LogConfig:
        def __init__(self, **options):
            self.options = options

    pb.InferMode = InferMode
    infer.LogConfig = LogConfig
    infer.init = Mock(return_value=True)
    infer.deinit = Mock(return_value=True)
    infer.L2Infer = Mock(
        return_value=[SimpleNamespace(success=True, error=None, data=np.array([[10], [20]], dtype=np.int8))]
    )
    for name in ("Calib", "SetOcState", "ReplaceOcParams", "UpdateLogConfig"):
        setattr(infer, name, Mock(return_value=SimpleNamespace(success=True, error="")))
    infer.GetOcInfo = Mock(return_value=SimpleNamespace(status=SimpleNamespace(success=True, error="")))

    bw01d.evb, bw01d.hb_ctrl, bw01d.passthrough, bw01d.l2_ctrl, bw01d.infer = evb, hb_ctrl, passthrough, l2_ctrl, infer
    wtm.bw01d = bw01d
    for name, module in {
        "wtm": wtm, "wtm.bw01d": bw01d, "wtm.bw01d.evb": evb, "wtm.bw01d.hb_ctrl": hb_ctrl,
        "wtm.bw01d.passthrough": passthrough, "wtm.bw01d.l2_ctrl": l2_ctrl, "wtm.bw01d.infer": infer,
        "infer_service_pb2": pb,
    }.items():  # fmt: skip
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(
        bw01d=bw01d,
        evb=evb,
        hb_ctrl=hb_ctrl,
        passthrough=passthrough,
        l2_ctrl=l2_ctrl,
        infer=infer,
        pb=pb,
        channel=channel,
    )


@pytest.fixture(autouse=True)
def configure_pytester(pytester: pytest.Pytester) -> None:
    """临时 pytest 子项目不继承根项目配置，需要单独指定 asyncio fixture 的 loop scope"""
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function\n")
