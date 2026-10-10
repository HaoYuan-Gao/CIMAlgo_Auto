"""BW01D 设备适配器的 SDK 调用契约；所有测试使用模拟通道。"""

import sys
from subprocess import CompletedProcess
from threading import Event
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest

from utils.device_ops import Bw01dDeviceOps, DeviceError, DeviceOps, DeviceTarget


@pytest.fixture
def sdk(monkeypatch):
    channel = Mock()
    response = SimpleNamespace(status_code=0, data=b"result")
    bw01d = ModuleType("wtm.bw01d")
    bw01d.init = Mock(return_value=channel)
    bw01d.set_phy_l2 = Mock(return_value=response)
    bw01d.hb_ctrl = SimpleNamespace(fc_select_layer=Mock(return_value=response))
    bw01d.l2_ctrl = SimpleNamespace(
        **{
            name: Mock(return_value=response)
            for name in (
                "upload_program",
                "run_program",
                "get_program_result",
                "get_program_progress",
                "map_program",
                "infer",
            )
        }
    )
    handle = object()
    bw01d.evb = SimpleNamespace(
        init=Mock(return_value=handle),
        deinit=Mock(),
        power_control=SimpleNamespace(set_all_channel_enabled=Mock(return_value=response)),
    )
    wtm = ModuleType("wtm")
    wtm.bw01d = bw01d
    monkeypatch.setitem(sys.modules, "wtm", wtm)
    monkeypatch.setitem(sys.modules, "wtm.bw01d", bw01d)
    return SimpleNamespace(bw01d=bw01d, channel=channel, handle=handle, response=response)


@pytest.fixture
def device(sdk):
    target = DeviceTarget(
        ip="192.0.2.1",
        bmc_ip="192.0.2.2",
        grpc_port=60010,
        evb_port=61007,
        server_port=60011,
    )
    return DeviceOps.for_target(target)


def test_init_connection_reuses_channel_and_close_is_idempotent(device, sdk):
    assert isinstance(device, Bw01dDeviceOps)
    assert device.init_connection() is sdk.channel
    assert device.init_connection() is sdk.channel
    sdk.bw01d.init.assert_called_once_with({"ip": "192.0.2.1", "port": 60010, "is_cluster": False, "test_stage": "CP3"})

    device.close()
    device.close()
    sdk.channel.close.assert_called_once_with()


def test_grpc_api_calls_use_the_cached_channel_and_preserve_responses(device, sdk):
    assert device.set_phy_l2([1, 2], timeout=12) is sdk.response
    sdk.bw01d.set_phy_l2.assert_called_once_with(sdk.channel, [1, 2], timeout=12)

    assert device.select_layer(1, 2, cim_trim0=b"a", cim_trim1=b"b") is sdk.response
    sdk.bw01d.hb_ctrl.fc_select_layer.assert_called_once_with(
        sdk.channel, hb_id=1, tsv_sel0=2, tsv_sel1=255, timeout=30, cim_trim0=b"a", cim_trim1=b"b"
    )

    assert device.upload_program(4, "program.tar", chip_id=2, max_wait_seconds=65) is sdk.response
    sdk.bw01d.l2_ctrl.upload_program.assert_called_once_with(
        sdk.channel, 4, "program.tar", max_wait_seconds=65, chip_id=2
    )
    assert device.run_program(4, 8, 9) is sdk.response
    sdk.bw01d.l2_ctrl.run_program.assert_called_once_with(sdk.channel, 4, 8, 9, max_wait_seconds=7200)
    assert device.get_program_result(4, "results") is sdk.response
    sdk.bw01d.l2_ctrl.get_program_result.assert_called_once_with(sdk.channel, 4, "results", max_wait_seconds=30)
    assert device.get_program_progress(4) is sdk.response
    sdk.bw01d.l2_ctrl.get_program_progress.assert_called_once_with(sdk.channel, 4, max_wait_seconds=5)
    assert device.map_program(4, "program.tar", 8, 9, "results", chip_id=2) is sdk.response
    sdk.bw01d.l2_ctrl.map_program.assert_called_once_with(
        sdk.channel, 4, "program.tar", 8, 9, output_dir="results", max_wait_seconds=7200, chip_id=2
    )
    assert device.infer(4, "results") is sdk.response
    sdk.bw01d.l2_ctrl.infer.assert_called_once_with(sdk.channel, 4, output_dir="results", max_wait_seconds=90)
    sdk.bw01d.init.assert_called_once()


def test_sdk_failure_becomes_device_error(device, sdk):
    sdk.bw01d.l2_ctrl.run_program.return_value = SimpleNamespace(status_code=1, error="烧录失败")

    with pytest.raises(DeviceError, match="烧录失败"):
        device.run_program(4, 8, 9)


def test_program_polls_each_active_l2_and_returns_results(device, sdk, monkeypatch, tmp_path):
    assert Bw01dDeviceOps.program is DeviceOps.program
    bars = []

    class ProgressBar:
        def __init__(self, **options):
            self.options = options
            self.n = options.get("initial", 0)
            self.updates = []
            bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def update(self, amount):
            self.n += amount
            self.updates.append(amount)

    monkeypatch.setattr("utils.device_ops.base.tqdm", ProgressBar)
    started = {0: Event(), 1: Event()}
    release = {0: Event(), 1: Event()}
    polls = {0: 0, 1: 0}

    def map_program(_channel, l2_id, _tar, _k, _d, **_kwargs):
        started[l2_id].set()
        assert release[l2_id].wait(timeout=2)
        return sdk.response

    def progress(_channel, l2_id, **_kwargs):
        assert started[l2_id].wait(timeout=2)
        polls[l2_id] += 1
        if polls[l2_id] == 1:
            assert not any(bar.options["desc"] == f"L2_{l2_id}" for bar in bars)
            return SimpleNamespace(status_code=0, content={"data": {"progress": 0}})
        release[l2_id].set()
        return SimpleNamespace(status_code=0, content={"data": {"progress": 35 + l2_id}})

    sdk.bw01d.l2_ctrl.map_program.side_effect = map_program
    sdk.bw01d.l2_ctrl.get_program_progress.side_effect = progress
    results = device.program(
        [(0, tmp_path / "zero.tar"), (1, tmp_path / "one.tar")],
        8,
        9,
        tmp_path / "program",
        max_wait_seconds=120,
        max_workers=2,
        poll_interval=0.01,
    )

    assert results == {0: sdk.response, 1: sdk.response}
    assert all(bar.options["bar_format"] == "{desc}: {percentage:.0f}%|{bar}| [{elapsed}]" for bar in bars)
    assert {bar.options["initial"] for bar in bars} == {35, 36}
    assert [bar.n for bar in bars] == [100, 100]
    assert all(any(0 < amount < 100 for amount in bar.updates) for bar in bars)
    assert sdk.bw01d.l2_ctrl.map_program.call_count == 2
    sdk.bw01d.l2_ctrl.map_program.assert_has_calls(
        [
            call(
                sdk.channel,
                0,
                str(tmp_path / "zero.tar"),
                8,
                9,
                output_dir=str(tmp_path / "program/L2_0"),
                max_wait_seconds=120,
                chip_id=0,
            ),
            call(
                sdk.channel,
                1,
                str(tmp_path / "one.tar"),
                8,
                9,
                output_dir=str(tmp_path / "program/L2_1"),
                max_wait_seconds=120,
                chip_id=0,
            ),
        ],
        any_order=True,
    )
    sdk.bw01d.l2_ctrl.upload_program.assert_not_called()
    sdk.bw01d.l2_ctrl.run_program.assert_not_called()
    sdk.bw01d.l2_ctrl.get_program_result.assert_not_called()


def test_program_rejects_duplicate_l2_ids(device, tmp_path):
    with pytest.raises(DeviceError, match="L2 ID 不能重复"):
        device.program([(0, "a.tar"), (0, "b.tar")], 8, 9, tmp_path)


def test_program_failure_does_not_complete_progress_bar(device, sdk, monkeypatch, tmp_path):
    bar = MagicMock()
    bar.__enter__.return_value = bar
    bar.n = 0
    monkeypatch.setattr("utils.device_ops.base.tqdm", Mock(return_value=bar))
    sdk.bw01d.l2_ctrl.map_program.return_value = SimpleNamespace(status_code=1, error="烧录失败")

    with pytest.raises(DeviceError, match="L2_0.*烧录失败"):
        device.program([(0, "program.tar")], 8, 9, tmp_path, poll_interval=0.01)

    bar.update.assert_not_called()
    sdk.bw01d.l2_ctrl.map_program.assert_called_once()
    sdk.bw01d.l2_ctrl.get_program_result.assert_not_called()


def test_control_commands_add_site_only_for_firmware(device, monkeypatch):
    outputs = iter(("2.3.3\n", "v1.1.0\n", ""))
    run = Mock(side_effect=lambda command, **kwargs: CompletedProcess(command, 0, next(outputs), ""))
    monkeypatch.setattr("utils.device_ops.bw01d.subprocess.run", run)

    assert device.get_sdk_version() == "2.3.3"
    assert device.get_firmware_version() == "v1.1.0"
    device.switch_firmware_version("v1.2.0")

    assert [invocation.args[0] for invocation in run.call_args_list] == [
        ["bw01d_ctrl", "get-sdk-version", "--server", "192.0.2.1:60011"],
        ["bw01d_ctrl", "get-n45-version", "--server", "192.0.2.1:60011", "--site", "1"],
        ["bw01d_ctrl", "switch-n45-version", "--server", "192.0.2.1:60011", "v1.2.0", "--site", "1"],
    ]


def test_sdk_version_command_uses_default_server_port_without_site(device, monkeypatch):
    device.target = DeviceTarget(ip="192.0.2.1", bmc_ip="192.0.2.2")
    run = Mock(return_value=CompletedProcess([], 0, "2.3.3\n", ""))
    monkeypatch.setattr("utils.device_ops.bw01d.subprocess.run", run)

    assert device.get_sdk_version() == "2.3.3"
    assert run.call_args.args[0] == ["bw01d_ctrl", "get-sdk-version", "--server", "192.0.2.1:60001"]


def test_control_command_without_chip_args_omits_site(device, monkeypatch):
    run = Mock(return_value=CompletedProcess([], 0, "v1.2.0\n", ""))
    monkeypatch.setattr("utils.device_ops.bw01d.subprocess.run", run)

    assert device._run_control("list-n45-firmware", timeout=30) == "v1.2.0"
    assert run.call_args.args[0] == ["bw01d_ctrl", "list-n45-firmware", "--server", "192.0.2.1:60011"]


def test_evb_power_calls_release_the_serial_handle(device, sdk):
    assert device.evb_power_set_all_enabled() is sdk.response
    assert device.evb_power_set_all_disabled() is sdk.response

    assert sdk.bw01d.evb.power_control.set_all_channel_enabled.call_args_list == [
        call(sdk.handle, True),
        call(sdk.handle, False),
    ]
    assert sdk.bw01d.evb.init.call_args_list == [
        call("192.0.2.1:61007", timeout=2),
        call("192.0.2.1:61007", timeout=2),
    ]
    assert sdk.bw01d.evb.deinit.call_args_list == [call(sdk.handle), call(sdk.handle)]


def test_power_cycle_waits_between_power_off_and_on(device, monkeypatch):
    operations = Mock()
    monkeypatch.setattr(device, "evb_power_set_all_disabled", operations.power_off)
    monkeypatch.setattr("utils.device_ops.bw01d.time.sleep", operations.sleep)
    monkeypatch.setattr(device, "evb_power_set_all_enabled", operations.power_on)

    device.power_cycle()
    device.power_cycle(off_seconds=2)

    assert operations.mock_calls == [
        call.power_off(),
        call.sleep(5),
        call.power_on(),
        call.power_off(),
        call.sleep(2),
        call.power_on(),
    ]
