from collections.abc import Sequence
from subprocess import CompletedProcess

import pytest

from device_ctrl import (
    DeviceError,
    DeviceOps,
    DeviceTarget,
    DeviceWorkflows,
    LayerSelection,
)
from device_ctrl.sdk import Wtm233DeviceOps


class FakeDeviceOps(DeviceOps):
    def __init__(self, versions: Sequence[str] = ("2.3.3",)):
        super().__init__(
            DeviceTarget(
                ip="192.0.2.1",
                grpc_port=60000,
                power_address="192.0.2.1:61006",
                control_address="192.0.2.1:60001",
                power_backend="evb",
            )
        )
        self.calls = []
        self._versions = iter(versions)

    def enable_power_channels(self) -> None:
        self.calls.append(("enable_power_channels",))

    def disable_power_channels(self) -> None:
        self.calls.append(("disable_power_channels",))

    def set_phy_l2(self, ids: Sequence[int]) -> None:
        self.calls.append(("set_phy_l2", list(ids)))

    def select_layer(self, hb_id: int, tsv_sel0: int, tsv_sel1: int) -> None:
        self.calls.append(("select_layer", hb_id, tsv_sel0, tsv_sel1))

    def get_n45_version(self) -> str:
        self.calls.append(("get_n45_version",))
        return next(self._versions)

    def switch_n45_version(self, version: str) -> None:
        self.calls.append(("switch_n45_version", version))

    def close(self) -> None:
        self.calls.append(("close",))


def _target(**overrides):
    defaults = dict(
        ip="192.0.2.1",
        grpc_port=60000,
        power_address="192.0.2.1:61006",
        control_address="192.0.2.1:60001",
        power_backend="evb",
    )
    return DeviceTarget(**{**defaults, **overrides})


def test_for_target_selects_registered_driver():
    device = DeviceOps.for_target(_target(driver="wtm_233"))

    assert isinstance(device, Wtm233DeviceOps)


def test_for_target_rejects_unknown_driver():
    with pytest.raises(DeviceError, match="unknown.*没有实现"):
        DeviceOps.for_target(_target(driver="unknown"))


def test_power_cycle_channels_preserves_order():
    device = FakeDeviceOps()
    sleeps = []

    DeviceWorkflows(device, sleep=sleeps.append).power_cycle_channels(5)

    assert device.calls == [
        ("disable_power_channels",),
        ("enable_power_channels",),
    ]
    assert sleeps == [5]


def test_configure_layers_sets_phy_before_selecting_layers():
    device = FakeDeviceOps()

    DeviceWorkflows(device).configure_layers(
        [0, 1, 2],
        [
            LayerSelection(hb_id=0, tsv_sel0=0, tsv_sel1=255),
            LayerSelection(hb_id=1, tsv_sel0=0, tsv_sel1=255),
        ],
    )

    assert device.calls == [
        ("set_phy_l2", [0, 1, 2]),
        ("select_layer", 0, 0, 255),
        ("select_layer", 1, 0, 255),
    ]


def test_ensure_n45_version_switches_and_verifies():
    device = FakeDeviceOps(["2.3.2", "2.3.3"])

    switched = DeviceWorkflows(device).ensure_n45_version("2.3.3")

    assert switched is True
    assert device.calls == [
        ("get_n45_version",),
        ("switch_n45_version", "2.3.3"),
        ("get_n45_version",),
    ]


def test_ensure_n45_version_rejects_mismatch_after_switch():
    device = FakeDeviceOps(["2.3.2", "2.3.2"])

    with pytest.raises(DeviceError, match="期望 2.3.3，实际 2.3.2"):
        DeviceWorkflows(device).ensure_n45_version("2.3.3")


def test_get_n45_version_uses_explicit_server_and_site(monkeypatch):
    target = DeviceTarget(
        ip="192.0.2.1",
        grpc_port=60010,
        power_address="192.0.2.1:61006",
        control_address="192.0.2.1:60001",
        power_backend="evb",
    )
    device = Wtm233DeviceOps(target)
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        return CompletedProcess(command, 0, stdout="2.3.3\n", stderr="")

    monkeypatch.setattr("device_ctrl.sdk.wtm_233.subprocess.run", fake_run)

    assert device.get_n45_version() == "2.3.3"
    assert commands[0][0] == [
        "bw01d_ctrl",
        "get-n45-version",
        "--server",
        "192.0.2.1:60001",
        "--site",
        "1",
    ]
    assert commands[0][1]["timeout"] == 60
