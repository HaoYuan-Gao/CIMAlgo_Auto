import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import utils.device_lock as device_lock


class TextReader:
    def __init__(self, *lines):
        self.lines = list(lines)

    async def readline(self):
        if self.lines:
            return self.lines.pop(0)
        # 持锁进程在 LOCKED 之后不再输出, 直到释放前一直保持阻塞
        await asyncio.Event().wait()

    async def read(self):
        remaining = "".join(self.lines)
        self.lines.clear()
        return remaining


def make_remote(status=None, *stdout, stderr="", peer=("10.27.17.45", 22)):
    # status 为 None 表示远程脚本一直没有输出状态行
    process = SimpleNamespace(
        stdout=TextReader(*([status, *stdout] if status is not None else [])),
        stderr=TextReader(stderr),
        stdin=SimpleNamespace(write_eof=Mock()),
        wait=AsyncMock(return_value=None),
        terminate=Mock(),
        close=Mock(),
        wait_closed=AsyncMock(return_value=None),
    )
    connection = SimpleNamespace(
        create_process=AsyncMock(return_value=process),
        get_extra_info=Mock(return_value=peer),
        close=Mock(),
        wait_closed=AsyncMock(return_value=None),
    )
    return connection, process


def test_acquire_holds_until_release_and_keeps_connection_open():
    connection, process = make_remote("LOCKED\n")
    lock = device_lock.RemoteDeviceLock(connection, case="test_conv", owner="alice")

    async def exercise():
        assert await lock.acquire() is lock
        assert lock.acquired
        process.stdin.write_eof.assert_not_called()
        await lock.release()

    asyncio.run(exercise())

    assert not lock.acquired
    assert lock.host == "10.27.17.45:22"
    command = connection.create_process.await_args.args[0]
    assert "flock -n 9" in command
    assert '"owner":"alice"' in command
    assert '"case":"test_conv"' in command
    process.stdin.write_eof.assert_called_once_with()
    process.close.assert_not_called()
    # 连接由调用方持有, 释放锁不关闭它
    connection.close.assert_not_called()
    connection.wait_closed.assert_not_awaited()


def test_busy_device_reports_owner_and_closes_only_process():
    lock_info = (
        '{"owner":"other","case":"running_case","runner_host":"runner-02",'
        '"pid":123,"started_at":"2026-10-15T16:30:00+08:00"}'
    )
    connection, process = make_remote("BUSY\n", f"{lock_info}\n", peer=("2001:db8::1", 2222, 0, 0))
    lock = device_lock.RemoteDeviceLock(connection, case="new_case")

    with pytest.raises(device_lock.DeviceBusyError) as caught:
        asyncio.run(lock.acquire())

    assert caught.value.host == "[2001:db8::1]:2222"
    assert caught.value.lock_info == lock_info
    assert '"owner": "other"' in str(caught.value)
    assert "running_case" in str(caught.value)
    assert "runner-02" in str(caught.value)
    assert "2026-10-15T16:30:00+08:00" in str(caught.value)
    assert not lock.acquired
    process.close.assert_called_once_with()
    process.wait_closed.assert_awaited_once_with()
    connection.close.assert_not_called()


def test_remote_error_is_not_reported_as_busy():
    connection, process = make_remote("ERROR\n", stderr="远程设备未安装 flock\n")
    lock = device_lock.RemoteDeviceLock(connection, case="test_conv")

    with pytest.raises(device_lock.DeviceLockError, match="未安装 flock"):
        asyncio.run(lock.acquire())

    process.close.assert_called_once_with()
    connection.close.assert_not_called()


def test_acquire_timeout_closes_process_and_keeps_connection():
    connection, process = make_remote()  # 远程脚本一直不输出状态
    lock = device_lock.RemoteDeviceLock(connection, case="test_conv", acquire_timeout=0.01)

    with pytest.raises(TimeoutError):
        asyncio.run(lock.acquire())

    assert not lock.acquired
    process.close.assert_called_once_with()
    process.wait_closed.assert_awaited_once_with()
    connection.close.assert_not_called()


def test_acquire_twice_is_rejected():
    connection, _ = make_remote("LOCKED\n")
    lock = device_lock.RemoteDeviceLock(connection, case="test_conv")

    async def exercise():
        await lock.acquire()
        with pytest.raises(device_lock.DeviceLockError, match="已由当前实例持有"):
            await lock.acquire()
        await lock.release()

    asyncio.run(exercise())
    assert connection.create_process.await_count == 1


def test_release_without_acquire_is_safe():
    connection, _ = make_remote("LOCKED\n")
    lock = device_lock.RemoteDeviceLock(connection, case="test_conv")

    asyncio.run(lock.release())
    assert not lock.acquired
    connection.create_process.assert_not_awaited()


def test_context_manager_acquires_and_releases():
    connection, process = make_remote("LOCKED\n")

    async def exercise():
        async with device_lock.RemoteDeviceLock(connection, case="test_conv", owner="alice") as lock:
            assert lock.acquired
            assert lock.case == "test_conv"
            assert lock.owner == "alice"
        assert not lock.acquired

    asyncio.run(exercise())
    process.stdin.write_eof.assert_called_once_with()
    connection.close.assert_not_called()


def test_owner_contains_debugging_metadata():
    owner = json.loads(device_lock.make_owner("test_conv", "alice"))

    assert owner["case"] == "test_conv"
    assert owner["owner"] == "alice"
    assert owner["runner_host"]
    assert owner["pid"] > 0
    assert owner["started_at"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"connection": None, "case": "test"},
        {"case": ""},
        {"case": "test", "owner": ""},
        {"case": "test", "acquire_timeout": 0},
    ],
)
def test_rejects_invalid_configuration(kwargs):
    connection, _ = make_remote("LOCKED\n")
    kwargs.setdefault("connection", connection)

    with pytest.raises(ValueError):
        device_lock.RemoteDeviceLock(**kwargs)
