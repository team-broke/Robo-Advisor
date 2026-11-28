"""Process liveness decisions must not reclaim uncertain lock owners."""

import ctypes
import os

import pytest

from robo_advisor import process


def test_current_and_missing_process():
    assert process.process_alive(os.getpid()) is True
    assert process.process_alive(999999999) is False


@pytest.mark.parametrize("pid", [0, -1, 0x100000000, True, "123"])
def test_invalid_pid_is_not_probed(pid):
    assert process.process_alive(pid) is False


@pytest.mark.parametrize("error,expected", [(ProcessLookupError, False), (PermissionError, True)])
def test_posix_known_errors(monkeypatch, error, expected):
    def fail(pid, signal):
        raise error

    monkeypatch.setattr(process.os, "kill", fail)
    assert process._posix_alive(123) is expected


def test_posix_unexpected_error_is_not_treated_as_stale(monkeypatch):
    def fail(pid, signal):
        raise OSError("unexpected process lookup error")

    monkeypatch.setattr(process.os, "kill", fail)
    with pytest.raises(OSError, match="unexpected"):
        process._posix_alive(123)


class FakeFunction:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.result


@pytest.mark.parametrize(
    "handle,error,status,expected",
    [
        (None, 87, None, False),
        (None, 5, None, True),
        (1, 0, 0, False),
        (1, 0, 258, True),
    ],
)
def test_windows_process_handle_states(monkeypatch, handle, error, status, expected):
    kernel = type("Kernel", (), {})()
    kernel.OpenProcess = FakeFunction(handle)
    kernel.WaitForSingleObject = FakeFunction(status)
    kernel.CloseHandle = FakeFunction(True)
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: kernel, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: error, raising=False)
    assert process._windows_alive(123) is expected
    assert kernel.OpenProcess.calls == [(0x00100000, False, 123)]
    assert kernel.CloseHandle.calls == ([(1,)] if handle else [])


@pytest.mark.parametrize("handle,error,status", [(None, 123, None), (1, 123, 0xFFFFFFFF)])
def test_windows_unexpected_error_is_not_treated_as_stale(monkeypatch, handle, error, status):
    kernel = type("Kernel", (), {})()
    kernel.OpenProcess = FakeFunction(handle)
    kernel.WaitForSingleObject = FakeFunction(status)
    kernel.CloseHandle = FakeFunction(True)
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: kernel, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: error, raising=False)
    with pytest.raises(OSError):
        process._windows_alive(123)
    assert kernel.CloseHandle.calls == ([(1,)] if handle else [])
