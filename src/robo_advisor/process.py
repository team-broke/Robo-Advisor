"""Cross-platform process liveness check for stale lock files."""

from __future__ import annotations

import os


def _posix_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OverflowError):
        return False
    except PermissionError:
        return True
    return True


def _windows_alive(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    open_process.restype = wintypes.HANDLE
    wait = kernel32.WaitForSingleObject
    wait.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    wait.restype = wintypes.DWORD
    close = kernel32.CloseHandle
    close.argtypes = (wintypes.HANDLE,)
    close.restype = wintypes.BOOL

    handle = open_process(0x00100000, False, pid)  # SYNCHRONIZE
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: no such PID
            return False
        if error == 5:  # ERROR_ACCESS_DENIED: do not reclaim an uncertain lock
            return True
        raise OSError(error, "OpenProcess failed")
    try:
        status = wait(handle, 0)
        if status == 0:  # WAIT_OBJECT_0: process has exited
            return False
        if status == 258:  # WAIT_TIMEOUT: process is still running
            return True
        raise OSError(ctypes.get_last_error(), "WaitForSingleObject failed")
    finally:
        if not close(handle):
            raise OSError(ctypes.get_last_error(), "CloseHandle failed")


def process_alive(pid: int) -> bool:
    """Return False only when the PID is invalid or its process has exited."""
    if not isinstance(pid, int) or isinstance(pid, bool) or not 0 < pid <= 0xFFFFFFFF:
        return False
    if os.name == "nt":
        return _windows_alive(pid)
    return _posix_alive(pid)
