"""Windows job-object process lifecycle helpers shared by every TTS engine
that owns a resident/child OS process (Piper's resident ``piper.exe`` pool,
Orpheus's loopback ``llama-server`` child).

A Windows Job Object with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` makes the OS
kill every assigned process when the job's last handle closes — i.e. when
*this* (parent) process dies, by crash, ``terminate()``, or clean exit. The
hub stops a backend with ``TerminateProcess`` (no atexit/finally), so a
grandchild process spawned in its own process group would otherwise leak —
holding GPU VRAM and its internal port. Assigning it to a job makes the OS
reap it whenever we go away.
"""

from __future__ import annotations

import subprocess
import sys

from ..no_window import NO_WINDOW


def _win_kill_on_close_job():
    """Create a Windows Job Object that kills every assigned process when
    its last handle closes — i.e. when *this* (parent) process dies, by
    crash, ``terminate()``, or clean exit.

    Returns the job handle (the caller must keep it alive) or ``None`` on
    non-Windows / failure (callers fall back to the explicit ``terminate`` in
    ``close``).
    """
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    JobObjectExtendedLimitInformation = 9

    class _BASIC(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _EXTENDED(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BASIC),
            ("IoInfo", _IO),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]

    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = _EXTENDED()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(
        job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
    ):
        k32.CloseHandle(job)
        return None
    return job


def _assign_to_job(job, proc: "subprocess.Popen") -> bool:
    """Assign ``proc`` to a Windows job handle from :func:`_win_kill_on_close_job`."""
    if job is None or sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    return bool(k32.AssignProcessToJobObject(job, int(proc._handle)))
