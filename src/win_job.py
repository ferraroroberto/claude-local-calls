"""Windows Job Object containment for hub subprocesses (#468).

The single implementation of "assign a child to a job that kills its whole
membership when the last handle closes". Two callers (#470 folded the second
in, which had grown its own ~70-line ctypes copy):

* ``tests/e2e``'s ``hub_url`` fixture, containing a throwaway test hub and
  every backend it spawns — the incident this module was written for, below.
* The TTS engines that own an OS child (``tts_engines/piper.py``'s resident
  ``piper.exe`` pool, ``tts_engines/orpheus.py``'s loopback ``llama-server``).
  The hub stops a backend with ``TerminateProcess`` — no ``atexit``, no
  ``finally`` — so a grandchild in its own process group would otherwise leak,
  holding GPU VRAM and its internal port.

A verification run (``tests/e2e``'s ``hub_url`` fixture) boots a throwaway
hub as a plain ``subprocess.Popen``. If a test drives an ``on_demand``
backend row (``src.backend_process.start``), that backend is spawned with
``CREATE_NEW_PROCESS_GROUP`` (see ``server_process.WIN_NEW_GROUP``) —
deliberately detached so ``CTRL_BREAK_EVENT`` aimed at the hub during its
own polite shutdown doesn't propagate to it. That's correct for the
*primary* checkout, where a backend surviving a hub restart is the intended
behaviour (``inherit_running_backends``). It's wrong for a throwaway test
hub: the fixture's teardown only ``proc.terminate()``s the hub PID itself,
which never reaches a grandchild in a different process group, so the
backend (e.g. an on-demand ``kokoro`` ``tts_server``) outlives the hub that
spawned it and pins the worktree it's rooted in (#468).

A Windows Job Object fixes this without caring about process-group
boundaries: assigning the hub's ``Popen`` to a job created with
``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` automatically pulls in every future
descendant (subprocess creation joins the creator's job unless it opts out
with ``CREATE_BREAKAWAY_FROM_JOB``, which nothing here does). Two distinct
guarantees fall out of that one flag:

* :func:`terminate_and_close` kills every *current* member of the job, even
  ones whose immediate parent already exited — exactly the observed
  incident (the hub PID's parent was gone, the ``tts_server`` grandchild
  wasn't). Call it from the fixture's ``finally:`` block.
* If the process holding the job handle (the test harness itself) is
  killed before reaching that ``finally:``, Windows closes its handles for
  it — including the job handle — which fires ``KILL_ON_JOB_CLOSE`` and
  reaps every member with no application code needing to run at all. This
  is what survives the spawner being killed mid-run, not just a clean exit.

No-op (returns ``None`` / ``False``) on non-Windows and on any API failure
— containment is best-effort so a broken job object never blocks a test
run from booting or tearing down its hub.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from typing import Optional

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _JobObjectExtendedLimitInformation = 9
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100

    # Explicit argtypes/restype are required, not cosmetic: ctypes defaults
    # an undeclared restype to a 32-bit ``c_int``, which silently truncates
    # a HANDLE (pointer-sized, 64 bits on Win64) instead of raising — the
    # calls would "work" against low-numbered handles in a short-lived test
    # and then misbehave the day a handle value doesn't fit in 32 bits.
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
    )
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    _kernel32.TerminateJobObject.restype = wintypes.BOOL
    _kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


def create_kill_on_close_job(name: Optional[str] = None) -> Optional[int]:
    """Create a Job Object whose members all die when its last handle closes.

    Returns the job HANDLE as a plain ``int``, or ``None`` on non-Windows /
    on any API failure — callers must treat ``None`` as "no containment
    available" and fall back to their prior best-effort teardown.
    """
    if not _IS_WINDOWS:
        return None
    try:
        handle = _kernel32.CreateJobObjectW(None, name)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = _kernel32.SetInformationJobObject(
            handle, _JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info),
        )
        if not ok:
            err = ctypes.WinError(ctypes.get_last_error())
            _kernel32.CloseHandle(handle)
            raise err
        return handle
    except Exception:  # noqa: BLE001 — containment is best-effort
        logger.warning("could not create kill-on-close job object", exc_info=True)
        return None


def assign_pid(job_handle: Optional[int], pid: int) -> bool:
    """Add process ``pid`` (and, transitively, its future children) to
    ``job_handle``. Best-effort: never raises, returns ``False`` on failure."""
    if not _IS_WINDOWS or not job_handle:
        return False
    proc_handle = _kernel32.OpenProcess(
        _PROCESS_TERMINATE | _PROCESS_SET_QUOTA, False, pid
    )
    if not proc_handle:
        logger.warning(
            "OpenProcess(%s) failed for job assignment: %s",
            pid, ctypes.WinError(ctypes.get_last_error()),
        )
        return False
    try:
        ok = _kernel32.AssignProcessToJobObject(job_handle, proc_handle)
        if not ok:
            logger.warning(
                "AssignProcessToJobObject(%s) failed: %s",
                pid, ctypes.WinError(ctypes.get_last_error()),
            )
        return bool(ok)
    finally:
        _kernel32.CloseHandle(proc_handle)


def terminate_and_close(job_handle: Optional[int]) -> None:
    """Kill every process currently in ``job_handle`` and release it.

    Explicit ``TerminateJobObject`` gives immediate, deterministic cleanup
    on the happy path (a clean fixture teardown) and reaches a member whose
    immediate parent has already exited — the exact shape of #468's
    incident. The ``KILL_ON_JOB_CLOSE`` flag set at creation is what makes
    the same cleanup happen automatically if this line is never reached at
    all (the caller itself killed mid-run) — Windows closes a dying
    process's handles for it. No-op if ``job_handle`` is ``None``.
    """
    if not _IS_WINDOWS or not job_handle:
        return
    try:
        _kernel32.TerminateJobObject(job_handle, 1)
    except Exception:  # noqa: BLE001 — best-effort teardown
        logger.warning("TerminateJobObject failed", exc_info=True)
    finally:
        try:
            _kernel32.CloseHandle(job_handle)
        except Exception:  # noqa: BLE001
            pass
