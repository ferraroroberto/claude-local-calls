"""Unit tests for ``ProcessSupervisor.stop_popen``'s Windows escalation (#530).

Regression for the divergence the #530 audit flagged: every ``stop_popen``
caller (``backend_process.stop``, ``tts_engines.orpheus``) sent
``CTRL_BREAK_EVENT`` and immediately called ``terminate()`` — a hard
``TerminateProcess`` on Windows — with no grace period for the graceful
signal to actually land. ``tray/tray.py``'s hand-rolled ``HubProcess.stop``
had always waited between the two, which is the behavior that actually
lets a process shut down cleanly. #530 folds the correct sequence back into
the shared helper (and drops the tray's now-redundant duplicate).

Fakes a ``subprocess.Popen``-shaped object rather than spawning a real
process — the escalation *sequencing* is what's under test, not OS signal
delivery, and a fake keeps this deterministic and platform-independent to
run (only the Windows branch is exercised via ``monkeypatch.setattr(sys,
"platform", "win32")``).
"""

from __future__ import annotations

import subprocess
import sys

from src.process_supervisor import ProcessSupervisor


class _FakePopen:
    """Records calls; ``dies_after`` names which call makes ``wait()`` succeed."""

    def __init__(self, *, dies_after: str) -> None:
        self.dies_after = dies_after
        self.calls: list[str] = []
        self._alive = True

    def send_signal(self, _sig) -> None:
        self.calls.append("send_signal")
        if self.dies_after == "send_signal":
            self._alive = False

    def terminate(self) -> None:
        self.calls.append("terminate")
        if self.dies_after == "terminate":
            self._alive = False

    def kill(self) -> None:
        self.calls.append("kill")
        self._alive = False

    def wait(self, timeout: float) -> None:
        self.calls.append("wait")
        if self._alive:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)


def test_ctrl_break_given_a_grace_period_before_terminate(monkeypatch):
    """A process that dies from CTRL_BREAK alone must never see terminate()."""
    monkeypatch.setattr(sys, "platform", "win32")
    proc = _FakePopen(dies_after="send_signal")

    ok, msg = ProcessSupervisor.stop_popen(proc, terminate_timeout=5, kill_timeout=5)

    assert ok and msg == "stopped"
    assert proc.calls == ["send_signal", "wait"]   # no terminate/kill needed


def test_escalates_to_terminate_only_after_ctrl_break_grace_expires(monkeypatch):
    """A process that ignores CTRL_BREAK escalates to terminate() next."""
    monkeypatch.setattr(sys, "platform", "win32")
    proc = _FakePopen(dies_after="terminate")

    ok, msg = ProcessSupervisor.stop_popen(proc, terminate_timeout=5, kill_timeout=5)

    assert ok and msg == "stopped"
    # CTRL_BREAK's own wait must time out (raise) *before* terminate() is
    # ever called — the exact ordering the pre-#530 code skipped.
    assert proc.calls == ["send_signal", "wait", "terminate", "wait"]


def test_escalates_to_kill_as_last_resort(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    proc = _FakePopen(dies_after="kill")

    ok, msg = ProcessSupervisor.stop_popen(proc, terminate_timeout=5, kill_timeout=5)

    assert ok and msg == "stopped"
    assert proc.calls == ["send_signal", "wait", "terminate", "wait", "kill", "wait"]


def test_non_windows_skips_ctrl_break_entirely(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    proc = _FakePopen(dies_after="terminate")

    ok, msg = ProcessSupervisor.stop_popen(proc, terminate_timeout=5, kill_timeout=5)

    assert ok and msg == "stopped"
    assert proc.calls == ["terminate", "wait"]   # no send_signal on non-Windows
