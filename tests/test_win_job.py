"""Regression coverage for src/win_job.py's Job Object containment (#468).

Proves the two guarantees the e2e ``hub_url`` fixture now relies on:

* ``terminate_and_close`` kills every current job member, even one whose
  immediate parent has already exited — the exact shape of the reported
  incident (hub PID's parent gone, its ``tts_server`` grandchild alive).
* A process holding a kill-on-close job's only handle being force-killed
  reaps every member automatically, with no cooperating code running in
  the victim — the "spawner killed mid-run" guarantee, exercised via the
  ``_win_job_harness`` helper process.

Windows-only: Job Objects are a Windows API and the reported bug is
Windows-specific (this repo's only deployment target).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from src import win_job
from src.no_window import NO_WINDOW

PROJECT_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Job Objects are a Windows-only API"
)


def _wait_until(predicate, timeout: float = 10.0, interval: float = 0.1) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_terminate_and_close_kills_grandchild_after_parent_already_exited():
    job = win_job.create_kill_on_close_job()
    assert job is not None

    # A short-lived parent that spawns a long-lived grandchild and prints
    # its PID, then exits immediately — reproducing "parent already gone,
    # grandchild survives" without waiting on a real backend to load.
    parent = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess, sys; "
         "gc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], "
         "creationflags=subprocess.CREATE_NO_WINDOW); "
         "print(gc.pid, flush=True)"],
        stdout=subprocess.PIPE, text=True,
        creationflags=NO_WINDOW,
    )
    assert win_job.assign_pid(job, parent.pid)
    grandchild_pid = int(parent.stdout.readline().strip())
    parent.wait(timeout=5)  # parent exits on its own; grandchild keeps sleeping

    assert psutil.pid_exists(grandchild_pid), "grandchild should still be alive here"

    win_job.terminate_and_close(job)

    assert _wait_until(lambda: not psutil.pid_exists(grandchild_pid)), (
        "grandchild survived terminate_and_close despite its parent "
        "already being gone — this is the #468 incident shape"
    )


def test_kill_on_close_reaps_grandchild_when_owner_is_force_killed(tmp_path):
    marker = tmp_path / "job_harness.json"
    harness = subprocess.Popen(
        [sys.executable, "-m", "tests._win_job_harness", str(marker)],
        cwd=str(PROJECT_ROOT),
        creationflags=NO_WINDOW,
    )
    try:
        assert _wait_until(marker.exists), "harness never reported its PIDs"
        info = json.loads(marker.read_text(encoding="utf-8"))
        assert "error" not in info, info.get("error")
        grandchild_pid = info["grandchild_pid"]
        assert psutil.pid_exists(grandchild_pid)

        # Simulate "the spawning process is killed mid-run" (#468's scope
        # note) — force-kill the harness directly, same shape as an
        # impatient worktree teardown killing a hung verification process.
        subprocess.run(
            ["taskkill", "/F", "/PID", str(harness.pid)],
            capture_output=True, creationflags=NO_WINDOW,
        )

        assert _wait_until(lambda: not psutil.pid_exists(grandchild_pid)), (
            "grandchild survived its owner being force-killed — "
            "KILL_ON_JOB_CLOSE did not fire"
        )
    finally:
        try:
            harness.wait(timeout=5)
        except subprocess.TimeoutExpired:
            harness.kill()
