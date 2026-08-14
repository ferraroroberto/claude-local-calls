"""Regression coverage for ``run_backend._self_contain`` (#507).

Proves the actual ``python -m src.run_backend hub`` launch surface — the
manual/verification path a build agent uses to run a worktree's own hub,
never the tray's supervised ``-m src.server`` — binds every backend it
spawns to its own process lifetime: force-killing this process (the "agent
that is killed, times out, or ends its turn early" shape from #507) reaps a
grandchild spawned exactly the way ``backend_process.start`` spawns an
on-demand TTS/whisper/llama backend (``CREATE_NEW_PROCESS_GROUP`` —
deliberately detached, so this is *not* already covered by a plain
``proc.terminate()`` or a CTRL_BREAK reaching the process group).

Twice on 2026-08-13 this exact shape (a kokoro ``tts_server`` outliving the
verification hub that spawned it) pinned a worktree and halted an unrelated
fleet cleanup run. The regression here is on the abnormally-terminated path
specifically — the clean-exit path already tears backends down via
``server_lifecycle.stop_backend_children`` and was never the reported bug.

Windows-only: the underlying Job Object mechanism is Windows-specific (this
repo's only deployment target).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

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


def test_killed_verification_hub_reaps_its_spawned_backend(tmp_path):
    marker = tmp_path / "run_hub_job_harness.json"
    harness = subprocess.Popen(
        [sys.executable, "-m", "tests._run_hub_job_harness", str(marker)],
        cwd=str(PROJECT_ROOT),
        creationflags=NO_WINDOW,
    )
    try:
        assert _wait_until(marker.exists), "harness never reported its PIDs"
        info = json.loads(marker.read_text(encoding="utf-8"))
        grandchild_pid = info["grandchild_pid"]
        assert psutil.pid_exists(grandchild_pid)

        # Simulate an abnormally-terminated verification run (#507): the
        # hub process is force-killed, not gracefully shut down, so its
        # ASGI shutdown handler (stop_backend_children) never runs.
        subprocess.run(
            ["taskkill", "/F", "/PID", str(harness.pid)],
            capture_output=True, creationflags=NO_WINDOW,
        )

        assert _wait_until(lambda: not psutil.pid_exists(grandchild_pid)), (
            "backend survived the verification hub being force-killed — "
            "KILL_ON_JOB_CLOSE did not fire (#507)"
        )
    finally:
        try:
            harness.wait(timeout=5)
        except subprocess.TimeoutExpired:
            harness.kill()
