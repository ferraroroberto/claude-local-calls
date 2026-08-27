"""Shared subprocess lifecycle helpers for hub and model backends.

The hub process and each local model backend have different launch details
(log pipe vs. log file, singleton vs. keyed state, inherited PID support), but
share the same start/stop lifecycle: reject already-running processes, adopt a
reachable external listener, spawn with UTF-8 env, and terminate with a polite
then forceful shutdown.  This module keeps that workflow in one place while the
callers provide their process-specific policy.
"""

from __future__ import annotations

import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, IO, Optional


@dataclass(frozen=True)
class SpawnSpec:
    """Concrete subprocess launch configuration."""

    cmd: list[str]
    cwd: Path
    env: dict[str, str]
    stdout: int | IO[bytes] | IO[str] | None
    stderr: int | IO[bytes] | IO[str] | None
    text: bool = False
    encoding: str | None = None
    errors: str | None = None
    bufsize: int = -1
    creationflags: int = 0


class ProcessSupervisor:
    """Parameterised process start/stop workflow.

    ``start`` owns the common orchestration; callers only supply how to check
    reachability, find external PIDs, build a ``SpawnSpec``, and record the
    resulting ``Popen``. ``stop_popen`` centralises the Windows CTRL_BREAK /
    terminate / kill fallback sequence while preserving per-caller timeouts.
    """

    def __init__(
        self,
        *,
        already_running: Callable[[], bool],
        reachable: Callable[[], bool],
        external_pid: Callable[[], Optional[int]],
        build_spawn_spec: Callable[[], SpawnSpec],
        set_process: Callable[[Optional[subprocess.Popen]], None],
        on_spawned: Callable[[subprocess.Popen], None] | None = None,
        adopt_message: str,
    ) -> None:
        self._already_running = already_running
        self._reachable = reachable
        self._external_pid = external_pid
        self._build_spawn_spec = build_spawn_spec
        self._set_process = set_process
        self._on_spawned = on_spawned
        self._adopt_message = adopt_message

    def start(self) -> tuple[bool, str]:
        if self._already_running():
            return False, "already running"

        if self._reachable():
            ext_pid = self._external_pid()
            suffix = f" (PID {ext_pid})" if ext_pid else ""
            return True, f"{self._adopt_message}{suffix}"

        try:
            spec = self._build_spawn_spec()
            proc = subprocess.Popen(
                spec.cmd,
                cwd=str(spec.cwd),
                stdout=spec.stdout,
                stderr=spec.stderr,
                text=spec.text,
                encoding=spec.encoding,
                errors=spec.errors,
                bufsize=spec.bufsize,
                env=spec.env,
                creationflags=spec.creationflags,
            )
        except Exception as e:
            return False, f"failed to launch: {e}"

        self._set_process(proc)
        if self._on_spawned is not None:
            self._on_spawned(proc)
        return True, f"started (pid={proc.pid})"

    @staticmethod
    def stop_popen(
        proc: subprocess.Popen,
        *,
        terminate_timeout: float,
        kill_timeout: float,
    ) -> tuple[bool, str]:
        try:
            if sys.platform == "win32":
                try:
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                except (OSError, ValueError):
                    pass
                else:
                    # Give the child's own graceful-shutdown handler a
                    # chance to run before escalating — terminate() is an
                    # immediate hard TerminateProcess on Windows, so calling
                    # it right away (the pre-#530 behavior here) never let a
                    # just-sent CTRL_BREAK actually land. tray/tray.py's
                    # HubProcess.stop always waited at this point; every
                    # other stop_popen caller escalated instantly instead —
                    # a divergence that made this the *worse* of two
                    # duplicate stop sequences (#530).
                    try:
                        proc.wait(timeout=terminate_timeout)
                        return True, "stopped"
                    except subprocess.TimeoutExpired:
                        pass
            proc.terminate()
            try:
                proc.wait(timeout=terminate_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=kill_timeout)
        except Exception as e:
            return False, f"error stopping: {e}"
        return True, "stopped"
