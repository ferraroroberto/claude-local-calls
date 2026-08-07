"""Manage the FastAPI hub as a subprocess, for the tray and the admin SPA.

Keeps a singleton `Popen` + a background reader thread that drains
stdout/stderr into a thread-safe ring buffer. State lives on a
module-level singleton so it survives across calls: the tray imports
this module once and drives the hub for its whole lifetime, and the
admin SPA — mounted in-process at ``/admin`` inside the hub — imports
it to read ownership state and adopt/force-stop the running hub across
many requests. There is no per-interaction script rerun; one long-lived
import owns one handle.

**Ownership model.** A single hub process binds :8000; whoever spawned
it owns it. Other observers (the SPA's own hub router talking to the
hub it lives inside, the tray, ``run_hub.bat`` invoked while the tray
is up) can *adopt* the running hub: they see it as reachable but don't
try to start a duplicate and don't tear it down on their own exit.
Three states:

* ``OWNERSHIP_OURS`` — we hold a live ``Popen``; ``stop()`` will tear
  it down and our log ring has its stdout.
* ``OWNERSHIP_EXTERNAL`` — port is held by someone else's process. We
  can talk to it through the network and we can force-kill it via
  :func:`force_stop_external`, but we have no log tail (Windows can't
  attach to another process's stdout post-hoc).
* ``OWNERSHIP_NONE`` — nothing on the port; safe to ``start()``.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import socket
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Deque, Optional

from .http_client import get_sync_client
from .no_window import NO_WINDOW
from .process_supervisor import ProcessSupervisor, SpawnSpec

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Uvicorn binds on all interfaces so other machines on the LAN can reach
# the server. Health checks + the canonical "self" URL still use loopback.
BIND_HOST = "0.0.0.0"
LOCAL_HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{LOCAL_HOST}:{PORT}"
RING_MAX = 1000

OWNERSHIP_OURS = "ours"
OWNERSHIP_EXTERNAL = "external"
OWNERSHIP_NONE = "none"

# On Windows, give the child its own process group so CTRL_BREAK_EVENT
# during stop() doesn't propagate to the tray launcher, and suppress the
# console so silent parents (pythonw, e.g. the tray) don't spawn a
# stray cmd window.
WIN_NEW_GROUP = (
    subprocess.CREATE_NEW_PROCESS_GROUP | NO_WINDOW if sys.platform == "win32" else 0
)


def lan_ip() -> Optional[str]:
    """Best-effort LAN IP of this machine.

    Uses the UDP-connect trick: no packet is actually sent, but the OS
    routing table picks the outbound interface, which is the address
    other machines on the LAN should use to reach us. Returns None if
    no route is available (fully offline).
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def lan_url() -> Optional[str]:
    ip = lan_ip()
    return f"http://{ip}:{PORT}" if ip else None


class _ServerState:
    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.log: Deque[str] = deque(maxlen=RING_MAX)
        self.lock = threading.Lock()
        self.reader: Optional[threading.Thread] = None


_STATE = _ServerState()


def is_running() -> bool:
    p = _STATE.proc
    return p is not None and p.poll() is None


def is_reachable(timeout: float = 1.5) -> bool:
    try:
        r = get_sync_client().get(f"{BASE_URL}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def log_lines() -> list[str]:
    with _STATE.lock:
        return list(_STATE.log)


def clear_log() -> None:
    with _STATE.lock:
        _STATE.log.clear()


def _reader(proc: subprocess.Popen) -> None:
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        with _STATE.lock:
            _STATE.log.append(line)


def start() -> tuple[bool, str]:
    def build_spawn_spec() -> SpawnSpec:
        clear_log()
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        return SpawnSpec(
            cmd=[sys.executable, "-m", "src.server"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            creationflags=WIN_NEW_GROUP,
        )

    def on_spawned(proc: subprocess.Popen) -> None:
        t = threading.Thread(target=_reader, args=(proc,), daemon=True)
        t.start()
        _STATE.reader = t

    return ProcessSupervisor(
        already_running=is_running,
        reachable=lambda: is_reachable(timeout=0.4),
        external_pid=external_pid,
        build_spawn_spec=build_spawn_spec,
        set_process=lambda proc: setattr(_STATE, "proc", proc),
        on_spawned=on_spawned,
        adopt_message="adopted external hub",
    ).start()


def stop() -> tuple[bool, str]:
    p = _STATE.proc
    if p is None or p.poll() is not None:
        _STATE.proc = None
        return False, "not running"

    ok, msg = ProcessSupervisor.stop_popen(
        p,
        terminate_timeout=5,
        kill_timeout=5,
    )
    if not ok:
        return ok, msg

    _STATE.proc = None
    return True, "stopped"


def pid() -> Optional[int]:
    p = _STATE.proc
    if p is None or p.poll() is not None:
        return None
    return p.pid


def snapshot_listening_pids() -> dict[int, list[int]]:
    """One-shot map of every listening TCP port → PID list.

    The legacy :func:`find_port_pids` shells out per-port. The admin
    webapp's Models tab queries ownership + pid for every backend on
    every poll — O(N) ``netstat`` invocations at ~1 s each adds up
    fast. This consolidates the lookup into a single in-process call
    via :func:`psutil.net_connections` (~2 ms for ~70 sockets), with
    netstat / lsof kept as the fallback if psutil refuses (Windows
    sometimes denies access without admin for system-wide queries).

    Returns an empty dict if all paths fail — callers must tolerate
    that and treat it as "no information; fall back to ``[]``".
    """
    try:
        import psutil

        result: dict[int, set[int]] = {}
        for conn in psutil.net_connections(kind="tcp"):
            if conn.status != psutil.CONN_LISTEN:
                continue
            if conn.laddr is None or not conn.pid:
                continue
            result.setdefault(conn.laddr.port, set()).add(conn.pid)
        if result:
            return {p: sorted(pids) for p, pids in result.items()}
    except ImportError:
        pass
    except (psutil.AccessDenied, RuntimeError):
        pass
    except Exception:  # noqa: BLE001 — never let observability poison the hub
        pass

    # Fallback: shell out. ~1 s on Windows; only triggered if psutil
    # refused (admin denial on a locked-down box) or hit an OS error.
    result_fb: dict[int, set[int]] = {}
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=5,
                creationflags=NO_WINDOW,
            ).stdout
            line_re = re.compile(
                r"\s*TCP\s+\S+:(\d+)\s+\S+\s+LISTENING\s+(\d+)"
            )
            for line in out.splitlines():
                m = line_re.match(line)
                if not m:
                    continue
                result_fb.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
        else:
            out = subprocess.run(
                ["lsof", "-nP", "-iTCP", "-sTCP:LISTEN", "-FpPn"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            pid: int | None = None
            for line in out.splitlines():
                if line.startswith("p"):
                    try:
                        pid = int(line[1:])
                    except ValueError:
                        pid = None
                elif line.startswith("n") and pid is not None:
                    tail = line[1:]
                    if ":" in tail:
                        try:
                            port = int(tail.rsplit(":", 1)[-1])
                        except ValueError:
                            continue
                        result_fb.setdefault(port, set()).add(pid)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return {}
    return {p: sorted(pids) for p, pids in result_fb.items()}


def find_port_pids(port: int) -> list[int]:
    """Return PIDs of processes listening on `port`, if any.

    Cross-platform: uses `netstat` on Windows, `lsof` on macOS/Linux.
    Returns [] if nothing is listening or the tool isn't available.

    Note: under ``pythonw`` (e.g. when called from the tray) Windows
    Terminal will spawn a fresh window for any console child unless we
    pass ``NO_WINDOW``. Callers that need ports for *many*
    sockets in one tick should prefer :func:`snapshot_listening_pids`
    to avoid spawning N netstat / lsof processes.
    """
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=5,
                creationflags=NO_WINDOW,
            ).stdout
            pids: set[int] = set()
            for line in out.splitlines():
                if "LISTENING" not in line:
                    continue
                # columns: Proto  LocalAddress  ForeignAddress  State  PID
                m = re.search(rf":{port}\b.*LISTENING\s+(\d+)", line)
                if m:
                    pids.add(int(m.group(1)))
            return sorted(pids)
        else:
            out = subprocess.run(
                ["lsof", "-nP", "-a", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            pids = sorted({int(x) for x in out.split() if x.strip().isdigit()})
            logger.info("ℹ️ listener lookup for TCP :%s resolved PID(s) %s", port, pids)
            return pids
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


def resolve_ownership(running: bool, port: int) -> str:
    """Shared tri-state (``OURS``/``EXTERNAL``/``NONE``) check for one port.

    Factored out of :func:`ownership` so :mod:`backend_process` — which
    tracks a *dict* of per-model states instead of one singleton — can
    reuse the identical decision instead of reimplementing it per model
    (issue #242).
    """
    if running:
        return OWNERSHIP_OURS
    if find_port_pids(port):
        return OWNERSHIP_EXTERNAL
    return OWNERSHIP_NONE


def resolve_external_pid(running: bool, port: int) -> Optional[int]:
    """Shared "who's holding this port" lookup — see :func:`resolve_ownership`."""
    if running:
        return None
    pids = find_port_pids(port)
    return pids[0] if pids else None


def ownership() -> str:
    """Return ``OWNERSHIP_OURS`` / ``EXTERNAL`` / ``NONE`` for the hub port."""
    return resolve_ownership(is_running(), PORT)


def external_pid() -> Optional[int]:
    """PID of the external port-holder, or ``None`` if we own it / port is free."""
    return resolve_external_pid(is_running(), PORT)


def force_stop_external() -> tuple[bool, str]:
    """Force-kill whoever currently holds :8000, if it's not us.

    Use this when the user wants to reclaim the port — e.g. clicking
    "Stop & take over" in the Server tab, or after a tray crash left a
    detached pythonw owning the hub.
    """
    target = external_pid()
    if target is None:
        return False, f"no external process on port {PORT}"
    return kill_pid(target)


def kill_pid(target_pid: int) -> tuple[bool, str]:
    """Force-kill a PID. Uses taskkill on Windows, SIGKILL elsewhere."""
    try:
        if sys.platform == "win32":
            r = subprocess.run(
                ["taskkill", "/F", "/PID", str(target_pid)],
                capture_output=True, text=True, timeout=5,
                creationflags=NO_WINDOW,
            )
            if r.returncode == 0:
                return True, f"killed pid {target_pid}"
            return False, (r.stderr or r.stdout or "taskkill failed").strip()
        else:
            os.kill(target_pid, signal.SIGKILL)
            return True, f"killed pid {target_pid}"
    except ProcessLookupError:
        return True, f"pid {target_pid} already gone"
    except Exception as e:
        return False, f"error killing {target_pid}: {e}"


