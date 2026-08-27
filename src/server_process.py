"""Port/PID utilities for the hub's own listening port, plus the tri-state
ownership check shared with per-model backend tracking.

**Ownership model.** A single hub process binds :8000; whoever spawned it
owns it. Callers that want to know who currently holds a port — the admin
SPA's hub/models routers, ``backend_process.py`` for per-model ports —
call :func:`resolve_ownership` with their own ``running`` flag (they track
their own process handle; this module doesn't):

* ``OWNERSHIP_OURS`` — the caller reports it holds the process itself.
* ``OWNERSHIP_EXTERNAL`` — port is held by someone else's process. We
  can talk to it through the network and force-kill it via
  :func:`kill_pid`, but we have no log tail (Windows can't attach to
  another process's stdout post-hoc).
* ``OWNERSHIP_NONE`` — nothing on the port.

This module does not spawn or own the hub process itself — the tray
drives the hub through its own ``HubProcess`` (``tray/tray.py``) and the
admin restart goes through ``src/hub_process_control.py``'s
``_respawn_watchdog``. What's left here is the cross-platform PID/port
lookup (``find_port_pids``, ``snapshot_listening_pids``, ``kill_pid``) and
the ownership tri-state, both still used by ``backend_process.py`` and the
admin routers.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import socket
import subprocess
import sys
from typing import Optional

from .host_profile import hub_bind_host, hub_port
from .http_client import get_sync_client
from .no_window import NO_WINDOW

logger = logging.getLogger(__name__)

# Uvicorn binds on all interfaces so other machines on the LAN can reach
# the server. Health checks + the canonical "self" URL still use loopback.
BIND_HOST = hub_bind_host()
LOCAL_HOST = "127.0.0.1"
PORT = hub_port()
BASE_URL = f"http://{LOCAL_HOST}:{PORT}"

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


def is_reachable(timeout: float = 1.5) -> bool:
    try:
        r = get_sync_client().get(f"{BASE_URL}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


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
                capture_output=True, encoding="oem", errors="replace", timeout=5,
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
                capture_output=True, encoding="oem", errors="replace", timeout=5,
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

    Shared so :mod:`backend_process` — which tracks a *dict* of per-model
    states instead of a single process handle — reuses the identical
    decision instead of reimplementing it per model (issue #242). This
    module has no process handle of its own for the hub's port, so its
    own callers always pass ``running=False`` (see :func:`external_pid`).
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


def external_pid() -> Optional[int]:
    """PID currently holding the hub's own port, if any.

    This module doesn't spawn or track the hub process itself (see the
    module docstring), so this is always a lookup of *someone else's*
    process — equivalent to ``resolve_external_pid(False, PORT)``. Used by
    ``run_backend.py``'s manual/verification hub launcher to report who
    already owns :8000 when adopting instead of spawning."""
    return resolve_external_pid(False, PORT)


def kill_pid(target_pid: int) -> tuple[bool, str]:
    """Force-kill a PID. Uses taskkill on Windows, SIGKILL elsewhere."""
    try:
        if sys.platform == "win32":
            r = subprocess.run(
                ["taskkill", "/F", "/PID", str(target_pid)],
                capture_output=True, encoding="oem", errors="replace", timeout=5,
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


