"""Manage the FastAPI server as a subprocess, for the Streamlit UI.

Keeps a singleton `Popen` + a background reader thread that drains
stdout/stderr into a thread-safe ring buffer the UI can poll on each
rerun. Streamlit reruns the script on every interaction, so we stash
state on a module-level singleton rather than `st.session_state`
(which is per-session and would break if the browser tab reloads).
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Deque, Optional

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Uvicorn binds on all interfaces so other machines on the LAN can reach
# the server. Health checks + the canonical "self" URL still use loopback.
BIND_HOST = "0.0.0.0"
LOCAL_HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{LOCAL_HOST}:{PORT}"
RING_MAX = 1000


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
        r = httpx.get(f"{BASE_URL}/health", timeout=timeout)
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
    if is_running():
        return False, "already running"

    clear_log()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "src.server"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
    except Exception as e:
        return False, f"failed to launch: {e}"

    _STATE.proc = proc
    t = threading.Thread(target=_reader, args=(proc,), daemon=True)
    t.start()
    _STATE.reader = t
    return True, f"started (pid={proc.pid})"


def stop() -> tuple[bool, str]:
    p = _STATE.proc
    if p is None or p.poll() is not None:
        _STATE.proc = None
        return False, "not running"

    try:
        if sys.platform == "win32":
            p.send_signal(signal.CTRL_BREAK_EVENT)  # best effort
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=5)
    except Exception as e:
        return False, f"error stopping: {e}"

    _STATE.proc = None
    return True, "stopped"


def pid() -> Optional[int]:
    p = _STATE.proc
    if p is None or p.poll() is not None:
        return None
    return p.pid
