"""Per-model llama-server process manager (parallel to server_process.py).

Each enabled openai-backed model in the registry gets its own singleton
process + log ring buffer here. Keyed by model id ("qwen", "glm"). Used
by the Streamlit Models view to start/stop individual backends and tail
their output without any global state entanglement.
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
from typing import Deque, Dict, Optional

import httpx

from .model_registry import Model, enabled_models, resolve as resolve_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_LLAMA = PROJECT_ROOT / "vendor" / "llama.cpp"
RING_MAX = 1000


def _server_binary() -> Path:
    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    return VENDOR_LLAMA / name


class _BackendState:
    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.log: Deque[str] = deque(maxlen=RING_MAX)
        self.lock = threading.Lock()
        self.reader: Optional[threading.Thread] = None


_STATES: Dict[str, _BackendState] = {}


def _state_for(model_id: str) -> _BackendState:
    state = _STATES.get(model_id)
    if state is None:
        state = _BackendState()
        _STATES[model_id] = state
    return state


def is_running(model_id: str) -> bool:
    p = _state_for(model_id).proc
    return p is not None and p.poll() is None


def pid(model_id: str) -> Optional[int]:
    p = _state_for(model_id).proc
    if p is None or p.poll() is not None:
        return None
    return p.pid


def is_reachable(model: Model, timeout: float = 1.5) -> bool:
    if not model.url:
        return False
    base = model.url.rstrip("/v1").rstrip("/")
    try:
        r = httpx.get(f"{base}/health", timeout=timeout)
        if r.status_code == 200:
            return True
    except Exception:
        pass
    # llama-server /v1/models is always available once loaded
    try:
        r = httpx.get(f"{model.url}/models", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def log_lines(model_id: str) -> list[str]:
    state = _state_for(model_id)
    with state.lock:
        return list(state.log)


def clear_log(model_id: str) -> None:
    state = _state_for(model_id)
    with state.lock:
        state.log.clear()


def _reader(state: _BackendState, proc: subprocess.Popen) -> None:
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        with state.lock:
            state.log.append(line)


def build_command(model: Model) -> list[str]:
    bin_path = _server_binary()
    if not bin_path.exists():
        raise RuntimeError(f"llama-server not found at {bin_path} - run scripts/install_llama_cpp.py")
    if not model.model_path:
        raise RuntimeError(f"model {model.id} has no model_path")
    model_path = (PROJECT_ROOT / model.model_path).resolve()
    if not model_path.exists():
        raise RuntimeError(f"GGUF not found at {model_path} - run scripts/download_models.py --only {model.id}")
    cmd = [
        str(bin_path),
        "-m", str(model_path),
        "--host", "0.0.0.0",
        "--port", str(model.port),
    ]
    cmd.extend(model.args or [])
    return cmd


def start(model_id: str) -> tuple[bool, str]:
    model = resolve_model_by_id(model_id)
    if model is None:
        return False, f"model {model_id!r} not enabled on this host"
    if is_running(model_id):
        return False, "already running"

    state = _state_for(model_id)
    clear_log(model_id)

    try:
        cmd = build_command(model)
    except Exception as e:
        return False, str(e)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # Help llama-server find the cudart DLLs shipped next to it.
    if sys.platform == "win32":
        env["PATH"] = str(VENDOR_LLAMA) + os.pathsep + env.get("PATH", "")

    try:
        proc = subprocess.Popen(
            cmd,
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

    state.proc = proc
    t = threading.Thread(target=_reader, args=(state, proc), daemon=True)
    t.start()
    state.reader = t
    return True, f"started (pid={proc.pid})"


def stop(model_id: str) -> tuple[bool, str]:
    state = _state_for(model_id)
    p = state.proc
    if p is None or p.poll() is not None:
        state.proc = None
        return False, "not running"
    try:
        if sys.platform == "win32":
            try:
                p.send_signal(signal.CTRL_BREAK_EVENT)
            except Exception:
                pass
        p.terminate()
        try:
            p.wait(timeout=8)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=5)
    except Exception as e:
        return False, f"error stopping: {e}"
    state.proc = None
    return True, "stopped"


def resolve_model_by_id(model_id: str) -> Optional[Model]:
    for m in enabled_models():
        if m.id == model_id:
            return m
    return None


def running_backends() -> Dict[str, Model]:
    """Return {model_id: Model} for each backend whose process is alive."""
    out: Dict[str, Model] = {}
    for m in enabled_models():
        if m.backend == "openai" and is_running(m.id):
            out[m.id] = m
    return out
