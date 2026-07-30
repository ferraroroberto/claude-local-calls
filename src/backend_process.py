"""Per-model backend process manager.

Each enabled local model in the registry gets its own singleton process +
per-backend log file (``data/logs/backend-<id>.log``) here. Keyed by model
id ("qwen", "glm", "whisper"). Used by the admin SPA's Models tab, the
tray, and the per-model launcher scripts to start/stop individual backends
and tail their output without global-state entanglement. The log file is
written by the child (not a hub-owned pipe), so it stays readable across a
hub restart and an inherited backend never writes into a closed pipe.

Two engine families share this manager:
  - `llama-server` for chat/completion GGUF models (qwen, glm, gemma4*)
  - `whisper-server` for whisper.cpp ASR (OpenAI-compatible /v1/audio/*)
The shape differences (binary location, -m vs --model flag, health
endpoint) are absorbed in `build_command` and `is_reachable`.

Ownership semantics mirror :mod:`src.server_process` — see its module
docstring. ``start(model_id)`` adopts an already-reachable backend on
the model's port instead of spawning a duplicate; ``stop(model_id)``
only stops what we spawned. Use :func:`force_stop_external` to reclaim
a port held by someone else.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from .host_profile import resolve as resolve_host
from .http_client import get_sync_client
from .model_registry import Model, enabled_models, local_models, resolve as resolve_model
from .process_supervisor import ProcessSupervisor, SpawnSpec
from .server_process import (
    OWNERSHIP_NONE,
    WIN_NEW_GROUP,
    kill_pid,
    resolve_external_pid,
    resolve_ownership,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_LLAMA = PROJECT_ROOT / "vendor" / "llama.cpp"
VENDOR_WHISPER = PROJECT_ROOT / "vendor" / "whisper.cpp"
LOG_DIR = PROJECT_ROOT / "data" / "logs"
# Tail size returned by ``log_lines`` — replaces the old 1000-line ring.
LOG_TAIL_LINES = 400


def _log_path(model_id: str) -> Path:
    """Per-backend log file: ``data/logs/backend-<id>.log`` (child-owned)."""
    return LOG_DIR / f"backend-{model_id}.log"


def _roll_log(model_id: str) -> Path:
    """Roll the previous run's log to ``.log.1`` and return the fresh path.

    Bounds growth to two files per backend (current + one backup) without
    needing the hub to manage rotation mid-run: the child owns the fd, so
    we can only rotate at spawn time. Best-effort — a failed roll never
    blocks a launch.
    """
    path = _log_path(model_id)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        if path.exists():
            backup = path.with_suffix(".log.1")
            backup.unlink(missing_ok=True)
            path.replace(backup)
    except OSError:
        pass
    return path


def llama_server_binary() -> Path:
    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    return VENDOR_LLAMA / name


def whisper_server_binary() -> Path:
    name = "whisper-server.exe" if sys.platform == "win32" else "whisper-server"
    return VENDOR_WHISPER / name


def _is_whisper(model: Model) -> bool:
    return (
        model.engine == "whisper-server"
        or model.engine == "whisper-server-lazy"
        or model.backend == "whisper"
    )


def _is_lazy_whisper(model: Model) -> bool:
    return model.engine == "whisper-server-lazy"


def vendor_dir_for(model: Model) -> Path:
    return VENDOR_WHISPER if _is_whisper(model) else VENDOR_LLAMA


class _BackendState:
    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        # PID of a backend the hub inherited at startup — a process the
        # hub didn't itself spawn but recognises as one of its model
        # binaries listening on the right port. ``stop`` taskkill's it
        # instead of calling ``proc.terminate``. Its stdout still lands in
        # the per-backend log file (the child owns that fd), so ``log_lines``
        # returns its tail across a hub restart — see ``_log_path``.
        self.inherited_pid: Optional[int] = None


_STATES: Dict[str, _BackendState] = {}

# Set true by the admin /admin/api/hub/restart endpoint just before it
# signals the hub to exit. The shutdown handler reads it and SKIPS tearing
# the backend children down, so they survive the restart and the respawned
# hub re-adopts them via ``inherit_running_backends`` (shown as "running").
# Without this, a restart kills the very survivors inheritance exists to
# reclaim. Process-local: the respawned hub starts with it false.
_restart_pending = False


def set_restart_pending(value: bool = True) -> None:
    """Mark (or clear) that a hub restart is in flight — see ``_restart_pending``."""
    global _restart_pending
    _restart_pending = bool(value)


def restart_pending() -> bool:
    """True while a hub restart is in flight and backends must be left alive."""
    return _restart_pending


def _state_for(model_id: str) -> _BackendState:
    state = _STATES.get(model_id)
    if state is None:
        state = _BackendState()
        _STATES[model_id] = state
    return state


def is_running(model_id: str) -> bool:
    state = _state_for(model_id)
    p = state.proc
    if p is not None and p.poll() is None:
        return True
    return _inherited_alive(state)


def is_inherited(model_id: str) -> bool:
    """True iff this model is alive via an inherited PID (not a Popen we own)."""
    state = _state_for(model_id)
    return state.proc is None and _inherited_alive(state)


def inherited_foreign(model_id: str) -> bool:
    """True iff this model is alive via an inherited PID whose process has
    **no tie to this repo** — an external sibling's server on a mutex-shared
    port, adopted for control but never spawned by the hub (#431). The
    canonical case: voice-transcriber's own ``whisper-server`` holding :8090
    on the tower — the hub can route to it, but the fleet summary must not
    claim the hub runs it.

    Checks exe + command line + cwd for the repo path, not exe alone: a
    hub-spawned python shim (``tts_server``) resolves its exe to the *base*
    interpreter outside the repo (the venv redirector — one hub, two PIDs),
    but its command line names ``<repo>\\.venv\\Scripts\\python(w).exe -m
    src.tts_server``, so any repo-path mention marks it ours. Best-effort:
    an unreadable process (access denied, racing exit) reads as not-foreign
    rather than guessing.
    """
    state = _state_for(model_id)
    if not (state.proc is None and _inherited_alive(state)):
        return False
    try:
        import psutil

        proc = psutil.Process(state.inherited_pid)
        probes = [proc.exe() or ""]
        for getter in (proc.cmdline, proc.cwd):
            try:
                value = getter()
                probes.extend(value if isinstance(value, list) else [value or ""])
            except Exception:  # noqa: BLE001 — per-probe best-effort
                pass
        root = str(PROJECT_ROOT).lower().replace("/", "\\")
        return not any(
            root in str(p).lower().replace("/", "\\") for p in probes
        )
    except Exception:  # noqa: BLE001 — display-only hint, never block
        return False


def pid(model_id: str) -> Optional[int]:
    state = _state_for(model_id)
    p = state.proc
    if p is not None and p.poll() is None:
        return p.pid
    if _inherited_alive(state):
        return state.inherited_pid
    return None


def _inherited_alive(state: "_BackendState") -> bool:
    pid_ = state.inherited_pid
    if pid_ is None:
        return False
    try:
        import psutil

        if not psutil.pid_exists(pid_):
            state.inherited_pid = None
            return False
        # Verify it's still the same process — PID reuse on Windows is
        # aggressive; if the create-time has changed, our PID is stale.
        proc = psutil.Process(pid_)
        if proc.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
            state.inherited_pid = None
            return False
        return True
    except Exception:  # noqa: BLE001
        state.inherited_pid = None
        return False


def is_reachable(model: Model, timeout: float = 1.5) -> bool:
    if not model.url:
        return False
    # NB: strip the literal "/v1" suffix, not a character set. `str.rstrip`
    # takes a set of chars, so `"...:8091/v1".rstrip("/v1")` eats the port's
    # trailing "1" too and yields ":809" — a dead port. removesuffix is exact.
    base = model.url.removesuffix("/v1").rstrip("/")
    if model.engine in ("whisper-server", "whisper-server-lazy"):
        # whisper.cpp server has no /health; GET / returns 200 once loaded.
        # Engine-specific, not `_is_whisper` (backend == "whisper") — a
        # whisper-*shaped* backend on a different engine (e.g. Parakeet's
        # `engine: parakeet-server`, #138) has its own real /health route.
        try:
            r = get_sync_client().get(f"{base}/", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False
    try:
        r = get_sync_client().get(f"{base}/health", timeout=timeout)
        if r.status_code == 200:
            return True
    except Exception:
        pass
    # llama-server /v1/models is always available once loaded
    try:
        r = get_sync_client().get(f"{model.url}/models", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def probe_health(model: Model, timeout: float = 1.5) -> Optional[Dict[str, Any]]:
    """GET the backend's own ``/health`` and return the parsed JSON body, or
    ``None`` if unreachable / non-JSON.

    Unlike :func:`is_reachable` (a boolean liveness gate used everywhere),
    this is for callers that need a field out of the body itself — e.g. the
    admin Models tab reading ``tts_server.py``'s reported ``device``
    (cuda/cpu/mps) off a running TTS backend's ``/health``.
    """
    if not model.url:
        return None
    base = model.url.removesuffix("/v1").rstrip("/")
    try:
        r = httpx.get(f"{base}/health", timeout=timeout)
        if r.status_code == 200:
            body = r.json()
            return body if isinstance(body, dict) else None
    except Exception:
        pass
    return None


def log_lines(model_id: str, limit: int = LOG_TAIL_LINES) -> list[str]:
    """Tail of the backend's log file (``data/logs/backend-<id>.log``).

    Reads the file the child writes its stdout/stderr to, so it works for
    a backend we spawned *and* one we inherited across a hub restart (the
    child owns the fd). Returns ``[]`` if the backend has never started.
    """
    path = _log_path(model_id)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except FileNotFoundError:
        return []
    except OSError:
        return []
    return lines[-limit:] if limit else lines


def clear_log(model_id: str) -> None:
    """Truncate the backend's log file (no-op if it doesn't exist yet)."""
    path = _log_path(model_id)
    try:
        path.open("w", encoding="utf-8").close()
    except OSError:
        pass


def _whisper_boost_args(existing_args: list[str]) -> list[str]:
    """Return extra whisper-server args to enable vocabulary boosting (#91).

    When a whisper row opts into ``--carry-initial-prompt`` (which is only
    honoured with ``--max-context > 0`` — proven on v1.8.6), source the
    initial prompt from the committed dictionary's ``boost_terms`` so the
    boosting vocabulary lives in one place
    (``config/transcription_glossary.json``, shared with the #90
    replacement rules). No-op if boosting isn't requested or the row
    already supplies its own ``--prompt``.
    """
    if "--carry-initial-prompt" not in existing_args or "--prompt" in existing_args:
        return []
    from .transcription_glossary import load_boost_terms

    terms = load_boost_terms()
    if not terms:
        return []
    return ["--prompt", "Glossary: " + ", ".join(terms) + "."]


def build_command(model: Model) -> list[str]:
    # TTS rows run the in-repo FastAPI shim (src/tts_server), which owns the
    # heavy engine (Chatterbox in-process / Orpheus via a llama-server child).
    # Checked first: a chatterbox row has no model_path (weights come from the
    # HF cache), so it must skip the model_path guard below.
    if model.engine == "tts-server":
        cmd = [sys.executable, "-m", "src.tts_server", "--model-id", model.id]
        cmd.extend(model.args or [])
        return cmd

    # Parakeet (#138): the in-repo FastAPI shim (src/parakeet_server) owns a
    # persistent FluidAudio Swift subprocess. No model_path — the CoreML
    # weights are fetched/cached by FluidAudio itself on first load, not by
    # this repo's download_models.py, so it must skip the model_path guard
    # below same as the tts-server branch above.
    if model.engine == "parakeet-server":
        cmd = [sys.executable, "-m", "src.parakeet_server", "--model-id", model.id]
        cmd.extend(model.args or [])
        return cmd

    if not model.model_path:
        raise RuntimeError(f"model {model.id} has no model_path")
    model_path = (PROJECT_ROOT / model.model_path).resolve()

    if _is_lazy_whisper(model):
        # The proxy itself doesn't need the model on disk to start — it
        # only needs whisper-server present. We still surface a clear
        # error if the model is missing, since the first POST would fail.
        bin_path = whisper_server_binary()
        if not bin_path.exists():
            raise RuntimeError(
                f"whisper-server not found at {bin_path} - run scripts/install_whisper_cpp.py"
            )
        if not model_path.exists():
            raise RuntimeError(
                f"whisper model not found at {model_path} - run scripts/download_models.py --only {model.id}"
            )
        return [
            sys.executable, "-m", "src.whisper_translate_proxy",
            "--model-id", model.id,
        ]

    if _is_whisper(model):
        bin_path = whisper_server_binary()
        if not bin_path.exists():
            raise RuntimeError(
                f"whisper-server not found at {bin_path} - run scripts/install_whisper_cpp.py"
            )
        if not model_path.exists():
            raise RuntimeError(
                f"whisper model not found at {model_path} - run scripts/download_models.py --only {model.id}"
            )
        cmd = [
            str(bin_path),
            "--host", "0.0.0.0",
            "--port", str(model.port),
            "--model", str(model_path),
        ]
        args = list(model.args or [])
        cmd.extend(args)
        cmd.extend(_whisper_boost_args(args))
        return cmd

    bin_path = llama_server_binary()
    if not bin_path.exists():
        raise RuntimeError(f"llama-server not found at {bin_path} - run scripts/install_llama_cpp.py")
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
    # Chain-aware ownership guard (#342): any host in the model's ordered
    # ``hosts:`` chain may spawn it locally (the failover engine relies on
    # this to bring a model up on a fallback candidate); a host *outside*
    # the chain never does. A bare single-``host:`` row has a one-element
    # chain, so the pre-#342 refusal is byte-identical.
    chain = list(getattr(model, "hosts", None) or [])
    active = resolve_host()
    if chain and active.id not in chain:
        return False, (
            f"model {model_id!r} is owned by host(s) {chain!r} — "
            "start it there, or via the admin API which proxies this "
            "call automatically"
        )
    if is_running(model_id):
        return False, "already running"

    # Adopt an external instance already listening on this model's port.
    if is_reachable(model, timeout=0.4):
        ext = external_pid(model_id)
        suffix = f" (PID {ext})" if ext else ""
        return True, f"adopted external instance{suffix}"

    state = _state_for(model_id)

    try:
        cmd = build_command(model)
    except Exception as e:
        return False, str(e)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # Help the server find the cudart DLLs shipped next to its binary.
    if sys.platform == "win32":
        env["PATH"] = str(vendor_dir_for(model)) + os.pathsep + env.get("PATH", "")

    # Redirect stdout/stderr to a child-owned log file instead of a hub-owned
    # pipe. The child keeps its own fd, so the log survives a hub restart and
    # an inherited backend never writes into a closed pipe (the [Errno 22]
    # class that made #104 possible). The file is also readable on disk and
    # via ``log_lines`` / the admin log endpoint. Roll first for a fresh log.
    log_file = _roll_log(model_id).open("ab")

    def build_spawn_spec() -> SpawnSpec:
        return SpawnSpec(
            cmd=cmd,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=WIN_NEW_GROUP,
        )

    def on_spawned(proc: subprocess.Popen) -> None:
        # The child duplicated the handle at spawn; the hub no longer needs
        # its own copy (and keeping it open would pin the file). Closing it is
        # exactly what makes the log restart-safe — only the child holds the fd.
        log_file.close()

    try:
        return ProcessSupervisor(
            already_running=lambda: is_running(model_id),
            reachable=lambda: is_reachable(model, timeout=0.4),
            external_pid=lambda: external_pid(model_id),
            build_spawn_spec=build_spawn_spec,
            set_process=lambda proc: setattr(state, "proc", proc),
            on_spawned=on_spawned,
            adopt_message="adopted external instance",
        ).start()
    finally:
        if not log_file.closed:
            log_file.close()


def stop(model_id: str) -> tuple[bool, str]:
    state = _state_for(model_id)
    p = state.proc
    # Inherited backend: we don't hold a Popen handle, so polite shutdown
    # isn't an option — taskkill the PID directly.
    if p is None:
        if _inherited_alive(state):
            pid_ = state.inherited_pid
            state.inherited_pid = None
            ok, msg = kill_pid(int(pid_))
            return ok, msg
        return False, "not running"
    if p.poll() is not None:
        state.proc = None
        return False, "not running"
    ok, msg = ProcessSupervisor.stop_popen(
        p,
        terminate_timeout=8,
        kill_timeout=5,
    )
    if not ok:
        return ok, msg
    state.proc = None
    return True, "stopped"


def inherit_running_backends() -> int:
    """Adopt any model-backend process the hub finds on one of its ports.

    Called once at hub startup. Without this, a hub restart leaves the
    previous hub's children alive on their ports — the new hub sees
    them as "external" (adopted) and the UI shows the disabled-Stop
    state. With inheritance, the new hub treats them as ours, the UI
    shows them as running, and Stop force-kills the PID directly.

    Returns the number of backends inherited.
    """
    from .server_process import snapshot_listening_pids

    try:
        import psutil
    except ImportError:
        return 0

    listening = snapshot_listening_pids()
    count = 0
    for m in local_models():
        if m.backend not in ("openai", "whisper", "tts"):
            continue
        if not m.port:
            continue
        if is_running(m.id):
            continue  # already ours (Popen or earlier-inherited)
        pids = listening.get(m.port) or []
        if not pids:
            continue
        candidate = pids[0]
        try:
            proc = psutil.Process(candidate)
            exe = (proc.exe() or "").lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
        if _looks_like_backend_binary(exe, m):
            state = _state_for(m.id)
            state.inherited_pid = candidate
            count += 1
            import logging
            logging.getLogger(__name__).info(
                "📎 Inherited %s on :%s (PID %s) — log tail at %s",
                m.id, m.port, candidate, _log_path(m.id),
            )
    return count


def _looks_like_backend_binary(exe: str, model: "Model") -> bool:
    """Heuristic: does this executable look like the binary we'd spawn for ``model``?"""
    exe = (exe or "").lower()
    if model.engine in ("whisper-server", "whisper-server-lazy") or model.backend == "whisper":
        return "whisper-server" in exe or exe.endswith("whisper-server.exe")
    # Default: llama.cpp's llama-server. The lazy-whisper proxy runs as
    # ``python -m src.whisper_translate_proxy`` — recognise pythonw too.
    return (
        "llama-server" in exe
        or "python" in exe  # whisper_translate_proxy.py path
    )


def resolve_model_by_id(model_id: str) -> Optional[Model]:
    for m in enabled_models():
        if m.id == model_id:
            return m
    return None


def resolve_model_for_engine(model_id: str, expected_engine: str) -> Model:
    """Resolve ``model_id`` and assert it's wired to ``expected_engine``, or
    raise ``SystemExit`` with a message naming the mismatch.

    Shared by the shim servers each of which handles exactly one engine —
    ``parakeet_server.py`` (``parakeet-server``), ``tts_server.py``
    (``tts-server``), ``whisper_translate_proxy.py``
    (``whisper-server-lazy``). ``SystemExit`` (not a plain exception) because
    every caller is a ``build_app(model_id)`` bring-up path meant to abort
    the process on a config mismatch, not to be caught and handled.
    """
    model = resolve_model_by_id(model_id)
    if model is None:
        raise SystemExit(
            f"model {model_id!r} not enabled on this host — "
            f"add it to the host's enabled list in config/models.yaml"
        )
    if model.engine != expected_engine:
        raise SystemExit(
            f"model {model_id!r} has engine={model.engine!r}; "
            f"only engine={expected_engine} is supported here"
        )
    return model


def running_backends() -> Dict[str, Model]:
    """Return {model_id: Model} for each local backend whose process is alive."""
    out: Dict[str, Model] = {}
    for m in local_models():
        if m.backend in ("openai", "whisper", "tts") and is_running(m.id):
            out[m.id] = m
    return out


def ownership(model_id: str) -> str:
    """Tri-state ownership of the port for *model_id* — see server_process docstring."""
    model = resolve_model_by_id(model_id)
    if model is None or model.port is None:
        return OWNERSHIP_NONE
    return resolve_ownership(is_running(model_id), model.port)


def external_pid(model_id: str) -> Optional[int]:
    """PID holding *model_id*'s port if it isn't us, else ``None``."""
    model = resolve_model_by_id(model_id)
    if model is None or model.port is None:
        return None
    return resolve_external_pid(is_running(model_id), model.port)


def force_stop_external(model_id: str) -> tuple[bool, str]:
    """Force-kill whoever currently holds *model_id*'s port, if it's not us."""
    target = external_pid(model_id)
    if target is None:
        return False, "no external process on this model's port"
    return kill_pid(target)
