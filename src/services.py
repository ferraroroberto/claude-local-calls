"""Host-side service helpers — Docker engine + Langfuse stack (issue #27).

The hub depends on a running Docker engine for the Langfuse observability
stack, but Docker Desktop is user-managed and silently down after a reboot
is a common failure mode. This module gives the admin SPA's Hub tab a way
to (a) tell the user that Docker / Langfuse are down and (b) bring them
back up with one button.

Everything here is best-effort and soft-failing: probes have short
timeouts, launches return structured step logs rather than raising, and
the Langfuse health probe degrades cleanly when the SDK / containers
are not present.

Sibling to ``server_process.py`` (hub-process lifecycle) and
``backend_process.py`` (per-model llama-server / whisper-server
lifecycle). The cross-host peer-probe trio (``remote_models`` /
``peer_health`` / ``hub_peers``) lives in ``remote_stats.py`` (issue
#533 — that module already owns peer I/O); the optional AgentsView
process lifecycle lives in ``agentsview_service.py``. :func:`poll_until`
is shared with that module's own ``wait_for_agentsview``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.http_client import get_async_client
from src.no_window import NO_WINDOW
from src.observability import langfuse_host

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCKER_PROBE_TIMEOUT_S = 2.0
LANGFUSE_PROBE_TIMEOUT_S = 2.0

# Used by POST /admin/api/services/launch — total budget for `docker info`
# to start succeeding after we spawn Docker Desktop. The engine usually
# comes up in 10-30 s on Windows; allow some slack.
DOCKER_READY_TIMEOUT_S = 90.0
# Same idea for Langfuse after `start_langfuse.bat` returns — image pulls
# already happened on first run, so steady-state is ~30 s.
LANGFUSE_READY_TIMEOUT_S = 90.0


# Windows install candidates for Docker Desktop. First-existing wins.
# Probe both Program Files and the per-user install location.
_WINDOWS_DOCKER_DESKTOP_CANDIDATES = (
    r"C:\Program Files\Docker\Docker\Docker Desktop.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Docker\Docker\Docker Desktop.exe"),
)


# ---------------------------------------------------------------- helpers


def find_docker_desktop() -> Optional[Path]:
    """Locate the Docker Desktop executable on this host, or None.

    Windows-only: macOS launches Docker via ``open -a Docker`` and Linux
    typically runs the engine under systemd with no GUI to launch.
    """
    if sys.platform != "win32":
        return None
    for candidate in _WINDOWS_DOCKER_DESKTOP_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            return p
    return None


def langfuse_start_script() -> Path:
    """Return the platform-appropriate start_langfuse script path."""
    if sys.platform == "win32":
        return PROJECT_ROOT / "start_langfuse.bat"
    return PROJECT_ROOT / "start_langfuse.sh"


def langfuse_stop_script() -> Path:
    """Return the platform-appropriate stop_langfuse script path."""
    if sys.platform == "win32":
        return PROJECT_ROOT / "stop_langfuse.bat"
    return PROJECT_ROOT / "stop_langfuse.sh"


# ---------------------------------------------------------------- docker


def _docker_info_sync(timeout_s: float) -> Dict[str, Any]:
    """Blocking half of :func:`docker_status`, run off-thread.

    ``asyncio.create_subprocess_exec`` has no Windows implementation
    under ``SelectorEventLoop`` (only ``ProactorEventLoop`` supports
    subprocess pipes there) — since #223 wired the hub's uvicorn to the
    selector loop, spawning ``docker info`` via the async subprocess API
    raises ``NotImplementedError`` on every call. A blocking
    ``subprocess.run`` in a worker thread sidesteps the event loop's
    subprocess transport entirely, so it works under either loop policy.
    """
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            timeout=timeout_s,
            # CREATE_NO_WINDOW on Windows so this poll (fired every few
            # seconds while the Hub tab is open) doesn't flash a console
            # window — matching system_stats.gpu_stats / claude_cli.
            creationflags=NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        return {"running": False, "error": f"`docker info` timed out after {timeout_s:.1f}s"}
    except OSError as exc:
        return {"running": False, "error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode == 0:
        version = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
        return {"running": True, "error": "", "server_version": version}
    # Daemon down — keep the first line of stderr for the UI.
    err = (proc.stderr or b"").decode("utf-8", errors="replace").strip().splitlines()
    first = err[0] if err else f"exit {proc.returncode}"
    return {"running": False, "error": first[:200]}


async def docker_status(timeout_s: float = DOCKER_PROBE_TIMEOUT_S) -> Dict[str, Any]:
    """Probe the Docker engine. Returns ``{running, error}``.

    Uses ``docker info`` with a short timeout. Treats both "docker
    binary missing" and "daemon pipe missing" as ``running=False`` —
    the SPA card only needs the binary state.
    """
    if shutil.which("docker") is None:
        return {"running": False, "error": "docker CLI not on PATH"}
    return await asyncio.to_thread(_docker_info_sync, timeout_s)


# ---------------------------------------------------------------- langfuse


async def langfuse_health(timeout_s: float = LANGFUSE_PROBE_TIMEOUT_S) -> Dict[str, Any]:
    """Probe Langfuse's public health endpoint.

    Returns ``{reachable, status_code, error, host}``. ``reachable`` is
    True only when the server returns < 500; auth keys are optional for
    the health endpoint itself.
    """
    host = langfuse_host()
    try:
        # Shared pooled client (#165/#392): a fresh AsyncClient builds an SSL
        # context (~0.26 s on Windows, cert-store scan) *on the event loop* —
        # the SPA polls this constantly, and those constructions stacked into
        # multi-second loop stalls that tripped 5 s e2e httpx timeouts.
        r = await get_async_client().get(f"{host}/api/public/health", timeout=timeout_s)
        return {
            "reachable": r.status_code < 500,
            "status_code": r.status_code,
            "error": "" if r.status_code < 500 else f"HTTP {r.status_code}",
            "host": host,
        }
    except Exception as exc:  # noqa: BLE001 — network / connection / DNS
        return {
            "reachable": False,
            "status_code": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "host": host,
        }


# ---------------------------------------------------------------- launch


def _spawn_docker_desktop(exe: Path) -> None:
    """Start Docker Desktop detached so it survives the request.

    Windows: CREATE_NEW_PROCESS_GROUP + CREATE_NO_WINDOW keeps it alive
    after the uvicorn worker that handled the launch request moves on and
    suppresses any console window. ``DETACHED_PROCESS`` is deliberately
    omitted — it's mutually exclusive with ``CREATE_NO_WINDOW`` per the
    Win32 CreateProcess docs, and combining them lets Windows Terminal
    (as the default terminal host) host a console window anyway.
    """
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | NO_WINDOW
    subprocess.Popen(
        [str(exe)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        close_fds=True,
    )


async def poll_until(
    check: Callable[[], Awaitable[bool]],
    timeout_s: float,
    poll_s: float,
) -> bool:
    """Poll ``check()`` until it returns ``True`` or the budget expires.

    Shared "poll until ready" loop for ``wait_for_docker`` /
    ``wait_for_langfuse`` (here) and ``agentsview_service.wait_for_agentsview``
    — each used to carry its own near-identical copy differing only in which
    status coroutine to await and which readiness key to check; that
    difference now lives in the caller's closure. Public (no leading
    underscore) because it is shared across module boundaries.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if await check():
            return True
        await asyncio.sleep(poll_s)
    return False


async def wait_for_docker(
    timeout_s: float = DOCKER_READY_TIMEOUT_S,
    poll_s: float = 2.0,
) -> bool:
    """Poll ``docker info`` until it succeeds or the budget expires."""
    async def _check() -> bool:
        info = await docker_status(timeout_s=DOCKER_PROBE_TIMEOUT_S)
        return bool(info["running"])

    return await poll_until(_check, timeout_s, poll_s)


async def wait_for_langfuse(
    timeout_s: float = LANGFUSE_READY_TIMEOUT_S,
    poll_s: float = 3.0,
) -> bool:
    """Poll the Langfuse health endpoint until it responds < 500."""
    async def _check() -> bool:
        info = await langfuse_health(timeout_s=LANGFUSE_PROBE_TIMEOUT_S)
        return bool(info["reachable"])

    return await poll_until(_check, timeout_s, poll_s)


def _run_langfuse_start_script_sync() -> Dict[str, Any]:
    """Blocking half of :func:`_run_langfuse_start_script`, run off-thread.

    Same ``SelectorEventLoop``-has-no-Windows-subprocess-support issue as
    :func:`_docker_info_sync` — a blocking ``subprocess.run`` in a worker
    thread avoids the event loop's subprocess transport entirely.
    """
    script = langfuse_start_script()
    if not script.exists():
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"start script not found: {script}",
        }
    if sys.platform == "win32":
        cmd = ["cmd.exe", "/c", str(script)]
    else:
        cmd = ["/bin/sh", str(script)]
    try:
        proc = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), capture_output=True, timeout=120.0,
            creationflags=NO_WINDOW,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or b"").decode("utf-8", errors="replace"),
            "stderr": (proc.stderr or b"").decode("utf-8", errors="replace"),
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "start_langfuse script timed out after 120 s",
        }
    except OSError as exc:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


async def _run_langfuse_start_script() -> Dict[str, Any]:
    """Run ``start_langfuse.{bat,sh}`` and capture the result.

    Returns ``{ok, returncode, stdout, stderr}``. The script itself is
    idempotent (``docker compose up -d``) so calling it again on an
    already-running stack is a fast no-op.
    """
    return await asyncio.to_thread(_run_langfuse_start_script_sync)


async def start_docker_desktop() -> Dict[str, Any]:
    """Start Docker Desktop if the engine is down.

    Returns ``{ok, steps}`` with a single ``docker_engine`` step — same
    contract as :func:`launch_stack` / ``agentsview_service.launch_agentsview``
    so the SPA renders all three identically. Factored out of :func:`launch_stack`
    (issue #284) so the Services card's individual Docker Start button can
    drive just this step without also touching Langfuse.
    """
    steps: List[Dict[str, str]] = []
    info = await docker_status()
    if info["running"]:
        steps.append({"name": "docker_engine", "status": "skipped", "detail": "engine already up"})
        return {"ok": True, "steps": steps}
    if sys.platform != "win32":
        steps.append({
            "name": "docker_engine",
            "status": "error",
            "detail": (
                "auto-launch is Windows-only — start Docker manually "
                "(`open -a Docker` on macOS, `sudo systemctl start docker` on Linux)"
            ),
        })
        return {"ok": False, "steps": steps}
    exe = find_docker_desktop()
    if exe is None:
        steps.append({
            "name": "docker_engine",
            "status": "error",
            "detail": (
                "Docker Desktop install not found in Program Files or LOCALAPPDATA — "
                "install it from docker.com/products/docker-desktop"
            ),
        })
        return {"ok": False, "steps": steps}
    try:
        _spawn_docker_desktop(exe)
    except OSError as exc:
        steps.append({
            "name": "docker_engine",
            "status": "error",
            "detail": f"spawn failed: {type(exc).__name__}: {exc}",
        })
        return {"ok": False, "steps": steps}
    ready = await wait_for_docker()
    if not ready:
        steps.append({
            "name": "docker_engine",
            "status": "error",
            "detail": (
                f"engine still not responsive after {DOCKER_READY_TIMEOUT_S:.0f}s — "
                "Docker Desktop may have shown a prompt; check the system tray"
            ),
        })
        return {"ok": False, "steps": steps}
    steps.append({
        "name": "docker_engine",
        "status": "ok",
        "detail": f"started Docker Desktop ({exe})",
    })
    return {"ok": True, "steps": steps}


async def start_langfuse() -> Dict[str, Any]:
    """Start the Langfuse stack if it isn't reachable.

    Returns ``{ok, steps}`` with a single ``langfuse_stack`` step. Does
    not start Docker itself — the start script fails fast with an
    actionable message if the engine is down. Factored out of
    :func:`launch_stack` (issue #284); see :func:`start_docker_desktop`.
    """
    steps: List[Dict[str, str]] = []
    health = await langfuse_health()
    if health["reachable"]:
        steps.append({"name": "langfuse_stack", "status": "skipped", "detail": "stack already up"})
        return {"ok": True, "steps": steps}

    result = await _run_langfuse_start_script()
    if not result["ok"]:
        # First line of stderr is usually the actionable bit.
        err_lines = [ln for ln in (result["stderr"] or "").splitlines() if ln.strip()]
        detail = err_lines[0][:200] if err_lines else f"exit {result['returncode']}"
        steps.append({"name": "langfuse_stack", "status": "error", "detail": detail})
        return {"ok": False, "steps": steps}
    ready = await wait_for_langfuse()
    if not ready:
        steps.append({
            "name": "langfuse_stack",
            "status": "error",
            "detail": (
                f"containers started but /api/public/health unreachable after "
                f"{LANGFUSE_READY_TIMEOUT_S:.0f}s — check `docker compose -f "
                "docker/langfuse/docker-compose.yml ps` for container errors"
            ),
        })
        return {"ok": False, "steps": steps}

    steps.append({
        "name": "langfuse_stack",
        "status": "ok",
        "detail": "containers started and health endpoint responding",
    })
    return {"ok": True, "steps": steps}


async def launch_stack() -> Dict[str, Any]:
    """End-to-end recovery: start Docker Desktop if down, then Langfuse.

    Returns ``{ok, steps: [{name, status, detail}]}``. The first error
    short-circuits the chain — Langfuse is never attempted if Docker
    didn't come up.
    """
    docker_result = await start_docker_desktop()
    if not docker_result["ok"]:
        return docker_result
    langfuse_result = await start_langfuse()
    return {
        "ok": langfuse_result["ok"],
        "steps": docker_result["steps"] + langfuse_result["steps"],
    }


# ------------------------------------------------------------------ stop
# Issue #284 — the Services card's Start-only buttons get Stop siblings,
# mirroring the Models tab's per-row start/stop pattern.

DOCKER_STOP_TIMEOUT_S = 60.0
LANGFUSE_STOP_TIMEOUT_S = 60.0


def _stop_docker_desktop_sync(timeout_s: float) -> Dict[str, Any]:
    """Blocking half of :func:`stop_docker_desktop`, run off-thread.

    Uses the official ``docker desktop stop`` CLI (bundled with recent
    Docker Desktop releases — confirmed present via ``docker desktop
    --help``) rather than killing the process tree, so the WSL2 VM gets
    a clean shutdown instead of an unclean kill.
    """
    try:
        proc = subprocess.run(
            ["docker", "desktop", "stop"],
            capture_output=True,
            timeout=timeout_s,
            creationflags=NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "detail": f"`docker desktop stop` timed out after {timeout_s:.0f}s"}
    except OSError as exc:
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    if proc.returncode == 0:
        return {"ok": True, "detail": "Docker Desktop stopped"}
    err = (proc.stderr or b"").decode("utf-8", errors="replace").strip().splitlines()
    first = err[0] if err else f"exit {proc.returncode}"
    return {"ok": False, "detail": first[:200]}


async def stop_docker_desktop(timeout_s: float = DOCKER_STOP_TIMEOUT_S) -> Dict[str, Any]:
    """Stop Docker Desktop via its CLI. Returns ``{ok, steps}`` (one step).

    Stopping Docker also takes the Langfuse containers down with it —
    they can't run without the engine — so the Services card will show
    both as down on its next poll.
    """
    info = await docker_status()
    if not info["running"]:
        return {"ok": True, "steps": [{"name": "docker_engine", "status": "skipped", "detail": "already down"}]}
    result = await asyncio.to_thread(_stop_docker_desktop_sync, timeout_s)
    status = "ok" if result["ok"] else "error"
    return {"ok": result["ok"], "steps": [{"name": "docker_engine", "status": status, "detail": result["detail"]}]}


def _run_langfuse_stop_script_sync() -> Dict[str, Any]:
    """Blocking half of :func:`stop_langfuse`, run off-thread. Sibling of
    :func:`_run_langfuse_start_script_sync`."""
    script = langfuse_stop_script()
    if not script.exists():
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"stop script not found: {script}",
        }
    if sys.platform == "win32":
        cmd = ["cmd.exe", "/c", str(script)]
    else:
        cmd = ["/bin/sh", str(script)]
    try:
        proc = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), capture_output=True, timeout=LANGFUSE_STOP_TIMEOUT_S,
            creationflags=NO_WINDOW,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or b"").decode("utf-8", errors="replace"),
            "stderr": (proc.stderr or b"").decode("utf-8", errors="replace"),
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"stop_langfuse script timed out after {LANGFUSE_STOP_TIMEOUT_S:.0f}s",
        }
    except OSError as exc:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


async def stop_langfuse() -> Dict[str, Any]:
    """Stop the Langfuse stack (``docker compose ... down``).

    Returns ``{ok, steps}`` (one ``langfuse_stack`` step). Idempotent —
    a no-op on an already-stopped stack.
    """
    health = await langfuse_health()
    if not health["reachable"]:
        return {"ok": True, "steps": [{"name": "langfuse_stack", "status": "skipped", "detail": "already down"}]}
    result = await asyncio.to_thread(_run_langfuse_stop_script_sync)
    if not result["ok"]:
        err_lines = [ln for ln in (result["stderr"] or "").splitlines() if ln.strip()]
        detail = err_lines[0][:200] if err_lines else f"exit {result['returncode']}"
        return {"ok": False, "steps": [{"name": "langfuse_stack", "status": "error", "detail": detail}]}
    return {"ok": True, "steps": [{"name": "langfuse_stack", "status": "ok", "detail": "containers stopped"}]}
