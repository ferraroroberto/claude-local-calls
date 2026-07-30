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
lifecycle).
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
from urllib.parse import urlparse

from src.host_profile import HostProfile, hub_port
from src.http_client import get_async_client
from src.no_window import NO_WINDOW
from src.observability import langfuse_host

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCKER_PROBE_TIMEOUT_S = 2.0
LANGFUSE_PROBE_TIMEOUT_S = 2.0
# /admin/api/models on the remote peer health-probes every local model it
# owns before responding (observed 2-5.5s under normal load), so this needs
# real headroom above that, not just network RTT — this only guards the
# admin Models-tab merge poll, not any request hot path.
REMOTE_HUB_PROBE_TIMEOUT_S = 8.0

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


# ------------------------------------------------------------ remote hosts


async def remote_models(
    owner: HostProfile, timeout_s: float = REMOTE_HUB_PROBE_TIMEOUT_S
) -> Optional[List[Dict[str, Any]]]:
    """GET ``{owner's hub}/admin/api/models`` — used to merge a remote
    host's own model rows into this hub's Models tab (#178).

    Returns ``None`` (not ``[]``) on any failure — lets the caller tell
    "peer unreachable" apart from "peer reachable, reports zero models"
    and fall back to a locally-synthesized offline row instead of
    silently dropping the model from the list.
    """
    from src import remote_stats
    from src.remote_proxy import remote_auth_token

    address = await remote_stats.dial_address_async(owner)
    if not address:
        return None
    base = f"http://{address}:{hub_port()}"
    token = remote_auth_token(owner.id)
    headers = {"Authorization": f"Bearer {token}"} if token else None
    try:
        # local_only=true: two bidirectionally cross-enabled hosts would
        # otherwise recurse into each other's /api/models forever.
        # Shared pooled client — see langfuse_health() (#165/#392).
        r = await get_async_client().get(
            f"{base}/admin/api/models",
            params={"local_only": "true"},
            headers=headers,
            timeout=timeout_s,
        )
        if r.status_code >= 400:
            return None
        body = r.json()
        rows = body.get("models") if isinstance(body, dict) else None
        return rows if isinstance(rows, list) else None
    except Exception:  # noqa: BLE001 — network / connection / DNS / bad JSON
        return None


async def peer_health(
    host_id: str, timeout_s: float = REMOTE_HUB_PROBE_TIMEOUT_S
) -> Dict[str, Any]:
    """Probe any hub-running peer host's own hub `/health` endpoint (#179,
    generalized from the Mac-Mini-only original in #372).

    Clone of ``langfuse_health()``'s try/timeout shape, but the address
    comes from ``remote_stats.dial_address`` (the host's LAN ``address:``,
    falling back to its ``tailscale:`` name when the LAN path is dead — #396)
    + ``hub_port()`` — the same single source of truth #178's remote proxy
    already resolves against, not a new env var. When reachable, also compares build
    identity against the peer's ``/admin/api/version`` (#181) — its own
    try/except so a reachable-but-erroring version fetch never flips
    ``reachable`` back to ``False``.
    """
    from src import remote_stats
    from src.build_info import git_sha
    from src.host_profile import get_host

    owner = get_host(host_id)
    address = await remote_stats.dial_address_async(owner) if owner is not None else None
    if owner is None or not address:
        return {"reachable": False, "error": f"host {host_id!r} has no address configured", "address": None}
    base = f"http://{address}:{hub_port()}"
    try:
        # Shared pooled client — see langfuse_health() (#165/#392).
        r = await get_async_client().get(f"{base}/health", timeout=timeout_s)
        result: Dict[str, Any] = {
            "reachable": r.status_code < 500,
            "status_code": r.status_code,
            "error": "" if r.status_code < 500 else f"HTTP {r.status_code}",
            "address": base,
        }
    except Exception as exc:  # noqa: BLE001 — network / connection / DNS
        return {
            "reachable": False,
            "status_code": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "address": base,
        }

    local_sha = git_sha()
    result["local_git_sha"] = local_sha
    result["remote_git_sha"] = None
    result["git_sha_match"] = None
    if result["reachable"]:
        try:
            v = await get_async_client().get(f"{base}/admin/api/version", timeout=timeout_s)
            remote_sha = v.json().get("git_sha") if v.status_code < 500 else None
            result["remote_git_sha"] = remote_sha
            result["git_sha_match"] = (
                remote_sha is not None
                and remote_sha != "unknown"
                and local_sha != "unknown"
                and remote_sha == local_sha
            )
        except Exception:  # noqa: BLE001 — version probe is best-effort
            pass
    return result


async def hub_peers(active_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every other hub-running host's reachability + build identity (#372).

    Generalizes the old Mac-Mini-only Services-card probe into a peer list:
    every declared host besides ``active_id`` that runs its own hub — has at
    least one launchable local model, the same test the fleet placement grid
    already applies per host row (``model_registry.hub_peer_ids``) — is
    probed in parallel via :func:`peer_health`. Drives the Services card's
    per-peer status/detail/Wake/Sync rows; a future satellite with a
    non-empty ``enabled:`` list appears here automatically, no code change.
    """
    from src.host_profile import get_host, resolve as resolve_host
    from src.model_registry import hub_peer_ids

    active = active_id if active_id is not None else resolve_host().id
    peer_ids = hub_peer_ids(active)

    async def _one(host_id: str) -> Dict[str, Any]:
        owner = get_host(host_id)
        health = await peer_health(host_id)
        return {
            "host_id": host_id,
            "display_name": (owner.display_name if owner else None) or host_id,
            **health,
        }

    return list(await asyncio.gather(*(_one(hid) for hid in peer_ids)))


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


async def _poll_until(
    check: Callable[[], Awaitable[bool]],
    timeout_s: float,
    poll_s: float,
) -> bool:
    """Poll ``check()`` until it returns ``True`` or the budget expires.

    Shared "poll until ready" loop for ``wait_for_docker`` /
    ``wait_for_langfuse`` / ``wait_for_agentsview`` — each used to carry its
    own near-identical copy differing only in which status coroutine to
    await and which readiness key to check; that difference now lives in
    the caller's closure.
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

    return await _poll_until(_check, timeout_s, poll_s)


async def wait_for_langfuse(
    timeout_s: float = LANGFUSE_READY_TIMEOUT_S,
    poll_s: float = 3.0,
) -> bool:
    """Poll the Langfuse health endpoint until it responds < 500."""
    async def _check() -> bool:
        info = await langfuse_health(timeout_s=LANGFUSE_PROBE_TIMEOUT_S)
        return bool(info["reachable"])

    return await _poll_until(_check, timeout_s, poll_s)


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
    contract as :func:`launch_stack` / :func:`launch_agentsview` so the
    SPA renders all three identically. Factored out of :func:`launch_stack`
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


# ------------------------------------------------------------- agentsview
# Optional external AgentsView server (issue #280) — feeds the Code tab's
# AGY vendor. Same optional-service shape as Langfuse: short probe, launch
# helper, soft-fail everywhere. Never installed into the hub's .venv — the
# exe resolves from AGENTSVIEW_EXE, the dedicated .venv-agentsview/, or PATH.

AGENTSVIEW_PROBE_TIMEOUT_S = 2.0
# First-ever `agentsview serve` does a full index sync across every agent's
# session dirs before it starts listening (observed ~1-2 min on this host);
# steady-state restarts come up in seconds.
AGENTSVIEW_READY_TIMEOUT_S = 180.0


def agentsview_exe() -> Optional[str]:
    """Resolve the agentsview executable, or ``None`` when not installed.

    Order: ``AGENTSVIEW_EXE`` env → the repo-local dedicated venv
    (``.venv-agentsview/``, kept separate from the hub's own ``.venv`` per
    #280's isolation rule) → PATH (pipx install).
    """
    env = os.environ.get("AGENTSVIEW_EXE", "").strip()
    if env:
        return env if Path(env).exists() else None
    bin_dir = "Scripts" if sys.platform == "win32" else "bin"
    name = "agentsview.exe" if sys.platform == "win32" else "agentsview"
    local = PROJECT_ROOT / ".venv-agentsview" / bin_dir / name
    if local.exists():
        return str(local)
    return shutil.which("agentsview")


async def agentsview_health(
    timeout_s: float = AGENTSVIEW_PROBE_TIMEOUT_S,
) -> Dict[str, Any]:
    """Probe AgentsView's ``/api/ping``.

    Returns ``{reachable, status_code, error, host, version, installed}``.
    ``reachable`` requires the responder to identify as agentsview (the
    port drifts when :8080 is busy, so a foreign squatter must read as
    down, not up). ``installed`` reports whether the exe resolves — the
    Hub tab uses it to word the down-state hint.
    """
    from src.agentsview_usage import _base_url

    host = _base_url()
    installed = agentsview_exe() is not None
    if not host:
        return {
            "reachable": False,
            "status_code": 0,
            "error": "disabled (AGENTSVIEW_BASE_URL is empty)",
            "host": "",
            "version": "",
            "installed": installed,
        }
    try:
        # Shared pooled client — see langfuse_health() (#165/#392).
        r = await get_async_client().get(f"{host}/api/ping", timeout=timeout_s)
        body = r.json() if r.status_code < 500 else {}
        is_av = bool(body.get("ok")) and "agentsview" in str(body.get("service", ""))
        return {
            "reachable": is_av,
            "status_code": r.status_code,
            "error": "" if is_av else f"not agentsview (HTTP {r.status_code})",
            "host": host,
            "version": str(body.get("version") or ""),
            "installed": installed,
        }
    except Exception as exc:  # noqa: BLE001 — network / connection / DNS
        return {
            "reachable": False,
            "status_code": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "host": host,
            "version": "",
            "installed": installed,
        }


async def wait_for_agentsview(
    timeout_s: float = AGENTSVIEW_READY_TIMEOUT_S,
    poll_s: float = 3.0,
) -> bool:
    """Poll ``/api/ping`` until AgentsView responds (initial sync can be slow)."""
    async def _check() -> bool:
        info = await agentsview_health()
        return bool(info["reachable"])

    return await _poll_until(_check, timeout_s, poll_s)


def _spawn_agentsview(exe: str) -> None:
    """Start ``agentsview serve`` detached (same idiom as Docker Desktop).

    Telemetry and the update check are disabled in the child env — the hub
    launches a quiet, loopback-only indexer. ``DETACHED_PROCESS`` is
    deliberately omitted from the creation flags — see
    ``_spawn_docker_desktop`` for why.
    """
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | NO_WINDOW
    env = dict(os.environ)
    env.setdefault("AGENTSVIEW_TELEMETRY_ENABLED", "0")
    env.setdefault("AGENTSVIEW_DISABLE_UPDATE_CHECK", "1")
    subprocess.Popen(
        [exe, "serve"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        close_fds=True,
        env=env,
    )


async def launch_agentsview() -> Dict[str, Any]:
    """Start AgentsView if it isn't already serving.

    Returns the same ``{ok, steps}`` shape as :func:`launch_stack` so the
    SPA and the startup autostart log render it identically.
    """
    steps: List[Dict[str, str]] = []
    health = await agentsview_health()
    if health["reachable"]:
        steps.append({"name": "agentsview", "status": "skipped", "detail": "already serving"})
        return {"ok": True, "steps": steps}
    if not health["host"]:
        steps.append({
            "name": "agentsview",
            "status": "skipped",
            "detail": "disabled (AGENTSVIEW_BASE_URL is empty)",
        })
        return {"ok": True, "steps": steps}
    exe = agentsview_exe()
    if exe is None:
        steps.append({
            "name": "agentsview",
            "status": "error",
            "detail": (
                "agentsview not installed — see docs/code-usage-agentsview.md "
                "(.venv-agentsview or `pipx install agentsview`)"
            ),
        })
        return {"ok": False, "steps": steps}
    try:
        _spawn_agentsview(exe)
    except OSError as exc:
        steps.append({
            "name": "agentsview",
            "status": "error",
            "detail": f"spawn failed: {type(exc).__name__}: {exc}",
        })
        return {"ok": False, "steps": steps}
    ready = await wait_for_agentsview()
    if not ready:
        steps.append({
            "name": "agentsview",
            "status": "error",
            "detail": (
                f"spawned but /api/ping unreachable after "
                f"{AGENTSVIEW_READY_TIMEOUT_S:.0f}s — first run's initial index "
                "sync can be slow; it may still come up"
            ),
        })
        return {"ok": False, "steps": steps}
    steps.append({"name": "agentsview", "status": "ok", "detail": f"started {exe}"})
    return {"ok": True, "steps": steps}


async def stop_agentsview() -> Dict[str, Any]:
    """Stop AgentsView by killing whoever holds its port (issue #284).

    AgentsView is spawned detached with no PID tracked by the hub — same
    fire-and-forget shape as Docker Desktop — so the only reliable way to
    stop it later is by the port it's listening on, reusing
    ``server_process.find_port_pids`` / ``kill_pid``, the same port-based
    kill idiom ``force_stop_external`` already uses for adopted models.
    """
    from src.agentsview_usage import _base_url
    from src.server_process import find_port_pids, kill_pid

    host = _base_url()
    if not host:
        return {"ok": True, "steps": [{"name": "agentsview", "status": "skipped", "detail": "disabled (AGENTSVIEW_BASE_URL is empty)"}]}
    port = urlparse(host).port
    if not port:
        return {"ok": False, "steps": [{"name": "agentsview", "status": "error", "detail": f"could not parse a port from {host!r}"}]}

    pids = await asyncio.to_thread(find_port_pids, port)
    if not pids:
        return {"ok": True, "steps": [{"name": "agentsview", "status": "skipped", "detail": "already down"}]}

    failures = []
    for pid in pids:
        ok, msg = await asyncio.to_thread(kill_pid, pid)
        if not ok:
            failures.append(msg)
    if failures:
        return {"ok": False, "steps": [{"name": "agentsview", "status": "error", "detail": "; ".join(failures)[:200]}]}
    return {"ok": True, "steps": [{"name": "agentsview", "status": "ok", "detail": f"stopped pid(s) {', '.join(map(str, pids))}"}]}
