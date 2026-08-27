"""Optional external AgentsView server process lifecycle (issue #280).

Feeds the Code tab's AGY vendor. Same optional-service shape as
``services.py``'s Langfuse handling: short probe, launch helper,
soft-fail everywhere. Never installed into the hub's own ``.venv`` — the
exe resolves from ``AGENTSVIEW_EXE``, the dedicated ``.venv-agentsview/``,
or PATH.

Split out of ``services.py`` (issue #533) — that module's docstring
advertised "Docker engine + Langfuse stack" while this whole AgentsView
lifecycle (probe/launch/stop) had also grown in there, unrelated to
either. Reuses :func:`src.services.poll_until` rather than duplicating
the "poll until ready" loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from src.http_client import get_async_client
from src.server_process import WIN_NEW_GROUP
from src.services import PROJECT_ROOT, poll_until

logger = logging.getLogger(__name__)


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
        # Shared pooled client — see services.langfuse_health() (#165/#392).
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

    return await poll_until(_check, timeout_s, poll_s)


def _spawn_agentsview(exe: str) -> None:
    """Start ``agentsview serve`` detached (same idiom as Docker Desktop).

    Telemetry and the update check are disabled in the child env — the hub
    launches a quiet, loopback-only indexer. ``DETACHED_PROCESS`` is
    deliberately omitted from the creation flags — see
    ``services._spawn_docker_desktop`` for why.
    """
    env = dict(os.environ)
    env.setdefault("AGENTSVIEW_TELEMETRY_ENABLED", "0")
    env.setdefault("AGENTSVIEW_DISABLE_UPDATE_CHECK", "1")
    subprocess.Popen(
        [exe, "serve"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=WIN_NEW_GROUP,
        close_fds=True,
        env=env,
    )


async def launch_agentsview() -> Dict[str, Any]:
    """Start AgentsView if it isn't already serving.

    Returns the same ``{ok, steps}`` shape as ``services.launch_stack`` so
    the SPA and the startup autostart log render it identically.
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
