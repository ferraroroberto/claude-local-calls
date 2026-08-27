"""In-browser SSH terminal via app-launcher's session-host (#309, Step 3).

We deliberately **reuse** app-launcher's PTY/xterm engine rather than
rebuilding a ConPTY/WebSocket/xterm stack in the hub. app-launcher already
runs its session-host as a standalone loopback HTTP+WS service
(``python -m app.session_host.server``, default ``127.0.0.1:8446``) whose
``PtySession`` can spawn any ``cmd /c <exe> <flags>`` and stream it over a
WebSocket. The hub creates an ``ssh <user>@<host>`` session there and proxies
the WebSocket to the browser (the pump lives in the router, which owns the
client WebSocket); the hub applies its own auth via the existing middleware.

The session-host gates the spawned command through app-launcher's
``src/agents.py::AGENTS`` registry, which registers the ``ssh`` agent
(command ``ssh``, caller-supplied ``<user>@<host>`` flags) — so
:func:`create_ssh_session` relies on that registration already being in
place rather than working around its absence.

Windows-only, which is fine: the session-host is ConPTY-based and the hub
host (tower) is Windows.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import httpx

from src import remote_stats
from src.host_profile import HostProfile

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Same env var + default app-launcher's own webapp uses to find the
# session-host, so the two stay pointed at the same loopback service.
SESSION_HOST_PORT_ENV = "LAUNCHER_SESSION_HOST_PORT"
DEFAULT_SESSION_HOST_PORT = 8446

# The app-launcher agent id registered in src/agents.py::AGENTS. Kept as one
# named constant so the create call and any error text agree.
SSH_AGENT_ID = "ssh"

_PROBE_TIMEOUT_S = 2.0
_CREATE_TIMEOUT_S = 8.0


def session_host_base() -> str:
    """Loopback base URL of app-launcher's session-host."""
    port = os.environ.get(SESSION_HOST_PORT_ENV) or DEFAULT_SESSION_HOST_PORT
    return f"http://127.0.0.1:{port}"


def session_host_ws_url(session_id: str) -> str:
    """Upstream WebSocket URL for a session's stream (role=phone so the
    session-host honours our resize frames — role=pc is the mirror side)."""
    port = os.environ.get(SESSION_HOST_PORT_ENV) or DEFAULT_SESSION_HOST_PORT
    return f"ws://127.0.0.1:{port}/sessions/{session_id}/ws?role=phone"


def _upstream_detail(r: "httpx.Response") -> str:
    """Best-effort extraction of the session-host's own error text.

    FastAPI's ``HTTPException`` responses are ``{"detail": "..."}``; fall
    back to raw response text (truncated) for anything else so a caller
    always sees the real upstream reason instead of a guessed one."""
    try:
        body = r.json()
        detail = body.get("detail") if isinstance(body, dict) else None
        if detail:
            return str(detail)
    except Exception:  # noqa: BLE001 — non-JSON body
        pass
    return r.text.strip()[:200]


async def terminal_status() -> Dict[str, Any]:
    """Is the in-browser SSH terminal available on this host?

    Probes the session-host ``/healthz``. Returns
    ``{available, reason, session_host}``. ``available`` only means the
    engine is reachable — a session create can still fail for other reasons
    (see :func:`create_ssh_session`)."""
    base = session_host_base()
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S) as client:
            r = await client.get(f"{base}/healthz")
        if r.status_code < 500:
            return {"available": True, "reason": "", "session_host": base}
        return {"available": False, "reason": f"session-host HTTP {r.status_code}", "session_host": base}
    except Exception as exc:  # noqa: BLE001 — network / connection
        logger.debug("session-host probe failed: %s", exc)
        return {
            "available": False,
            "reason": "app-launcher session-host not reachable on this host",
            "session_host": base,
        }


async def create_ssh_session(
    host: HostProfile, *, cols: int = 120, rows: int = 30
) -> Dict[str, Any]:
    """Create an ``ssh <user>@<host>`` PTY session on the session-host.

    Returns ``{ok, session_id, error}``. Any 4xx/5xx from the session-host
    surfaces its own ``detail`` (or, failing that, raw response text) rather
    than a guessed cause — see :func:`_upstream_detail`."""
    if not host.can_ssh:
        return {"ok": False, "session_id": None, "error": "host has no SSH target configured"}
    # LAN address while it answers, tailnet name when it doesn't (#396).
    address = await remote_stats.dial_address_async(host)
    target = f"{host.ssh_user}@{address}"
    payload = {
        "kind": "pty",
        "agent": SSH_AGENT_ID,
        "flags": target,
        "project_dir": str(PROJECT_ROOT),
        "name": f"ssh {host.display_name or host.id}",
        "label": host.display_name or host.id,
        "cols": cols,
        "rows": rows,
    }
    base = session_host_base()
    try:
        async with httpx.AsyncClient(timeout=_CREATE_TIMEOUT_S) as client:
            r = await client.post(f"{base}/sessions", json=payload)
    except Exception as exc:  # noqa: BLE001 — network / connection
        logger.warning("⚠️ ssh session create failed: %s", exc)
        return {"ok": False, "session_id": None, "error": "session-host not reachable"}
    if r.status_code >= 400:
        detail = _upstream_detail(r)
        error = f"session-host HTTP {r.status_code}"
        if detail:
            error = f"{error}: {detail}"
        return {"ok": False, "session_id": None, "error": error}
    try:
        body = r.json()
        sid = body.get("session_id") or body.get("id")
    except Exception:  # noqa: BLE001 — bad JSON
        sid = None
    if not sid:
        return {"ok": False, "session_id": None, "error": "session-host returned no session id"}
    return {"ok": True, "session_id": sid, "error": ""}
