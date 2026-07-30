"""Machines console API (#309) — the fleet machine console behind the SPA's
Machines tab.

  * ``GET  /admin/api/machines/status`` — every enrolled machine as a probed
    card (reachability, uptime, actions, host stats). The tab's poll.
  * ``GET  /admin/api/machines/self`` — this machine's own detailed snapshot
    (CPU/RAM/GPU/disk/uptime/version). Also fetched cross-host by a peer's
    aggregator, so it rides the normal bearer-token/loopback middleware.
  * ``POST /admin/api/machines/{id}/reboot`` and ``.../shutdown`` —
    destructive power actions over the hub user's own general SSH (#311). The
    active hub host is refused (it is the excluded destructive case).
  * ``POST /admin/api/machines/{id}/wake`` — fire-and-forget Wake-on-LAN
    (#356) for a machine with a configured ``mac``. Unlike reboot/shutdown
    this legitimately targets a powered-down box, so it is not gated on
    reachability — only on having a MAC and not being the hub's own host.
  * ``GET  /admin/api/machines/{id}/rdp`` — download a generated ``.rdp``
    launcher for the machine's Remote-Desktop action.
  * ``GET  /admin/api/machines/terminal/status`` — is the in-browser SSH
    terminal engine (app-launcher's session-host) available here?
  * ``WS   /admin/api/machines/{id}/terminal`` — proxy an ``ssh`` PTY session
    from app-launcher's session-host to the browser (Step 3).

The read-only probes are loopback-bypass-safe like the other admin reads;
the power actions ride the normal bearer-token middleware (they trigger real
remote/host actions), matching ``hosts.py``'s bootstrap/sync stance. The
terminal proxy applies the *same* rules but has to do it in-route via
``middleware.authorize_websocket``: the middleware is a
``BaseHTTPMiddleware`` and so never runs for a ``websocket``-scope
connection.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

import websockets
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from starlette.responses import Response

from src import machine_console, remote_bootstrap, ssh_terminal
from src.host_profile import get_host, resolve
from src.wake_on_lan import WakeOnLanError, send_wake

from ..middleware import authorize_websocket

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/machines/status")
async def machines_status() -> Dict[str, Any]:
    """Every enrolled machine as a probed card (the Machines tab poll)."""
    return await machine_console.machines_status()


@router.get("/api/machines/self")
async def machine_self() -> Dict[str, Any]:
    """This machine's own detailed snapshot — used for its own card and
    fetched cross-host by a peer's aggregator."""
    return await machine_console.self_snapshot()


def _require_power_target(host_id: str):
    """Resolve a reboot/shutdown target or raise. Refuses the active hub
    host (the excluded destructive case) and SSH-less peers."""
    if host_id == resolve().id:
        raise HTTPException(status_code=400, detail="refusing to power-cycle the hub host")
    host = get_host(host_id)
    if host is None:
        raise HTTPException(status_code=404, detail=f"unknown machine {host_id!r}")
    if not host.can_ssh:
        raise HTTPException(status_code=400, detail=f"{host_id!r} has no SSH channel for power actions")
    return host


@router.post("/api/machines/{host_id}/reboot")
async def machine_reboot(host_id: str) -> Dict[str, Any]:
    _require_power_target(host_id)
    logger.info("♻️ /admin/api/machines/%s/reboot", host_id)
    result = await remote_bootstrap.reboot_host(host_id)
    if not result["ok"]:
        raise HTTPException(status_code=502, detail=result["detail"])
    return result


@router.post("/api/machines/{host_id}/shutdown")
async def machine_shutdown(host_id: str) -> Dict[str, Any]:
    _require_power_target(host_id)
    logger.info("⏻ /admin/api/machines/%s/shutdown", host_id)
    result = await remote_bootstrap.shutdown_host(host_id)
    if not result["ok"]:
        raise HTTPException(status_code=502, detail=result["detail"])
    return result


def _require_wake_target(host_id: str):
    """Resolve a wake target or raise. Unlike reboot/shutdown, wake
    legitimately targets a box that is powered down — no reachability
    check — but it still refuses the hub's own host (waking a box the hub
    is already running on is meaningless) and requires a configured `mac`."""
    if host_id == resolve().id:
        raise HTTPException(status_code=400, detail="refusing to wake the hub host")
    host = get_host(host_id)
    if host is None:
        raise HTTPException(status_code=404, detail=f"unknown machine {host_id!r}")
    if not host.mac:
        raise HTTPException(status_code=400, detail=f"{host_id!r} has no MAC address configured for wake")
    return host


@router.post("/api/machines/{host_id}/wake")
async def machine_wake(host_id: str) -> Dict[str, Any]:
    """Fire-and-forget Wake-on-LAN (#356) — no confirmation loop, no
    reachability polling; the caller only learns the packet was sent."""
    host = _require_wake_target(host_id)
    logger.info("🌅 /admin/api/machines/%s/wake", host_id)
    try:
        send_wake(host.mac)
    except WakeOnLanError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"ok": True, "sent": True}


@router.get("/api/machines/{host_id}/rdp")
async def machine_rdp(host_id: str) -> Response:
    """Download a generated ``.rdp`` launcher for the machine."""
    generated = machine_console.rdp_file(host_id)
    if generated is None:
        raise HTTPException(status_code=404, detail=f"{host_id!r} has no RDP target configured")
    filename, content = generated
    logger.info("🖥️ /admin/api/machines/%s/rdp", host_id)
    return Response(
        content=content,
        media_type="application/x-rdp",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/api/machines/terminal/status")
async def terminal_status() -> Dict[str, Any]:
    """Is the in-browser SSH terminal engine available on this host?"""
    return await ssh_terminal.terminal_status()


def _admin_token_getter(websocket: WebSocket):
    """Read the admin token off the sub-app's state, matching the accessor
    ``app_web/server.py`` hands :class:`BearerTokenMiddleware`. Deferred to
    call time (not handshake time) because ``webapp_config`` is reloaded on
    the app's state, so a cached value would go stale."""
    return lambda: getattr(
        getattr(websocket.app.state, "webapp_config", None), "auth_token", ""
    )


@router.websocket("/api/machines/{host_id}/terminal")
async def machine_terminal(websocket: WebSocket, host_id: str) -> None:
    """Proxy an ``ssh <user>@<host>`` PTY session (app-launcher's session-
    host) to the browser (#309, Step 3).

    The handshake is gated *before* ``accept()`` by
    ``middleware.authorize_websocket`` — ``BaseHTTPMiddleware`` (and so
    ``BearerTokenMiddleware``) only runs for ``http`` scope and never sees a
    websocket, so this route carries the check itself rather than inheriting
    it. Same trust rules as every other admin route (loopback bypass,
    proxied-loopback detection, ``extra_allowlist``, then bearer header or
    ``?token=`` — which is what the SPA already appends).

    Creates the session upstream, then pumps frames both ways until either
    side closes. On any setup failure the socket is accepted just long
    enough to send one JSON error frame, so the SPA can show why the
    terminal didn't open instead of a silent close."""
    if not await authorize_websocket(websocket, _admin_token_getter(websocket)):
        return
    await websocket.accept()
    host = get_host(host_id)
    if host is None or not host.can_ssh:
        await _close_with_error(websocket, "machine has no SSH terminal")
        return

    created = await ssh_terminal.create_ssh_session(host)
    if not created["ok"]:
        await _close_with_error(websocket, created["error"])
        return

    upstream_url = ssh_terminal.session_host_ws_url(created["session_id"])
    try:
        async with websockets.connect(upstream_url, max_size=None) as upstream:
            await _pump(websocket, upstream)
    except Exception as exc:  # noqa: BLE001 — connection / proxy failure
        logger.warning("⚠️ terminal proxy for %s failed: %s", host_id, exc)
        await _close_with_error(websocket, "terminal stream ended")


async def _close_with_error(websocket: WebSocket, message: str) -> None:
    """Best-effort: send one error frame, then close."""
    try:
        await websocket.send_json({"type": "error", "message": message})
    except Exception:  # noqa: BLE001 — socket may already be gone
        pass
    try:
        await websocket.close()
    except Exception:  # noqa: BLE001
        pass


async def _pump(client: WebSocket, upstream: "websockets.WebSocketClientProtocol") -> None:
    """Bidirectional relay between the browser and the session-host.

    Browser→upstream frames are JSON (``input``/``resize``) and upstream→
    browser frames are raw terminal text — the session-host's documented
    wire contract; we relay bytes/text verbatim without interpreting them."""

    async def browser_to_upstream() -> None:
        try:
            while True:
                msg = await client.receive_text()
                await upstream.send(msg)
        except (WebSocketDisconnect, Exception):  # noqa: BLE001
            return

    async def upstream_to_browser() -> None:
        try:
            async for msg in upstream:
                if isinstance(msg, bytes):
                    await client.send_bytes(msg)
                else:
                    await client.send_text(msg)
        except Exception:  # noqa: BLE001
            return

    done, pending = await asyncio.wait(
        {asyncio.create_task(browser_to_upstream()), asyncio.create_task(upstream_to_browser())},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
