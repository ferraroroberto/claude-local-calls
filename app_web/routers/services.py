"""Services tab API — Docker engine + Langfuse stack health & launch.

Surfaces the host-side ``src.services`` (Docker/Langfuse), ``src.remote_stats``
(hub-peer probing, #533) and ``src.agentsview_service`` (AgentsView, #533)
helpers behind these endpoints:

  * ``GET  /admin/api/services/status`` — quick combined probe used by
    the Hub tab's services card; polls every few seconds while the Hub
    tab is active.
  * ``POST /admin/api/services/launch`` — start Docker Desktop (if
    down) then run ``start_langfuse.bat`` / ``.sh``. Returns a final
    step log when the chain settles — the request can take ~30-90 s on
    a cold start. Idempotent: a no-op when both services are already up.
  * ``POST /admin/api/services/{docker,langfuse,agentsview}/start`` and
    ``.../stop`` (issue #284) — per-service controls mirroring the
    Models tab's per-model start/stop, for the Services card's
    individual row buttons. All idempotent.

Every endpoint is loopback-bypass-safe (the bearer-token middleware
exempts loopback) so the SPA can call them without an admin token on
``127.0.0.1``.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Awaitable, Dict

from fastapi import APIRouter

from src import agentsview_service, remote_stats
from src import services as svc
from src.host_profile import resolve as resolve_host

logger = logging.getLogger(__name__)
router = APIRouter()


async def _service_action(
    path: str, label: str, verb: str, coro: Awaitable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run one service start/stop/launch coroutine and log its outcome
    uniformly (#530: was seven repeats of the same five-line body — await,
    log an emoji + ``result["steps"]`` keyed on ``result["ok"]``, return it
    — mirroring the SPA-side collapse already done for this exact shape in
    ``app_web/static/hub.js``'s ``onServiceAction``).
    """
    icon = "🛑" if verb == "stop" else "🚀"
    logger.info("%s /admin/api/services/%s", icon, path)
    result = await coro
    if result["ok"]:
        logger.info("✅ %s %s: %s", label, verb, result["steps"])
    else:
        logger.warning("⚠️ %s %s failed: %s", label, verb, result["steps"])
    return result


@router.get("/api/services/status")
async def services_status() -> Dict[str, Any]:
    """Combined Docker + Langfuse + hub-peer probe.

    All probes have a short timeout so the worst case is a few seconds
    when everything is down; in steady state each returns in <100 ms.
    """
    docker = await svc.docker_status()
    langfuse = await svc.langfuse_health()
    agentsview = await agentsview_service.agentsview_health()
    # Informational only (#179) — every other hub-running host (#372: was a
    # single hardcoded Mac Mini, now a generic peer list) gets probed in
    # parallel; the indicator exists to tell each peer's own story.
    active = resolve_host()
    peers = await remote_stats.hub_peers(active.id)

    # The "launch" button is only meaningful on platforms where we know
    # how to spawn Docker Desktop. Surface that to the SPA so the card
    # can render an explanatory line instead of a button on macOS/Linux.
    launchable = (
        sys.platform == "win32"
        and svc.find_docker_desktop() is not None
    )

    return {
        "docker": docker,
        "langfuse": langfuse,
        "agentsview": agentsview,
        "peers": peers,
        "launchable": launchable,
        "platform": sys.platform,
    }


@router.post("/api/services/agentsview/launch")
async def services_agentsview_launch() -> Dict[str, Any]:
    """Start the optional AgentsView server (issue #280). Returns a step log."""
    return await _service_action(
        "agentsview/launch", "agentsview", "launch", agentsview_service.launch_agentsview())


@router.post("/api/services/launch")
async def services_launch() -> Dict[str, Any]:
    """Start Docker Desktop + the Langfuse stack. Returns a step log."""
    return await _service_action("launch", "services", "launch", svc.launch_stack())


@router.post("/api/services/agentsview/stop")
async def services_agentsview_stop() -> Dict[str, Any]:
    """Stop AgentsView (issue #284). Returns a step log."""
    return await _service_action(
        "agentsview/stop", "agentsview", "stop", agentsview_service.stop_agentsview())


@router.post("/api/services/docker/start")
async def services_docker_start() -> Dict[str, Any]:
    """Start Docker Desktop only (issue #284). Returns a step log."""
    return await _service_action("docker/start", "docker", "start", svc.start_docker_desktop())


@router.post("/api/services/docker/stop")
async def services_docker_stop() -> Dict[str, Any]:
    """Stop Docker Desktop via its CLI (issue #284). Returns a step log."""
    return await _service_action("docker/stop", "docker", "stop", svc.stop_docker_desktop())


@router.post("/api/services/langfuse/start")
async def services_langfuse_start() -> Dict[str, Any]:
    """Start the Langfuse stack only, without touching Docker (issue #284)."""
    return await _service_action("langfuse/start", "langfuse", "start", svc.start_langfuse())


@router.post("/api/services/langfuse/stop")
async def services_langfuse_stop() -> Dict[str, Any]:
    """Stop the Langfuse stack (issue #284). Returns a step log."""
    return await _service_action("langfuse/stop", "langfuse", "stop", svc.stop_langfuse())
