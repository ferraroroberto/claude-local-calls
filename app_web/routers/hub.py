"""Hub tab API — status, control, live request stream, log tail.

Endpoints (all under /admin/api/hub):
  * GET  /status            — pid, uptime, local/lan URLs, build identity
  * POST /stop              — graceful shutdown (the page will then 502)
  * POST /restart           — spawn a watchdog that respawns ``src.server``
  * GET  /log/tail          — SSE stream of root-logger lines
  * GET  /log/recent        — non-SSE seed (last N lines)
  * GET  /stats             — 5-minute ring of RAM/CPU/GPU samples (sparklines)
  * GET  /requests/stream   — SSE stream of every routed /v1/* request
  * GET  /requests/recent   — non-SSE seed (last N records)
  * GET  /errors/recent     — non-2xx ring
  * GET  /counters          — per-backend counters since hub start

The old install tab's endpoints (``/admin/api/install/*``) split into their
own ``install.py`` router (issue #533) — see that module. The platform
process-supervision helpers behind /stop and /restart (self-signal,
launchd/systemd, respawn-watchdog spawn) likewise moved to
``src/hub_process_control.py`` — they raise no FastAPI ``HTTPException``, so
per ``app_web/admin_forward.py``'s layering rule they don't belong in
``app_web``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from src.hub_log import HUB_LOG
from src.hub_observability import OBS
from src.hub_process_control import (
    _delayed_darwin_bootout,
    _delayed_shutdown,
    _delayed_systemctl,
    _spawn_respawn_watchdog,
    _under_systemd,
)
from src.server_process import lan_ip

from ._helpers import sse_stream

logger = logging.getLogger(__name__)
router = APIRouter()


# ----------------------------------------------------------------- helpers


def _hub_port() -> int:
    from src.host_profile import hub_port

    return int(hub_port())


# ---------------------------------------------------------------- status

@router.get("/api/hub/status")
async def hub_status(request: Request) -> Dict[str, Any]:
    from src.host_profile import resolve as resolve_host

    port = _hub_port()
    lan = lan_ip()
    uptime_s = max(0.0, time.time() - OBS.started_at())
    return {
        "running": True,  # we ARE the hub — if you can read this, it's up
        "pid": os.getpid(),
        "port": port,
        "local_url": f"http://127.0.0.1:{port}",
        "lan_url": f"http://{lan}:{port}" if lan else "",
        "started_at": OBS.started_at(),
        "uptime_s": round(uptime_s, 1),
        "host": resolve_host().id,
    }


# ---------------------------------------------------------------- control
# The self-signal / launchd / systemd / respawn-watchdog helpers these two
# endpoints drive now live in src/hub_process_control.py (issue #533) — pure
# process supervision, no FastAPI.

@router.post("/api/hub/stop")
async def hub_stop() -> Dict[str, Any]:
    if sys.platform == "darwin":
        from src.install import LAUNCHAGENT_LABEL

        logger.info("🛑 /admin/api/hub/stop — launchctl bootout (unload; a signaled exit alone respawns under KeepAlive)")
        _delayed_darwin_bootout(LAUNCHAGENT_LABEL)
        return {"ok": True, "detail": "hub will exit shortly (LaunchAgent unloaded)"}

    if _under_systemd():
        # Restart=always respawns a bare self-SIGTERM, so a deliberate stop must
        # go through systemd itself (the systemd analogue of the Mac's bootout).
        logger.info("🛑 /admin/api/hub/stop — systemctl stop (Restart=always respawns a bare signal)")
        _delayed_systemctl("stop")
        return {"ok": True, "detail": "hub will stop shortly (systemctl stop)"}

    logger.info("🛑 /admin/api/hub/stop — scheduling self-shutdown")
    _delayed_shutdown()
    return {"ok": True, "detail": "hub will exit shortly"}


@router.post("/api/hub/restart")
async def hub_restart() -> Dict[str, Any]:
    # Tell the shutdown handler to leave the model backends running so the
    # respawned hub adopts them, instead of killing the survivors that
    # inherit_running_backends() exists to reclaim.
    from src import backend_process as bp

    bp.set_restart_pending(True)

    if sys.platform == "darwin":
        # On darwin the LaunchAgent (#181) is the sole supervisor — its
        # KeepAlive.SuccessfulExit=false only respawns on an *abnormal*
        # exit, so spawning our own detached respawn-watchdog here would
        # race it: two processes competing for the same port. Instead, ask
        # launchd itself to kill+relaunch the job; no self-exit needed,
        # launchd already owns that half.
        from src.install import LAUNCHAGENT_LABEL

        logger.info("🔄 /admin/api/hub/restart — launchctl kickstart")
        subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{LAUNCHAGENT_LABEL}"],
            capture_output=True,
        )
        return {"ok": True, "detail": "hub will restart shortly via launchd"}

    if _under_systemd():
        # systemd is the sole supervisor here — like launchd on the Mac, drive
        # the restart through it rather than racing a self-spawned watchdog
        # against Restart=always for the same port. restart_pending (set above)
        # keeps model backends alive for the respawned hub to adopt.
        logger.info("🔄 /admin/api/hub/restart — systemctl restart")
        _delayed_systemctl("restart")
        return {"ok": True, "detail": "hub will restart shortly via systemd"}

    logger.info("🔄 /admin/api/hub/restart — spawning respawn watchdog")
    _spawn_respawn_watchdog()
    _delayed_shutdown(delay=0.8)
    return {"ok": True, "detail": "hub will restart shortly"}


# ----------------------------------------------------------------- log tail

@router.get("/api/hub/log/recent")
async def log_recent(limit: int = 400) -> Dict[str, Any]:
    return {"lines": HUB_LOG.lines(limit=max(1, min(limit, 2000)))}


@router.get("/api/hub/log/tail")
async def log_tail(request: Request) -> StreamingResponse:
    return sse_stream(
        request, HUB_LOG.subscribe, HUB_LOG.unsubscribe,
        seed=HUB_LOG.lines(limit=200),
    )


# ----------------------------------------------------------------- requests

@router.get("/api/hub/requests/recent")
async def requests_recent(limit: int = 50) -> Dict[str, Any]:
    return {"requests": OBS.recent_requests(limit=max(1, min(limit, 200)))}


@router.get("/api/hub/requests/stream")
async def requests_stream(request: Request) -> StreamingResponse:
    from src.hub_observability import _rec_to_dict

    return sse_stream(
        request, OBS.subscribe, OBS.unsubscribe,
        seed=OBS.recent_requests(limit=20),
        to_dict=_rec_to_dict,
        reverse_seed=True,  # send oldest-first so order matches the stream
    )


@router.get("/api/hub/errors/recent")
async def errors_recent(limit: int = 50) -> Dict[str, Any]:
    return {"errors": OBS.recent_errors(limit=max(1, min(limit, 50)))}


@router.get("/api/hub/counters")
async def counters() -> Dict[str, Any]:
    return {"counters": OBS.counters_snapshot()}


# ----------------------------------------------------------------- stats

@router.get("/api/hub/stats")
async def stats() -> Dict[str, Any]:
    """gpu_stats() shells out to nvidia-smi (3s timeout). Keep it off the
    event loop so the rest of /admin stays snappy."""
    from src import system_stats

    ram = system_stats.ram_stats()
    cpu = system_stats.cpu_stats()
    gpus = await asyncio.to_thread(system_stats.gpu_stats)
    history = OBS.stats_snapshot()
    return {"ram": ram, "cpu": cpu, "gpus": gpus, "history": history}
