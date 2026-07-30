"""Hub tab API — status, control, live request stream, log tail, install.

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

Plus /admin/api/install/{status,fix-all} which fold in the old install tab.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.hub_log import HUB_LOG
from src.hub_observability import OBS
from src.server_process import WIN_NEW_GROUP, lan_ip

from ._helpers import maybe_json, sse_stream

logger = logging.getLogger(__name__)
router = APIRouter()


# ----------------------------------------------------------------- helpers

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


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

def _delayed_shutdown(delay: float = 0.4) -> None:
    """Signal ourselves to exit after ``delay`` seconds, so the HTTP
    response can flush first. Uvicorn handles SIGINT/SIGTERM as a clean
    shutdown on both Windows and POSIX."""

    def _runner() -> None:
        time.sleep(delay)
        try:
            if sys.platform == "win32":
                # signal.raise_signal arrived in 3.8 and works under
                # uvicorn's SIGINT handler.
                signal.raise_signal(signal.SIGINT)
            else:
                os.kill(os.getpid(), signal.SIGTERM)
        except Exception as exc:  # noqa: BLE001 — fall back
            logger.error("⚠️ shutdown signal failed: %s — using os._exit", exc)
            os._exit(0)

    import threading
    threading.Thread(target=_runner, daemon=True).start()


def _delayed_darwin_bootout(label: str, delay: float = 0.4) -> None:
    """Unload the LaunchAgent job entirely, so a deliberate stop actually
    stays stopped (#181).

    Confirmed empirically on this machine: launchd's ``KeepAlive`` respawns
    the job after *any* signal-terminated exit — a plain self-SIGTERM
    (``_delayed_shutdown``) and even an explicit ``launchctl stop`` both got
    immediately relaunched. ``launchctl bootout`` is the only thing that
    actually removes the job from launchd's active registry, so nothing is
    left to respawn. Bringing it back requires ``launchctl bootstrap``
    again — the ``bootstrap`` action in ``mac/bin/hub-remote-ctl.sh`` and
    ``src/install.py``'s ``_fix_launchagent()`` both already do this.
    """

    def _runner() -> None:
        time.sleep(delay)
        subprocess.run(
            ["launchctl", "bootout", f"gui/{os.getuid()}/{label}"],
            capture_output=True,
        )

    import threading
    threading.Thread(target=_runner, daemon=True).start()


def _under_systemd() -> bool:
    """Are we running as a systemd unit? (#341/#368)

    systemd sets ``INVOCATION_ID`` in every unit's environment — the standard
    "am I supervised by systemd" signal. Only then do the stop/restart
    endpoints drive ``systemctl`` (below): a dev running the hub by hand on
    Linux falls through to the plain self-signal path, which is correct there
    (nothing would respawn it). Gated on Linux so it never fires elsewhere.
    """
    return sys.platform.startswith("linux") and bool(os.environ.get("INVOCATION_ID"))


def _delayed_systemctl(verb: str, delay: float = 0.4) -> None:
    """Run ``sudo -n systemctl <verb> local-llm-hub`` after a short delay so the
    HTTP response flushes first (#368).

    ``stop``/``restart`` SIGTERM this very process (it lives in the unit's
    cgroup) — which is the point: unlike the plain self-SIGTERM path,
    ``Restart=always`` would immediately respawn a bare signal, so a *deliberate*
    stop must go through systemd itself. ``sudo -n`` never prompts; a missing
    passwordless-sudo rule is logged rather than hanging.
    """
    from src.install import SYSTEMD_UNIT_NAME

    def _runner() -> None:
        time.sleep(delay)
        r = subprocess.run(
            ["sudo", "-n", "systemctl", verb, SYSTEMD_UNIT_NAME],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            logger.error("⚠️ `sudo -n systemctl %s %s` failed: %s",
                         verb, SYSTEMD_UNIT_NAME, (r.stderr or "").strip())

    import threading
    threading.Thread(target=_runner, daemon=True).start()


def _restart_log_path() -> Path:
    """File the detached watchdog redirects the relaunched server into.

    The respawn is detached and outlives this process, so its stdout has
    nowhere to go in-process — and under ``pythonw`` there is no console
    at all. We give it a real file so (a) ``src.server``'s import-time
    logging write doesn't crash a console-less child, and (b) a failed
    restart leaves a diagnostic trail instead of vanishing silently.
    """
    return PROJECT_ROOT / "data" / "logs" / "restart.log"


def _spawn_respawn_watchdog() -> None:
    """Spawn a detached Python that waits for our PID to die then re-launches us.

    The relaunch is made the way ``src/server_process.start()`` spawns the
    hub — never a bare ``pythonw`` with no stdout. The actual wait/relaunch
    logic lives in ``src/_respawn_watchdog.py`` (issue #198 — this used to
    be a ~60-line string literal built up line-by-line and fed to
    ``python -c``, invisible to lint/type-check and one quoting slip away
    from a silently-failed restart). That module is deliberately
    stdlib-only with no import from any other ``src.*`` module: it's the
    thing recovering *from* a broken deploy, so it can't assume the rest
    of the hub's package still imports cleanly — only its own module and
    the empty ``src/__init__.py`` need to load.
    """
    parent_pid = os.getpid()
    port = _hub_port()
    log_path = _restart_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("🔄 respawn watchdog: relaunch log → %s", log_path)
    # Same flags src.server_process.start() uses for the hub itself —
    # DETACHED_PROCESS is deliberately omitted, it's mutually exclusive
    # with CREATE_NO_WINDOW per the Win32 CreateProcess docs (#282/#283).
    creationflags = WIN_NEW_GROUP
    # Capture the watchdog's own stdout/stderr to the same log so a
    # failure *before* it opens its own handle (e.g. a bad argv) is still
    # visible rather than swallowed by DEVNULL.
    wd_log = open(log_path, "a", encoding="utf-8", errors="replace")
    subprocess.Popen(
        [
            sys.executable, "-m", "src._respawn_watchdog",
            "--parent-pid", str(parent_pid),
            "--port", str(port),
            "--log-path", str(log_path),
            "--root", str(PROJECT_ROOT),
        ],
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=wd_log,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )


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


# ----------------------------------------------------------------- install

@router.get("/api/install/status")
async def install_status() -> Dict[str, Any]:
    """Run every install check off the event loop — many shell out to
    ``claude --version`` / ``nvidia-smi`` / ``llama-server --version``
    via blocking subprocess.run, which would otherwise pin the entire
    uvicorn worker for seconds while other admin requests queue up."""
    from src import install

    report = await asyncio.to_thread(install.run_all_checks)
    return {
        "worst_status": report.worst_status,
        "ok": report.ok,
        "checks": [asdict(c) for c in report.checks],
    }


@router.post("/api/install/fix")
async def install_fix(request: Request) -> Dict[str, Any]:
    """Run a single fix by ``fix_id``.

    Uses the brief use_cache=True report (issue #198): the admin UI always
    calls install_status() — which populates that cache — moments before a
    user clicks a fix button, so locating one check by fix_id doesn't need
    to force a second full (expensive) battery run.
    """
    from src import install

    body = await maybe_json(request)
    fix_id = (body or {}).get("fix_id")
    if not fix_id:
        raise HTTPException(status_code=400, detail="fix_id is required")
    report = await asyncio.to_thread(install.run_all_checks, use_cache=True)
    target = next((c for c in report.checks if c.fix_id == fix_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"no fixable check with fix_id={fix_id!r}")
    fn = install.fix_fn_for(target)
    if fn is None:
        raise HTTPException(status_code=400, detail=f"no fix function for {fix_id!r}")
    try:
        await asyncio.to_thread(fn)
    except Exception as exc:  # noqa: BLE001 — surface the failure to the UI
        raise HTTPException(status_code=500, detail=f"fix {fix_id!r} failed: {exc}")
    return {"ok": True, "fix_id": fix_id}


@router.post("/api/install/fix-all")
async def install_fix_all() -> Dict[str, Any]:
    """Run every currently-fixable check. Same brief use_cache=True reuse
    as install_fix() — see its docstring."""
    from src import install

    report = await asyncio.to_thread(install.run_all_checks, use_cache=True)
    ran: List[Dict[str, Any]] = []
    for c in report.checks:
        if c.status not in ("missing", "error"):
            continue
        fn = install.fix_fn_for(c)
        if fn is None:
            continue
        try:
            await asyncio.to_thread(fn)
            ran.append({"fix_id": c.fix_id, "ok": True})
        except Exception as exc:  # noqa: BLE001
            ran.append({"fix_id": c.fix_id, "ok": False, "error": str(exc)})
    return {"ran": ran}
