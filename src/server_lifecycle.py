"""FastAPI startup/shutdown lifecycle wiring + the background resource
sampler for the hub app.

Extracted out of ``server.py`` (issue #198) — that module's own docstring
already flagged ``/v1/images/*`` and ``/v1/audio/*`` as prior splits for
exactly this reason (keeping the god-module from growing further); the
startup/shutdown event bodies and the 2s-tick resource sampler were the
next-largest chunk still living inline, mixed in with route registration.

``server.py`` wires these on with ``app.add_event_handler(...)`` (the
non-decorator form of ``@app.on_event``) rather than decorating them here,
so the functions stay plain, directly-callable, and directly-testable —
``tests/test_restart_keepalive.py`` calls ``stop_backend_children()``
straight, with no FastAPI app in the loop.
"""

from __future__ import annotations

import asyncio
import logging
import time

from .hub_log import HUB_LOG
from .hub_observability import OBS

logger = logging.getLogger(__name__)

# Tasks spawned by wire_observatory_loop, tracked so stop_background_tasks
# can cancel them on shutdown -- see that function's docstring for why.
_BACKGROUND_TASKS: list[asyncio.Task] = []


async def stop_background_tasks() -> None:
    """Cancel every perpetual task ``wire_observatory_loop`` spawned.

    On a real process exit this wouldn't matter (the OS reclaims everything),
    but the ASGI lifespan also tears down on every ``with TestClient(app) as
    c:`` exit in the test suite -- and ``_fleet_reconcile_loop`` /
    ``_model_failover_loop`` (each starts with a boot-delay ``asyncio.sleep``,
    so a test's brief lifespan window usually catches them still sleeping)
    would otherwise keep running in that torn-down loop, mutating the very
    module-global state (``model_failover.TRACKER``, ``fleet_reconcile``'s
    peer-transport hooks) that ``test_fleet_reconcile.py`` /
    ``test_model_failover.py`` monkeypatch and assert on from the main thread
    -- the order-dependent full-suite pollution in issue #416."""
    tasks, _BACKGROUND_TASKS[:] = list(_BACKGROUND_TASKS), []
    if not tasks:
        return
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def stop_backend_children() -> None:
    """Tear down every model subprocess the hub spawned.

    The hub owns its backend children (since the tray drives them via
    the admin API). Without this, a clean ``CTRL+C`` would leave
    orphan ``llama-server`` / ``whisper-server`` processes holding
    their ports until the user logged out.

    Exception: on an admin **restart** the children must survive so the
    respawned hub re-adopts them (``inherit_running_backends``). The
    restart endpoint sets ``backend_process.restart_pending()`` before
    signalling shutdown; we honour it by skipping teardown.
    """
    from . import backend_process as bp
    from . import http_client

    try:
        await http_client.aclose()
        http_client.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("shutdown: closing shared httpx clients raised: %s", exc)

    if bp.restart_pending():
        survivors = list(bp.running_backends().keys())
        logger.info(
            "shutdown: restart in progress — leaving %d backend(s) running for adoption: %s",
            len(survivors), survivors,
        )
        return

    for model_id in list(bp.running_backends().keys()):
        try:
            ok, msg = bp.stop(model_id)
            logger.info("shutdown: stop %s -> %s %s", model_id, ok, msg)
        except Exception as exc:  # noqa: BLE001
            logger.warning("shutdown: stop %s raised: %s", model_id, exc)


async def stop_diagnostics_sampler() -> None:
    """Gracefully finish any diagnostics capture on hub shutdown (#316).

    The sampler is an in-process asyncio task (issue #315), so a capture cannot
    survive a process restart regardless — draining it here finalizes the run as
    ``stopped`` instead of leaving it to be recovered as an ``orphan`` on the
    next boot. It also closes a test-isolation hazard: without this, a capture's
    ``to_thread`` worker (or the scheduled-snapshot loop) can outlive the app and
    write through the module-global ``db_path`` into the *next* test's database,
    surfacing as an intermittent ``database is locked``. Best-effort — a shutdown
    must never hang on it (``stop_run`` is itself bounded by a grace timeout)."""
    try:
        from .diagnostics import sampler
        await sampler.stop_scheduled_snapshots()
        if sampler.is_capturing():
            await sampler.stop_run()
    except Exception as exc:  # noqa: BLE001
        logger.warning("shutdown: stopping diagnostics sampler raised: %s", exc)


async def wire_observatory_loop() -> None:
    """Capture the running event loop so the synchronous middleware can
    fan out SSE events from non-async callers."""
    loop = asyncio.get_running_loop()
    OBS.attach_loop(loop)
    HUB_LOG.attach_loop(loop)
    # Start the resource sampler. 2s tick × 150 samples = 5 min ring.
    _BACKGROUND_TASKS.append(loop.create_task(_resource_sampler()))

    # Inherit any backend process left running on one of our ports by a
    # previous hub instance. Without this, every hub restart shows the
    # surviving model backends as "adopted" rather than "running".
    try:
        from . import backend_process as bp
        inherited = await asyncio.to_thread(bp.inherit_running_backends)
        if inherited:
            logger.info("📎 Inherited %d running backend(s) from a previous hub", inherited)
    except Exception as exc:  # noqa: BLE001
        logger.warning("inherit_running_backends failed: %s", exc)

    # The hub owns configured backend autostart so every launch surface
    # (tray, run_hub.bat, python -m src.run_backend hub) behaves the same.
    _BACKGROUND_TASKS.append(loop.create_task(_autostart_configured_backends()))
    # Same idea for Docker/Langfuse/AgentsView (issue #265) — the startup
    # profile's non-model service flags.
    _BACKGROUND_TASKS.append(loop.create_task(_autostart_services()))
    # Fleet always-on control plane (issue #353): converge every host to its
    # registry-derived desired state on boot + every few minutes (#430 — the
    # placement comes from models.yaml's hosts chains + startup policy). Since
    # #374 this is also the *sole* peer wake/sync mechanism (a peer with
    # desired models is woken + started here; one with none is left asleep).
    _BACKGROUND_TASKS.append(loop.create_task(_fleet_reconcile_loop()))
    # Dynamic model fallback (issue #342): probe the hosts of every
    # multi-host ``hosts:`` chain and move ownership down/up the chain with
    # hysteresis. The task exits immediately when no enabled model declares
    # a multi-host chain, so pre-#342 configs pay nothing.
    _BACKGROUND_TASKS.append(loop.create_task(_model_failover_loop()))
    # Diagnostics (issue #315): close any capture orphaned by a previous hub
    # and arm the scheduled snapshot if it's enabled. No task is created when
    # it's off — the feature costs nothing until asked for.
    _BACKGROUND_TASKS.append(loop.create_task(_init_diagnostics()))
    # On-demand idle watchdog (issue #422): periodically unloads a
    # ``startup: on_demand`` model that has sat idle past its
    # ``idle_unload_minutes`` window. Cheap when no row opts in — the pass
    # scans the registry and finds no candidates.
    _BACKGROUND_TASKS.append(loop.create_task(_on_demand_idle_loop()))
    # Config drift convergence (#424): on the single write host only, poll
    # hub peers' models.yaml sha and re-fire the #181 sync at any satellite
    # that missed a push-triggered sync (e.g. it was powered off). The task
    # returns immediately on every non-writer host.
    _BACKGROUND_TASKS.append(loop.create_task(_config_drift_loop()))


async def _autostart_configured_backends() -> None:
    # The registry-derived desired set (#430): eager rows preferred-owned by
    # (or unowned and launchable on) this host. This is how an offline
    # satellite applies its models when it powers up — its own synced
    # models.yaml, no tower involvement needed.
    from . import backend_process as bp
    from .model_registry import desired_model_ids

    model_ids = desired_model_ids()
    if not model_ids:
        return
    logger.info("autostart: configured backend set: %s", model_ids)
    for model_id in model_ids:
        try:
            ok, msg = await asyncio.to_thread(bp.start, model_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("autostart: %s raised: %s", model_id, exc)
            continue
        if ok or "already running" in msg.lower():
            logger.info("autostart: %s -> %s", model_id, msg)
        else:
            logger.warning("autostart: %s -> %s", model_id, msg)


async def _init_diagnostics() -> None:
    """Diagnostics startup housekeeping (issue #315).

    The sampler lives in-process, so a run still marked ``running`` in the DB
    belongs to a hub that died mid-capture — close it so a stale row never
    looks like a live capture. Then arm the opt-in scheduled snapshot. Both
    are best-effort: diagnostics must never keep the hub from starting.
    """
    try:
        from .diagnostics import sampler, settings as diag_settings, store

        await asyncio.to_thread(store.close_orphan_runs)
        cfg = diag_settings.load_settings()
        if cfg.scheduled_enabled:
            await sampler.start_scheduled_snapshots(
                cfg.scheduled_interval_hours, retention_days=cfg.retention_days,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("diagnostics init failed: %s", exc)


async def _autostart_services() -> None:
    """Bring up Docker/Langfuse + AgentsView per the startup profile (#265).

    Best-effort and soft-failing, same spirit as ``services.launch_stack()``
    itself — a slow/unreachable Docker Desktop must never block the hub from
    finishing its own startup. Peer wake/sync used to live here too (the legacy
    ``mac_mini_sync`` branch); since #374 it is owned entirely by the fleet
    reconcile loop (``_fleet_reconcile_loop`` → ``fleet_reconcile.reconcile_once``),
    driven since #430 by the registry-derived desired state (``models.yaml``'s
    ``hosts:`` chains + ``startup:`` policy) as the sole cross-host source of
    truth.
    """
    from . import services as svc
    from .startup_profile import load_startup_profile

    profile = load_startup_profile()

    if profile.docker or profile.langfuse:
        try:
            result = await svc.launch_stack()
            logger.info(
                "autostart: services launch %s: %s",
                "ok" if result["ok"] else "failed",
                result["steps"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("autostart: services launch raised: %s", exc)

    if profile.agentsview:
        try:
            result = await svc.launch_agentsview()
            logger.info(
                "autostart: agentsview launch %s: %s",
                "ok" if result["ok"] else "failed",
                result["steps"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("autostart: agentsview launch raised: %s", exc)


async def _fleet_reconcile_loop() -> None:
    """Periodic fleet convergence (issue #353).

    A short boot delay lets ``_autostart_configured_backends`` +
    ``inherit_running_backends`` settle so the first pass sees accurate running
    state (and never races them into a double-spawn), then an additive
    ``reconcile_once`` every ``FLEET_RECONCILE_INTERVAL_S``. Soft-failing, same
    spirit as the sampler: a reconcile error must never take the loop (or the
    hub) down. No-ops cheaply when no placement is configured.
    """
    from . import fleet_reconcile

    await asyncio.sleep(fleet_reconcile.FLEET_RECONCILE_BOOT_DELAY_S)
    while True:
        try:
            results = await fleet_reconcile.reconcile_once()
            if results:
                logger.info("🛰️ fleet reconcile pass: %s", list(results.keys()))
        except Exception as exc:  # noqa: BLE001 — loop must not die
            logger.warning("fleet reconcile pass raised: %s", exc)
        await asyncio.sleep(fleet_reconcile.FLEET_RECONCILE_INTERVAL_S)


async def _model_failover_loop() -> None:
    """Dynamic model fallback across ordered host chains (issue #342).

    Thin lifecycle shim — the probe/decide/act cycle, the hysteresis rules,
    and the no-chains early exit all live in ``src.model_failover``
    (``failover_loop``), keeping the engine unit-testable with no FastAPI
    app in the loop. The same boot delay as the fleet reconcile loop lets
    autostart + backend inheritance settle before the first probe pass.
    """
    from . import model_failover

    try:
        await model_failover.failover_loop(boot_delay_s=20.0)
    except Exception as exc:  # noqa: BLE001 — the shim must not crash startup
        logger.warning("model failover loop exited abnormally: %s", exc)


async def _config_drift_loop() -> None:
    """Periodic models.yaml drift sync across hub peers (issue #424).

    Thin lifecycle shim — the write-host gate, the per-(peer, sha) attempt
    memory, and the sync call live in ``src.config_write``
    (``config_drift_sync_loop``), keeping the policy unit-testable.
    """
    from . import config_write

    try:
        await config_write.config_drift_sync_loop()
    except Exception as exc:  # noqa: BLE001 — the shim must not crash startup
        logger.warning("config drift loop exited abnormally: %s", exc)


async def _on_demand_idle_loop() -> None:
    """Idle unload for ``startup: on_demand`` models (issue #422).

    Thin lifecycle shim — the sweep cadence, the idle decision, and the
    in-flight guard live in ``src.on_demand`` (``idle_unload_loop``),
    keeping the policy unit-testable with no FastAPI app in the loop.
    """
    from . import on_demand

    try:
        await on_demand.idle_unload_loop()
    except Exception as exc:  # noqa: BLE001 — the shim must not crash startup
        logger.warning("on-demand idle loop exited abnormally: %s", exc)


async def _resource_sampler() -> None:
    """Background task that samples RAM + CPU + GPU usage every 2 s.

    The sampling itself runs in a worker thread (#392): ``gpu_stats()``
    shells out to nvidia-smi, which stalls for seconds when the GPU is
    busy serving models — calling it inline here blocked the whole event
    loop every 2 s, and any HTTP client with a 5 s timeout (the e2e
    suite's httpx calls) would intermittently trip on an otherwise-idle
    endpoint. Same off-the-loop treatment ``/api/hub/stats`` already has.
    """
    from . import system_stats
    from .hub_observability import StatSample

    def _sample_sync():
        return (
            system_stats.ram_stats(),
            system_stats.cpu_stats(),
            system_stats.gpu_stats(),
        )

    while True:
        try:
            ram, cpu, gpus = await asyncio.to_thread(_sample_sync)
            gpu0_vram = None
            gpu0_util = None
            if gpus:
                first = gpus[0]
                gpu0_vram = first.get("vram_percent")
                gpu0_util = first.get("util_percent")
            OBS.record_stat(
                StatSample(
                    ts=time.time(),
                    ram_percent=float(ram.get("percent", 0.0)),
                    cpu_percent=float(cpu.get("percent", 0.0)),
                    gpu0_vram_percent=gpu0_vram,
                    gpu0_util_percent=gpu0_util,
                )
            )
        except Exception:  # noqa: BLE001 — sampler must not die
            pass
        await asyncio.sleep(2.0)


def register(app) -> None:
    """Attach the hub's startup/shutdown handlers to ``app``.

    Calls ``app.on_event(event_type)`` as a plain function (its decorator
    return value) rather than using ``@app.on_event(...)`` sugar directly
    on these functions, so the handlers themselves stay plain module-level
    callables — importable and directly testable without needing a
    FastAPI app (``tests/test_restart_keepalive.py`` calls
    ``stop_backend_children()`` straight). Starlette's ``Router`` dropped
    ``add_event_handler`` in favor of lifespan context managers; ``on_event``
    is FastAPI's still-supported (if deprecated) escape hatch for the
    decorator-less registration this module needs.
    """
    app.on_event("shutdown")(stop_background_tasks)
    app.on_event("shutdown")(stop_backend_children)
    app.on_event("shutdown")(stop_diagnostics_sampler)
    app.on_event("startup")(wire_observatory_loop)
