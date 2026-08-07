"""On-demand model lifecycle — spawn on first request, unload when idle (#422).

A ``models.yaml`` row may declare ``startup: on_demand`` (default ``eager``,
today's always-on behavior). Such a row is never started eagerly — not by
hub autostart (``model_registry.desired_model_ids`` excludes it), not by
the fleet reconcile loop (same derived set, #430), not by the failover
engine (guards on ``Model.startup``). Instead:

* **Load on first request** — the dispatch paths that reach a local backend
  (chat completions, the Anthropic-shape ``/v1/messages`` translation, and
  the ``/v1/audio/speech`` proxy) call :func:`ensure_ready` before
  forwarding. When the backend isn't up, the request spawns it via
  ``backend_process.start`` and blocks polling ``is_reachable`` until the
  process answers — the same request-waits-for-readiness pattern as the
  lazy whisper proxy (``src/whisper_translate_proxy._ChildSupervisor``),
  lifted from a per-process shim into the hub itself.
* **Idle unload** — ``idle_unload_minutes: <int>`` arms the watchdog loop
  (``server_lifecycle`` wires :func:`idle_unload_loop`): after that many
  minutes without a request the hub stops the backend. A model stopped this
  way (or by hand via the admin API) stays down — the reconcile/failover
  guards above are what keep the supervisor from resurrecting it, the
  regression observed live in #422.
* **VRAM budget warning, not arbitration** — before an on-demand spawn, the
  sum of the running local models' ``est_vram_mb`` plus the candidate's is
  checked against the host's ``vram_mb`` ceiling (#375 grid math). Overcommit
  logs a loud warning and proceeds: WDDM overcommit degrades transiently and
  idle unload recovers it. Eviction arbitration is explicitly out of scope.

State is process-local and in-memory (last-request timestamps, in-flight
request counts) — an unload decision never outlives the hub that made it.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

from .model_registry import Model, STARTUP_ON_DEMAND

logger = logging.getLogger(__name__)

# How long the first request waits for a cold backend to come up. Sized for
# the heaviest on-demand row (gemma4_26b's 13.4 GB IQ4_XS read from NVMe) with
# the same generosity as the orpheus shim's llama child (180 s deadline).
READY_DEADLINE_S = 180.0
READY_POLL_S = 1.0
# Idle watchdog cadence — coarse on purpose; unload windows are minutes.
IDLE_CHECK_INTERVAL_S = 60.0

_LOCK = threading.Lock()
# model_id -> monotonic timestamp of the last request activity (start or end).
_LAST_USED: Dict[str, float] = {}
# model_id -> number of requests currently being served. A model with work
# in flight is never idle, however long ago its last *arrival* was.
_IN_FLIGHT: Dict[str, int] = {}
# Per-model spawn locks so two concurrent first-requests can't race
# ``backend_process.start`` into a double spawn / port clash.
_SPAWN_LOCKS: Dict[str, threading.Lock] = {}


class OnDemandNotReady(RuntimeError):
    """The on-demand backend failed to come up within the deadline."""


def is_on_demand(model: Model) -> bool:
    """True when ``model`` declared ``startup: on_demand`` (#422)."""
    return getattr(model, "startup", None) == STARTUP_ON_DEMAND


def _spawn_lock(model_id: str) -> threading.Lock:
    with _LOCK:
        lock = _SPAWN_LOCKS.get(model_id)
        if lock is None:
            lock = threading.Lock()
            _SPAWN_LOCKS[model_id] = lock
        return lock


def request_started(model_id: str, now: Optional[float] = None) -> None:
    """Mark a request against ``model_id`` as in flight (call before dispatch)."""
    t = time.monotonic() if now is None else now
    with _LOCK:
        _IN_FLIGHT[model_id] = _IN_FLIGHT.get(model_id, 0) + 1
        _LAST_USED[model_id] = t


def request_finished(model_id: str, now: Optional[float] = None) -> None:
    """Mark a request against ``model_id`` as done (call in a ``finally``)."""
    t = time.monotonic() if now is None else now
    with _LOCK:
        _IN_FLIGHT[model_id] = max(0, _IN_FLIGHT.get(model_id, 0) - 1)
        _LAST_USED[model_id] = t


class RequestTracking:
    """In-flight bookkeeping for one request against one model (#470).

    Every dispatch site that can reach a local ``startup: on_demand`` backend
    has to pair a :func:`request_started` with exactly one
    :func:`request_finished`, or the idle-unload window stays pinned open
    forever. That pairing used to be hand-copied at four sites across three
    modules — worst case, one ``request_started`` balanced by three separately
    placed ``request_finished`` calls in a branched streaming function, where
    a fifth exit path added later silently leaks. Obtain one of these from
    :func:`tracking` instead:

    * ``with tracking(model, remote):`` — the whole request is served inside
      the block (the two non-streaming chat paths).
    * ``track = tracking(...); track.start()`` … ``track.finish()`` — the
      response outlives the dispatch function, so the generator's ``finally``
      closes it (the SSE passthrough).
    * ``with tracking(...) as track:`` … ``track.detach()`` — the ``with``
      covers every exit path up to the point ownership is handed to a
      streaming generator, which then calls ``track.finish()``.

    ``finish`` is idempotent, so a belt-and-braces second call (or a detach
    that races a raise) can never double-decrement the in-flight count.
    """

    __slots__ = ("model_id", "active", "_detached", "_finished")

    def __init__(self, model_id: str, active: bool) -> None:
        self.model_id = model_id
        self.active = active
        self._detached = False
        self._finished = False

    def start(self) -> "RequestTracking":
        if self.active:
            request_started(self.model_id)
        return self

    def finish(self) -> None:
        if self.active and not self._finished:
            self._finished = True
            request_finished(self.model_id)

    def detach(self) -> None:
        """Hand the in-flight mark to a caller that outlives this ``with``
        block — ``__exit__`` stops finishing it, so ``finish()`` must be
        called explicitly (a streaming generator's ``finally``)."""
        self._detached = True

    def __enter__(self) -> "RequestTracking":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self._detached:
            self.finish()
        return False


def tracking(model: Model, remote: Optional[str] = None) -> RequestTracking:
    """Tracker for one request against ``model`` (#470).

    Tracking is a no-op unless the request is served *locally* (``remote`` is
    the resolved peer base URL, ``None`` when this hub serves it) by a row
    declaring ``startup: on_demand`` — a remote hop is the peer hub's to
    account for, and an eager row is never unloaded. See
    :class:`RequestTracking` for the three call shapes.
    """
    return RequestTracking(model.id, remote is None and is_on_demand(model))


def _idle_state(model_id: str) -> tuple[int, Optional[float]]:
    with _LOCK:
        return _IN_FLIGHT.get(model_id, 0), _LAST_USED.get(model_id)


def _warn_on_vram_overcommit(model: Model) -> Optional[int]:
    """Log a loud warning when loading ``model`` would exceed the host's
    ``vram_mb`` ceiling (#375 math: sum of running local models' static
    ``est_vram_mb`` estimates + the candidate's). Advisory only — the load
    always proceeds. Returns the projected total (MB) when it overcommits,
    ``None`` otherwise (including hosts with no ceiling).
    """
    from . import backend_process as bp
    from .host_profile import resolve as resolve_host
    from .model_registry import cpu_resident_map

    try:
        host = resolve_host()
        ceiling = host.vram_mb
    except Exception:  # noqa: BLE001 — hostless tooling contexts
        ceiling = None
    if not ceiling:
        return None
    # CPU-resident rows on this host (piper, -ng whisper rows, a chain's
    # degraded cpu tier) hold no GPU VRAM — excluding them keeps the headroom
    # math honest (#431), same exclusion as the fleet summary's capacity sum.
    cpu = cpu_resident_map().get(host.id, set())
    running = bp.running_backends()
    projected = sum(
        m.est_vram_mb or 0
        for mid, m in running.items()
        if mid != model.id and mid not in cpu
    ) + (0 if model.id in cpu else model.est_vram_mb or 0)
    if projected <= ceiling:
        return None
    logger.warning(
        "⚠️ VRAM overcommit: loading %s (~%s MB) puts the running set at "
        "~%s MB against this host's %s MB ceiling — proceeding anyway "
        "(WDDM overcommit degrades transiently; idle unload recovers it)",
        model.id, model.est_vram_mb or 0, projected, ceiling,
    )
    return projected


def ensure_ready(model: Model, deadline_s: float = READY_DEADLINE_S) -> None:
    """Make sure an on-demand local backend is up before a request hits it.

    No-op for eager rows, virtual aliases, and models currently served by a
    *remote* host (the owning hub runs its own on-demand lifecycle). For a
    cold local on-demand backend this spawns it and blocks (the caller runs
    in a worker thread / ``to_thread``) polling ``is_reachable`` until it
    answers or ``deadline_s`` expires — raising :class:`OnDemandNotReady`
    so the route can surface a distinct 503.
    """
    from . import backend_process as bp
    from .remote_proxy import remote_base_url

    if not is_on_demand(model) or model.virtual or not model.port:
        return
    if remote_base_url(model) is not None:
        return
    if bp.is_reachable(model, timeout=0.4):
        return

    with _spawn_lock(model.id):
        # Re-check under the lock — a concurrent first-request may have
        # already brought the backend up while this one waited.
        if bp.is_reachable(model, timeout=0.4):
            return
        _warn_on_vram_overcommit(model)
        ok, msg = bp.start(model.id)
        # "already running" means a process exists but isn't answering yet
        # (mid-load) — that's exactly what the readiness poll below absorbs.
        if not ok and "already running" not in msg.lower():
            raise OnDemandNotReady(f"on-demand start of {model.id!r} failed: {msg}")
        logger.info("⏳ on-demand: loading %s (%s) — request waits for readiness",
                    model.id, msg)
        started = time.monotonic()
        while time.monotonic() - started < deadline_s:
            if bp.is_reachable(model, timeout=1.0):
                logger.info("✅ on-demand: %s ready after %.1fs",
                            model.id, time.monotonic() - started)
                return
            time.sleep(READY_POLL_S)
        raise OnDemandNotReady(
            f"on-demand backend {model.id!r} did not become ready within "
            f"{deadline_s:.0f}s — check data/logs/backend-{model.id}.log"
        )


def idle_unload_pass(now: Optional[float] = None) -> Dict[str, Any]:
    """One idle-unload sweep over the local on-demand models.

    A running ``on_demand`` row with ``idle_unload_minutes`` set is stopped
    when it has no requests in flight and its last activity is older than
    the window. A model that is up but has never been touched this hub-run
    (e.g. inherited across a restart) starts its idle clock at first sight,
    so it still unloads one full window later rather than immediately or
    never. Returns ``{model_id: action}`` for observability/tests.
    """
    from . import backend_process as bp
    from .model_registry import local_models

    t = time.monotonic() if now is None else now
    results: Dict[str, Any] = {}
    try:
        candidates = [
            m for m in local_models()
            if is_on_demand(m) and not m.virtual and m.idle_unload_minutes
        ]
    except Exception as exc:  # noqa: BLE001 — config error must not kill the loop
        logger.warning("on-demand idle sweep: config scan failed: %s", exc)
        return results

    for m in candidates:
        if not bp.is_running(m.id):
            continue
        in_flight, last_used = _idle_state(m.id)
        if last_used is None:
            # First sighting of an already-running instance — arm the clock.
            with _LOCK:
                _LAST_USED.setdefault(m.id, t)
            results[m.id] = "armed"
            continue
        if in_flight > 0:
            results[m.id] = "busy"
            continue
        idle_s = t - last_used
        window_s = float(m.idle_unload_minutes) * 60.0
        if idle_s < window_s:
            results[m.id] = "warm"
            continue
        ok, msg = bp.stop(m.id)
        logger.info(
            "💤 on-demand: unloaded %s after %.0f min idle (window %s min) -> %s %s",
            m.id, idle_s / 60.0, m.idle_unload_minutes, ok, msg,
        )
        results[m.id] = "unloaded" if ok else f"stop failed: {msg}"
    return results


async def idle_unload_loop() -> None:
    """Background watchdog wired by ``server_lifecycle`` — periodic
    :func:`idle_unload_pass`, off-loop (``bp`` calls shell out / probe)."""
    import asyncio

    while True:
        await asyncio.sleep(IDLE_CHECK_INTERVAL_S)
        try:
            await asyncio.to_thread(idle_unload_pass)
        except Exception as exc:  # noqa: BLE001 — watchdog must not die
            logger.warning("on-demand idle sweep raised: %s", exc)


def reset() -> None:
    """Drop all in-memory state (tests)."""
    with _LOCK:
        _LAST_USED.clear()
        _IN_FLIGHT.clear()
        _SPAWN_LOCKS.clear()
