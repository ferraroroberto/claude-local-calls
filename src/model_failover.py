"""Dynamic model fallback across an ordered host chain (#342).

A model row may declare ``hosts: [a, b, c]`` — an ordered preference list
(``src.model_registry._parse_host_chain``). This module turns that static
chain into a *live* ownership decision — "which host serves ``model=X``
right now" — and keeps it true as hosts die and return:

* **Ownership resolution** — the model's owner is the first chain candidate
  that is reachable and has the model in its ``enabled:`` list. Every hub
  resolves requests against that owner (``effective_owner`` feeds
  ``remote_proxy.remote_base_url``) and proxies there via the existing #178
  cross-host path.
* **Failover** — when the current owner stays unreachable past
  ``fail_after_s``, ownership moves to the next reachable candidate.
* **Failback with hysteresis** — when a more-preferred host returns, it must
  stay reachable for ``failback_after_s`` (a *continuous* stability window)
  before ownership hands back — a repeatedly-rebooting host never
  accumulates the window, so ownership cannot flap. ``policy: sticky``
  disables automatic hand-back entirely (ownership stays put until the
  current owner itself dies).
* **Decentralized process actions** — each hub acts only on *itself*: when
  the tracker says "this host is now the effective owner" and the model
  isn't running locally, it starts it (``backend_process.start`` — which
  applies the ``cpu: true`` degraded-offload args on a flagged host); when
  ownership moves away, it stops **only** an instance this engine itself
  started (never a hand-started or autostarted one — same additive spirit
  as ``fleet_reconcile``). No master is needed for failover, so the chain
  keeps working when the control node itself is the host that died; the
  deterministic tie-break (chain order + reachability) means two live
  candidates always agree on who owns routing.

Backward compatibility is structural: a model with a single-host chain (any
bare ``host:`` row) never consults the tracker — ``effective_owner`` returns
``model.host`` statically — and when **no** enabled model declares a
multi-host chain the background loop exits immediately, so existing
deployments gain zero probes, zero tasks, zero behavior change.

**Contract with fleet_reconcile.py (#411):** this engine's ``fail_after_s``
window competes with ``fleet_reconcile.py``'s own always-on convergence, which
used to SSH-resurrect a deliberately-stopped peer within seconds — well inside
this window — so failover could never observe a real outage. That race is
closed on the reconcile side: arm ``src.fleet_maintenance`` for the target host
before inducing an outage (via ``/admin/api/fleet-maintenance``) so reconcile
stands down and this engine's probe/decide/act cycle gets a clean read.

Reachability reuses ``services.peer_health`` (the same dial-resolver probe
the Machines tab and fleet reconcile already use — no second prober), and
the observation loop is the only writer of tracker state; request paths
only read the current owner (cheap, lock-guarded dict lookup).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .host_profile import _load_config
from .model_registry import Model

logger = logging.getLogger(__name__)

POLICY_AUTO = "auto"
POLICY_STICKY = "sticky"


@dataclass(frozen=True)
class FailoverSettings:
    """Tunables for the failover engine — ``failover:`` in models.yaml."""

    probe_interval_s: float = 30.0
    # The current owner must be *continuously* unreachable this long before
    # ownership moves down-chain (a single missed probe is not a failover).
    fail_after_s: float = 90.0
    # A more-preferred host must be *continuously* reachable this long before
    # ownership hands back (the anti-flap hysteresis window).
    failback_after_s: float = 600.0
    policy: str = POLICY_AUTO  # "auto" (hand back) | "sticky" (stay put)


def load_settings() -> FailoverSettings:
    """Read the optional top-level ``failover:`` block from models.yaml."""
    raw = _load_config().get("failover") or {}
    if not isinstance(raw, dict):
        raw = {}
    policy = str(raw.get("policy") or POLICY_AUTO).strip().lower()
    if policy not in (POLICY_AUTO, POLICY_STICKY):
        logger.warning("failover: unknown policy %r — using %r", policy, POLICY_AUTO)
        policy = POLICY_AUTO
    defaults = FailoverSettings()
    return FailoverSettings(
        probe_interval_s=float(raw.get("probe_interval_s", defaults.probe_interval_s)),
        fail_after_s=float(raw.get("fail_after_s", defaults.fail_after_s)),
        failback_after_s=float(raw.get("failback_after_s", defaults.failback_after_s)),
        policy=policy,
    )


@dataclass
class _HostObservation:
    reachable: bool
    since: float  # when the host *entered* its current reachable/unreachable state


class OwnershipTracker:
    """Pure ownership state machine — observations in, owner decisions out.

    Deliberately framework-free and clock-injected (every mutator takes
    ``now``) so the hysteresis behavior is deterministic under test. One
    instance per process (module-level ``TRACKER``); the observation loop is
    the only writer, request paths call :meth:`owner_for` (read-only).
    """

    def __init__(self, settings: Optional[FailoverSettings] = None) -> None:
        self._lock = threading.Lock()
        self._settings = settings
        self._hosts: Dict[str, _HostObservation] = {}
        self._owner: Dict[str, str] = {}

    @property
    def settings(self) -> FailoverSettings:
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #
    def observe_host(self, host_id: str, reachable: bool, now: float) -> None:
        """Record a probe result. ``since`` only advances on a state *change*,
        so "up for N seconds" means continuously up across every probe."""
        with self._lock:
            st = self._hosts.get(host_id)
            if st is None or st.reachable != bool(reachable):
                self._hosts[host_id] = _HostObservation(bool(reachable), now)

    def _is_up(self, host_id: str) -> Optional[bool]:
        st = self._hosts.get(host_id)
        return None if st is None else st.reachable

    def _state_age(self, host_id: str, now: float) -> float:
        st = self._hosts.get(host_id)
        return 0.0 if st is None else max(0.0, now - st.since)

    # ------------------------------------------------------------------ #
    # Decisions
    # ------------------------------------------------------------------ #
    def owner_for(self, model_id: str) -> Optional[str]:
        """Current tracked owner (read-only; ``None`` until first decide)."""
        with self._lock:
            return self._owner.get(model_id)

    def decide(self, model_id: str, chain: List[str], now: float) -> Optional[str]:
        """Re-evaluate ownership of ``model_id`` over ``chain`` at ``now``.

        Rules (chain order is the deterministic tie-break throughout):

        * first call → the first candidate observed up; with no observations
          yet, the first candidate — i.e. the static preferred owner, so
          behavior before the first probe cycle equals the pre-#342 world.
        * owner down ≥ ``fail_after_s`` → the first candidate currently up
          (may be up- or down-chain of the dead owner). Nothing up → owner
          unchanged (there is nowhere better to point).
        * a strictly more-preferred candidate up ≥ ``failback_after_s`` →
          hand back (``policy: auto`` only; ``sticky`` never hands back).
        """
        if not chain:
            return None
        s = self.settings
        with self._lock:
            current = self._owner.get(model_id)
            if current not in chain:
                current = None

            if current is None:
                pick = next((h for h in chain if self._is_up(h)), None)
                if pick is None:
                    # Nothing observed up: prefer an unprobed candidate (we
                    # don't know it's down) over a known-down one.
                    pick = next((h for h in chain if self._is_up(h) is None), chain[0])
                self._owner[model_id] = pick
                return pick

            if (
                self._is_up(current) is False
                and self._state_age(current, now) >= s.fail_after_s
            ):
                pick = next((h for h in chain if self._is_up(h)), None)
                if pick is not None and pick != current:
                    self._owner[model_id] = pick
                    return pick
                return current

            if s.policy == POLICY_AUTO:
                for h in chain:
                    if h == current:
                        break
                    if (
                        self._is_up(h)
                        and self._state_age(h, now) >= s.failback_after_s
                    ):
                        self._owner[model_id] = h
                        return h
            return current

    def reset(self) -> None:
        """Drop all state (tests / config reload)."""
        with self._lock:
            self._hosts.clear()
            self._owner.clear()
        self._settings = None


TRACKER = OwnershipTracker()

# Model ids whose local process *this engine* started (failover bring-up).
# Only these are ever stopped on failback — a hand-started or autostarted
# instance is never touched (additive, same contract as fleet_reconcile).
_ENGINE_STARTED: set = set()


# ---------------------------------------------------------------------- #
# Chain + effective-owner resolution (request-path surface)
# ---------------------------------------------------------------------- #
def eligible_chain(model: Model) -> List[str]:
    """``model.host_chain`` filtered to hosts that actually list the model in
    their ``enabled:`` — a chain entry that never enabled the model can't
    serve it, so it is skipped at resolution time (issue #342's "(b) has the
    model installed/enabled"). Unknown host ids are dropped too.
    """
    from .host_profile import get_host

    out: List[str] = []
    # ``getattr`` (not the property) so duck-typed test fakes without a
    # ``hosts`` field read as unowned — same defensiveness the pre-#342
    # code applied to ``model.host``.
    for host_id in list(getattr(model, "hosts", None) or []):
        profile = get_host(host_id)
        if profile is not None and model.id in profile.enabled:
            out.append(host_id)
    return out


def effective_owner(model: Model) -> Optional[str]:
    """The host currently serving ``model`` — the dynamic counterpart of the
    static ``model.host``.

    Fast path: a single-host or unowned row returns ``model.host`` with no
    tracker involvement at all — pre-#342 rows keep pre-#342 cost and
    behavior. A multi-host row returns the tracker's current owner, falling
    back to the preferred (first) candidate until the loop has observed.
    """
    chain = list(getattr(model, "hosts", None) or [])
    if len(chain) <= 1:
        return getattr(model, "host", None)
    eligible = eligible_chain(model)
    if not eligible:
        return getattr(model, "host", None)
    if len(eligible) == 1:
        return eligible[0]
    return TRACKER.owner_for(model.id) or eligible[0]


def multi_host_models() -> List[Model]:
    """Enabled models with a real (multi-candidate) failover chain."""
    from .model_registry import enabled_models

    return [m for m in enabled_models() if len(eligible_chain(m)) > 1]


# ---------------------------------------------------------------------- #
# The observation/action pass + background loop
# ---------------------------------------------------------------------- #
async def _default_probe(host_id: str) -> bool:
    """Reachability of a chain candidate — the active host is trivially up
    (this code is running on it); peers reuse ``services.peer_health`` (the
    Machines-tab prober — LAN address with tailnet fallback, #396)."""
    from .host_profile import resolve as resolve_host

    if host_id == resolve_host().id:
        return True
    from . import services

    health = await services.peer_health(host_id)
    return bool(health.get("reachable"))


async def failover_pass(
    *,
    now: Optional[float] = None,
    probe: Optional[Callable[[str], Awaitable[bool]]] = None,
    start_local: Optional[Callable[[str], Awaitable[Any]]] = None,
    stop_local: Optional[Callable[[str], Awaitable[Any]]] = None,
    tracker: Optional[OwnershipTracker] = None,
) -> Dict[str, Any]:
    """One probe → decide → act cycle. All seams injectable for tests.

    Actions are strictly local (see module docstring): start the model here
    when this host became the effective owner; stop it here when ownership
    moved away **and** this engine had started it.
    """
    from .host_profile import resolve as resolve_host

    trk = tracker if tracker is not None else TRACKER
    t = time.time() if now is None else now
    do_probe = probe if probe is not None else _default_probe
    models = multi_host_models()
    if not models:
        return {}

    active_id = resolve_host().id
    chains = {m.id: eligible_chain(m) for m in models}
    hosts = sorted({h for chain in chains.values() for h in chain})
    for host_id in hosts:
        try:
            reachable = await do_probe(host_id)
        except Exception as exc:  # noqa: BLE001 — a probe error reads as down
            logger.warning("failover: probe %s raised: %s", host_id, exc)
            reachable = False
        trk.observe_host(host_id, reachable, t)

    results: Dict[str, Any] = {}
    for m in models:
        chain = chains[m.id]
        prev = trk.owner_for(m.id)
        owner = trk.decide(m.id, chain, t)
        if prev is not None and owner != prev:
            logger.info(
                "🔀 failover: %s owner %s -> %s (chain %s)", m.id, prev, owner, chain
            )
        entry: Dict[str, Any] = {"owner": owner, "previous": prev, "chain": chain}
        try:
            if owner == active_id:
                entry["action"] = await _ensure_running_local(m, start_local)
            else:
                entry["action"] = await _release_local(m, stop_local)
        except Exception as exc:  # noqa: BLE001 — one bad model must not abort the pass
            logger.warning("failover: action for %s raised: %s", m.id, exc)
            entry["action"] = {"error": str(exc)}
        results[m.id] = entry
    return results


async def _ensure_running_local(
    model: Model, start_local: Optional[Callable[[str], Awaitable[Any]]]
) -> Optional[Dict[str, Any]]:
    """Start ``model`` on this host if it isn't already serving (idempotent —
    ``backend_process.start`` adopts a live port and no-ops when running)."""
    from . import backend_process as bp

    if model.virtual or model.backend not in ("openai", "whisper", "tts"):
        return None
    # ``startup: on_demand`` (#422): ownership can move here, but nothing
    # spawns the model until a request does — an idle-unloaded on-demand
    # model must never be resurrected by this engine.
    if getattr(model, "startup", None) == "on_demand":
        return None
    if await asyncio.to_thread(bp.is_running, model.id):
        return None
    if start_local is not None:
        result = await start_local(model.id)
        _ENGINE_STARTED.add(model.id)
        return {"started": True, "detail": result}
    ok, msg = await asyncio.to_thread(bp.start, model.id)
    # An *adopted* external instance existed before this engine acted — it is
    # not engine-started, so failback must leave it alone (additive contract).
    if ok and "adopted" not in msg.lower():
        _ENGINE_STARTED.add(model.id)
    logger.info("failover: local start %s -> %s (%s)", model.id, ok, msg)
    return {"started": bool(ok), "detail": msg}


async def _release_local(
    model: Model, stop_local: Optional[Callable[[str], Awaitable[Any]]]
) -> Optional[Dict[str, Any]]:
    """Stop a *failover-started* local instance after ownership moved away.
    Never touches an instance this engine didn't start."""
    from . import backend_process as bp

    if model.id not in _ENGINE_STARTED:
        return None
    if not await asyncio.to_thread(bp.is_running, model.id):
        _ENGINE_STARTED.discard(model.id)
        return None
    if stop_local is not None:
        result = await stop_local(model.id)
        _ENGINE_STARTED.discard(model.id)
        return {"stopped": True, "detail": result}
    ok, msg = await asyncio.to_thread(bp.stop, model.id)
    _ENGINE_STARTED.discard(model.id)
    logger.info("failover: local stop %s -> %s (%s)", model.id, ok, msg)
    return {"stopped": bool(ok), "detail": msg}


async def failover_loop(boot_delay_s: float = 20.0) -> None:
    """Background loop wired by ``server_lifecycle`` — exits immediately when
    no enabled model declares a multi-host chain (the config contract is
    edit-YAML-then-restart, so there is nothing to re-check later)."""
    await asyncio.sleep(boot_delay_s)
    try:
        if not multi_host_models():
            logger.debug("failover: no multi-host model chains configured — loop idle")
            return
    except Exception as exc:  # noqa: BLE001
        logger.warning("failover: config scan failed: %s", exc)
        return
    settings = TRACKER.settings
    logger.info(
        "🔀 failover engine active: probe every %.0fs, fail after %.0fs, "
        "failback after %.0fs, policy %s",
        settings.probe_interval_s, settings.fail_after_s,
        settings.failback_after_s, settings.policy,
    )
    while True:
        try:
            await failover_pass()
        except Exception as exc:  # noqa: BLE001 — loop must not die
            logger.warning("failover pass raised: %s", exc)
        await asyncio.sleep(settings.probe_interval_s)
