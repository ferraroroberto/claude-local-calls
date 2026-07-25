"""Fleet reconcile — converge each host to its desired placement (#353).

Step 2 of the always-on control plane. Reads ``config/fleet_placement.json``
(``src/fleet_placement.py``) and, for every host with placed models, makes the
live fleet match intent:

  * an **unreachable** satellite that ``can_ssh`` → wake it via
    ``remote_bootstrap.bootstrap_host`` (the tower holds the forced-command
    key); once it answers, converge it in the same pass;
  * a **reachable remote** host → write its ``startup_profile`` through to the
    desired set (so it self-boots correctly on its *own* next reboot, via
    Step 1's host-addressable profile API) and start any placed model not up;
  * the **local** (control-node) host → start its placed models via
    ``backend_process.start`` directly.

The periodic pass (:func:`reconcile_once`) is **additive**: it starts missing
placed models but never stops one that was started by hand and isn't placed.
Explicit un-placement is a separate, deliberate action
(:func:`apply_placement_change`, driven by the placement API's PATCH) — it stops
the removed models and drops them from the host's profile.

**Maintenance gate (#411):** this loop's own always-on convergence used to race
``model_failover.py`` (#342) — a deliberately-stopped peer got SSH-resurrected
here within seconds, well inside ``model_failover``'s ``fail_after_s`` window,
so failover could never trigger on a drill-induced outage. :func:`reconcile_once`
now checks ``src.fleet_maintenance.is_under_maintenance`` per host and skips a
drained host entirely (no wake, no bootstrap, no profile write, no start) until
its window expires or is cleared via ``/admin/api/fleet-maintenance`` — see
that module's docstring for the full contract.

Everything leans on existing idempotency: ``backend_process.start`` adopts a
reachable port and no-ops if already running, and a forwarded ``/start`` returns
409 "already running" — both treated as success here — so the loop is safe to
run every few minutes forever. Framework-free (raw ``httpx`` peer calls,
soft-failing dicts) exactly like ``remote_bootstrap``, so it stays unit-testable
with no FastAPI app in the loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List

import httpx

from .wake_on_lan import WakeOnLanError, send_wake

logger = logging.getLogger(__name__)

# The reconcile cadence (issue #353): a pass on boot, then every few minutes.
# Env-overridable so a test rig or a jumpier fleet can retune without a code
# change ("configurable" per the issue). A short boot delay lets local autostart
# + backend inheritance settle first, so the first pass sees accurate running
# state instead of racing ``_autostart_configured_backends`` into a double-spawn.
FLEET_RECONCILE_INTERVAL_S = float(
    os.environ.get("LOCAL_LLM_HUB_FLEET_RECONCILE_INTERVAL_S", "300")
)
FLEET_RECONCILE_BOOT_DELAY_S = float(
    os.environ.get("LOCAL_LLM_HUB_FLEET_RECONCILE_BOOT_DELAY_S", "20")
)
_PEER_TIMEOUT_S = 30.0


# --------------------------------------------------------------------------- #
# Peer transport — raw httpx, soft-failing (no FastAPI HTTPException in a loop).
# --------------------------------------------------------------------------- #
def _peer_base(owner: Any) -> str:
    """Peer hub base URL via the #396 dial resolver — LAN address while it
    answers, tailnet name when it doesn't. The preceding ``peer_health`` call
    in every converge path has already warmed the resolver's last-known-good
    cache, so this is a dict lookup in practice, not a probe."""
    from . import remote_stats
    from .host_profile import hub_port
    address = remote_stats.dial_address(owner) or owner.address
    return f"http://{address}:{hub_port()}"


def _peer_headers(host_id: str) -> Dict[str, str]:
    from .remote_proxy import remote_auth_token
    token = remote_auth_token(host_id)
    return {"Authorization": f"Bearer {token}"} if token else {}


async def _remote_write_profile(host_id: str, base: str, models: List[str]) -> Dict[str, Any]:
    """PATCH a peer's own ``startup_profile`` to the desired model set.

    Sends only ``{"models": [...]}`` — the peer's profile PATCH merges it over
    that host's docker/langfuse/etc. flags and validates the ids against its
    *own* launchable set (#265). So the satellite self-boots the placed set on
    its next reboot even if the tower is down.
    """
    try:
        async with httpx.AsyncClient(timeout=_PEER_TIMEOUT_S) as client:
            r = await client.patch(
                f"{base}/admin/api/startup-profile",
                json={"models": list(models)},
                headers=_peer_headers(host_id),
            )
        return {"ok": r.status_code < 400, "status": r.status_code}
    except httpx.HTTPError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


async def _remote_model_action(host_id: str, base: str, model_id: str, action: str) -> Dict[str, Any]:
    """POST ``/admin/api/models/{id}/{start|stop}`` to a peer hub.

    409 ("already running" on start / "not running" on stop) is a benign no-op —
    the whole point of the additive loop is that repeated starts converge — so it
    counts as ``ok``.
    """
    try:
        async with httpx.AsyncClient(timeout=_PEER_TIMEOUT_S) as client:
            r = await client.post(
                f"{base}/admin/api/models/{model_id}/{action}",
                headers=_peer_headers(host_id),
            )
    except httpx.HTTPError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": r.status_code < 400 or r.status_code == 409, "status": r.status_code}


# --------------------------------------------------------------------------- #
# Per-host convergence.
# --------------------------------------------------------------------------- #
async def _reconcile_local(desired: List[str]) -> Dict[str, Any]:
    """Start the control node's own placed models directly (idempotent)."""
    from . import backend_process as bp

    started: List[Dict[str, Any]] = []
    for model_id in desired:
        try:
            ok, msg = await asyncio.to_thread(bp.start, model_id)
        except Exception as exc:  # noqa: BLE001 — a bad row must not abort the pass
            logger.warning("fleet reconcile: local start %s raised: %s", model_id, exc)
            started.append({"id": model_id, "ok": False, "detail": str(exc)})
            continue
        no_op = "already running" in msg.lower()
        started.append({"id": model_id, "ok": bool(ok) or no_op, "detail": msg})
        logger.info("fleet reconcile: local %s -> %s", model_id, msg)
    return {"local": True, "reachable": True, "started": started}


async def _reconcile_remote(host_id: str, desired: List[str]) -> Dict[str, Any]:
    """Wake (if needed) + write-through profile + start placed models on a peer."""
    from . import remote_bootstrap, services
    from .host_profile import get_host

    owner = get_host(host_id)
    if owner is None or not owner.address:
        return {"reachable": False, "error": "no address configured"}

    health = await services.peer_health(host_id)
    reachable = bool(health.get("reachable"))
    woke: Any = None
    wol_sent = False
    if not reachable:
        # True power-on beneath the SSH bootstrap (#364, Phase 2 of #356): a
        # MAC-registered satellite may be fully off, where SSH can't reach it.
        # Fire-and-continue by design — a cold boot takes minutes while this
        # pass's reachability budget is seconds, so we never wait on the wake
        # here; the next periodic pass finds the box up and converges.
        if owner.mac:
            try:
                await asyncio.to_thread(send_wake, owner.mac)
                wol_sent = True
                logger.info("fleet reconcile: WOL packet sent to %s (%s)", host_id, owner.mac)
            except WakeOnLanError as exc:
                logger.warning("fleet reconcile: WOL send to %s failed: %s", host_id, exc)
        if not owner.can_ssh:
            return {"reachable": False, "wol_sent": wol_sent, "error": "unreachable, cannot ssh"}
        woke = await remote_bootstrap.bootstrap_host(host_id)
        reachable = bool(woke.get("ok"))
        logger.info("fleet reconcile: wake %s -> reachable=%s", host_id, reachable)
        if not reachable:
            # Couldn't bring it up this pass — the next pass will retry.
            return {"reachable": False, "wol_sent": wol_sent, "woke": woke}

    base = _peer_base(owner)
    profile = await _remote_write_profile(host_id, base, desired)
    started: List[Dict[str, Any]] = []
    for model_id in desired:
        started.append({"id": model_id, **await _remote_model_action(host_id, base, model_id, "start")})
    return {
        "reachable": True,
        "wol_sent": wol_sent,
        "woke": woke,
        "profile_written": profile.get("ok"),
        "started": started,
    }


async def _reconcile_host(host_id: str, desired: List[str], active_id: str) -> Dict[str, Any]:
    if host_id == active_id:
        return await _reconcile_local(desired)
    return await _reconcile_remote(host_id, desired)


async def reconcile_once() -> Dict[str, Any]:
    """One additive convergence pass over the whole fleet placement.

    Starts every placed-but-not-running model (waking an offline can-ssh
    satellite first). Never stops anything — an un-placement is the caller's
    explicit :func:`apply_placement_change`, not this loop's job. A host with an
    empty placement is skipped entirely: no models to run means no reason to
    wake it.
    """
    from . import fleet_maintenance
    from .fleet_placement import load_fleet_placement
    from .host_profile import resolve as resolve_host

    placement = load_fleet_placement()
    active_id = resolve_host().id
    results: Dict[str, Any] = {}
    for host_id, desired in placement.items():
        if not desired:
            continue
        if fleet_maintenance.is_under_maintenance(host_id):
            logger.info("fleet reconcile: host %s is under maintenance — skipping", host_id)
            results[host_id] = {"maintenance": True, "reachable": None}
            continue
        try:
            results[host_id] = await _reconcile_host(host_id, list(desired), active_id)
        except Exception as exc:  # noqa: BLE001 — one bad host must not abort the sweep
            logger.warning("fleet reconcile: host %s raised: %s", host_id, exc)
            results[host_id] = {"ok": False, "error": str(exc)}
    return results


# --------------------------------------------------------------------------- #
# Explicit un-placement (PATCH-driven) — the one path allowed to stop things.
# --------------------------------------------------------------------------- #
def _drop_local_profile_models(removed: List[str]) -> None:
    """Drop un-placed ids from the *local* startup profile so they don't
    resurrect on the next reboot. Remote hosts get the same effect for free —
    the reconcile write-through rewrites their profile to the new desired set.
    """
    from .startup_profile import load_startup_profile, save_startup_profile

    current = load_startup_profile().as_dict()
    kept = [m for m in current.get("models", []) if m not in set(removed)]
    if kept != current.get("models", []):
        save_startup_profile({**current, "models": kept})


async def _unplace(host_id: str, removed: List[str], active_id: str) -> List[Dict[str, Any]]:
    from . import backend_process as bp
    from .host_profile import get_host

    results: List[Dict[str, Any]] = []
    if host_id == active_id:
        for model_id in removed:
            try:
                ok, msg = await asyncio.to_thread(bp.stop, model_id)
            except Exception as exc:  # noqa: BLE001
                results.append({"id": model_id, "ok": False, "detail": str(exc)})
                continue
            results.append({"id": model_id, "ok": bool(ok), "detail": msg})
            logger.info("fleet reconcile: local stop %s -> %s", model_id, msg)
        _drop_local_profile_models(removed)
        return results

    owner = get_host(host_id)
    if owner is None or not owner.address:
        return [{"id": m, "ok": False, "error": "no address configured"} for m in removed]
    base = _peer_base(owner)
    for model_id in removed:
        results.append({"id": model_id, **await _remote_model_action(host_id, base, model_id, "stop")})
    return results


async def apply_placement_change(
    host_id: str, old_ids: List[str], new_ids: List[str], active_id: str
) -> Dict[str, Any]:
    """Apply a single host's placement delta immediately (PATCH-driven, #353).

    Removed ids are **stopped** and de-profiled (the only path that stops a
    model — the periodic loop never does). Then the host is converged additively
    to ``new_ids`` (start newly-placed / write-through the peer profile). A
    remote host's de-profiling happens implicitly in the converge write-through,
    which rewrites its profile to exactly ``new_ids`` — including when
    ``new_ids`` is empty (#360): skipping the converge entirely there left the
    peer's profile holding the un-placed model, resurrecting it on the peer's
    next reboot. The empty-set write-through goes straight to the profile PATCH
    (reachability-checked, soft-fail) — never the wake/bootstrap machinery,
    which exists to *start* things, not to erase a line from a profile.
    """
    removed = [m for m in old_ids if m not in set(new_ids)]
    stopped = await _unplace(host_id, removed, active_id) if removed else []
    if new_ids:
        converged = await _reconcile_host(host_id, list(new_ids), active_id)
    elif removed and host_id != active_id:
        converged = await _deprofile_remote(host_id)
    else:
        converged = {}
    return {"stopped": stopped, "converged": converged}


async def _deprofile_remote(host_id: str) -> Dict[str, Any]:
    """Rewrite a remote peer's profile to no models (last-model un-place, #360).

    Reachable peer → PATCH ``{"models": []}``; unreachable/unknown peer →
    soft-fail (the stale profile entry survives until the peer is next up, but
    the un-place itself never errors on it)."""
    from . import services
    from .host_profile import get_host

    owner = get_host(host_id)
    if owner is None or not owner.address:
        return {"reachable": False, "error": "no address configured"}
    health = await services.peer_health(host_id)
    if not health.get("reachable"):
        return {"reachable": False, "profile_written": False}
    profile = await _remote_write_profile(host_id, _peer_base(owner), [])
    return {"reachable": True, "profile_written": profile.get("ok")}
