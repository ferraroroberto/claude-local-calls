"""Host-scoped maintenance/drain marker — the contract gap between
``fleet_reconcile.py`` (#353) and ``model_failover.py`` (#342), closed by
#411.

Both loops hold independent opinions about a down peer hub: reconcile treats
it as drift and SSH-wakes it every ``FLEET_RECONCILE_INTERVAL_S`` (+ a boot
pass); failover treats it as a dead owner and moves ownership after
``fail_after_s`` of continuous downtime. Reconcile usually wins the race
before failover's window is ever satisfied. This module gives an operator an
explicit, host-scoped way to tell reconcile "leave this host alone for a
while" — armed/cleared via the ``/admin/api/fleet-maintenance`` router,
replacing the old workaround of hand-editing
``LOCAL_LLM_HUB_FLEET_RECONCILE_INTERVAL_S``/``..._BOOT_DELAY_S`` in ``.env``.

Mirrors ``src/startup_profile.py``'s conventions: tolerant load (a broken or
absent file never raises), atomic save, a path-keyed cache. Local to the
tower — ``config/fleet_maintenance.json`` is gitignored, no committed example
(absent file = nothing under maintenance).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAINTENANCE_PATH = PROJECT_ROOT / "config" / "fleet_maintenance.json"

# A drain covers the default fail_after_s (90s) with comfortable margin
# without being so long a forgotten toggle wedges a host out of reconcile
# indefinitely — MAX_MAINTENANCE_S is the hard ceiling on any single request.
DEFAULT_MAINTENANCE_S = 900.0
MAX_MAINTENANCE_S = 3600.0

# Parsed cache keyed by the resolved path — same shape as
# startup_profile._PROFILE_CACHE so swapping DEFAULT_MAINTENANCE_PATH in
# tests transparently busts the cache instead of returning a stale hit.
_MAINTENANCE_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {}


def _coerce(data: Any) -> Dict[str, Dict[str, Any]]:
    """Best-effort shape a loaded JSON blob into ``{host_id: {until, reason}}``.

    Tolerant by construction — ``load_fleet_maintenance`` must never raise, so
    a non-dict blob becomes ``{}`` and a malformed row is dropped rather than
    crashing the reconcile loop that reads it.
    """
    if not isinstance(data, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for host_id, row in data.items():
        if not isinstance(row, dict):
            continue
        try:
            until = float(row.get("until"))
        except (TypeError, ValueError):
            continue
        out[str(host_id)] = {"until": until, "reason": str(row.get("reason") or "")}
    return out


def load_fleet_maintenance(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Load the maintenance map. Missing/unparseable file → an empty mapping."""
    target = Path(path) if path else DEFAULT_MAINTENANCE_PATH
    key = str(target)
    cached = _MAINTENANCE_CACHE.get(key)
    if cached is not None:
        return cached

    if not target.exists():
        result: Dict[str, Dict[str, Any]] = {}
    else:
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("⚠️ could not load fleet maintenance %s: %s", target, exc)
            data = None
        result = _coerce(data)

    _MAINTENANCE_CACHE[key] = result
    return result


def _save(data: Dict[str, Dict[str, Any]], path: Optional[str] = None) -> None:
    target = Path(path) if path else DEFAULT_MAINTENANCE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    _MAINTENANCE_CACHE.pop(str(target), None)


def is_under_maintenance(
    host_id: str, now: Optional[float] = None, path: Optional[str] = None
) -> bool:
    """Is ``host_id`` currently drained? An expired ``until`` reads as
    ``False`` with no write needed — expiry is derived from time, not a
    distinct state transition."""
    t = time.time() if now is None else now
    row = load_fleet_maintenance(path).get(host_id)
    return row is not None and float(row["until"]) > t


def maintenance_status(
    now: Optional[float] = None, path: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """Active-only view for the API — expired rows dropped, each entry
    annotated with ``remaining_s``."""
    t = time.time() if now is None else now
    out: Dict[str, Dict[str, Any]] = {}
    for host_id, row in load_fleet_maintenance(path).items():
        until = float(row["until"])
        if until > t:
            out[host_id] = {"until": until, "reason": row["reason"], "remaining_s": until - t}
    return out


def set_maintenance(
    host_id: str,
    duration_s: float = DEFAULT_MAINTENANCE_S,
    reason: str = "",
    now: Optional[float] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Arm a drain window for ``host_id``. Raises ``ValueError`` on an
    unknown host or a non-positive duration; silently clamps an
    over-long duration to ``MAX_MAINTENANCE_S`` rather than rejecting it —
    a fat-fingered value should still arm a bounded window, not error out.
    Opportunistically prunes any other already-expired rows on write, so
    the file self-cleans without a separate sweep task.
    """
    from src.host_profile import get_host

    if get_host(host_id) is None:
        raise ValueError(f"unknown host {host_id!r}")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    duration_s = min(float(duration_s), MAX_MAINTENANCE_S)

    t = time.time() if now is None else now
    current = dict(load_fleet_maintenance(path))
    current = {h: row for h, row in current.items() if float(row["until"]) > t}
    until = t + duration_s
    current[host_id] = {"until": until, "reason": str(reason or "")}
    _save(current, path)
    logger.info(
        "🚧 fleet maintenance: %s armed for %.0fs (until %.0f)%s",
        host_id, duration_s, until, f" — {reason}" if reason else "",
    )
    return {"host_id": host_id, "until": until, "reason": str(reason or ""), "remaining_s": duration_s}


def clear_maintenance(host_id: str, path: Optional[str] = None) -> bool:
    """Clear ``host_id``'s drain window. Idempotent — returns whether a row
    was actually present."""
    current = dict(load_fleet_maintenance(path))
    present = current.pop(host_id, None) is not None
    if present:
        _save(current, path)
        logger.info("🚧 fleet maintenance: %s cleared", host_id)
    return present
