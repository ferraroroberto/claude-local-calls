"""Fleet maintenance API (#411) — the tower's drain-marker control surface
that closes the ``fleet_reconcile.py``/``model_failover.py`` race.

  * ``GET    /api/fleet-maintenance``            → active (non-expired) drains.
  * ``POST   /api/fleet-maintenance/{host_id}``   → arm a drain window.
  * ``DELETE /api/fleet-maintenance/{host_id}``   → clear it, then run one
    immediate reconcile pass so "reconcile resumes healing once cleared" is
    instantly observable instead of waiting up to
    ``FLEET_RECONCILE_INTERVAL_S``.

Local to the tower exactly like ``fleet_placement``'s router — the same
control-plane surface, same thin-CRUD-shell shape.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from src import fleet_maintenance as fm
from src import fleet_reconcile

from ._helpers import maybe_json

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/fleet-maintenance")
async def get_fleet_maintenance() -> Dict[str, Any]:
    return {"maintenance": fm.maintenance_status()}


@router.post("/api/fleet-maintenance/{host_id}")
async def arm_fleet_maintenance(host_id: str, request: Request) -> Dict[str, Any]:
    body = await maybe_json(request)
    duration_s = body.get("duration_s", fm.DEFAULT_MAINTENANCE_S)
    reason = body.get("reason", "")
    try:
        duration_s = float(duration_s)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="duration_s must be a number")
    try:
        entry = fm.set_maintenance(host_id, duration_s=duration_s, reason=reason)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, **entry}


@router.delete("/api/fleet-maintenance/{host_id}")
async def clear_fleet_maintenance(host_id: str) -> Dict[str, Any]:
    cleared = fm.clear_maintenance(host_id)
    results = await fleet_reconcile.reconcile_once()
    return {"ok": True, "cleared": cleared, "reconcile": results}
