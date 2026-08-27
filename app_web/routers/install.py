"""Install tab API — first-run checks + one-click fixes.

Endpoints (all under /admin/api/install):
  * GET  /status    — run every check, return worst_status/ok/checks
  * POST /fix       — run one fix by fix_id
  * POST /fix-all   — run every currently-fixable check

Split out of ``hub.py`` (issue #533) — that router's own docstring flagged
this as "bolted on": these three endpoints wrap ``src/install.py``'s
check/fix battery and have nothing to do with the hub-process
status/control/log/request-stream surface the rest of that file owns.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request

from ._helpers import maybe_json

router = APIRouter()


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
