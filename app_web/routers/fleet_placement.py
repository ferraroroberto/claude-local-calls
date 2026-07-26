"""Fleet placement API — the fleet's registry-derived desired-state view.

Step 2 of the always-on control plane (#353), re-sourced in #430: the desired
placement is **derived from ``config/models.yaml``**
(``model_registry.desired_placement()`` — ``startup: eager`` rows on their
preferred chain host), not a separate editable file — the old
``config/fleet_placement.json`` + its PATCH surface were retired because they
duplicated (and drifted from) the registry's ``hosts:`` chains + ``startup:``
policy. This router is now a read-only status surface plus the on-demand
reconcile trigger; changing *what runs where* is a ``models.yaml`` edit
(``/admin/api/config/*`` cards or swap-model), applied by the reconcile loop.

  * ``GET   /api/fleet-placement`` → derived placement + per-host status.
  * ``POST  /api/fleet-placement/reconcile`` → run one additive convergence pass
    on demand (the loop already does this on boot + every few minutes).

Local to the tower (the control node) in practice, but harmless anywhere —
every hub derives the same placement from the same synced registry.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter

from src import backend_process as bp
from src import fleet_reconcile, remote_stats, services as svc, system_stats
from src.host_profile import HostProfile, all_hosts, resolve as resolve_host
from src.model_registry import (
    all_models,
    cpu_resident_map,
    desired_placement,
    launchable_local_ids,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Live-running badges come from a manageable peer's own hub models API. Bound it
# short so a peer that's powered on but whose hub is slow/absent doesn't stall an
# on-demand tab-open — the box's own TCP liveness (below) already settles online
# vs offline; the models call only enriches the badges.
_GRID_PROBE_TIMEOUT_S = 2.5


def _display_names() -> Dict[str, str]:
    return {m.id: m.display_name for m in all_models()}


def _vram_estimates() -> Dict[str, int]:
    """``{model_id: est_vram_mb}`` for every model that declares a footprint
    (#375). A row without ``est_vram_mb`` is absent — the capacity sum treats a
    missing id as 0, so subscription/virtual/CPU rows contribute nothing."""
    return {m.id: m.est_vram_mb for m in all_models() if m.est_vram_mb is not None}


def _device_hints() -> Dict[str, Dict[str, str]]:
    """``{host_id: {model_id: "cpu"}}`` — CPU residency is per *(model, host)*,
    so the summary can show a small 'cpu' hint per row (#387) — a model
    contributing 0 to the VRAM sum reads as *intentionally exempt*, not as an
    omission. Thin view over ``model_registry.cpu_resident_map()`` (#431) —
    the same source the capacity sum excludes, so hint and math can't drift.
    """
    return {
        host_id: {mid: "cpu" for mid in ids}
        for host_id, ids in cpu_resident_map().items()
    }


def _gpu_snapshot(gpus: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """First GPU's live ``{used_mb, total_mb}`` out of a stats probe (#436) —
    the same figures the Machines tab renders, so the two tabs can never
    disagree while both are live. ``None`` when the host reports no GPU
    metric (no nvidia-smi — the Mac's unified memory, a failed probe), in
    which case the UI falls back to the ``~`` static estimate. First entry
    only: every probe in this fleet reports at most one GPU per host."""
    for g in gpus or []:
        if g.get("total_mb"):
            return {"used_mb": g.get("used_mb"), "total_mb": g.get("total_mb")}
    return None


def _capacity(
    profile: HostProfile,
    placed: List[str],
    running: List[str],
    vram: Dict[str, int],
    devices: Dict[str, str],
) -> Dict[str, Any]:
    """The host's VRAM headroom against its declared ceiling (#375).

    Sums ``est_vram_mb`` over the union of *placed* (desired) and *running*
    (live) model ids — a model can be either without the other, and both draw
    VRAM. Rows resident on **CPU on this host** (``devices`` — piper, ``-ng``
    whisper rows, a chain's degraded ``cpu: true`` tier) are excluded: they
    hold no GPU VRAM, so counting them faked an overcommit on the tower
    (#431). The result is **advisory**: ``capacity_warning`` is True only when
    the host declares a ``vram_mb`` ceiling AND the estimate exceeds it. A
    host with no ceiling (Apple-silicon unified memory, managed-only boxes)
    never warns — ``vram_mb`` is None and the sum is reported for context only.
    """
    considered = list(dict.fromkeys([*placed, *running]))
    est = sum(vram.get(m, 0) for m in considered if devices.get(m) != "cpu")
    ceiling = profile.vram_mb
    return {
        "vram_mb": ceiling,
        "est_vram_mb": est,
        "capacity_warning": ceiling is not None and est > ceiling,
    }


async def _host_status(
    profile: HostProfile,
    active_id: str,
    placement: Dict[str, List[str]],
    names: Dict[str, str],
    vram: Dict[str, int],
    devices: Dict[str, str],
) -> Dict[str, Any]:
    """One host's grid row: its launchable models, live status, capacity
    headroom, and whether the control plane can manage it.

    Reachability is the **hub-independent TCP liveness** the Machines tab uses
    (``remote_stats.is_reachable`` — *is the box powered on?*), not a hub
    ``/health`` probe, so a managed-only satellite that runs no hub (``gaming``,
    ``openclaw``) still reads "online" honestly. ``runs_hub`` (a host declares
    launchable models) tells the UI whether a hub answers there; a host with
    none is shown with an honest note rather than an empty grid cell.
    """
    hid = profile.id
    eligible_ids = launchable_local_ids(profile)
    eligible_set = set(eligible_ids)
    eligible = [
        {"id": m, "display_name": names.get(m, m), "device": devices.get(m)}
        for m in eligible_ids
    ]
    placed = placement.get(hid, [])
    runs_hub = bool(eligible_ids)  # only a host with launchable models runs this hub

    base = {
        "id": hid, "display_name": profile.display_name or hid,
        "icon": profile.icon or ("monitor" if hid == active_id else "server"),
        "can_ssh": profile.can_ssh, "runs_hub": runs_hub,
        "eligible": eligible, "placed": placed,
        # Display-only capacity context (#431) — total system RAM where the
        # machines registry documents it; None where the fact isn't known.
        "ram_mb": profile.ram_mb,
        # Live RAM snapshot (#434) — the same {used_gb, total_gb, percent}
        # block the Machines tab shows, so the capacity line can read
        # used/total instead of just the declared fact. Filled per branch
        # below; None wherever no live figure is available (host off, no SSH),
        # in which case the UI falls back to the declared `ram_mb` total.
        "ram": None,
        # Live GPU snapshot (#436) — {used_mb, total_mb} from the same probe
        # plumbing the Machines tab reads (local nvidia-smi, cached SSH stats
        # on peers), so the capacity line and the Machines tab can never
        # disagree. None where no live figure exists (host off, no GPU
        # metric) — the UI then falls back to the ~est_vram_mb estimate vs
        # the declared ceiling.
        "gpu": None,
    }

    if hid == active_id:
        # Only the launchable models that are up — so a summary row reads
        # honestly. Excludes subscription + virtual rows.
        running = [m for m in bp.running_backends().keys() if m in eligible_set]
        # A live backend whose adopted PID is a *foreign* process (an external
        # sibling on a mutex-shared port — voice-transcriber's whisper-server
        # on :8090) is flagged so the summary can label it distinctly rather
        # than claim the hub runs it (#431).
        external = [m for m in running if bp.inherited_foreign(m)]
        return {
            **base, "local": True, "reachable": True, "dormant": False,
            "running": running, "external": external,
            "ram": system_stats.ram_stats(),
            "gpu": _gpu_snapshot(system_stats.gpu_stats()),
            **_capacity(profile, placed, running, vram, devices),
        }

    # A peer: liveness by TCP connect (is the box on?), independent of whether it
    # runs a hub. A dormant node is never live-probed (it's declared powered down).
    reachable = False if profile.dormant else await remote_stats.is_reachable(profile)
    running: List[str] = []
    ram = None
    gpu = None
    if reachable and runs_hub:
        # Only a hub-running peer exposes a models API for live running badges.
        rows = await svc.remote_models(profile, timeout_s=_GRID_PROBE_TIMEOUT_S) or []
        running = [
            r["id"] for r in rows
            if isinstance(r, dict) and r.get("id") in eligible_set and r.get("reachable")
        ]
    if reachable and profile.can_ssh:
        # Live RAM + GPU over the same cached SSH probe the Machines tab uses
        # (#434, GPU #436) — remote_stats.collect keeps a 30 s cache and this
        # GET is tab-triggered (never polled), so no SSH storm. Best-effort: a
        # failed probe leaves both None and the UI shows the declared total /
        # the ~estimate.
        stats = await remote_stats.collect(profile)
        ram = stats.get("ram") if stats else None
        gpu = _gpu_snapshot(stats.get("gpus")) if stats else None
    return {
        **base, "local": False, "reachable": reachable,
        "dormant": profile.dormant, "running": running, "external": [],
        "ram": ram, "gpu": gpu,
        **_capacity(profile, placed, running, vram, devices),
    }


@router.get("/api/fleet-placement")
async def get_fleet_placement() -> Dict[str, Any]:
    """Registry-derived desired placement + a row for **every** fleet host: its
    launchable models, live liveness, and whether it's manageable from here."""
    placement = desired_placement()
    active_id = resolve_host().id
    names = _display_names()
    vram = _vram_estimates()
    devices = _device_hints()
    statuses = await asyncio.gather(
        *(
            _host_status(h, active_id, placement, names, vram, devices.get(h.id, {}))
            for h in all_hosts()
        )
    )
    return {"placement": placement, "hosts": list(statuses)}


@router.post("/api/fleet-placement/reconcile")
async def reconcile_now() -> Dict[str, Any]:
    """Run one additive convergence pass on demand (same as the periodic loop)."""
    return {"ok": True, "results": await fleet_reconcile.reconcile_once()}
