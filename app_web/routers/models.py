"""Models tab API — per-backend tile state + start/stop/force-stop/ping."""

from __future__ import annotations

import asyncio
import io
import logging
import time
import wave
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from src import backend_process as bp
from src import config_write
from src import services as svc
from src.host_profile import all_hosts, get_host, resolve as resolve_host
from src.model_failover import effective_owner
from src.model_registry import (
    Model,
    cpu_resident_map,
    enabled_models,
    resolve as resolve_model,
)
from src.remote_proxy import remote_auth_token_for_model, remote_base_url
from app_web.admin_forward import forward_admin_request
from src.server_process import (
    OWNERSHIP_EXTERNAL,
    OWNERSHIP_NONE,
    OWNERSHIP_OURS,
    snapshot_listening_pids,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _remote_admin_headers(model: Model) -> Dict[str, str]:
    token = remote_auth_token_for_model(model)
    return {"Authorization": f"Bearer {token}"} if token else {}


def _offline_remote_row(m: Model, host_id: str) -> Dict[str, Any]:
    """Fallback row for a remote-owned model when its owning hub couldn't
    be reached — shown as unreachable rather than silently dropped from
    the list (a remote host being offline shouldn't hide the model that
    normally lives there).
    """
    return {
        "id": m.id,
        "display_name": m.display_name,
        "backend": m.backend,
        "engine": m.engine,
        "port": m.port,
        "url": None,
        "aliases": list(m.aliases or []),
        "controllable": m.backend in ("openai", "whisper", "tts") and not m.virtual,
        "ownership": OWNERSHIP_NONE,
        "pid": None,
        "reachable": False,
        "model_path": m.model_path,
        "host": host_id,
        "host_unreachable": True,
    }


def _add_failover_fields(row: Dict[str, Any], m: Model, owner_id: str) -> None:
    """Annotate a tile row with #342 chain state — only for multi-host rows.

    ``preferred_host`` is the chain's first candidate; ``failover: true``
    flags that the model is currently served off-preference (failed over),
    which the SPA renders as an amber hint on the tile's meta line.
    """
    if len(m.host_chain) <= 1:
        return
    row["preferred_host"] = m.host_chain[0]
    row["failover"] = owner_id != m.host_chain[0]


def _add_placement_fields(
    row: Dict[str, Any], m: Model, cpu_map: Dict[str, set]
) -> None:
    """Annotate a tile row with its declared placement *intent* (#423) — the
    Phase 1 (#422) registry fields the read-only placement card renders.

    Config-derived from the local registry, never trusted from a peer: the
    YAML is the fleet-wide source of truth and every hub reads the same file,
    so stamping it locally keeps the payload shape independent of the owning
    hub's version. Subscription rows (claude/gemini) have no placement
    concept — no chain, no process, no VRAM — so they carry no ``placement``
    key at all rather than an empty shell the UI would have to special-case.
    """
    if m.backend in ("claude", "gemini"):
        return
    row["placement"] = {
        # Declared chain in priority order; ``cpu`` marks the *effective*
        # device on that host (#434) — sourced from
        # ``model_registry.cpu_resident_map`` (the same source the fleet
        # summary's device hints read), so an always-CPU row (piper, a
        # whisper ``-ng`` row, #265) is tagged everywhere, not just a chain's
        # degraded ``cpu: true`` tier (#342). A bare ``host:`` row is a
        # 1-element chain.
        "chain": [
            {"id": h, "cpu": m.id in cpu_map.get(h, ())} for h in m.host_chain
        ],
        "startup": m.startup,
        "idle_unload_minutes": m.idle_unload_minutes,
        "est_vram_mb": m.est_vram_mb,
        # #424: a virtual alias shares its parent row's process — its
        # placement is the parent's, so the editor never opens on it.
        "editable": not m.virtual,
    }


def _config_block() -> Dict[str, Any]:
    """Config-as-code context for the Models tab (#424): the models.yaml
    HEAD sha (the drift-visible config version), whether *this* hub may
    write (single-writer contract — tower only), and the full fleet host
    list the chain editor offers (with ceilings for context)."""
    return {
        "sha": config_write.config_sha(),
        "write_enabled": config_write.is_write_host(),
        "write_host": config_write.write_host_id(),
        "fleet_hosts": [{"id": h.id, "vram_mb": h.vram_mb} for h in all_hosts()],
    }


def _require_model(model_id: str) -> Model:
    """Resolve ``model_id`` via ``bp.resolve_model_by_id`` or raise the same
    404 every start/stop/force-stop/log handler needs — the identical
    resolve-or-404 block each one opened with before this helper existed.
    """
    target = bp.resolve_model_by_id(model_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"model {model_id!r} not enabled")
    return target


async def _forward_admin_call(
    target: Model, method: str, suffix: str, **kwargs: Any
) -> Dict[str, Any]:
    """Forward an admin models-API call to the host that actually owns
    ``target`` (#178) — used by start/stop/force-stop/log when the
    resolved model isn't local. Mirrors the local handlers' error shape
    (404/400/409 from the remote surface verbatim; 502 if the remote
    hub itself is unreachable) via the shared ``forward_admin_request``.
    """
    remote = remote_base_url(target)
    assert remote is not None
    owner = effective_owner(target) or target.host
    return await forward_admin_request(
        remote,
        f"/admin/api/models/{target.id}{suffix}",
        method=method,
        headers=_remote_admin_headers(target),
        unreachable_detail=f"host {owner!r} (owns {target.id!r}) unreachable",
        **kwargs,
    )


def _ownership_from_snapshot(m: Model, listening: Dict[int, list]) -> tuple[str, Any]:
    """Compute (ownership, pid) for a controllable model from a port→pids map.

    Avoids the per-model netstat invocation that ``bp.ownership`` does
    when checking each model in isolation.
    """
    if bp.is_running(m.id):
        return OWNERSHIP_OURS, bp.pid(m.id)
    if not m.port:
        return OWNERSHIP_NONE, None
    pids = listening.get(m.port) or []
    if pids:
        return OWNERSHIP_EXTERNAL, pids[0]
    return OWNERSHIP_NONE, None


@router.get("/api/models")
async def list_models_for_admin(local_only: bool = False) -> Dict[str, Any]:
    """Per-tile state for every enabled model — local rows computed here,
    plus (#178) any remote-owned rows merged in from the host that
    actually runs them.

    ``local_only=true`` skips the remote-merge step and returns just this
    host's own rows — used when a peer hub fetches *this* endpoint to
    build its own merge (``svc.remote_models``). Without it, two
    bidirectionally cross-enabled hosts recurse into each other forever:
    A's merge calls B's `/api/models`, which (unless told not to) tries to
    merge in A's rows by calling A's `/api/models` again, and so on.

    Two pieces are expensive: probing reachability over HTTP per backend
    (each costs up to 0.5 s) and resolving port → PID via netstat (one
    shell-out per call). We fan the HTTP probes out concurrently, and
    do a single netstat snapshot up front instead of one per backend —
    O(N) → O(1) subprocesses, O(N) → O(0.5 s) wall time when all
    backends are alive.

    A reachable TTS row also gets a second, equally-fanned-out probe for
    its resolved ``device`` (cuda/cpu/mps) — see ``_probe_device`` below.
    """
    active = resolve_host()
    all_enabled = list(enabled_models())
    # Split by *effective* owner (#342): a multi-host-chain model counts as
    # local while this host currently serves it (failover), and as remote
    # while another candidate does — so the tiles always describe the host
    # actually running the process. Single-host rows resolve statically to
    # their ``host:`` exactly as before.
    owner_by_id = {m.id: effective_owner(m) for m in all_enabled}
    local_models = [m for m in all_enabled if owner_by_id[m.id] in (None, active.id)]
    remote_owned = [m for m in all_enabled if owner_by_id[m.id] not in (None, active.id)]

    # psutil gives us every listening port in ~2 ms — use it both for
    # ownership *and* as a cheap reachability gate so we never fire an
    # HTTP probe at a port that isn't bound.
    listening = await asyncio.to_thread(snapshot_listening_pids)

    async def _probe_reach(m: Model) -> bool:
        if m.backend == "claude" or m.backend == "gemini":
            # Subscription-backed — always "live" if the hub itself
            # answered, which the caller already knows it did.
            return True
        if not m.port or m.port not in listening:
            # Port isn't bound → definitely not reachable; skip the
            # 1-second-per-dead-backend HTTP probe.
            return False
        return await asyncio.to_thread(bp.is_reachable, m, 0.4)

    reach_results = await asyncio.gather(*(_probe_reach(m) for m in local_models))

    async def _probe_device(m: Model, reachable: bool) -> Optional[str]:
        # TTS backends resolve a real device (cuda/cpu/mps) at load time and
        # report it on their own /health (tts_server.py's state.device) —
        # other backends have no comparable concept. Only probe a row that's
        # already known-reachable, and only trust a fully-resolved value: a
        # backend still loading reports its raw "auto" arg, and surfacing
        # that would be actively misleading — omit rather than guess wrong.
        if not reachable or m.backend != "tts":
            return None
        body = await asyncio.to_thread(bp.probe_health, m, 0.4)
        dev = body.get("device") if isinstance(body, dict) else None
        if isinstance(dev, str) and dev.strip().lower() in ("cpu", "cuda", "mps"):
            return dev.strip().lower()
        return None

    device_results = await asyncio.gather(
        *(_probe_device(m, reachable) for m, reachable in zip(local_models, reach_results))
    )

    rows: List[Dict[str, Any]] = []
    cpu_map = cpu_resident_map()  # effective per-(model, host) device (#434)
    for m, reachable, device in zip(local_models, reach_results, device_results):
        # Virtual aliases share an existing backend's port and own no process,
        # so they're reachable but never independently start/stop-able.
        controllable = m.backend in ("openai", "whisper", "tts") and not m.virtual
        own = OWNERSHIP_NONE
        pid: Any = None
        if controllable:
            own, pid = _ownership_from_snapshot(m, listening)
        row: Dict[str, Any] = {
            "id": m.id,
            "display_name": m.display_name,
            "backend": m.backend,
            "engine": m.engine,
            "port": m.port,
            "url": m.url,
            "aliases": list(m.aliases or []),
            "controllable": controllable,
            "ownership": own,
            "pid": pid,
            "reachable": bool(reachable),
            "model_path": m.model_path,
            "host": active.id,
        }
        _add_failover_fields(row, m, active.id)
        _add_placement_fields(row, m, cpu_map)
        if device:
            row["device"] = device
        rows.append(row)

    if local_only:
        return {"models": rows, "config": _config_block()}

    # Remote-owned rows: one fetch per distinct owning host, merged in.
    # Trust the owner's own reachable/ownership/pid values — this hub has
    # no local visibility into another machine's ports.
    owners: Dict[str, List[Model]] = {}
    for m in remote_owned:
        owners.setdefault(owner_by_id[m.id], []).append(m)

    for host_id, models_for_host in owners.items():
        owner_profile = get_host(host_id)
        fetched = await svc.remote_models(owner_profile) if owner_profile else None
        by_id = {r.get("id"): r for r in fetched if isinstance(r, dict)} if fetched is not None else None
        for m in models_for_host:
            remote_row = by_id.get(m.id) if by_id is not None else None
            if remote_row is not None:
                row = dict(remote_row)
                row.setdefault("host", host_id)
            else:
                row = _offline_remote_row(m, host_id)
            _add_failover_fields(row, m, host_id)
            _add_placement_fields(row, m, cpu_map)
            rows.append(row)

    return {"models": rows, "config": _config_block()}


# --------------------------------------------------------------------- #
# Editable placement (#424) — the write-through-to-git path. Tower-only;
# validation + the git transaction live in src.config_write.
# --------------------------------------------------------------------- #

# Strong references to in-flight peer-sync tasks — a bare create_task result
# is GC-eligible and the sync would silently die mid-flight.
_PEER_SYNC_TASKS: set = set()


def _schedule_peer_sync() -> None:
    """Fire the #181 sync (git pull + hub restart) at every hub peer after a
    successful config push — the immediate leg of propagation; the periodic
    drift loop (``config_drift_sync_loop``) is the catch-up net."""
    from src import remote_bootstrap
    from src.model_registry import hub_peer_ids

    async def _sync(peer: str) -> None:
        try:
            result = await remote_bootstrap.sync_host(peer)
            logger.info("🔃 post-write sync of %s: %s", peer, result)
        except Exception as exc:  # noqa: BLE001 — drift loop retries later
            logger.warning("post-write sync of %s raised: %s", peer, exc)

    for peer in hub_peer_ids():
        task = asyncio.create_task(_sync(peer))
        _PEER_SYNC_TASKS.add(task)
        task.add_done_callback(_PEER_SYNC_TASKS.discard)


@router.put("/api/models/{model_id}/placement")
async def model_placement_update(model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Update a model's placement in config/models.yaml via the git-backed
    write path (#424): validate (schema + #375 VRAM budget, hard-reject) →
    comment-preserving YAML edit → config-bot commit → push to origin main
    → background peer sync. 403 on every host but the declared writer.
    """
    if not config_write.is_write_host():
        writer = config_write.write_host_id() or "(none configured)"
        raise HTTPException(
            status_code=403,
            detail=f"config writes are only allowed on host {writer!r} — "
                   f"this hub is {resolve_host().id!r}",
        )
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    chain, shape_errors = config_write.normalize_chain(payload.get("hosts"))
    if shape_errors:
        raise HTTPException(status_code=400, detail="; ".join(shape_errors))
    startup = str(payload.get("startup") or "").strip().lower()
    raw_idle = payload.get("idle_unload_minutes")
    idle: Any = raw_idle
    if isinstance(raw_idle, float) and raw_idle.is_integer():
        idle = int(raw_idle)
    try:
        result = await asyncio.to_thread(
            config_write.apply_placement, model_id, chain, startup, idle
        )
    except config_write.ConfigWriteError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))
    if result.get("changed"):
        _schedule_peer_sync()
    return result


@router.post("/api/models/{model_id}/start")
async def model_start(model_id: str) -> Dict[str, Any]:
    target = _require_model(model_id)
    if remote_base_url(target):
        return await _forward_admin_call(target, "POST", "/start")
    if target.virtual:
        raise HTTPException(
            status_code=400,
            detail=f"model {model_id!r} is a virtual alias of another backend — nothing to start",
        )
    if not (target.backend in ("openai", "whisper", "tts")):
        raise HTTPException(
            status_code=400,
            detail=f"backend {target.backend!r} has no managed process (subscription-backed)",
        )
    ok, msg = bp.start(model_id)
    if not ok:
        # "already running" is OK in the SPA — surface as 409 so the UI can ignore.
        raise HTTPException(status_code=409, detail=msg)
    return {"ok": True, "detail": msg}


@router.post("/api/models/{model_id}/stop")
async def model_stop(model_id: str) -> Dict[str, Any]:
    target = _require_model(model_id)
    if remote_base_url(target):
        return await _forward_admin_call(target, "POST", "/stop")
    ok, msg = bp.stop(model_id)
    if not ok:
        raise HTTPException(status_code=409, detail=msg)
    return {"ok": True, "detail": msg}


@router.post("/api/models/{model_id}/force-stop")
async def model_force_stop(model_id: str) -> Dict[str, Any]:
    """Force-kill whatever process holds this model's port.

    Use when the hub doesn't own the process — e.g. a stale backend
    from a previous tray session, or a llama-server someone started
    by hand. taskkill on Windows, SIGKILL on POSIX. The hub doesn't
    know what's listening — that's the whole point — so the caller
    is implicitly saying "I take responsibility for this PID".
    """
    target = _require_model(model_id)
    if remote_base_url(target):
        return await _forward_admin_call(target, "POST", "/force-stop")
    ok, msg = bp.force_stop_external(model_id)
    if not ok:
        raise HTTPException(status_code=409, detail=msg)
    return {"ok": True, "detail": msg}


@router.get("/api/models/{model_id}/log")
async def model_log(model_id: str, limit: int = 400) -> Dict[str, Any]:
    """Tail of a managed backend's log file (``data/logs/backend-<id>.log``).

    Readable for a backend the hub spawned *and* one it inherited across a
    restart — the child owns the log fd. Empty ``lines`` (200) when the
    backend has never started; 404 only for an unknown/subscription-backed
    model that has no managed process.
    """
    target = _require_model(model_id)
    if remote_base_url(target):
        return await _forward_admin_call(target, "GET", "/log", params={"limit": limit})
    if not (target.backend in ("openai", "whisper", "tts")):
        raise HTTPException(
            status_code=400,
            detail=f"backend {target.backend!r} has no managed process (subscription-backed)",
        )
    limit = max(1, min(limit, bp.LOG_TAIL_LINES * 10))
    lines = await asyncio.to_thread(bp.log_lines, model_id, limit)
    return {
        "id": model_id,
        "lines": lines,
        "path": f"data/logs/backend-{model_id}.log",
    }


def _silent_wav(seconds: float = 0.5, rate: int = 16000) -> bytes:
    """A tiny mono 16-bit PCM WAV of silence, built in memory.

    Just enough for whisper-server to decode and return a (blank)
    transcription — proves the backend can actually run inference, not
    merely that its port is open. 0.5s (not the original 0.1s): FluidAudio's
    Parakeet worker (#138) rejects anything under ~0.3s as invalidAudioData;
    whisper.cpp tolerates any length so the longer default doesn't regress it.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


def _ping_result(r: Any, latency_ms: float) -> Dict[str, Any]:
    """Shape a backend probe response into the tile's ping payload."""
    body: Dict[str, Any] = {}
    try:
        body = r.json()
    except Exception:  # noqa: BLE001
        body = {"raw": r.text[:300]}
    usage = body.get("usage") if isinstance(body, dict) else None
    return {
        "ok": r.is_success,
        "status": r.status_code,
        "latency_ms": round(latency_ms, 1),
        "usage": usage or {},
        "error": "" if r.is_success else (body.get("detail") if isinstance(body, dict) else str(r.status_code)),
    }


@router.post("/api/models/{model_id}/ping")
async def model_ping(model_id: str) -> Dict[str, Any]:
    """Probe the backend through the hub and report latency.

    Confirms the backend actually answers, not just that the port is open.
    The probe is protocol-aware: chat/ASR backends speak different APIs, so
    a chat ping at a whisper row would always 400. Whisper rows get a real
    audio transcription probe instead; everything else gets a 1-token chat
    probe. For subscription-backed claude/gemini rows the alias resolves
    inside the hub the same way as any other request.
    """
    target = bp.resolve_model_by_id(model_id)
    if target is None:
        # Could still be a claude/gemini row — those aren't backed by
        # backend_process but are still resolvable in the registry.
        target = resolve_model(model_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"unknown model {model_id!r}")

    import httpx
    from src.host_profile import hub_port

    port = hub_port()
    if target.backend == "whisper":
        # Whisper speaks the OpenAI audio API, not chat — send a tiny silent
        # clip to the hub's transcription proxy (model=display_name routes it
        # to this exact backend and keeps the hit in the observability ring).
        url = f"http://127.0.0.1:{port}/v1/audio/transcriptions"
        files = {"file": ("ping.wav", _silent_wav(), "audio/wav")}
        data = {"model": target.display_name}
        t0 = time.monotonic_ns()
        try:
            # Generous timeout: a lazy/CPU whisper backend may cold-load.
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(url, files=files, data=data)
        except httpx.HTTPError as exc:
            return {
                "ok": False,
                "status": 0,
                "latency_ms": (time.monotonic_ns() - t0) / 1e6,
                "error": str(exc),
            }
        return _ping_result(r, (time.monotonic_ns() - t0) / 1e6)

    if target.backend == "tts":
        # TTS speaks the OpenAI /v1/audio/speech shape, not chat — synthesize
        # a short phrase through the hub's proxy (model=display_name routes it
        # to this exact backend and keeps the hit in the observability ring).
        url = f"http://127.0.0.1:{port}/v1/audio/speech"
        payload = {"model": target.display_name, "input": "ping", "response_format": "wav"}
        t0 = time.monotonic_ns()
        try:
            # Generous timeout: a cold TTS backend may still be warming weights.
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            return {
                "ok": False,
                "status": 0,
                "latency_ms": (time.monotonic_ns() - t0) / 1e6,
                "error": str(exc),
            }
        # Audio bytes aren't JSON — _ping_result would mis-parse them, so
        # shape the result directly (ok = 2xx, no usage payload for audio).
        latency_ms = (time.monotonic_ns() - t0) / 1e6
        return {
            "ok": r.is_success,
            "status": r.status_code,
            "latency_ms": round(latency_ms, 1),
            "usage": {"audio_bytes": len(r.content)} if r.is_success else {},
            "error": "" if r.is_success else r.text[:300],
        }

    url = f"http://127.0.0.1:{port}/v1/messages"
    payload = {
        "model": target.display_name,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "ping"}],
    }
    t0 = time.monotonic_ns()
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(url, json=payload)
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "status": 0,
            "latency_ms": (time.monotonic_ns() - t0) / 1e6,
            "error": str(exc),
        }
    return _ping_result(r, (time.monotonic_ns() - t0) / 1e6)
