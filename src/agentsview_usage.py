"""AgentsView-sourced usage records for agents the hub has no native parser for.

AgentsView (``kenn-io/agentsview``) is an **optional, external** local service
— its own install (pipx / separate venv, never this repo's ``.venv``), its own
process (``agentsview serve``, default ``http://127.0.0.1:8080``), never
started or managed by the hub (issue #280).  It indexes 40+ coding-agent
session formats (including Gemini/``agy``, whose protobuf storage #279
declined to reverse-engineer) and exposes a REST API; this module polls that
API and emits records in the shared ``UsageRecord`` shape so they flow
through the same Code-tab summary builder as native data.

Scope guard: only the agents in the curated ``_AGENT_VENDOR_MAP`` are fetched
(the ``agy`` pair today) — natives (claude/codex/copilot) and unwanted slugs
(cursor/cowork/pi/…) are never touched, so AgentsView can neither override a
native source (#152 found its Claude totals run low) nor spam the vendor
selector.

Design constraints (mirroring the Langfuse pattern in
``app_web/routers/telemetry.py`` and the parser conventions in
``codex_usage.py``):
- **Read-only** — GETs only.
- **Never on the request path** — the summary endpoint is ``async def`` on the
  event loop, so all HTTP happens in a daemon refresh thread; callers get an
  in-memory snapshot instantly (stale-kick with a 60 s TTL).
- **Never raises** — unreachable/absent AgentsView degrades to the last-known
  snapshot with ``reachable=False``; a fresh process just has no records.
- **Session cache** — completed sessions are immutable, so per-session usage
  is fetched once and cached; only sessions started within the active window
  are re-fetched.

Config: ``AGENTSVIEW_BASE_URL`` env (default ``http://127.0.0.1:8080``); an
**empty string disables the integration entirely** (no probe — also how tests
stay hermetic).  Named to avoid AgentsView's own ``AGENTSVIEW_*`` env vars.
API note: aggregate endpoints are camelCase, ``/sessions/{id}/usage`` is
snake_case — mapped per endpoint below.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import httpx

from src.usage_common import (
    UsageRecord,
    encode_project_key,
    parse_iso_ts,
    project_pretty,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Curated AgentsView-agent -> hub-vendor map: only these agents are fetched
# and surfaced, everything else AgentsView knows about is ignored (the user
# explicitly scoped #280 down from surface-everything: Claude/Codex/Copilot
# stay native, and slugs like cursor/cowork/pi are noise on this machine).
# `agy` merges AgentsView's two Antigravity-related slugs: `gemini` (the
# hub's agy-routed one-shot calls) and `antigravity-cli` (interactive
# sessions) — one AGY button in the SPA.
_AGENT_VENDOR_MAP = {
    "gemini": "agy",
    "antigravity-cli": "agy",
}

# The curated vendors this module can ever emit — always valid selector
# values even while AgentsView is down/absent, because code_usage_history
# keeps their rollups queryable across hub restarts (#280).
KNOWN_VENDORS = sorted(set(_AGENT_VENDOR_MAP.values()))

_DEFAULT_BASE_URL = "http://127.0.0.1:8080"
_REFRESH_TTL_S = 60.0          # snapshot freshness window (2 SPA polls)
_CONNECT_TIMEOUT_S = 1.0       # loopback: refused connections fail instantly
_READ_TIMEOUT_S = 10.0
_SESSIONS_PAGE_LIMIT = 500     # API max per page
_MAX_SESSIONS_PER_AGENT = 2000  # safety valve on the per-session fan-out
_ACTIVE_WINDOW_DAYS = 2        # sessions this recent may still accrue → re-fetch


def _base_url() -> str:
    """AgentsView base URL from env; empty string = integration disabled."""
    raw = os.environ.get("AGENTSVIEW_BASE_URL")
    if raw is None:
        return _DEFAULT_BASE_URL
    return raw.strip().rstrip("/")


# ---------------------------------------------------------------------------
# Snapshot state (module-level singleton)
# ---------------------------------------------------------------------------


@dataclass
class _Snapshot:
    records: List[UsageRecord] = field(default_factory=list)
    vendors: List[str] = field(default_factory=list)  # sorted, non-native only
    reachable: bool = False
    error: str = ""
    version: str = ""
    fetched_at: float = 0.0  # epoch of the last refresh *attempt*


_snapshot = _Snapshot()
_lock = threading.Lock()
_refresh_in_flight = False
_was_reachable: Optional[bool] = None  # transition-only logging
# session_id -> records; completed sessions are immutable so this only grows
# (bounded by _MAX_SESSIONS_PER_AGENT per agent).
_session_cache: Dict[str, List[UsageRecord]] = {}


def _reset_for_tests() -> None:
    """Wipe all module state (test isolation)."""
    global _snapshot, _refresh_in_flight, _was_reachable
    with _lock:
        _snapshot = _Snapshot()
        _refresh_in_flight = False
        _was_reachable = None
        _session_cache.clear()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def all_records() -> List[UsageRecord]:
    """Return the current snapshot's records; kick a background refresh if
    stale.  Never blocks, never raises."""
    _kick_refresh_if_stale()
    with _lock:
        return list(_snapshot.records)


def discovered_vendors() -> List[str]:
    """Vendors AgentsView has data for that the hub lacks natively.

    Snapshot-only — **no refresh kick**, because this sits on the router's
    vendor-validation path.
    """
    with _lock:
        return list(_snapshot.vendors)


def status() -> dict:
    """Reachability block for the summary response (the Code-tab mirror of
    the Telemetry tab's ``langfuse_reachable``)."""
    with _lock:
        snap = _snapshot
        return {
            "enabled": bool(_base_url()),
            "reachable": snap.reachable,
            "base_url": _base_url(),
            "vendors": list(snap.vendors),
            "error": snap.error,
            "version": snap.version,
            "last_refresh": (
                datetime.fromtimestamp(snap.fetched_at, tz=timezone.utc).isoformat()
                if snap.fetched_at
                else None
            ),
        }


# ---------------------------------------------------------------------------
# Refresh machinery
# ---------------------------------------------------------------------------


def _kick_refresh_if_stale() -> None:
    global _refresh_in_flight
    if not _base_url():
        return
    with _lock:
        if _refresh_in_flight:
            return
        if time.time() - _snapshot.fetched_at < _REFRESH_TTL_S:
            return
        _refresh_in_flight = True
    threading.Thread(
        target=_refresh_worker, daemon=True, name="agentsview-refresh"
    ).start()


def _refresh_worker() -> None:
    global _refresh_in_flight
    try:
        _refresh()
    except Exception as exc:  # belt-and-braces: the worker must never die loud
        _log.warning("⚠️ agentsview: unexpected refresh error: %s", exc)
        _set_unreachable(str(exc))
    finally:
        with _lock:
            _refresh_in_flight = False


def _build_client() -> httpx.Client:
    """One short-lived client per refresh (≤1/min) — also the test seam."""
    return httpx.Client(
        base_url=_base_url(),
        timeout=httpx.Timeout(_READ_TIMEOUT_S, connect=_CONNECT_TIMEOUT_S),
    )


def _refresh() -> None:
    """Synchronous refresh core (tests call this directly, no thread)."""
    global _was_reachable
    try:
        with _build_client() as client:
            ping = client.get("/api/ping")
            ping.raise_for_status()
            info = ping.json()
            # Guard against a foreign service squatting the port (AgentsView
            # itself drifts to another port when 8080 is busy).
            if not info.get("ok") or "agentsview" not in str(
                info.get("service", "")
            ):
                _set_unreachable(
                    f"service on {_base_url()} is not agentsview: "
                    f"{info.get('service')!r}"
                )
                return
            version = str(info.get("version") or "")

            # include_one_shot: without it, agents whose sessions are all
            # one-shot (e.g. the hub's own agy-routed gemini calls) don't
            # appear in the list at all (verified live on v0.37.5).
            agents_resp = client.get(
                "/api/v1/agents", params={"include_one_shot": "true"}
            )
            if agents_resp.status_code >= 400:
                agents_resp = client.get("/api/v1/agents")
            agents_resp.raise_for_status()
            agents = _extract_agent_slugs(agents_resp.json())
            gap_agents = sorted(a for a in agents if a in _AGENT_VENDOR_MAP)
            vendors = sorted({_AGENT_VENDOR_MAP[a] for a in gap_agents})

            for agent in gap_agents:
                try:
                    _fetch_agent_sessions(client, agent)
                except Exception as exc:
                    # Per-agent degradation: skip this agent, keep the rest.
                    _log.warning(
                        "⚠️ agentsview: agent %r fetch failed, skipping: %s",
                        agent,
                        exc,
                    )
    except Exception as exc:
        _set_unreachable(str(exc))
        return

    records: List[UsageRecord] = []
    for recs in _session_cache.values():
        records.extend(recs)

    with _lock:
        _snapshot.records = records
        _snapshot.vendors = vendors
        _snapshot.reachable = True
        _snapshot.error = ""
        _snapshot.version = version
        _snapshot.fetched_at = time.time()
    if _was_reachable is not True:
        _log.info(
            "ℹ️ agentsview: reachable (v%s) — %d vendors, %d records",
            version,
            len(vendors),
            len(records),
        )
    _was_reachable = True


def _set_unreachable(err: str) -> None:
    """Mark unreachable but keep the last-known records/vendors visible."""
    global _was_reachable
    with _lock:
        _snapshot.reachable = False
        _snapshot.error = err
        _snapshot.fetched_at = time.time()
    if _was_reachable is not False:
        _log.warning("⚠️ agentsview: unreachable at %s: %s", _base_url(), err)
    _was_reachable = False


# ---------------------------------------------------------------------------
# Fetching & mapping
# ---------------------------------------------------------------------------


def _extract_agent_slugs(payload: object) -> List[str]:
    """Accept both a bare list and an ``{"agents": [...]}`` wrapper; entries
    may be strings or ``{"name"|"agent"|"id": ...}`` objects."""
    items = payload.get("agents", payload) if isinstance(payload, dict) else payload
    slugs: List[str] = []
    for item in items or []:
        if isinstance(item, str):
            slugs.append(item.lower())
        elif isinstance(item, dict):
            for key in ("name", "agent", "id", "slug"):
                if item.get(key):
                    slugs.append(str(item[key]).lower())
                    break
    return slugs


def _fetch_agent_sessions(client: httpx.Client, agent: str) -> None:
    """List one agent's sessions and cache per-session usage records."""
    active_since = datetime.now(tz=timezone.utc) - timedelta(
        days=_ACTIVE_WINDOW_DAYS
    )
    cursor: Optional[str] = None
    seen = 0
    include_one_shot = True  # hub-routed agy calls are often one-shot
    while seen < _MAX_SESSIONS_PER_AGENT:
        params: Dict[str, str] = {
            "agent": agent,
            "limit": str(_SESSIONS_PAGE_LIMIT),
        }
        if include_one_shot:
            params["include_one_shot"] = "true"
        if cursor:
            params["cursor"] = cursor
        resp = client.get("/api/v1/sessions", params=params)
        if resp.status_code >= 400 and include_one_shot:
            # Older/newer servers may reject the flag — retry without it.
            include_one_shot = False
            continue
        resp.raise_for_status()
        body = resp.json()
        sessions = body.get("sessions") or []
        if not sessions:
            break
        for sess in sessions:
            sid = str(
                sess.get("id") or sess.get("sessionId") or sess.get("session_id") or ""
            )
            if not sid:
                continue
            seen += 1
            ts = parse_iso_ts(
                sess.get("startedAt") or sess.get("started_at") or sess.get("date")
            )
            if sid in _session_cache and ts < active_since:
                continue  # completed session already cached — immutable
            usage = _fetch_session_usage(client, sid)
            _session_cache[sid] = _records_for_session(agent, sid, sess, usage)
            if seen >= _MAX_SESSIONS_PER_AGENT:
                break
        cursor = body.get("next_cursor") or body.get("nextCursor")
        if not cursor:
            break


def _fetch_session_usage(client: httpx.Client, session_id: str) -> dict:
    resp = client.get(
        f"/api/v1/sessions/{session_id}/usage", params={"breakdown": "true"}
    )
    if resp.status_code == 404:
        return {}
    resp.raise_for_status()
    return resp.json() or {}


def _project_fields(raw_project: Optional[str], agent: str) -> tuple:
    """Map AgentsView's ``project`` (path or plain name) onto the shared
    project key/name so real paths group with native vendors' records."""
    raw = (raw_project or "").strip()
    if not raw:
        return agent, "(unknown)"
    looks_like_path = (":\\" in raw) or ("/" in raw) or ("\\" in raw)
    if looks_like_path:
        key = encode_project_key(raw)
        return key, project_pretty(key)
    return raw, raw


def _int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _records_for_session(
    agent: str, session_id: str, sess: dict, usage: dict
) -> List[UsageRecord]:
    """Synthesize ``UsageRecord`` rows for one session.

    Preferred: one record per breakdown row (per-call granularity).  Fallback
    when breakdown rows are absent or carry no model: a single session-level
    record with whatever fields exist — partial data beats dropped data (#280).
    Timestamps are session-granular (``startedAt``), same precedent as the
    Copilot CLI parser.  Session-level ``cost_usd`` rides ``credits_usd`` on
    the first record only, so totals are exact and never double-counted.
    """
    vendor = _AGENT_VENDOR_MAP.get(agent, agent)
    ts = parse_iso_ts(
        sess.get("startedAt") or sess.get("started_at") or sess.get("date")
    )
    project_key, project_name = _project_fields(sess.get("project"), agent)
    session_cost = float(usage.get("cost_usd") or 0.0)
    models = usage.get("models") or []
    fallback_model = str(models[0]) if models else "unknown"

    rows = usage.get("breakdown") or []
    records: List[UsageRecord] = []
    row_costs_present = any(r.get("cost_usd") for r in rows if isinstance(r, dict))
    for row in rows:
        if not isinstance(row, dict):
            continue
        model = str(
            row.get("model") or row.get("model_name") or row.get("modelName") or ""
        )
        if not model:
            continue
        records.append(
            UsageRecord(
                session_id=session_id,
                project_key=project_key,
                project_name=project_name,
                model=model,
                # Rows carry a real per-call timestamp (verified v0.37.5);
                # fall back to the session start when absent.
                ts=parse_iso_ts(row["timestamp"]) if row.get("timestamp") else ts,
                input_tokens=_int(row.get("input_tokens")),
                output_tokens=_int(row.get("output_tokens")),
                cache_creation_tokens=_int(row.get("cache_creation_input_tokens")),
                cache_read_tokens=_int(row.get("cache_read_input_tokens")),
                vendor=vendor,
                credits_usd=float(row.get("cost_usd") or 0.0),
            )
        )
    if records:
        if not row_costs_present and session_cost:
            records[0].credits_usd = session_cost
        return records

    # Session-level fallback (no usable breakdown). Zero-token, cost-only
    # records are valid — no fabricated numbers.
    return [
        UsageRecord(
            session_id=session_id,
            project_key=project_key,
            project_name=project_name,
            model=fallback_model,
            ts=ts,
            input_tokens=_int(usage.get("total_input_tokens")),
            output_tokens=_int(usage.get("total_output_tokens")),
            cache_creation_tokens=0,
            cache_read_tokens=0,
            vendor=vendor,
            credits_usd=session_cost,
        )
    ]
