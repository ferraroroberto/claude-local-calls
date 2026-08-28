"""GitHub Copilot billing-API daily credits poller (issue #231, part B).

The Copilot CLI/VS Code session parsers in ``copilot_usage.py`` only see
usage the local machine actually wrote to disk — sessions that crashed
before a clean shutdown, or spend from other machines/IDEs, are invisible to
them. This module is the *authoritative* counterpart: GitHub's billing API
reports the real per-day x per-model AI Credit spend for the account,
independent of which device generated it. Trade-off: no session/project
attribution is possible from this endpoint, so it is surfaced as its own
"official credits" card rather than folded into the session-shaped tables
``code_usage.py`` builds for Claude/Codex/Copilot-local (issue #231 design
decision).

Design constraints mirror ``code_usage.py``/``codex_usage.py`` as closely as
a *remote* API allows:
- **Read-only** — GET only, never mutates anything on GitHub.
- **Passive** — nothing runs on any request path; the SPA polls this on its
  own cadence, gated to when the Copilot vendor tab is actually selected.
- **Cached, not hammered** — per-day granularity means one HTTP call per day
  in the window; a naive re-fetch-everything-every-poll would be 14+ calls
  every 30s. Past days are immutable and cached forever; only "today" (still
  accruing) is refreshed, and only if the cached copy is stale.

Auth: a fine-grained GitHub PAT with the "Plan: read-only" user permission,
read from ``GITHUB_COPILOT_BILLING_PAT`` (``.env``, loaded the same way as
Langfuse's keys in ``observability.py``). Unset PAT, a 404 (account not on
the enhanced billing platform), or any other HTTP error all degrade to
``{"available": False, "reason": ...}`` rather than raising — the Code tab
must render fine with zero Copilot billing configured.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

from src.http_client import get_async_client

_log = logging.getLogger(__name__)

_PAT_ENV = "GITHUB_COPILOT_BILLING_PAT"
_API_BASE = "https://api.github.com"
_TODAY_REFRESH_SECS = 300  # re-fetch "today" if the cached copy is older than this
_UNAVAILABLE_TTL_SECS = 300  # don't hammer a broken PAT/endpoint every poll

_username_cache: Optional[str] = None

# One entry per (year, month, day); {"items": [...], "fetched_at": epoch,
# "settled": bool}. "settled" is True only once a fetch for that day has
# happened after its own UTC date was already in the past — i.e. the data
# is guaranteed immutable. A day cached while it was still "today" gets one
# forced re-fetch after the date rolls over (see `needs_settle_refresh`
# below) before it is cached forever.
_day_cache: Dict[date, Dict[str, Any]] = {}

# Sticky "the whole feature is unavailable" state, so a missing/broken PAT
# doesn't retry the network on every 30 s poll.
_unavailable: Optional[Dict[str, Any]] = None


def _headers(pat: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def _resolve_username(client: httpx.AsyncClient, pat: str) -> str:
    resp = await client.get(f"{_API_BASE}/user", headers=_headers(pat), timeout=10.0)
    resp.raise_for_status()
    login = (resp.json() or {}).get("login")
    if not login:
        raise ValueError("GitHub /user response had no 'login'")
    return login


async def _fetch_day(
    client: httpx.AsyncClient, pat: str, username: str, d: date
) -> List[Dict[str, Any]]:
    """Return the raw ``usageItems`` for one calendar day."""
    resp = await client.get(
        f"{_API_BASE}/users/{username}/settings/billing/ai_credit/usage",
        headers=_headers(pat),
        params={"year": d.year, "month": d.month, "day": d.day},
        timeout=15.0,
    )
    resp.raise_for_status()
    body = resp.json() or {}
    items = body.get("usageItems")
    return items if isinstance(items, list) else []


def _set_unavailable(reason: str) -> Dict[str, Any]:
    global _unavailable
    _unavailable = {"reason": reason, "at": time.time()}
    return _unavailable


def _still_unavailable() -> Optional[str]:
    if _unavailable is None:
        return None
    if time.time() - _unavailable["at"] > _UNAVAILABLE_TTL_SECS:
        return None
    return _unavailable["reason"]


def _degraded(reason: str) -> Dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "daily": [],
        "as_of": datetime.now(tz=timezone.utc).isoformat(),
    }


async def get_daily_credits(days: int = 14) -> Dict[str, Any]:
    """Return per-day x per-model AI Credit spend for the last ``days`` days.

    Shape: ``{"available": bool, "reason": str|None, "daily": [{"date",
    "model", "credits", "usd"}], "as_of": iso}``. Degrades cleanly (never
    raises) when no PAT is configured or the account isn't on the enhanced
    billing platform.
    """
    pat = os.environ.get(_PAT_ENV, "").strip()
    if not pat:
        return _degraded("no PAT configured")

    sticky_reason = _still_unavailable()
    if sticky_reason is not None:
        return _degraded(sticky_reason)

    client = get_async_client()

    global _username_cache
    if _username_cache is None:
        try:
            _username_cache = await _resolve_username(client, pat)
        except (httpx.HTTPError, ValueError) as exc:
            reason = f"could not resolve GitHub username: {exc}"
            _log.warning("⚠️ copilot_billing: %s", reason)
            return _degraded(_set_unavailable(reason)["reason"])

    today = datetime.now(tz=timezone.utc).date()
    window = [today - timedelta(days=i) for i in range(days - 1, -1, -1)]

    rows: List[Dict[str, Any]] = []
    for i, d in enumerate(window):
        cached = _day_cache.get(d)
        is_today = d == today
        stale = cached is not None and is_today and (
            time.time() - cached["fetched_at"] > _TODAY_REFRESH_SECS
        )
        # A day cached while it was still "today" is not yet immutable —
        # `settled` is only set once a fetch happens after that day's UTC
        # date has passed. Force one final re-fetch the first time we see
        # such an entry with the date rolled over, so spend from the last
        # `_TODAY_REFRESH_SECS` before midnight isn't cached forever with
        # a gap (#529).
        needs_settle_refresh = (
            cached is not None and not is_today and not cached.get("settled", False)
        )
        if cached is None or stale or needs_settle_refresh:
            try:
                items = await _fetch_day(client, pat, _username_cache, d)
            except httpx.HTTPStatusError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    reason = "GitHub billing API returned 404 (account likely not on the enhanced billing platform)"
                    _log.info("ℹ️ copilot_billing: %s", reason)
                    return _degraded(_set_unavailable(reason)["reason"])
                # Non-404 error for one day: skip this day, keep the rest of
                # the window (don't fail the whole card over one bad day).
                _log.warning("⚠️ copilot_billing: %s failed: %s", d, exc)
                continue
            except httpx.HTTPError as exc:
                _log.warning("⚠️ copilot_billing: network error fetching %s: %s", d, exc)
                continue
            _day_cache[d] = {
                "items": items,
                "fetched_at": time.time(),
                "settled": not is_today,
            }
            cached = _day_cache[d]

        rows.extend(_aggregate_day(d, cached["items"]))

    return {
        "available": True,
        "reason": None,
        "daily": rows,
        "as_of": datetime.now(tz=timezone.utc).isoformat(),
    }


def _aggregate_day(d: date, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sum ``netAmount`` per model for one day's raw usageItems.

    Field names (``model``, ``netAmount``) are per the issue's own API
    investigation, not yet verified against a live response — the acceptance
    criterion ("daily chart matches the GitHub billing UI") should be
    checked with a real PAT before relying on the totals here; an unexpected
    shape degrades to zero credits for that item rather than raising.
    """
    by_model: Dict[str, float] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        model = item.get("model") or item.get("sku") or "unknown"
        net = item.get("netAmount")
        try:
            by_model[model] = by_model.get(model, 0.0) + float(net or 0.0)
        except (TypeError, ValueError):
            continue
    return [
        {"date": d.isoformat(), "model": model, "credits": credits, "usd": credits * 0.01}
        for model, credits in by_model.items()
    ]
