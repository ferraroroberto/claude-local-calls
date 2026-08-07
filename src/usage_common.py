"""Shared substrate for every vendor's usage parser (Claude/Codex/Copilot/AgentsView).

Hoisted out of ``code_usage.py`` (#451) — that module started as the
Claude-only parser its docstring still describes, but ``codex_usage.py``,
``copilot_usage.py`` and ``agentsview_usage.py`` had all grown to import its
underscore-private names as a de facto shared API (``_UsageRecord``,
``_FileStats``, ``_load_cached``, ``_encode_project_key``, ``_project_pretty``,
``_parse_iso_ts``), and ``claude_code_otel.py`` imported ``_model_display`` the
same way. A leading underscore claims "module-private", but the real contract
was already cross-module — a rename inside ``code_usage.py`` could silently
break four importers with nothing in the naming to warn about it. This module
is that contract made explicit: the record shape, the mtime-cache wrapper,
and the small parsing/display helpers every vendor parser needs, under public
names. ``code_usage.py`` (Claude JSONL parsing + the ``get_summary``
aggregation API) and the vendor parsers all import from here.

``_period_since`` (with the ``_today_utc`` it is built on) was missed by that
sweep and kept being reached for across the boundary by ``claude_code_otel.py``;
it moved here as :func:`period_since` / :func:`today_utc` in #471.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

_log = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    """One aggregated usage record (one assistant / agent API call).

    Shared across vendors (issue #71): Claude Code records carry
    ``vendor="claude"``; Codex records (from ``codex_usage.py``) carry
    ``vendor="codex"``.  The two trailing fields have defaults so the
    Claude parser, which never sets them, is unaffected.
    """

    session_id: str
    project_key: str       # encoded dir name, e.g. "E--automation-local-llm-hub"
    project_name: str      # pretty-printed project name
    model: str
    ts: datetime
    input_tokens: int      # net new prompt tokens (Codex: incl. cached subset)
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    # Codex-only: reasoning tokens, a *subset* of output_tokens (never added).
    reasoning_output_tokens: int = 0
    vendor: str = "claude"
    # Copilot: exact billed USD for this record (AI Credits, not a
    # rate-table estimate) — usage_pricing.record_costs() returns this
    # directly rather than pricing tokens against a $/Mtok table (issue #231).
    # AgentsView-sourced vendors reuse this field for AgentsView's reported
    # cost (#280).
    credits_usd: float = 0.0
    # Aggregation weight: 1 for a real parsed call; a code_usage_history
    # synthetic rollup row carries the N calls it summarises (#280).
    requests: int = 1


@dataclass
class FileStats:
    """Cached parse result for one JSONL file."""

    mtime: float
    entries: List["UsageRecord"]


def load_cached(
    path: Path,
    cache: Dict[str, FileStats],
    parse_fn: Callable[[Path], List["UsageRecord"]],
) -> List["UsageRecord"]:
    """Return cached records for ``path``, re-parsing via ``parse_fn`` only
    when the file's mtime has changed since the last call.

    Shared by the Claude/Codex/Copilot(CLI)/Copilot(VS Code) usage parsers —
    each used to carry its own copy of this "stat mtime -> compare to cached
    FileStats.mtime -> reparse if changed -> store -> return" wrapper,
    differing only in which cache dict and parse function to use.
    """
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []

    key = str(path)
    cached = cache.get(key)
    if cached is not None and cached.mtime == mtime:
        return cached.entries

    entries = parse_fn(path)
    cache[key] = FileStats(mtime=mtime, entries=entries)
    return entries


def parse_iso_ts(raw: Optional[str]) -> datetime:
    """Parse an ISO-8601 timestamp (bare, ``Z``-suffixed, or offset-aware);
    fall back to ``now()`` on any failure.

    Shared by the Claude/Codex/Copilot/AgentsView usage parsers — each used
    to hand-roll its own copy of this with a *different* exception set and a
    different ``Z`` normalization, so the same malformed timestamp was
    handled differently depending on which vendor wrote it. ``replace("Z",
    "+00:00")`` (rather than ``rstrip("Z")``) also correctly preserves a
    non-UTC offset already present in the string instead of overwriting it.
    """
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError):
        return datetime.now(tz=timezone.utc)
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def today_utc() -> date:
    """Today's date in UTC — the anchor every period window is measured from."""
    return datetime.now(tz=timezone.utc).date()


def period_since(period: str) -> Optional[date]:
    """Return the earliest date (UTC) that belongs to ``period``, or None for all-time.

    Shared by ``code_usage.get_summary`` and ``claude_code_otel``'s per-period
    rollup, which must agree on where a window starts or the OTel top-up would
    cover a different span than the records it tops up.
    """
    today = today_utc()
    if period == "today":
        return today
    if period == "week":
        return today - timedelta(days=6)
    if period == "month":
        return today - timedelta(days=29)
    # "all"
    return None


def encode_project_key(path: str) -> str:
    """Encode a raw filesystem path into the project-key form Claude Code uses.

    ``E:\\automation\\local-llm-hub`` → ``E--automation-local-llm-hub``.
    Shared so Codex records (whose source carries a raw ``cwd``) group under
    the same key as Claude records for the same project.
    """
    return path.replace(":\\", "--").replace("\\", "-").replace("/", "-")


_WORKSPACE_ROOT_SEGMENT = "automation"


def project_pretty(key: str) -> str:
    """Turn an encoded project key into a readable name.

    Drops the drive-letter prefix, then collapses the shared ``automation``
    workspace-root segment so the per-project table reads as the folder name
    and fits on mobile without horizontal scroll (issue #71)::

        E--automation-local-llm-hub → local-llm-hub
        E--automation              → automation   (the workspace root itself)
        C--Users-rober--some-path  → some-path    (not under automation: unchanged)
    """
    # Drop the drive-letter prefix (up to and including the first "--").
    parts = key.split("--", 1)
    tail = parts[-1] if len(parts) > 1 else key
    # Projects live under E:\automation\<name>; show just <name>. The bare
    # workspace root keeps its own name.
    prefix = _WORKSPACE_ROOT_SEGMENT + "-"
    if tail.startswith(prefix):
        return tail[len(prefix):]
    return tail


def model_display(model: str) -> str:
    """Shorten model IDs to a human-readable label.

    Claude families collapse to Fable / Opus / Sonnet / Haiku.  Codex
    (OpenAI) ids pass through with a readable label (``gpt-5.5`` →
    ``GPT-5.5``, ``gpt-5.5-pro`` → ``GPT-5.5 Pro``) rather than being
    forced into a Claude family.  Anything else is returned verbatim.
    """
    m = model.lower()
    if "fable" in m:
        return "Fable"
    if "opus" in m:
        return "Opus"
    if "sonnet" in m:
        return "Sonnet"
    if "haiku" in m:
        return "Haiku"
    if m.startswith("gpt"):
        return model.upper().replace("-PRO", " Pro")
    return model
