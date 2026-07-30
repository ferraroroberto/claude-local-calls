"""Host-side Claude Code usage parser.

Reads the JSONL session logs that Claude Code writes under
``~/.claude/projects/<encoded-path>/*.jsonl`` and aggregates them into
summaries suitable for the admin SPA's ``Cld`` tab.

Design constraints (from issue #20):
- **Read-only** — we never modify the JSONL files.
- **Zero subprocesses** — we parse the files ourselves rather than
  shelling out to ``bunx ccusage``.
- **Passive** — nothing runs on the Claude Code request path; the SPA
  polls this module on its own 30 s interval.
- **Mtime cache** — each file is only re-parsed when its mtime changes,
  so repeated polls are cheap.

Token fields used from each ``assistant`` entry::

    message.usage.input_tokens                   # net new prompt tokens
    message.usage.output_tokens                  # generated tokens
    message.usage.cache_creation_input_tokens    # tokens written to cache
    message.usage.cache_read_input_tokens        # tokens served from cache

``total_in`` for display purposes = ``input_tokens + cache_creation_input_tokens``
(both are "charged" in the Pro billing model).  ``cache_read`` is kept
separately so the SPA can show it with a visual distinction.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CLAUDE_PROJECTS_DIR: Path = Path.home() / ".claude" / "projects"

# Anthropic API list prices (USD per million tokens), keyed by model family.
# Loaded once from config/claude_pricing.json; this dict is the fallback used
# when that file is missing or unreadable, so cost display degrades gracefully.
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
_PRICING_PATH: Path = _PROJECT_ROOT / "config" / "claude_pricing.json"
_PRICING_FALLBACK: Dict[str, Dict[str, float]] = {
    "Fable":  {"input": 10.0, "output": 50.0, "cache_write": 12.50, "cache_read": 1.00},
    "Opus":   {"input": 5.0, "output": 25.0, "cache_write": 6.25, "cache_read": 0.50},
    "Sonnet": {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30},
    "Haiku":  {"input": 1.0, "output": 5.0,  "cache_write": 1.25, "cache_read": 0.10},
}

# OpenAI API list prices (USD per million tokens), keyed by display model id
# (what _model_display returns for a Codex model, e.g. "GPT-5.5").  Used to
# show the equivalent metered-API cost of host-side Codex usage.  Loaded once
# from config/openai_pricing.json; this dict is the fallback when that file is
# missing or unreadable.  Codex's cached_input tokens are a *subset* of input
# (not additive), so the cost path prices the non-cached remainder at "input"
# and the cached portion at "cached_input".
_OPENAI_PRICING_PATH: Path = _PROJECT_ROOT / "config" / "openai_pricing.json"
_OPENAI_PRICING_FALLBACK: Dict[str, Dict[str, float]] = {
    "GPT-5.6-SOL": {"input": 5.0,  "cached_input": 0.50, "output": 30.0},
    "GPT-5.5":     {"input": 5.0,  "cached_input": 0.50, "output": 30.0},
    "GPT-5.5 Pro": {"input": 30.0, "cached_input": 0.0,  "output": 180.0},
    "GPT-5.4":     {"input": 2.5,  "cached_input": 0.25, "output": 15.0},
}

# Gemini API list prices in USD per million tokens, keyed by family (what
# _gemini_family collapses an AGY model name to).  Used to show the
# equivalent metered-API cost of AGY/Antigravity usage (#280) — an estimate
# against Google list prices, same idea as Codex vs OpenAI.  AGY cache reads
# are reported separately/additively (Claude-style), priced at "cache_read".
_GEMINI_PRICING_PATH: Path = _PROJECT_ROOT / "config" / "gemini_pricing.json"
_GEMINI_PRICING_FALLBACK: Dict[str, Dict[str, float]] = {
    "pro":        {"input": 2.0,  "output": 12.0, "cache_read": 0.20},
    "flash":      {"input": 0.30, "output": 2.50, "cache_read": 0.03},
    "flash-lite": {"input": 0.10, "output": 0.40, "cache_read": 0.01},
}

# Memoized price tables, keyed by cache slot ("claude" / "openai" / "gemini");
# populated lazily by ``_load_priced_table``.
_priced_table_cache: Dict[str, Dict[str, Dict[str, float]]] = {}

# How many recent sessions to return in the summary.
_MAX_RECENT_SESSIONS = 15

# How many days of daily history to return.
_MAX_DAILY_DAYS = 14

# How many weeks / months of history to return for the trend charts.
_MAX_CHART_WEEKS = 12
_MAX_CHART_MONTHS = 12


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass
class _FileStats:
    """Cached parse result for one JSONL file."""

    mtime: float
    entries: List["_UsageRecord"]


@dataclass
class _UsageRecord:
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
    # rate-table estimate) — _record_costs() returns this directly rather
    # than pricing tokens against a $/Mtok table (issue #231).  AgentsView-
    # sourced vendors reuse this field for AgentsView's reported cost (#280).
    credits_usd: float = 0.0
    # Aggregation weight: 1 for a real parsed call; a code_usage_history
    # synthetic rollup row carries the N calls it summarises (#280).
    requests: int = 1


# ---------------------------------------------------------------------------
# File-level mtime cache (module-level singleton)
# ---------------------------------------------------------------------------

_file_cache: Dict[str, _FileStats] = {}


def _load_cached(
    path: Path,
    cache: Dict[str, _FileStats],
    parse_fn: Callable[[Path], List["_UsageRecord"]],
) -> List["_UsageRecord"]:
    """Return cached records for ``path``, re-parsing via ``parse_fn`` only
    when the file's mtime has changed since the last call.

    Shared by the Claude/Codex/Copilot(CLI)/Copilot(VS Code) usage parsers —
    each used to carry its own copy of this "stat mtime -> compare to cached
    _FileStats.mtime -> reparse if changed -> store -> return" wrapper,
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
    cache[key] = _FileStats(mtime=mtime, entries=entries)
    return entries


def _parse_iso_ts(raw: Optional[str]) -> datetime:
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


def _encode_project_key(path: str) -> str:
    """Encode a raw filesystem path into the project-key form Claude Code uses.

    ``E:\\automation\\local-llm-hub`` → ``E--automation-local-llm-hub``.
    Shared so Codex records (whose source carries a raw ``cwd``) group under
    the same key as Claude records for the same project.
    """
    return path.replace(":\\", "--").replace("\\", "-").replace("/", "-")


_WORKSPACE_ROOT_SEGMENT = "automation"


def _project_pretty(key: str) -> str:
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


def _parse_jsonl_file(path: Path, project_key: str) -> List[_UsageRecord]:
    """Parse one JSONL file and return usage records."""
    records: List[_UsageRecord] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if obj.get("type") != "assistant":
                    continue

                msg = obj.get("message") or {}
                usage = msg.get("usage") or {}
                if not usage:
                    continue

                # Timestamp — fall back gracefully.
                ts = _parse_iso_ts(obj.get("timestamp", ""))

                model = msg.get("model") or "unknown"
                session_id = obj.get("sessionId") or str(path.stem)

                records.append(
                    _UsageRecord(
                        session_id=session_id,
                        project_key=project_key,
                        project_name=_project_pretty(project_key),
                        model=model,
                        ts=ts,
                        input_tokens=int(usage.get("input_tokens") or 0),
                        output_tokens=int(usage.get("output_tokens") or 0),
                        cache_creation_tokens=int(
                            usage.get("cache_creation_input_tokens") or 0
                        ),
                        cache_read_tokens=int(
                            usage.get("cache_read_input_tokens") or 0
                        ),
                    )
                )
    except OSError as exc:
        _log.warning("⚠️ code_usage: cannot read %s: %s", path, exc)
    return records


def _load_file(path: Path, project_key: str) -> List[_UsageRecord]:
    """Return cached records, re-parsing only when the file has changed."""
    return _load_cached(path, _file_cache, lambda p: _parse_jsonl_file(p, project_key))


def _claude_records() -> List[_UsageRecord]:
    """Scan all Claude Code project JSONL files and return every usage record."""
    records: List[_UsageRecord] = []

    if not _CLAUDE_PROJECTS_DIR.exists():
        return records

    try:
        project_dirs = [p for p in _CLAUDE_PROJECTS_DIR.iterdir() if p.is_dir()]
    except OSError as exc:
        _log.warning("⚠️ code_usage: cannot list %s: %s", _CLAUDE_PROJECTS_DIR, exc)
        return records

    for proj_dir in project_dirs:
        project_key = proj_dir.name
        try:
            jsonl_files = list(proj_dir.glob("*.jsonl"))
            # Newer Claude Code also writes per-session directories holding
            # sub-agent transcripts (projects/<proj>/<session>/subagents/
            # agent-*.jsonl, same line format) — the Task-tool usage that
            # the flat files never carried (#68's blind spot, now partially
            # local). Lines carry the parent sessionId so they group with
            # their session where present.
            jsonl_files += list(proj_dir.glob("*/subagents/agent-*.jsonl"))
        except OSError:
            continue
        for jf in jsonl_files:
            records.extend(_load_file(jf, project_key))

    return records


_VALID_VENDORS = {"claude", "codex", "copilot", "all"}  # native set + "all"


def is_valid_vendor(vendor: str) -> bool:
    """Native vendors/"all" plus the curated AgentsView vendors (#280).

    ``KNOWN_VENDORS`` (e.g. ``agy``) stay valid even while AgentsView is
    down or absent — their history rollups remain queryable.  Snapshot-only
    lookups, no network, so this is safe on the router's validation path.
    """
    if vendor in _VALID_VENDORS:
        return True
    from src import agentsview_usage
    return (
        vendor in agentsview_usage.KNOWN_VENDORS
        or vendor in agentsview_usage.discovered_vendors()
    )


def _gather_records(vendor: str = "all") -> List[_UsageRecord]:
    """Return usage records for the requested vendor(s).

    ``vendor`` is ``claude | codex | copilot | all`` or an AgentsView-sourced
    agent slug (issue #280).  Vendor modules are imported lazily so they can
    import shared helpers from this module without a cycle.
    """
    records: List[_UsageRecord] = []
    if vendor in ("all", "claude"):
        records.extend(_claude_records())
    if vendor in ("all", "codex"):
        from src import codex_usage
        records.extend(codex_usage.all_records())
    if vendor in ("all", "copilot"):
        from src import copilot_usage
        records.extend(copilot_usage.all_records())
    if vendor == "all" or vendor not in _VALID_VENDORS:
        # AgentsView gap-fill vendors (never claude/codex/copilot — the
        # client excludes natives at the source, so "all" can't double-count).
        from src import agentsview_usage
        av = agentsview_usage.all_records()
        if vendor != "all":
            av = [r for r in av if r.vendor == vendor]
        records.extend(av)
    return records


# ---------------------------------------------------------------------------
# Public API — aggregation helpers
# ---------------------------------------------------------------------------


def _today_utc() -> date:
    return datetime.now(tz=timezone.utc).date()


def _tok_k(n: int) -> float:
    """Round to one decimal in thousands."""
    return round(n / 1000, 1)


def _model_display(model: str) -> str:
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


def _load_priced_table(
    path: Path,
    top_key: str,
    fallback: Dict[str, Dict[str, float]],
    cache_slot: str,
) -> Dict[str, Dict[str, float]]:
    """Return a per-family/per-model price table, loaded once from config and cached.

    Shared loader for the Claude/OpenAI/Gemini pricing tables (each used to
    carry its own near-identical ~35-line copy differing only in the config
    path, the JSON top-level key, and which cache slot to memoize into).
    Falls back to ``fallback`` when ``path`` is missing or malformed, so the
    cost display never hard-fails on a fresh checkout.
    """
    cached = _priced_table_cache.get(cache_slot)
    if cached is not None:
        return cached

    pricing: Dict[str, Dict[str, float]] = dict(fallback)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        table = raw.get(top_key) or {}
        if isinstance(table, dict) and table:
            pricing = {
                name: {k: float(v) for k, v in rates.items()}
                for name, rates in table.items()
                if isinstance(rates, dict)
            }
    except (OSError, ValueError, TypeError) as exc:
        _log.warning(
            "⚠️ code_usage: using fallback pricing (%s unreadable): %s",
            path, exc,
        )
    _priced_table_cache[cache_slot] = pricing
    return pricing


def _load_pricing() -> Dict[str, Dict[str, float]]:
    """Return the per-family Claude price table, loaded once from config and cached."""
    return _load_priced_table(_PRICING_PATH, "families", _PRICING_FALLBACK, "claude")


def _load_openai_pricing() -> Dict[str, Dict[str, float]]:
    """Return the per-model OpenAI price table, loaded once and cached."""
    return _load_priced_table(_OPENAI_PRICING_PATH, "models", _OPENAI_PRICING_FALLBACK, "openai")


def _gemini_family(model: str) -> str:
    """Collapse an AGY model name to a Gemini pricing family.

    AgentsView surfaces both raw ids (``gemini-3.1-pro-preview``) and
    display names (``Gemini 3.1 Pro (High)``) — family matching by
    substring covers both.  Unknown names return "" (prices at zero, no
    fabricated cost).
    """
    m = model.lower().replace(" ", "-")
    if "gemini" not in m:
        return ""
    if "flash-lite" in m:
        return "flash-lite"
    if "flash" in m:
        return "flash"
    if "pro" in m:
        return "pro"
    return ""


def _load_gemini_pricing() -> Dict[str, Dict[str, float]]:
    """Return the per-family Gemini price table, loaded once and cached."""
    return _load_priced_table(_GEMINI_PRICING_PATH, "families", _GEMINI_PRICING_FALLBACK, "gemini")


def _record_costs(r: "_UsageRecord") -> Tuple[float, float, float]:
    """Return ``(input_cost, output_cost, cache_read_cost)`` in USD for one record.

    Priced against the record's own model, so a mixed-model / mixed-vendor
    period is summed correctly.  Unknown models price at zero (no fabricated
    cost).  The cost maps into the same three tiles the SPA shows.

    Claude: input-tile cost folds in cache-creation tokens (5-min cache-write
    rate) to mirror the tile (``input + cache_creation``).

    Codex: ``cached_input`` tokens are a *subset* of ``input_tokens``, so the
    non-cached remainder is priced at the input rate and the cached portion at
    the (cheaper) cached_input rate — no double counting.  Reasoning tokens are
    already inside ``output_tokens`` and bill at the output rate.  The >272K
    long-context surcharge (2x input / 1.5x output) is not modelled — this is an
    estimate, and per-request context size isn't tracked.

    AGY (#280): priced per tile against Gemini API list prices
    (``config/gemini_pricing.json``), same estimate-vs-list-prices idea as
    Codex — AgentsView's own ``cost_usd`` is unreliable here (it can't price
    the display-name model ids ``antigravity-cli`` sessions carry).  Cache
    reads are reported separately/additively (Claude-style), priced at the
    implicit-caching rate.

    Copilot and any other AgentsView-sourced vendor: ``credits_usd`` carries
    the record's own dollar figure — Copilot's *exact* billed AI Credits
    (#231) — returned directly as the input-cost slot so it still nets into
    the same three-tile total the SPA sums, never re-priced against a hub
    rate table.  This must be checked before the Claude family-substring
    fallback below, or a Copilot call that resolved to e.g.
    ``claude-sonnet-4.5`` would get silently re-priced at the Anthropic
    subscription rate instead of its own credit charge.
    """
    if r.vendor == "agy":
        rates = _load_gemini_pricing().get(_gemini_family(r.model))
        if not rates:
            return 0.0, 0.0, 0.0
        input_cost = r.input_tokens * rates.get("input", 0.0) / 1_000_000
        output_cost = r.output_tokens * rates.get("output", 0.0) / 1_000_000
        cache_read_cost = (
            r.cache_read_tokens * rates.get("cache_read", 0.0) / 1_000_000
        )
        return input_cost, output_cost, cache_read_cost

    if r.vendor not in ("claude", "codex"):
        return r.credits_usd, 0.0, 0.0

    if r.vendor == "codex":
        rates = _load_openai_pricing().get(_model_display(r.model))
        if not rates:
            return 0.0, 0.0, 0.0
        non_cached_input = max(r.input_tokens - r.cache_read_tokens, 0)
        input_cost = non_cached_input * rates.get("input", 0.0) / 1_000_000
        output_cost = r.output_tokens * rates.get("output", 0.0) / 1_000_000
        cache_read_cost = (
            r.cache_read_tokens * rates.get("cached_input", 0.0) / 1_000_000
        )
        return input_cost, output_cost, cache_read_cost

    rates = _load_pricing().get(_model_display(r.model))
    if not rates:
        return 0.0, 0.0, 0.0
    input_cost = (
        r.input_tokens * rates.get("input", 0.0)
        + r.cache_creation_tokens * rates.get("cache_write", 0.0)
    ) / 1_000_000
    output_cost = r.output_tokens * rates.get("output", 0.0) / 1_000_000
    cache_read_cost = r.cache_read_tokens * rates.get("cache_read", 0.0) / 1_000_000
    return input_cost, output_cost, cache_read_cost


def _week_start(d: date) -> date:
    """Return the Monday of the ISO week containing d."""
    from datetime import timedelta
    return d - timedelta(days=d.weekday())


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _build_time_series(
    records: List[_UsageRecord], period: str, today: date
) -> list:
    """Build oldest-first time-series buckets with per-model breakdown for the chart.

    Returned list is empty for ``period == "all"`` (unbounded x-axis is not useful).
    Each bucket: ``{"label": str, "models": {family: {input_tokens, output_tokens, requests}}}``.
    ``input_tokens`` already folds in ``cache_creation_tokens`` so the chart shows billed in.
    """
    from datetime import timedelta

    if period == "all":
        return []

    if period == "today":
        buckets = [today - timedelta(days=i) for i in range(_MAX_DAILY_DAYS - 1, -1, -1)]
    elif period == "week":
        this_mon = _week_start(today)
        buckets = [this_mon - timedelta(weeks=i) for i in range(_MAX_CHART_WEEKS - 1, -1, -1)]
    else:  # month
        ym: List[tuple] = []
        y, m = today.year, today.month
        for _ in range(_MAX_CHART_MONTHS):
            ym.append((y, m))
            m -= 1
            if m == 0:
                m, y = 12, y - 1
        buckets = [date(yr, mo, 1) for yr, mo in reversed(ym)]

    bucket_set = set(buckets)
    bmap: Dict[date, Dict[str, dict]] = {b: {} for b in buckets}

    for r in records:
        rd = r.ts.astimezone(timezone.utc).date()
        if period == "today":
            bk = rd
        elif period == "week":
            bk = _week_start(rd)
        else:
            bk = _month_start(rd)
        if bk not in bucket_set:
            continue
        family = _model_display(r.model)
        if family not in bmap[bk]:
            bmap[bk][family] = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "requests": 0}
        slot = bmap[bk][family]
        slot["input_tokens"] += r.input_tokens + r.cache_creation_tokens
        slot["output_tokens"] += r.output_tokens
        slot["cache_read_tokens"] += r.cache_read_tokens
        slot["requests"] += 1

    result = []
    for b in buckets:
        lbl = b.strftime("%b %Y") if period == "month" else b.strftime("%b ") + str(b.day)
        result.append({"label": lbl, "models": bmap[b]})
    return result


def _build_prev_totals(
    records: List[_UsageRecord], period: str, today: date
) -> Optional[dict]:
    """Return aggregate counts for the period immediately preceding the current window.

    today  → yesterday
    week   → 7 days ending last Sunday (today−13 .. today−7)
    month  → 30-day window ending 30 days ago (today−59 .. today−30)
    all    → None (omitted from response)

    For a non-"all" period the dict is always returned (zero-filled when the
    preceding window had no activity), so the SPA can show a "new" badge for a
    metric whose prior value was 0 instead of hiding the comparison entirely —
    e.g. a vendor like Codex that has no data in the previous week (issue #71).
    """
    from datetime import timedelta

    if period == "all":
        return None

    if period == "today":
        lo = hi = today - timedelta(days=1)
    elif period == "week":
        lo, hi = today - timedelta(days=13), today - timedelta(days=7)
    else:  # month
        lo, hi = today - timedelta(days=59), today - timedelta(days=30)

    acc = {
        "input_tokens": 0, "output_tokens": 0,
        "cache_creation_tokens": 0, "cache_read_tokens": 0, "requests": 0,
    }
    for r in records:
        d = r.ts.astimezone(timezone.utc).date()
        if lo <= d <= hi:
            acc["input_tokens"] += r.input_tokens
            acc["output_tokens"] += r.output_tokens
            acc["cache_creation_tokens"] += r.cache_creation_tokens
            acc["cache_read_tokens"] += r.cache_read_tokens
            acc["requests"] += 1
    return acc


_VALID_PERIODS = {"today", "week", "month", "all"}


def _period_since(period: str) -> Optional[date]:
    """Return the earliest date (UTC) that belongs to ``period``, or None for all-time."""
    from datetime import timedelta
    today = _today_utc()
    if period == "today":
        return today
    if period == "week":
        return today - timedelta(days=6)
    if period == "month":
        return today - timedelta(days=29)
    # "all"
    return None


_OTEL_UNTRACKED_KEY = "otel-untracked"
_OTEL_FIELD_MAP = {
    "input": "input_tokens",
    "output": "output_tokens",
    "cache_read": "cache_read_tokens",
    "cache_creation": "cache_creation_tokens",
}


def _otel_delta_records(records: List[_UsageRecord]) -> List[_UsageRecord]:
    """Claude usage the transcript sources never saw (#280 follow-up).

    Sessions bridged through claude.ai/code export OTel metrics (#68) but
    write no usage records to the local JSONL transcripts, so the OTel store
    is a superset of the transcript view whenever telemetry is enabled.  Per
    ``(day, model family)``, any per-field OTel excess over the claude
    records already gathered (live + history) becomes one synthetic record:
    ``vendor="claude"``, project ``(untracked)`` (OTel carries no project or
    session attribution), ``requests=0`` (no request count in the token
    metrics — token/cost tiles stay accurate, the requests tile only counts
    transcript-sourced calls).

    Recomputed from ``claude_code_otel``'s own persisted store on every call
    and **never folded into code_usage_history** — persisting a derived
    delta alongside its inputs would double-count when the inputs shift
    (e.g. partial pruning of a day).  Never raises.
    """
    try:
        from src import claude_code_otel
        otel_rows = claude_code_otel.get_usage_summary("all")["rows"]
    except Exception as exc:  # noqa: BLE001 — OTel store must never break the tab
        _log.warning("⚠️ code_usage: OTel delta unavailable: %s", exc)
        return []
    if not otel_rows:
        return []

    otel: Dict[Tuple[str, str], Dict[str, int]] = {}
    for row in otel_rows:
        key = (row["date"], _model_display(str(row.get("model") or "unknown")))
        acc = otel.setdefault(key, {f: 0 for f in _OTEL_FIELD_MAP.values()})
        for src_field, dst_field in _OTEL_FIELD_MAP.items():
            acc[dst_field] += int(row.get(src_field) or 0)

    covered: Dict[Tuple[str, str], Dict[str, int]] = {}
    for r in records:
        if r.vendor != "claude":
            continue
        key = (r.ts.astimezone(timezone.utc).date().isoformat(), _model_display(r.model))
        acc = covered.setdefault(key, {f: 0 for f in _OTEL_FIELD_MAP.values()})
        acc["input_tokens"] += r.input_tokens
        acc["output_tokens"] += r.output_tokens
        acc["cache_read_tokens"] += r.cache_read_tokens
        acc["cache_creation_tokens"] += r.cache_creation_tokens

    out: List[_UsageRecord] = []
    for (day, family), sums in otel.items():
        have = covered.get((day, family), {})
        delta = {
            f: max(sums[f] - have.get(f, 0), 0)
            for f in _OTEL_FIELD_MAP.values()
        }
        if not any(delta.values()):
            continue
        d = date.fromisoformat(day)
        out.append(
            _UsageRecord(
                session_id=f"otel:{day}",
                project_key=_OTEL_UNTRACKED_KEY,
                project_name="(untracked)",
                model=family,
                ts=datetime(d.year, d.month, d.day, 12, tzinfo=timezone.utc),
                input_tokens=delta["input_tokens"],
                output_tokens=delta["output_tokens"],
                cache_creation_tokens=delta["cache_creation_tokens"],
                cache_read_tokens=delta["cache_read_tokens"],
                vendor="claude",
                requests=0,
            )
        )
    return out


def get_summary(period: str = "today", vendor: str = "all") -> dict:
    """Return a summary dict consumed by the Cld tab.

    ``period`` is one of ``today | week | month | all``.
    ``vendor`` is ``claude | codex | copilot | all`` (issues #71, #231) or a
    dynamically-discovered AgentsView agent slug, e.g. ``gemini`` (#280).

    Keys returned:
      period     — echoed back
      vendor     — echoed back
      totals     — aggregate token counts for the requested period
      daily      — per-day list (last _MAX_DAILY_DAYS days, newest first; all-time)
      by_model   — per-model-family breakdown for the requested period
      by_project — per-project breakdown for the requested period
      by_vendor  — per-vendor breakdown for the requested period
      recent_sessions — last _MAX_RECENT_SESSIONS sessions (all-time)
    """
    if period not in _VALID_PERIODS:
        period = "today"
    if not is_valid_vendor(vendor):
        vendor = "all"

    records = _gather_records(vendor)
    # Durable daily rollups (#280): fold what the parsers see right now into
    # the on-disk snapshot, then extend with synthetic records for days the
    # sources have already pruned (Claude Code deletes transcripts after
    # ~30 days) — "All" keeps growing instead of silently equalling "Month".
    from src import code_usage_history as _history
    _history.update_from_records(records)
    records = records + _history.synthetic_records(records, vendor)
    # Claude sessions bridged through claude.ai/code write no usage to the
    # local transcripts at all — the hub's OTel receiver (#68) is the only
    # channel that sees them. Top the Claude picture up with per-(day, model)
    # deltas vs everything above; recomputed each call from OTel's own
    # durable store, deliberately NOT persisted into the history snapshot.
    if vendor in ("all", "claude"):
        records = records + _otel_delta_records(records)
    today = _today_utc()
    since = _period_since(period)

    # ---- helpers ----
    def blank_counts() -> dict:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "reasoning_output_tokens": 0,
            "requests": 0,
        }

    def add_record(acc: dict, r: _UsageRecord) -> None:
        acc["input_tokens"] += r.input_tokens
        acc["output_tokens"] += r.output_tokens
        acc["cache_creation_tokens"] += r.cache_creation_tokens
        acc["cache_read_tokens"] += r.cache_read_tokens
        acc["reasoning_output_tokens"] += r.reasoning_output_tokens
        acc["requests"] += r.requests

    def in_period(r: _UsageRecord) -> bool:
        return since is None or r.ts.astimezone(timezone.utc).date() >= since

    # ---- totals for the requested period (with equivalent API cost) ----
    totals = blank_counts()
    cost_acc = {"input_cost": 0.0, "output_cost": 0.0, "cache_read_cost": 0.0}
    for r in records:
        if in_period(r):
            add_record(totals, r)
            ic, oc, crc = _record_costs(r)
            cost_acc["input_cost"] += ic
            cost_acc["output_cost"] += oc
            cost_acc["cache_read_cost"] += crc
    totals.update(cost_acc)

    # ---- daily buckets (always last _MAX_DAILY_DAYS calendar days) ----
    daily_map: Dict[date, dict] = {}
    for r in records:
        d = r.ts.astimezone(timezone.utc).date()
        if d not in daily_map:
            daily_map[d] = {"date": d.isoformat(), **blank_counts()}
        add_record(daily_map[d], r)

    sorted_days = sorted(daily_map.keys(), reverse=True)
    daily_list = [daily_map[d] for d in sorted_days[:_MAX_DAILY_DAYS]]

    # ---- per-model breakdown (period-scoped) ----
    model_map: Dict[str, dict] = {}
    for r in records:
        if not in_period(r):
            continue
        label = _model_display(r.model)
        if label not in model_map:
            model_map[label] = {"model": label, **blank_counts()}
        add_record(model_map[label], r)
    by_model = sorted(
        model_map.values(), key=lambda x: x["requests"], reverse=True
    )

    # ---- per-project breakdown (period-scoped) ----
    proj_map: Dict[str, dict] = {}
    for r in records:
        if not in_period(r):
            continue
        key = r.project_key
        if key not in proj_map:
            proj_map[key] = {
                "project_key": key,
                "project": r.project_name,
                **blank_counts(),
            }
        add_record(proj_map[key], r)
    by_project = sorted(
        proj_map.values(), key=lambda x: x["requests"], reverse=True
    )

    # ---- per-vendor breakdown (period-scoped, with equivalent API cost) ----
    vendor_map: Dict[str, dict] = {}
    for r in records:
        if not in_period(r):
            continue
        row = vendor_map.get(r.vendor)
        if row is None:
            row = vendor_map[r.vendor] = {
                "vendor": r.vendor,
                **blank_counts(),
                "input_cost": 0.0,
                "output_cost": 0.0,
                "cache_read_cost": 0.0,
            }
        add_record(row, r)
        ic, oc, crc = _record_costs(r)
        row["input_cost"] += ic
        row["output_cost"] += oc
        row["cache_read_cost"] += crc
    by_vendor = sorted(
        vendor_map.values(), key=lambda x: x["requests"], reverse=True
    )

    # ---- recent sessions (always all-time, newest first) ----
    session_map: Dict[Tuple[str, str], dict] = {}
    for r in records:
        k = (r.project_key, r.session_id)
        if k not in session_map:
            session_map[k] = {
                "session_id": r.session_id,
                "project_key": r.project_key,
                "project": r.project_name,
                "model": r.model,
                "first_ts": r.ts.isoformat(),
                "last_ts": r.ts.isoformat(),
                **blank_counts(),
            }
        s = session_map[k]
        add_record(s, r)
        if r.ts.isoformat() < s["first_ts"]:
            s["first_ts"] = r.ts.isoformat()
        if r.ts.isoformat() > s["last_ts"]:
            s["last_ts"] = r.ts.isoformat()
        s["model"] = r.model

    sessions_sorted = sorted(
        session_map.values(), key=lambda x: x["last_ts"], reverse=True
    )
    recent_sessions = sessions_sorted[:_MAX_RECENT_SESSIONS]

    time_series = _build_time_series(records, period, today)
    prev_totals = _build_prev_totals(records, period, today)

    result: dict = {
        "period": period,
        "vendor": vendor,
        "totals": totals,
        "daily": daily_list,
        "by_model": by_model,
        "by_project": by_project,
        "by_vendor": by_vendor,
        "recent_sessions": recent_sessions,
        "time_series": time_series,
    }
    if prev_totals is not None:
        result["prev_totals"] = prev_totals
    return result


def get_today_totals_for_project(project_dir: str) -> Optional[dict]:
    """Quick helper for the status-line script (not used by the SPA router).

    ``project_dir`` is the raw filesystem path, e.g. ``E:\\automation\\local-llm-hub``.
    Returns a dict with ``input_tokens``, ``output_tokens``, ``cache_read_tokens``,
    ``cache_creation_tokens``, ``requests`` for today, or ``None`` if no data.
    """
    # Encode the path the same way Claude Code does.
    encoded = _encode_project_key(project_dir)
    proj_dir = _CLAUDE_PROJECTS_DIR / encoded
    if not proj_dir.is_dir():
        return None

    today = _today_utc()
    acc = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "requests": 0,
    }
    found = False
    for jf in proj_dir.glob("*.jsonl"):
        for r in _load_file(jf, encoded):
            if r.ts.astimezone(timezone.utc).date() == today:
                acc["input_tokens"] += r.input_tokens
                acc["output_tokens"] += r.output_tokens
                acc["cache_creation_tokens"] += r.cache_creation_tokens
                acc["cache_read_tokens"] += r.cache_read_tokens
                acc["requests"] += 1
                found = True

    return acc if found else None
