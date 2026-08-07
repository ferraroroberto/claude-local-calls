"""Time-series bucketing and period-over-period comparison for the Cld tab.

Extracted out of ``code_usage.py`` (#451) — chart bucketing (weekly/monthly
rollups, the previous-period comparison) was one of five unrelated concerns
that module mixed together. ``code_usage.py`` imports :func:`build_time_series`
and :func:`build_prev_totals` for its ``get_summary`` aggregation; nothing
here depends back on it.
"""

from __future__ import annotations

from datetime import date, timedelta, timezone
from typing import Dict, List, Optional

from src.usage_common import UsageRecord, model_display

# How many days of daily history to return (also bounds code_usage.get_summary's
# plain "daily" list, not just the "today"-period chart buckets here).
MAX_DAILY_DAYS = 14

# How many weeks / months of history to return for the trend charts.
MAX_CHART_WEEKS = 12
MAX_CHART_MONTHS = 12


def _week_start(d: date) -> date:
    """Return the Monday of the ISO week containing d."""
    return d - timedelta(days=d.weekday())


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def build_time_series(
    records: List[UsageRecord], period: str, today: date
) -> list:
    """Build oldest-first time-series buckets with per-model breakdown for the chart.

    Returned list is empty for ``period == "all"`` (unbounded x-axis is not useful).
    Each bucket: ``{"label": str, "models": {family: {input_tokens, output_tokens, requests}}}``.
    ``input_tokens`` already folds in ``cache_creation_tokens`` so the chart shows billed in.
    """
    if period == "all":
        return []

    if period == "today":
        buckets = [today - timedelta(days=i) for i in range(MAX_DAILY_DAYS - 1, -1, -1)]
    elif period == "week":
        this_mon = _week_start(today)
        buckets = [this_mon - timedelta(weeks=i) for i in range(MAX_CHART_WEEKS - 1, -1, -1)]
    else:  # month
        ym: List[tuple] = []
        y, m = today.year, today.month
        for _ in range(MAX_CHART_MONTHS):
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
        family = model_display(r.model)
        if family not in bmap[bk]:
            bmap[bk][family] = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "requests": 0}
        slot = bmap[bk][family]
        slot["input_tokens"] += r.input_tokens + r.cache_creation_tokens
        slot["output_tokens"] += r.output_tokens
        slot["cache_read_tokens"] += r.cache_read_tokens
        # ``requests`` is an aggregation weight, not a flag (see UsageRecord):
        # a code_usage_history synthetic rollup row carries the N calls it
        # summarises, and an OTel-derived record carries 0. Counting 1 per row
        # undercounted every pruned day and overcounted every OTel delta (#474).
        slot["requests"] += r.requests

    result = []
    for b in buckets:
        lbl = b.strftime("%b %Y") if period == "month" else b.strftime("%b ") + str(b.day)
        result.append({"label": lbl, "models": bmap[b]})
    return result


def build_prev_totals(
    records: List[UsageRecord], period: str, today: date
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
            # Weight, not a flag — same reason as build_time_series (#474).
            # The "month" window (days 30-59 back) is always entirely past
            # Claude Code's ~30-day transcript retention, so it is sourced
            # from synthetic rollup rows alone and undercounted by 1-per-row.
            acc["requests"] += r.requests
    return acc
