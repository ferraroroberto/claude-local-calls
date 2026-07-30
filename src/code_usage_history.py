"""Durable daily rollups of Code-tab usage — outlives transcript pruning.

Claude Code deletes session transcripts after ~30 days (``cleanupPeriodDays``),
so the live parsers can never show more than a rolling month and the "All"
period silently equals "Month" (#280 follow-up, verified 2026-07-12: the
oldest file on disk is always ~30 days old).  This module snapshots what the
parsers *do* see into ``data/code_usage_history.json`` — one entry per
``(date, vendor, model, project)`` — so history accumulates from today
forward and keeps counting after the source files are pruned.

Semantics:
- **Max-merge on write**: a live day's totals only ever grow while its files
  exist and drop when files get pruned — keeping the field-wise maximum per
  key preserves each day's high-water mark without double counting.
- **Cutoff on read**: synthetic records are emitted only for days *older*
  than the oldest live record of the same vendor (or for vendors with no
  live records at all), so a day is sourced from exactly one place — live
  files while they exist, the snapshot after they're pruned.
- Aggregated entries carry a ``requests`` weight; ``UsageRecord.requests``
  feeds the summary's request counting so one synthetic row counts as the
  N calls it rolls up.

Same tolerant-load / atomic-write contract as ``startup_profile.py`` — a
missing or corrupt history file must never break the Code tab; it just
starts accumulating again.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.usage_common import UsageRecord

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_HISTORY_PATH: Path = _PROJECT_ROOT / "data" / "code_usage_history.json"
_HISTORY_PATH: Path = _DEFAULT_HISTORY_PATH

# Persist at most this often — the SPA polls every 30 s and a save is an
# atomic full-file rewrite, so batch a few polls per write.
_SAVE_MIN_INTERVAL_S = 120.0

_SUM_FIELDS = (
    "requests",
    "input_tokens",
    "output_tokens",
    "cache_creation_tokens",
    "cache_read_tokens",
    "reasoning_output_tokens",
    "credits_usd",
)

_lock = threading.Lock()
_entries: Optional[Dict[str, dict]] = None  # key -> entry dict, lazy-loaded
_dirty = False
_last_save = 0.0


def _reset_for_tests(path: Optional[Path] = None) -> None:
    """Wipe state and point at ``path`` (``None`` restores the default)."""
    global _HISTORY_PATH, _entries, _dirty, _last_save
    with _lock:
        _HISTORY_PATH = path if path is not None else _DEFAULT_HISTORY_PATH
        _entries = None
        _dirty = False
        _last_save = 0.0


def _key(day: str, vendor: str, model: str, project_key: str) -> str:
    return f"{day}|{vendor}|{model}|{project_key}"


def _load_locked() -> Dict[str, dict]:
    """Load entries from disk (call with ``_lock`` held)."""
    global _entries
    if _entries is not None:
        return _entries
    _entries = {}
    try:
        raw = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
        entries = raw.get("entries")
        if isinstance(entries, dict):
            _entries = {
                k: v for k, v in entries.items() if isinstance(v, dict)
            }
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as exc:
        logger.warning(
            "⚠️ code_usage_history: %s unreadable (%s) — starting fresh",
            _HISTORY_PATH, exc,
        )
    return _entries


def _save_locked() -> None:
    global _dirty, _last_save
    try:
        _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _HISTORY_PATH.with_suffix(".json.tmp")
        payload = {
            "updated": datetime.now(tz=timezone.utc).isoformat(),
            "entries": _entries or {},
        }
        tmp.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        os.replace(tmp, _HISTORY_PATH)
        _dirty = False
        _last_save = time.time()
    except OSError as exc:
        logger.warning("⚠️ code_usage_history: save failed: %s", exc)


def update_from_records(records: List[UsageRecord]) -> None:
    """Fold live records into the snapshot (max-merge per day/vendor/model/
    project) and persist opportunistically.  Never raises."""
    global _dirty
    if not records:
        return
    # Aggregate outside the lock — the records list is ours to walk.
    fresh: Dict[str, dict] = {}
    for r in records:
        day = r.ts.astimezone(timezone.utc).date().isoformat()
        k = _key(day, r.vendor, r.model, r.project_key)
        e = fresh.get(k)
        if e is None:
            e = fresh[k] = {
                "date": day,
                "vendor": r.vendor,
                "model": r.model,
                "project_key": r.project_key,
                "project_name": r.project_name,
                **{f: 0 for f in _SUM_FIELDS},
            }
        e["requests"] += getattr(r, "requests", 1)
        e["input_tokens"] += r.input_tokens
        e["output_tokens"] += r.output_tokens
        e["cache_creation_tokens"] += r.cache_creation_tokens
        e["cache_read_tokens"] += r.cache_read_tokens
        e["reasoning_output_tokens"] += r.reasoning_output_tokens
        e["credits_usd"] += r.credits_usd

    with _lock:
        entries = _load_locked()
        for k, e in fresh.items():
            stored = entries.get(k)
            if stored is None:
                entries[k] = e
                _dirty = True
                continue
            for f in _SUM_FIELDS:
                if e[f] > stored.get(f, 0):
                    stored[f] = e[f]
                    _dirty = True
        if _dirty and time.time() - _last_save >= _SAVE_MIN_INTERVAL_S:
            _save_locked()


def synthetic_records(
    live_records: List[UsageRecord], vendor: str
) -> List[UsageRecord]:
    """Records for days the live parsers no longer see.

    Per vendor, only days strictly older than the oldest live record are
    synthesized (a vendor with no live records contributes all its history),
    so no day is ever counted from both sources.
    """
    live_min: Dict[str, date] = {}
    for r in live_records:
        d = r.ts.astimezone(timezone.utc).date()
        v = r.vendor
        if v not in live_min or d < live_min[v]:
            live_min[v] = d

    out: List[UsageRecord] = []
    with _lock:
        entries = list(_load_locked().values())
    for e in entries:
        try:
            v = e["vendor"]
            if vendor != "all" and v != vendor:
                continue
            d = date.fromisoformat(e["date"])
            cutoff = live_min.get(v)
            if cutoff is not None and d >= cutoff:
                continue
            out.append(
                UsageRecord(
                    session_id=f"history:{e['date']}",
                    project_key=e.get("project_key") or "history",
                    project_name=e.get("project_name") or "(history)",
                    model=e.get("model") or "unknown",
                    ts=datetime(d.year, d.month, d.day, 12, tzinfo=timezone.utc),
                    input_tokens=int(e.get("input_tokens") or 0),
                    output_tokens=int(e.get("output_tokens") or 0),
                    cache_creation_tokens=int(e.get("cache_creation_tokens") or 0),
                    cache_read_tokens=int(e.get("cache_read_tokens") or 0),
                    reasoning_output_tokens=int(
                        e.get("reasoning_output_tokens") or 0
                    ),
                    vendor=v,
                    credits_usd=float(e.get("credits_usd") or 0.0),
                    requests=int(e.get("requests") or 1),
                )
            )
        except (KeyError, ValueError, TypeError):
            continue  # one malformed entry never breaks the tab
    return out
