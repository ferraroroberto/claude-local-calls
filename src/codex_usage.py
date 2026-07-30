"""Host-side Codex (OpenAI) usage parser.

Reads the rollout JSONL session logs Codex writes under
``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`` and emits usage records in
the shared ``UsageRecord`` shape (tagged ``vendor="codex"``), so they flow
through the same Code-tab summary builder as Claude data (issue #71).

Design constraints mirror ``code_usage.py``:
- **Read-only** — we never modify the rollout files.
- **Zero subprocesses** — we parse the files ourselves.
- **Passive** — nothing runs on any request path; the SPA polls on its own
  30 s interval.
- **Mtime cache** — each file is only re-parsed when its mtime changes.

Token mapping — one ``UsageRecord`` per ``token_count`` event, using the
per-turn delta ``last_token_usage`` (**never** the cumulative
``total_token_usage``, which would massively double-count when summed)::

    last_token_usage.input_tokens            -> input_tokens   (incl. cached)
    last_token_usage.cached_input_tokens     -> cache_read_tokens
    last_token_usage.output_tokens           -> output_tokens  (incl. reasoning)
    last_token_usage.reasoning_output_tokens -> reasoning_output_tokens

Cross-vendor semantic note: for Codex the ``cached_input_tokens`` are a
*subset* of ``input_tokens`` (the cost path in ``code_usage.py`` prices the
non-cached remainder at the input rate and the cached portion at the cached
rate), whereas Claude's ``cache_read_input_tokens`` are reported
separately/additively.  ``cache_creation_tokens`` is always 0 for Codex.  The
``reasoning_output_tokens`` are a subset of ``output_tokens`` and are carried
for display only — never summed into output.

``model`` and ``cwd`` come from the most recent ``turn_context`` entry seen
while walking the file (seeded from ``session_meta``); ``cwd`` is encoded into
the same project key Claude uses so the same repo groups across vendors.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.usage_common import (
    FileStats,
    UsageRecord,
    encode_project_key,
    load_cached,
    parse_iso_ts,
    project_pretty,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CODEX_SESSIONS_DIR: Path = Path.home() / ".codex" / "sessions"

# File-level mtime cache (module-level singleton), independent of the Claude one.
_file_cache: Dict[str, FileStats] = {}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_rollout_file(path: Path) -> List[UsageRecord]:
    """Parse one rollout JSONL file and return usage records."""
    records: List[UsageRecord] = []
    cur_model = "unknown"
    cur_cwd: Optional[str] = None
    cur_session_id = path.stem  # rollout-<ts>-<uuid>; overridden by session_meta.id

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

                etype = obj.get("type")
                payload = obj.get("payload") or {}

                if etype == "session_meta":
                    cur_session_id = payload.get("id") or cur_session_id
                    if payload.get("cwd"):
                        cur_cwd = payload["cwd"]
                    continue

                if etype == "turn_context":
                    if payload.get("model"):
                        cur_model = payload["model"]
                    if payload.get("cwd"):
                        cur_cwd = payload["cwd"]
                    continue

                if etype == "event_msg" and payload.get("type") == "token_count":
                    info = payload.get("info") or {}
                    usage = info.get("last_token_usage") or {}
                    if not usage:
                        continue
                    # Skip token_count events emitted before the first
                    # turn_context (no model yet) — these are unattributable
                    # noise, not real coding turns.
                    if cur_model == "unknown":
                        continue
                    key = encode_project_key(cur_cwd) if cur_cwd else "codex"
                    records.append(
                        UsageRecord(
                            session_id=cur_session_id,
                            project_key=key,
                            project_name=(
                                project_pretty(key) if cur_cwd else "(unknown)"
                            ),
                            model=cur_model,
                            ts=parse_iso_ts(obj.get("timestamp")),
                            input_tokens=int(usage.get("input_tokens") or 0),
                            output_tokens=int(usage.get("output_tokens") or 0),
                            cache_creation_tokens=0,
                            cache_read_tokens=int(
                                usage.get("cached_input_tokens") or 0
                            ),
                            reasoning_output_tokens=int(
                                usage.get("reasoning_output_tokens") or 0
                            ),
                            vendor="codex",
                        )
                    )
    except OSError as exc:
        _log.warning("⚠️ codex_usage: cannot read %s: %s", path, exc)
    return records


def _load_file(path: Path) -> List[UsageRecord]:
    """Return cached records, re-parsing only when the file has changed."""
    return load_cached(path, _file_cache, _parse_rollout_file)


def all_records() -> List[UsageRecord]:
    """Scan all Codex rollout files and return every usage record."""
    records: List[UsageRecord] = []

    if not _CODEX_SESSIONS_DIR.exists():
        return records

    try:
        rollout_files = list(_CODEX_SESSIONS_DIR.rglob("rollout-*.jsonl"))
    except OSError as exc:
        _log.warning(
            "⚠️ codex_usage: cannot scan %s: %s", _CODEX_SESSIONS_DIR, exc
        )
        return records

    for rf in rollout_files:
        records.extend(_load_file(rf))

    return records
