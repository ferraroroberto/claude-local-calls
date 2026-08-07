"""Claude Code OTel metrics receiver — data layer (issue #68).

Claude Code (any host session, including sub-agents spawned via the Task
tool) can export ``claude_code.token.usage`` / ``claude_code.cost.usage``
OTel metrics over OTLP/HTTP. The hub's own OTel pipeline
(``src/observability.py``) only ever *exports* to Langfuse — this module is
the receive side: parse what Claude Code pushes at ``POST /v1/metrics``
(``src/server_otel_receiver.py``), persist it, and roll it up for the OTel
tab's "Claude Code (host CLI)" panel.

This closes the gap documented in #66: the Code tab's JSONL parser
(``code_usage.py``) never sees sub-agent API calls (Claude Code doesn't write
them to session transcripts), so per-model totals there silently undercount
whenever sub-agents run. OTel is the only channel that carries them.

Design notes:

- **Delta temporality, verified empirically.** Both metrics are exported as
  ``Sum`` with ``aggregation_temporality=DELTA`` (confirmed by capturing a
  real export from a throwaway ``claude -p`` call during planning) — each
  data point already *is* the incremental delta since the last export, so
  ingestion just sums values directly. No last-value bookkeeping, no special
  handling for a Claude Code process restart.
- **Data-minimization.** Data points also carry identity attributes
  (``user.id``, ``user.email``, ``user.account_uuid``, ``user.account_id``,
  ``organization.id``, ``session.id``, ``terminal.type``) — these are read
  off the protobuf but never stored or logged. Only ``model``,
  ``query_source``, (for the token metric) ``type``, and (when the sending
  session set it) ``project.name`` are persisted.
- **Project attribution is automatic, via fleet-config, not this repo**
  (issue #234, automated by ``fleet-config``#310). Unlike the Code tab's
  JSONL source — where "project" is trivially the session file's own
  directory — Claude Code's OTel metrics carry no cwd/project attribute by
  default. Verified empirically that setting ``OTEL_RESOURCE_ATTRIBUTES``
  (e.g. ``project.name=<repo>``) before launching ``claude`` *does* flatten
  onto every data point's own attributes (not just the resource level), so
  it round-trips through this receiver correctly. The host's ``fleet-config``
  repo wires a ``claude`` shell-function wrapper into ``$PROFILE`` that sets
  this automatically from the current git repo — see that repo's
  ``docs/otel-project-attribution.md`` for the mechanism (deliberately not
  duplicated here: this repo is just the one consumer of an attribute a
  different repo is responsible for setting). ``docs/telemetry-langfuse.md``
  covers the manual fallback for invocations outside an interactive shell.
- **Plain JSONL, no DB** — matches the rest of this repo's usage-tracking
  (Claude Code's own session logs, ``code_usage.py``'s parser). No rotation;
  revisit only if this becomes a real problem (same posture as the JSONL
  files this module is a workaround for).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.usage_common import model_display, period_since

_log = logging.getLogger(__name__)

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
_DATA_DIR: Path = _PROJECT_ROOT / "data" / "telemetry"
_DATA_FILE: Path = _DATA_DIR / "claude_code_otel_usage.jsonl"

_TOKEN_METRIC = "claude_code.token.usage"
_COST_METRIC = "claude_code.cost.usage"

# Maps the OTel `type` attribute on claude_code.token.usage data points to
# our row field names.
_TOKEN_TYPE_FIELD = {
    "input": "input",
    "output": "output",
    "cacheRead": "cache_read",
    "cacheCreation": "cache_creation",
}

_PERIODS = {"today", "week", "month", "all"}


@dataclass
class UsagePoint:
    """One ingested data point, PII already stripped."""

    ts: datetime
    metric: str  # "token" | "cost"
    model: str
    query_source: str
    token_type: Optional[str]  # only set when metric == "token"
    value: float
    project: Optional[str]  # from OTEL_RESOURCE_ATTRIBUTES' project.name, if set


def _attr_value(value: Any) -> Any:
    """Pull the scalar out of an OTel protobuf ``AnyValue``."""
    kind = value.WhichOneof("value")
    if kind is None:
        return None
    return getattr(value, kind)


def parse_export_request(raw: bytes) -> List[UsagePoint]:
    """Decode an OTLP ``ExportMetricsServiceRequest`` into :class:`UsagePoint`\\ s.

    Only ``claude_code.token.usage`` / ``claude_code.cost.usage`` Sum metrics
    are kept; anything else in the export is ignored. Raises on malformed
    protobuf — callers (``ingest_export_request``) catch and log.
    """
    from opentelemetry.proto.collector.metrics.v1 import metrics_service_pb2

    req = metrics_service_pb2.ExportMetricsServiceRequest()
    req.ParseFromString(raw)

    points: List[UsagePoint] = []
    for resource_metrics in req.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name not in (_TOKEN_METRIC, _COST_METRIC):
                    continue
                if metric.WhichOneof("data") != "sum":
                    continue
                metric_kind = "token" if metric.name == _TOKEN_METRIC else "cost"
                for dp in metric.sum.data_points:
                    attrs = {a.key: _attr_value(a.value) for a in dp.attributes}
                    value = dp.as_double if dp.HasField("as_double") else float(dp.as_int)
                    ts = datetime.fromtimestamp(dp.time_unix_nano / 1e9, tz=timezone.utc)
                    token_type = None
                    if metric_kind == "token":
                        token_type = attrs.get("type")
                    project = attrs.get("project.name")
                    points.append(
                        UsagePoint(
                            ts=ts,
                            metric=metric_kind,
                            model=str(attrs.get("model") or "unknown"),
                            query_source=str(attrs.get("query_source") or "unknown"),
                            token_type=str(token_type) if token_type else None,
                            value=float(value),
                            project=str(project) if project else None,
                        )
                    )
    return points


def ingest_export_request(raw: bytes) -> int:
    """Parse + append an OTLP metrics export to the persisted JSONL log.

    Never raises — a malformed export is logged at ``warning`` and dropped,
    matching this repo's "telemetry must never break the request" posture
    (here applied to the receive side: a bad export must never surface as an
    error to the Claude Code CLI that sent it). Returns the number of points
    ingested (0 on failure or an export with nothing relevant in it).
    """
    try:
        points = parse_export_request(raw)
    except Exception:  # noqa: BLE001
        _log.warning("⚠️ failed to parse OTLP metrics export", exc_info=True)
        return 0
    if not points:
        return 0
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        with _DATA_FILE.open("a", encoding="utf-8") as f:
            for p in points:
                f.write(
                    json.dumps(
                        {
                            "ts": p.ts.isoformat(),
                            "metric": p.metric,
                            "model": p.model,
                            "query_source": p.query_source,
                            "token_type": p.token_type,
                            "value": p.value,
                            "project": p.project,
                        }
                    )
                    + "\n"
                )
    except Exception:  # noqa: BLE001
        _log.warning("⚠️ failed to persist Claude Code OTel usage", exc_info=True)
        return 0
    return len(points)


# --------------------------------------------------------------------- reads

@dataclass
class _FileCache:
    mtime: float
    points: List[UsagePoint]


_file_cache: Optional[_FileCache] = None


def _load_points() -> List[UsagePoint]:
    """Reparse the JSONL log only when its mtime changes (same technique as
    ``code_usage.py``'s ``_file_cache``, simpler since it's a single file)."""
    global _file_cache
    try:
        mtime = _DATA_FILE.stat().st_mtime
    except OSError:
        return []

    if _file_cache is not None and _file_cache.mtime == mtime:
        return _file_cache.points

    points: List[UsagePoint] = []
    try:
        with _DATA_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    points.append(
                        UsagePoint(
                            ts=datetime.fromisoformat(row["ts"]),
                            metric=row["metric"],
                            model=row["model"],
                            query_source=row["query_source"],
                            token_type=row.get("token_type"),
                            value=float(row["value"]),
                            project=row.get("project"),
                        )
                    )
                except Exception:  # noqa: BLE001
                    continue
    except OSError:
        return []

    _file_cache = _FileCache(mtime=mtime, points=points)
    return points


def get_usage_summary(period: str = "today") -> Dict[str, Any]:
    """Per-(date, model, query_source, project) rollup for the OTel tab's
    Claude Code panel.

    ``period`` is one of ``today | week | month | all``; unrecognised values
    fall back to ``all`` (unbounded) rather than raising. Rows are broken out
    by day (issue #233) — each ingested point already carries a timestamp, so
    this attributes cost/usage to the day it actually happened rather than
    collapsing the whole selected window into one aggregate row per model.
    ``project`` is ``None`` (rendered "—" by the SPA) for the vast majority of
    sessions, which never set ``OTEL_RESOURCE_ATTRIBUTES`` — see #234.
    """
    if period not in _PERIODS:
        period = "all"

    points = _load_points()
    since = period_since(period)
    if since is not None:
        points = [p for p in points if p.ts.date() >= since]

    rows: Dict[Tuple[str, str, str, Optional[str]], Dict[str, float]] = {}
    for p in points:
        key = (p.ts.date().isoformat(), model_display(p.model), p.query_source, p.project)
        row = rows.setdefault(
            key,
            {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_creation": 0.0, "cost_usd": 0.0},
        )
        if p.metric == "cost":
            row["cost_usd"] += p.value
        else:
            field = _TOKEN_TYPE_FIELD.get(p.token_type or "")
            if field:
                row[field] += p.value

    def _round_row(vals: Dict[str, float]) -> Dict[str, Any]:
        return {
            "input": int(vals["input"]),
            "output": int(vals["output"]),
            "cache_read": int(vals["cache_read"]),
            "cache_creation": int(vals["cache_creation"]),
            "cost_usd": round(vals["cost_usd"], 6),
        }

    # Most recent day first; within a day, sorted by model/source/project
    # (two-pass stable sort — first by model/source/project, then by date desc).
    by_model_source = sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][3] or ""))
    by_date_desc = sorted(by_model_source, key=lambda kv: kv[0][0], reverse=True)

    out_rows = []
    totals = {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_creation": 0.0, "cost_usd": 0.0}
    for (day, model, source, project), vals in by_date_desc:
        out_rows.append(
            {"date": day, "model": model, "query_source": source, "project": project, **_round_row(vals)}
        )
    for vals in rows.values():
        for k in totals:
            totals[k] += vals[k]

    return {
        "period": period,
        "rows": out_rows,
        "totals": _round_row(totals),
        "source": "otel",
    }


def _reset_for_tests() -> None:
    """Tests-only — wipe the mtime cache so a fresh file read happens."""
    global _file_cache
    _file_cache = None
