"""Unit tests for code_usage._otel_delta_records (#280 follow-up).

Bridged claude.ai/code sessions only exist in the OTel store — these tests
pin the per-(day, family) delta semantics: fill what transcripts miss, never
double-count what they already carry, degrade to nothing when OTel is empty
or broken.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src import claude_code_otel, code_usage
from src.usage_common import UsageRecord


def _otel_rows(monkeypatch, rows):
    monkeypatch.setattr(
        claude_code_otel,
        "get_usage_summary",
        lambda period="all": {"rows": rows, "totals": {}, "period": period, "source": "otel"},
    )


def _jsonl_rec(day: str, model: str = "claude-fable-5", inp: int = 0, out: int = 0):
    return UsageRecord(
        session_id="s1",
        project_key="E--automation-demo",
        project_name="demo",
        model=model,
        ts=datetime.fromisoformat(day + "T09:00:00+00:00"),
        input_tokens=inp,
        output_tokens=out,
        cache_creation_tokens=0,
        cache_read_tokens=0,
    )


def test_bridged_only_day_fills_from_otel(monkeypatch):
    _otel_rows(monkeypatch, [
        {"date": "2026-07-12", "model": "Fable", "query_source": "main",
         "project": None, "input": 100, "output": 500, "cache_read": 2000, "cache_creation": 300},
    ])
    deltas = code_usage._otel_delta_records([])
    assert len(deltas) == 1
    d = deltas[0]
    assert d.vendor == "claude"
    assert d.project_name == "(untracked)"
    assert d.model == "Fable"
    assert (d.input_tokens, d.output_tokens) == (100, 500)
    assert d.cache_read_tokens == 2000 and d.cache_creation_tokens == 300
    assert d.requests == 0  # OTel token metrics carry no request count


def test_transcript_covered_day_yields_no_delta(monkeypatch):
    _otel_rows(monkeypatch, [
        {"date": "2026-07-10", "model": "Fable", "query_source": "main",
         "project": None, "input": 100, "output": 500, "cache_read": 0, "cache_creation": 0},
    ])
    covered = [_jsonl_rec("2026-07-10", inp=150, out=500)]
    assert code_usage._otel_delta_records(covered) == []


def test_partial_coverage_fills_only_the_excess(monkeypatch):
    _otel_rows(monkeypatch, [
        {"date": "2026-07-10", "model": "Fable", "query_source": "main",
         "project": None, "input": 100, "output": 500, "cache_read": 0, "cache_creation": 0},
        # A second source row for the same day/family aggregates first.
        {"date": "2026-07-10", "model": "Fable", "query_source": "auxiliary",
         "project": None, "input": 50, "output": 100, "cache_read": 0, "cache_creation": 0},
    ])
    covered = [_jsonl_rec("2026-07-10", inp=120, out=650)]
    deltas = code_usage._otel_delta_records(covered)
    assert len(deltas) == 1
    assert deltas[0].input_tokens == 30   # 150 otel - 120 covered
    assert deltas[0].output_tokens == 0   # covered exceeds otel -> clamped


def test_non_claude_records_do_not_mask_otel(monkeypatch):
    _otel_rows(monkeypatch, [
        {"date": "2026-07-10", "model": "Fable", "query_source": "main",
         "project": None, "input": 10, "output": 10, "cache_read": 0, "cache_creation": 0},
    ])
    codex = _jsonl_rec("2026-07-10", model="gpt-5.5", inp=999, out=999)
    codex.vendor = "codex"
    deltas = code_usage._otel_delta_records([codex])
    assert len(deltas) == 1 and deltas[0].input_tokens == 10


def test_otel_failure_degrades_to_empty(monkeypatch):
    def boom(period="all"):
        raise OSError("store unreadable")
    monkeypatch.setattr(claude_code_otel, "get_usage_summary", boom)
    assert code_usage._otel_delta_records([]) == []


def test_summary_end_to_end_with_bridged_day(monkeypatch, tmp_path):
    _otel_rows(monkeypatch, [
        {"date": datetime.now(tz=timezone.utc).date().isoformat(), "model": "Fable",
         "query_source": "main", "project": None,
         "input": 10, "output": 250, "cache_read": 40, "cache_creation": 5},
    ])
    monkeypatch.setattr(code_usage, "_gather_records", lambda vendor="all": [])
    summary = code_usage.get_summary("today", "claude")
    assert summary["totals"]["output_tokens"] == 250
    assert summary["totals"]["requests"] == 0
    assert any(p["project"] == "(untracked)" for p in summary["by_project"])
