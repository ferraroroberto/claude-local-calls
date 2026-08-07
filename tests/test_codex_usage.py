"""Unit tests for the Codex (OpenAI) usage parser (issue #71).

These run without a real ``~/.codex`` tree: each test writes a synthetic
rollout JSONL file into a temp dir and points the parser at it.

The most important guarantee is that the parser uses the per-turn delta
``last_token_usage`` and never the cumulative ``total_token_usage`` — summing
the cumulative field would massively double-count.
"""

from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path

import pytest

from src import code_usage, codex_usage, usage_charts, usage_common, usage_pricing


def _write_rollout(path: Path, lines: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(obj) for obj in lines) + "\n", encoding="utf-8"
    )


def _token_count(ts: str, last: dict, total: dict) -> dict:
    return {
        "timestamp": ts,
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {"last_token_usage": last, "total_token_usage": total},
        },
    }


@pytest.fixture
def codex_dir(tmp_path, monkeypatch):
    """Point the parser at a temp sessions dir and clear its mtime cache."""
    sessions = tmp_path / "sessions"
    monkeypatch.setattr(codex_usage, "_CODEX_SESSIONS_DIR", sessions)
    codex_usage._file_cache.clear()
    return sessions


def test_uses_last_not_cumulative(codex_dir):
    """Two turns: assert totals reflect the per-turn deltas, not cumulative."""
    f = codex_dir / "2026" / "06" / "05" / "rollout-a.jsonl"
    _write_rollout(f, [
        {"type": "session_meta", "payload": {"id": "sess-1", "cwd": "E:\\automation\\demo"}},
        {"type": "turn_context", "payload": {"model": "gpt-5.5", "cwd": "E:\\automation\\demo"}},
        _token_count(
            "2026-06-05T20:25:43.521Z",
            last={"input_tokens": 100, "cached_input_tokens": 10, "output_tokens": 20, "reasoning_output_tokens": 5},
            total={"input_tokens": 100, "cached_input_tokens": 10, "output_tokens": 20, "reasoning_output_tokens": 5},
        ),
        _token_count(
            "2026-06-05T20:26:00.000Z",
            last={"input_tokens": 50, "cached_input_tokens": 5, "output_tokens": 10, "reasoning_output_tokens": 2},
            total={"input_tokens": 150, "cached_input_tokens": 15, "output_tokens": 30, "reasoning_output_tokens": 7},
        ),
    ])

    records = codex_usage.all_records()
    assert len(records) == 2
    # Per-turn deltas: 100 + 50 = 150 (NOT 100 + 150 = 250 from cumulative).
    assert sum(r.input_tokens for r in records) == 150
    assert sum(r.output_tokens for r in records) == 30
    assert sum(r.cache_read_tokens for r in records) == 15
    assert sum(r.reasoning_output_tokens for r in records) == 7
    assert all(r.vendor == "codex" for r in records)
    assert all(r.cache_creation_tokens == 0 for r in records)


def test_model_and_project_attribution(codex_dir):
    """Model comes from turn_context; project key matches Claude's encoding."""
    f = codex_dir / "2026" / "06" / "05" / "rollout-b.jsonl"
    _write_rollout(f, [
        {"type": "session_meta", "payload": {"id": "sess-2", "cwd": "E:\\automation\\local-llm-hub"}},
        {"type": "turn_context", "payload": {"model": "gpt-5.5", "cwd": "E:\\automation\\local-llm-hub"}},
        _token_count(
            "2026-06-05T20:25:43.521Z",
            last={"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 1, "reasoning_output_tokens": 0},
            total={"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 1, "reasoning_output_tokens": 0},
        ),
    ])

    rec = codex_usage.all_records()[0]
    assert rec.model == "gpt-5.5"
    assert rec.session_id == "sess-2"
    assert rec.project_key == usage_common.encode_project_key("E:\\automation\\local-llm-hub")
    assert rec.ts.tzinfo is timezone.utc


def test_empty_when_dir_absent(codex_dir):
    """Missing sessions dir yields no records, no error."""
    assert codex_usage.all_records() == []


def test_codex_cost_prices_cached_subset(codex_dir):
    """Codex cost prices the non-cached input remainder + cached portion separately."""
    # 1M input of which 200k cached, 100k output, gpt-5.5 → $5 / $0.50 / $30 per M.
    r = usage_common.UsageRecord(
        session_id="s", project_key="k", project_name="k", model="gpt-5.5",
        ts=codex_usage.parse_iso_ts("2026-06-05T20:00:00Z"),
        input_tokens=1_000_000, output_tokens=100_000,
        cache_creation_tokens=0, cache_read_tokens=200_000,
        reasoning_output_tokens=0, vendor="codex",
    )
    input_cost, output_cost, cache_cost = usage_pricing.record_costs(r)
    # non-cached input 800k * $5/M = $4.00 ; cached 200k * $0.50/M = $0.10 ; output 100k * $30/M = $3.00
    assert input_cost == pytest.approx(4.0)
    assert cache_cost == pytest.approx(0.10)
    assert output_cost == pytest.approx(3.0)


def test_claude_fable_cost_uses_fable_family_rates():
    """claude-fable-5 records price at the Fable family rate, not $0 (falling
    through model_display unmatched would silently zero the cost)."""
    # 1M input, 200k of it cache reads, 100k output. Fable: $10 / $1 / $50 per M.
    r = usage_common.UsageRecord(
        session_id="s", project_key="k", project_name="k", model="claude-fable-5",
        ts=codex_usage.parse_iso_ts("2026-07-09T20:00:00Z"),
        input_tokens=1_000_000, output_tokens=100_000,
        cache_creation_tokens=0, cache_read_tokens=200_000,
    )
    input_cost, output_cost, cache_cost = usage_pricing.record_costs(r)
    assert input_cost == pytest.approx(10.0)
    assert cache_cost == pytest.approx(0.20)
    assert output_cost == pytest.approx(5.0)


def test_prev_totals_zero_filled_not_none():
    """A non-'all' period returns a zero-filled prev dict (not None) when the
    preceding window had no activity, so the SPA can render a 'new' badge
    instead of hiding the comparison (issue #71)."""
    from datetime import date, datetime, timezone

    today = date(2026, 6, 6)
    # Record inside the CURRENT week → the previous-week window is empty.
    r = usage_common.UsageRecord(
        session_id="s", project_key="k", project_name="k", model="gpt-5.5",
        ts=datetime(2026, 6, 5, 12, tzinfo=timezone.utc),
        input_tokens=10, output_tokens=1, cache_creation_tokens=0,
        cache_read_tokens=0, reasoning_output_tokens=0, vendor="codex",
    )
    prev = usage_charts.build_prev_totals([r], "week", today)
    assert prev is not None and prev["requests"] == 0
    # All-time still omits the comparison.
    assert usage_charts.build_prev_totals([r], "all", today) is None


def test_summary_by_vendor_merges(codex_dir, monkeypatch):
    """get_summary('all','codex') tags codex and surfaces a by_vendor row."""
    f = codex_dir / "2026" / "06" / "05" / "rollout-c.jsonl"
    _write_rollout(f, [
        {"type": "session_meta", "payload": {"id": "sess-3", "cwd": "E:\\automation\\demo"}},
        {"type": "turn_context", "payload": {"model": "gpt-5.5", "cwd": "E:\\automation\\demo"}},
        _token_count(
            "2026-06-05T20:25:43.521Z",
            last={"input_tokens": 100, "cached_input_tokens": 0, "output_tokens": 10, "reasoning_output_tokens": 3},
            total={"input_tokens": 100, "cached_input_tokens": 0, "output_tokens": 10, "reasoning_output_tokens": 3},
        ),
    ])

    summary = code_usage.get_summary("all", "codex")
    assert summary["vendor"] == "codex"
    vendors = {row["vendor"] for row in summary["by_vendor"]}
    assert vendors == {"codex"}
    assert summary["totals"]["reasoning_output_tokens"] == 3


def test_chart_buckets_and_prev_totals_use_requests_weight():
    """issue #474: usage_charts must accumulate the ``requests`` *weight*, not
    1-per-row. A synthetic rollup row carries the N calls it summarises and an
    OTel-derived row carries 0, so counting rows undercounted every day past
    Claude Code's ~30-day transcript retention and overcounted OTel deltas."""
    from datetime import date, datetime, timezone

    from src import usage_charts

    today = date(2026, 6, 30)

    def _r(day: datetime, requests: int) -> usage_common.UsageRecord:
        return usage_common.UsageRecord(
            session_id="s", project_key="k", project_name="k",
            model="claude-sonnet-5", ts=day,
            input_tokens=1, output_tokens=1, cache_creation_tokens=0,
            cache_read_tokens=0, requests=requests,
        )

    # A rollup row worth 7 calls plus an OTel row worth 0, same day.
    in_bucket = datetime(2026, 6, 30, 12, tzinfo=timezone.utc)
    series = usage_charts.build_time_series(
        [_r(in_bucket, 7), _r(in_bucket, 0)], "today", today,
    )
    assert series[-1]["models"]["Sonnet"]["requests"] == 7

    # Previous "month" window is days 30-59 back — always entirely synthetic.
    prev_day = datetime(2026, 5, 20, 12, tzinfo=timezone.utc)  # 41 days back
    prev = usage_charts.build_prev_totals(
        [_r(prev_day, 7), _r(prev_day, 0)], "month", today,
    )
    assert prev is not None and prev["requests"] == 7
