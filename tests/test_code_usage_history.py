"""Unit tests for src/code_usage_history.py (#280 follow-up).

The conftest autouse fixture points the snapshot at a per-test temp file;
these tests exercise the max-merge write path, the per-vendor cutoff read
path, and the requests-weight flow through get_summary.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src import code_usage, code_usage_history as hist
from src.usage_common import UsageRecord


def _rec(day: str, vendor: str = "claude", model: str = "claude-sonnet-5",
         out: int = 100, sid: str = "s1") -> UsageRecord:
    return UsageRecord(
        session_id=sid,
        project_key="E--automation-demo",
        project_name="demo",
        model=model,
        ts=datetime.fromisoformat(day + "T10:00:00+00:00"),
        input_tokens=10,
        output_tokens=out,
        cache_creation_tokens=0,
        cache_read_tokens=5,
        vendor=vendor,
    )


def test_max_merge_and_cutoff_no_double_count():
    live = [_rec("2026-07-01"), _rec("2026-07-01", sid="s2"), _rec("2026-07-10")]
    hist.update_from_records(live)

    # While the day is still live, no synthetic record duplicates it.
    assert hist.synthetic_records(live, "all") == []

    # Files for 2026-07-01 get pruned: only that day comes back, with the
    # rolled-up requests weight and summed tokens.
    pruned = [_rec("2026-07-10")]
    hist.update_from_records(pruned)  # shrunken view must not shrink history
    synth = hist.synthetic_records(pruned, "all")
    assert len(synth) == 1
    s = synth[0]
    assert s.ts.date().isoformat() == "2026-07-01"
    assert s.requests == 2
    assert s.output_tokens == 200
    assert s.vendor == "claude"


def test_vendor_scoping_and_absent_vendor():
    hist.update_from_records([_rec("2026-07-01"), _rec("2026-07-02", vendor="agy", model="gemini-3-pro")])
    # agy has no live records at all -> all its history applies.
    synth = hist.synthetic_records([_rec("2026-07-05")], "agy")
    assert [r.vendor for r in synth] == ["agy"]
    # Vendor filter excludes other vendors' history.
    assert all(r.vendor == "claude" for r in hist.synthetic_records([], "claude"))


def test_requests_weight_flows_through_summary(monkeypatch):
    # No live records at all: summary is fed purely by synthetic history.
    hist.update_from_records([_rec("2026-06-01"), _rec("2026-06-01", sid="s2")])
    monkeypatch.setattr(code_usage, "_gather_records", lambda vendor="all": [])
    summary = code_usage.get_summary("all", "claude")
    assert summary["totals"]["requests"] == 2
    assert summary["totals"]["output_tokens"] == 200


def test_corrupt_file_starts_fresh(tmp_path):
    p = tmp_path / "hist.json"
    p.write_text("{not json", encoding="utf-8")
    hist._reset_for_tests(p)
    hist.update_from_records([_rec("2026-07-01")])  # must not raise
    assert len(hist.synthetic_records([], "all")) == 1


def test_persists_across_reload(tmp_path):
    p = tmp_path / "hist.json"
    hist._reset_for_tests(p)
    hist.update_from_records([_rec("2026-07-01")])
    with hist._lock:
        hist._save_locked()
    hist._reset_for_tests(p)  # simulate hub restart
    synth = hist.synthetic_records([], "all")
    assert len(synth) == 1
    assert synth[0].output_tokens == 100
