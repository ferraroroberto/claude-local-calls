"""Unit tests for src/fleet_maintenance.py (issue #411).

Host-scoped drain marker: set/is_under/clear round-trip, lazy expiry (an
entry with ``until`` in the past reads as not-under-maintenance with no
write needed), duration clamping, unknown-host rejection, and opportunistic
pruning of expired rows on write.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

import pytest  # noqa: E402

from src import fleet_maintenance as fm  # noqa: E402


def _isolate(monkeypatch, tmp_path, initial=None):
    target = tmp_path / "fleet_maintenance.json"
    if initial is not None:
        target.write_text(json.dumps(initial), encoding="utf-8")
    monkeypatch.setattr(fm, "DEFAULT_MAINTENANCE_PATH", target)
    fm._MAINTENANCE_CACHE.clear()
    return target


def test_load_missing_returns_empty(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    assert fm.load_fleet_maintenance() == {}


def test_load_tolerant_on_garbage(monkeypatch, tmp_path):
    target = _isolate(monkeypatch, tmp_path)
    target.write_text("{ not json", encoding="utf-8")
    assert fm.load_fleet_maintenance() == {}


def test_load_drops_malformed_rows(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path, {
        "gaming": {"until": 1000.0, "reason": "drill"},
        "bad-until": {"until": "not-a-number"},
        "not-a-dict": "oops",
    })
    got = fm.load_fleet_maintenance()
    assert got == {"gaming": {"until": 1000.0, "reason": "drill"}}


def test_set_maintenance_arms_and_is_under_maintenance(monkeypatch, tmp_path):
    target = _isolate(monkeypatch, tmp_path)
    entry = fm.set_maintenance("gaming", duration_s=300.0, reason="drill", now=1000.0)
    assert entry["host_id"] == "gaming"
    assert entry["until"] == 1300.0
    assert fm.is_under_maintenance("gaming", now=1000.0) is True
    assert fm.is_under_maintenance("gaming", now=1299.0) is True
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk["gaming"]["until"] == 1300.0
    assert on_disk["gaming"]["reason"] == "drill"


def test_is_under_maintenance_false_after_expiry(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    fm.set_maintenance("gaming", duration_s=300.0, now=1000.0)
    assert fm.is_under_maintenance("gaming", now=1300.0) is False
    assert fm.is_under_maintenance("gaming", now=5000.0) is False


def test_is_under_maintenance_false_for_unknown_marker(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    assert fm.is_under_maintenance("gaming") is False


def test_set_maintenance_clamps_duration(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    entry = fm.set_maintenance("gaming", duration_s=fm.MAX_MAINTENANCE_S * 10, now=1000.0)
    assert entry["until"] == 1000.0 + fm.MAX_MAINTENANCE_S


def test_set_maintenance_rejects_non_positive_duration(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        fm.set_maintenance("gaming", duration_s=0)
    with pytest.raises(ValueError):
        fm.set_maintenance("gaming", duration_s=-5)


def test_set_maintenance_rejects_unknown_host(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        fm.set_maintenance("ghost-host", duration_s=300.0)


def test_set_maintenance_prunes_expired_rows_on_write(monkeypatch, tmp_path):
    target = _isolate(monkeypatch, tmp_path, {
        "mac-mini-m4": {"until": 500.0, "reason": "stale"},
    })
    fm.set_maintenance("gaming", duration_s=300.0, now=1000.0)
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert "mac-mini-m4" not in on_disk  # expired at now=1000 (until=500) -> pruned
    assert "gaming" in on_disk


def test_clear_maintenance_is_idempotent(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    fm.set_maintenance("gaming", duration_s=300.0, now=1000.0)
    assert fm.clear_maintenance("gaming") is True
    assert fm.is_under_maintenance("gaming", now=1000.0) is False
    assert fm.clear_maintenance("gaming") is False  # already gone


def test_maintenance_status_drops_expired_and_adds_remaining_s(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path, {
        "gaming": {"until": 1300.0, "reason": "drill"},
        "mac-mini-m4": {"until": 500.0, "reason": "old"},
    })
    status = fm.maintenance_status(now=1000.0)
    assert set(status) == {"gaming"}
    assert status["gaming"]["remaining_s"] == 300.0


def test_save_busts_cache(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    assert fm.load_fleet_maintenance() == {}
    fm.set_maintenance("gaming", duration_s=300.0, now=1000.0)
    assert "gaming" in fm.load_fleet_maintenance()
