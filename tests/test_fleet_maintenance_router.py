"""Unit tests for app_web/routers/fleet_maintenance.py (issue #411).

GET reflects active drains; POST arms (clamping/validating); DELETE clears
and triggers one immediate reconcile pass.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

from fastapi.testclient import TestClient  # noqa: E402

from src import fleet_maintenance as fm  # noqa: E402
from src import fleet_reconcile  # noqa: E402
from src import server as server_mod  # noqa: E402


def _isolate(monkeypatch, tmp_path, initial=None):
    target = tmp_path / "fleet_maintenance.json"
    if initial is not None:
        target.write_text(json.dumps(initial), encoding="utf-8")
    monkeypatch.setattr(fm, "DEFAULT_MAINTENANCE_PATH", target)
    fm._MAINTENANCE_CACHE.clear()
    return target


def test_get_reflects_armed_host(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path, {"gaming": {"until": 9999999999.0, "reason": "drill"}})
    client = TestClient(server_mod.app)
    r = client.get("/admin/api/fleet-maintenance")
    assert r.status_code == 200, r.text
    assert "gaming" in r.json()["maintenance"]


def test_post_arms_with_defaults(monkeypatch, tmp_path):
    target = _isolate(monkeypatch, tmp_path)
    client = TestClient(server_mod.app)
    r = client.post("/admin/api/fleet-maintenance/gaming")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["host_id"] == "gaming"
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert "gaming" in on_disk


def test_post_clamps_duration(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    client = TestClient(server_mod.app)
    r = client.post("/admin/api/fleet-maintenance/gaming", json={"duration_s": fm.MAX_MAINTENANCE_S * 5})
    assert r.status_code == 200, r.text
    assert r.json()["remaining_s"] == fm.MAX_MAINTENANCE_S


def test_post_unknown_host_400(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    client = TestClient(server_mod.app)
    r = client.post("/admin/api/fleet-maintenance/ghost-host")
    assert r.status_code == 400


def test_post_bad_duration_400(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    client = TestClient(server_mod.app)
    r = client.post("/admin/api/fleet-maintenance/gaming", json={"duration_s": "not-a-number"})
    assert r.status_code == 400
    r = client.post("/admin/api/fleet-maintenance/gaming", json={"duration_s": -5})
    assert r.status_code == 400


def test_delete_clears_and_triggers_reconcile(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path, {"gaming": {"until": 9999999999.0, "reason": "drill"}})

    reconcile_calls = {"n": 0}

    async def fake_once():
        reconcile_calls["n"] += 1
        return {"gaming": {"reachable": True}}

    monkeypatch.setattr(fleet_reconcile, "reconcile_once", fake_once)
    client = TestClient(server_mod.app)
    r = client.delete("/admin/api/fleet-maintenance/gaming")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["cleared"] is True
    assert body["reconcile"] == {"gaming": {"reachable": True}}
    assert reconcile_calls["n"] == 1
    assert fm.is_under_maintenance("gaming") is False


def test_delete_when_absent_is_idempotent(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)

    async def fake_once():
        return {}

    monkeypatch.setattr(fleet_reconcile, "reconcile_once", fake_once)
    client = TestClient(server_mod.app)
    r = client.delete("/admin/api/fleet-maintenance/gaming")
    assert r.status_code == 200, r.text
    assert r.json()["cleared"] is False
