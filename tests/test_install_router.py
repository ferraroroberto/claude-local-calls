"""HTTP-level tests for app_web/routers/install.py's install endpoints
(issue #198): install_fix/install_fix_all now call
install.run_all_checks(use_cache=True) instead of forcing a second full
battery run just to locate one check by fix_id. (Router split out of
hub.py in issue #533 — endpoint paths are unchanged.)
"""

from __future__ import annotations

import os

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

from fastapi.testclient import TestClient

from src import install as install_mod
from src import server as server_mod


def _reset_cache(monkeypatch):
    monkeypatch.setattr(install_mod, "_cached_report", None)
    monkeypatch.setattr(install_mod, "_cached_at", 0.0)


def test_install_fix_requires_fix_id():
    client = TestClient(server_mod.app)
    r = client.post("/admin/api/install/fix", json={})
    assert r.status_code == 400


def test_install_fix_404_for_unknown_fix_id(monkeypatch):
    _reset_cache(monkeypatch)
    client = TestClient(server_mod.app)
    r = client.post("/admin/api/install/fix", json={"fix_id": "not-a-real-fix-id"})
    assert r.status_code == 404


def test_install_fix_reuses_cache_populated_by_status(monkeypatch):
    """A status call followed by a fix call must not re-run the battery a
    second time — the fix call should hit the cache install_status warmed."""
    _reset_cache(monkeypatch)
    calls = {"n": 0}

    def _stub():
        calls["n"] += 1
        from src.install import Check
        return Check(
            "deps", "stub", "missing",
            fix_id="deps", fix_label="pip install",
        )

    monkeypatch.setattr(install_mod, "_check_deps", _stub)
    # Neutralize the fix function so the test doesn't actually pip install.
    ran = {"called": False}
    monkeypatch.setattr(install_mod, "_fix_deps", lambda: ran.__setitem__("called", True))

    client = TestClient(server_mod.app)
    r1 = client.get("/admin/api/install/status")
    assert r1.status_code == 200
    assert calls["n"] == 1

    r2 = client.post("/admin/api/install/fix", json={"fix_id": "deps"})
    assert r2.status_code == 200, r2.text
    assert r2.json()["ok"] is True
    assert calls["n"] == 1  # reused install_status's cached report
    assert ran["called"] is True
