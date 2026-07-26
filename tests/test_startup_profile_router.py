"""Unit tests for app_web/routers/startup_profile.py (issues #265, #430).

GET returns the current profile + service items; PATCH merges a partial
payload over the current profile, validates, and persists. Services only
since #430 — the model autostart set derives from config/models.yaml.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

from fastapi.testclient import TestClient  # noqa: E402

from src import server as server_mod  # noqa: E402
from src import startup_profile as sp  # noqa: E402


def _isolate_profile(monkeypatch, tmp_path, initial=None):
    target = tmp_path / "startup_profile.json"
    if initial is not None:
        target.write_text(json.dumps(initial), encoding="utf-8")
    monkeypatch.setattr(sp, "DEFAULT_PROFILE_PATH", target)
    return target


def test_get_returns_profile_and_service_items(monkeypatch, tmp_path):
    _isolate_profile(monkeypatch, tmp_path, {"docker": True, "langfuse": False})
    client = TestClient(server_mod.app)
    r = client.get("/admin/api/startup-profile")
    assert r.status_code == 200
    body = r.json()
    assert body["profile"]["docker"] is True
    assert body["profile"]["langfuse"] is False
    # #430: no models key on the profile, no eligible-models list in the body.
    assert "models" not in body["profile"]
    assert "models" not in body
    service_ids = {s["id"] for s in body["services"]}
    # mac_mini_sync retired in #374 — peer sync is reconcile-driven now.
    assert service_ids == {"docker", "langfuse", "agentsview"}


def test_patch_merges_partial_payload(monkeypatch, tmp_path):
    target = _isolate_profile(monkeypatch, tmp_path, {"docker": True, "langfuse": True})
    client = TestClient(server_mod.app)
    r = client.patch("/admin/api/startup-profile", json={"docker": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["profile"]["docker"] is False
    # Untouched fields survive the merge.
    assert body["profile"]["langfuse"] is True

    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk["docker"] is False


def test_patch_with_legacy_models_key_is_ignored(monkeypatch, tmp_path):
    """#430 migration: an old caller PATCHing the retired ``models`` list gets
    a clean 200 and the key is silently dropped — never persisted, never 400."""
    target = _isolate_profile(monkeypatch, tmp_path, {"docker": True, "langfuse": True})
    client = TestClient(server_mod.app)
    r = client.patch("/admin/api/startup-profile", json={"models": ["piper"]})
    assert r.status_code == 200, r.text
    assert "models" not in r.json()["profile"]
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert "models" not in on_disk


# --------------------------------------------------------------------------- #
# ?host= — host-addressed profile (remote forwarding, #352)
# --------------------------------------------------------------------------- #
import app_web.routers.startup_profile as spr  # noqa: E402


def _fake_remote(monkeypatch, base="http://10.0.0.9:8000", known=True):
    """Make host 'mac' resolve to a reachable remote peer; capture the forward."""
    captured: dict = {}

    async def fake_forward(base_url, path, *, method, headers=None, unreachable_detail="", **kw):
        captured.update(base=base_url, path=path, method=method, headers=headers, kw=kw)
        return {"ok": True, "profile": {"docker": False, "langfuse": True, "agentsview": True},
                "services": []}

    monkeypatch.setattr(spr, "get_host", lambda h: object() if known else None)
    monkeypatch.setattr(spr, "remote_base_url_for_host", lambda h: base if known else None)
    monkeypatch.setattr(spr, "forward_admin_request", fake_forward)
    return captured


def test_get_host_self_is_served_locally(monkeypatch, tmp_path):
    _isolate_profile(monkeypatch, tmp_path, {"docker": True, "langfuse": False})
    # No forward mock — if it tried to forward, it would fail.
    client = TestClient(server_mod.app)
    r = client.get("/admin/api/startup-profile?host=tower")
    assert r.status_code == 200
    assert r.json()["profile"]["langfuse"] is False


def test_get_forwards_to_remote_host(monkeypatch, tmp_path):
    _isolate_profile(monkeypatch, tmp_path, {"docker": True})
    captured = _fake_remote(monkeypatch)
    client = TestClient(server_mod.app)
    r = client.get("/admin/api/startup-profile?host=mac")
    assert r.status_code == 200
    assert r.json()["profile"]["docker"] is False  # the peer's profile
    assert captured["base"] == "http://10.0.0.9:8000"
    assert captured["path"] == "/admin/api/startup-profile"
    assert captured["method"] == "GET"


def test_patch_forwards_body_to_remote_host(monkeypatch, tmp_path):
    _isolate_profile(monkeypatch, tmp_path, {"docker": True})
    captured = _fake_remote(monkeypatch)
    client = TestClient(server_mod.app)
    r = client.patch("/admin/api/startup-profile?host=mac", json={"docker": False})
    assert r.status_code == 200
    assert captured["method"] == "PATCH"
    assert captured["kw"].get("json") == {"docker": False}


def test_unknown_host_404(monkeypatch, tmp_path):
    _isolate_profile(monkeypatch, tmp_path, {"docker": True})
    monkeypatch.setattr(spr, "get_host", lambda h: None)  # unknown id
    client = TestClient(server_mod.app)
    r = client.get("/admin/api/startup-profile?host=ghost")
    assert r.status_code == 404


def test_known_host_without_address_400(monkeypatch, tmp_path):
    _isolate_profile(monkeypatch, tmp_path, {"docker": True})
    monkeypatch.setattr(spr, "get_host", lambda h: object())   # known
    monkeypatch.setattr(spr, "remote_base_url_for_host", lambda h: None)  # but no address
    client = TestClient(server_mod.app)
    r = client.get("/admin/api/startup-profile?host=openclaw")
    assert r.status_code == 400


def test_remote_base_url_for_host_empty_and_self_are_none():
    from src.remote_proxy import remote_base_url_for_host
    assert remote_base_url_for_host("") is None
    assert remote_base_url_for_host(None) is None
    assert remote_base_url_for_host("tower") is None  # the active host
