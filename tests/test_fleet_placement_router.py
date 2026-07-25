"""Unit tests for app_web/routers/fleet_placement.py (issue #353).

GET returns the placement + per-host status; PATCH merges a partial update,
persists it, and applies the delta; validation rejects an unknown host.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

from fastapi.testclient import TestClient  # noqa: E402

from app_web.routers import fleet_placement as fpr  # noqa: E402
from src import backend_process as bp  # noqa: E402
from src import fleet_placement as fp  # noqa: E402
from src import fleet_reconcile, remote_stats, services as svc  # noqa: E402
from src import server as server_mod  # noqa: E402


def _isolate(monkeypatch, tmp_path, initial=None):
    target = tmp_path / "fleet_placement.json"
    if initial is not None:
        target.write_text(json.dumps(initial), encoding="utf-8")
    monkeypatch.setattr(fp, "DEFAULT_PLACEMENT_PATH", target)
    monkeypatch.setattr(fp, "EXAMPLE_PLACEMENT_PATH", tmp_path / "none.json")
    fp._PLACEMENT_CACHE.clear()
    return target


def _stub_status(monkeypatch, reachable=True):
    """Keep GET off the network: local snapshot + a reachable Mac Mini. Peer
    liveness is the hub-independent TCP probe (remote_stats.is_reachable), not a
    hub /health call — the same signal the Machines tab uses."""
    monkeypatch.setattr(bp, "running_backends", lambda: {"piper": object()})

    async def is_reachable(host):
        return reachable

    async def remote_models(owner, **kw):
        return [{"id": "parakeet", "reachable": True}]

    monkeypatch.setattr(remote_stats, "is_reachable", is_reachable)
    monkeypatch.setattr(svc, "remote_models", remote_models)


def test_get_lists_every_fleet_host_with_manageability(monkeypatch, tmp_path):
    """Every configured fleet host gets a row. A managed-only satellite that
    runs no hub (openclaw — no launchable models) is shown with runs_hub=False
    and an empty eligible list (the UI renders the "not placeable here" note),
    never silently dropped — using the box's own TCP liveness for its
    online/offline state, not a hub probe it doesn't answer. gaming graduated
    to a placeable voice-pair host in #323, then gained the remaining two
    whisper backends in #370."""
    _isolate(monkeypatch, tmp_path, {})
    monkeypatch.setattr(bp, "running_backends", lambda: {})

    async def is_reachable(host):
        return host.id == "gaming"  # gaming powered on; other peers off

    async def remote_models(owner, **kw):
        return []

    monkeypatch.setattr(remote_stats, "is_reachable", is_reachable)
    monkeypatch.setattr(svc, "remote_models", remote_models)

    client = TestClient(server_mod.app)
    body = client.get("/admin/api/fleet-placement").json()
    hosts = {h["id"]: h for h in body["hosts"]}

    # Full inventory — nothing dropped.
    assert {"tower", "mac-mini-m4", "gaming", "openclaw"} <= set(hosts)
    # Managed-only satellite: reachable by TCP, but no hub / nothing to place.
    assert hosts["openclaw"]["runs_hub"] is False
    assert hosts["openclaw"]["eligible"] == []
    assert hosts["openclaw"]["reachable"] is False
    # gaming is a placeable voice-quartet host since #323/#370.
    assert hosts["gaming"]["runs_hub"] is True
    assert {e["id"] for e in hosts["gaming"]["eligible"]} == {
        "whisper", "orpheus", "whisper_translate", "whisper_vanilla",
    }
    assert hosts["gaming"]["reachable"] is True   # powered on (TCP liveness)
    # Manageable hosts still carry their launchable models.
    assert hosts["mac-mini-m4"]["runs_hub"] is True
    assert hosts["tower"]["local"] is True


def test_get_returns_placement_and_host_status(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path, {"tower": ["piper"], "mac-mini-m4": ["parakeet"]})
    _stub_status(monkeypatch)
    client = TestClient(server_mod.app)
    r = client.get("/admin/api/fleet-placement")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["placement"] == {"tower": ["piper"], "mac-mini-m4": ["parakeet"]}
    hosts = {h["id"]: h for h in body["hosts"]}
    assert hosts["tower"]["local"] is True
    assert hosts["tower"]["running"] == ["piper"]
    assert hosts["mac-mini-m4"]["reachable"] is True
    assert hosts["mac-mini-m4"]["placed"] == ["parakeet"]
    # eligible carries display names for the grid to render
    assert all("display_name" in e for e in hosts["mac-mini-m4"]["eligible"])
    # parakeet is 0-VRAM but runs on the Mac's ANE, not CPU — no device hint
    by_id = {e["id"]: e for e in hosts["mac-mini-m4"]["eligible"]}
    assert by_id["parakeet"]["device"] is None


def _stub_gaming_online(monkeypatch):
    """GET off the network with gaming powered on — the capacity tests drive the
    warning off gaming's *placed* set, so its liveness just needs to be True."""
    monkeypatch.setattr(bp, "running_backends", lambda: {})

    async def is_reachable(host):
        return True

    async def remote_models(owner, **kw):
        return []  # no live-running badges — placed set is what capacity sums

    monkeypatch.setattr(remote_stats, "is_reachable", is_reachable)
    monkeypatch.setattr(svc, "remote_models", remote_models)


def test_capacity_warning_when_over_ceiling(monkeypatch, tmp_path):
    """A host whose placed models' est_vram_mb sum exceeds its declared
    ``vram_mb`` ceiling carries capacity_warning=True (advisory, #375). gaming
    declares an 8192 MB ceiling; two stubbed 5000 MB models overcommit it."""
    _isolate(monkeypatch, tmp_path, {"gaming": ["whisper", "orpheus"]})
    _stub_gaming_online(monkeypatch)
    monkeypatch.setattr(fpr, "_vram_estimates", lambda: {"whisper": 5000, "orpheus": 5000})

    client = TestClient(server_mod.app)
    hosts = {h["id"]: h for h in client.get("/admin/api/fleet-placement").json()["hosts"]}
    g = hosts["gaming"]
    assert g["vram_mb"] == 8192
    assert g["est_vram_mb"] == 10000
    assert g["capacity_warning"] is True


def test_no_capacity_warning_when_under_ceiling(monkeypatch, tmp_path):
    """gaming's voice pair (whisper 2000 + orpheus 2200 with SNAC on CPU,
    #422 = 4200 MB from the committed config) sits under its 8192 MB ceiling
    — the real config must not raise a false positive."""
    _isolate(monkeypatch, tmp_path, {"gaming": ["whisper", "orpheus"]})
    _stub_gaming_online(monkeypatch)

    client = TestClient(server_mod.app)
    hosts = {h["id"]: h for h in client.get("/admin/api/fleet-placement").json()["hosts"]}
    g = hosts["gaming"]
    assert g["vram_mb"] == 8192
    assert g["est_vram_mb"] == 4200  # 2000 + 2200, from config/models.yaml
    assert g["capacity_warning"] is False


def test_no_capacity_warning_with_full_voice_quartet(monkeypatch, tmp_path):
    """gaming's worst-case voice set (whisper 2000 + orpheus 2200 as #422's
    failover tenant + whisper_translate 0 + whisper_vanilla 2000 = 6200 MB
    from the committed config) sits under its 8192 MB ceiling — the real
    config must not raise a false positive with all four placed together."""
    _isolate(
        monkeypatch, tmp_path,
        {"gaming": ["whisper", "orpheus", "whisper_translate", "whisper_vanilla"]},
    )
    _stub_gaming_online(monkeypatch)

    client = TestClient(server_mod.app)
    hosts = {h["id"]: h for h in client.get("/admin/api/fleet-placement").json()["hosts"]}
    g = hosts["gaming"]
    assert g["vram_mb"] == 8192
    assert g["est_vram_mb"] == 6200  # 2000 + 2200 + 0 + 2000, from config/models.yaml
    assert g["capacity_warning"] is False


def test_host_without_ceiling_never_warns(monkeypatch, tmp_path):
    """The Apple-silicon Mac Mini declares no ``vram_mb`` (unified memory), so
    it never warns even with a huge placed footprint — ceiling is None."""
    _isolate(monkeypatch, tmp_path, {"mac-mini-m4": ["parakeet"]})
    _stub_gaming_online(monkeypatch)
    monkeypatch.setattr(fpr, "_vram_estimates", lambda: {"parakeet": 99999})

    client = TestClient(server_mod.app)
    hosts = {h["id"]: h for h in client.get("/admin/api/fleet-placement").json()["hosts"]}
    m = hosts["mac-mini-m4"]
    assert m["vram_mb"] is None
    assert m["est_vram_mb"] == 99999
    assert m["capacity_warning"] is False


def test_eligible_entries_mark_cpu_models(monkeypatch, tmp_path):
    """CPU-resident models (piper, whisper_translate) carry
    ``device: "cpu"`` in their eligible entry (#387); a GPU-backed model
    (whisper) and the ANE-resident, also-0-VRAM parakeet do not."""
    _isolate(monkeypatch, tmp_path, {})
    _stub_gaming_online(monkeypatch)

    client = TestClient(server_mod.app)
    hosts = {h["id"]: h for h in client.get("/admin/api/fleet-placement").json()["hosts"]}

    gaming = {e["id"]: e for e in hosts["gaming"]["eligible"]}
    assert gaming["whisper_translate"]["device"] == "cpu"  # whisper-server -ng
    assert gaming["whisper"]["device"] is None
    assert gaming["orpheus"]["device"] is None

    tower = {e["id"]: e for e in hosts["tower"]["eligible"]}
    assert tower["piper"]["device"] == "cpu"  # tts_engine: piper hardcodes CPU


def test_cpu_chain_tier_marks_only_the_flagged_host(monkeypatch, tmp_path):
    """A failover chain's ``cpu: true`` tier is CPU on *that host only* (#405).

    Regression guard: the hint used to be a single ``{model_id: "cpu"}`` dict
    built from ``all_models()``, whose args carry the active host's CPU-offload
    rewrite baked in — so flagging tower as whisper's degraded last resort
    labelled whisper "cpu" on gaming and mac-mini-m4 too, i.e. the grid claimed
    the GPU-preferred members ran it on CPU.
    """
    _isolate(monkeypatch, tmp_path, {})
    _stub_gaming_online(monkeypatch)

    client = TestClient(server_mod.app)
    hosts = {h["id"]: h for h in client.get("/admin/api/fleet-placement").json()["hosts"]}

    def device(host_id: str, model_id: str):
        entry = {e["id"]: e for e in hosts[host_id]["eligible"]}.get(model_id)
        return entry["device"] if entry else None

    # whisper's production chain is [gaming, mac-mini-m4, {id: tower, cpu: true}]
    assert device("tower", "whisper") == "cpu"          # the flagged degraded tier
    assert device("gaming", "whisper") is None          # GPU-preferred member
    assert device("mac-mini-m4", "whisper") is None     # GPU-preferred member

    # An always-CPU row stays CPU everywhere it is eligible — the per-host
    # split must not regress the #387 static case.
    assert device("gaming", "whisper_translate") == "cpu"


def test_patch_merges_persists_and_applies(monkeypatch, tmp_path):
    target = _isolate(monkeypatch, tmp_path, {"tower": ["piper"], "mac-mini-m4": ["parakeet"]})
    applied_calls = []

    async def fake_apply(host_id, old_ids, new_ids, active_id):
        applied_calls.append((host_id, tuple(old_ids), tuple(new_ids)))
        return {"stopped": [], "converged": {}}

    monkeypatch.setattr(fleet_reconcile, "apply_placement_change", fake_apply)

    client = TestClient(server_mod.app)
    r = client.patch("/admin/api/fleet-placement", json={"mac-mini-m4": ["parakeet", "qwen"]})
    assert r.status_code == 200, r.text
    body = r.json()
    # tower untouched by the merge; mac-mini replaced
    assert body["placement"] == {"tower": ["piper"], "mac-mini-m4": ["parakeet", "qwen"]}
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk["mac-mini-m4"] == ["parakeet", "qwen"]
    # only the touched host had its delta applied, with the right old→new
    assert applied_calls == [("mac-mini-m4", ("parakeet",), ("parakeet", "qwen"))]


def test_patch_unknown_host_400(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path, {})
    monkeypatch.setattr(fleet_reconcile, "apply_placement_change", _never)
    client = TestClient(server_mod.app)
    r = client.patch("/admin/api/fleet-placement", json={"ghost-host": ["whisper"]})
    assert r.status_code == 400


def test_reconcile_endpoint_runs_a_pass(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path, {})

    async def fake_once():
        return {"mac-mini-m4": {"reachable": True}}

    monkeypatch.setattr(fleet_reconcile, "reconcile_once", fake_once)
    client = TestClient(server_mod.app)
    r = client.post("/admin/api/fleet-placement/reconcile")
    assert r.status_code == 200
    assert r.json()["results"]["mac-mini-m4"]["reachable"] is True


async def _never(*args, **kwargs):  # pragma: no cover — must not be called
    raise AssertionError("apply_placement_change should not run on a rejected PATCH")
