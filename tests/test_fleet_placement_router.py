"""Unit tests for app_web/routers/fleet_placement.py (issues #353, #430).

GET returns the registry-derived placement + per-host status (read-only since
#430 — the PATCH surface and config/fleet_placement.json behind it were
retired; desired state comes from models.yaml's hosts chains + startup
policy). POST /reconcile still triggers a convergence pass on demand.
"""

from __future__ import annotations

import os

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

from fastapi.testclient import TestClient  # noqa: E402

from app_web.routers import fleet_placement as fpr  # noqa: E402
from src import backend_process as bp  # noqa: E402
from src import fleet_reconcile, remote_stats, services as svc, system_stats  # noqa: E402
from src import server as server_mod  # noqa: E402


def _stub_collect(monkeypatch, stats=None):
    """Keep the live probes off the network/subprocess: remote_stats.collect
    would SSH a reachable peer for its stats snapshot (#434), and the local
    GPU snapshot (#436) would spawn a real nvidia-smi. Tests that assert the
    live-GPU path override gpu_stats after calling this."""
    async def collect(host):
        return stats

    monkeypatch.setattr(remote_stats, "collect", collect)
    monkeypatch.setattr(system_stats, "gpu_stats", lambda: [])


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
    _stub_collect(monkeypatch)


def test_get_lists_every_fleet_host_with_manageability(monkeypatch):
    """Every configured fleet host gets a row. A managed-only satellite that
    runs no hub (openclaw — no launchable models) is shown with runs_hub=False
    and an empty eligible list (the UI renders an honest note), never silently
    dropped — using the box's own TCP liveness for its online/offline state,
    not a hub probe it doesn't answer."""
    monkeypatch.setattr(bp, "running_backends", lambda: {})

    async def is_reachable(host):
        return host.id == "gaming"  # gaming powered on; other peers off

    async def remote_models(owner, **kw):
        return []

    monkeypatch.setattr(remote_stats, "is_reachable", is_reachable)
    monkeypatch.setattr(svc, "remote_models", remote_models)
    _stub_collect(monkeypatch)

    client = TestClient(server_mod.app)
    body = client.get("/admin/api/fleet-placement").json()
    hosts = {h["id"]: h for h in body["hosts"]}

    # Full inventory — nothing dropped.
    assert {"tower", "mac-mini-m4", "gaming", "openclaw"} <= set(hosts)
    # Managed-only satellite: reachable by TCP, but no hub / nothing to place.
    assert hosts["openclaw"]["runs_hub"] is False
    assert hosts["openclaw"]["eligible"] == []
    assert hosts["openclaw"]["reachable"] is False
    # gaming is a hub-running satellite since #323/#370.
    assert hosts["gaming"]["runs_hub"] is True
    assert {e["id"] for e in hosts["gaming"]["eligible"]} == {
        "whisper", "orpheus", "whisper_translate", "whisper_vanilla",
    }
    assert hosts["gaming"]["reachable"] is True   # powered on (TCP liveness)
    # Manageable hosts still carry their launchable models.
    assert hosts["mac-mini-m4"]["runs_hub"] is True
    assert hosts["tower"]["local"] is True


def test_get_returns_registry_derived_placement(monkeypatch):
    """The placement map is derived from the committed config/models.yaml
    (#430): eager rows on their preferred chain host; on_demand rows
    (gemma4_26b, gemma4_e4b, chatterbox, kokoro) never appear."""
    _stub_status(monkeypatch)
    client = TestClient(server_mod.app)
    r = client.get("/admin/api/fleet-placement")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["placement"] == {
        "tower": ["qwen35_4b", "piper", "orpheus"],
        "mac-mini-m4": ["qwen", "parakeet"],
        "gaming": ["whisper", "whisper_translate", "whisper_vanilla"],
    }
    hosts = {h["id"]: h for h in body["hosts"]}
    assert hosts["tower"]["local"] is True
    assert hosts["tower"]["running"] == ["piper"]
    assert hosts["tower"]["placed"] == ["qwen35_4b", "piper", "orpheus"]
    assert hosts["mac-mini-m4"]["reachable"] is True
    assert hosts["mac-mini-m4"]["placed"] == ["qwen", "parakeet"]
    # eligible carries display names for the grid to render
    assert all("display_name" in e for e in hosts["mac-mini-m4"]["eligible"])
    # parakeet is 0-VRAM but runs on the Mac's ANE, not CPU — no device hint
    by_id = {e["id"]: e for e in hosts["mac-mini-m4"]["eligible"]}
    assert by_id["parakeet"]["device"] is None


def test_patch_is_gone(monkeypatch):
    """#430: desired state is registry-derived — the old editable PATCH
    surface must not exist any more (405 Method Not Allowed)."""
    _stub_status(monkeypatch)
    client = TestClient(server_mod.app)
    r = client.patch("/admin/api/fleet-placement", json={"tower": ["piper"]})
    assert r.status_code == 405


def _stub_gaming_online(monkeypatch):
    """GET off the network with every peer powered on — the capacity tests
    drive the warning off the derived desired set, so liveness just needs to
    be True."""
    monkeypatch.setattr(bp, "running_backends", lambda: {})

    async def is_reachable(host):
        return True

    async def remote_models(owner, **kw):
        return []  # no live-running badges — the desired set is what capacity sums

    monkeypatch.setattr(remote_stats, "is_reachable", is_reachable)
    monkeypatch.setattr(svc, "remote_models", remote_models)
    _stub_collect(monkeypatch)


def test_capacity_warning_when_over_ceiling(monkeypatch):
    """A host whose desired models' est_vram_mb sum exceeds its declared
    ``vram_mb`` ceiling carries capacity_warning=True (advisory, #375). gaming
    declares an 8192 MB ceiling; two stubbed 5000 MB estimates overcommit it."""
    _stub_gaming_online(monkeypatch)
    monkeypatch.setattr(
        fpr, "_vram_estimates", lambda: {"whisper": 5000, "whisper_vanilla": 5000}
    )

    client = TestClient(server_mod.app)
    hosts = {h["id"]: h for h in client.get("/admin/api/fleet-placement").json()["hosts"]}
    g = hosts["gaming"]
    assert g["vram_mb"] == 8192
    assert g["est_vram_mb"] == 10000
    assert g["capacity_warning"] is True


def test_no_capacity_warning_from_committed_config(monkeypatch):
    """gaming's derived desired set (the whisper trio: 2000 + 0 + 2000 =
    4000 MB from the committed config) sits under its 8192 MB ceiling — the
    real config must not raise a false positive."""
    _stub_gaming_online(monkeypatch)

    client = TestClient(server_mod.app)
    hosts = {h["id"]: h for h in client.get("/admin/api/fleet-placement").json()["hosts"]}
    g = hosts["gaming"]
    assert g["vram_mb"] == 8192
    assert g["est_vram_mb"] == 4000  # 2000 + 0 + 2000, from config/models.yaml
    assert g["capacity_warning"] is False


def test_host_without_ceiling_never_warns(monkeypatch):
    """The Apple-silicon Mac Mini declares no ``vram_mb`` (unified memory), so
    it never warns even with a huge desired footprint — ceiling is None."""
    _stub_gaming_online(monkeypatch)
    monkeypatch.setattr(fpr, "_vram_estimates", lambda: {"parakeet": 99999})

    client = TestClient(server_mod.app)
    hosts = {h["id"]: h for h in client.get("/admin/api/fleet-placement").json()["hosts"]}
    m = hosts["mac-mini-m4"]
    assert m["vram_mb"] is None
    assert m["est_vram_mb"] == 99999
    assert m["capacity_warning"] is False


def test_eligible_entries_mark_cpu_models(monkeypatch):
    """CPU-resident models (piper, whisper_translate) carry
    ``device: "cpu"`` in their eligible entry (#387); a GPU-backed model
    (whisper) and the ANE-resident, also-0-VRAM parakeet do not."""
    _stub_gaming_online(monkeypatch)

    client = TestClient(server_mod.app)
    hosts = {h["id"]: h for h in client.get("/admin/api/fleet-placement").json()["hosts"]}

    gaming = {e["id"]: e for e in hosts["gaming"]["eligible"]}
    assert gaming["whisper_translate"]["device"] == "cpu"  # whisper-server -ng
    assert gaming["whisper"]["device"] is None
    assert gaming["orpheus"]["device"] is None

    tower = {e["id"]: e for e in hosts["tower"]["eligible"]}
    assert tower["piper"]["device"] == "cpu"  # tts_engine: piper hardcodes CPU


def test_cpu_chain_tier_marks_only_the_flagged_host(monkeypatch):
    """A failover chain's ``cpu: true`` tier is CPU on *that host only* (#405).

    Regression guard: the hint used to be a single ``{model_id: "cpu"}`` dict
    built from ``all_models()``, whose args carry the active host's CPU-offload
    rewrite baked in — so flagging tower as whisper's degraded last resort
    labelled whisper "cpu" on gaming and mac-mini-m4 too, i.e. the grid claimed
    the GPU-preferred members ran it on CPU.
    """
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


def test_capacity_excludes_cpu_resident_rows(monkeypatch):
    """CPU-resident rows must not count against the GPU ceiling (#431): piper
    (always-CPU engine) and whisper (tower is its degraded ``cpu: true`` chain
    tier) contribute nothing to the tower's sum even while running — the fix
    for the false "Over VRAM capacity ~19 GB / 16 GB" on the tower. The same
    rows still count on hosts where they are GPU-resident (whisper on gaming)."""
    monkeypatch.setattr(
        bp, "running_backends", lambda: {"piper": object(), "whisper": object()}
    )

    async def is_reachable(host):
        return True

    async def remote_models(owner, **kw):
        return []

    monkeypatch.setattr(remote_stats, "is_reachable", is_reachable)
    monkeypatch.setattr(svc, "remote_models", remote_models)
    _stub_collect(monkeypatch)
    monkeypatch.setattr(
        fpr, "_vram_estimates",
        lambda: {"piper": 3000, "whisper": 2000, "qwen35_4b": 2100, "orpheus": 2200},
    )

    client = TestClient(server_mod.app)
    hosts = {h["id"]: h for h in client.get("/admin/api/fleet-placement").json()["hosts"]}

    t = hosts["tower"]
    # Considered set: placed [qwen35_4b, piper, orpheus] ∪ running [piper,
    # whisper] — piper + whisper are CPU on the tower, so only the GPU rows sum.
    assert t["est_vram_mb"] == 2100 + 2200
    assert t["capacity_warning"] is False
    # CPU residency is per host: whisper still counts on GPU-preferred gaming.
    assert hosts["gaming"]["est_vram_mb"] == 2000


def test_ram_mb_surfaces_where_documented(monkeypatch):
    """``ram_mb`` (display-only, #431) rides each host row where machines.md
    documents the fact — tower 128 GB, mac 16 GB unified (#434), gaming
    16 GB — and is None elsewhere."""
    _stub_status(monkeypatch)
    client = TestClient(server_mod.app)
    hosts = {h["id"]: h for h in client.get("/admin/api/fleet-placement").json()["hosts"]}
    assert hosts["tower"]["ram_mb"] == 131072
    assert hosts["gaming"]["ram_mb"] == 16384
    assert hosts["mac-mini-m4"]["ram_mb"] == 16384
    assert hosts["openclaw"]["ram_mb"] is None


def test_live_ram_block_rides_reachable_ssh_peers(monkeypatch):
    """#434: the capacity line reads RAM used/total live where available —
    the local host snapshots itself; a reachable SSH peer carries the
    ``ram`` block from the (cached) Machines-tab stats probe; a peer whose
    probe fails (or a host that is off) reads None and the UI falls back to
    the declared total."""
    _stub_status(monkeypatch)
    _stub_collect(
        monkeypatch,
        {"ram": {"used_gb": 5.2, "total_gb": 15.6, "percent": 33.3}},
    )

    client = TestClient(server_mod.app)
    hosts = {h["id"]: h for h in client.get("/admin/api/fleet-placement").json()["hosts"]}

    # Local host: psutil snapshot — live values, assert shape not numbers.
    assert set(hosts["tower"]["ram"]) == {"used_gb", "total_gb", "percent"}
    # Reachable SSH peer: the collected block, verbatim.
    assert hosts["mac-mini-m4"]["ram"] == {
        "used_gb": 5.2, "total_gb": 15.6, "percent": 33.3,
    }


def test_live_ram_none_when_probe_fails(monkeypatch):
    """A reachable peer whose SSH stats probe fails degrades to ram=None —
    never an error, never a fabricated figure."""
    _stub_status(monkeypatch)  # _stub_collect(None) baked in

    client = TestClient(server_mod.app)
    hosts = {h["id"]: h for h in client.get("/admin/api/fleet-placement").json()["hosts"]}
    assert hosts["mac-mini-m4"]["ram"] is None
    assert hosts["openclaw"]["ram"] is None


def test_live_gpu_block_rides_local_and_ssh_peers(monkeypatch):
    """#436: the capacity line reads GPU used/total live from the same probe
    plumbing the Machines tab uses — local nvidia-smi snapshot on the hub
    host, the cached SSH stats probe on reachable peers — so the two tabs
    can never disagree while both are live. First GPU only."""
    _stub_status(monkeypatch)
    _stub_collect(monkeypatch, {
        "ram": {"used_gb": 5.2, "total_gb": 15.6, "percent": 33.3},
        "gpus": [{"name": "GTX 1070", "used_mb": 1900.0, "total_mb": 8192.0,
                  "vram_percent": 23.2, "util_percent": 2.0}],
    })
    # After _stub_collect (which parks gpu_stats at []) — the local live path.
    monkeypatch.setattr(
        system_stats, "gpu_stats",
        lambda: [{"name": "RTX 5080", "used_mb": 3100.0, "total_mb": 16384.0,
                  "vram_percent": 18.9, "util_percent": 5.0}],
    )

    client = TestClient(server_mod.app)
    hosts = {h["id"]: h for h in client.get("/admin/api/fleet-placement").json()["hosts"]}
    # Local host: the nvidia-smi snapshot, reduced to {used_mb, total_mb}.
    assert hosts["tower"]["gpu"] == {"used_mb": 3100.0, "total_mb": 16384.0}
    # Reachable SSH peer: the collected block's first GPU, same reduction.
    assert hosts["mac-mini-m4"]["gpu"] == {"used_mb": 1900.0, "total_mb": 8192.0}


def test_live_gpu_none_without_probe(monkeypatch):
    """A host with no live GPU metric — no nvidia-smi (the Mac's unified
    memory), a failed SSH probe, or a powered-off box — degrades to
    gpu=None so the UI falls back to the ~estimate, never a fabricated
    figure (#436)."""
    _stub_status(monkeypatch)  # _stub_collect(None) baked in
    monkeypatch.setattr(system_stats, "gpu_stats", lambda: [])

    client = TestClient(server_mod.app)
    hosts = {h["id"]: h for h in client.get("/admin/api/fleet-placement").json()["hosts"]}
    assert hosts["tower"]["gpu"] is None
    assert hosts["mac-mini-m4"]["gpu"] is None
    assert hosts["openclaw"]["gpu"] is None


def test_foreign_adopted_backend_flagged_external(monkeypatch):
    """A live local backend whose adopted PID is a foreign process (an
    external sibling on a mutex-shared port — voice-transcriber's
    whisper-server on :8090) is listed in ``external`` so the summary labels
    it distinctly instead of claiming the hub runs it (#431)."""
    _stub_status(monkeypatch)
    monkeypatch.setattr(bp, "running_backends", lambda: {"whisper": object()})
    monkeypatch.setattr(bp, "inherited_foreign", lambda mid: mid == "whisper")

    client = TestClient(server_mod.app)
    hosts = {h["id"]: h for h in client.get("/admin/api/fleet-placement").json()["hosts"]}
    assert hosts["tower"]["running"] == ["whisper"]
    assert hosts["tower"]["external"] == ["whisper"]
    # Peers never carry the flag — their liveness is a reachability probe.
    assert hosts["gaming"]["external"] == []


def test_reconcile_endpoint_runs_a_pass(monkeypatch):
    async def fake_once():
        return {"mac-mini-m4": {"reachable": True}}

    monkeypatch.setattr(fleet_reconcile, "reconcile_once", fake_once)
    client = TestClient(server_mod.app)
    r = client.post("/admin/api/fleet-placement/reconcile")
    assert r.status_code == 200
    assert r.json()["results"]["mac-mini-m4"]["reachable"] is True
