"""Diagnostics API + report layer (#315).

Covers the router contract (status/start/stop/runs/summary/report/export,
settings validation, the single-capture guard) and the report layer's
summary/drift/markdown output. The sampler is exercised through a real
one-shot so the whole path — collect, persist, evaluate, render — is proven
end to end without waiting on a timed run.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from src import backend_process, model_registry, server as server_mod, services
from src.diagnostics import report, sampler, settings as diag_settings, store


def _stub_real_backend_ops(monkeypatch):
    """Keep the real ``server_mod.app`` lifespan hermetic (issue #416 follow-up).

    Entering ``TestClient(server_mod.app)`` runs the *real* ASGI lifespan —
    ``wire_observatory_loop`` inherits/autostarts this machine's actually-configured
    backends (registry-derived, ``config/models.yaml``) and its
    shutdown counterpart really stops them. This suite doesn't exercise any of that
    (it's purely the diagnostics API), so without these stubs every test in this
    file was starting/killing this dev box's real ``qwen``/``piper`` backend
    processes on every ``with TestClient(...)`` enter/exit.
    """
    monkeypatch.setattr(model_registry, "desired_model_ids", lambda *a, **kw: [])
    monkeypatch.setattr(backend_process, "inherit_running_backends", lambda: 0)
    monkeypatch.setattr(backend_process, "start", lambda model_id: (True, "stubbed (test)"))
    monkeypatch.setattr(backend_process, "stop", lambda model_id: (True, "stubbed (test)"))

    async def _no_launch():
        return {"ok": True, "steps": []}

    monkeypatch.setattr(services, "launch_stack", _no_launch)
    monkeypatch.setattr(services, "launch_agentsview", _no_launch)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Isolated DB + settings per test so nothing touches real runtime data."""
    _stub_real_backend_ops(monkeypatch)
    store.set_db_path(tmp_path / "diag.db")
    diag_settings.set_settings_path(tmp_path / "diagnostics_settings.json")
    with TestClient(server_mod.app) as c:
        yield c
    store.set_db_path(None)
    diag_settings.set_settings_path(None)


def _sample(ts, *, ram=40.0, cpu=10.0, process_count=2):
    """One synthetic tick. ``ram`` is the *percent* — the helper builds the
    full dict so a caller can't accidentally replace it with a scalar."""
    return store.SystemSample(
        ts=ts, cpu_percent=cpu, per_core=[cpu], load_avg=None,
        ram={"used_gb": 4.0, "total_gb": 16.0, "percent": ram},
        swap={"used_gb": 0.0, "total_gb": 8.0, "percent": 0.0},
        disk={"used_gb": 10.0, "total_gb": 500.0, "percent": 20.0},
        disk_io={}, net_io={}, gpus=[], process_count=process_count,
    )


def _proc(pid, name, app_id, rss_mb=10):
    return {"pid": pid, "ppid": 1, "name": name, "cmdline": f"/bin/{name}",
            "app_id": app_id, "cpu_percent": 1.0, "rss_bytes": int(rss_mb * 1024 ** 2),
            "num_threads": 2, "status": "running", "create_time": 1.0}


def _seed_run(*, machine_id="pc-test", apps=(("app-launcher", 3),), ports=(), ram=40.0):
    """Create a finished run with one tick of synthetic content."""
    run_id = store.create_run(machine_id=machine_id, os_name="TestOS",
                              interval_s=15.0, duration_s=60.0)
    procs, pid = [], 1
    for app_id, count in apps:
        for _ in range(count):
            procs.append(_proc(pid, "python.exe", app_id))
            pid += 1
    port_rows = [{"port": p, "proto": "tcp", "pid": 1, "name": "python.exe",
                  "app_id": a} for p, a in ports]
    store.write_sample(run_id, _sample(time.time(), ram=ram, process_count=len(procs)),
                       procs, port_rows)
    store.finish_run(run_id)
    return run_id


# ------------------------------------------------------------------ status


def test_status_is_idle_with_no_capture(client):
    body = client.get("/admin/api/diagnostics/status").json()
    assert body["capturing"] is False
    assert body["active"] is None
    assert body["settings"]["scheduled_enabled"] is False   # off by default
    assert body["limits"]["max_duration_s"] == sampler.MAX_DURATION_S


def test_runs_list_is_empty_initially(client):
    assert client.get("/admin/api/diagnostics/runs").json()["runs"] == []


# ------------------------------------------------------------- capture flow


def test_start_stop_capture_round_trip(client):
    started = client.post("/admin/api/diagnostics/start",
                          json={"duration_s": 600, "interval_s": 5})
    assert started.status_code == 200
    run_id = started.json()["active"]["run_id"]

    status = client.get("/admin/api/diagnostics/status").json()
    assert status["capturing"] is True
    assert status["active"]["run_id"] == run_id

    assert client.post("/admin/api/diagnostics/stop", json={}).json()["stopped"] is True
    assert client.get("/admin/api/diagnostics/status").json()["capturing"] is False
    assert store.get_run(run_id)["status"] == "stopped"


def test_second_capture_is_refused_while_one_runs(client):
    """Two concurrent captures would double the observer effect and interleave
    in the store."""
    client.post("/admin/api/diagnostics/start", json={"duration_s": 600})
    conflict = client.post("/admin/api/diagnostics/start", json={"duration_s": 600})
    assert conflict.status_code == 409
    client.post("/admin/api/diagnostics/stop", json={})


def test_start_rejects_non_numeric_input(client):
    res = client.post("/admin/api/diagnostics/start", json={"duration_s": "soon"})
    assert res.status_code == 400


def test_interval_and_duration_are_clamped(client):
    """A pathological interval must not turn the sampler into the load it is
    supposed to measure."""
    body = client.post("/admin/api/diagnostics/start",
                       json={"interval_s": 0.001, "duration_s": 10 ** 9}).json()
    assert body["active"]["interval_s"] == sampler.MIN_INTERVAL_S
    assert body["active"]["duration_s"] == sampler.MAX_DURATION_S
    client.post("/admin/api/diagnostics/stop", json={})


def test_one_shot_snapshot_captures_and_evaluates(client):
    """The full real path: scan this machine, persist, verdict, summarize."""
    body = client.post("/admin/api/diagnostics/snapshot", json={}).json()
    run_id = body["run_id"]

    summary = client.get(f"/admin/api/diagnostics/runs/{run_id}").json()
    assert summary["run"]["sample_count"] == 1
    assert summary["run"]["trigger"] == "one-shot"
    assert summary["apps"], "a real machine must attribute at least one app"
    assert summary["verdict"]["level"] in {"healthy", "warning", "critical"}
    # CPU must be a real reading, not the 0.0 a cold psutil call returns.
    assert summary["resources"]["cpu"]["avg"] is not None


def test_stop_when_idle_is_safe(client):
    assert client.post("/admin/api/diagnostics/stop", json={}).json()["stopped"] is False


def test_stop_leaves_no_write_in_flight(client):
    """Stopping must be *graceful*, not a mid-tick cancel.

    The sampler does its work through ``asyncio.to_thread``, and cancelling
    the await does not stop the worker thread — so a cancel-based stop could
    land a sample in SQLite after the caller believed the run was over, and
    skip the finalize because the awaits in the ``finally`` re-raise
    ``CancelledError``. That surfaced as intermittent 'database is locked'
    failures in unrelated tests. Once stop returns, the run must be closed and
    its row count must be final."""
    client.post("/admin/api/diagnostics/start", json={"duration_s": 600, "interval_s": 5})
    run_id = client.get("/admin/api/diagnostics/status").json()["active"]["run_id"]
    time.sleep(2.5)  # let at least one tick land

    assert client.post("/admin/api/diagnostics/stop", json={}).json()["stopped"] is True

    run = store.get_run(run_id)
    assert run["status"] == "stopped", "a stopped run must not be left 'running'"
    assert run["ended_at"] is not None
    assert run["verdict_level"] is not None, "the finalize (verdict) was skipped"

    settled = len(store.samples(run_id))
    time.sleep(1.5)
    assert len(store.samples(run_id)) == settled, "a write landed after stop returned"


def test_stop_signal_does_not_abort_the_next_run(client):
    """A stale stop signal must not kill the following capture on tick one."""
    client.post("/admin/api/diagnostics/start", json={"duration_s": 600, "interval_s": 5})
    client.post("/admin/api/diagnostics/stop", json={})

    client.post("/admin/api/diagnostics/start", json={"duration_s": 600, "interval_s": 5})
    status = client.get("/admin/api/diagnostics/status").json()
    assert status["capturing"] is True, "the new run was aborted by a stale stop signal"
    client.post("/admin/api/diagnostics/stop", json={})


# ----------------------------------------------------------------- run reads


def test_summary_of_unknown_run_is_404(client):
    assert client.get("/admin/api/diagnostics/runs/nope").status_code == 404
    assert client.get("/admin/api/diagnostics/runs/nope/drift").status_code == 404
    assert client.get("/admin/api/diagnostics/runs/nope/report").status_code == 404
    assert client.get("/admin/api/diagnostics/runs/nope/export").status_code == 404


def test_summary_reports_apps_not_raw_process_names(client):
    run_id = _seed_run(apps=(("app-launcher", 3), ("llama.cpp", 1)))
    apps = {a["app_id"]: a for a in client.get(f"/admin/api/diagnostics/runs/{run_id}").json()["apps"]}
    assert apps["app-launcher"]["peak_procs"] == 3
    assert apps["llama.cpp"]["peak_procs"] == 1


def test_markdown_report_is_self_contained(client):
    run_id = _seed_run(ports=((8000, "local-llm-hub"),))
    res = client.get(f"/admin/api/diagnostics/runs/{run_id}/report")
    assert res.status_code == 200
    assert "attachment" in res.headers.get("content-disposition", "")
    text = res.text
    assert "# Machine diagnostics" in text
    assert "## Findings" in text and "## Load by app" in text
    assert "app-launcher" in text and "8000" in text


def test_export_carries_every_layer(client):
    run_id = _seed_run()
    body = client.get(f"/admin/api/diagnostics/runs/{run_id}/export").json()
    assert {"run", "samples", "process_aggregates", "app_aggregates",
            "ports", "process_timeline"} <= set(body)
    assert body["run"]["run_id"] == run_id
    assert len(body["samples"]) == 1


def test_delete_run(client):
    run_id = _seed_run()
    assert client.delete(f"/admin/api/diagnostics/runs/{run_id}").status_code == 200
    assert client.get(f"/admin/api/diagnostics/runs/{run_id}").status_code == 404
    assert client.delete(f"/admin/api/diagnostics/runs/{run_id}").status_code == 404


def test_reevaluate_rereads_thresholds(client):
    run_id = _seed_run()
    body = client.post(f"/admin/api/diagnostics/runs/{run_id}/evaluate").json()
    assert body["ok"] is True and body["level"] in {"healthy", "warning", "critical"}


def test_app_cpu_is_normalized_to_percent_of_machine(client):
    """psutil reports per-process CPU per *core*, so an app's summed figure can
    exceed 100% on a multi-core box. Rendering that raw beside a 25% machine-
    wide reading looks like a bug — it must be divided by the core count."""
    run_id = store.create_run(machine_id="pc-test", os_name="TestOS", interval_s=15.0,
                              duration_s=60.0, params={"cpu_count": 16})
    procs = [{"pid": i, "ppid": 1, "name": "svchost.exe", "cmdline": f"/bin/svchost {i}",
              "app_id": "windows-services", "cpu_percent": 100.0,
              "rss_bytes": 1024 ** 2, "num_threads": 2, "status": "running",
              "create_time": 1.0} for i in range(1, 17)]
    store.write_sample(run_id, _sample(time.time(), process_count=16), procs, [])
    store.finish_run(run_id)

    app = client.get(f"/admin/api/diagnostics/runs/{run_id}").json()["apps"][0]
    # 16 processes x 100% of one core / 16 cores == 100% of the machine.
    assert app["peak_cpu"] == 100.0


def test_cpu_count_comes_from_the_run_not_the_reader(client):
    """An exported DB read on different hardware must still normalize against
    the machine the capture came from."""
    run_id = store.create_run(machine_id="other-box", os_name="TestOS", interval_s=15.0,
                              duration_s=60.0, params={"cpu_count": 4})
    procs = [{"pid": i, "ppid": 1, "name": "x", "cmdline": "x", "app_id": "a",
              "cpu_percent": 100.0, "rss_bytes": 1024 ** 2, "num_threads": 1,
              "status": "running", "create_time": 1.0} for i in range(1, 5)]
    store.write_sample(run_id, _sample(time.time(), process_count=4), procs, [])
    store.finish_run(run_id)
    data = report.summary(run_id)
    assert data["cpu_count"] == 4
    assert data["apps"][0]["peak_cpu"] == 100.0


# ------------------------------------------------------------------- drift


def test_baseline_and_drift(client):
    base_id = _seed_run(apps=(("app-launcher", 2),), ram=40.0)
    assert client.post(f"/admin/api/diagnostics/runs/{base_id}/baseline").status_code == 200

    later_id = _seed_run(apps=(("app-launcher", 5), ("newapp", 1)), ram=70.0)
    drift = client.get(f"/admin/api/diagnostics/runs/{later_id}/drift").json()

    assert drift["baseline"]["run_id"] == base_id
    assert "newapp" in drift["new_apps"]
    row = next(r for r in drift["apps"] if r["app_id"] == "app-launcher")
    assert row["procs_before"] == 2 and row["procs_now"] == 5 and row["procs_delta"] == 3
    ram = next(c for c in drift["changes"] if c["label"] == "RAM peak %")
    assert ram["before"] == 40.0 and ram["now"] == 70.0 and ram["delta"] == 30.0


def test_drift_without_a_baseline_is_empty_not_an_error(client):
    run_id = _seed_run()
    body = client.get(f"/admin/api/diagnostics/runs/{run_id}/drift").json()
    assert body["baseline"] is None and body["changes"] == []


def test_drift_reports_new_and_gone_ports(client):
    base_id = _seed_run(ports=((8000, "local-llm-hub"),))
    client.post(f"/admin/api/diagnostics/runs/{base_id}/baseline")
    later_id = _seed_run(ports=((9999, "mystery"),))
    ports = {p["port"]: p["status"] for p in
             client.get(f"/admin/api/diagnostics/runs/{later_id}/drift").json()["ports"]}
    assert ports == {9999: "new", 8000: "gone"}


def test_baseline_is_resolved_per_machine(client):
    """A baseline from another machine must never be compared against."""
    other = _seed_run(machine_id="mac-mini")
    store.set_baseline(other)
    mine = _seed_run(machine_id="pc-test")
    assert report.drift(mine)["baseline"] is None


def test_baseline_of_unknown_run_is_404(client):
    assert client.post("/admin/api/diagnostics/runs/nope/baseline").status_code == 404


# ---------------------------------------------------------------- settings


def test_settings_round_trip_and_clamping(client):
    res = client.put("/admin/api/diagnostics/settings", json={
        "retention_days": 99999, "scheduled_enabled": False,
        "scheduled_interval_hours": 0.1,
    })
    saved = res.json()["settings"]
    assert saved["retention_days"] == diag_settings.MAX_RETENTION_DAYS
    assert saved["scheduled_interval_hours"] == diag_settings.MIN_SCHEDULE_HOURS
    assert diag_settings.load_settings().retention_days == diag_settings.MAX_RETENTION_DAYS


def test_enabling_the_schedule_arms_it_immediately(client):
    """A toggle the user must restart the hub to activate is a toggle that
    lies."""
    body = client.put("/admin/api/diagnostics/settings", json={
        "scheduled_enabled": True, "scheduled_interval_hours": 24, "retention_days": 90,
    }).json()
    assert body["settings"]["scheduled_active"] is True

    off = client.put("/admin/api/diagnostics/settings", json={
        "scheduled_enabled": False, "scheduled_interval_hours": 24, "retention_days": 90,
    }).json()
    assert off["settings"]["scheduled_active"] is False


def test_settings_survive_a_reload(client, tmp_path):
    client.put("/admin/api/diagnostics/settings", json={"retention_days": 30})
    diag_settings._cache.clear()
    assert diag_settings.load_settings().retention_days == 30


def test_broken_settings_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    diag_settings.set_settings_path(path)
    try:
        assert diag_settings.load_settings().retention_days == 90
    finally:
        diag_settings.set_settings_path(None)


def test_shutdown_drains_an_active_capture(tmp_path, monkeypatch):
    """A hub shutdown must gracefully stop an in-flight capture, not abandon it.

    Beyond finalizing the run as `stopped` instead of orphaning it, this is what
    stops a capture's worker thread from outliving the app and writing through
    the module-global `db_path` into the next test's DB — the intermittent
    `database is locked` the router suite showed before #316 added the drain."""
    _stub_real_backend_ops(monkeypatch)
    store.set_db_path(tmp_path / "diag.db")
    diag_settings.set_settings_path(tmp_path / "s.json")
    try:
        with TestClient(server_mod.app) as c:
            c.post("/admin/api/diagnostics/start", json={"duration_s": 600, "interval_s": 5})
            assert sampler.is_capturing() is True
        # Leaving the context ran the app's shutdown handlers.
        assert sampler.is_capturing() is False
    finally:
        store.set_db_path(None)
        diag_settings.set_settings_path(None)


# --------------------------------------------------------- ingest endpoint (#316)


def _remote_payload(*, platform="linux", machine="peer"):
    """A minimal valid portable-capture document for the ingest endpoint."""
    proc = {"pid": 1, "ppid": 1, "name": "python3",
            "cmdline": "/opt/automation/grocery/.venv/bin/python3 -m app",
            "cpu_percent": 5.0, "rss_bytes": 20 * 1024 ** 2, "num_threads": 2,
            "status": "running", "create_time": 1.0}
    sample = {"ts": 1000.0, "cpu_percent": 10.0, "per_core": [10.0], "load_avg": None,
              "ram": {"used_gb": 4.0, "total_gb": 16.0, "percent": 30.0}, "swap": {},
              "disk": {"percent": 20.0}, "disk_io": {}, "net_io": {}, "gpus": [],
              "process_count": 1, "ports_denied": False, "processes": [proc], "ports": []}
    return {"schema": "llm-hub-diagnostics-capture/1", "machine": machine,
            "hostname": machine, "os": f"{platform}-os", "platform": platform,
            "cpu_count": 8, "interval_s": 15.0, "started_at": 1000.0,
            "ended_at": 1060.0, "samples": [sample]}


def test_ingest_endpoint_creates_a_remote_run(client):
    body = client.post("/admin/api/diagnostics/ingest", json=_remote_payload()).json()
    assert body["ok"] is True
    run_id = body["run_id"]
    # It appears in the run list like any other run, tagged remote.
    runs = client.get("/admin/api/diagnostics/runs").json()["runs"]
    row = next(r for r in runs if r["run_id"] == run_id)
    assert row["trigger"] == "remote"
    # And its summary attributes centrally at ingest.
    summary = client.get(f"/admin/api/diagnostics/runs/{run_id}").json()
    assert "grocery" in {a["app_id"] for a in summary["apps"]}


def test_ingest_endpoint_accepts_the_machine_override_envelope(client):
    body = client.post("/admin/api/diagnostics/ingest", json={
        "payload": _remote_payload(machine="hostname-only"), "machine": "openclaw",
    }).json()
    summary = client.get(f"/admin/api/diagnostics/runs/{body['run_id']}").json()
    assert summary["run"]["machine_id"] == "openclaw"


def test_ingest_endpoint_refuses_garbage_with_400(client):
    resp = client.post("/admin/api/diagnostics/ingest",
                       json={"schema": "nope", "platform": "linux", "samples": []})
    assert resp.status_code == 400
