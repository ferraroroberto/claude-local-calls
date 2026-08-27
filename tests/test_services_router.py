"""Unit tests for app_web/routers/services.py + src/services.py (issue #27).

Verifies the status endpoint shape, the launch endpoint orchestration
(with the heavy helpers monkeypatched out — we don't want to spawn
Docker Desktop or run docker compose from a unit test), and a couple
of pure-Python behaviours on the helpers module. The peer-probe trio
(``peer_health``/``hub_peers``) moved to ``src/remote_stats.py`` (#533) —
patched there below, not on ``svc``.
"""

from __future__ import annotations

import asyncio
import sys

from fastapi.testclient import TestClient

from src import remote_stats
from src import server as server_mod
from src import services as svc


def _client() -> TestClient:
    return TestClient(server_mod.app)


def _run(coro):
    """Run a coroutine on a fresh thread+loop.

    ``asyncio.run()`` (and ``loop.run_until_complete()`` on the main
    thread) raise when an outer loop is already running — which happens
    in this suite after the Playwright e2e tests have started one.
    Running on a worker thread guarantees a clean asyncio context.
    """
    import threading

    bucket: dict = {}

    def _worker() -> None:
        loop = asyncio.new_event_loop()
        try:
            bucket["value"] = loop.run_until_complete(coro)
        except BaseException as exc:  # noqa: BLE001 — re-raised in caller
            bucket["error"] = exc
        finally:
            loop.close()

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    if "error" in bucket:
        raise bucket["error"]
    return bucket.get("value")


# ----------------------------------------------------------------- helpers


def test_find_docker_desktop_returns_none_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert svc.find_docker_desktop() is None


def test_langfuse_start_script_path_per_platform(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert svc.langfuse_start_script().name == "start_langfuse.bat"
    monkeypatch.setattr(sys, "platform", "linux")
    assert svc.langfuse_start_script().name == "start_langfuse.sh"


def test_docker_status_missing_binary(monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda _: None)
    result = _run(svc.docker_status())
    assert result["running"] is False
    assert "PATH" in result["error"]


def _run_on_selector_loop(coro):
    """Like ``_run`` but pins the worker thread's loop to ``SelectorEventLoop``.

    Regression pin for #225: the hub's uvicorn servers run under
    ``asyncio.SelectorEventLoop`` on Windows since #223, and
    ``SelectorEventLoop`` has no subprocess support there —
    ``asyncio.create_subprocess_exec`` raises ``NotImplementedError``
    under it. ``docker_status()``/``_run_langfuse_start_script()`` must
    use a thread-executor path that doesn't depend on the loop's
    subprocess transport, so this must pass on any platform + loop combo.
    """
    import threading

    bucket: dict = {}

    def _worker() -> None:
        loop = (
            asyncio.SelectorEventLoop()
            if sys.platform == "win32"
            else asyncio.new_event_loop()
        )
        try:
            bucket["value"] = loop.run_until_complete(coro)
        except BaseException as exc:  # noqa: BLE001 — re-raised in caller
            bucket["error"] = exc
        finally:
            loop.close()

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    if "error" in bucket:
        raise bucket["error"]
    return bucket.get("value")


def test_docker_status_does_not_raise_under_selector_event_loop():
    """#225: docker_status() must not NotImplementedError under Selector.

    Doesn't assert on Docker's actual running state (host-dependent) —
    only that the call completes and returns the well-formed dict shape
    instead of blowing up through the ASGI app as a 500.
    """
    result = _run_on_selector_loop(svc.docker_status())
    assert isinstance(result, dict)
    assert "running" in result
    assert "error" in result


def test_run_langfuse_start_script_does_not_raise_under_selector_event_loop(monkeypatch):
    """#225: same NotImplementedError trap on the launch-endpoint path."""
    monkeypatch.setattr(
        svc, "langfuse_start_script", lambda: svc.PROJECT_ROOT / "does-not-exist.bat"
    )
    result = _run_on_selector_loop(svc._run_langfuse_start_script())
    assert result["ok"] is False
    assert "start script not found" in result["stderr"]


def test_langfuse_health_unreachable_returns_clean_payload(monkeypatch):
    """When httpx can't talk to Langfuse, the helper must return a
    well-formed dict, not raise — the SPA card depends on that."""

    class _BoomClient:
        async def get(self, *a, **kw):
            raise RuntimeError("synthetic network failure")

    # langfuse_health uses the shared pooled client (#392), not a fresh
    # per-call httpx.AsyncClient — patch the getter.
    monkeypatch.setattr(svc, "get_async_client", lambda: _BoomClient())
    result = _run(svc.langfuse_health())
    assert result["reachable"] is False
    assert "synthetic network failure" in result["error"]
    assert result["host"]  # always set


async def _fake_peer_health(host_id, timeout_s=None):
    """Stub for remote_stats.peer_health — no real network I/O against any peer."""
    return {
        "reachable": host_id == "gaming",
        "status_code": 200 if host_id == "gaming" else 0,
        "error": "" if host_id == "gaming" else "down",
        "address": "http://x:8000",
        "local_git_sha": "abc123",
        "remote_git_sha": "abc123" if host_id == "gaming" else None,
        "git_sha_match": host_id == "gaming",
    }


def test_services_status_endpoint_shape(monkeypatch):
    monkeypatch.setattr(remote_stats, "peer_health", _fake_peer_health)
    r = _client().get("/admin/api/services/status")
    assert r.status_code == 200, r.text
    body = r.json()
    for key in ("docker", "langfuse", "peers", "launchable", "platform"):
        assert key in body, body
    assert isinstance(body["docker"], dict)
    assert "running" in body["docker"]
    assert isinstance(body["langfuse"], dict)
    assert "reachable" in body["langfuse"]
    assert isinstance(body["launchable"], bool)
    assert isinstance(body["peers"], list)


def test_services_status_peers_cover_every_hub_running_host(monkeypatch):
    """#372: gaming (owns a Linux hub since #323/#370's non-empty `enabled:`)
    and mac-mini-m4 both surface as peer rows; openclaw (managed-only, no
    `enabled:`) must not — same `runs_hub` test the fleet placement grid
    already applies per host (model_registry.hub_peer_ids)."""
    monkeypatch.setattr(remote_stats, "peer_health", _fake_peer_health)
    r = _client().get("/admin/api/services/status")
    assert r.status_code == 200, r.text
    peers = r.json()["peers"]
    peer_ids = {p["host_id"] for p in peers}
    assert peer_ids == {"mac-mini-m4", "gaming"}
    assert "openclaw" not in peer_ids
    assert "tower" not in peer_ids  # never lists the active host as its own peer

    gaming = next(p for p in peers if p["host_id"] == "gaming")
    assert gaming["display_name"] == "gaming"
    assert gaming["reachable"] is True
    assert gaming["git_sha_match"] is True

    mac_mini = next(p for p in peers if p["host_id"] == "mac-mini-m4")
    assert mac_mini["display_name"] == "Mac Mini M4"
    assert mac_mini["reachable"] is False


def test_hub_peer_ids_excludes_managed_only_and_active_host():
    """Pure-Python pin on the shared helper backing the peers list — no
    router/network involved (#372)."""
    from src.model_registry import hub_peer_ids

    peers = hub_peer_ids("tower")
    assert set(peers) == {"mac-mini-m4", "gaming"}
    assert "openclaw" not in peers
    assert "tower" not in peers


# ----------------------------------------------------------------- /launch


def test_services_launch_skips_when_both_up(monkeypatch):
    """Both services already up → both steps come back as `skipped`."""

    async def _docker_up(*a, **kw):
        return {"running": True, "error": "", "server_version": "test"}

    async def _lf_up(*a, **kw):
        return {"reachable": True, "status_code": 200, "error": "", "host": "x"}

    monkeypatch.setattr(svc, "docker_status", _docker_up)
    monkeypatch.setattr(svc, "langfuse_health", _lf_up)

    r = _client().post("/admin/api/services/launch")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    names = {s["name"]: s["status"] for s in body["steps"]}
    assert names == {"docker_engine": "skipped", "langfuse_stack": "skipped"}


def test_services_launch_errors_off_windows_when_docker_down(monkeypatch):
    """Non-Windows hosts can't auto-launch Docker — surface a clear error."""

    async def _docker_down(*a, **kw):
        return {"running": False, "error": "down"}

    async def _lf_down(*a, **kw):
        return {"reachable": False, "status_code": 0, "error": "down", "host": "x"}

    monkeypatch.setattr(svc, "docker_status", _docker_down)
    monkeypatch.setattr(svc, "langfuse_health", _lf_down)
    monkeypatch.setattr(svc.sys, "platform", "linux")

    r = _client().post("/admin/api/services/launch")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is False
    err_step = body["steps"][0]
    assert err_step["name"] == "docker_engine"
    assert err_step["status"] == "error"
    assert "Windows-only" in err_step["detail"]


def test_services_launch_runs_langfuse_script_when_docker_up(monkeypatch):
    """Docker up, Langfuse down → only the langfuse step runs."""

    async def _docker_up(*a, **kw):
        return {"running": True, "error": "", "server_version": "test"}

    async def _lf_down(*a, **kw):
        return {"reachable": False, "status_code": 0, "error": "down", "host": "x"}

    async def _run_script_ok():
        return {"ok": True, "returncode": 0, "stdout": "", "stderr": ""}

    async def _ready(*a, **kw):
        return True

    monkeypatch.setattr(svc, "docker_status", _docker_up)
    monkeypatch.setattr(svc, "langfuse_health", _lf_down)
    monkeypatch.setattr(svc, "_run_langfuse_start_script", _run_script_ok)
    monkeypatch.setattr(svc, "wait_for_langfuse", _ready)

    r = _client().post("/admin/api/services/launch")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    steps = {s["name"]: s["status"] for s in body["steps"]}
    assert steps["docker_engine"] == "skipped"
    assert steps["langfuse_stack"] == "ok"


# ----------------------------------------------- individual start/stop (#284)


def test_langfuse_stop_script_path_per_platform(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert svc.langfuse_stop_script().name == "stop_langfuse.bat"
    monkeypatch.setattr(sys, "platform", "linux")
    assert svc.langfuse_stop_script().name == "stop_langfuse.sh"


def test_docker_start_endpoint_skips_when_already_up(monkeypatch):
    async def _docker_up(*a, **kw):
        return {"running": True, "error": "", "server_version": "test"}

    monkeypatch.setattr(svc, "docker_status", _docker_up)
    r = _client().post("/admin/api/services/docker/start")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["steps"][0] == {"name": "docker_engine", "status": "skipped", "detail": "engine already up"}


def test_docker_stop_endpoint_skips_when_already_down(monkeypatch):
    async def _docker_down(*a, **kw):
        return {"running": False, "error": "down"}

    monkeypatch.setattr(svc, "docker_status", _docker_down)
    r = _client().post("/admin/api/services/docker/stop")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["steps"][0] == {"name": "docker_engine", "status": "skipped", "detail": "already down"}


def test_docker_stop_endpoint_runs_cli_when_running(monkeypatch):
    async def _docker_up(*a, **kw):
        return {"running": True, "error": "", "server_version": "test"}

    def _stop_ok(timeout_s):
        return {"ok": True, "detail": "Docker Desktop stopped"}

    monkeypatch.setattr(svc, "docker_status", _docker_up)
    monkeypatch.setattr(svc, "_stop_docker_desktop_sync", _stop_ok)
    r = _client().post("/admin/api/services/docker/stop")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["steps"][0] == {"name": "docker_engine", "status": "ok", "detail": "Docker Desktop stopped"}


def test_langfuse_start_endpoint_skips_when_already_up(monkeypatch):
    async def _lf_up(*a, **kw):
        return {"reachable": True, "status_code": 200, "error": "", "host": "x"}

    monkeypatch.setattr(svc, "langfuse_health", _lf_up)
    r = _client().post("/admin/api/services/langfuse/start")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["steps"][0] == {"name": "langfuse_stack", "status": "skipped", "detail": "stack already up"}


def test_langfuse_stop_endpoint_skips_when_already_down(monkeypatch):
    async def _lf_down(*a, **kw):
        return {"reachable": False, "status_code": 0, "error": "down", "host": "x"}

    monkeypatch.setattr(svc, "langfuse_health", _lf_down)
    r = _client().post("/admin/api/services/langfuse/stop")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["steps"][0] == {"name": "langfuse_stack", "status": "skipped", "detail": "already down"}


def test_langfuse_stop_endpoint_runs_script_when_reachable(monkeypatch):
    async def _lf_up(*a, **kw):
        return {"reachable": True, "status_code": 200, "error": "", "host": "x"}

    def _stop_script_ok():
        return {"ok": True, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(svc, "langfuse_health", _lf_up)
    monkeypatch.setattr(svc, "_run_langfuse_stop_script_sync", _stop_script_ok)
    r = _client().post("/admin/api/services/langfuse/stop")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["steps"][0] == {"name": "langfuse_stack", "status": "ok", "detail": "containers stopped"}


def test_agentsview_stop_endpoint_skips_when_disabled(monkeypatch):
    monkeypatch.setattr("src.agentsview_usage._base_url", lambda: "")
    r = _client().post("/admin/api/services/agentsview/stop")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["steps"][0]["status"] == "skipped"


def test_agentsview_stop_endpoint_skips_when_already_down(monkeypatch):
    monkeypatch.setattr("src.agentsview_usage._base_url", lambda: "http://127.0.0.1:8080")
    monkeypatch.setattr("src.server_process.find_port_pids", lambda port: [])
    r = _client().post("/admin/api/services/agentsview/stop")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["steps"][0] == {"name": "agentsview", "status": "skipped", "detail": "already down"}


def test_agentsview_stop_endpoint_kills_port_holder(monkeypatch):
    monkeypatch.setattr("src.agentsview_usage._base_url", lambda: "http://127.0.0.1:8080")
    monkeypatch.setattr("src.server_process.find_port_pids", lambda port: [4242])
    monkeypatch.setattr("src.server_process.kill_pid", lambda pid: (True, f"killed pid {pid}"))
    r = _client().post("/admin/api/services/agentsview/stop")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert "4242" in body["steps"][0]["detail"]
