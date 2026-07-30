"""Unit tests for src/agentsview_usage.py (issue #280).

All HTTP is faked with ``httpx.MockTransport`` via the ``_build_client`` test
seam — no network, no real AgentsView.  Tests drive ``_refresh()``
synchronously (no daemon thread) and assert the snapshot/mapping behaviour:
records in the shared ``UsageRecord`` shape, native vendors never fetched,
graceful degradation when the service is absent or foreign.
"""

from __future__ import annotations

from datetime import timezone

import httpx
import pytest

from src import agentsview_usage as av
from src import code_usage, usage_pricing


@pytest.fixture(autouse=True)
def _isolate_module_state():
    """The snapshot is a module-level singleton — wipe it around every test so
    seeded records never leak into other test files."""
    av._reset_for_tests()
    yield
    av._reset_for_tests()


_PING_OK = {"ok": True, "service": "agentsview", "version": "0.37.5", "pid": 1}

_SESSION = {
    "id": "gemini-sess-1",
    "project": "E:\\automation\\grocery-shopping-automation",
    "startedAt": "2026-07-10T08:30:00Z",
}

_USAGE = {
    "session_id": "gemini-sess-1",
    "agent": "gemini",
    "total_output_tokens": 12303,
    "peak_context_tokens": 14552,
    "has_token_data": True,
    "cost_usd": 0.18,
    "models": ["gemini-3-pro"],
    "breakdown": [
        {
            "model": "gemini-3-pro",
            "timestamp": "2026-07-10T09:00:00Z",  # per-call ts (v0.37.5)
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 250,
            "cost_usd": 0.10,
        },
        {
            "model": "gemini-3-pro",
            "input_tokens": 2000,
            "output_tokens": 700,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 800,
            "cost_usd": 0.08,
        },
    ],
}


def _install(monkeypatch, handler):
    """Point the module at a MockTransport client and reset its state."""
    monkeypatch.setenv("AGENTSVIEW_BASE_URL", "http://av.test")

    def _fake_client():
        return httpx.Client(
            transport=httpx.MockTransport(handler), base_url="http://av.test"
        )

    monkeypatch.setattr(av, "_build_client", _fake_client)
    av._reset_for_tests()


def _std_handler(request: httpx.Request) -> httpx.Response:
    """A healthy AgentsView with one gemini session (and native claude data)."""
    path = request.url.path
    if path == "/api/ping":
        return httpx.Response(200, json=_PING_OK)
    if path == "/api/v1/agents":
        # Mixed shapes on purpose: object entries (live v0.37.5) + the
        # vscode-copilot split slug that must map to the native exclusion.
        return httpx.Response(
            200,
            json={
                "agents": [
                    {"name": "claude", "session_count": 5},
                    {"name": "vscode-copilot", "session_count": 1},
                    {"name": "gemini", "session_count": 2},
                ]
            },
        )
    if path == "/api/v1/sessions":
        assert request.url.params.get("agent") != "claude", (
            "native vendor must never be fetched from AgentsView"
        )
        return httpx.Response(200, json={"sessions": [_SESSION], "total": 1})
    if path == "/api/v1/sessions/gemini-sess-1/usage":
        return httpx.Response(200, json=_USAGE)
    return httpx.Response(404, json={"error": {"code": "not_found"}})


def test_maps_session_to_records(monkeypatch):
    _install(monkeypatch, _std_handler)
    av._refresh()

    records = av.all_records()
    assert len(records) == 2  # one per breakdown row
    r = records[0]
    assert r.vendor == "agy"  # gemini slug maps to the merged agy vendor
    assert r.session_id == "gemini-sess-1"
    assert r.model == "gemini-3-pro"
    assert r.ts.tzinfo == timezone.utc
    # First row carries its own per-call timestamp; the second falls back to
    # the session's startedAt.
    assert r.ts.isoformat().startswith("2026-07-10T09:00")
    assert records[1].ts.isoformat().startswith("2026-07-10T08:30")
    assert r.input_tokens == 1000
    assert r.output_tokens == 500
    assert r.cache_read_tokens == 250
    assert r.credits_usd == pytest.approx(0.10)
    # Path-shaped project groups under the same key native vendors use.
    assert r.project_key == "E--automation-grocery-shopping-automation"
    assert r.project_name == "grocery-shopping-automation"

    assert av.discovered_vendors() == ["agy"]
    st = av.status()
    assert st["reachable"] is True
    assert st["version"] == "0.37.5"
    assert st["vendors"] == ["agy"]


def test_native_agents_never_fetched(monkeypatch):
    _install(monkeypatch, _std_handler)  # handler asserts agent != claude
    av._refresh()
    assert av.discovered_vendors() == ["agy"]  # unmapped slugs ignored
    assert all(r.vendor != "claude" for r in av.all_records())


def test_unreachable_degrades(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused")

    _install(monkeypatch, handler)
    av._refresh()  # must not raise
    assert av.all_records() == []
    st = av.status()
    assert st["reachable"] is False
    assert st["error"]


def test_foreign_service_on_port(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={"ok": True, "service": "some-other-app"})

    _install(monkeypatch, handler)
    av._refresh()
    assert av.status()["reachable"] is False
    assert "not agentsview" in av.status()["error"]


def test_partial_fields_cost_only(monkeypatch):
    usage = {
        "session_id": "gemini-sess-1",
        "has_token_data": False,
        "cost_usd": 0.05,
        "models": [],
        "breakdown": [],
    }

    def handler(request):
        path = request.url.path
        if path == "/api/ping":
            return httpx.Response(200, json=_PING_OK)
        if path == "/api/v1/agents":
            return httpx.Response(200, json=["gemini"])  # bare-list shape
        if path == "/api/v1/sessions":
            return httpx.Response(200, json={"sessions": [_SESSION]})
        if path == "/api/v1/sessions/gemini-sess-1/usage":
            return httpx.Response(200, json=usage)
        return httpx.Response(404)

    _install(monkeypatch, handler)
    av._refresh()
    records = av.all_records()
    assert len(records) == 1
    r = records[0]
    assert r.model == "unknown"
    assert r.input_tokens == 0 and r.output_tokens == 0
    assert r.credits_usd == pytest.approx(0.05)


def test_session_cost_not_double_counted(monkeypatch):
    """Rows without per-row cost: session cost lands on the first record only."""
    usage = dict(_USAGE)
    usage["breakdown"] = [
        {k: v for k, v in row.items() if k != "cost_usd"}
        for row in _USAGE["breakdown"]
    ]

    def handler(request):
        path = request.url.path
        if path == "/api/ping":
            return httpx.Response(200, json=_PING_OK)
        if path == "/api/v1/agents":
            return httpx.Response(200, json={"agents": ["gemini"]})
        if path == "/api/v1/sessions":
            return httpx.Response(200, json={"sessions": [_SESSION]})
        if path == "/api/v1/sessions/gemini-sess-1/usage":
            return httpx.Response(200, json=usage)
        return httpx.Response(404)

    _install(monkeypatch, handler)
    av._refresh()
    records = av.all_records()
    assert sum(r.credits_usd for r in records) == pytest.approx(0.18)


def test_snapshot_survives_outage(monkeypatch):
    _install(monkeypatch, _std_handler)
    av._refresh()
    assert len(av.all_records()) == 2

    def down(request):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(
        av,
        "_build_client",
        lambda: httpx.Client(
            transport=httpx.MockTransport(down), base_url="http://av.test"
        ),
    )
    av._refresh()
    # Last-known data stays visible; only the flag flips.
    assert len(av.all_records()) == 2
    assert av.discovered_vendors() == ["agy"]
    st = av.status()
    assert st["reachable"] is False
    assert st["error"]


def test_disabled_env_never_probes(monkeypatch):
    def handler(request):  # pragma: no cover - must never run
        raise AssertionError("disabled integration must not probe")

    _install(monkeypatch, handler)
    monkeypatch.setenv("AGENTSVIEW_BASE_URL", "")
    av._kick_refresh_if_stale()  # no thread, no probe
    assert av.all_records() == []
    assert av.status()["enabled"] is False


def test_gather_and_summary_dynamic_vendor(monkeypatch):
    _install(monkeypatch, _std_handler)
    av._refresh()
    monkeypatch.setenv("AGENTSVIEW_BASE_URL", "")  # freeze: snapshot-only

    assert code_usage.is_valid_vendor("agy") is True
    assert code_usage.is_valid_vendor("bogus") is False

    summary = code_usage.get_summary("all", "agy")
    assert summary["vendor"] == "agy"
    vendors = {row["vendor"] for row in summary["by_vendor"]}
    assert vendors == {"agy"}
    assert summary["totals"]["output_tokens"] == 1200  # 500 + 700

    # AGY records price per tile against the Gemini list-price table (#280) —
    # AgentsView's own cost_usd is ignored (it can't price display-name ids).
    rec = av.all_records()[0]  # gemini-3-pro: 1000 in / 500 out / 250 cache
    input_cost, output_cost, cache_cost = usage_pricing.record_costs(rec)
    assert input_cost == pytest.approx(1000 * 2.0 / 1e6)
    assert output_cost == pytest.approx(500 * 12.0 / 1e6)
    assert cache_cost == pytest.approx(250 * 0.20 / 1e6)
    # Display-name ids collapse to the same family as raw ids.
    assert usage_pricing.gemini_family("Gemini 3.1 Pro (High)") == "pro"
    assert usage_pricing.gemini_family("gemini-3.1-flash-lite-preview") == "flash-lite"
    assert usage_pricing.gemini_family("Gemini 3.5 Flash (High)") == "flash"
    assert usage_pricing.gemini_family("unknown") == ""
