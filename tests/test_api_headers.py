"""Anthropic header parity on the wire (#461).

Covers the three halves of the issue that live outside the auth gate
(``x-api-key`` is exercised in ``tests/test_bearer_middleware.py``, next to
the rest of the bearer-token coverage):

1. ``request-id`` on every ``/v1/messages`` response, equal to
   ``X-Trace-Id`` when a trace ID exists.
2. ``anthropic-version`` echoed, defaulted when absent.
3. ``anthropic-beta`` echoed but never reported as honoured.

Unit tests run with ``OTEL_SDK_DISABLED=true`` (tests/conftest.py), so the
no-span fallback path is the one under test here — which is exactly the
path that would otherwise have left ``request-id`` silently absent.
"""

from __future__ import annotations

import os

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

from fastapi.testclient import TestClient

from src import anthropic_headers
from src import server as server_mod


def _client() -> TestClient:
    return TestClient(server_mod.app)


def _messages(client: TestClient, headers=None):
    # An empty messages list 400s in the route body — the response still
    # travels back out through the outermost middleware, which is the layer
    # under test. Keeps the test off any real backend.
    return client.post(
        "/v1/messages",
        json={"model": "claude-haiku-4-5", "max_tokens": 16, "messages": []},
        headers=headers or {},
    )


# ---- the pure header builder ----

def test_default_version_when_client_sends_none():
    out = dict(anthropic_headers.parity_headers([]))
    assert out[b"anthropic-version"] == anthropic_headers.DEFAULT_ANTHROPIC_VERSION.encode()
    assert b"anthropic-beta" not in out
    assert b"warning" not in out


def test_version_is_echoed():
    out = dict(anthropic_headers.parity_headers([(b"anthropic-version", b"2024-10-22")]))
    assert out[b"anthropic-version"] == b"2024-10-22"


def test_betas_echoed_and_flagged_unimplemented():
    out = dict(
        anthropic_headers.parity_headers(
            [(b"anthropic-beta", b"prompt-caching-2024-07-31, context-1m-2025-08-07")]
        )
    )
    assert out[b"anthropic-beta"] == b"prompt-caching-2024-07-31, context-1m-2025-08-07"
    warning = out[b"warning"].decode()
    assert warning.startswith("299 local-llm-hub ")
    # Both names are named as NOT implemented — the echo must never read as
    # an acknowledgement of support.
    assert "not implemented" in warning
    assert "prompt-caching-2024-07-31" in warning
    assert "context-1m-2025-08-07" in warning


def test_implemented_beta_gets_no_warning(monkeypatch):
    monkeypatch.setattr(
        anthropic_headers, "IMPLEMENTED_BETAS", frozenset({"already-shipped"})
    )
    out = dict(anthropic_headers.parity_headers([(b"anthropic-beta", b"already-shipped")]))
    assert out[b"anthropic-beta"] == b"already-shipped"
    assert b"warning" not in out


def test_repeated_beta_headers_are_merged():
    out = dict(
        anthropic_headers.parity_headers(
            [(b"anthropic-beta", b"one"), (b"anthropic-beta", b"two")]
        )
    )
    assert out[b"anthropic-beta"] == b"one, two"


def test_echo_is_sanitised():
    """A client value never gets to split a header or unbalance the Warning."""
    out = dict(
        anthropic_headers.parity_headers(
            [(b"anthropic-version", b'2023-06-01"\r\nx-injected: 1')]
        )
    )
    value = out[b"anthropic-version"].decode()
    assert "\r" not in value and "\n" not in value and '"' not in value
    assert value.startswith("2023-06-01")


# ---- the wire ----

def test_messages_response_carries_request_id():
    r = _messages(_client())
    assert r.status_code == 400
    request_id = r.headers.get("request-id")
    assert request_id and len(request_id) == 32


def test_request_id_matches_x_trace_id_when_present():
    r = _messages(_client())
    trace_id = r.headers.get("x-trace-id")
    if trace_id:  # only when the OTel SDK is live
        assert r.headers["request-id"] == trace_id


def test_messages_response_echoes_default_version():
    r = _messages(_client())
    assert r.headers["anthropic-version"] == anthropic_headers.DEFAULT_ANTHROPIC_VERSION


def test_messages_response_echoes_client_version_and_beta():
    r = _messages(_client(), headers={
        "anthropic-version": "2024-10-22",
        "anthropic-beta": "some-beta-2026-01-01",
    })
    assert r.headers["anthropic-version"] == "2024-10-22"
    assert r.headers["anthropic-beta"] == "some-beta-2026-01-01"
    assert "not implemented" in r.headers["warning"]


def test_request_without_version_still_succeeds():
    r = _client().get("/v1/models")
    assert r.status_code == 200
    assert "request-id" in r.headers


def test_openai_shape_route_gets_no_anthropic_headers():
    """The parity headers are Anthropic-shape only; request-id is universal."""
    r = _client().post(
        "/v1/chat/completions",
        json={"model": "does-not-exist", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert "anthropic-version" not in r.headers
    assert "request-id" in r.headers


def test_static_assets_get_no_request_id():
    r = _client().get("/admin/static/does-not-exist.js")
    assert "request-id" not in r.headers
