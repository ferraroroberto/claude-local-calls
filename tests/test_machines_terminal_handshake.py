"""Regression tests for the Machines-tab terminal websocket handshake.

``BearerTokenMiddleware`` is a ``BaseHTTPMiddleware``, which Starlette only
runs for ``http``-scope connections — a ``@router.websocket(...)`` route is
invisible to it. The terminal route therefore has to apply the trust rules
itself (``middleware.authorize_websocket``) before ``accept()``.

These tests pin that the websocket handshake reaches the *same* verdict as
the equivalent HTTP request in ``test_bearer_middleware.py``: a caller that
would be refused over HTTP must also be refused here, and one that would be
allowed must still get through. The upstream session factory is patched out
throughout, so an accepted handshake never dials a real machine — what's
under test is purely who gets past the handshake.
"""

from __future__ import annotations

import os

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app_web import server as admin_server
from app_web.routers import machines as machines_router
from src.webapp_config import WebappConfig

# X-Forwarded-For makes the proxied-loopback check fire, so the TestClient's
# pseudo-host stops counting as loopback and a token is actually required —
# the same trick test_bearer_middleware.py uses, and the same shape a real
# Cloudflare-tunnel / tailscale-serve caller arrives with.
_PROXY_HEADERS = {"X-Forwarded-For": "203.0.113.5"}

_TOKEN = "secret123"
_WS_PATH = "/api/machines/tower/terminal"


@pytest.fixture(autouse=True)
def _never_dial_upstream(monkeypatch):
    """Fail loudly if an accepted handshake tries to open a real session."""

    async def _refuse(*_a, **_kw):
        return {"ok": False, "error": "stubbed in tests"}

    monkeypatch.setattr(machines_router.ssh_terminal, "create_ssh_session", _refuse)


def _client(token: str = _TOKEN) -> TestClient:
    app = admin_server.create_app()
    app.state.webapp_config = WebappConfig(auth_token=token)
    return TestClient(app)


def test_handshake_refused_without_token():
    """A non-loopback caller with no credential must not get a socket."""
    with _client() as client:
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect(_WS_PATH, headers=_PROXY_HEADERS):
                pass
    assert excinfo.value.code == 1008


def test_handshake_refused_with_wrong_token():
    with _client() as client:
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect(
                _WS_PATH, headers={**_PROXY_HEADERS, "Authorization": "Bearer nope"}
            ):
                pass
    assert excinfo.value.code == 1008


def test_handshake_allowed_with_correct_bearer_token():
    """The credentialled path still works — the fix must not lock out the SPA."""
    with _client() as client:
        with client.websocket_connect(
            _WS_PATH,
            headers={**_PROXY_HEADERS, "Authorization": f"Bearer {_TOKEN}"},
        ) as ws:
            # Accepted, then closed by the stubbed session factory's error frame.
            assert ws.receive_json()["type"] == "error"


def test_handshake_allowed_with_query_token():
    """``?token=`` is what machines_terminal.js actually appends."""
    with _client() as client:
        with client.websocket_connect(
            f"{_WS_PATH}?token={_TOKEN}", headers=_PROXY_HEADERS
        ) as ws:
            assert ws.receive_json()["type"] == "error"


def test_handshake_allowed_from_loopback_without_token():
    """No proxy headers -> genuine loopback -> same bypass the HTTP path grants."""
    with _client() as client:
        with client.websocket_connect(_WS_PATH) as ws:
            assert ws.receive_json()["type"] == "error"


def test_handshake_allowed_when_no_token_is_configured():
    """A hub with no token set is unauthenticated by configuration, not by
    accident — the guard must not invent a credential requirement."""
    with _client(token="") as client:
        with client.websocket_connect(_WS_PATH, headers=_PROXY_HEADERS) as ws:
            assert ws.receive_json()["type"] == "error"
