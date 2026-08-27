"""Regression tests for the request-origin arm of ``_caller_is_trusted``.

``app_web/middleware.py`` grants three credential-free trust rules — no token
configured, loopback source, allowlisted peer — that key off where a connection
came from. A browser can be made to open a connection from those same places on
behalf of a page served from somewhere else, so a request the browser reports as
started off-box must present the token instead of inheriting that trust.

The boundary is the one `src/cors_policy.py` already draws, and these tests pin
both sides of it: an off-box origin loses the bypass, while every legitimate
caller shape keeps it — the SPA's own fetches, a sister webapp on another
loopback port (the capability `cors_policy` exists to provide), a plain
SDK/curl client sending no browser headers at all, a token-bearing client, and
read-only requests.
"""

from __future__ import annotations

import os

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

from fastapi.testclient import TestClient

from app_web import server as admin_server
from src.webapp_config import WebappConfig

# An unrouted admin path, POSTed. The gate runs as middleware, ahead of
# routing, so this isolates the decision under test with no side effects: a
# refused request is 401 and an admitted one falls through to 404. Pointing
# these at a real mutation (``/api/services/docker/stop`` and friends) would
# make the passing assertions actually stop the developer's Docker.
MUTATING_PATH = "/api/__request_origin_probe__"
READ_ONLY_PATH = "/api/webauthn/status"

ADMITTED = 404  # past the gate, no such route
REFUSED = 401

TOKEN = "secret123"


def _admin_client(token: str = TOKEN) -> TestClient:
    app = admin_server.create_app()
    app.state.webapp_config = WebappConfig(auth_token=token)
    return TestClient(app)


def test_loopback_bypass_refused_for_off_box_origin():
    """The gap this file exists for: loopback source, no credential, but the
    browser reports the request was started by a page from off the machine."""
    client = _admin_client()
    r = client.post(MUTATING_PATH, headers={"Origin": "https://example.com"})
    assert r.status_code == REFUSED


def test_lookalike_origin_is_refused():
    """``LOOPBACK_ORIGIN_REGEX`` is matched with ``fullmatch``, so a hostname
    that merely starts with a loopback literal must not pass."""
    client = _admin_client()
    r = client.post(
        MUTATING_PATH, headers={"Origin": "http://127.0.0.1.evil.example"}
    )
    assert r.status_code == REFUSED


def test_fetch_metadata_fallback_refused_when_no_origin_header():
    """A browser that omits Origin is still identifiable by fetch metadata."""
    client = _admin_client()
    r = client.post(MUTATING_PATH, headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == REFUSED


def test_token_still_wins_for_an_off_box_origin():
    """Holding the credential is sufficient regardless of who started it."""
    client = _admin_client()
    r = client.post(
        MUTATING_PATH,
        headers={"Origin": "https://example.com", "Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == ADMITTED


def test_spa_same_origin_post_still_bypasses():
    """The admin SPA's own fetches must be unaffected."""
    client = _admin_client()
    r = client.post(
        MUTATING_PATH,
        headers={"Sec-Fetch-Site": "same-origin", "Origin": "http://127.0.0.1:8000"},
    )
    assert r.status_code == ADMITTED


def test_sister_webapp_on_another_loopback_port_still_bypasses():
    """`src/cors_policy.py` exists so a sister webapp on this machine can call
    the hub from the browser instead of proxying through its own backend, and
    its docstring rests that on the loopback bypass. A different loopback port
    (and `localhost` vs `127.0.0.1`, which fetch metadata calls cross-site) must
    therefore keep working."""
    client = _admin_client()
    for origin in ("http://127.0.0.1:8765", "http://localhost:8765", "http://[::1]:9000"):
        r = client.post(MUTATING_PATH, headers={"Origin": origin})
        assert r.status_code == ADMITTED, origin


def test_non_browser_client_without_fetch_metadata_still_bypasses():
    """curl / an SDK / the tray / a peer hub send neither header and must keep
    the loopback bypass — they carry no ambient credential to borrow."""
    client = _admin_client()
    r = client.post(MUTATING_PATH)
    assert r.status_code == ADMITTED


def test_read_only_request_from_off_box_origin_is_unaffected():
    """Safe methods are already covered by the CORS read barrier; gating them
    here would only break ordinary asset loads."""
    client = _admin_client()
    r = client.get(READ_ONLY_PATH, headers={"Origin": "https://example.com"})
    assert r.status_code == 200


def test_token_less_hub_still_refuses_an_off_box_origin():
    """With no token configured every caller is trusted — except this one,
    which is the only protection such a hub has."""
    client = _admin_client(token="")
    assert client.post(MUTATING_PATH).status_code == ADMITTED
    r = client.post(MUTATING_PATH, headers={"Origin": "https://example.com"})
    assert r.status_code == REFUSED
