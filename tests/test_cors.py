"""CORS for browser-based clients (#462).

Two halves, and the second is the one that matters:

1. the policy is what it claims — loopback allowed, named origins allowed,
   everything else refused, no wildcard reachable from config;
2. enabling it did not open a second way in. A preflight is answered
   without a token *because a browser never sends one on a preflight*,
   while the real request behind it still meets the bearer gate exactly as
   ``tests/test_bearer_middleware.py`` asserts.

The app under test is the real ``src.server.app``, so the middleware order
(CORS outermost, outside ``ParentBearerTokenMiddleware``) is exercised
rather than assumed.
"""

from __future__ import annotations

import os

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

from fastapi.testclient import TestClient

from src import cors_policy
from src import server as server_mod
from src import webapp_config as webapp_config_mod
from src.webapp_config import WebappConfig

LOOPBACK_ORIGIN = "http://localhost:3000"
FOREIGN_ORIGIN = "https://evil.example"
# Forces the bearer gate to treat the caller as non-loopback.
_PROXY_HEADERS = {"X-Forwarded-For": "203.0.113.5"}


def _client() -> TestClient:
    return TestClient(server_mod.app)


def _preflight(client: TestClient, origin: str, *, method: str = "POST",
               headers: str = "content-type", extra=None):
    return client.options(
        "/v1/messages",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": headers,
            **(extra or {}),
        },
    )


# ---- the pure policy ----

def test_loopback_origins_match_the_regex():
    import re

    pattern = re.compile(cors_policy.LOOPBACK_ORIGIN_REGEX)
    for origin in (
        "http://localhost",
        "http://localhost:8501",
        "https://localhost:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.2:5173",
        "http://[::1]:1234",
    ):
        assert pattern.fullmatch(origin), origin


def test_lookalike_origins_do_not_match_the_regex():
    import re

    pattern = re.compile(cors_policy.LOOPBACK_ORIGIN_REGEX)
    for origin in (
        "http://127.0.0.1.evil.example",
        "http://localhost.evil.example",
        "https://notlocalhost",
        "http://192.168.1.10:8000",
        "ftp://localhost",
    ):
        assert pattern.fullmatch(origin) is None, origin


def test_wildcard_origin_is_dropped_not_honoured():
    """The one thing the issue puts out of scope: no wildcard ever ships."""
    assert cors_policy.resolve_allowed_origins(["*"]) == []
    assert cors_policy.resolve_allowed_origins(["https://a.example", "*"]) == [
        "https://a.example"
    ]
    # A wildcard smuggled into an otherwise-plausible entry goes too.
    assert cors_policy.resolve_allowed_origins(["https://*.example"]) == []


def test_origins_are_normalised():
    assert cors_policy.resolve_allowed_origins(
        ["  https://Llm.Example.com/  ", "", "https://llm.example.com"]
    ) == ["https://llm.example.com"]


def test_default_config_is_loopback_only():
    kwargs = cors_policy.cors_kwargs(None)
    assert kwargs["allow_origins"] == []
    assert kwargs["allow_origin_regex"] == cors_policy.LOOPBACK_ORIGIN_REGEX
    # Credentialed CORS is the risk class we decline to take on at all.
    assert kwargs["allow_credentials"] is False


def test_sdk_headers_are_allowed():
    allowed = {h.lower() for h in cors_policy.CORS_ALLOW_HEADERS}
    for header in (
        "authorization",
        "content-type",
        "x-api-key",
        "anthropic-version",
        "anthropic-beta",
        "x-stainless-lang",
        "openai-beta",
    ):
        assert header in allowed


def test_sample_config_carries_the_setting():
    """A fresh clone must be able to see the knob without reading the source."""
    import json

    raw = json.loads(
        webapp_config_mod.SAMPLE_CONFIG_PATH.read_text(encoding="utf-8")
    )
    assert raw["cors_allow_origins"] == []


def test_config_round_trips_the_origins(tmp_path):
    path = tmp_path / "webapp_config.json"
    webapp_config_mod.save_webapp_config(
        WebappConfig(cors_allow_origins=["https://llm.example.com"]), path
    )
    assert webapp_config_mod.load_webapp_config(path).cors_allow_origins == [
        "https://llm.example.com"
    ]


# ---- the wire ----

def test_preflight_from_loopback_origin_succeeds():
    r = _preflight(_client(), LOOPBACK_ORIGIN)
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == LOOPBACK_ORIGIN
    allowed = r.headers["access-control-allow-headers"].lower()
    assert "authorization" in allowed and "x-api-key" in allowed


def test_preflight_succeeds_without_a_token(monkeypatch):
    """A browser sends no credentials on a preflight — if the bearer gate saw
    it, every cross-origin call would die before it started."""
    monkeypatch.setattr(
        webapp_config_mod,
        "load_webapp_config",
        lambda *a, **k: WebappConfig(auth_token="parentsecret"),
    )
    r = _preflight(_client(), LOOPBACK_ORIGIN, extra=_PROXY_HEADERS)
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == LOOPBACK_ORIGIN


def test_preflight_from_foreign_origin_is_refused():
    r = _preflight(_client(), FOREIGN_ORIGIN)
    assert r.status_code != 200
    assert "access-control-allow-origin" not in r.headers


def test_foreign_origin_cannot_read_a_real_response():
    """The request still runs (CORS never blocked a request, only the read) —
    the browser is denied the response because no ACAO comes back."""
    r = _client().get("/v1/models", headers={"Origin": FOREIGN_ORIGIN})
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers


def test_named_origin_is_allowed():
    """An origin added to the config is honoured — the extension path the
    issue asks for, exercised against a freshly built app so the config is
    actually read at middleware-construction time."""
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/ping")
    def _ping():
        return {"ok": True}

    cors_policy.install_cors(
        app, WebappConfig(cors_allow_origins=["https://llm.example.com"])
    )
    client = TestClient(app)
    r = client.get("/ping", headers={"Origin": "https://llm.example.com"})
    assert r.headers["access-control-allow-origin"] == "https://llm.example.com"
    r = client.get("/ping", headers={"Origin": FOREIGN_ORIGIN})
    assert "access-control-allow-origin" not in r.headers


def test_trace_headers_are_readable_by_page_javascript():
    """A response header a browser cannot read is the same as not sending it —
    so the #4 and #461 contracts have to be on expose_headers."""
    r = _client().get("/v1/models", headers={"Origin": LOOPBACK_ORIGIN})
    assert r.status_code == 200
    exposed = {
        h.strip().lower()
        for h in r.headers["access-control-expose-headers"].split(",")
    }
    assert "x-trace-id" in exposed
    assert "request-id" in exposed
    assert "anthropic-version" in exposed
    assert "warning" in exposed
    # And the header is genuinely on the response, not just permitted.
    assert r.headers.get("request-id")


def test_bearer_gate_still_rejects_non_loopback_without_token(monkeypatch):
    """CORS is not an auth bypass — the real request behind the preflight
    meets exactly the gate it met before."""
    monkeypatch.setattr(
        webapp_config_mod,
        "load_webapp_config",
        lambda *a, **k: WebappConfig(auth_token="parentsecret"),
    )
    r = _client().get(
        "/v1/models", headers={**_PROXY_HEADERS, "Origin": LOOPBACK_ORIGIN}
    )
    assert r.status_code == 401
