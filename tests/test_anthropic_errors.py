"""Anthropic-shaped error envelopes on /v1/messages (#460).

Three things are under test, and the third is the one that matters most:

1. ``/v1/messages`` errors serialise as
   ``{"type": "error", "error": {"type": ..., "message": ...}}`` with the
   right ``error.type`` for the status.
2. ``/v1/chat/completions`` — and every non-Anthropic route — keeps
   FastAPI's ``{"detail": ...}`` shape, because OpenAI-shape callers parse
   a different envelope. A blanket app-level handler would break this.
3. The real ``anthropic`` SDK, driven against the app, raises its typed
   exception *and* sees the envelope in ``exc.body`` — demonstrated, not
   asserted from the shape alone.
"""

from __future__ import annotations

import os

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

import anthropic
import pytest
from fastapi.testclient import TestClient

from src import anthropic_errors
from src import chat_translation as chat_translation_mod
from src import server as server_mod


def _client() -> TestClient:
    return TestClient(server_mod.app)


def _empty_messages_request(client: TestClient):
    return client.post(
        "/v1/messages",
        json={"model": "claude-haiku-4-5", "max_tokens": 16, "messages": []},
    )


# ---- the status -> error.type mapping ----

@pytest.mark.parametrize(
    "status,expected",
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (422, "invalid_request_error"),
        (429, "rate_limit_error"),
        (500, "api_error"),
        (502, "api_error"),
        (503, "overloaded_error"),
        (504, "api_error"),
        # Unmapped statuses still land on a valid enum member.
        (418, "invalid_request_error"),
        (599, "api_error"),
    ],
)
def test_error_type_mapping(status, expected):
    assert anthropic_errors.anthropic_error_type(status) == expected


def test_only_anthropic_shape_paths_match():
    assert anthropic_errors.is_anthropic_shape_path("/v1/messages")
    assert anthropic_errors.is_anthropic_shape_path("/v1/messages/count_tokens")
    assert not anthropic_errors.is_anthropic_shape_path("/v1/chat/completions")
    assert not anthropic_errors.is_anthropic_shape_path("/v1/models")
    assert not anthropic_errors.is_anthropic_shape_path("/v1/messages-ish")


# ---- the wire ----

def test_messages_400_uses_anthropic_envelope():
    r = _empty_messages_request(_client())
    assert r.status_code == 400
    assert r.json() == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "messages must not be empty",
        },
    }


def test_messages_502_maps_to_api_error(monkeypatch):
    from src.claude_cli import ClaudeCLIError

    def fake_call(prompt, *, model=None, system=None, attachments=None, timeout=600.0):
        raise ClaudeCLIError("upstream exploded")

    monkeypatch.setattr(chat_translation_mod, "call_claude", fake_call)

    r = _client().post(
        "/v1/messages",
        json={
            "model": "claude-haiku-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 502
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "api_error"
    assert "upstream exploded" in body["error"]["message"]


def test_messages_body_validation_uses_anthropic_envelope():
    """A malformed body is an error on this route too — same envelope."""
    r = _client().post("/v1/messages", json={"max_tokens": 16})
    assert r.status_code == 422
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    # Readable, and it names the offending field.
    assert "model" in body["error"]["message"]
    assert "detail" not in body


# ---- the OpenAI-shape route must be untouched ----

def test_chat_completions_error_shape_unchanged():
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "does-not-exist",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 400
    body = r.json()
    assert "unknown model" in body["detail"]
    assert "error" not in body
    assert "type" not in body


def test_chat_completions_validation_shape_unchanged():
    r = _client().post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 422
    body = r.json()
    assert isinstance(body["detail"], list)
    assert "error" not in body


def test_other_routes_keep_default_shape():
    """Non-chat routes (here: the admin sub-app's auth gate) are untouched."""
    r = _client().get("/v1/models/does-not-exist")
    assert r.status_code == 404
    assert r.json() == {"detail": "Not Found"}


# ---- observability ring still sees these as errors ----

def test_error_still_lands_in_the_observability_ring():
    from src.hub_observability import OBS

    r = _empty_messages_request(_client())
    assert r.status_code == 400
    # recent_errors() is newest-first.
    newest = OBS.recent_errors(limit=1)[0]
    assert newest["path"] == "/v1/messages"
    assert newest["status"] == 400


# ---- the anthropic SDK, driven for real ----

def test_anthropic_sdk_raises_typed_error_with_envelope():
    """The SDK's typed exception, against the hub app, over real HTTP plumbing.

    ``TestClient`` is an ``httpx.Client`` subclass, so the SDK can use it as
    its transport — the request goes through the SDK's own request builder,
    error classifier and response parser, not a hand-rolled stand-in.
    """
    sdk = anthropic.Anthropic(
        api_key="local-dummy",
        base_url="http://testserver",
        http_client=_client(),
        max_retries=0,
    )
    with pytest.raises(anthropic.BadRequestError) as excinfo:
        sdk.messages.create(model="claude-haiku-4-5", max_tokens=16, messages=[])

    exc = excinfo.value
    assert exc.status_code == 400
    assert exc.body == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "messages must not be empty",
        },
    }
