"""Model-name → backend routing in src.server.

Exercises the /v1/messages path for an openai-backed model (dispatched to
mocked `call_openai_chat`) plus the unknown-model 400 branch. Claude
routing is already covered by test_server.py.
"""

from __future__ import annotations

import os

os.environ.setdefault("CLAUDE_LOCAL_CALLS_HOST", "pc-cuda")

from fastapi.testclient import TestClient

from src import server as server_mod


def _fake_openai_response(text: str = "pong"):
    return {
        "id": "chatcmpl-xyz",
        "object": "chat.completion",
        "model": "qwen3.5-9b",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


def test_messages_routes_openai_backend(monkeypatch):
    captured = {}

    def fake_call(base_url, model, messages, *, max_tokens=None, temperature=None, timeout=600.0, extra=None):
        captured["base_url"] = base_url
        captured["model"] = model
        captured["messages"] = messages
        captured["max_tokens"] = max_tokens
        return _fake_openai_response("pong")

    monkeypatch.setattr(server_mod, "call_openai_chat", fake_call)

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "qwen3.5-9b",
            "max_tokens": 64,
            "system": "Answer briefly.",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["content"] == [{"type": "text", "text": "pong"}]
    assert body["model"] == "qwen3.5-9b"
    assert body["stop_reason"] == "end_turn"

    # System prompt was prepended to the OpenAI messages.
    assert captured["model"] == "qwen3.5-9b"
    assert captured["messages"][0] == {"role": "system", "content": "Answer briefly."}
    assert captured["messages"][1]["role"] == "user"
    assert "127.0.0.1:8081" in captured["base_url"]


def test_messages_unknown_model_400():
    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "does-not-exist",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 400
    assert "unknown model" in r.json()["detail"]


def test_chat_completions_passthrough_openai(monkeypatch):
    def fake_call(base_url, model, messages, *, max_tokens=None, temperature=None, timeout=600.0, extra=None):
        return _fake_openai_response("hi")

    monkeypatch.setattr(server_mod, "call_openai_chat", fake_call)

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-9b",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    # Passthrough preserves OpenAI shape.
    assert body["choices"][0]["message"]["content"] == "hi"
    assert body["object"] == "chat.completion"


def test_list_models_includes_enabled():
    client = TestClient(server_mod.app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = {entry["id"] for entry in r.json()["data"]}
    assert "qwen3.5-9b" in ids
    assert "claude-haiku-4-5" in ids
