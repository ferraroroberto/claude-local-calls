"""Model-name → backend routing in src.server.

Exercises the /v1/messages path for an openai-backed model (dispatched to
mocked `call_openai_chat`) plus the unknown-model 400 branch. Claude
routing is already covered by test_server.py.
"""

from __future__ import annotations

import os

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

from fastapi.testclient import TestClient

from src import chat_translation as chat_translation_mod
from src import server as server_mod


def _fake_openai_response(text: str = "pong"):
    return {
        "id": "chatcmpl-xyz",
        "object": "chat.completion",
        "model": "qwen3.5-4b",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


def test_messages_routes_openai_backend(monkeypatch):
    captured = {}

    def fake_call(base_url, model, messages, *, max_tokens=None, temperature=None, timeout=600.0, extra=None, headers=None):
        captured["base_url"] = base_url
        captured["model"] = model
        captured["messages"] = messages
        captured["max_tokens"] = max_tokens
        return _fake_openai_response("pong")

    monkeypatch.setattr(chat_translation_mod, "call_openai_chat", fake_call)

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "qwen3.5-4b",
            "max_tokens": 64,
            "system": "Answer briefly.",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["content"] == [{"type": "text", "text": "pong"}]
    assert body["model"] == "qwen3.5-4b"
    assert body["stop_reason"] == "end_turn"

    # System prompt was prepended to the OpenAI messages.
    assert captured["model"] == "qwen3.5-4b"
    assert captured["messages"][0] == {"role": "system", "content": "Answer briefly."}
    assert captured["messages"][1]["role"] == "user"
    assert "127.0.0.1:8088" in captured["base_url"]


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
    # Anthropic error envelope on the Anthropic-shape route (#460).
    assert "unknown model" in r.json()["error"]["message"]


def test_chat_completions_passthrough_openai(monkeypatch):
    def fake_call(base_url, model, messages, *, max_tokens=None, temperature=None, timeout=600.0, extra=None, headers=None):
        return _fake_openai_response("hi")

    monkeypatch.setattr(server_mod, "call_openai_chat", fake_call)

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-4b",
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
    assert "qwen3.5-4b" in ids
    # Claude rows + their stable aliases.
    assert "claude-haiku-4-5" in ids
    assert "claude_haiku" in ids
    assert "claude_sonnet" in ids
    assert "claude_opus" in ids
    # Gemini subscription path is always enabled, like Claude.
    assert "Gemini 3.1 Pro" in ids
    assert "gemini_pro" in ids
    assert "gemini_flash" in ids
    assert "gemini_lite" in ids


def test_messages_routes_gemini_backend(monkeypatch):
    captured = {}

    def fake_call(prompt, *, model=None, system=None, attachments=None, timeout=600.0):
        captured["prompt"] = prompt
        captured["model"] = model
        captured["system"] = system
        return {
            "type": "result",
            "is_error": False,
            "result": "g-pong",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    monkeypatch.setattr(chat_translation_mod, "call_gemini", fake_call)

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "Gemini 3.1 Pro",
            "max_tokens": 64,
            "system": "Answer briefly.",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["content"] == [{"type": "text", "text": "g-pong"}]
    assert body["model"] == "Gemini 3.1 Pro"
    assert captured["model"] == "Gemini 3.1 Pro"
    assert captured["system"] == "Answer briefly."


def test_messages_routes_gemini_alias(monkeypatch):
    """`gemini_pro` alias resolves to display_name `Gemini 3.1 Pro`."""
    captured = {}

    def fake_call(prompt, *, model=None, system=None, attachments=None, timeout=600.0):
        captured["model"] = model
        return {
            "type": "result", "is_error": False, "result": "ok",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    monkeypatch.setattr(chat_translation_mod, "call_gemini", fake_call)

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "gemini_pro",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    # Alias → underlying display_name handed to the CLI.
    assert captured["model"] == "Gemini 3.1 Pro"


def test_messages_routes_claude_alias(monkeypatch):
    """`claude_sonnet` alias resolves to display_name `claude-sonnet-4-6`."""
    captured = {}

    def fake_call(prompt, *, model=None, system=None, attachments=None, timeout=600.0):
        captured["model"] = model
        return {
            "type": "result", "is_error": False, "result": "ok",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    monkeypatch.setattr(chat_translation_mod, "call_claude", fake_call)

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "claude_sonnet",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    # Alias → CLI receives the real model version, not the alias.
    assert captured["model"] == "claude-sonnet-4-6"


def test_chat_completions_routes_gemini(monkeypatch):
    def fake_call(prompt, *, model=None, system=None, attachments=None, timeout=600.0):
        return {
            "type": "result", "is_error": False, "result": "chat-ok",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    monkeypatch.setattr(server_mod, "call_gemini", fake_call)

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "Gemini 3.6 Flash",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "chat-ok"
    assert body["object"] == "chat.completion"


def test_chat_completions_multi_turn_flattens_same_as_messages(monkeypatch):
    """issue #195: /v1/chat/completions and /v1/messages must produce the
    same prompt shape for the same conversation — they now share
    _flatten_messages via _openai_messages_to_anthropic instead of each
    hand-rolling its own scaffold."""
    captured = {}

    def fake_call(prompt, *, model=None, system=None, attachments=None, timeout=600.0):
        captured["prompt"] = prompt
        captured["system"] = system
        return {
            "type": "result", "is_error": False, "result": "ok",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    monkeypatch.setattr(server_mod, "call_claude", fake_call)

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-haiku-4-5",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "how are you"},
            ],
        },
    )
    assert r.status_code == 200, r.text
    assert captured["system"] == "Be brief."
    assert "Previous conversation:" in captured["prompt"]
    assert "User: hi" in captured["prompt"]
    assert "Assistant: hello" in captured["prompt"]
    assert "how are you" in captured["prompt"]


def test_chat_completions_single_turn_unwraps_like_messages(monkeypatch):
    """Single-turn stays a bare prompt (no framing) on both routes."""
    captured = {}

    def fake_call(prompt, *, model=None, system=None, attachments=None, timeout=600.0):
        captured["prompt"] = prompt
        return {
            "type": "result", "is_error": False, "result": "ok",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    monkeypatch.setattr(server_mod, "call_claude", fake_call)

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "Capital of France?"}],
        },
    )
    assert r.status_code == 200, r.text
    assert captured["prompt"] == "Capital of France?"


def test_chat_completions_refuses_image_url_instead_of_dropping_it(monkeypatch):
    """issue #474: an OpenAI-shape vision message routed to a claude/gemini
    backend used to have its image part silently dropped and be answered as if
    the text were the whole question. It must now refuse with an explicit 400,
    matching the text-only guard the Anthropic-shape route already applies."""
    called = {"n": 0}

    def fake_call(prompt, *, model=None, system=None, attachments=None, timeout=600.0):
        called["n"] += 1
        return {
            "type": "result", "is_error": False, "result": "ok",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    monkeypatch.setattr(server_mod, "call_claude", fake_call)

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-haiku-4-5",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "text", "text": "What is in this image?"},
                ],
            }],
        },
    )
    assert r.status_code == 400, r.text
    assert "image_url" in r.json()["detail"]
    assert "/v1/messages" in r.json()["detail"]
    # The backend was never reached — no silent partial answer.
    assert called["n"] == 0


def test_chat_completions_still_accepts_all_text_parts(monkeypatch):
    """The refusal is scoped to non-text parts: an all-text list still joins."""
    captured = {}

    def fake_call(prompt, *, model=None, system=None, attachments=None, timeout=600.0):
        captured["prompt"] = prompt
        return {
            "type": "result", "is_error": False, "result": "ok",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    monkeypatch.setattr(server_mod, "call_claude", fake_call)

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-haiku-4-5",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Capital of France?"},
                    {"type": "text", "text": "One word."},
                ],
            }],
        },
    )
    assert r.status_code == 200, r.text
    assert captured["prompt"] == "Capital of France?\nOne word."
