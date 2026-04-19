"""Unit tests for the server — no real `claude` calls.

We monkeypatch `call_claude` so the tests are fast and deterministic.
For the real end-to-end smoke test, see `scripts/smoke_test.py`.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src import server as server_mod


def _fake_envelope(text: str = "Hello from fake Claude"):
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": text,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 7},
    }


def test_health():
    client = TestClient(server_mod.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_messages_single_turn(monkeypatch):
    seen = {}

    def fake_call(prompt, *, model=None, system=None, timeout=600.0):
        seen["prompt"] = prompt
        seen["model"] = model
        seen["system"] = system
        return _fake_envelope("Paris")

    monkeypatch.setattr(server_mod, "call_claude", fake_call)

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "claude-haiku-4-5",
            "max_tokens": 64,
            "system": "Answer in one word.",
            "messages": [{"role": "user", "content": "Capital of France?"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"] == [{"type": "text", "text": "Paris"}]
    assert body["model"] == "claude-haiku-4-5"
    assert body["stop_reason"] == "end_turn"
    assert body["usage"]["output_tokens"] == 7
    assert seen["prompt"] == "Capital of France?"
    assert seen["system"] == "Answer in one word."
    assert seen["model"] == "claude-haiku-4-5"


def test_messages_multi_turn_flattens(monkeypatch):
    captured = {}

    def fake_call(prompt, *, model=None, system=None, timeout=600.0):
        captured["prompt"] = prompt
        return _fake_envelope("ok")

    monkeypatch.setattr(server_mod, "call_claude", fake_call)

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "claude-haiku-4-5",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "how are you"},
            ],
        },
    )
    assert r.status_code == 200
    assert "Previous conversation:" in captured["prompt"]
    assert "User: hi" in captured["prompt"]
    assert "Assistant: hello" in captured["prompt"]
    assert "how are you" in captured["prompt"]


def test_messages_cli_error_returns_502(monkeypatch):
    from src.claude_cli import ClaudeCLIError

    def fake_call(prompt, *, model=None, system=None, timeout=600.0):
        raise ClaudeCLIError("boom")

    monkeypatch.setattr(server_mod, "call_claude", fake_call)

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/messages",
        json={
            "model": "claude-haiku-4-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 502
    assert "boom" in r.json()["detail"]
