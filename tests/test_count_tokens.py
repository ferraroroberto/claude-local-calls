"""POST /v1/messages/count_tokens (#463).

The behaviour that matters is not "does it return a number" but "does it
tell the truth about the number":

1. llama-server backends are counted by the upstream's own tokenizer via
   ``/apply-template`` + ``/tokenize`` — ``exact: true``, no warning.
2. Subscription-CLI backends (claude/gemini) have no tokenizer, so the
   response is flagged ``exact: false`` and carries a ``warning``. An
   estimate must never be returned dressed as a measurement.
3. Errors match ``/v1/messages`` exactly — same 400 for an unknown model,
   the same non-chat-backend rejection, the same Anthropic error envelope.
"""

from __future__ import annotations

import os

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

import httpx  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src import server as server_mod  # noqa: E402
from src import token_counting as tc  # noqa: E402

CLAUDE_MODEL = "claude-haiku-4-5"
LLAMA_MODEL = "qwen3.5-4b"


def _client() -> TestClient:
    return TestClient(server_mod.app)


def _body(model: str, **extra):
    payload = {"model": model, "messages": [{"role": "user", "content": "Count to three."}]}
    payload.update(extra)
    return payload


# ---------------------------------------------------------------- fake upstream

class _FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload


class _FakeClient:
    """llama-server stand-in: records calls, answers /apply-template + /tokenize."""

    def __init__(self, *, template: bool = True, tokens: int = 23):
        self.template = template
        self.tokens = tokens
        self.calls: list[tuple[str, dict]] = []

    def post(self, url, *, json=None, headers=None, timeout=None):
        self.calls.append((url, json or {}))
        if url.endswith("/apply-template"):
            if not self.template:
                return _FakeResponse(404, text="not found")
            rendered = "<|im_start|>" + "|".join(
                str(m.get("content") or "") for m in json["messages"]
            )
            return _FakeResponse(200, {"prompt": rendered})
        if url.endswith("/tokenize"):
            return _FakeResponse(200, {"tokens": list(range(self.tokens))})
        raise AssertionError(f"unexpected upstream call: {url}")


def _patch_upstream(monkeypatch, client) -> None:
    monkeypatch.setattr(tc, "get_sync_client", lambda: client)


@pytest.fixture(autouse=True)
def _never_spawn_backends(monkeypatch):
    """count_tokens calls ensure_ready for on-demand rows; keep tests hermetic."""
    from src import on_demand

    monkeypatch.setattr(on_demand, "ensure_ready", lambda model, **kw: None)


# ----------------------------------------------------- llama-server: exact path

def test_llama_server_count_is_exact_and_uses_the_chat_template(monkeypatch):
    fake = _FakeClient(tokens=23)
    _patch_upstream(monkeypatch, fake)

    r = _client().post(
        "/v1/messages/count_tokens",
        json=_body(LLAMA_MODEL, system="You are terse."),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["input_tokens"] == 23
    assert body["exact"] is True
    assert body["method"] == tc.METHOD_LLAMA_TOKENIZER
    assert body["model"] == LLAMA_MODEL
    assert body["backend"] == "openai"
    assert "warning" not in body

    urls = [url for url, _ in fake.calls]
    assert urls == [
        "http://127.0.0.1:8088/apply-template",
        "http://127.0.0.1:8088/tokenize",
    ]
    # The system prompt reaches the template, and the rendered prompt (not the
    # raw message text) is what gets tokenized.
    rendered_messages = fake.calls[0][1]["messages"]
    assert rendered_messages[0] == {"role": "system", "content": "You are terse."}
    assert fake.calls[1][1]["content"].startswith("<|im_start|>")
    # add_special mirrors what llama-server does for a real completion.
    assert fake.calls[1][1]["add_special"] is True


def test_llama_server_without_apply_template_is_flagged_approximate(monkeypatch):
    _patch_upstream(monkeypatch, _FakeClient(template=False, tokens=7))

    r = _client().post("/v1/messages/count_tokens", json=_body(LLAMA_MODEL))
    assert r.status_code == 200
    body = r.json()
    assert body["input_tokens"] == 7
    assert body["exact"] is False
    assert body["method"] == tc.METHOD_LLAMA_TOKENIZER_UNTEMPLATED
    assert body["warning"].startswith("APPROXIMATE")


def test_unreachable_llama_server_is_a_502(monkeypatch):
    class _Dead:
        def post(self, *a, **kw):
            raise httpx.ConnectError("All connection attempts failed")

    _patch_upstream(monkeypatch, _Dead())

    r = _client().post("/v1/messages/count_tokens", json=_body(LLAMA_MODEL))
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "api_error"


# ------------------------------------------------- subscription CLI: approximate

def test_claude_count_is_returned_but_flagged_as_an_estimate():
    r = _client().post(
        "/v1/messages/count_tokens",
        json=_body(CLAUDE_MODEL, system="You are terse."),
    )
    assert r.status_code == 200
    body = r.json()
    # "You are terse." + "\n" + "Count to three." == 30 chars, /4 -> 8.
    assert body["input_tokens"] == 8
    assert body["exact"] is False
    assert body["method"] == tc.METHOD_CHAR_HEURISTIC
    assert body["backend"] == "claude"
    assert body["warning"].startswith("APPROXIMATE")
    # The warning has to say why, not just that.
    assert "claude -p" in body["warning"]


def test_image_blocks_are_called_out_as_uncounted():
    r = _client().post(
        "/v1/messages/count_tokens",
        json={
            "model": CLAUDE_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": "aGk=",
                    }},
                ],
            }],
        },
    )
    assert r.status_code == 200
    assert "image/document blocks" in r.json()["warning"]


# ------------------------------------------------------- parity with /v1/messages

def test_unknown_model_fails_like_v1_messages():
    client = _client()
    count = client.post("/v1/messages/count_tokens", json=_body("no-such-model"))
    messages = client.post(
        "/v1/messages", json=_body("no-such-model", max_tokens=16)
    )
    assert count.status_code == messages.status_code == 400
    assert count.json() == messages.json()


def test_empty_messages_is_a_400_in_the_anthropic_envelope():
    r = _client().post("/v1/messages/count_tokens", json={
        "model": CLAUDE_MODEL, "messages": [],
    })
    assert r.status_code == 400
    assert r.json() == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "messages must not be empty",
        },
    }


@pytest.mark.parametrize(
    "model_name,expected",
    [
        ("whisper-large-v3-turbo", "ASR backend"),
        ("piper-tts", "TTS backend"),
    ],
)
def test_non_chat_backend_is_rejected_the_same_way(model_name, expected):
    from src.model_registry import resolve as resolve_model

    if resolve_model(model_name) is None:
        pytest.skip(f"{model_name} not enabled on this host")
    r = _client().post("/v1/messages/count_tokens", json=_body(model_name))
    assert r.status_code == 400
    assert expected in r.json()["error"]["message"]


def test_anthropic_sdk_count_tokens_works_against_the_hub(monkeypatch):
    """The real SDK's ``messages.count_tokens``, over its own plumbing.

    ``TestClient`` is an ``httpx.Client`` subclass, so the SDK can use it as
    its transport — this exercises the SDK's request builder and response
    parser, not a hand-rolled stand-in, which is the whole point of "API
    parity". The hub's extra honesty fields must not break its model.
    """
    import anthropic

    _patch_upstream(monkeypatch, _FakeClient(tokens=23))
    sdk = anthropic.Anthropic(
        api_key="local-dummy",
        base_url="http://testserver",
        http_client=_client(),
        max_retries=0,
    )
    counted = sdk.messages.count_tokens(
        model=LLAMA_MODEL, messages=[{"role": "user", "content": "Count to three."}],
    )
    assert counted.input_tokens == 23


def test_endpoint_is_advertised_on_info():
    r = _client().get("/info")
    assert r.json()["endpoints"]["count_tokens"] == "POST /v1/messages/count_tokens"
