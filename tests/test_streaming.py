"""Tests for SSE streaming and ``<think>`` stripping in both API shapes.

Covers:

- ``ThinkStripper`` over chunked input (tag straddles boundary).
- ``strip_think_blocks`` on a complete string.
- ``clean_openai_response`` (non-stream): strips think tags and folds
  ``reasoning_content`` into empty ``content``.
- ``iter_cleaned_sse``: full SSE filter pipeline yields cleaned
  ``data:`` frames and passes through ``[DONE]`` / blank lines.
- ``/v1/chat/completions`` with ``stream=true``: returns SSE
  (``text/event-stream``) and proxies cleaned chunks.
- ``/v1/chat/completions`` non-stream: response has think blocks
  removed.
- ``/v1/messages`` with ``stream=true``: Claude CLI and llama-server events
  become ordered, text-only Anthropic SSE without changing buffered calls.
"""

from __future__ import annotations

import asyncio
import json
import os
from io import StringIO
from typing import Iterator, List

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

import pytest
from fastapi.testclient import TestClient

from src import openai_upstream as upstream_mod
from src import claude_cli as claude_cli_mod
from src import server as server_mod
from src.chat_translation import AnthropicStreamState, iter_claude_anthropic_sse
from src.hub_observability import OBS
from src.openai_upstream import (
    ThinkStripper,
    clean_openai_response,
    iter_cleaned_sse,
    strip_think_blocks,
)


# ---- ThinkStripper unit tests ----

def test_strip_think_blocks_complete_string():
    src = "Before <think>secret</think>after"
    assert strip_think_blocks(src) == "Before after"


def test_strip_think_blocks_multiline():
    src = "x<think>\nlots\nof\nthink\n</think>y"
    assert strip_think_blocks(src) == "xy"


def test_strip_think_blocks_unterminated_drops_tail():
    # Truncated mid-thought (e.g. max_tokens cut it off): no close tag
    # to match, so the naive regex would leave the raw block in place.
    src = "answer is...<think>still thinking when stream died"
    assert strip_think_blocks(src) == "answer is..."


def test_strip_think_blocks_unterminated_only_think():
    src = "<think>never got to an answer"
    assert strip_think_blocks(src) == ""


def test_strip_think_blocks_well_formed_then_unterminated():
    src = "<think>done</think>Final answer<think>more thinking cut off"
    assert strip_think_blocks(src) == "Final answer"


def test_think_stripper_split_open_tag():
    s = ThinkStripper()
    out1 = s.feed("hello <thi")
    out2 = s.feed("nk>secret</think>world")
    out3 = s.flush()
    assert out1 + out2 + out3 == "hello world"


def test_think_stripper_split_close_tag():
    s = ThinkStripper()
    out1 = s.feed("a<think>thinking ab")
    out2 = s.feed("out it</thi")
    out3 = s.feed("nk>b")
    out4 = s.flush()
    assert out1 + out2 + out3 + out4 == "ab"


def test_think_stripper_no_tags_pass_through():
    s = ThinkStripper()
    parts = ["he", "ll", "o ", "wor", "ld"]
    out = "".join(s.feed(p) for p in parts) + s.flush()
    assert out == "hello world"


def test_think_stripper_unterminated_drops_tail():
    # Stream cut off mid-thinking: nothing to recover.
    s = ThinkStripper()
    out = s.feed("answer is...<think>still thinking when stream died")
    assert out == "answer is..."
    assert s.flush() == ""


# ---- clean_openai_response (non-stream) ----

def _resp(content: str = "", reasoning_content: str = "") -> dict:
    msg = {"role": "assistant", "content": content}
    if reasoning_content:
        msg["reasoning_content"] = reasoning_content
    return {
        "id": "x", "object": "chat.completion", "model": "qwen3.5-4b",
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_clean_openai_response_strips_think():
    r = _resp(content="<think>plan</think>The answer is 4.")
    clean_openai_response(r)
    assert r["choices"][0]["message"]["content"] == "The answer is 4."


def test_clean_openai_response_folds_reasoning_when_content_empty():
    r = _resp(content="", reasoning_content="The answer is 4.")
    clean_openai_response(r)
    assert r["choices"][0]["message"]["content"] == "The answer is 4."


def test_clean_openai_response_prefers_content_when_present():
    r = _resp(content="Direct answer", reasoning_content="Long reasoning")
    clean_openai_response(r)
    # Don't clobber a real content field with reasoning.
    assert r["choices"][0]["message"]["content"] == "Direct answer"


def test_clean_openai_response_drops_unterminated_think_on_truncation():
    r = _resp(content="<think>never finished reasoning")
    r["choices"][0]["finish_reason"] = "length"
    clean_openai_response(r)
    # A truncated response should degrade to empty content, not leak
    # raw <think> internals to the caller.
    assert r["choices"][0]["message"]["content"] == ""
    assert r["choices"][0]["finish_reason"] == "length"


# ---- iter_cleaned_sse pipeline ----

def _sse_lines(*chunks: dict) -> List[str]:
    lines: List[str] = []
    for c in chunks:
        lines.append("data: " + json.dumps(c))
        lines.append("")  # SSE record terminator
    lines.append("data: [DONE]")
    return lines


def _delta(content: str = "", reasoning_content: str = "") -> dict:
    delta = {}
    if content:
        delta["content"] = content
    if reasoning_content:
        delta["reasoning_content"] = reasoning_content
    return {
        "id": "x", "object": "chat.completion.chunk", "model": "qwen3.5-4b",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }


def test_iter_cleaned_sse_strips_think_across_chunks():
    raw = _sse_lines(
        _delta("Hello <thi"),
        _delta("nk>secret</think>world"),
        _delta(" friend"),
    )
    cleaned: List[str] = list(iter_cleaned_sse(iter(raw)))
    # Reassemble the content deltas the client would see.
    seen_content = ""
    for line in cleaned:
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            continue
        obj = json.loads(payload)
        delta = obj["choices"][0].get("delta", {})
        seen_content += delta.get("content") or ""
    assert seen_content == "Hello world friend"


def test_iter_cleaned_sse_passes_done_unchanged():
    raw = ["data: [DONE]"]
    out = list(iter_cleaned_sse(iter(raw)))
    assert out == ["data: [DONE]"]


def test_iter_cleaned_sse_passes_blank_lines():
    raw = ["", "data: [DONE]"]
    out = list(iter_cleaned_sse(iter(raw)))
    assert out == ["", "data: [DONE]"]


def test_iter_cleaned_sse_flushes_trailing_tag_lookalike_before_done():
    """#529: a stream whose last content delta ends near a ``<`` leaves
    ``ThinkStripper`` holding back that tail as a possible split open-tag.
    Without a flush before ``[DONE]``, that tail is silently dropped."""
    raw = _sse_lines(
        _delta("Answer: done."),
        _delta("</p>"),
    )
    cleaned: List[str] = list(iter_cleaned_sse(iter(raw)))
    assert cleaned[-1] == "data: [DONE]"

    seen_content = ""
    for line in cleaned:
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            continue
        obj = json.loads(payload)
        delta = obj["choices"][0].get("delta", {})
        seen_content += delta.get("content") or ""
    assert seen_content == "Answer: done.</p>"


# ---- Anthropic Messages SSE ----

def _anthropic_data(body: str) -> List[dict]:
    return [
        json.loads(line[len("data:"):].strip())
        for line in body.splitlines()
        if line.startswith("data:")
    ]


def _claude_stream_records(*texts: str) -> Iterator[dict]:
    yield {
        "type": "stream_event",
        "event": {
            "type": "message_start",
            "message": {
                "id": "msg_cli",
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 3,
                    "cache_creation_input_tokens": 4,
                },
            },
        },
    }
    yield {
        "type": "stream_event",
        "event": {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "private"},
        },
    }
    for text in texts:
        yield {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": text},
            },
        }
    yield {
        "type": "stream_event",
        "event": {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 7},
        },
    }
    yield {"type": "stream_event", "event": {"type": "message_stop"}}


def test_claude_stream_filters_wrapper_and_implicit_thinking():
    state = AnthropicStreamState("claude_haiku")
    body = "".join(iter_claude_anthropic_sse(
        _claude_stream_records("Hello ", "world"), state,
    ))
    events = _anthropic_data(body)

    assert [event["type"] for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[0]["message"]["id"] == "msg_cli"
    assert events[0]["message"]["model"] == "claude_haiku"
    assert "private" not in body
    assert state.text == "Hello world"
    assert state.input_tokens == 12
    assert state.output_tokens == 7
    assert state.cache_read_tokens == 3
    assert state.cache_write_tokens == 4


def test_messages_claude_streaming_http_contract(monkeypatch):
    captured: dict = {}

    def fake_stream(prompt, *, model=None, system=None, attachments=None,
                    timeout=600.0) -> Iterator[dict]:
        captured.update(prompt=prompt, model=model, system=system)
        yield from _claude_stream_records("Hello ", "Claude")

    monkeypatch.setattr(server_mod, "call_claude_stream", fake_stream)
    client = TestClient(server_mod.app)
    with client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "claude_haiku",
            "stream": True,
            "max_tokens": 20,
            "system": "Be terse",
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    events = _anthropic_data(body)
    deltas = [
        event["delta"]["text"]
        for event in events
        if event["type"] == "content_block_delta"
    ]
    assert deltas == ["Hello ", "Claude"]
    assert captured == {
        "prompt": "hi",
        "model": "claude-haiku-4-5",
        "system": "Be terse",
    }
    record = next(
        item for item in OBS.recent_requests() if item["model"] == "claude_haiku"
    )
    assert record["in_tok"] == 12
    assert record["out_tok"] == 7
    assert record["stop_reason"] == "end_turn"


def test_messages_openai_stream_translates_and_strips_thinking(monkeypatch):
    captured: dict = {}

    def fake_stream(base_url, model, messages, *, max_tokens=None, temperature=None,
                    timeout=600.0, extra=None, headers=None) -> Iterator[str]:
        captured["extra"] = extra
        usage = {
            "id": "x",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        yield from _sse_lines(
            _delta("<think>hidden"),
            _delta(" plan</think>Hello "),
            _delta("local"),
            usage,
        )

    monkeypatch.setattr(server_mod, "_ensure_backend_ready", lambda model: None)
    monkeypatch.setattr(server_mod, "call_openai_chat_stream", fake_stream)
    client = TestClient(server_mod.app)
    with client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "qwen3.5-4b",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    events = _anthropic_data(body)
    assert [event["type"] for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert "hidden" not in body
    assert "".join(
        event["delta"]["text"]
        for event in events
        if event["type"] == "content_block_delta"
    ) == "Hello local"
    assert events[-2]["usage"]["output_tokens"] == 3
    assert events[-2]["delta"]["stop_reason"] == "end_turn"
    assert captured["extra"] == {"stream_options": {"include_usage": True}}


def test_messages_stream_failure_is_anthropic_error_event(monkeypatch):
    def fake_stream(*args, **kwargs) -> Iterator[dict]:
        raise server_mod.ClaudeCLIError("stream exploded")
        yield  # pragma: no cover

    monkeypatch.setattr(server_mod, "call_claude_stream", fake_stream)
    client = TestClient(server_mod.app)
    with client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "claude_haiku",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert _anthropic_data(body) == [{
        "type": "error",
        "error": {"type": "api_error", "message": "stream exploded"},
    }]


def test_messages_non_streaming_remains_buffered(monkeypatch):
    monkeypatch.setattr(server_mod, "_run_claude_backend", lambda model, req: {
        "result": "buffered",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    })
    response = TestClient(server_mod.app).post(
        "/v1/messages",
        json={
            "model": "claude_haiku",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["content"] == [{"type": "text", "text": "buffered"}]


class _FakeClaudeProcess:
    def __init__(self, stdout: str) -> None:
        self.stdin = StringIO()
        self.stdout = StringIO(stdout)
        self.stderr = StringIO()
        self.return_code = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.return_code

    def wait(self, timeout=None):
        if self.return_code is None:
            self.return_code = 0
        return self.return_code

    def terminate(self):
        self.terminated = True
        self.return_code = -15

    def kill(self):
        self.killed = True
        self.return_code = -9


def test_claude_stream_disconnect_terminates_owned_process(monkeypatch):
    process = _FakeClaudeProcess('{"type":"stream_event","event":{"type":"message_start"}}\n')
    captured: dict = {}

    def fake_popen(*args, **kwargs):
        captured.update(kwargs)
        return process

    monkeypatch.setattr(claude_cli_mod.subprocess, "Popen", fake_popen)
    records = claude_cli_mod.call_claude_stream("hi")
    assert next(records)["type"] == "stream_event"
    records.close()

    assert process.terminated
    assert captured["creationflags"] == claude_cli_mod.NO_WINDOW


async def _disconnect_after_first_body(response) -> None:
    body_started = asyncio.Event()
    request_sent = False

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await body_started.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            body_started.set()
            await asyncio.Event().wait()

    await response(
        {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}},
        receive,
        send,
    )


def test_messages_http_disconnect_closes_claude_stream(monkeypatch):
    class CloseAwareRecords:
        def __init__(self) -> None:
            self.closed = False
            self.sent_start = False

        def __iter__(self):
            return self

        def __next__(self):
            if not self.sent_start:
                self.sent_start = True
                return {
                    "type": "stream_event",
                    "event": {
                        "type": "message_start",
                        "message": {"id": "msg_disconnect", "usage": {}},
                    },
                }
            return {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "more"},
                },
            }

        def close(self) -> None:
            self.closed = True

    records = CloseAwareRecords()
    monkeypatch.setattr(server_mod, "call_claude_stream", lambda *a, **k: records)
    request = server_mod.MessagesRequest(
        model="claude_haiku",
        stream=True,
        max_tokens=20,
        messages=[{"role": "user", "content": "hi"}],
    )
    response = server_mod._stream_anthropic_response(
        server_mod._resolve(request.model),
        request,
        ctx=None,
        span=None,
        client_id="test",
        start_ns=0,
    )

    asyncio.run(_disconnect_after_first_body(response))
    assert records.closed


def test_messages_http_disconnect_finishes_on_demand_stream(monkeypatch):
    class CloseAwareRaw:
        def __init__(self) -> None:
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            return 'data: {"choices":[{"delta":{"content":"more"}}]}\n\n'

        def close(self) -> None:
            self.closed = True

    class Lease:
        def __init__(self) -> None:
            self.finished = False

        def start(self):
            return self

        def finish(self) -> None:
            self.finished = True

    raw = CloseAwareRaw()
    lease = Lease()
    monkeypatch.setattr(server_mod, "_ensure_backend_ready", lambda model: None)
    monkeypatch.setattr(server_mod, "call_openai_chat_stream", lambda *a, **k: raw)
    monkeypatch.setattr(server_mod._on_demand, "tracking", lambda *a, **k: lease)
    request = server_mod.MessagesRequest(
        model="qwen3.5-4b",
        stream=True,
        max_tokens=20,
        messages=[{"role": "user", "content": "hi"}],
    )
    response = server_mod._stream_anthropic_response(
        server_mod._resolve(request.model),
        request,
        ctx=None,
        span=None,
        client_id="test",
        start_ns=0,
    )

    asyncio.run(_disconnect_after_first_body(response))
    assert raw.closed
    assert lease.finished


def test_claude_stream_timeout_has_distinct_error_and_kills(monkeypatch):
    process = _FakeClaudeProcess("")

    class ImmediateTimer:
        daemon = False

        def __init__(self, timeout, callback):
            self.callback = callback

        def start(self):
            self.callback()

        def cancel(self):
            pass

    monkeypatch.setattr(claude_cli_mod.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(claude_cli_mod.threading, "Timer", ImmediateTimer)

    with pytest.raises(claude_cli_mod.ClaudeCLIError, match="timed out after 3s"):
        list(claude_cli_mod.call_claude_stream("hi", timeout=3))
    assert process.killed


# ---- end-to-end against /v1/chat/completions ----

def test_chat_completions_strips_think_non_stream(monkeypatch):
    def fake_call(base_url, model, messages, *, max_tokens=None, temperature=None,
                  timeout=600.0, extra=None, headers=None):
        return {
            "id": "x", "object": "chat.completion", "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "<think>step 1\nstep 2</think>final answer",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
        }

    monkeypatch.setattr(server_mod, "call_openai_chat", fake_call)

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-4b",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "final answer"


def test_chat_completions_streaming_proxies_sse(monkeypatch):
    """End-to-end: stream=true returns SSE with cleaned content deltas."""

    def fake_stream(base_url, model, messages, *, max_tokens=None, temperature=None,
                    timeout=600.0, extra=None, headers=None) -> Iterator[str]:
        for line in _sse_lines(
            _delta("<think>plan"),
            _delta(" some</think>"),
            _delta("Hello "),
            _delta("world!"),
        ):
            yield line

    monkeypatch.setattr(server_mod, "call_openai_chat_stream", fake_stream)

    client = TestClient(server_mod.app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-4b",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())

    # Parse out the data frames.
    reassembled = ""
    saw_done = False
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        obj = json.loads(payload)
        delta = obj["choices"][0].get("delta", {})
        reassembled += delta.get("content") or ""
    assert reassembled == "Hello world!"
    assert saw_done


def test_chat_completions_streaming_usage_populated_from_trailing_frame(monkeypatch):
    """usage_in/usage_out must be captured from the trailing usage frame.

    llama-server (--jinja mode) emits the usage object on a final chunk that
    arrives *after* all content deltas, i.e. after first_token_ns is set.
    This test verifies the fix: usage must be non-zero even when the usage
    frame is not the first data frame.
    """

    def fake_stream(base_url, model, messages, *, max_tokens=None, temperature=None,
                    timeout=600.0, extra=None, headers=None) -> Iterator[str]:
        # Two content deltas first (these set first_token_ns), then a trailing
        # usage-only chunk (no choices/delta, just a usage field).
        content_chunks = [_delta("Hello "), _delta("world!")]
        usage_chunk = {
            "id": "x", "object": "chat.completion.chunk", "model": model,
            "choices": [],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
        }
        for line in _sse_lines(*content_chunks, usage_chunk):
            yield line

    monkeypatch.setattr(server_mod, "call_openai_chat_stream", fake_stream)

    # Patch record_genai_metrics to capture what usage values were recorded.
    recorded: dict = {}

    def fake_record(*, model, backend, route, client_id, duration_ms,
                    input_tokens=0, output_tokens=0, error_type=""):
        recorded["input_tokens"] = input_tokens
        recorded["output_tokens"] = output_tokens

    monkeypatch.setattr(server_mod, "record_genai_metrics", fake_record)

    client = TestClient(server_mod.app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-4b",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as r:
        assert r.status_code == 200
        _ = "".join(r.iter_text())

    assert recorded.get("input_tokens") == 12, f"input_tokens should be 12, got {recorded.get('input_tokens')}"
    assert recorded.get("output_tokens") == 7, f"output_tokens should be 7, got {recorded.get('output_tokens')}"


def test_chat_completions_stream_upstream_error(monkeypatch):
    def fake_stream(*args, **kwargs):
        raise upstream_mod.UpstreamError("boom")
        yield  # pragma: no cover

    monkeypatch.setattr(server_mod, "call_openai_chat_stream", fake_stream)

    client = TestClient(server_mod.app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-4b",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as r:
        assert r.status_code == 200  # SSE always opens 200
        body = "".join(r.iter_text())
    assert "boom" in body
    assert "[DONE]" in body


# ---- response_format / chat_template_kwargs passthrough (issue #159) ----

_RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "ok", "schema": {"type": "object"}}}
_TEMPLATE_KWARGS = {"enable_thinking": False}


def test_chat_completions_forwards_structured_params_non_stream(monkeypatch):
    """response_format + chat_template_kwargs reach the upstream `extra` (non-stream)."""
    captured: dict = {}

    def fake_call(base_url, model, messages, *, max_tokens=None, temperature=None,
                  timeout=600.0, extra=None, headers=None):
        captured["extra"] = extra
        return {
            "id": "x", "object": "chat.completion", "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "{}"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(server_mod, "call_openai_chat", fake_call)

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-4b",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": _RESPONSE_FORMAT,
            "chat_template_kwargs": _TEMPLATE_KWARGS,
        },
    )
    assert r.status_code == 200, r.text
    extra = captured["extra"]
    assert extra is not None
    assert extra["response_format"] == _RESPONSE_FORMAT
    assert extra["chat_template_kwargs"] == _TEMPLATE_KWARGS


def test_chat_completions_forwards_structured_params_stream(monkeypatch):
    """response_format + chat_template_kwargs reach the upstream `extra` (stream)."""
    captured: dict = {}

    def fake_stream(base_url, model, messages, *, max_tokens=None, temperature=None,
                    timeout=600.0, extra=None, headers=None) -> Iterator[str]:
        captured["extra"] = extra
        for line in _sse_lines(_delta("ok")):
            yield line

    monkeypatch.setattr(server_mod, "call_openai_chat_stream", fake_stream)

    client = TestClient(server_mod.app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-4b",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": _RESPONSE_FORMAT,
            "chat_template_kwargs": _TEMPLATE_KWARGS,
        },
    ) as r:
        assert r.status_code == 200
        _ = "".join(r.iter_text())

    extra = captured["extra"]
    assert extra is not None
    assert extra["response_format"] == _RESPONSE_FORMAT
    assert extra["chat_template_kwargs"] == _TEMPLATE_KWARGS


def test_chat_completions_omits_structured_params_when_absent(monkeypatch):
    """No response_format/chat_template_kwargs in `extra` when the client omits them."""
    captured: dict = {}

    def fake_call(base_url, model, messages, *, max_tokens=None, temperature=None,
                  timeout=600.0, extra=None, headers=None):
        captured["extra"] = extra
        return {
            "id": "x", "object": "chat.completion", "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(server_mod, "call_openai_chat", fake_call)

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-4b",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200, r.text
    # extra collapses to None when nothing optional is sent.
    assert captured["extra"] is None


# ---- no-think virtual alias injection (issue #161) ----
# These hit the real config/models.yaml (host tower set at module import),
# so they also verify the qwen35_4b_nothink wiring end-to-end.

def _capture_call(captured: dict):
    def fake_call(base_url, model, messages, *, max_tokens=None, temperature=None,
                  timeout=600.0, extra=None, headers=None):
        captured["extra"] = extra
        captured["base_url"] = base_url
        return {
            "id": "x", "object": "chat.completion", "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    return fake_call


@pytest.mark.parametrize("model", ["qwen3.5-4b-nothink", "agentic_light_nothink"])
def test_nothink_alias_injects_chat_template_kwargs(monkeypatch, model):
    """Both names for the no-think row fold enable_thinking:false into the
    upstream payload AND route to qwen's :8088 backend — no second process.

    `agentic_light_nothink` is covered explicitly because it is the string
    Home Assistant's extended_openai_conversation sends; #489 moved the bare
    `agentic_light` alias onto this same row but deliberately RETAINED this
    name, so a regression here breaks a live consumer silently.
    """
    captured: dict = {}
    monkeypatch.setattr(server_mod, "call_openai_chat", _capture_call(captured))

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200, r.text
    assert captured["extra"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert captured["base_url"] == "http://127.0.0.1:8088/v1"   # shares qwen


def test_agentic_light_defaults_to_nothink(monkeypatch):
    """`agentic_light` is the NO-THINK lane by default (#489).

    Inverts the pre-#489 contract, where the bare role alias was
    thinking-capable. The role is OpenClaw's fast lane — latency-sensitive,
    short-output work — and on #486's prompt set the thinking row ran ~3x
    slower and truncated on 17/23 at a 1024-token budget. Reasoning did not
    disappear; it moved one alias away (see the test below).
    """
    captured: dict = {}
    monkeypatch.setattr(server_mod, "call_openai_chat", _capture_call(captured))

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "agentic_light",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200, r.text
    assert captured["extra"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert captured["base_url"] == "http://127.0.0.1:8088/v1"


def test_agentic_light_think_stays_thinking_capable(monkeypatch):
    """`agentic_light_think` is the escape hatch that keeps #489 revertible.

    A consumer that genuinely wants reasoning changes one string and nothing
    else — same backend, same port, no overlay.
    """
    captured: dict = {}
    monkeypatch.setattr(server_mod, "call_openai_chat", _capture_call(captured))

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "agentic_light_think",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200, r.text
    # No tools, no inject_extra → extra collapses to None.
    assert captured["extra"] is None
    assert captured["base_url"] == "http://127.0.0.1:8088/v1"


def test_nothink_alias_caller_chat_template_kwargs_wins(monkeypatch):
    """A caller that sends its own chat_template_kwargs overrides the injected
    default (caller wins)."""
    captured: dict = {}
    monkeypatch.setattr(server_mod, "call_openai_chat", _capture_call(captured))

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-4b-nothink",
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    assert r.status_code == 200, r.text
    assert captured["extra"] == {"chat_template_kwargs": {"enable_thinking": True}}
