"""Tests for Anthropic-shape tool use against the local ``openai`` backends.

Covers issue #552 — ``POST /v1/messages`` accepting ``tools`` / ``tool_choice``
and translating ``tool_use`` / ``tool_result`` content blocks against
llama-server's OpenAI function-calling shape, buffered and streaming:

- Request translation: Anthropic ``tools`` → OpenAI ``tools``, ``tool_choice``
  keywords, ``tool_use`` blocks → ``tool_calls``, and the message-count change
  where one user turn of ``tool_result`` blocks becomes N ``tool`` messages.
- Buffered response: ``tool_calls`` → ``tool_use`` blocks with
  ``stop_reason: "tool_use"``, and 502 on arguments that don't parse.
- Streaming: fragments accumulated per ``tool_calls[].index`` and replayed as
  ``content_block_start`` / ``input_json_delta`` / ``content_block_stop``.
- The full two-leg round trip, asserting what the hub actually sent upstream.
- Loud 400s: tools on a CLI backend, and malformed tool definitions.
- Text-only requests unchanged in both modes (no #550 regression).
"""

from __future__ import annotations

import json
import os
from typing import Iterator, List

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

import pytest
from fastapi.testclient import TestClient

from src import chat_translation as chat_mod
from src import server as server_mod
from src import server_common as server_common_mod
from src.chat_translation import (
    AnthropicStreamState,
    iter_openai_anthropic_sse,
)
from src.openai_upstream import (
    UpstreamError,
    anthropic_to_openai_messages,
    anthropic_tool_choice_to_openai,
    anthropic_tools_to_openai,
    openai_to_anthropic_envelope,
)

LOCAL_MODEL = "qwen3.5-4b"

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Look up the weather",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


# ---- request translation ----

def test_tools_translate_to_openai_functions():
    assert anthropic_tools_to_openai([WEATHER_TOOL]) == [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": WEATHER_TOOL["input_schema"],
            "description": "Look up the weather",
        },
    }]


def test_tool_without_input_schema_is_rejected():
    # Anthropic's server-side tools ({"type": "web_search_20250305"}) carry no
    # input_schema and have no local equivalent.
    with pytest.raises(ValueError, match="input_schema"):
        anthropic_tools_to_openai([{"type": "web_search_20250305", "name": "web_search"}])


def test_tool_without_name_is_rejected():
    with pytest.raises(ValueError, match="name"):
        anthropic_tools_to_openai([{"input_schema": {"type": "object"}}])


@pytest.mark.parametrize(
    "choice, expected",
    [
        (None, None),
        ({"type": "auto"}, "auto"),
        ({"type": "any"}, "required"),
        ({"type": "none"}, "none"),
        ("auto", "auto"),
        (
            {"type": "tool", "name": "get_weather"},
            {"type": "function", "function": {"name": "get_weather"}},
        ),
    ],
)
def test_tool_choice_translation(choice, expected):
    assert anthropic_tool_choice_to_openai(choice) == expected


def test_tool_choice_named_tool_needs_a_name():
    with pytest.raises(ValueError, match="requires a 'name'"):
        anthropic_tool_choice_to_openai({"type": "tool"})


def test_tool_choice_unknown_type_is_rejected():
    with pytest.raises(ValueError, match="unsupported tool_choice"):
        anthropic_tool_choice_to_openai({"type": "telepathy"})


def test_tool_use_block_becomes_openai_tool_calls():
    messages = anthropic_to_openai_messages(
        [
            {"role": "user", "content": "weather in Brussels?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Checking."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "Brussels"},
                    },
                ],
            },
        ],
        None,
    )
    assert messages[1] == {
        "role": "assistant",
        "content": "Checking.",
        "tool_calls": [{
            "id": "toolu_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"city": "Brussels"}, ensure_ascii=False),
            },
        }],
    }


def test_tool_results_become_their_own_tool_messages():
    # Anthropic puts tool results on a *user* turn; OpenAI models each as its
    # own `tool` message. Two results in one turn therefore expand to two.
    messages = anthropic_to_openai_messages(
        [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "17C"},
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "rain"},
            ],
        }],
        None,
    )
    assert messages == [
        {"role": "tool", "tool_call_id": "toolu_1", "content": "17C"},
        {"role": "tool", "tool_call_id": "toolu_2", "content": "rain"},
    ]


def test_tool_result_only_turn_adds_no_empty_user_message():
    messages = anthropic_to_openai_messages(
        [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "17C"},
            ],
        }],
        None,
    )
    assert [m["role"] for m in messages] == ["tool"]


def test_tool_results_precede_user_text_from_the_same_turn():
    # An OpenAI `tool` message must directly follow the assistant turn that
    # called it, so results are emitted ahead of any text sharing the turn.
    messages = anthropic_to_openai_messages(
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": "and tomorrow?"},
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "17C"},
            ],
        }],
        None,
    )
    assert messages == [
        {"role": "tool", "tool_call_id": "toolu_1", "content": "17C"},
        {"role": "user", "content": "and tomorrow?"},
    ]


def test_tool_result_block_list_is_flattened_to_text():
    messages = anthropic_to_openai_messages(
        [{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": [
                    {"type": "text", "text": "17C"},
                    {"type": "image", "source": {"type": "base64", "data": "x"}},
                ],
            }],
        }],
        None,
    )
    # The image is named rather than dropped silently — this backend is
    # text-only and the caller should be able to see that in the transcript.
    assert messages[0]["content"] == "17C\n[image omitted: this backend is text-only]"


def test_text_only_messages_translate_unchanged():
    assert anthropic_to_openai_messages(
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "be terse",
    ) == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]


# ---- buffered response translation ----

def _tool_call_response(arguments: str = '{"city": "Brussels"}') -> dict:
    return {
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": arguments},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 11, "completion_tokens": 5},
    }


def test_envelope_carries_tool_use_blocks():
    envelope = openai_to_anthropic_envelope(_tool_call_response())
    assert envelope["content"] == [{
        "type": "tool_use",
        "id": "call_abc",
        "name": "get_weather",
        "input": {"city": "Brussels"},
    }]
    assert envelope["stop_reason"] == "tool_use"
    assert envelope["usage"]["input_tokens"] == 11


def test_envelope_keeps_text_alongside_a_tool_call():
    response = _tool_call_response()
    response["choices"][0]["message"]["content"] = "<think>hm</think>Looking it up."
    envelope = openai_to_anthropic_envelope(response)
    assert [block["type"] for block in envelope["content"]] == ["text", "tool_use"]
    # Think-stripping still applies to the text half.
    assert envelope["content"][0]["text"] == "Looking it up."
    assert envelope["result"] == "Looking it up."


def test_tool_use_wins_over_a_stop_finish_reason():
    # llama-server does not always set finish_reason="tool_calls".
    response = _tool_call_response()
    response["choices"][0]["finish_reason"] = "stop"
    assert openai_to_anthropic_envelope(response)["stop_reason"] == "tool_use"


def test_unparseable_tool_arguments_raise_rather_than_defaulting():
    # Degrading to {} would hand the caller a plausible tool call to execute.
    with pytest.raises(UpstreamError, match="unparseable arguments"):
        openai_to_anthropic_envelope(_tool_call_response(arguments='{"city": "Brus'))


def test_empty_tool_arguments_are_an_empty_object():
    envelope = openai_to_anthropic_envelope(_tool_call_response(arguments=""))
    assert envelope["content"][0]["input"] == {}


def test_text_only_envelope_is_a_single_text_block():
    envelope = openai_to_anthropic_envelope({
        "choices": [{"index": 0, "message": {"content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
    })
    assert envelope["content"] == [{"type": "text", "text": "hello"}]
    assert envelope["stop_reason"] == "end_turn"


# ---- buffered HTTP route ----

def test_messages_buffered_tool_call_round_trip(monkeypatch):
    """Both legs of a real round trip, asserting what reached the upstream."""
    sent: List[dict] = []
    replies = [
        _tool_call_response(),
        {
            "choices": [{
                "index": 0,
                "message": {"content": "It is 17C in Brussels."},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 6},
        },
    ]

    def fake_call(base_url, model, messages, *, max_tokens=None, temperature=None,
                  timeout=600.0, extra=None, headers=None):
        sent.append({"messages": messages, "extra": extra})
        return replies[len(sent) - 1]

    monkeypatch.setattr(chat_mod, "call_openai_chat", fake_call)
    monkeypatch.setattr(
        server_common_mod, "ensure_backend_ready_or_503", lambda model: None
    )
    client = TestClient(server_mod.app)

    first = client.post("/v1/messages", json={
        "model": LOCAL_MODEL,
        "max_tokens": 64,
        "tools": [WEATHER_TOOL],
        "tool_choice": {"type": "auto"},
        "messages": [{"role": "user", "content": "weather in Brussels?"}],
    })
    assert first.status_code == 200
    body = first.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"] == [{
        "type": "tool_use",
        "id": "call_abc",
        "name": "get_weather",
        "input": {"city": "Brussels"},
    }]
    # Tool definitions actually reached llama-server.
    assert sent[0]["extra"]["tool_choice"] == "auto"
    assert sent[0]["extra"]["tools"][0]["function"]["name"] == "get_weather"

    # Second leg: hand the tool_use block back with its result.
    second = client.post("/v1/messages", json={
        "model": LOCAL_MODEL,
        "max_tokens": 64,
        "tools": [WEATHER_TOOL],
        "messages": [
            {"role": "user", "content": "weather in Brussels?"},
            {"role": "assistant", "content": body["content"]},
            {"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": "call_abc",
                "content": "17C, clear",
            }]},
        ],
    })
    assert second.status_code == 200
    assert second.json()["content"] == [
        {"type": "text", "text": "It is 17C in Brussels."}
    ]
    assert second.json()["stop_reason"] == "end_turn"
    # The upstream saw a well-formed OpenAI tool conversation: the assistant
    # turn with tool_calls, then the matching tool message.
    assert [m["role"] for m in sent[1]["messages"]] == ["user", "assistant", "tool"]
    assert sent[1]["messages"][1]["tool_calls"][0]["id"] == "call_abc"
    assert sent[1]["messages"][2]["tool_call_id"] == "call_abc"


def test_messages_buffered_without_tools_sends_no_tool_params(monkeypatch):
    sent: List[dict] = []

    def fake_call(base_url, model, messages, *, max_tokens=None, temperature=None,
                  timeout=600.0, extra=None, headers=None):
        sent.append({"extra": extra})
        return {
            "choices": [{"index": 0, "message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    monkeypatch.setattr(chat_mod, "call_openai_chat", fake_call)
    monkeypatch.setattr(
        server_common_mod, "ensure_backend_ready_or_503", lambda model: None
    )
    client = TestClient(server_mod.app)
    response = client.post("/v1/messages", json={
        "model": LOCAL_MODEL,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": "hi"}]
    assert sent[0]["extra"] is None


def test_malformed_tool_definition_is_a_400():
    # Validated before the on-demand spin-up, so no backend patch is needed.
    client = TestClient(server_mod.app)
    response = client.post("/v1/messages", json={
        "model": LOCAL_MODEL,
        "tools": [{"name": "broken"}],
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert response.status_code == 400
    assert "input_schema" in response.json()["error"]["message"]


# ---- CLI-backend refusal ----

@pytest.mark.parametrize("model", ["claude_haiku", "gemini_pro"])
@pytest.mark.parametrize("stream", [False, True])
def test_tools_on_a_cli_backend_are_refused(model, stream):
    client = TestClient(server_mod.app)
    response = client.post("/v1/messages", json={
        "model": model,
        "stream": stream,
        "tools": [WEATHER_TOOL],
        "messages": [{"role": "user", "content": "weather?"}],
    })
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert "'tools' parameter" in message
    assert "openai-backend" in message


def test_tool_blocks_on_a_cli_backend_are_refused():
    # No `tools` parameter, but the transcript still carries tool blocks the
    # flatten-to-one-prompt dispatch would drop on the floor.
    client = TestClient(server_mod.app)
    response = client.post("/v1/messages", json={
        "model": "claude_haiku",
        "messages": [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}},
            ]},
        ],
    })
    assert response.status_code == 400
    assert "tool_use / tool_result" in response.json()["error"]["message"]


def test_plain_text_on_a_cli_backend_still_works(monkeypatch):
    monkeypatch.setattr(
        server_mod, "_run_claude_backend",
        lambda model, req: {"result": "hi", "stop_reason": "end_turn", "usage": {}},
    )
    client = TestClient(server_mod.app)
    response = client.post("/v1/messages", json={
        "model": "claude_haiku",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": "hi"}]


# ---- streaming translation ----

def _tool_chunk(index: int, *, call_id: str = "", name: str = "",
                arguments: str = "", finish: str | None = None) -> dict:
    call: dict = {"index": index}
    if call_id:
        call["id"] = call_id
    function: dict = {}
    if name:
        function["name"] = name
    if arguments:
        function["arguments"] = arguments
    if function:
        call["function"] = function
    return {
        "id": "x",
        "object": "chat.completion.chunk",
        "model": LOCAL_MODEL,
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": [call]},
            "finish_reason": finish,
        }],
    }


def _sse(*chunks: dict) -> Iterator[str]:
    for chunk in chunks:
        yield "data: " + json.dumps(chunk)
        yield ""
    yield "data: [DONE]"


def _events(body: str) -> List[dict]:
    return [
        json.loads(line[len("data:"):].strip())
        for line in body.splitlines()
        if line.startswith("data:")
    ]


def test_stream_tool_call_fragments_reassemble():
    state = AnthropicStreamState(LOCAL_MODEL)
    body = "".join(iter_openai_anthropic_sse(
        _sse(
            _tool_chunk(0, call_id="call_1", name="get_weather", arguments='{"ci'),
            _tool_chunk(0, arguments='ty": "Bru'),
            _tool_chunk(0, arguments='ssels"}', finish="tool_calls"),
        ),
        state,
    ))
    events = _events(body)
    assert [event["type"] for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    start = events[1]
    assert start["index"] == 0
    assert start["content_block"] == {
        "type": "tool_use", "id": "call_1", "name": "get_weather", "input": {},
    }
    delta = events[2]["delta"]
    assert delta["type"] == "input_json_delta"
    # The accumulated JSON is what the SDK will parse into `input`.
    assert json.loads(delta["partial_json"]) == {"city": "Brussels"}
    assert events[-2]["delta"]["stop_reason"] == "tool_use"
    assert state.stop_reason == "tool_use"


def test_stream_text_then_tool_call_uses_separate_block_indices():
    state = AnthropicStreamState(LOCAL_MODEL)
    text_chunk = {
        "id": "x", "object": "chat.completion.chunk", "model": LOCAL_MODEL,
        "choices": [{"index": 0, "delta": {"content": "Checking."}, "finish_reason": None}],
    }
    body = "".join(iter_openai_anthropic_sse(
        _sse(
            text_chunk,
            _tool_chunk(0, call_id="call_1", name="get_weather",
                        arguments='{"city": "Brussels"}', finish="tool_calls"),
        ),
        state,
    ))
    events = _events(body)
    assert [event["type"] for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[1]["content_block"]["type"] == "text"
    assert events[1]["index"] == 0
    assert events[4]["content_block"]["type"] == "tool_use"
    assert events[4]["index"] == 1
    # Every block is closed before the next opens, and none is reused.
    stops = [event["index"] for event in events if event["type"] == "content_block_stop"]
    assert stops == [0, 1]
    assert state.text == "Checking."


def test_stream_parallel_tool_calls_become_sequential_blocks():
    state = AnthropicStreamState(LOCAL_MODEL)
    body = "".join(iter_openai_anthropic_sse(
        _sse(
            _tool_chunk(0, call_id="call_1", name="get_weather", arguments='{"city": "A"}'),
            _tool_chunk(1, call_id="call_2", name="get_weather", arguments='{"city": "B"}'),
        ),
        state,
    ))
    events = _events(body)
    starts = [e for e in events if e["type"] == "content_block_start"]
    assert [s["index"] for s in starts] == [0, 1]
    assert [s["content_block"]["id"] for s in starts] == ["call_1", "call_2"]


def test_stream_nameless_tool_call_is_dropped_and_logged(caplog):
    state = AnthropicStreamState(LOCAL_MODEL)
    with caplog.at_level("WARNING"):
        body = "".join(iter_openai_anthropic_sse(
            _sse(_tool_chunk(0, call_id="call_1", arguments='{"city": "A"}')),
            state,
        ))
    events = _events(body)
    assert not [e for e in events if e["type"] == "content_block_start"
                and e["content_block"]["type"] == "tool_use"]
    assert "no function name" in caplog.text


def test_stream_reports_end_turn_when_every_tool_call_was_unusable(caplog):
    # Degenerate upstream: finish_reason says tool_calls, but the only call
    # has no name. "tool_use" with no tool_use block is not a valid Anthropic
    # message and would send an agent loop hunting for a call that isn't there.
    state = AnthropicStreamState(LOCAL_MODEL)
    with caplog.at_level("WARNING"):
        body = "".join(iter_openai_anthropic_sse(
            _sse(_tool_chunk(0, call_id="call_1", arguments="{}", finish="tool_calls")),
            state,
        ))
    events = _events(body)
    assert not [e for e in events if e["type"] == "content_block_start"
                and e["content_block"]["type"] == "tool_use"]
    assert state.stop_reason == "end_turn"
    assert events[-2]["delta"]["stop_reason"] == "end_turn"
    assert "no usable tool call" in caplog.text


def test_stream_text_only_sequence_is_unchanged():
    # Guards the #550 streaming contract against the block-index rework.
    state = AnthropicStreamState(LOCAL_MODEL)
    chunks = [{
        "id": "x", "object": "chat.completion.chunk", "model": LOCAL_MODEL,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    } for text in ("Hello ", "world")]
    body = "".join(iter_openai_anthropic_sse(_sse(*chunks), state))
    events = _events(body)
    assert [event["type"] for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert all(
        event["index"] == 0
        for event in events
        if event["type"].startswith("content_block")
    )
    assert state.stop_reason == "end_turn"


# ---- streaming HTTP route ----

def test_messages_stream_tool_call_end_to_end(monkeypatch):
    captured: dict = {}

    def fake_stream(base_url, model, messages, *, max_tokens=None, temperature=None,
                    timeout=600.0, extra=None, headers=None) -> Iterator[str]:
        captured["extra"] = extra
        captured["messages"] = messages
        yield from _sse(
            _tool_chunk(0, call_id="call_1", name="get_weather", arguments='{"city"'),
            _tool_chunk(0, arguments=': "Brussels"}', finish="tool_calls"),
        )

    monkeypatch.setattr(server_mod, "_ensure_backend_ready", lambda model: None)
    monkeypatch.setattr(server_mod, "call_openai_chat_stream", fake_stream)
    client = TestClient(server_mod.app)
    with client.stream("POST", "/v1/messages", json={
        "model": LOCAL_MODEL,
        "stream": True,
        "tools": [WEATHER_TOOL],
        "tool_choice": {"type": "any"},
        "messages": [{"role": "user", "content": "weather in Brussels?"}],
    }) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    events = _events(body)
    start = next(e for e in events if e["type"] == "content_block_start")
    assert start["content_block"]["name"] == "get_weather"
    payload = "".join(
        e["delta"]["partial_json"]
        for e in events
        if e["type"] == "content_block_delta"
    )
    assert json.loads(payload) == {"city": "Brussels"}
    assert events[-2]["delta"]["stop_reason"] == "tool_use"
    # Tool params rode alongside the streaming options rather than replacing them.
    assert captured["extra"]["tool_choice"] == "required"
    assert captured["extra"]["stream_options"] == {"include_usage": True}
    assert captured["extra"]["tools"][0]["function"]["name"] == "get_weather"
