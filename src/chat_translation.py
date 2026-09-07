"""Chat-shape translation: request/response schemas, media-block extraction,
prompt flattening, Anthropic SSE event adapters, and per-backend dispatch
shared by the ``/v1/messages`` and ``/v1/chat/completions`` routes in
``server.py``.

Split out of ``server.py`` (issue #245) — the Pydantic schemas, the
Anthropic content-block media extractor, the multi-turn prompt flattener, and
the three per-backend dispatchers (``_run_claude_backend`` /
``_run_gemini_backend`` / ``_run_openai_backend``) were the one part of
``server.py`` that hadn't yet had the splitting treatment already applied to
the audio/images/lifecycle concerns (``server_audio_asr.py``/``server_audio_tts.py``, ``server_images.py``,
``server_lifecycle.py``). ``server.py`` keeps the endpoint handlers
themselves (including the OpenAI SSE passthrough generator) and the FastAPI
app/middleware assembly.

A leaf module with no dependency on ``server.py``'s ``app`` — mirrors
``server_common.py``'s reason for existing, so route modules (and this one)
can import each other without a circular import back into ``server.py``.
"""

from __future__ import annotations

import base64
import json
import logging
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from fastapi import HTTPException
from pydantic import BaseModel

from .claude_cli import ClaudeCLIError, call_claude
from .gemini_cli import GeminiCLIError, call_gemini
from .model_registry import Model
from .openai_upstream import (
    UpstreamError,
    anthropic_to_openai_messages,
    anthropic_tool_choice_to_openai,
    anthropic_tools_to_openai,
    call_openai_chat,
    openai_to_anthropic_envelope,
)
from .remote_proxy import remote_auth_token_for_model, remote_base_url

logger = logging.getLogger(__name__)


@dataclass
class AnthropicStreamState:
    """Mutable result accumulated while translating one Anthropic SSE stream.

    ``next_index`` / ``open_index`` / ``text_index`` track content-block
    allocation. Anthropic keeps exactly one block open at a time and numbers
    them in emission order, so indices are handed out as blocks are opened
    rather than fixed in advance — a text-only stream still gets a single
    block at index 0, unchanged from before tool support (#552).
    """

    requested_model: str
    message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:24]}")
    text_parts: List[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    stop_reason: str = "end_turn"
    message_started: bool = False
    message_delta_sent: bool = False
    message_stopped: bool = False
    next_index: int = 0
    open_index: Optional[int] = None
    text_index: Optional[int] = None
    # Upstream tool-call fragments keyed by OpenAI ``tool_calls[].index``:
    # ``{"id": str, "name": str, "args": [fragment, ...]}``.
    tool_calls: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "".join(self.text_parts)


def _anthropic_sse(event: Dict[str, Any]) -> str:
    """Encode one Anthropic event with both its SSE name and JSON body."""
    return (
        f"event: {event['type']}\n"
        f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    )


def _set_anthropic_usage(state: AnthropicStreamState, usage: Dict[str, Any]) -> None:
    state.input_tokens = max(
        state.input_tokens, int(usage.get("input_tokens", 0) or 0)
    )
    state.output_tokens = max(
        state.output_tokens, int(usage.get("output_tokens", 0) or 0)
    )
    state.cache_read_tokens = max(
        state.cache_read_tokens,
        int(usage.get("cache_read_input_tokens", 0) or 0),
    )
    state.cache_write_tokens = max(
        state.cache_write_tokens,
        int(usage.get("cache_creation_input_tokens", 0) or 0),
    )


def _start_anthropic_stream(state: AnthropicStreamState) -> List[str]:
    if state.message_started:
        return []
    state.message_started = True
    event = {
        "type": "message_start",
        "message": {
            "id": state.message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": state.requested_model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": state.input_tokens,
                "output_tokens": 0,
                "cache_creation_input_tokens": state.cache_write_tokens,
                "cache_read_input_tokens": state.cache_read_tokens,
            },
        },
    }
    return [_anthropic_sse(event)]


def _close_open_block(state: AnthropicStreamState) -> List[str]:
    """Close whichever content block is currently open, if any."""
    if state.open_index is None:
        return []
    index = state.open_index
    state.open_index = None
    return [_anthropic_sse({"type": "content_block_stop", "index": index})]


def _start_text_block(state: AnthropicStreamState) -> List[str]:
    events = _start_anthropic_stream(state)
    if state.text_index is not None and state.open_index == state.text_index:
        return events
    # Text arriving after a tool block opens a *new* text block rather than
    # reopening the old one — Anthropic allows several text blocks per
    # message, and a closed block cannot take further deltas.
    events += _close_open_block(state)
    state.text_index = state.next_index
    state.next_index += 1
    state.open_index = state.text_index
    events.append(_anthropic_sse({
        "type": "content_block_start",
        "index": state.text_index,
        "content_block": {"type": "text", "text": ""},
    }))
    return events


def _text_delta(state: AnthropicStreamState, text: str) -> List[str]:
    if not text:
        return []
    events = _start_text_block(state)
    state.text_parts.append(text)
    events.append(_anthropic_sse({
        "type": "content_block_delta",
        "index": state.text_index,
        "delta": {"type": "text_delta", "text": text},
    }))
    return events


def _emit_tool_blocks(state: AnthropicStreamState) -> List[str]:
    """Replay accumulated upstream tool calls as Anthropic ``tool_use`` blocks.

    Buffered to the end of the stream rather than forwarded live. Anthropic's
    wire format keeps exactly one content block open at a time, while OpenAI
    may interleave fragments from several ``tool_calls[].index`` values in any
    order — replaying at the end is the only translation that stays valid for
    both. Nothing incremental is lost in practice: llama-server parses tool
    calls out of the completed generation, so it has no partial arguments to
    stream in the first place.

    The accumulated fragments are emitted as a single ``input_json_delta``,
    which is a complete JSON object by construction — the caller's accumulator
    sees exactly what the buffered route would have returned.
    """
    events: List[str] = []
    emitted = 0
    for openai_index in sorted(state.tool_calls):
        entry = state.tool_calls[openai_index]
        name = entry.get("name") or ""
        partial = "".join(entry.get("args") or []) or "{}"
        if not name:
            logger.warning(
                "dropping upstream tool call at index %s: no function name "
                "(id=%r args=%r)",
                openai_index, entry.get("id"), partial[:200],
            )
            continue
        events += _close_open_block(state)
        index = state.next_index
        state.next_index += 1
        state.open_index = index
        events.append(_anthropic_sse({
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": entry.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": name,
                "input": {},
            },
        }))
        events.append(_anthropic_sse({
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial},
        }))
        emitted += 1
    if emitted:
        # Authoritative over the upstream's finish_reason, which llama-server
        # does not always set to "tool_calls" when it emits one.
        state.stop_reason = "tool_use"
    elif state.stop_reason == "tool_use":
        # The upstream reported "tool_calls" but every call was unusable and
        # got dropped above. A "tool_use" stop reason with no tool_use block
        # is not a valid Anthropic message — and it would send an agent loop
        # hunting for a call that isn't there.
        logger.warning(
            "upstream finished with tool_calls but emitted no usable tool "
            "call; reporting stop_reason=end_turn instead"
        )
        state.stop_reason = "end_turn"
    return events


def _finish_anthropic_stream(state: AnthropicStreamState) -> List[str]:
    if state.message_stopped:
        return []
    if state.next_index == 0 and not state.tool_calls:
        # Nothing was ever emitted (empty upstream response) — an Anthropic
        # message always carries at least one content block.
        events = _start_text_block(state)
    else:
        events = _start_anthropic_stream(state)
    events += _emit_tool_blocks(state)
    events += _close_open_block(state)
    if not state.message_delta_sent:
        events.append(_anthropic_sse({
            "type": "message_delta",
            "delta": {
                "stop_reason": state.stop_reason,
                "stop_sequence": None,
            },
            "usage": {"output_tokens": state.output_tokens},
        }))
        state.message_delta_sent = True
    events.append(_anthropic_sse({"type": "message_stop"}))
    state.message_stopped = True
    return events


def iter_claude_anthropic_sse(
    records: Iterator[Dict[str, Any]],
    state: AnthropicStreamState,
) -> Iterator[str]:
    """Filter Claude Code JSONL records into a text-only Anthropic stream.

    Claude Code's wrapper emits native Anthropic events nested under
    ``stream_event`` plus its own lifecycle records. Its current models may
    also emit implicit thinking blocks. Those are intentionally filtered here:
    explicit extended-thinking support is a separate API contract, while this
    route has historically exposed only final assistant text.
    """
    for record in records:
        if record.get("type") == "result":
            if record.get("is_error"):
                raise ClaudeCLIError(
                    f"claude -p returned is_error=true: {str(record)[:300]}"
                )
            _set_anthropic_usage(state, record.get("usage") or {})
            if not state.text_parts and record.get("result"):
                yield from _text_delta(state, str(record["result"]))
            continue
        if record.get("type") != "stream_event":
            continue
        event = record.get("event") or {}
        event_type = event.get("type")
        if event_type == "message_start":
            message = event.get("message") or {}
            if message.get("id"):
                state.message_id = str(message["id"])
            _set_anthropic_usage(state, message.get("usage") or {})
            yield from _start_anthropic_stream(state)
        elif event_type == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                yield from _text_delta(state, str(delta.get("text") or ""))
        elif event_type == "message_delta":
            delta = event.get("delta") or {}
            if delta.get("stop_reason"):
                state.stop_reason = str(delta["stop_reason"])
            _set_anthropic_usage(state, event.get("usage") or {})
        elif event_type == "message_stop":
            # Wait for Claude Code's following ``result`` record before
            # closing the downstream stream. It carries the authoritative
            # success/error bit and final usage; emitting message_stop first
            # could otherwise produce an invalid error-after-stop sequence.
            continue
    yield from _finish_anthropic_stream(state)


_OPENAI_STOP_TO_ANTHROPIC = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


def iter_openai_anthropic_sse(
    lines: Iterator[str],
    state: AnthropicStreamState,
) -> Iterator[str]:
    """Translate cleaned OpenAI SSE lines into Anthropic Messages events."""
    yield from _start_anthropic_stream(state)
    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        usage = chunk.get("usage") or {}
        state.input_tokens = max(
            state.input_tokens, int(usage.get("prompt_tokens", 0) or 0)
        )
        state.output_tokens = max(
            state.output_tokens, int(usage.get("completion_tokens", 0) or 0)
        )
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str) and content:
                yield from _text_delta(state, content)
            _accumulate_tool_calls(state, delta.get("tool_calls"))
            finish = choice.get("finish_reason")
            if finish:
                state.stop_reason = _OPENAI_STOP_TO_ANTHROPIC.get(
                    str(finish), "end_turn"
                )
    yield from _finish_anthropic_stream(state)


def _accumulate_tool_calls(
    state: AnthropicStreamState,
    tool_calls: Optional[List[Dict[str, Any]]],
) -> None:
    """Fold one chunk's ``delta.tool_calls`` into ``state.tool_calls``.

    OpenAI spreads a single call across chunks: the first carries ``id`` and
    ``function.name``, later ones append ``function.arguments`` fragments that
    are only valid JSON once concatenated. ``index`` is what ties them
    together, so it keys the accumulator rather than list position.
    """
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        index = int(call.get("index", 0) or 0)
        entry = state.tool_calls.setdefault(index, {"id": "", "name": "", "args": []})
        if call.get("id"):
            entry["id"] = str(call["id"])
        function = call.get("function") or {}
        if function.get("name"):
            entry["name"] = str(function["name"])
        arguments = function.get("arguments")
        if isinstance(arguments, str) and arguments:
            entry["args"].append(arguments)


def iter_buffered_anthropic_sse(
    envelope: Dict[str, Any],
    state: AnthropicStreamState,
) -> Iterator[str]:
    """Shape a buffered backend result as one valid Anthropic SSE sequence."""
    _set_anthropic_usage(state, envelope.get("usage") or {})
    state.stop_reason = str(envelope.get("stop_reason") or "end_turn")
    yield from _start_anthropic_stream(state)
    yield from _text_delta(state, str(envelope.get("result") or ""))
    yield from _finish_anthropic_stream(state)


def anthropic_stream_error(exc: Exception) -> str:
    """Return the Anthropic error event used after an SSE response has begun."""
    return _anthropic_sse({
        "type": "error",
        "error": {"type": "api_error", "message": str(exc)},
    })


# ---- shared content-block helpers (unchanged shape) ----

class ContentBlock(BaseModel):
    type: str
    text: Optional[str] = None
    # Anthropic image block: {"type": "image", "source": {"type": "base64",
    # "media_type": "image/png", "data": "<b64>"}} or {"type": "url",
    # "url": "https://..."}. Kept loose to forward fields we don't model.
    source: Optional[Dict[str, Any]] = None
    # tool_use block (assistant turn): {"type": "tool_use", "id": "toolu_...",
    # "name": "get_weather", "input": {...}}.
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    # tool_result block (user turn): {"type": "tool_result", "tool_use_id":
    # "toolu_...", "content": str | [block, ...], "is_error": bool}.
    tool_use_id: Optional[str] = None
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    is_error: Optional[bool] = None


class Message(BaseModel):
    role: str
    content: Union[str, List[ContentBlock]]


class MessagesRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = None
    system: Optional[Union[str, List[ContentBlock]]] = None
    stream: bool = False
    temperature: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    # Anthropic tool definitions ({"name", "description", "input_schema"}) and
    # tool_choice ({"type": "auto"|"any"|"none"|"tool", "name": ...}). Served
    # by the local `openai` backends; refused on the CLI backends (#552).
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


# Content-block types that only mean anything on a tool-capable backend.
_TOOL_BLOCK_TYPES = ("tool_use", "tool_result")


def openai_tool_params(req: MessagesRequest) -> Dict[str, Any]:
    """Extra upstream parameters carrying this request's tool definitions.

    Empty when the caller sent no ``tools``, so a plain chat request reaches
    llama-server byte-identical to before. Translation errors are the
    caller's (a malformed tool definition), so they surface as 400 rather
    than ``UpstreamError``'s 502.
    """
    if not req.tools:
        return {}
    try:
        extra: Dict[str, Any] = {"tools": anthropic_tools_to_openai(req.tools)}
        choice = anthropic_tool_choice_to_openai(req.tool_choice)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if choice is not None:
        extra["tool_choice"] = choice
    return extra


def reject_tools_on_cli_backend(model: Model, req: MessagesRequest) -> None:
    """400 when tool use is asked of a backend that cannot serve it.

    The ``claude`` / ``gemini`` dispatch flattens a conversation into one text
    prompt (``_flatten_messages``), so ``tools`` and ``tool_use`` /
    ``tool_result`` blocks have nowhere to go. Dropping them silently would
    answer in prose a caller that asked for a tool call — the well-formed
    wrong answer #474 refused for non-text parts on the OpenAI shape.
    """
    if model.backend not in ("claude", "gemini"):
        return
    if req.tools:
        unsupported = "a 'tools' parameter"
    elif any(
        isinstance(m.content, list)
        and any(b.type in _TOOL_BLOCK_TYPES for b in m.content)
        for m in req.messages
    ):
        unsupported = "tool_use / tool_result content blocks"
    else:
        return
    raise HTTPException(
        status_code=400,
        detail=(
            f"backend {model.id!r} ({model.display_name}) cannot serve "
            f"{unsupported}: the CLI backends flatten a conversation into a "
            "single text prompt. Route tool-use requests to a local "
            "openai-backend model instead."
        ),
    )


def _content_to_text(content: Union[str, List[ContentBlock]]) -> str:
    if isinstance(content, str):
        return content
    parts: List[str] = []
    for block in content:
        if block.type == "text" and block.text:
            parts.append(block.text)
    return "\n".join(parts)


def _system_to_text(system: Optional[Union[str, List[ContentBlock]]]) -> Optional[str]:
    if system is None:
        return None
    if isinstance(system, str):
        return system
    return _content_to_text(system) or None


_EXT_BY_MEDIA_TYPE = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
    "application/pdf": "pdf",
    # Text/data document types — the CLI paths can attach any file, so a
    # caller may send a `document` block carrying one of these. Unknown
    # media types fall back to `.bin`, which the CLIs still read as bytes.
    "text/plain": "txt",
    "text/markdown": "md",
    "application/json": "json",
    "text/csv": "csv",
    "application/xml": "xml",
    "text/xml": "xml",
    "text/html": "html",
    "application/x-yaml": "yaml",
    "text/yaml": "yaml",
}

# Content-block types extracted to temp files for the multimodal CLI paths,
# mapped to (filename stem, default media_type when the block omits one).
_MEDIA_BLOCK_TYPES = {
    "image": ("img", "image/png"),
    "document": ("doc", "application/pdf"),
}


@contextmanager
def _extract_media_blocks(
    messages: List[Message],
) -> Iterator[Tuple[List[Message], List[Path]]]:
    """Pull media content blocks out of messages, write them to a temp dir.

    Handles Anthropic-style ``image`` and ``document`` (PDF) blocks. Yields
    ``(stripped_messages, attachment_paths)``. Stripped messages keep only
    text blocks so the existing flattener works unchanged. The temp dir and
    its contents are removed when the context exits, which must not happen
    until after the backend subprocess returns.

    Only ``source.type == "base64"`` blocks are written to disk.
    ``source.type == "url"`` is forwarded as a text reference to the URL
    since neither CLI fetches remote URLs on our behalf — fetching needs
    `httpx.get` first, which we can add later if a caller actually needs it.
    """
    attachment_paths: List[Path] = []
    stripped: List[Message] = []
    has_media = any(
        isinstance(m.content, list)
        and any(b.type in _MEDIA_BLOCK_TYPES for b in m.content)
        for m in messages
    )

    if not has_media:
        # Fast path — no temp dir at all when there's nothing to extract.
        yield messages, []
        return

    with tempfile.TemporaryDirectory(prefix="hub-media-") as td:
        td_path = Path(td)
        for msg in messages:
            if isinstance(msg.content, str):
                stripped.append(msg)
                continue
            kept: List[ContentBlock] = []
            for block in msg.content:
                if block.type not in _MEDIA_BLOCK_TYPES or not block.source:
                    kept.append(block)
                    continue
                stem, default_media = _MEDIA_BLOCK_TYPES[block.type]
                src = block.source
                stype = src.get("type")
                if stype == "base64":
                    data_b64 = src.get("data") or ""
                    media = src.get("media_type", default_media)
                    ext = _EXT_BY_MEDIA_TYPE.get(media, "bin")
                    fname = f"{stem}_{len(attachment_paths)}.{ext}"
                    fpath = td_path / fname
                    try:
                        fpath.write_bytes(base64.b64decode(data_b64))
                    except Exception as e:
                        raise HTTPException(
                            status_code=400,
                            detail=f"bad {block.type} block: {e}",
                        )
                    attachment_paths.append(fpath)
                elif stype == "url":
                    url = src.get("url", "")
                    kept.append(
                        ContentBlock(type="text", text=f"[{block.type} url: {url}]")
                    )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"unsupported {block.type} source.type {stype!r}",
                    )
            # Keep at least an empty text block so flatteners don't crash.
            if not kept:
                kept = [ContentBlock(type="text", text="")]
            stripped.append(Message(role=msg.role, content=kept))
        yield stripped, attachment_paths


def _flatten_messages(messages: List[Message]) -> str:
    """Flatten multi-turn into one prompt for the claude/gemini CLI dispatch.

    Shared by both the Anthropic-shape ``/v1/messages`` route (via
    ``_run_claude_backend``/``_run_gemini_backend``) and the OpenAI-shape
    ``/v1/chat/completions`` route (via ``_openai_messages_to_anthropic``
    below) — one prompt scaffold, so a format change applied here reaches
    both routes instead of only whichever one happened to get edited.
    """
    if not messages:
        raise ValueError("messages must not be empty")
    if len(messages) == 1 and messages[0].role == "user":
        return _content_to_text(messages[0].content)
    lines: List[str] = ["Previous conversation:"]
    for m in messages[:-1]:
        label = "User" if m.role == "user" else "Assistant"
        lines.append(f"{label}: {_content_to_text(m.content)}")
    last = messages[-1]
    lines.append("")
    lines.append(f"Current {last.role} message:")
    lines.append(_content_to_text(last.content))
    return "\n".join(lines)


def _openai_messages_to_anthropic(
    messages: List[Dict[str, Any]],
    model_label: Optional[str] = None,
) -> Tuple[List[Message], Optional[str]]:
    """Normalize OpenAI-shape dict messages into Anthropic-shape ``Message``
    objects plus an extracted system prompt, so ``/v1/chat/completions`` can
    reuse ``_flatten_messages`` instead of hand-rolling its own prompt
    scaffold (issue #195 — the two routes previously diverged silently).

    Non-text content parts (``image_url``, ``input_audio``, ``file``, …) are
    **refused with a 400** rather than dropped (issue #474). This route
    flattens a conversation down to a single text prompt for the claude /
    gemini CLI dispatch, so silently keeping only the ``text`` parts returned a
    well-formed 200 that answered a question the caller never asked. Refusing
    loudly matches ``_run_openai_backend``'s guard for the same input on the
    Anthropic-shape route; media for these backends goes to ``/v1/messages``,
    which carries it through ``_extract_media_blocks``.
    """
    sys_text: Optional[str] = None
    turns: List[Message] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            unsupported = sorted({
                str(p.get("type") or "unknown") if isinstance(p, dict) else type(p).__name__
                for p in content
                if not (isinstance(p, dict) and p.get("type") == "text")
            })
            if unsupported:
                who = f"{model_label!r} " if model_label else ""
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"backend {who}cannot accept {', '.join(unsupported)} content "
                        "on /v1/chat/completions — this route flattens messages to a "
                        "text prompt. Send image/document input to POST /v1/messages "
                        "instead."
                    ),
                )
            content = "\n".join(
                p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
            )
        if role == "system":
            sys_text = content
        else:
            turns.append(Message(role=role, content=content))
    return turns, sys_text


# ---- routing ----

def _run_claude_backend(model: Model, req: MessagesRequest) -> Dict[str, Any]:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    reject_tools_on_cli_backend(model, req)
    system = _system_to_text(req.system)
    with _extract_media_blocks(req.messages) as (msgs, attachments):
        prompt = _flatten_messages(msgs)
        try:
            return call_claude(
                # Use resolved display_name so version-free aliases
                # (e.g. `claude_haiku`) hit the right CLI model.
                prompt, model=model.display_name, system=system,
                attachments=attachments or None,
            )
        except ClaudeCLIError as e:
            raise HTTPException(status_code=502, detail=str(e))


def _run_gemini_backend(model: Model, req: MessagesRequest) -> Dict[str, Any]:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    reject_tools_on_cli_backend(model, req)
    system = _system_to_text(req.system)
    with _extract_media_blocks(req.messages) as (msgs, attachments):
        prompt = _flatten_messages(msgs)
        try:
            return call_gemini(
                prompt, model=model.display_name, system=system,
                attachments=attachments or None,
            )
        except GeminiCLIError as e:
            raise HTTPException(status_code=502, detail=str(e))


def _remote_headers(model: Model) -> Optional[Dict[str, str]]:
    """``Authorization`` header for a remote-hub call, if a token is
    configured for that host — see ``remote_proxy.remote_auth_token``.
    Most setups rely on the receiving hub's IP allowlist instead, so this
    is commonly ``None``.
    """
    token = remote_auth_token_for_model(model)
    return {"Authorization": f"Bearer {token}"} if token else None


def _run_openai_backend(model: Model, req: MessagesRequest) -> Dict[str, Any]:
    # Validated before the on-demand spin-up below: a malformed tool
    # definition is a 400 and shouldn't cold-start a model to discover it.
    extra = openai_tool_params(req)
    # On-demand lifecycle (#422): a cold ``startup: on_demand`` local backend
    # is spawned here and the request blocks until it answers (503 on load
    # failure) — same hook the OpenAI-shape route applies in server.py.
    from .server_common import ensure_backend_ready_or_503
    ensure_backend_ready_or_503(model)
    remote = remote_base_url(model)
    base_url = f"{remote}/v1" if remote else model.url
    if not base_url:
        raise HTTPException(status_code=500, detail=f"model {model.id} has no url")
    if any(
        isinstance(m.content, list)
        and any(b.type in ("image", "document") for b in m.content)
        for m in req.messages
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"backend {model.id!r} ({model.display_name}) is text-only. "
                "Route image/document requests to a claude-* or gemini-* "
                "model instead."
            ),
        )
    messages = anthropic_to_openai_messages(
        [m.model_dump() for m in req.messages],
        _system_to_text(req.system),
    )
    from . import on_demand as _on_demand
    try:
        with _on_demand.tracking(model, remote):
            raw = call_openai_chat(
                base_url,
                model=model.id if remote else model.display_name,
                messages=messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                extra=extra or None,
                headers=_remote_headers(model) if remote else None,
            )
    except UpstreamError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return openai_to_anthropic_envelope(raw)
