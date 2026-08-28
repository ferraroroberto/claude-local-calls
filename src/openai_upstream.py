"""Adapter to call an OpenAI-compatible upstream (e.g. llama-server).

Used by the hub when routing to local Qwen / GLM backends. Helpers:

- ``call_openai_chat()``: POST {base_url}/chat/completions, return dict.
- ``call_openai_chat_stream()``: POST with ``stream: true``, yield raw
  SSE byte chunks from the upstream (used to proxy SSE through the hub
  without translating shapes).
- ``openai_to_anthropic_envelope()``: shape the response into the same
  dict the existing claude-path code translates into an Anthropic
  ``/v1/messages`` response.
- ``strip_think_blocks()`` / ``ThinkStripper``: scrub ``<think>...``
  ``</think>`` segments from text or from streamed deltas, with carry
  over so a tag split across SSE chunks is still cleaned correctly.
- ``clean_openai_response()`` / ``clean_openai_chunk()``: in-place
  cleanup that folds ``reasoning_content`` into ``content`` when the
  upstream emits the answer in the reasoning channel, and strips any
  ``<think>`` tags left in ``content`` (Qwen3-style models when run
  with ``--reasoning-format none``).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterator, List, Optional

import httpx

from .http_client import get_sync_client


class UpstreamError(RuntimeError):
    pass


def anthropic_to_openai_messages(
    messages: List[Dict[str, Any]],
    system: Optional[str],
) -> List[Dict[str, Any]]:
    """Flatten Anthropic message blocks into OpenAI-shape messages.

    Anthropic allows `content` to be a list of content blocks; OpenAI
    expects a string for text-only messages. This only handles text
    blocks; images/tool_use are dropped in phase 1.
    """
    out: List[Dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                btype = (block.get("type") if isinstance(block, dict) else getattr(block, "type", None))
                btext = (block.get("text") if isinstance(block, dict) else getattr(block, "text", None))
                if btype == "text" and btext:
                    parts.append(btext)
            content = "\n".join(parts)
        out.append({"role": role, "content": content or ""})
    return out


def call_openai_chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    timeout: float = 600.0,
    extra: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {"model": model, "messages": messages}
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if extra:
        payload.update(extra)
    try:
        r = get_sync_client().post(url, json=payload, headers=headers, timeout=timeout)
    except httpx.HTTPError as e:
        raise UpstreamError(f"upstream {url} unreachable: {e}") from e
    if r.status_code >= 400:
        raise UpstreamError(f"upstream {url} HTTP {r.status_code}: {r.text[:500]}")
    try:
        return r.json()
    except Exception as e:
        raise UpstreamError(f"upstream returned non-JSON: {r.text[:200]!r}") from e


def call_openai_chat_stream(
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    timeout: float = 600.0,
    extra: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Iterator[str]:
    """POST with ``stream: true`` and yield SSE *lines* from the upstream.

    Each yielded item is a single line of the SSE body (without the
    trailing ``\\n``). The caller re-emits these as ``line + "\\n"``
    plus the SSE record terminator. Lines may be empty (record
    separators), comments (``: keepalive``), or ``data: ...`` payloads.
    Yields nothing extra after the upstream closes — callers are
    responsible for ensuring a final ``data: [DONE]`` if the upstream
    didn't already send one (llama-server does).
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": True}
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if extra:
        payload.update(extra)
    req_headers = {"Accept": "text/event-stream", **(headers or {})}
    try:
        with get_sync_client().stream(
            "POST", url, json=payload, headers=req_headers, timeout=timeout
        ) as r:
            if r.status_code >= 400:
                body = r.read().decode("utf-8", errors="replace")
                raise UpstreamError(
                    f"upstream {url} HTTP {r.status_code}: {body[:500]}"
                )
            for line in r.iter_lines():
                yield line
    except httpx.HTTPError as e:
        raise UpstreamError(f"upstream {url} unreachable: {e}") from e


# -----------------------------------------------------------------------
# Thinking / reasoning cleanup
# -----------------------------------------------------------------------
#
# Qwen3-style models (and others) emit chain-of-thought between
# ``<think>`` and ``</think>`` tags when run with
# ``--reasoning-format none``. OpenAI-shape clients (e.g. openClaw's
# vllm provider) only read ``message.content`` / ``delta.content`` and
# don't know what to do with raw thinking text — it pollutes tool
# decisions and confuses downstream parsers. We strip those blocks
# server-side so callers see only the final answer.
#
# When the upstream uses ``--reasoning-format deepseek`` (the default
# in some llama.cpp builds), the thinking lands in
# ``message.reasoning_content`` instead. If ``content`` is empty we
# fold ``reasoning_content`` into ``content`` so the caller still gets
# *something* — better than an empty response.

_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)


def strip_think_blocks(text: str) -> str:
    """Remove ``<think>...</think>`` segments from a complete string.

    Used for non-streaming responses where we have the full content in
    one shot. For streaming, use :class:`ThinkStripper` instead — it
    keeps a buffer so a tag split across SSE chunks still gets stripped.

    An ``<think>`` left open with no matching close (the upstream was
    truncated mid-thought, e.g. ``max_tokens`` cut it off) has no pair
    for ``_THINK_RE`` to match. Drop from that open tag to the end
    rather than leaking raw reasoning into ``content`` — mirrors
    :meth:`ThinkStripper.flush`'s truncated-stream handling.
    """
    if not text:
        return text
    text = _THINK_RE.sub("", text)
    m = _THINK_OPEN_RE.search(text)
    if m is not None:
        text = text[: m.start()]
    return text


class ThinkStripper:
    """Stateful filter that removes ``<think>...</think>`` from a stream.

    Maintains a small buffer so a tag straddling two upstream chunks is
    still recognised. The contract:

    - ``feed(chunk)`` returns the safe-to-emit prefix from this chunk
      (after concatenation with any retained buffer).
    - ``flush()`` returns whatever's left when the stream closes —
      only the bytes we held back waiting to see if a tag would form.

    Three internal modes:

    - ``out``: outside any think block. We retain a small tail (up to
      ``len('<think')``) in case a partial tag appears at the end.
    - ``in``: inside a think block. We discard everything until we see
      a closing ``</think>``.
    - ``in_tail``: inside a think block, retaining a short tail so a
      ``</think>`` split across chunks is recognised.
    """

    _OPEN_PREFIX_MAX = len("<think")
    _CLOSE_PREFIX_MAX = len("</think")

    def __init__(self) -> None:
        self._buf = ""
        self._mode = "out"  # "out" | "in"

    def feed(self, chunk: str) -> str:
        if not chunk and not self._buf:
            return ""
        self._buf += chunk
        out_parts: List[str] = []
        while True:
            if self._mode == "out":
                m = _THINK_OPEN_RE.search(self._buf)
                if m is not None:
                    out_parts.append(self._buf[: m.start()])
                    self._buf = self._buf[m.end():]
                    self._mode = "in"
                    continue
                # No full open tag. If a ``<`` appears within the last
                # ``_OPEN_PREFIX_MAX`` chars, retain from that ``<``
                # onward — it might be the start of a split open tag.
                # Otherwise emit everything.
                tail_len = self._OPEN_PREFIX_MAX
                tail_start = max(0, len(self._buf) - tail_len)
                lt_in_tail = self._buf.find("<", tail_start)
                if lt_in_tail >= 0:
                    out_parts.append(self._buf[:lt_in_tail])
                    self._buf = self._buf[lt_in_tail:]
                else:
                    out_parts.append(self._buf)
                    self._buf = ""
                break
            # mode == "in"
            close_idx = self._buf.lower().find("</think>")
            if close_idx >= 0:
                self._buf = self._buf[close_idx + len("</think>"):]
                self._mode = "out"
                continue
            # No close tag yet. Retain a tail so a split close tag is
            # still recognised; drop everything before it (still
            # thinking).
            keep_from = max(0, len(self._buf) - self._CLOSE_PREFIX_MAX)
            self._buf = self._buf[keep_from:]
            break
        return "".join(out_parts)

    def flush(self) -> str:
        # If we're outside any think block, anything we held back was
        # an innocent ``<...`` lookalike — emit it verbatim.
        # If we're inside a block at end-of-stream, drop the buffer:
        # the upstream cut off mid-thinking, no answer to recover.
        out = self._buf if self._mode == "out" else ""
        self._buf = ""
        return out


def _fold_reasoning_into_content(message: Dict[str, Any]) -> None:
    """If ``content`` is empty but ``reasoning_content`` isn't, swap them.

    Mutates ``message`` in place. Used for non-streaming responses;
    streaming fold is handled per-delta in :func:`clean_openai_chunk`.
    """
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    if (not content) and reasoning:
        message["content"] = reasoning
        message["reasoning_content"] = ""


def clean_openai_response(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Strip ``<think>`` blocks and fold reasoning_content for non-stream.

    Returns ``resp`` (mutated) for chaining. Safe to call on any
    OpenAI-shape chat completion dict; no-op on shapes that don't
    contain ``choices``.
    """
    for choice in resp.get("choices") or []:
        msg = choice.get("message")
        if not isinstance(msg, dict):
            continue
        _fold_reasoning_into_content(msg)
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = strip_think_blocks(content)
    return resp


def clean_openai_chunk(
    chunk: Dict[str, Any],
    strippers: Dict[int, ThinkStripper],
) -> Dict[str, Any]:
    """Apply think-strip + reasoning fold to a single SSE chunk dict.

    ``strippers`` is a per-stream cache keyed by ``choice.index`` so a
    ``<think>`` tag spanning multiple chunks is still recognised.
    Mutates and returns ``chunk``.
    """
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        # Fold reasoning delta into content delta when content is empty.
        # llama-server with --reasoning-format=deepseek emits
        # ``reasoning_content`` deltas separately; OpenAI-shape clients
        # ignore them. Treat them as part of the answer if and only if
        # the model never produced a real ``content`` delta on this
        # stream — that's still better than emitting nothing.
        if not delta.get("content") and delta.get("reasoning_content"):
            delta["content"] = delta["reasoning_content"]
            delta["reasoning_content"] = ""
        text = delta.get("content")
        if isinstance(text, str) and text:
            idx = int(choice.get("index", 0) or 0)
            stripper = strippers.get(idx)
            if stripper is None:
                stripper = ThinkStripper()
                strippers[idx] = stripper
            delta["content"] = stripper.feed(text)
    return chunk


def iter_cleaned_sse(raw_lines: Iterator[str]) -> Iterator[str]:
    """Filter raw SSE lines, applying think-strip to ``data:`` payloads.

    Yields cleaned SSE lines (without trailing newline). The caller is
    responsible for joining lines back with ``\\n`` (each line + ``\\n``
    is the wire format).

    Each choice's :class:`ThinkStripper` retains a short lookahead tail
    (see its docstring) that is never emitted by ``feed()`` alone. Before
    forwarding ``data: [DONE]``, every stripper is flushed and any
    held-back text is emitted as one final synthetic delta per choice —
    otherwise a stream whose last content delta happens to end near a
    ``<`` silently loses that tail.
    """
    strippers: Dict[int, ThinkStripper] = {}
    last_meta: Dict[str, Any] = {}
    for line in raw_lines:
        if not line.startswith("data:"):
            yield line
            continue
        payload = line[len("data:"):].lstrip()
        if payload == "":
            yield line
            continue
        if payload == "[DONE]":
            for idx in sorted(strippers):
                tail = strippers[idx].flush()
                if not tail:
                    continue
                yield "data: " + json.dumps(
                    {
                        **last_meta,
                        "choices": [
                            {
                                "index": idx,
                                "delta": {"content": tail},
                                "finish_reason": None,
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            yield line
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            # Not JSON we recognise — pass through unchanged. Better to
            # forward an oddly shaped line than to drop it silently.
            yield line
            continue
        clean_openai_chunk(obj, strippers)
        last_meta = {k: v for k, v in obj.items() if k != "choices"}
        yield "data: " + json.dumps(obj, ensure_ascii=False)


def openai_response_text(resp: Dict[str, Any]) -> str:
    """Extract assistant text from an OpenAI chat completion.

    Strips ``<think>`` blocks and falls back to ``reasoning_content``
    if ``content`` is empty.
    """
    choices = resp.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    text = msg.get("content") or ""
    if not text:
        # Qwen3/GLM reasoning models put the answer in reasoning_content
        # when --jinja is on and the client doesn't opt out of thinking.
        text = msg.get("reasoning_content") or ""
    return strip_think_blocks(text)


_STOP_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


def openai_to_anthropic_envelope(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Shape an OpenAI response into the envelope src.server consumes.

    The hub's `_envelope_to_anthropic` expects `{"result": str,
    "stop_reason": str, "usage": {...}}`. This builds exactly that.
    """
    usage = resp.get("usage") or {}
    finish = (resp.get("choices") or [{}])[0].get("finish_reason") or "stop"
    return {
        "result": openai_response_text(resp),
        "stop_reason": _STOP_MAP.get(finish, "end_turn"),
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }
