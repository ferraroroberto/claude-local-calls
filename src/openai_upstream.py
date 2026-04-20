"""Adapter to call an OpenAI-compatible upstream (e.g. llama-server).

Used by the hub when routing to local Qwen / GLM backends. Two helpers:

- `call_openai_chat()`: POST {base_url}/chat/completions, return dict
- `openai_to_anthropic_envelope()`: shape the response into the same
  dict the existing claude-path code translates into an Anthropic
  `/v1/messages` response.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx


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
        r = httpx.post(url, json=payload, timeout=timeout)
    except httpx.HTTPError as e:
        raise UpstreamError(f"upstream {url} unreachable: {e}") from e
    if r.status_code >= 400:
        raise UpstreamError(f"upstream {url} HTTP {r.status_code}: {r.text[:500]}")
    try:
        return r.json()
    except Exception as e:
        raise UpstreamError(f"upstream returned non-JSON: {r.text[:200]!r}") from e


def openai_response_text(resp: Dict[str, Any]) -> str:
    choices = resp.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    text = msg.get("content") or ""
    if text:
        return text
    # Qwen3/GLM reasoning models put the answer in reasoning_content
    # when --jinja is on and the client doesn't opt out of thinking.
    return msg.get("reasoning_content") or ""


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
