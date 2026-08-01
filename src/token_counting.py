"""Prompt-token counting behind ``POST /v1/messages/count_tokens`` (#463).

The honest count differs per backend family, so this module resolves it per
backend instead of applying one global guess:

* ``openai`` (llama-server) — **exact**. The upstream owns the tokenizer, so
  the hub asks it: ``POST /apply-template`` renders the messages through the
  model's own chat template, and ``POST /tokenize`` counts the rendered
  prompt. That is the same text ``/completion`` would tokenize, so the number
  equals the ``usage.prompt_tokens`` a real request reports.
* ``claude`` / ``gemini`` — **approximate, and labelled as such**. Both are
  driven through a subscription CLI (``claude -p`` / ``agy``), neither of
  which exposes a tokenizer or a count-tokens call, and neither vendor
  publishes an offline tokenizer for the current model generation. There is
  no measurement to return, so the response carries ``exact: false`` plus a
  ``warning`` naming the method. A confidently-wrong count is worse than an
  absent endpoint, because it gets budgeted against.

Remote-owned models (``host:``/``hosts:`` pointing at another box) are
proxied to that host's own hub, which repeats this decision locally with the
backend actually in front of it.

Leaf module: imports only other leaf modules, never ``server.py``, so the
route handler can import it without a cycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from .chat_translation import (
    MessagesRequest,
    _content_to_text,
    _remote_headers,
    _system_to_text,
)
from .http_client import get_sync_client
from .model_registry import Model
from .openai_upstream import anthropic_to_openai_messages
from .remote_proxy import remote_base_url

# Characters per token for the heuristic used on the subscription-CLI
# backends. The widely-published English rule of thumb; it is a budgeting
# hint, not a measurement, which is exactly why every response built from it
# carries ``exact: false``.
APPROX_CHARS_PER_TOKEN = 4.0

# Method identifiers returned in the response body — stable slugs a client can
# branch on, with the human explanation carried by ``warning``.
METHOD_LLAMA_TOKENIZER = "llama_server_tokenizer"
METHOD_LLAMA_TOKENIZER_UNTEMPLATED = "llama_server_tokenizer_untemplated"
METHOD_CHAR_HEURISTIC = "character_heuristic"

_HEURISTIC_REASON = {
    "claude": (
        "the claude-* rows run through the local `claude -p` CLI on your "
        "subscription, which exposes no tokenizer and no count-tokens call"
    ),
    "gemini": (
        "the gemini-* rows run through the `agy` Antigravity CLI on your "
        "Google sign-in, which exposes no tokenizer and no count-tokens call"
    ),
}

_UPSTREAM_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class TokenCount:
    """One backend's answer: the number, how it was reached, and whether it
    is a measurement or an estimate."""

    input_tokens: int
    method: str
    exact: bool
    warning: Optional[str] = None


# ---------------------------------------------------------------- llama-server


def _llama_base_url(model: Model) -> str:
    """Root URL of the model's llama-server (its ``/v1`` sibling paths
    ``/tokenize`` and ``/apply-template`` hang off the root, not off ``/v1``)."""
    url = model.url
    if not url:
        raise HTTPException(status_code=500, detail="model has no url")
    return url.removesuffix("/v1").rstrip("/")


def _post_upstream(url: str, payload: Dict[str, Any]) -> httpx.Response:
    try:
        return get_sync_client().post(url, json=payload, timeout=_UPSTREAM_TIMEOUT_S)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"upstream {url} unreachable: {exc}"
        ) from exc


def _apply_chat_template(base: str, messages: List[Dict[str, Any]]) -> Optional[str]:
    """Render ``messages`` through the model's own chat template.

    Returns the rendered prompt, or ``None`` when the upstream is an older
    llama-server without ``/apply-template`` — the caller then falls back to
    an untemplated (and therefore approximate) count rather than failing.
    """
    url = f"{base}/apply-template"
    r = _post_upstream(url, {"messages": messages})
    if r.status_code in (404, 405, 501):
        return None
    if r.status_code >= 400:
        raise HTTPException(
            status_code=502, detail=f"upstream {url} HTTP {r.status_code}: {r.text[:300]}"
        )
    try:
        prompt = r.json().get("prompt")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"upstream {url} returned non-JSON: {r.text[:200]!r}"
        ) from exc
    return prompt if isinstance(prompt, str) else None


def _tokenize(base: str, text: str) -> int:
    """Token count of ``text`` per the upstream's own tokenizer.

    ``add_special: true`` mirrors what llama-server does for a real
    completion, so the count matches the ``prompt_tokens`` it would report.
    """
    url = f"{base}/tokenize"
    r = _post_upstream(url, {"content": text, "add_special": True})
    if r.status_code >= 400:
        raise HTTPException(
            status_code=502, detail=f"upstream {url} HTTP {r.status_code}: {r.text[:300]}"
        )
    try:
        tokens = r.json().get("tokens")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"upstream {url} returned non-JSON: {r.text[:200]!r}"
        ) from exc
    if not isinstance(tokens, list):
        raise HTTPException(
            status_code=502, detail=f"upstream {url} returned no token list"
        )
    return len(tokens)


def _count_via_llama_server(model: Model, req: MessagesRequest) -> TokenCount:
    base = _llama_base_url(model)
    oai_messages = anthropic_to_openai_messages(
        [m.model_dump() for m in req.messages], _system_to_text(req.system)
    )
    prompt = _apply_chat_template(base, oai_messages)
    if prompt is not None:
        return TokenCount(
            input_tokens=_tokenize(base, prompt),
            method=METHOD_LLAMA_TOKENIZER,
            exact=True,
        )
    joined = "\n".join(str(m.get("content") or "") for m in oai_messages)
    return TokenCount(
        input_tokens=_tokenize(base, joined),
        method=METHOD_LLAMA_TOKENIZER_UNTEMPLATED,
        exact=False,
        warning=(
            "APPROXIMATE: this llama-server build has no /apply-template, so the "
            "message text was tokenized without the model's chat template. The "
            "vocabulary is exact but the per-turn template tokens are missing — "
            "the real prompt is larger. Upgrade llama-server for an exact count."
        ),
    )


# ------------------------------------------------------------------ heuristic


def _prompt_text(req: MessagesRequest) -> str:
    parts: List[str] = []
    system = _system_to_text(req.system)
    if system:
        parts.append(system)
    for message in req.messages:
        parts.append(_content_to_text(message.content))
    return "\n".join(parts)


def _has_non_text_blocks(req: MessagesRequest) -> bool:
    for message in req.messages:
        if isinstance(message.content, list) and any(
            block.type != "text" for block in message.content
        ):
            return True
    return False


def _count_via_heuristic(model: Model, req: MessagesRequest) -> TokenCount:
    text = _prompt_text(req)
    reason = _HEURISTIC_REASON.get(
        model.backend, f"the {model.backend!r} backend exposes no tokenizer"
    )
    warning = (
        f"APPROXIMATE — NOT a measurement: {reason}. This number is "
        f"len(text)/{APPROX_CHARS_PER_TOKEN:g} over the caller's own message and "
        "system text only. It excludes the backend CLI's own system prompt and "
        "tool definitions, which dominate the input_tokens a real /v1/messages "
        "call reports. Treat it as a rough budget hint, never as a bill."
    )
    if _has_non_text_blocks(req):
        warning += (
            " The request also carries image/document blocks, whose tokens are "
            "not counted at all."
        )
    return TokenCount(
        input_tokens=math.ceil(len(text) / APPROX_CHARS_PER_TOKEN),
        method=METHOD_CHAR_HEURISTIC,
        exact=False,
        warning=warning,
    )


# --------------------------------------------------------------- remote hosts


def _count_via_remote_hub(
    model: Model, req: MessagesRequest, remote: str
) -> Dict[str, Any]:
    """Forward to the owning host's hub, which counts against the backend it
    actually runs. Its answer (exact or approximate) is passed through."""
    url = f"{remote}/v1/messages/count_tokens"
    body = req.model_dump(exclude_none=True)
    body["model"] = model.id
    try:
        r = get_sync_client().post(
            url, json=body, headers=_remote_headers(model), timeout=_UPSTREAM_TIMEOUT_S
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"remote hub {url} unreachable: {exc}"
        ) from exc
    if r.status_code >= 400:
        raise HTTPException(
            status_code=r.status_code if r.status_code < 500 else 502,
            detail=f"remote hub {url} HTTP {r.status_code}: {r.text[:300]}",
        )
    try:
        payload = r.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"remote hub {url} returned non-JSON: {r.text[:200]!r}"
        ) from exc
    if not isinstance(payload, dict) or "input_tokens" not in payload:
        raise HTTPException(
            status_code=502, detail=f"remote hub {url} returned no input_tokens"
        )
    # Echo back the name the caller asked for, not the remote's registry id.
    payload["model"] = req.model
    return payload


# -------------------------------------------------------------------- public


def count_tokens(model: Model, req: MessagesRequest) -> Dict[str, Any]:
    """Count the input tokens ``req`` would consume on ``model``.

    Returns the ``POST /v1/messages/count_tokens`` response body: Anthropic's
    ``input_tokens`` plus the hub's own honesty fields (``exact``, ``method``,
    and ``warning`` whenever ``exact`` is false).
    """
    remote = remote_base_url(model)
    if remote:
        return _count_via_remote_hub(model, req, remote)

    if model.backend == "openai":
        # A cold ``startup: on_demand`` backend owns the only tokenizer that
        # can answer exactly, so it is spawned here exactly as a real request
        # would spawn it (distinct 503 on load failure).
        from .server_common import ensure_backend_ready_or_503

        ensure_backend_ready_or_503(model)
        result = _count_via_llama_server(model, req)
    else:
        result = _count_via_heuristic(model, req)

    payload: Dict[str, Any] = {
        "input_tokens": result.input_tokens,
        "model": req.model,
        "backend": model.backend,
        "method": result.method,
        "exact": result.exact,
    }
    if not result.exact:
        payload["warning"] = result.warning or "APPROXIMATE: not a measured count."
    return payload
