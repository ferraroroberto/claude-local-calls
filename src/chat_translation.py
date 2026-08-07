"""Chat-shape translation: request/response schemas, media-block extraction,
prompt flattening, and per-backend dispatch shared by the ``/v1/messages``
and ``/v1/chat/completions`` routes in ``server.py``.

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
import tempfile
from contextlib import contextmanager
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
    call_openai_chat,
    openai_to_anthropic_envelope,
)
from .remote_proxy import remote_auth_token_for_model, remote_base_url


# ---- shared content-block helpers (unchanged shape) ----

class ContentBlock(BaseModel):
    type: str
    text: Optional[str] = None
    # Anthropic image block: {"type": "image", "source": {"type": "base64",
    # "media_type": "image/png", "data": "<b64>"}} or {"type": "url",
    # "url": "https://..."}. Kept loose to forward fields we don't model.
    source: Optional[Dict[str, Any]] = None


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
                headers=_remote_headers(model) if remote else None,
            )
    except UpstreamError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return openai_to_anthropic_envelope(raw)
