"""Local multi-model hub: Anthropic-compatible and OpenAI-compatible endpoints.

Routes each request by the `model` field to one of the backends in
`config/models.yaml`:

- claude-*                -> `claude -p` subprocess (existing behaviour)
- qwen3.5-9b / qwen*      -> llama-server at 127.0.0.1:8081 (/v1)
- glm-4.5-air / glm*      -> llama-server at 127.0.0.1:8082 (/v1)

Two shapes exposed:
  * POST /v1/messages          - Anthropic shape (drop-in for the SDK)
  * POST /v1/chat/completions  - OpenAI shape (passthrough/translation)
  * GET  /v1/models            - union of enabled names (both shapes)

Caveats (phase 1): text-only content. No streaming. No tool_use round-trip
for non-claude backends (OpenAI-shape callers get tool use natively from
llama-server; Anthropic-shape callers targeting qwen/glm are text-only).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from .claude_cli import ClaudeCLIError, call_claude
from .host_profile import hub_bind_host, hub_port
from .landing import LANDING_HTML
from .model_registry import Model, enabled_models, resolve as resolve_model
from .openai_upstream import (
    UpstreamError,
    anthropic_to_openai_messages,
    call_openai_chat,
    openai_response_text,
    openai_to_anthropic_envelope,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---- shared content-block helpers (unchanged shape) ----

class ContentBlock(BaseModel):
    type: str
    text: Optional[str] = None


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


def _flatten_messages(messages: List[Message]) -> str:
    """Flatten multi-turn into one prompt for `claude -p` (Claude path only)."""
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


def _envelope_to_anthropic(env: Dict[str, Any], requested_model: str) -> Dict[str, Any]:
    text = env.get("result") or ""
    usage_raw = env.get("usage") or {}
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": requested_model,
        "stop_reason": env.get("stop_reason") or "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage_raw.get("input_tokens", 0) or 0),
            "output_tokens": int(usage_raw.get("output_tokens", 0) or 0),
            "cache_creation_input_tokens": int(
                usage_raw.get("cache_creation_input_tokens", 0) or 0
            ),
            "cache_read_input_tokens": int(
                usage_raw.get("cache_read_input_tokens", 0) or 0
            ),
        },
    }


# ---- routing ----

def _resolve(model_name: str) -> Model:
    m = resolve_model(model_name)
    if m is None:
        known = [m.display_name for m in enabled_models()]
        raise HTTPException(
            status_code=400,
            detail=f"unknown model {model_name!r}. available on this host: {known}",
        )
    return m


def _run_claude_backend(req: MessagesRequest) -> Dict[str, Any]:
    prompt = _flatten_messages(req.messages)
    system = _system_to_text(req.system)
    try:
        return call_claude(prompt, model=req.model, system=system)
    except ClaudeCLIError as e:
        raise HTTPException(status_code=502, detail=str(e))


def _run_openai_backend(model: Model, req: MessagesRequest) -> Dict[str, Any]:
    if not model.url:
        raise HTTPException(status_code=500, detail=f"model {model.id} has no url")
    messages = anthropic_to_openai_messages(
        [m.model_dump() for m in req.messages],
        _system_to_text(req.system),
    )
    try:
        raw = call_openai_chat(
            model.url,
            model=model.display_name,
            messages=messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    except UpstreamError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return openai_to_anthropic_envelope(raw)


# ---- FastAPI app ----

app = FastAPI(title="claude-local-calls", version="0.2.0")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root() -> HTMLResponse:
    return HTMLResponse(LANDING_HTML)


@app.get("/info", include_in_schema=False)
def info() -> Dict[str, Any]:
    return {
        "name": "claude-local-calls",
        "version": app.version,
        "description": "Multi-model hub: Anthropic-shape + OpenAI-shape over Claude / Qwen / GLM.",
        "endpoints": {
            "health": "GET /health",
            "messages": "POST /v1/messages",
            "chat_completions": "POST /v1/chat/completions",
            "models": "GET /v1/models",
            "docs": "GET /docs",
        },
        "models": sorted({m.display_name for m in enabled_models()}),
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    data = []
    for m in enabled_models():
        for name in m.all_names:
            data.append({
                "id": name,
                "object": "model",
                "owned_by": m.backend,
                "backend": m.backend,
            })
    return {"object": "list", "data": data}


@app.post("/v1/messages")
def messages(req: MessagesRequest) -> JSONResponse:
    if req.stream:
        logger.warning("stream=true requested - returning non-streaming response")

    model = _resolve(req.model)
    logger.info("/v1/messages model=%s backend=%s", req.model, model.backend)

    if model.backend == "claude":
        env = _run_claude_backend(req)
    elif model.backend == "openai":
        env = _run_openai_backend(model, req)
    elif model.backend == "whisper":
        raise HTTPException(
            status_code=400,
            detail=(
                f"{req.model!r} is an ASR backend, not a chat model. "
                f"POST audio to http://127.0.0.1:{model.port}/v1/audio/transcriptions instead."
            ),
        )
    else:
        raise HTTPException(status_code=500, detail=f"unknown backend {model.backend!r}")

    payload = _envelope_to_anthropic(env, req.model)
    u = payload["usage"]
    logger.info(
        "<- in=%d out=%d (cache_r=%d cache_w=%d) stop=%s backend=%s",
        u["input_tokens"], u["output_tokens"],
        u["cache_read_input_tokens"], u["cache_creation_input_tokens"],
        payload["stop_reason"], model.backend,
    )
    return JSONResponse(payload)


# ---- OpenAI-shape endpoint ----

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


def _wrap_as_openai(text: str, *, model_name: str, in_toks: int, out_toks: int, finish: str = "stop") -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": in_toks,
            "completion_tokens": out_toks,
            "total_tokens": in_toks + out_toks,
        },
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest) -> JSONResponse:
    if req.stream:
        logger.warning("stream=true requested - returning non-streaming response")

    model = _resolve(req.model)
    logger.info("/v1/chat/completions model=%s backend=%s", req.model, model.backend)

    if model.backend == "claude":
        # Flatten OpenAI messages into a claude -p prompt.
        sys_text: Optional[str] = None
        turns: List[str] = []
        for m in req.messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            if role == "system":
                sys_text = content
            elif role == "user":
                turns.append(f"User: {content}")
            else:
                turns.append(f"Assistant: {content}")
        prompt = "\n".join(turns) if len(turns) > 1 else (turns[0].split(": ", 1)[-1] if turns else "")
        try:
            env = call_claude(prompt, model=req.model, system=sys_text)
        except ClaudeCLIError as e:
            raise HTTPException(status_code=502, detail=str(e))
        text = env.get("result", "")
        usage = env.get("usage") or {}
        return JSONResponse(_wrap_as_openai(
            text, model_name=req.model,
            in_toks=int(usage.get("input_tokens", 0) or 0),
            out_toks=int(usage.get("output_tokens", 0) or 0),
        ))

    if model.backend == "whisper":
        raise HTTPException(
            status_code=400,
            detail=(
                f"{req.model!r} is an ASR backend, not a chat model. "
                f"POST audio to http://127.0.0.1:{model.port}/v1/audio/transcriptions instead."
            ),
        )

    if model.backend == "openai":
        if not model.url:
            raise HTTPException(status_code=500, detail="model has no url")
        extra: Dict[str, Any] = {}
        if req.tools is not None:
            extra["tools"] = req.tools
        if req.tool_choice is not None:
            extra["tool_choice"] = req.tool_choice
        try:
            raw = call_openai_chat(
                model.url,
                model=model.display_name,
                messages=req.messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                extra=extra or None,
            )
        except UpstreamError as e:
            raise HTTPException(status_code=502, detail=str(e))
        # Passthrough of upstream response (already OpenAI-shape).
        return JSONResponse(raw)

    raise HTTPException(status_code=500, detail=f"unknown backend {model.backend!r}")


def main() -> None:
    import uvicorn
    uvicorn.run("src.server:app", host=hub_bind_host(), port=hub_port(), reload=False)


if __name__ == "__main__":
    main()
