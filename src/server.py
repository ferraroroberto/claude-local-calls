"""Local Anthropic-compatible API server backed by `claude -p`.

Exposes POST /v1/messages with the same request/response shape as the
official Anthropic Messages API, so the `anthropic` SDK (or any client)
can be pointed at this server via `base_url="http://localhost:8000"`
and call through the user's local Claude Code auth instead of an API key.

Caveats (intentional — lightweight):
  * No streaming (no SSE). `stream: true` falls back to a single response.
  * Multi-turn is flattened into one prompt (see _flatten_messages).
  * Tool use / images / extended thinking not implemented.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from .claude_cli import ClaudeCLIError, call_claude
from .landing import LANDING_HTML

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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
    """Flatten a multi-turn exchange into a single prompt for `claude -p`.

    The last user message is the live turn; prior turns are prepended as
    labelled context. Single-turn calls pass straight through.
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


app = FastAPI(title="claude-local-calls", version="0.1.0")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root() -> HTMLResponse:
    return HTMLResponse(LANDING_HTML)


@app.get("/info", include_in_schema=False)
def info() -> Dict[str, Any]:
    return {
        "name": "claude-local-calls",
        "version": "0.1.0",
        "description": "Anthropic-compatible local API backed by `claude -p`.",
        "endpoints": {
            "health": "GET /health",
            "messages": "POST /v1/messages",
            "docs": "GET /docs",
            "redoc": "GET /redoc",
        },
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/messages")
def messages(req: MessagesRequest) -> JSONResponse:
    if req.stream:
        logger.warning("stream=true requested — returning non-streaming response")

    prompt = _flatten_messages(req.messages)
    system = _system_to_text(req.system)

    logger.info(
        "→ claude -p (model=%s, prompt_chars=%d, system=%s)",
        req.model, len(prompt), "yes" if system else "no",
    )
    try:
        env = call_claude(prompt, model=req.model, system=system)
    except ClaudeCLIError as e:
        raise HTTPException(status_code=502, detail=str(e))

    payload = _envelope_to_anthropic(env, req.model)
    u = payload["usage"]
    logger.info(
        "<- tokens in=%d out=%d (cache_read=%d cache_write=%d) stop=%s",
        u["input_tokens"], u["output_tokens"],
        u["cache_read_input_tokens"], u["cache_creation_input_tokens"],
        payload["stop_reason"],
    )
    return JSONResponse(payload)


def main() -> None:
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
