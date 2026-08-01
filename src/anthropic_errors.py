"""Anthropic-shaped error envelopes for the Anthropic-shape routes (#460).

The hub raises :class:`fastapi.HTTPException` throughout, so without this
module every error reaches the client as FastAPI's default
``{"detail": "..."}``. The real Anthropic API answers with::

    {"type": "error", "error": {"type": "invalid_request_error", "message": "..."}}

and the ``anthropic`` SDK's callers key their own handling off that
envelope — so a hub that claims to be a drop-in for the SDK has to speak
it too.

**Scoped by path on purpose.** ``/v1/chat/completions`` callers parse the
OpenAI envelope, not this one, so the handlers below delegate to FastAPI's
defaults for every path that is not Anthropic-shaped. A blanket app-level
rewrite would be a regression on the OpenAI route.

Statuses are never rewritten — only the body. The observability ring and
the ``record_genai_metrics`` call in :func:`src.server.messages` both run
before/around the handler and are unaffected.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import Request
from fastapi.exception_handlers import (
    http_exception_handler as _default_http_exception_handler,
    request_validation_exception_handler as _default_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

# Paths speaking the Anthropic wire shape. Prefix-matched so a future
# sibling (e.g. /v1/messages/count_tokens) is covered without an edit here.
ANTHROPIC_SHAPE_PREFIX = "/v1/messages"

# The statuses the hub actually raises, mapped onto Anthropic's error-type
# enum. 503 is the hub's "backend not ready / failed to load" status, for
# which `overloaded_error` is the truer description than a flat `api_error`.
_ERROR_TYPE_BY_STATUS: Dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "overloaded_error",
    504: "api_error",
    529: "overloaded_error",
}


def is_anthropic_shape_path(path: str) -> bool:
    """True when ``path`` is served by an Anthropic-shape route."""
    return path == ANTHROPIC_SHAPE_PREFIX or path.startswith(
        ANTHROPIC_SHAPE_PREFIX + "/"
    )


def anthropic_error_type(status_code: int) -> str:
    """Map an HTTP status onto Anthropic's ``error.type`` enum."""
    known = _ERROR_TYPE_BY_STATUS.get(status_code)
    if known is not None:
        return known
    return "invalid_request_error" if 400 <= status_code < 500 else "api_error"


def anthropic_error_payload(status_code: int, message: str) -> Dict[str, Any]:
    """Build the Anthropic error envelope for one status + message."""
    return {
        "type": "error",
        "error": {
            "type": anthropic_error_type(status_code),
            "message": message,
        },
    }


def _detail_to_message(detail: Any) -> str:
    """Flatten an ``HTTPException.detail`` into a readable message string."""
    if isinstance(detail, str):
        return detail
    if detail is None:
        return ""
    return str(detail)


def _validation_to_message(exc: RequestValidationError) -> str:
    """Flatten pydantic's error list into one readable sentence.

    FastAPI's 422 body is a list of per-field objects the Anthropic
    envelope has no room for, so it collapses to ``loc: msg`` pairs.
    """
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in (err.get("loc") or ()))
        msg = str(err.get("msg") or "invalid value")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts) or "invalid request body"


async def anthropic_http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> Response:
    """Serialise HTTPExceptions from Anthropic-shape routes as the envelope."""
    if exc.status_code < 400 or not is_anthropic_shape_path(request.url.path):
        return await _default_http_exception_handler(request, exc)
    headers: Optional[Dict[str, str]] = getattr(exc, "headers", None)
    return JSONResponse(
        anthropic_error_payload(exc.status_code, _detail_to_message(exc.detail)),
        status_code=exc.status_code,
        headers=headers,
    )


async def anthropic_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> Response:
    """Serialise 422 body-validation failures as the Anthropic envelope."""
    if not is_anthropic_shape_path(request.url.path):
        return await _default_validation_exception_handler(request, exc)
    return JSONResponse(
        anthropic_error_payload(422, _validation_to_message(exc)),
        status_code=422,
    )


def install_anthropic_error_handlers(app: Any) -> None:
    """Register both handlers on ``app``.

    Registering against Starlette's ``HTTPException`` also covers
    FastAPI's subclass — Starlette's exception middleware walks the MRO
    when it looks a handler up.
    """
    app.add_exception_handler(
        StarletteHTTPException, anthropic_http_exception_handler
    )
    app.add_exception_handler(
        RequestValidationError, anthropic_validation_exception_handler
    )
