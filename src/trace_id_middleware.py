"""Response header contract — X-Trace-Id, request-id, and Anthropic parity.

The hub speaks the W3C ``traceparent`` standard natively (OpenTelemetry
handles it transparently). But many small clients — voice-transcriber,
openClaw, ad-hoc curl scripts — find it easier to mint a UUID4 and
shove it in an ``X-Trace-Id`` header. This middleware bridges:

- **Inbound:** if a request carries ``X-Trace-Id`` but no ``traceparent``,
  derive a deterministic 128-bit OpenTelemetry trace ID from the
  client's value (BLAKE2b-keyed UUID4 → 16 bytes) and inject a
  synthesized ``traceparent`` header into the ASGI scope **before** the
  OTel FastAPI middleware sees it. Two calls with the same X-Trace-Id
  land in the same Langfuse trace.

- **Outbound:** every response gets ``X-Trace-Id`` set to the current
  span's trace ID (hex). Clients can read it after the call to attach
  feedback / scores later via ``POST /admin/api/trace/{id}/feedback``.

On top of that contract this middleware also speaks Anthropic's header
vocabulary (#461), because it is the one layer that sees every response —
including the 401s and 4xx envelopes raised before any route runs:

- **``request-id``** on every non-static response, aliased to the same
  trace ID ``X-Trace-Id`` carries, so one identifier ties the client's
  record, the hub log line, and the Langfuse trace together. When tracing
  is off there is no span to read, so a random 128-bit ID keeps the header
  unconditional rather than silently absent — in that mode there is no
  ``X-Trace-Id`` for it to disagree with.
- **``anthropic-version`` / ``anthropic-beta``**, on the Anthropic-shape
  routes only, per :mod:`src.anthropic_headers`.

``X-Trace-Id`` itself is untouched — existing clients (voice-transcriber,
openClaw, ad-hoc scripts) read it and must keep seeing exactly what they
saw before.

Sits OUTERMOST in the middleware stack — added after
``instrument_fastapi_app(app)`` so it wraps the OTel layer.
"""

from __future__ import annotations

import logging
import secrets
from typing import Any, Awaitable, Callable

from .anthropic_errors import is_anthropic_shape_path
from .anthropic_headers import parity_headers
from .observability import derive_trace_id_from_uuid

logger = logging.getLogger(__name__)

# Routes that should NEVER carry a trace ID (static assets, admin SPA
# polling) — adding the header doesn't break anything but wastes bytes
# and clutters the response shape. We still process headers on every
# request because the cost is a dict lookup.
_NOISE_PREFIXES = ("/admin/static",)

# Paths whose request-id is worth a log line. The whole point of emitting
# the header is that a caller can quote it in a support question and the
# request be findable — so the API surface is logged and the admin SPA's
# own polling is not.
_LOGGED_PREFIXES = ("/v1/",)


def _fallback_request_id() -> str:
    """A trace-ID-shaped random ID, for when OTel is disabled."""
    return format(secrets.randbits(128), "032x")


def _trace_id_hex_from_current_span() -> str:
    """Read the current OTel span's trace ID, hex-encoded. Empty when no
    span is active or when OTel is disabled."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is None:
            return ""
        ctx = span.get_span_context()
        if not ctx or not ctx.trace_id:
            return ""
        return format(ctx.trace_id, "032x")
    except Exception:  # noqa: BLE001
        return ""


class TraceIdHeaderMiddleware:
    """Pure-ASGI middleware — installed OUTERMOST.

    Pure-ASGI rather than ``BaseHTTPMiddleware`` because we need to
    mutate the request scope's headers list, which BaseHTTPMiddleware
    does not expose cleanly.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(
        self,
        scope: dict,
        receive: Callable[[], Awaitable[dict]],
        send: Callable[[dict], Awaitable[None]],
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "") or ""
        method = scope.get("method", "?")
        is_noise = any(path.startswith(p) for p in _NOISE_PREFIXES)
        request_headers = list(scope.get("headers") or [])
        scope = self._maybe_synthesize_traceparent(scope)

        async def send_wrapper(message: dict) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", []))
                # Never clobber a header the handler set itself.
                present = {name.lower() for name, _ in headers}
                tid_hex = _trace_id_hex_from_current_span()
                if tid_hex and b"x-trace-id" not in present:
                    headers.append((b"x-trace-id", tid_hex.encode("ascii")))
                if not is_noise and b"request-id" not in present:
                    request_id = tid_hex or _fallback_request_id()
                    headers.append((b"request-id", request_id.encode("ascii")))
                    if any(path.startswith(p) for p in _LOGGED_PREFIXES):
                        logger.info(
                            "ℹ️ request-id=%s %s %s -> %s",
                            request_id,
                            method,
                            path,
                            message.get("status", "?"),
                        )
                if not is_noise and is_anthropic_shape_path(path):
                    for name, value in parity_headers(request_headers):
                        if name not in present:
                            headers.append((name, value))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)

    @staticmethod
    def _maybe_synthesize_traceparent(scope: dict) -> dict:
        raw_headers = scope.get("headers") or []
        path = scope.get("path", "")
        if any(path.startswith(p) for p in _NOISE_PREFIXES):
            return scope

        # Header names in ASGI scope are lowercase bytes.
        has_traceparent = False
        x_trace_id_value = b""
        for name, value in raw_headers:
            if name == b"traceparent":
                has_traceparent = True
                break
            if name == b"x-trace-id" and not x_trace_id_value:
                x_trace_id_value = value

        if has_traceparent or not x_trace_id_value:
            return scope

        try:
            derived = derive_trace_id_from_uuid(
                x_trace_id_value.decode("ascii", errors="replace")
            )
        except Exception:  # noqa: BLE001
            derived = None
        if derived is None:
            return scope

        trace_id_hex = format(derived, "032x")
        # Random 64-bit span ID for the synthesized root span.
        span_id_hex = format(secrets.randbits(64), "016x")
        # 00 = version, 01 = sampled flag.
        traceparent = f"00-{trace_id_hex}-{span_id_hex}-01".encode("ascii")

        new_headers = list(raw_headers) + [(b"traceparent", traceparent)]
        new_scope = dict(scope)
        new_scope["headers"] = new_headers
        return new_scope
