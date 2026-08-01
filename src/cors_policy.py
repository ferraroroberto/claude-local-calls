"""CORS policy for browser-based clients (#462).

Without CORS headers a page's JavaScript cannot call the hub at all: the
preflight fails and the ``fetch`` never reaches a route, so every sister
webapp on this machine has to proxy through its own backend to talk to a
loopback service. This module is the whole policy — which origins may
call, which request headers they may send, and which response headers
they may *read* (a header a browser cannot read is the same as not
sending it).

Three deliberate choices, each of which is the security-relevant half:

- **Origins are never a wildcard.** Loopback origins are allowed
  unconditionally via :data:`LOOPBACK_ORIGIN_REGEX`; anything else has to
  be named in ``cors_allow_origins`` in ``config/webapp_config.json``. A
  ``"*"`` found in that list is dropped with a warning rather than
  honoured — ``allow_origins=["*"]`` is exactly the shape that turns a
  loopback-trusting service into one any page in the browser can drive.
- **Loopback is the default because it is already the trust boundary.**
  The bearer gate (``app_web/middleware.py``) lets a loopback caller
  through without a token, so letting a loopback *origin* read the
  response grants nothing it did not already have. A page served from a
  non-loopback origin still makes its request from 127.0.0.1 — so the
  origin check, not the client IP, is what keeps it from reading the
  answer.
- **Credentials are off.** The hub sets no cookies and uses no
  HTTP-auth realm; it authenticates from ``Authorization`` / ``x-api-key``
  / ``?token=``, which a plain (non-credentialed) cross-origin ``fetch``
  carries fine. Leaving ``allow_credentials`` off removes the
  credentialed-CORS risk class outright instead of relying on the origin
  list to contain it.

CORS is **not** an auth relaxation: the middleware is installed
outermost so a preflight ``OPTIONS`` is answered instead of 401'd, but a
real request still travels the full stack and meets the bearer gate
exactly as before.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .webapp_config import WebappConfig

logger = logging.getLogger(__name__)

# Every loopback origin, any port, http or https — matched with
# ``fullmatch`` by starlette, so ``http://127.0.0.1.evil.example`` does
# not slip through. Covers the whole 127/8 block plus ``localhost`` and
# the IPv6 literal, which is what a browser actually puts in ``Origin``.
LOOPBACK_ORIGIN_REGEX = (
    r"https?://(localhost|127(\.\d{1,3}){3}|\[::1\])(:\d{1,5})?"
)

# The hub exposes GET/POST routes plus the admin sub-app's mutations.
CORS_ALLOW_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]

# Request headers a browser may send. Named explicitly rather than "*" so
# the list doubles as documentation of the wire vocabulary the hub
# actually speaks — but it has to be *complete*, because a header missing
# here fails the preflight and the call never happens. The ``x-stainless-*``
# entries are the telemetry headers the generated Anthropic and OpenAI
# SDKs attach to every request; omitting them breaks both SDKs in-browser
# even though nothing reads them server-side.
CORS_ALLOW_HEADERS = [
    "accept",
    "authorization",
    "content-type",
    # Anthropic SDK
    "x-api-key",
    "anthropic-version",
    "anthropic-beta",
    "anthropic-dangerous-direct-browser-access",
    # OpenAI SDK
    "openai-organization",
    "openai-project",
    "openai-beta",
    # tracing (the X-Trace-Id contract + W3C traceparent)
    "x-trace-id",
    "traceparent",
    "tracestate",
    # generated-SDK telemetry, sent unconditionally by both SDKs
    "x-stainless-arch",
    "x-stainless-lang",
    "x-stainless-os",
    "x-stainless-package-version",
    "x-stainless-read-timeout",
    "x-stainless-retry-count",
    "x-stainless-runtime",
    "x-stainless-runtime-version",
    "x-stainless-timeout",
]

# Response headers page JavaScript may read. Only ``Cache-Control``,
# ``Content-Language``, ``Content-Type``, ``Expires``, ``Last-Modified``
# and ``Pragma`` are readable by default, so every header the hub adds as
# a contract has to be listed here or it is invisible to the caller —
# which is the same silence #461 closed on the server side.
CORS_EXPOSE_HEADERS = [
    "request-id",
    "x-trace-id",
    "anthropic-version",
    "anthropic-beta",
    "warning",
    # emitted on the 401 a mis-configured browser client hits first
    "www-authenticate",
]

# Preflight cache lifetime (seconds). Ten minutes keeps a chatty SDK from
# re-preflighting every call without pinning a stale policy for long.
CORS_MAX_AGE = 600


def resolve_allowed_origins(configured: Optional[Iterable[Any]]) -> List[str]:
    """Normalise the configured extra origins into a starlette-ready list.

    Drops blanks and any wildcard (with a warning — the default must be a
    named list, never ``*``), strips the trailing slash a hand-written
    config tends to grow, and lowercases: a browser's ``Origin`` header is
    scheme + host + port only, all case-insensitive, so a stored
    ``https://Example.com/`` would otherwise silently never match.
    """
    out: List[str] = []
    for entry in configured or []:
        origin = str(entry).strip().rstrip("/").lower()
        if not origin:
            continue
        if "*" in origin:
            logger.warning(
                "⚠️ ignoring wildcard CORS origin %r — name the origins "
                "explicitly in config/webapp_config.json",
                str(entry),
            )
            continue
        if origin in out:
            continue
        out.append(origin)
    return out


def cors_kwargs(configured: Optional[Iterable[Any]] = None) -> Dict[str, Any]:
    """Build the :class:`CORSMiddleware` keyword arguments for one config."""
    return {
        "allow_origins": resolve_allowed_origins(configured),
        "allow_origin_regex": LOOPBACK_ORIGIN_REGEX,
        "allow_credentials": False,
        "allow_methods": CORS_ALLOW_METHODS,
        "allow_headers": CORS_ALLOW_HEADERS,
        "expose_headers": CORS_EXPOSE_HEADERS,
        "max_age": CORS_MAX_AGE,
    }


def install_cors(app: FastAPI, cfg: Optional[WebappConfig] = None) -> List[str]:
    """Add the CORS middleware to ``app``, OUTERMOST.

    Must be the last ``add_middleware`` call on the app: starlette builds
    the stack so the last-added layer wraps every earlier one, and CORS
    has to sit outside :class:`ParentBearerTokenMiddleware` for a
    preflight ``OPTIONS`` — which carries no ``Authorization`` header, by
    the browser's design — to be answered rather than 401'd.

    ``cfg`` is a :class:`~src.webapp_config.WebappConfig` or ``None``
    (config unreadable → loopback-only, the safe default). Returns the
    resolved extra-origin list, for logging and tests.
    """
    kwargs = cors_kwargs(getattr(cfg, "cors_allow_origins", None))
    app.add_middleware(CORSMiddleware, **kwargs)
    extra = kwargs["allow_origins"]
    if extra:
        logger.info("🌐 CORS: loopback origins + %s", ", ".join(extra))
    else:
        logger.info("🌐 CORS: loopback origins only")
    return extra
