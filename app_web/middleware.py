"""Bearer-token middleware for the /admin sub-app.

Loopback callers (PC itself) bypass the bearer token. Non-loopback
callers must present ``Authorization: Bearer <token>``, ``x-api-key:
<token>`` (the Anthropic SDK's own credential header, #461), or
``?token=…`` on the URL (the last is what bookmarked / shared
URLs use; the SPA strips it from ``window.location`` on first load).

WebSocket handshakes are not seen by this middleware:
``BaseHTTPMiddleware`` only ever runs for ``http``-scope connections, so
a ``@router.websocket(...)`` route gets no coverage from it at all. Any
websocket route must therefore call :func:`authorize_websocket`
explicitly before ``accept()`` — the middleware cannot do it for you.
Most live-ops streams use SSE (plain HTTP, covered here); the Machines
tab's terminal proxy is the one websocket route, and it calls the guard.

The loopback and allowlist bypasses trust a caller for *where it connected
from*. A page served from off this machine can borrow those positions by having
the user's browser make the request, so a request whose ``Origin`` is not local
is held to the token instead — see :func:`_is_foreign_origin_request`, which
draws that line with `src/cors_policy.py`'s own definition of a local origin.
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
import re
from typing import Any, FrozenSet, List, Optional, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.cors_policy import LOOPBACK_ORIGIN_REGEX

logger = logging.getLogger(__name__)

# Loopback addresses bypass the bearer-token gate so local probes keep
# working. Tunnel traffic arrives with a non-loopback client IP and must
# present the token. ``testclient`` is starlette's pseudo-host for its
# in-process TestClient — treating it as loopback keeps pytest happy
# without forcing every fixture to inject a fake token.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})

# Headers that a reverse proxy sets when forwarding a request. The
# presence of ANY of these means the loopback ``request.client.host``
# is the proxy's own address, not the real client — so we must enforce
# the bearer token even though the TCP source is 127.0.0.1.
#
# Covers:
#   * tailscale serve     → ``X-Forwarded-For``, ``X-Forwarded-Proto``
#   * cloudflared tunnel  → ``cf-ray``, ``cf-connecting-ip``
#   * generic nginx/caddy → ``X-Forwarded-For``
PROXY_HEADERS = (
    "x-forwarded-for",
    "x-forwarded-proto",
    "x-forwarded-host",
    "cf-ray",
    "cf-connecting-ip",
)


def _is_proxied(headers) -> bool:
    for h in PROXY_HEADERS:
        if h in headers:
            return True
    return False

# Static + login + healthz/version are exempt: the login endpoint is
# how a phone-side user *gets* the token, and static assets must load
# before the login form can render. Paths here are the sub-app's view
# (the /admin mount prefix is stripped by starlette before we see them).
AUTH_EXEMPT_PREFIXES = ("/static/",)
AUTH_EXEMPT_EXACT = frozenset(
    {
        "",
        "/",            # SPA index — login overlay renders client-side
        "/api/login",
        "/api/healthz",
        "/api/version",
    }
)


# Methods with no side effects. A browser can already issue these at will
# (an <img>/<script>/<link> tag is enough) and the CORS policy in
# ``src/cors_policy.py`` is what stops another site from *reading* the answer,
# so gating them here would add no protection and would break ordinary
# same-origin asset loads.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

# One definition of "a local origin", shared with the CORS policy so the set of
# origins allowed to *read* a response and the set allowed to keep the
# credential-free bypasses can never drift apart.
_LOOPBACK_ORIGIN_RE = re.compile(LOOPBACK_ORIGIN_REGEX)


def _is_loopback_origin(origin: str) -> bool:
    return _LOOPBACK_ORIGIN_RE.fullmatch(origin.strip().lower()) is not None


def _is_foreign_origin_request(headers, method: Optional[str]) -> bool:
    """True when a browser reports this request was initiated off-box.

    ``Origin`` is the signal. A browser attaches it to every request that can
    change something (anything but GET/HEAD) and to every cross-origin one, and
    a page cannot forge it. "Foreign" here means exactly what
    :data:`~src.cors_policy.LOOPBACK_ORIGIN_REGEX` already defines as non-local,
    so this stays the same boundary the CORS policy draws — a sister webapp
    served from another loopback port is a first-class caller (that is the whole
    point of `src/cors_policy.py`) and keeps the bypasses; a page from off the
    machine does not.

    ``Sec-Fetch-Site`` is consulted only when no ``Origin`` is present, to cover
    a browser that omits it. A non-browser caller (curl, an SDK, the tray, a
    peer hub's httpx client) sends neither header and is never foreign: it holds
    no ambient credential for another site to borrow, so the trust rules apply
    to it unchanged.

    ``method`` is ``None`` for a websocket handshake, which is always checked —
    a handshake is nominally a GET, but the route it opens is not read-only.
    """
    if method is not None and method.upper() in SAFE_METHODS:
        return False
    origin = (headers.get("origin", "") or "").strip()
    if origin:
        return not _is_loopback_origin(origin)
    return (headers.get("sec-fetch-site", "") or "").strip().lower() == "cross-site"


def _presented_credential(headers, query_params) -> str:
    """The credential this request carries, from any of the accepted places."""
    auth_header = headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        presented = auth_header[7:].strip()
        if presented:
            return presented
    # ``x-api-key`` is how the Anthropic SDK authenticates (#461), so an SDK
    # pointed at the hub the ordinary way — ``api_key=<hub token>``, no
    # ``default_headers`` special-casing — presents its credential here.
    # Accepted as an equal alternative to the bearer token: same constant-time
    # compare, same 401 when it's wrong.
    presented = (headers.get("x-api-key", "") or "").strip()
    if presented:
        return presented
    return query_params.get("token", "").strip()


def _client_in_allowlist(client_host: str, allowlist: List[str]) -> bool:
    try:
        ip = ipaddress.ip_address(client_host)
    except ValueError:
        return False
    for entry in allowlist or []:
        try:
            if ip in ipaddress.ip_network(str(entry), strict=False):
                return True
        except ValueError:
            if client_host == str(entry):
                return True
    return False


def _caller_is_trusted(
    *,
    client_host: str,
    headers,
    query_params,
    app_state,
    get_token,
    method: Optional[str] = None,
) -> bool:
    """Core caller-identity decision, independent of ASGI scope type.

    Shared by :func:`_authenticate` (``http`` scope, via the two
    ``BaseHTTPMiddleware`` subclasses) and :func:`authorize_websocket`
    (``websocket`` scope, which no middleware ever sees) so both scopes
    resolve caller trust from exactly one implementation — a route reached
    over a websocket must not be easier to reach than the same route over
    HTTP. Returns ``True`` when the caller may proceed.

    The three source-based rules (no token configured, loopback,
    ``extra_allowlist``) are independent ORs — the order they're checked in
    doesn't change the outcome. The ``Origin`` check ahead of them is not: it
    decides *whether they apply at all*, because a request a browser made on
    behalf of an off-box page arrives from a trusted-looking source without
    being a trusted caller.

    Exempt *paths* are deliberately not part of this: they're an HTTP-only
    concern (login/static must load before a token exists), and a websocket
    route is never exempt.
    """
    token = (get_token() or "").strip()
    presented = _presented_credential(headers, query_params)

    # The three trust rules below (no token configured, loopback, allowlisted
    # peer) all key off *where the connection came from* rather than what it
    # carries. That is the right call for the callers they exist for — the tray,
    # a local script, a peer hub — but a page served from off this machine can
    # make the user's own browser open a connection from exactly those places.
    # Such a request names itself in ``Origin`` and has to present the token
    # like any other unrecognised caller.
    if _is_foreign_origin_request(headers, method):
        return bool(token and presented and hmac.compare_digest(presented, token))

    if not token:
        return True
    if client_host in LOOPBACK_HOSTS and not _is_proxied(headers):
        return True
    cfg = getattr(app_state, "webapp_config", None)
    extra = getattr(cfg, "extra_allowlist", []) if cfg else []
    if _client_in_allowlist(client_host, extra):
        return True

    return bool(presented and hmac.compare_digest(presented, token))


async def authorize_websocket(websocket, get_token) -> bool:
    """Bearer gate for a websocket handshake — call before ``accept()``.

    ``BaseHTTPMiddleware`` only runs for ``http``-scope connections, so
    :class:`BearerTokenMiddleware` never sees a websocket handshake and a
    ``@router.websocket(...)`` route is responsible for its own check.
    Applies the same trust rules as the HTTP path via
    :func:`_caller_is_trusted` (loopback bypass, proxied-loopback
    detection, ``extra_allowlist``, then ``Authorization: Bearer`` or
    ``?token=``).

    Returns ``True`` when the caller may proceed. Otherwise closes the
    socket with 1008 (policy violation) *without* accepting it and returns
    ``False``, so the caller should simply ``return``.
    """
    client_host = websocket.client.host if websocket.client else ""
    if _caller_is_trusted(
        client_host=client_host,
        headers=websocket.headers,
        query_params=websocket.query_params,
        app_state=websocket.app.state,
        get_token=get_token,
        method=None,  # a handshake is a GET, but the route it opens is not read-only
    ):
        return True
    logger.warning("⚠️ websocket handshake refused for %s", client_host or "?")
    await websocket.close(code=1008)
    return False


def _authenticate(
    request: Request,
    get_token,
    path: str,
    exempt_exact: FrozenSet[str],
    exempt_prefixes: Tuple[str, ...],
) -> Optional[JSONResponse]:
    """Shared bearer-token gate for both :class:`BearerTokenMiddleware` (the
    ``/admin`` sub-app) and :class:`ParentBearerTokenMiddleware` (the parent
    hub app) — same loopback/proxy detection, allowlist check, and token
    compare; only the exempt-path set differs between the two apps.

    Returns ``None`` when the request should proceed (caller calls
    ``call_next``), or the 401 :class:`JSONResponse` to return directly
    when it must be blocked.
    """
    if path in exempt_exact or any(path.startswith(p) for p in exempt_prefixes):
        return None
    if _caller_is_trusted(
        client_host=request.client.host if request.client else "",
        headers=request.headers,
        query_params=request.query_params,
        app_state=request.app.state,
        get_token=get_token,
        method=request.method,
    ):
        return None

    return JSONResponse(
        status_code=401,
        content={"detail": "missing or invalid bearer token"},
        headers={"WWW-Authenticate": 'Bearer realm="local-llm-hub"'},
    )


class BearerTokenMiddleware(BaseHTTPMiddleware):
    """Require Authorization: Bearer <token> on /admin endpoints (non-loopback only).

    Paths are the sub-app's view — starlette strips the /admin mount
    prefix before invoking us.
    """

    def __init__(self, app: Any, get_token):
        super().__init__(app)
        self._get_token = get_token

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # ``request.url.path`` here is the original *parent* path — e.g.
        # ``/admin/static/styles.css`` — because Starlette's ``BaseHTTPMiddleware``
        # runs BEFORE the parent's Mount strips the mount prefix. Strip
        # it manually so AUTH_EXEMPT_PREFIXES (``/static/``) matches.
        path = request.url.path
        if path.startswith("/admin"):
            path = path[len("/admin"):] or "/"

        blocked = _authenticate(
            request, self._get_token, path, AUTH_EXEMPT_EXACT, AUTH_EXEMPT_PREFIXES
        )
        if blocked is not None:
            return blocked
        return await call_next(request)


# ----------------------------------------------------------------- parent

# Paths on the *parent* hub app (not the /admin sub-app) that bypass
# the bearer token even for non-loopback callers. The /admin SPA mounts
# under /admin so it has its own enforcement; / itself redirects there.
PARENT_AUTH_EXEMPT_PREFIXES = ("/admin/", "/admin")
PARENT_AUTH_EXEMPT_EXACT = frozenset(
    {
        "/",
        "/health",
        "/info",
        "/favicon.ico",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)


class ParentBearerTokenMiddleware(BaseHTTPMiddleware):
    """Same bearer enforcement as :class:`BearerTokenMiddleware`, but for
    the parent hub app — so a Cloudflare-exposed hub can't have its
    /v1/messages or /v1/chat/completions endpoints hit anonymously.

    The /admin sub-app is exempted here because it owns its own middleware.
    """

    def __init__(self, app: Any, get_token):
        super().__init__(app)
        self._get_token = get_token

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        path = request.url.path
        blocked = _authenticate(
            request, self._get_token, path,
            PARENT_AUTH_EXEMPT_EXACT, PARENT_AUTH_EXEMPT_PREFIXES,
        )
        if blocked is not None:
            return blocked
        return await call_next(request)
