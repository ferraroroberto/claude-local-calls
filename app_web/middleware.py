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
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
from typing import Any, FrozenSet, List, Optional, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

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
) -> bool:
    """Core caller-identity decision, independent of ASGI scope type.

    Shared by :func:`_authenticate` (``http`` scope, via the two
    ``BaseHTTPMiddleware`` subclasses) and :func:`authorize_websocket`
    (``websocket`` scope, which no middleware ever sees) so both scopes
    resolve caller trust from exactly one implementation — a route reached
    over a websocket must not be easier to reach than the same route over
    HTTP. Returns ``True`` when the caller may proceed.

    Every ``True`` below is an independent OR — the *order* they're checked
    in doesn't change the outcome, only whether a token check is reached.
    Exempt *paths* are deliberately not part of this: they're an HTTP-only
    concern (login/static must load before a token exists), and a websocket
    route is never exempt.
    """
    token = (get_token() or "").strip()
    if not token:
        return True
    if client_host in LOOPBACK_HOSTS and not _is_proxied(headers):
        return True
    cfg = getattr(app_state, "webapp_config", None)
    extra = getattr(cfg, "extra_allowlist", []) if cfg else []
    if _client_in_allowlist(client_host, extra):
        return True

    presented = ""
    auth_header = headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        presented = auth_header[7:].strip()
    if not presented:
        # ``x-api-key`` is how the Anthropic SDK authenticates (#461), so an
        # SDK pointed at the hub the ordinary way — ``api_key=<hub token>``,
        # no ``default_headers`` special-casing — presents its credential
        # here. Accepted as an equal alternative to the bearer token: same
        # constant-time compare, same 401 when it's wrong.
        presented = (headers.get("x-api-key", "") or "").strip()
    if not presented:
        presented = query_params.get("token", "").strip()

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
