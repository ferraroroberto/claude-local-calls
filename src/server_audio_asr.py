"""Whisper ASR proxy routes (``/v1/audio/transcriptions``, ``/translations``,
``/health``).

Split out of ``server.py`` originally (the whisper transcription/translation
proxy and the TTS speech proxy together were ~300 lines of multipart-and-httpx
plumbing sitting between the chat routes), then split again out of the
combined ``server_audio.py`` (#451) once that file grew to 820 lines mixing
both audio directions with essentially no shared logic between them — see
``server_audio_tts.py`` for the speech-synthesis half and
``server_audio_common.py`` for the small handful of helpers both need.

The whisper-server already speaks the OpenAI ``/v1/audio/*`` shape, so the
hub mostly forwards bytes — the point of routing through here (rather than
hitting :8090/:8091 directly) is that the observability middleware records
the request in the live ring.

Routes are collected on a module-level :class:`fastapi.APIRouter` and mounted
onto the parent hub app by ``server.py`` via ``include_router``.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from .audio_proxy import build_whisper_upstream_request
from .http_client import get_async_client
from .model_registry import Model
from .remote_proxy import remote_base_url
from .server_audio_common import (
    HEADER_REQUESTED_MODEL,
    HEADER_SERVED_HOST,
    HEADER_SERVED_MODEL,
    _audio_upstream_error,
    _header_safe,
    _remote_audio_headers,
)
from .server_common import current_otel_span, safe_span, stash_trace_id_on_ctx

logger = logging.getLogger(__name__)

router = APIRouter()


def _whisper_model_for_request(model_name: str, *, default_role: str) -> Optional[Model]:
    """Pick a whisper-shaped backend for a request.

    If the caller passed ``model=...`` in the multipart form, try to
    resolve it through the registry first — this is the only path that
    can return a *remote* (``host:`` set) model, e.g. ``model=parakeet``.
    Otherwise fall back to a heuristic based on the endpoint's role,
    restricted to locally-owned backends only — the default role never
    silently starts proxying to a remote host:

      * ``audio_transcribe`` → first whisper backend whose id does NOT
        contain "translate" (the turbo / GPU one).
      * ``audio_translate`` → first whisper backend whose id DOES
        contain "translate" (the medium / CPU sibling).

    Returns ``None`` if no whisper backend is enabled on this host —
    the caller surfaces that as 503.
    """
    from .model_registry import local_models, resolve as _resolve_model

    if model_name:
        m = _resolve_model(model_name)
        if m and m.backend == "whisper" and m.port:
            return m

    whispers = [m for m in local_models() if m.backend == "whisper" and m.port]
    if not whispers:
        return None

    if default_role == "audio_translate":
        for m in whispers:
            if "translate" in m.id.lower():
                return m
    else:  # audio_transcribe — anything that isn't the translate sibling
        for m in whispers:
            if "translate" not in m.id.lower():
                return m

    return whispers[0]


class _BackendUnavailable(Exception):
    """Raised inside a single candidate attempt when that backend is *down* —
    a connection error/timeout or an upstream 502/503/504 — so ``_proxy_audio``
    can fail over to the next model in the role chain (#348). Carries the
    HTTPException to surface if this turns out to be the last candidate."""

    def __init__(self, http_exc: HTTPException) -> None:
        self.http_exc = http_exc


# Role aliases a caller may send as ``model=`` to explicitly ask for the
# failover chain; a *concrete* model id is honoured single-shot instead.
_ROLE_ALIASES = {"audio_transcribe", "audio_translate"}


def _whisper_chain_for_request(model_name: str, *, default_role: str) -> List[Model]:
    """Ordered whisper-shaped candidates to try for this request (#348).

    * An explicit **concrete** model (``model=whisper-vanilla``) → a one-element
      chain: honour it exactly, never fail over to a *different model*, and
      never fall through to the role chain. Preserves #128 — a caller that
      picked ``whisper-vanilla`` to escape the glossary must not silently land
      on turbo — and #412: when that model can't be served here the caller gets
      a 503 naming it, not a 200 produced by some other model. (Host-level
      failover for the *same* model id — the #342 chain — is not a
      substitution and still applies, inside :func:`_forward_to_candidate`.)
    * The **role** path (no ``model``, or a role alias like ``audio_transcribe``)
      → the configured ``roles.audio.<role>`` chain (``model_id`` + ``fallback``),
      resolved via the registry so it can include remote/cross-enabled rows
      (e.g. ``parakeet`` on the Mac). Falling back across models here is the
      whole point of the role — it is only made *observable* (#412), never
      blocked.
    * A ``model=`` the registry has never heard of (OpenAI clients must send
      something, and the SDK's default is ``whisper-1``) is not a request for
      one of our models — it addresses the role, exactly as before.
    * If the config chain is empty/unresolvable → the legacy local-only heuristic
      (:func:`_whisper_model_for_request`), so nothing regresses.

    Raises ``HTTPException`` for an explicit, *known* model id that this hub
    cannot serve — 400 when the id names a non-transcription model, 503 when
    the id names a transcription model this host does not serve. Neither of
    those is an *outage*: a backend that is genuinely down is a different
    condition, reported downstream by :func:`_audio_upstream_error` with a
    message naming the port (#147).
    """
    from .model_registry import audio_role_chain, resolve as _resolve_model, resolve_any

    if model_name and model_name.strip().lower() not in _ROLE_ALIASES:
        # Ownership test first, and case-insensitively (:func:`resolve_any`):
        # "is this one of *our* model ids at all?". An exact-match-only gate
        # let ``model=Whisper`` look identical to ``whisper-1`` and drop the
        # caller onto the role chain — a 200 from another model, the exact
        # behaviour #412 exists to abolish.
        known = resolve_any(model_name)
        if known is not None:
            # Serveability is a separate question, asked with the row's
            # *canonical* id so the host-scoped exact-match ``resolve`` — which
            # every other consumer relies on being exact — is left untouched.
            serveable = _resolve_model(known.id)
            if serveable is not None and serveable.backend == "whisper" and serveable.port:
                return [serveable]
            if known.backend != "whisper" or not known.port:
                logger.warning(
                    "⚠️ audio request rejected: %s is a '%s' model, not a "
                    "transcription backend",
                    known.id, known.backend,
                )
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"model '{model_name}' is not a transcription backend "
                        f"(it is a '{known.backend}' model) — /v1/audio/* only "
                        f"serves whisper-shaped models"
                    ),
                )
            # The *effective* owner, not the statically-preferred one: on a
            # failed-over multi-host row (#342) `known.host` still names the
            # first chain entry, which would point the operator at a machine
            # that no longer serves the model.
            from .model_failover import effective_owner

            owner = effective_owner(known) or known.host or ""
            logger.warning(
                "⚠️ audio request rejected: %s is a configured model this host "
                "does not serve (owner=%s) — refusing to substitute another model",
                known.id, owner or "unset",
            )
            owned_by = f" — host '{owner}' owns it" if owner else ""
            raise HTTPException(
                status_code=503,
                detail=(
                    f"model '{model_name}' is not enabled on this host{owned_by}. "
                    f"Nothing is down: this hub simply does not serve that model, "
                    f"and an explicit model id is never answered by a different "
                    f"one. Send the request to the owning host's hub, or address "
                    f"the role ('{default_role}', or omit model) if a fallback "
                    f"across models is acceptable."
                ),
            )
        # Unknown to the registry → a client-side placeholder, not a model
        # request: fall through to the role chain (and record what served).

    role_key = "translate" if default_role == "audio_translate" else "transcribe"
    chain: List[Model] = []
    seen: set = set()
    for mid in audio_role_chain(role_key):
        m = _resolve_model(mid)
        if m and m.backend == "whisper" and m.port and m.id not in seen:
            seen.add(m.id)
            chain.append(m)
    if chain:
        return chain

    m = _whisper_model_for_request(model_name, default_role=default_role)
    return [m] if m is not None else []


def _served_host_for(target: Model) -> str:
    """Host id that serves ``target`` from this hub's point of view (#412).

    The *effective* owner (#342) when the row declares one — that is the host
    the request is proxied to — and the active host otherwise (a row nobody
    owns is served right here). Best-effort: an unresolvable host profile
    yields ``""`` rather than failing a transcription that already succeeded.
    """
    from .host_profile import resolve as _resolve_host
    from .model_failover import effective_owner

    try:
        owner = effective_owner(target)
    except Exception:  # noqa: BLE001 — never fail a served request over a label
        owner = None
    if owner:
        return owner
    try:
        return _resolve_host().id
    except Exception:  # noqa: BLE001
        return ""


def _served_identity(target: Model, upstream, *, remote: bool) -> Tuple[str, str]:
    """Who actually served this response — ``(served_model, served_host)``.

    Normally that is the candidate this hub dispatched to, on its owning host.
    The exception is a **remote role hop**: ``_proxy_audio`` forwards the body
    with no ``model`` field, so the peer hub re-resolves the role against *its*
    own config and may legitimately answer off a different chain member. Its
    ``X-Hub-Served-Model`` / ``X-Hub-Served-Host`` are the honest answer and
    win; a peer too old to send them — or a bare whisper-server, which never
    does — falls back to what this hub dispatched (#412).
    """
    peer_model = ""
    peer_host = ""
    if remote:
        try:
            peer_model = _header_safe(str(upstream.headers.get(HEADER_SERVED_MODEL) or "").strip())
            peer_host = _header_safe(str(upstream.headers.get(HEADER_SERVED_HOST) or "").strip())
        except Exception:  # noqa: BLE001 — a malformed peer header is not fatal
            peer_model = peer_host = ""
    return peer_model or target.id, peer_host or _served_host_for(target)


def _observable_audio_error(exc: HTTPException, ctx, requested: str) -> HTTPException:
    """Make an audio *failure* as observable as an audio success (#412).

    A strict rejection is precisely the event an operator has to be able to
    see, yet it used to be the least visible thing the hub emitted: the reject
    fired before anything stamped ``request.state.obs_ctx``, so the ring
    recorded a blank row (``model=''``, ``backend=''``, counter key
    ``unknown``) and ``ObservatoryMiddleware`` could not recover it — its
    fallback peeks a *JSON* body and a multipart request never parses. Stamp
    the detail on the context and echo the same header trio a 200 carries,
    with the served pair empty because nothing served.
    """
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    if ctx is not None and not getattr(ctx, "error_detail", ""):
        ctx.error_detail = detail
    headers = dict(exc.headers or {})
    headers.setdefault(HEADER_REQUESTED_MODEL, _header_safe(requested))
    headers.setdefault(HEADER_SERVED_MODEL, _header_safe(getattr(ctx, "served_model", "") or ""))
    headers.setdefault(HEADER_SERVED_HOST, _header_safe(getattr(ctx, "served_host", "") or ""))
    exc.headers = headers
    return exc


async def _proxy_audio(request: Request, *, default_role: str, ctx_path: str) -> Response:
    """Forward a multipart audio request to a whisper backend, failing over
    across the role's model chain (#348).

    The whisper-server already speaks the OpenAI ``/v1/audio/*`` shape, so we
    forward the bytes/form and pass the response back — the point of going
    through the hub (vs hitting :8090/:8091/:8098 directly) is the observability
    ring. When a *role-addressed* request's primary backend is unavailable (a
    connection error/timeout or a 502/503/504), it transparently retries the
    next model in ``roles.audio.<role>`` instead of erroring — so a dead
    ``parakeet@mac`` falls through to whisper, never a failed dictation. That
    fallback is no longer *silent*: who served, and on which host, is stamped
    on the request record and echoed in the response headers (#412). An
    explicit ``model=`` is strict and never substituted at all — see
    :func:`_whisper_chain_for_request`. A real client error (4xx) or a 200 is
    returned as-is.

    For ``audio_translate`` the raw-bytes path can't be used: whisper-server
    exposes a single inference endpoint (``/v1/audio/transcriptions``) and wants
    whisper.cpp's ``translate=true`` boolean, not OpenAI's ``task=translate``.
    The form is parsed+rewritten once up front; its file bytes are read into
    ``files`` there, so the payload is safe to resend to each candidate.
    """
    body = await request.body()

    # Peek the ``model`` field out of the multipart body to choose a
    # backend. python-multipart parsing is overkill — the field shows
    # up as ``Content-Disposition: form-data; name="model"`` followed
    # by a couple of CRLF lines and the value. Best-effort regex.
    #
    # Scan the first 16 KB first (the cheap path — standard SDK clients
    # serialize plain form fields before the file part, so ``model``
    # lands in the head). Fall back to the whole body if it's not there:
    # a client that puts a large file *before* the model field would
    # otherwise misroute to the default turbo (#128) — silently landing
    # on the glossary path the caller chose ``whisper-vanilla`` to escape.
    model_name = ""
    try:
        pattern = rb'name="model"\r?\n\r?\n([^\r\n]+)'
        match = re.search(pattern, body[: 16 * 1024]) or re.search(pattern, body)
        if match:
            # Sanitize at the source: this value is echoed back in a response
            # header and the pattern above only excludes CR/LF, so a control
            # byte would otherwise reach h11 (see :func:`_header_safe`).
            model_name = _header_safe(
                match.group(1).decode("ascii", errors="ignore").strip()
            )
    except Exception:  # noqa: BLE001
        pass

    # Stamp the observability context *before* anything below can raise (#412).
    # A strict rejection is the event that most needs to be visible, and the
    # middleware cannot fill these in for a multipart request — see
    # :func:`_observable_audio_error`.
    ctx = getattr(request.state, "obs_ctx", None)
    requested = model_name or default_role
    if ctx is not None:
        ctx.model = requested
        ctx.backend = "whisper"

    try:
        return await _dispatch_audio(
            request, body=body, model_name=model_name, requested=requested,
            default_role=default_role, ctx_path=ctx_path, ctx=ctx,
        )
    except HTTPException as exc:
        raise _observable_audio_error(exc, ctx, requested)


async def _dispatch_audio(
    request: Request, *, body: bytes, model_name: str, requested: str,
    default_role: str, ctx_path: str, ctx,
) -> Response:
    """Resolve the chain, build the reusable payload, and walk the candidates.

    Split out of :func:`_proxy_audio` purely so every ``HTTPException`` raised
    on the way — strict rejection, malformed multipart, whole chain down — is
    funnelled through one observability wrapper instead of each raise site
    remembering to stamp the context (#412).
    """
    chain = _whisper_chain_for_request(model_name, default_role=default_role)
    if not chain:
        raise HTTPException(status_code=503, detail="no whisper backend enabled on this host")

    # Build the reusable upstream payload once — bytes-based, so it can be
    # resent to each candidate without re-reading the request stream.
    if default_role == "audio_translate":
        # whisper-server exposes a single inference path and wants whisper.cpp's
        # `translate=true` boolean, not OpenAI's `task=translate` string. Parse +
        # rewrite via the shared helper (the lazy-load shim in
        # whisper_translate_proxy.py calls the same one — issue #132).
        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid multipart body: {exc}")
        upload, data, files = await build_whisper_upstream_request(form)
        if upload is None:
            raise HTTPException(status_code=400, detail="missing required form field: file")
        send = {"files": files, "data": data}
        upstream_path = "/v1/audio/transcriptions"
    else:
        fwd_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() in {"content-type", "accept"}
        }
        send = {"content": body, "headers": fwd_headers}
        upstream_path = ctx_path

    span = current_otel_span()
    client = get_async_client()  # fetch once, reuse for every candidate

    last_http_exc: Optional[HTTPException] = None
    for idx, target in enumerate(chain):
        is_last = idx == len(chain) - 1
        try:
            response = await _forward_to_candidate(
                target, send, upstream_path, default_role, requested,
                ctx, span, client, connect_fast=not is_last,
            )
        except _BackendUnavailable as bu:
            last_http_exc = bu.http_exc
            if not is_last:
                logger.warning(
                    "🔁 audio failover: %s unavailable (%s) — trying next in chain",
                    target.id, bu.http_exc.status_code,
                )
            continue
        # Say who answered whenever it is not who this hub dispatched to: a
        # chain fallback (idx > 0), or a peer hub that re-resolved the role on
        # its own side. A degraded tier should be visible in the log, not only
        # in the request ring (#412).
        served = response.headers.get(HEADER_SERVED_MODEL) or target.id
        if idx > 0:
            logger.info(
                "ℹ️ audio substitution: requested=%s served=%s (role fallback, "
                "primary %s unavailable)",
                requested, served, chain[0].id,
            )
        elif served != target.id:
            logger.info(
                "ℹ️ audio substitution: requested=%s served=%s (peer %s re-resolved "
                "the role)",
                requested, served, response.headers.get(HEADER_SERVED_HOST) or "?",
            )
        return response
    raise last_http_exc or HTTPException(status_code=503, detail="no audio backend answered")


async def _forward_to_candidate(
    target: Model, send: dict, upstream_path: str, default_role: str,
    requested: str, ctx, span, client, *, connect_fast: bool,
) -> Response:
    """POST the prepared payload to one candidate backend and return its
    Response, or raise :class:`_BackendUnavailable` when the backend is down so
    the caller can try the next model (#348). Records obs/OTel against the model
    that actually served, and applies the #90 glossary to a 200 transcript.

    The request record carries *three* names (#412): ``model`` is what the
    caller asked for (stamped by :func:`_proxy_audio` before anything can
    fail), while ``served_model`` and ``served_host`` are the chain member and
    the machine that **answered** — so the Telemetry tab and
    ``/admin/api/hub/requests/recent`` show ``requested=audio_transcribe
    served=parakeet @mac-mini-m4`` instead of an indistinguishable success.

    The served pair is written only *after* an upstream answers: stamping it up
    front attributed a whole-chain 503 — and its error counter — to a model
    that never ran. The same trio is echoed as ``X-Hub-Requested-Model`` /
    ``X-Hub-Served-Model`` / ``X-Hub-Served-Host`` response headers: additive
    metadata every OpenAI/Anthropic client ignores.
    """
    import httpx as _httpx

    port = target.port
    remote = remote_base_url(target)

    if span is not None and hasattr(span, "set_attribute"):
        with safe_span("whisper_attrs"):
            span.set_attribute("gen_ai.system", "whisper")
            span.set_attribute("gen_ai.operation.name", default_role)
            span.set_attribute("gen_ai.request.model", requested)
            span.set_attribute("whisper.port", int(port))
            span.set_attribute("whisper.model_id", target.id)
    stash_trace_id_on_ctx(ctx, span)

    base = remote if remote else f"http://127.0.0.1:{port}"
    url = f"{base}{upstream_path}"
    headers = dict(send.get("headers") or {})
    if remote:
        headers.update(_remote_audio_headers(target) or {})
    # A dead primary should fail over fast (short connect); the last resort keeps
    # a patient connect. Read stays long — a transcription can take a while.
    timeout = _httpx.Timeout(4.0 if connect_fast else 30.0, read=300.0, write=60.0, pool=10.0)
    post_kwargs: dict = {"headers": headers or None, "timeout": timeout}
    if "files" in send:
        post_kwargs["files"] = send["files"]
        post_kwargs["data"] = send["data"]
    else:
        post_kwargs["content"] = send["content"]

    try:
        upstream = await client.post(url, **post_kwargs)
    except _httpx.HTTPError as exc:
        raise _BackendUnavailable(_audio_upstream_error(exc, backend="whisper-server", port=port))
    if upstream.status_code in (502, 503, 504):
        raise _BackendUnavailable(HTTPException(
            status_code=upstream.status_code,
            detail=f"whisper backend {target.id} unavailable ({upstream.status_code})",
        ))

    # A model has now genuinely answered — only here is it honest to say who
    # served (#412). Before this point every candidate may still fail over.
    served_model, served_host = _served_identity(target, upstream, remote=bool(remote))
    if ctx is not None:
        ctx.served_model = served_model
        ctx.served_host = served_host
    if span is not None and hasattr(span, "set_attribute"):
        with safe_span("whisper_served"):
            span.set_attribute("gen_ai.response.model", served_model)
            span.set_attribute("hub.served_host", served_host)

    # Apply the committed transcription glossary (issue #90) to a 200 transcript
    # before returning. Deterministic literal fixes (e.g. "cloud code" →
    # "Claude Code") for acoustically-strong errors biasing can't solve. Wrapped
    # defensively: a broken glossary must never break the passthrough.
    out_content = upstream.content
    if upstream.status_code == 200:
        try:
            from .transcription_glossary import apply_to_response, load_rules

            rules = load_rules()
            if rules:
                out_content = apply_to_response(
                    upstream.content, upstream.headers.get("content-type"), rules,
                )
        except Exception:  # noqa: BLE001 — never let post-processing fail the proxy
            out_content = upstream.content

    # Drop the peer hub's own X-Hub-* trio from the passthrough: this hub
    # re-states them below (folding in the peer's answer via
    # :func:`_served_identity`), and leaving the originals in would emit the
    # header twice with two different requested-model values.
    _drop = {
        "content-length", "transfer-encoding", "connection",
        HEADER_REQUESTED_MODEL.lower(), HEADER_SERVED_MODEL.lower(),
        HEADER_SERVED_HOST.lower(),
    }
    out_headers = {
        k: v for k, v in upstream.headers.items() if k.lower() not in _drop
    }
    out_headers[HEADER_REQUESTED_MODEL] = _header_safe(requested)
    out_headers[HEADER_SERVED_MODEL] = _header_safe(served_model)
    out_headers[HEADER_SERVED_HOST] = _header_safe(served_host)
    return Response(
        content=out_content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
        headers=out_headers,
    )


@router.post("/v1/audio/transcriptions")
async def audio_transcriptions(request: Request) -> Response:
    """Proxy transcription requests through the hub so they land in the
    observability ring. Clients that point directly at :8090 still
    work but are invisible to the admin UI — pointing at :8000 here
    makes them visible without changing the request shape.
    """
    return await _proxy_audio(
        request, default_role="audio_transcribe",
        ctx_path="/v1/audio/transcriptions",
    )


@router.post("/v1/audio/translations")
async def audio_translations(request: Request) -> Response:
    """Companion to :func:`audio_transcriptions` for the ``task=translate``
    case. Routes to the ``audio_translate`` role's port (medium, CPU).
    """
    return await _proxy_audio(
        request, default_role="audio_translate",
        ctx_path="/v1/audio/translations",
    )


@router.get("/v1/audio/health")
def audio_health() -> Response:
    """Probe-only liveness of the audio backends — lets a consumer preflight
    instead of discovering an outage one failed transcription at a time (#147).

    Reports each enabled whisper / TTS backend with its port and whether it is
    currently reachable (a cheap GET to the backend, never a transcription).
    ``status`` is ``ok`` when every enabled audio backend answers, ``degraded``
    when at least one is down, and ``none`` when no audio backend is enabled on
    this host. A degraded/none result returns HTTP 503 so a consumer can branch
    on the status code alone; ``ok`` returns 200.

    Defined as a sync route on purpose: ``is_reachable`` does blocking socket
    probes, so FastAPI runs it in a threadpool rather than stalling the loop.
    """
    import json as _json

    from .backend_process import is_reachable
    from .host_profile import resolve as _resolve_host
    from .model_failover import effective_owner
    from .model_registry import local_models

    backends = []
    # Backends this host *currently serves* only — a remote-owned row's
    # liveness is the owning host's own /v1/audio/health concern, not
    # something this loopback probe can answer correctly (see
    # app_web/routers/models.py for the cross-host merge that *does* surface
    # remote rows, in the admin UI). With #342 chains, ``local_models`` also
    # contains rows this host merely *stands by for* — those are filtered by
    # effective owner so a standby candidate is never reported "down" here.
    active_id = _resolve_host().id
    audio = [
        m for m in local_models()
        if m.backend in ("whisper", "tts") and m.port
        and effective_owner(m) in (None, active_id)
    ]
    for m in audio:
        reachable = is_reachable(m, timeout=1.0)
        backends.append({
            "id": m.id,
            "backend": m.backend,
            "port": m.port,
            "reachable": reachable,
        })

    if not backends:
        status, code = "none", 503
    elif all(b["reachable"] for b in backends):
        status, code = "ok", 200
    else:
        status, code = "degraded", 503

    return Response(
        content=_json.dumps({"status": status, "backends": backends}),
        status_code=code,
        media_type="application/json",
    )
