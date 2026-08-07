"""Text-to-speech proxy route (``/v1/audio/speech``).

Split out of ``server.py`` originally alongside the whisper ASR proxy, then
split again out of the combined ``server_audio.py`` (#451) — see
``server_audio_asr.py`` for the transcription/translation half and
``server_audio_common.py`` for the small handful of helpers both need.

Routes are collected on a module-level :class:`fastapi.APIRouter` and mounted
onto the parent hub app by ``server.py`` via ``include_router``.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from . import on_demand as _on_demand
from .http_client import get_async_client
from .model_registry import Model
from .remote_proxy import remote_base_url
from .server_audio_asr import router
from .server_audio_common import _audio_upstream_error, _remote_audio_headers
from .server_common import current_otel_span, ensure_backend_ready_or_503, safe_span, stash_trace_id_on_ctx

logger = logging.getLogger(__name__)


def _tts_model_for_request(model_name: str) -> Optional[Model]:
    """Pick a TTS backend for a ``/v1/audio/speech`` request.

    Resolve an explicit ``model`` through the registry first — the only path
    that can return a *remote* (``host:`` set) model, e.g. ``model=mac_say``.
    An unresolvable explicit model returns ``None`` rather than silently
    selecting an English backend. Only an omitted model falls back to the
    ``audio_speech`` role (Piper), then the first enabled *local* TTS
    backend — the default role never silently proxies to a remote host.
    Returns ``None`` if no TTS backend is enabled on this host.
    """
    from .model_registry import local_models, resolve as _resolve_model

    if model_name:
        m = _resolve_model(model_name)
        if m and m.backend == "tts" and m.port:
            return m
        return None

    tts = [m for m in local_models() if m.backend == "tts" and m.port]
    if not tts:
        return None
    for m in tts:
        if "audio_speech" in (m.aliases or []):
            return m
    return tts[0]


@router.post("/v1/audio/speech")
async def audio_speech(request: Request) -> Response:
    """Proxy text-to-speech requests through the hub so they land in the
    observability ring. The inverse of :func:`server_audio_asr.audio_transcriptions`.

    Body is the OpenAI JSON shape ``{model, input, voice, response_format,
    speed}`` (plus Chatterbox's ``exaggeration`` / ``cfg_weight``). Clients
    may also POST directly to the backend port (:8096 / :8092 / :8093 / :8095) for lower
    overhead, bypassing the hub's capture.
    """
    import json as _json

    import httpx as _httpx

    body = await request.body()
    model_name = ""
    stream_format = ""
    try:
        parsed = _json.loads(body or b"{}")
        if isinstance(parsed, dict):
            model_name = str(parsed.get("model") or "")
            stream_format = str(parsed.get("stream_format") or "").strip().lower()
    except Exception:  # noqa: BLE001
        pass

    target = _tts_model_for_request(model_name)
    if target is None:
        if model_name:
            raise HTTPException(
                status_code=400,
                detail=f"unknown or unsupported TTS model: {model_name}",
            )
        raise HTTPException(status_code=503, detail="no TTS backend enabled on this host")
    port = target.port
    remote = remote_base_url(target)

    ctx = getattr(request.state, "obs_ctx", None)
    if ctx is not None:
        ctx.model = model_name
        ctx.backend = "tts"

    span = current_otel_span()
    if span is not None and hasattr(span, "set_attribute"):
        with safe_span("tts_attrs"):
            span.set_attribute("gen_ai.system", "tts")
            span.set_attribute("gen_ai.operation.name", "audio_speech")
            if model_name:
                span.set_attribute("gen_ai.request.model", model_name)
            span.set_attribute("tts.port", int(port))
    stash_trace_id_on_ctx(ctx, span)

    upstream_url = f"{remote}/v1/audio/speech" if remote else f"http://127.0.0.1:{port}/v1/audio/speech"
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() in {"content-type", "accept"}
    }
    if remote:
        headers.update(_remote_audio_headers(target) or {})

    # On-demand lifecycle (#422): a cold ``startup: on_demand`` local TTS
    # backend is spawned here and the request blocks until it answers
    # (distinct 503 on load failure) — same hook as the chat routes. Off the
    # event loop: the readiness poll is a blocking sleep/probe cycle.
    if remote is None:
        import asyncio as _asyncio
        await _asyncio.to_thread(ensure_backend_ready_or_503, target)

    def _passthrough_headers(upstream) -> dict:
        return {
            k: v for k, v in upstream.headers.items()
            if k.lower() not in {"content-length", "transfer-encoding", "connection"}
        }

    # The in-flight mark covers *every* exit below by default (#470): the
    # ``with`` finishes it on a return or a raise, so a future branch can't
    # forget one and pin the idle-unload window open. Only the streaming
    # branch — whose response outlives this function — opts out, and only
    # after the upstream connection is established.
    with _on_demand.tracking(target, remote) as track:
        # Streaming synth: hold the upstream connection open and forward bytes
        # as they arrive, so time-to-first-audio stays low. The obs middleware
        # still records this entry on response, exactly like the chat-stream path.
        if stream_format == "audio":
            client = get_async_client()
            stream_cm = client.stream("POST", upstream_url, content=body, headers=headers)
            try:
                upstream = await stream_cm.__aenter__()
            except _httpx.HTTPError as exc:
                raise _audio_upstream_error(exc, backend="tts-server", port=port)
            track.detach()

            async def _forward():
                try:
                    async for piece in upstream.aiter_bytes():
                        yield piece
                finally:
                    track.finish()
                    await stream_cm.__aexit__(None, None, None)

            return StreamingResponse(
                _forward(),
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type"),
                headers=_passthrough_headers(upstream),
            )

        try:
            client = get_async_client()
            upstream = await client.post(upstream_url, content=body, headers=headers)
        except _httpx.HTTPError as exc:
            raise _audio_upstream_error(exc, backend="tts-server", port=port)

        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
            headers=_passthrough_headers(upstream),
        )
