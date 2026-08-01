"""Playground tab API — model dropdown + send (proxies in-process to /v1/messages)."""

from __future__ import annotations

import base64
import logging
import mimetypes
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from src.host_profile import hub_port
from src.http_client import get_async_client
from src.model_registry import enabled_models, resolve as resolve_model
from src.tts_engines import capabilities_for_engine

from .models import list_models_for_admin

logger = logging.getLogger(__name__)
router = APIRouter()

_IMAGE_MEDIA_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}


@router.get("/api/playground/models")
async def playground_models() -> Dict[str, Any]:
    """List every enabled model that can answer chat-style requests.

    Whisper rows are excluded — they're ASR-only and don't speak the
    /v1/messages shape.
    """
    rows: List[Dict[str, Any]] = []
    for m in enabled_models():
        if m.backend == "whisper":
            continue
        # Image-generation rows don't speak the chat shape — they're served
        # by the dedicated image card, not the chat dropdown.
        if getattr(m, "image_gen", False):
            continue
        rows.append(
            {
                "id": m.id,
                "display_name": m.display_name,
                "backend": m.backend,
                "aliases": list(m.aliases or []),
                "image_capable": m.backend in ("claude", "gemini"),
            }
        )
    return {"models": rows}


@router.get("/api/playground/image_models")
async def playground_image_models() -> Dict[str, Any]:
    """List enabled image-generation models for the Playground image card."""
    rows: List[Dict[str, Any]] = []
    for m in enabled_models():
        if getattr(m, "image_gen", False):
            rows.append(
                {
                    "id": m.id,
                    "display_name": m.display_name,
                    "backend": m.backend,
                    "aliases": list(m.aliases or []),
                }
            )
    return {"models": rows}


@router.get("/api/playground/tts_models")
async def playground_tts_models() -> Dict[str, Any]:
    """List configured TTS backends, runtime state, and UI capabilities."""
    runtime = await list_models_for_admin()
    reachable_by_id = {
        row.get("id"): bool(row.get("reachable"))
        for row in runtime.get("models", [])
        if isinstance(row, dict)
    }
    rows: List[Dict[str, Any]] = []
    for m in enabled_models():
        if m.backend != "tts":
            continue
        rows.append(
            {
                "id": m.id,
                "display_name": m.display_name,
                "engine": m.tts_engine,
                "aliases": list(m.aliases or []),
                "reachable": reachable_by_id.get(m.id, False),
                "capabilities": capabilities_for_engine(m.tts_engine or ""),
            }
        )
    rows.sort(key=lambda row: 0 if "audio_speech" in row.get("aliases", []) else 1)
    return {"models": rows}


@router.post("/api/playground/speak")
async def playground_speak(
    model: str = Form(...),
    input: str = Form(...),
    voice: str = Form(""),
    response_format: str = Form("wav"),
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5),
    speed: float = Form(1.0),
    stream: bool = Form(False),
) -> Response:
    """Synthesize speech through the hub's own ``/v1/audio/speech`` proxy.

    Same loopback-proxy pattern as :func:`playground_send` — the request
    lands in the observability ring like any external call. Returns the raw
    audio bytes for the SPA's ``<audio>`` player. With ``stream=true`` the
    hub's streaming shape (``stream_format: "audio"``) is forwarded through
    so the SPA can play audio as it synthesizes and time the first chunk.
    """
    import httpx

    target = resolve_model(model)
    if target is None or target.backend != "tts":
        raise HTTPException(status_code=400, detail=f"not a TTS model: {model!r}")
    if not input.strip():
        raise HTTPException(status_code=400, detail="input is empty")

    payload: Dict[str, Any] = {
        "model": target.display_name,
        "input": input,
        "voice": voice,
        "response_format": response_format,
        "exaggeration": exaggeration,
        "cfg_weight": cfg_weight,
        "speed": speed,
    }
    if stream:
        payload["stream_format"] = "audio"
    url = f"http://127.0.0.1:{hub_port()}/v1/audio/speech"

    if stream:
        client = get_async_client()
        stream_cm = client.stream("POST", url, json=payload, timeout=300.0)
        try:
            upstream = await stream_cm.__aenter__()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"upstream error: {exc}")
        if not upstream.is_success:
            detail = (await upstream.aread()).decode("utf-8", "replace") or f"HTTP {upstream.status_code}"
            status = upstream.status_code
            await stream_cm.__aexit__(None, None, None)
            raise HTTPException(status_code=status, detail=str(detail)[:500])

        async def _forward():
            try:
                async for piece in upstream.aiter_bytes():
                    yield piece
            finally:
                await stream_cm.__aexit__(None, None, None)

        out_headers = {}
        sr = upstream.headers.get("x-sample-rate")
        if sr:
            out_headers["X-Sample-Rate"] = sr
        return StreamingResponse(
            _forward(),
            media_type=upstream.headers.get("content-type", "audio/wav"),
            headers=out_headers,
        )

    try:
        r = await get_async_client().post(url, json=payload, timeout=300.0)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"upstream error: {exc}")
    if not r.is_success:
        detail = r.text or f"HTTP {r.status_code}"
        try:
            body = r.json()
            detail = body.get("detail") or detail
        except Exception:  # noqa: BLE001
            pass
        raise HTTPException(status_code=r.status_code, detail=str(detail)[:500])
    return Response(content=r.content, media_type=r.headers.get("content-type", "audio/wav"))


@router.post("/api/playground/send")
async def playground_send(
    model: str = Form(...),
    prompt: str = Form(...),
    max_tokens: int = Form(512),
    system: Optional[str] = Form(None),
    attachment: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """Send a single-turn prompt through the hub. Returns text + usage.

    We proxy through the hub's *own* ``/v1/messages`` endpoint over
    loopback so the routing/observability path is identical to a real
    external call — the playground gets recorded in the live request
    ring like any other request, which is exactly what an operator wants
    when sanity-checking a new backend.
    """
    import httpx

    target = resolve_model(model)
    if target is None:
        raise HTTPException(status_code=400, detail=f"unknown model {model!r}")

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    if attachment is not None and attachment.filename:
        raw = await attachment.read()
        suffix = (attachment.filename.rsplit(".", 1)[-1] or "").lower()
        b64 = base64.b64encode(raw).decode("ascii")
        if suffix in _IMAGE_MEDIA_TYPES:
            # Images route through the dedicated image block.
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": _IMAGE_MEDIA_TYPES[suffix],
                        "data": b64,
                    },
                }
            )
        else:
            # Everything else (PDF, JSON, CSV, code, …) is a document
            # block; the CLI attaches it as an @file the model can read.
            media = (
                mimetypes.guess_type(attachment.filename)[0]
                or attachment.content_type
                or "application/octet-stream"
            )
            content.append(
                {
                    "type": "document",
                    "source": {"type": "base64", "media_type": media, "data": b64},
                }
            )

    payload: Dict[str, Any] = {
        "model": target.display_name,
        "max_tokens": int(max_tokens),
        "messages": [{"role": "user", "content": content}],
    }
    if system:
        payload["system"] = system

    url = f"http://127.0.0.1:{hub_port()}/v1/messages"
    try:
        r = await get_async_client().post(url, json=payload, timeout=120.0)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"upstream error: {exc}")
    body: Dict[str, Any] = {}
    try:
        body = r.json()
    except Exception:  # noqa: BLE001
        body = {}
    if not r.is_success:
        # /v1/messages answers errors in the Anthropic envelope (#460);
        # `detail` is kept as a fallback for anything else on this path.
        err = body.get("error") if isinstance(body, dict) else None
        detail = (
            (err.get("message") if isinstance(err, dict) else None)
            or (body.get("detail") if isinstance(body, dict) else None)
            or r.text
            or f"HTTP {r.status_code}"
        )
        raise HTTPException(status_code=r.status_code, detail=str(detail)[:500])

    text = ""
    blocks = body.get("content") if isinstance(body, dict) else None
    if isinstance(blocks, list):
        text = "\n".join(
            (b.get("text") or "") for b in blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return {
        "text": text,
        "stop_reason": body.get("stop_reason"),
        "usage": body.get("usage") or {},
    }


@router.post("/api/playground/generate_image")
async def playground_generate_image(
    model: str = Form("gemini_image"),
    prompt: str = Form(...),
    image: Optional[UploadFile] = File(None),
) -> Response:
    """Generate (or edit) an image through the hub's own image endpoints.

    Same loopback-proxy pattern as :func:`playground_send`: with no upload it
    POSTs ``/v1/images/generations`` (text→image); with an upload it POSTs
    ``/v1/images/edits`` (image+prompt→edited image). Either way the request
    lands in the observability ring. Returns the raw image bytes for the SPA's
    ``<img>`` preview. Editing is slow (procedural-agentic), so the timeout is
    generous.
    """
    import httpx

    target = resolve_model(model)
    if target is None or not getattr(target, "image_gen", False):
        raise HTTPException(
            status_code=400, detail=f"not an image-generation model: {model!r}")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is empty")

    base = f"http://127.0.0.1:{hub_port()}"
    client = get_async_client()
    try:
        if image is not None and image.filename:
            raw = await image.read()
            files = {
                "image": (
                    image.filename, raw,
                    image.content_type or "application/octet-stream",
                )
            }
            data = {"model": model, "prompt": prompt}
            r = await client.post(
                base + "/v1/images/edits", files=files, data=data, timeout=900.0)
        else:
            r = await client.post(
                base + "/v1/images/generations",
                json={"model": model, "prompt": prompt},
                timeout=900.0,
            )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"upstream error: {exc}")

    if not r.is_success:
        detail = r.text or f"HTTP {r.status_code}"
        try:
            detail = r.json().get("detail") or detail
        except Exception:  # noqa: BLE001
            pass
        raise HTTPException(status_code=r.status_code, detail=str(detail)[:500])

    body = r.json()
    b64 = (body.get("data") or [{}])[0].get("b64_json")
    if not b64:
        raise HTTPException(status_code=502, detail="no image in hub response")
    img = base64.b64decode(b64)
    # Sniff the real format — artifacts are usually PNG but can be JPEG.
    from src.gemini_cli import _sniff_image_media_type

    media_type = _sniff_image_media_type(img) or "image/png"
    return Response(content=img, media_type=media_type)
