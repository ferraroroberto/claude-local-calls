"""Image generation + edit routes (OpenAI ``/v1/images/*`` shape).

Split out of ``server.py`` so the routing/chat core stays readable. Both routes
guard on a row flagged ``image_gen`` and 400 everything else.

Two image backends exist, and ``/v1/images/generations`` dispatches on
``model.backend``:

* ``gemini`` — Google's Imagen, driven through the Antigravity CLI's agentic
  tool harness (there is no Nano Banana picker model — issue #114). A
  subscription path with no local process.
* ``comfyui`` — FLUX.1 [dev] on this host's own GPU (#492). An ordinary
  ``models.yaml`` backend process, so it goes through the on-demand lifecycle
  (#422): the first request spawns ComfyUI and waits for it, and the idle
  watchdog unloads it again.

``/v1/images/edits`` stays **gemini-only** — editing on the ComfyUI path needs a
different workflow graph (img2img/inpaint) and is deliberately out of #492's
scope.

The routes are collected on a module-level :class:`fastapi.APIRouter` and
mounted onto the parent hub app by ``server.py`` via ``include_router``.
"""

from __future__ import annotations

import base64
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import on_demand
from .comfyui_client import ComfyUIError, checkpoint_name_for, generate_image
from .gemini_cli import GeminiCLIError, call_gemini_image
from .image_sizes import DEFAULT_SIZE, ImageSizeError, parse_size
from .model_registry import Model
from .observability import record_genai_metrics, set_genai_request_attrs
from .remote_proxy import remote_base_url
from .server_common import (
    client_id_from,
    current_otel_span,
    resolve_model_or_400,
    stash_trace_id_on_ctx,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Backends that can serve POST /v1/images/generations.
IMAGE_BACKENDS = ("gemini", "comfyui")


def _generate_via_comfyui(
    model: Model, prompt: str, width: int, height: int, refine: bool,
) -> dict:
    """Run one generation on the local ComfyUI backend, loading it if cold.

    ``ensure_ready`` spawns the process and blocks until it answers — safe here
    because FastAPI runs this ``def`` route in a worker thread. The tracking
    context keeps the idle watchdog from unloading the backend mid-generation,
    which matters more than usual here: a cold load plus a 20-step sample can
    outlast a short idle window on its own.
    """
    if remote_base_url(model) is not None:
        # Owned by another host. Unlike the chat paths there is no image proxy,
        # so say so precisely instead of failing later on a dead loopback port.
        raise HTTPException(
            status_code=503,
            detail=(
                f"model {model.id!r} is owned by host {model.host!r} and image "
                "generation is not proxied between hubs — call that host's hub "
                "directly."
            ),
        )
    try:
        on_demand.ensure_ready(model)
    except on_demand.OnDemandNotReady as e:
        raise HTTPException(status_code=503, detail=str(e))

    with on_demand.tracking(model):
        return generate_image(
            prompt,
            base_url=f"http://127.0.0.1:{model.port}",
            ckpt_name=checkpoint_name_for(model.model_path),
            width=width,
            height=height,
            refine=refine,
        )


class ImagesGenerationRequest(BaseModel):
    model: str
    prompt: str
    n: int = 1
    response_format: str = "b64_json"
    # Preset name ("4k") or "WIDTHxHEIGHT" (#497). Honoured by comfyui rows;
    # accepted and ignored by gemini, where Imagen controls its own dimensions
    # — deliberately not a 400, because OpenAI clients send `size` routinely
    # and rejecting it would break every existing caller.
    size: str = DEFAULT_SIZE
    # Second low-denoise pass over an upscaled image. Only meaningful for
    # sizes above the native ceiling; a no-op otherwise.
    refine: bool = False


@router.post("/v1/images/generations")
def images_generations(req: ImagesGenerationRequest, request: Request) -> JSONResponse:
    """Generate an image and return it OpenAI-shape (``data[].b64_json``).

    Routes to any row flagged ``image_gen`` on an image-capable backend —
    ``gemini`` (Imagen, subscription) or ``comfyui`` (FLUX, local GPU). Every
    other backend is text/audio-only and 400s. The call lands in the
    observability ring like other hub traffic.
    """
    model = resolve_model_or_400(req.model)
    if not (model.backend in IMAGE_BACKENDS and model.image_gen):
        raise HTTPException(
            status_code=400,
            detail=(
                f"model {req.model!r} ({model.display_name}) is not an "
                "image-generation model. Use 'gemini_image' (Imagen) or "
                "'flux1_local' (local FLUX) instead."
            ),
        )
    if req.n != 1:
        raise HTTPException(
            status_code=400,
            detail="only n=1 is supported for image generation",
        )
    if req.response_format != "b64_json":
        raise HTTPException(
            status_code=400,
            detail="only response_format='b64_json' is supported",
        )
    # Validate `size` only on a backend that honours it. Validating a field we
    # then ignore would be user-hostile in a specific way: an Imagen request
    # for "1920x1080" would be rejected with a message about FLUX's 16-pixel
    # grid, for a constraint that does not apply to the model being called.
    # Imagen ignores the field completely, so it ignores malformed values too.
    width = height = None
    if model.backend == "comfyui":
        try:
            width, height = parse_size(req.size)
        except ImageSizeError as e:
            raise HTTPException(status_code=400, detail=str(e))

    ctx = getattr(request.state, "obs_ctx", None)
    if ctx is not None:
        ctx.backend = model.backend
    logger.info("/v1/images/generations model=%s", req.model)

    client_id = client_id_from(request)
    span = current_otel_span()
    set_genai_request_attrs(
        span,
        model=req.model,
        backend=model.backend,
        operation="image_generation",
        client_id=client_id,
    )
    stash_trace_id_on_ctx(ctx, span)

    start_ns = time.monotonic_ns()
    try:
        if model.backend == "comfyui":
            out = _generate_via_comfyui(model, req.prompt, width, height, req.refine)
        else:
            # Imagen sizes its own output and ignores pixel hints, so req.size
            # is deliberately not forwarded here (docs/image-generation.md).
            out = call_gemini_image(req.prompt)
    except (GeminiCLIError, ComfyUIError) as e:
        record_genai_metrics(
            model=req.model, backend=model.backend,
            route="/v1/images/generations", client_id=client_id,
            duration_ms=(time.monotonic_ns() - start_ns) / 1e6,
            error_type=("comfyui_error" if isinstance(e, ComfyUIError)
                        else "gemini_cli_error"),
        )
        raise HTTPException(status_code=502, detail=str(e))
    except HTTPException as e:
        # _generate_via_comfyui's own 503s (remote-owned row, backend never
        # became ready) — record them before they propagate so a failed
        # on-demand load is visible in the ring rather than silently absent.
        record_genai_metrics(
            model=req.model, backend=model.backend,
            route="/v1/images/generations", client_id=client_id,
            duration_ms=(time.monotonic_ns() - start_ns) / 1e6,
            error_type=f"http_{e.status_code}",
        )
        raise

    b64 = base64.b64encode(out["image_bytes"]).decode("ascii")
    record_genai_metrics(
        model=req.model, backend=model.backend,
        route="/v1/images/generations", client_id=client_id,
        duration_ms=(time.monotonic_ns() - start_ns) / 1e6,
    )
    logger.info(
        "<- image bytes=%d media=%s backend=%s",
        len(out["image_bytes"]), out["media_type"], model.backend,
    )
    body = {
        "created": int(time.time()),
        "data": [{"b64_json": b64}],
    }
    # Non-OpenAI extension: the size actually produced. Present only when the
    # backend knows it — Imagen picks its own dimensions and does not report
    # them, and guessing would be worse than omitting the field.
    if out.get("width") and out.get("height"):
        body["size"] = f"{out['width']}x{out['height']}"
    return JSONResponse(body)


@router.post("/v1/images/edits")
async def images_edits(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(...),
    model: str = Form("gemini_image"),
    response_format: str = Form("b64_json"),
    n: int = Form(1),
) -> JSONResponse:
    """Edit an uploaded image (OpenAI ``/v1/images/edits`` shape, multipart).

    Routes to `agy`'s image path with the upload as a reference; the model
    edits it and the hub returns the result OpenAI-shape (``data[].b64_json``).
    Editing is agentic and procedural (`agy` often scripts the edit), so it is
    slower and best-effort. Same backend guard as generations.
    """
    resolved = resolve_model_or_400(model)
    if not (resolved.backend == "gemini" and resolved.image_gen):
        # Two distinct conditions, two distinct messages: a text model here is
        # a category error, whereas an image model on a non-gemini backend is
        # simply an unimplemented capability (#492 scoped edits to gemini).
        if resolved.image_gen:
            detail = (
                f"model {model!r} ({resolved.display_name}) can generate images "
                "but not edit them — editing is only implemented on the gemini "
                "backend. Use 'gemini_image' for edits."
            )
        else:
            detail = (
                f"model {model!r} ({resolved.display_name}) is not an "
                "image-generation model. Use 'gemini_image' instead."
            )
        raise HTTPException(status_code=400, detail=detail)
    if n != 1:
        raise HTTPException(status_code=400, detail="only n=1 is supported")
    if response_format != "b64_json":
        raise HTTPException(
            status_code=400,
            detail="only response_format='b64_json' is supported",
        )

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty image upload")

    ctx = getattr(request.state, "obs_ctx", None)
    if ctx is not None:
        ctx.model = model
        ctx.backend = resolved.backend
    logger.info("/v1/images/edits model=%s bytes=%d", model, len(raw))

    client_id = client_id_from(request)
    span = current_otel_span()
    set_genai_request_attrs(
        span, model=model, backend=resolved.backend,
        operation="image_edit", client_id=client_id,
    )
    stash_trace_id_on_ctx(ctx, span)

    suffix = Path(image.filename or "input.png").suffix or ".png"
    start_ns = time.monotonic_ns()
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(raw)
            tmp_path = Path(tf.name)
        try:
            out = call_gemini_image(prompt, reference_image=tmp_path)
        except GeminiCLIError as e:
            record_genai_metrics(
                model=model, backend=resolved.backend,
                route="/v1/images/edits", client_id=client_id,
                duration_ms=(time.monotonic_ns() - start_ns) / 1e6,
                error_type="gemini_cli_error",
            )
            raise HTTPException(status_code=502, detail=str(e))
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    b64 = base64.b64encode(out["image_bytes"]).decode("ascii")
    record_genai_metrics(
        model=model, backend=resolved.backend,
        route="/v1/images/edits", client_id=client_id,
        duration_ms=(time.monotonic_ns() - start_ns) / 1e6,
    )
    logger.info(
        "<- edited image bytes=%d media=%s", len(out["image_bytes"]),
        out["media_type"],
    )
    return JSONResponse({
        "created": int(time.time()),
        "data": [{"b64_json": b64}],
    })
