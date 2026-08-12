"""ComfyUI client — drives the local FLUX image backend (#492).

The hub's other image path (``gemini_cli.call_gemini_image``) shells out to the
Antigravity CLI and scrapes whatever artifact the agent leaves on disk. This one
is an ordinary HTTP client: ComfyUI runs as a normal ``models.yaml`` backend
process (``engine: comfyui-server``) and exposes a documented prompt API.

Three calls make a generation:

1. ``POST /prompt`` with a **workflow graph** in ComfyUI's "API format" —
   ``{node_id: {"class_type": ..., "inputs": {...}}}``, where an input wired to
   another node is the pair ``[node_id, output_index]``. Returns a
   ``prompt_id``.
2. ``GET /history/<prompt_id>`` — empty ``{}`` while the job is queued or
   running, then the finished record with its ``outputs`` and ``status``.
3. ``GET /view?filename=…&subfolder=…&type=…`` — the image bytes.

:func:`build_flux_workflow` writes the graph for FLUX.1 [dev]. Two details there
are load-bearing and easy to get wrong:

* **``EmptySD3LatentImage``, not ``EmptyLatentImage``** — FLUX uses a 16-channel
  latent. The SD-era node emits 4 channels and the sampler produces noise.
* **``cfg: 1.0`` plus a ``FluxGuidance`` node.** FLUX [dev] is
  guidance-*distilled*: real CFG is a no-op that merely doubles the work, and
  the guidance scale is instead injected into the conditioning. The negative
  prompt is wired only because ``KSampler`` requires the input.

Output uses ``PreviewImage`` (ComfyUI's ``temp`` type) rather than
``SaveImage``. The hub returns the bytes to the caller and has no reason to keep
a copy; ``temp`` is cleared by ComfyUI on startup, so the on-demand
load/idle-unload cycle (#422) garbage-collects generations for free instead of
growing an unbounded output directory nobody prunes.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .http_client import get_sync_client

logger = logging.getLogger(__name__)


class ComfyUIError(RuntimeError):
    """A generation could not be produced (unreachable, rejected, or failed)."""


# FLUX.1 [dev] reference settings. Steps/sampler/scheduler are the values
# ComfyUI's own FLUX dev template ships; guidance 3.5 is Black Forest Labs'
# published default for the distilled dev checkpoint.
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 20
DEFAULT_GUIDANCE = 3.5
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"

# A cold ComfyUI reads a 17 GB checkpoint off NVMe and, on a 16 GB card, streams
# part of it back and forth between host and device on every run. The first
# generation after a load is minutes; steady-state is far quicker.
DEFAULT_TIMEOUT_S = 600.0
POLL_INTERVAL_S = 1.0
# Per-HTTP-call timeouts. Distinct from the whole-generation budget above: a
# submit that hangs means ComfyUI is wedged, not that the image is slow.
SUBMIT_TIMEOUT_S = 30.0
POLL_TIMEOUT_S = 15.0
FETCH_TIMEOUT_S = 120.0

# Identifies this hub process in ComfyUI's queue/websocket bookkeeping. Purely
# informational — the hub polls /history rather than subscribing.
_CLIENT_ID = f"local-llm-hub-{uuid.uuid4().hex[:8]}"

# Node keys in the generated graph. Named rather than numbered so the wiring
# below reads as a graph instead of a puzzle.
_CKPT = "checkpoint"
_POSITIVE = "positive_prompt"
_NEGATIVE = "negative_prompt"
_GUIDANCE = "flux_guidance"
_LATENT = "empty_latent"
_SAMPLER = "sampler"
_DECODE = "vae_decode"
_OUTPUT = "output"
# Upscale / refine tail (#497), only present for above-native sizes.
_UPSCALE_MODEL = "upscale_model"
_UPSCALE = "upscale"
_RESIZE = "resize_exact"
_REFINE_ENCODE = "refine_encode"
_REFINE_SAMPLER = "refine_sampler"
_REFINE_DECODE = "refine_decode"

# ESRGAN-family 4x upscaler, provisioned into models/comfyui/upscale_models/ by
# scripts/install_comfyui.py (it is an engine asset, not a models.yaml row, so
# scripts/download_models.py does not know about it). 4x overshoots every preset
# we offer, so the ImageScale after it is always a *downsample* — the good
# direction.
DEFAULT_UPSCALE_MODEL = "4x-UltraSharp.pth"
# Second-pass denoise for the optional refine step. Low on purpose: this is
# meant to add real detail to interpolated pixels, and much above ~0.35 the
# model starts reinventing the composition it was handed.
DEFAULT_REFINE_DENOISE = 0.25
# Refine runs fewer steps than the base sample — it is finishing an image, not
# building one from noise.
DEFAULT_REFINE_STEPS = 10


@dataclass(frozen=True)
class GraphHandles:
    """Where the reusable edges live in a built workflow (#498).

    The two FLUX generations need structurally different graphs, but the
    upscale/refine tail is identical work in both. Rather than teach the tail
    about every graph, each builder reports the handful of edges it needs:
    the model, the VAE, the positive/negative conditioning, and the node keys
    of the final decode and output. ``add_upscale_tail`` then wires against
    these instead of hardcoding FLUX.1's node names.
    """

    model: List[Any]
    vae: List[Any]
    positive: List[Any]
    negative: List[Any]
    decode: str
    output: str


# FLUX.2 node keys — a distinct namespace from the FLUX.1 graph above.
_F2_UNET = "unet"
_F2_CLIP = "clip"
_F2_VAE = "vae"
_F2_POSITIVE = "positive_prompt"
_F2_NEGATIVE = "negative_prompt"
_F2_GUIDANCE = "flux_guidance"
_F2_GUIDER = "guider"
_F2_SIGMAS = "sigmas"
_F2_SAMPLER_SEL = "sampler_select"
_F2_NOISE = "noise"
_F2_LATENT = "empty_latent"
_F2_SAMPLER = "sampler"
_F2_DECODE = "vae_decode"

# FLUX.2 reference settings. Guidance 4.0 is Black Forest Labs' published
# default for the dev checkpoint; the distilled klein sibling wants fewer steps
# and lower guidance, hence the per-variant overrides in FLUX2_DEFAULTS.
DEFAULT_FLUX2_STEPS = 24
DEFAULT_FLUX2_GUIDANCE = 4.0
FLUX2_KLEIN_STEPS = 8
FLUX2_KLEIN_GUIDANCE = 3.0


def build_flux2_workflow(
    prompt: str,
    *,
    unet_name: str,
    clip_name: str,
    vae_name: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    steps: int = DEFAULT_FLUX2_STEPS,
    guidance: float = DEFAULT_FLUX2_GUIDANCE,
    seed: Optional[int] = None,
    weight_dtype: str = "fp8_e4m3fn",
) -> tuple[Dict[str, Any], GraphHandles]:
    """Build the FLUX.2 txt2img graph — a *split-loader* graph, unlike FLUX.1.

    FLUX.2 ships the transformer, its text encoder and the VAE as three
    separate files, so there is no ``CheckpointLoaderSimple`` to yield all
    three edges. Which encoder depends on the variant — Mistral-Small-24B for
    [dev], Qwen3-4B for [klein] — and they are **not** interchangeable; the
    caller supplies the right one via ``clip_name``. It also needs
    FLUX.2-specific nodes, confirmed against a live ``/object_info`` rather
    than assumed:

    * ``EmptyFlux2LatentImage`` — its own latent shape.
    * ``Flux2Scheduler`` — a **resolution-aware** sigma schedule taking width
      and height, which is why the plain ``KSampler`` path cannot be reused.
    * ``SamplerCustomAdvanced`` + ``BasicGuider`` — the guider carries a single
      conditioning, matching a guidance-distilled model, instead of KSampler's
      positive/negative pair with a CFG that would be a no-op.

    The transformer loads through the stock ``UNETLoader``. #498 also carried a
    ``UnetLoaderGGUF`` branch for the quantized 32B [dev] transformer, dropped
    with that row in #501 — see ``config/models.yaml``. Re-adding a ``.gguf``
    model means restoring both the loader branch and the ComfyUI-GGUF custom
    node, since stock ComfyUI cannot read that format at all.

    An empty negative prompt is built even though the sampling path ignores it:
    the shared refine tail runs a ``KSampler``, whose signature requires one.
    """
    if seed is None:
        seed = random.getrandbits(63)

    graph: Dict[str, Any] = {
        _F2_UNET: {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet_name, "weight_dtype": weight_dtype},
        },
        _F2_CLIP: {
            "class_type": "CLIPLoader",
            # `type` must be "flux2" — verified present in CLIPLoader's live
            # enum. The FLUX.1 value would silently mis-tokenize the prompt.
            "inputs": {"clip_name": clip_name, "type": "flux2"},
        },
        _F2_VAE: {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        _F2_POSITIVE: {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": [_F2_CLIP, 0]},
        },
        _F2_NEGATIVE: {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "", "clip": [_F2_CLIP, 0]},
        },
        _F2_GUIDANCE: {
            "class_type": "FluxGuidance",
            "inputs": {"conditioning": [_F2_POSITIVE, 0], "guidance": guidance},
        },
        _F2_GUIDER: {
            "class_type": "BasicGuider",
            "inputs": {"model": [_F2_UNET, 0], "conditioning": [_F2_GUIDANCE, 0]},
        },
        _F2_SIGMAS: {
            "class_type": "Flux2Scheduler",
            "inputs": {"steps": steps, "width": width, "height": height},
        },
        _F2_SAMPLER_SEL: {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": DEFAULT_SAMPLER},
        },
        _F2_NOISE: {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        _F2_LATENT: {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        _F2_SAMPLER: {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": [_F2_NOISE, 0],
                "guider": [_F2_GUIDER, 0],
                "sampler": [_F2_SAMPLER_SEL, 0],
                "sigmas": [_F2_SIGMAS, 0],
                "latent_image": [_F2_LATENT, 0],
            },
        },
        _F2_DECODE: {
            "class_type": "VAEDecode",
            # SamplerCustomAdvanced output 0 is the noisy latent, 1 is the
            # denoised one — decoding 0 yields a grainy image.
            "inputs": {"samples": [_F2_SAMPLER, 1], "vae": [_F2_VAE, 0]},
        },
        _OUTPUT: {
            "class_type": "PreviewImage",
            "inputs": {"images": [_F2_DECODE, 0]},
        },
    }
    handles = GraphHandles(
        model=[_F2_UNET, 0],
        vae=[_F2_VAE, 0],
        positive=[_F2_GUIDANCE, 0],
        negative=[_F2_NEGATIVE, 0],
        decode=_F2_DECODE,
        output=_OUTPUT,
    )
    return graph, handles


def flux1_handles() -> GraphHandles:
    """Handles for the FLUX.1 graph built by :func:`build_flux_workflow`."""
    return GraphHandles(
        model=[_CKPT, 0],
        vae=[_CKPT, 2],
        positive=[_GUIDANCE, 0],
        negative=[_NEGATIVE, 0],
        decode=_DECODE,
        output=_OUTPUT,
    )


def build_flux_workflow(
    prompt: str,
    ckpt_name: str,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the FLUX.1 [dev] txt2img graph in ComfyUI's API format.

    ``ckpt_name`` is the checkpoint's bare filename as ComfyUI sees it in its
    ``checkpoints`` search path (the all-in-one fp8 build bundles the
    transformer, both text encoders and the VAE, so one loader yields every
    edge this graph needs).
    """
    if seed is None:
        # ComfyUI validates seed as a 64-bit *signed* int; 63 bits keeps it
        # positive and in range. A fresh seed per call means repeated identical
        # prompts don't return a cached-looking identical image.
        seed = random.getrandbits(63)
    return {
        _CKPT: {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt_name},
        },
        _POSITIVE: {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": [_CKPT, 1]},
        },
        _NEGATIVE: {
            # Required by KSampler's signature; FLUX [dev] ignores it (see the
            # module docstring on guidance distillation).
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "", "clip": [_CKPT, 1]},
        },
        _GUIDANCE: {
            "class_type": "FluxGuidance",
            "inputs": {"conditioning": [_POSITIVE, 0], "guidance": guidance},
        },
        _LATENT: {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        _SAMPLER: {
            "class_type": "KSampler",
            "inputs": {
                "model": [_CKPT, 0],
                "positive": [_GUIDANCE, 0],
                "negative": [_NEGATIVE, 0],
                "latent_image": [_LATENT, 0],
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": DEFAULT_SAMPLER,
                "scheduler": DEFAULT_SCHEDULER,
                "denoise": 1.0,
            },
        },
        _DECODE: {
            "class_type": "VAEDecode",
            "inputs": {"samples": [_SAMPLER, 0], "vae": [_CKPT, 2]},
        },
        _OUTPUT: {
            "class_type": "PreviewImage",
            "inputs": {"images": [_DECODE, 0]},
        },
    }


def add_upscale_tail(
    workflow: Dict[str, Any],
    target_width: int,
    target_height: int,
    *,
    handles: Optional[GraphHandles] = None,
    refine: bool = False,
    refine_denoise: float = DEFAULT_REFINE_DENOISE,
    refine_steps: int = DEFAULT_REFINE_STEPS,
    seed: Optional[int] = None,
    upscale_model: str = DEFAULT_UPSCALE_MODEL,
) -> Dict[str, Any]:
    """Extend a base txt2img graph to reach an above-native target size (#497).

    Mutates and returns ``workflow``. The base graph must already sample at a
    native-safe size (see ``image_sizes.native_source_size``); this bolts on:

    ``upscale model -> 4x upscale -> scale to exact target [-> refine pass]``

    then re-points the output node at the end of that chain. The ``ImageScale``
    after the 4x pass is always a downsample, because 4x overshoots every size
    we offer — going down from a larger image is what keeps it sharp.

    ``refine`` adds a second low-denoise sampling pass over the upscaled image.
    Without it an upscale is interpolation: more pixels, no more detail. With
    it the model actually redraws at the higher resolution — roughly doubling
    generation time, which is why the caller opts in.
    """
    # Defaults to the FLUX.1 graph's edges so pre-#498 callers are unchanged.
    h = handles or flux1_handles()

    workflow[_UPSCALE_MODEL] = {
        "class_type": "UpscaleModelLoader",
        "inputs": {"model_name": upscale_model},
    }
    workflow[_UPSCALE] = {
        "class_type": "ImageUpscaleWithModel",
        "inputs": {"upscale_model": [_UPSCALE_MODEL, 0], "image": [h.decode, 0]},
    }
    workflow[_RESIZE] = {
        "class_type": "ImageScale",
        "inputs": {
            "image": [_UPSCALE, 0],
            "width": target_width,
            "height": target_height,
            "upscale_method": "lanczos",
            "crop": "disabled",
        },
    }
    tail = _RESIZE

    if refine:
        workflow[_REFINE_ENCODE] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": [_RESIZE, 0], "vae": list(h.vae)},
        }
        workflow[_REFINE_SAMPLER] = {
            "class_type": "KSampler",
            "inputs": {
                "model": list(h.model),
                "positive": list(h.positive),
                "negative": list(h.negative),
                "latent_image": [_REFINE_ENCODE, 0],
                # A *different* seed from the base pass on purpose: reusing it
                # would re-apply the same noise pattern the image already
                # carries and bias the refinement toward the original grain.
                "seed": (seed + 1) if seed is not None else random.getrandbits(63),
                "steps": refine_steps,
                "cfg": 1.0,
                "sampler_name": DEFAULT_SAMPLER,
                "scheduler": DEFAULT_SCHEDULER,
                "denoise": refine_denoise,
            },
        }
        workflow[_REFINE_DECODE] = {
            "class_type": "VAEDecode",
            "inputs": {"samples": [_REFINE_SAMPLER, 0], "vae": list(h.vae)},
        }
        tail = _REFINE_DECODE

    workflow[h.output]["inputs"]["images"] = [tail, 0]
    return workflow


def checkpoint_name_for(model_path: Optional[str]) -> str:
    """ComfyUI's name for a registry ``model_path`` — its bare filename.

    ``config/models.yaml`` stores a repo-relative path (so
    ``scripts/download_models.py`` knows where to put the weights); ComfyUI
    resolves checkpoints by filename within its own search path. One source of
    truth, two spellings.
    """
    if not model_path:
        raise ComfyUIError("model row has no model_path — cannot name a checkpoint")
    return Path(model_path).name


def _post_prompt(base_url: str, workflow: Dict[str, Any]) -> str:
    payload = {"prompt": workflow, "client_id": _CLIENT_ID}
    try:
        r = get_sync_client().post(
            f"{base_url}/prompt", json=payload, timeout=SUBMIT_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 — surfaced as a backend error
        raise ComfyUIError(f"ComfyUI unreachable at {base_url}: {exc}") from exc

    if r.status_code != 200:
        # A rejected graph comes back as 400 with per-node validation detail —
        # far more useful than the status line alone.
        detail = r.text[:800]
        try:
            body = r.json()
            node_errors = body.get("node_errors") or {}
            if node_errors:
                detail = f"{body.get('error')} node_errors={node_errors}"
        except Exception:  # noqa: BLE001 — non-JSON body, keep the raw text
            pass
        raise ComfyUIError(f"ComfyUI rejected the workflow ({r.status_code}): {detail}")

    prompt_id = (r.json() or {}).get("prompt_id")
    if not prompt_id:
        raise ComfyUIError(f"ComfyUI returned no prompt_id: {r.text[:300]}")
    return str(prompt_id)


def _await_history(base_url: str, prompt_id: str, timeout_s: float) -> Dict[str, Any]:
    """Poll ``/history/<id>`` until the job lands, then return its record.

    ComfyUI answers 200 with ``{}`` for a job that is still queued or running,
    so "empty" means *pending* rather than *missing* — the deadline is what
    distinguishes a slow generation from a wedged one.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = get_sync_client().get(
                f"{base_url}/history/{prompt_id}", timeout=POLL_TIMEOUT_S,
            )
            if r.status_code == 200:
                history = r.json() or {}
                record = history.get(prompt_id)
                if record:
                    return record
        except Exception as exc:  # noqa: BLE001 — transient; keep polling
            logger.debug("comfyui history poll failed (retrying): %s", exc)
        time.sleep(POLL_INTERVAL_S)

    raise ComfyUIError(
        f"ComfyUI did not finish prompt {prompt_id} within {timeout_s:.0f}s — "
        "check data/logs/backend-<model-id>.log"
    )


def _execution_error(record: Dict[str, Any]) -> Optional[str]:
    """A human-readable failure from a finished history record, else ``None``.

    Distinguishes "the graph ran and a node raised" (OOM, missing checkpoint)
    from "the graph ran and produced nothing" — different conditions deserve
    different messages.
    """
    status = record.get("status") or {}
    if status.get("status_str") != "error" and status.get("completed") is not False:
        return None
    for entry in status.get("messages") or []:
        # messages are ``[event_name, payload]`` pairs.
        if isinstance(entry, (list, tuple)) and len(entry) == 2 and entry[0] == "execution_error":
            payload = entry[1] or {}
            return (
                f"{payload.get('node_type', '?')} raised "
                f"{payload.get('exception_type', '?')}: "
                f"{payload.get('exception_message', '?')}"
            )
    return status.get("status_str") or "execution failed"


def _first_image_ref(record: Dict[str, Any]) -> Dict[str, Any]:
    for node_output in (record.get("outputs") or {}).values():
        for image in node_output.get("images") or []:
            if image.get("filename"):
                return image
    raise ComfyUIError("ComfyUI finished but produced no image output")


def _fetch_image(base_url: str, ref: Dict[str, Any]) -> tuple[bytes, str]:
    params = {
        "filename": ref["filename"],
        "subfolder": ref.get("subfolder", ""),
        "type": ref.get("type", "temp"),
    }
    try:
        r = get_sync_client().get(
            f"{base_url}/view", params=params, timeout=FETCH_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001
        raise ComfyUIError(f"could not fetch the generated image: {exc}") from exc
    if r.status_code != 200 or not r.content:
        raise ComfyUIError(
            f"could not fetch {ref['filename']} from ComfyUI ({r.status_code})"
        )

    # Unlike the `agy` path — which sniffs magic bytes because the agent
    # mislabels its own files — ComfyUI serves the artifact it wrote, so its
    # declared Content-Type is trustworthy. Fall back to the extension.
    content_type = (r.headers.get("content-type") or "").split(";")[0].strip()
    if content_type.startswith("image/"):
        media_type = content_type
    else:
        suffix = Path(ref["filename"]).suffix.lower()
        media_type = {".png": "image/png", ".jpg": "image/jpeg",
                      ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(suffix, "image/png")
    return r.content, media_type


@dataclass(frozen=True)
class ModelSpec:
    """Everything the client needs to build a graph for one registry row (#498).

    Built by ``server_images`` from a ``models.yaml`` row, so the client stays
    ignorant of the registry. ``workflow`` selects the graph; the file fields
    are ComfyUI's bare filenames (it resolves weights by name within its own
    search path, not by our repo-relative paths).
    """

    workflow: str = "flux1"
    ckpt_name: Optional[str] = None          # flux1: the all-in-one checkpoint
    unet_name: Optional[str] = None          # flux2: the transformer
    # flux2: the text encoder — Mistral-Small for dev, Qwen3-4B for klein.
    # Not interchangeable; see the note in config/models.yaml.
    clip_name: Optional[str] = None
    vae_name: Optional[str] = None           # flux2
    steps: Optional[int] = None
    guidance: Optional[float] = None


def _build_graph(
    spec: ModelSpec, prompt: str, width: int, height: int, seed: int,
) -> tuple[Dict[str, Any], GraphHandles]:
    if spec.workflow == "flux2":
        if not (spec.unet_name and spec.clip_name and spec.vae_name):
            raise ComfyUIError(
                "flux2 workflow needs unet/text_encoder/vae weights — check the "
                "row's model_path and extra_weights in config/models.yaml"
            )
        return build_flux2_workflow(
            prompt,
            unet_name=spec.unet_name, clip_name=spec.clip_name,
            vae_name=spec.vae_name,
            width=width, height=height,
            steps=spec.steps or DEFAULT_FLUX2_STEPS,
            guidance=spec.guidance if spec.guidance is not None else DEFAULT_FLUX2_GUIDANCE,
            seed=seed,
        )
    if not spec.ckpt_name:
        raise ComfyUIError("flux1 workflow needs a checkpoint — check model_path")
    graph = build_flux_workflow(
        prompt, spec.ckpt_name, width=width, height=height,
        steps=spec.steps or DEFAULT_STEPS,
        guidance=spec.guidance if spec.guidance is not None else DEFAULT_GUIDANCE,
        seed=seed,
    )
    return graph, flux1_handles()


def generate_image(
    prompt: str,
    *,
    base_url: str,
    spec: Optional[ModelSpec] = None,
    ckpt_name: Optional[str] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    seed: Optional[int] = None,
    refine: bool = False,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate one image with FLUX on a running ComfyUI.

    ``width``/``height`` are the *requested* output size. Anything above the
    native sampling ceiling (``image_sizes.NATIVE_MAX_PIXELS``) is sampled at
    the largest native-safe size of the same aspect ratio and then upscaled to
    the exact target — the caller asks for 4K and gets 4K back, with the
    two-stage work staying inside this function rather than leaking a two-call
    protocol into the API. ``refine`` adds a second low-denoise pass over the
    upscaled image; it is ignored for native-size requests, which have nothing
    to refine.

    Returns ``{"image_bytes", "media_type", "result_text", "width", "height"}``
    — a superset of what ``gemini_cli.call_gemini_image`` returns, so
    ``server_images`` still handles both backends' results identically.
    Raises :class:`ComfyUIError` on any failure.

    The caller is responsible for making sure the backend is up
    (``on_demand.ensure_ready``); this function does not spawn anything.
    """
    from .image_sizes import native_source_size, needs_upscale

    if spec is None:
        # Back-compat for the single-model call shape (#492/#497).
        spec = ModelSpec(workflow="flux1", ckpt_name=ckpt_name)
    if timeout_s is None:
        timeout_s = DEFAULT_TIMEOUT_S

    base = base_url.rstrip("/")
    upscaling = needs_upscale(width, height)
    src_w, src_h = native_source_size(width, height)
    if seed is None:
        seed = random.getrandbits(63)

    workflow, handles = _build_graph(spec, prompt, src_w, src_h, seed)
    if upscaling:
        add_upscale_tail(
            workflow, width, height, handles=handles, refine=refine, seed=seed,
        )

    started = time.monotonic()
    prompt_id = _post_prompt(base, workflow)
    if upscaling:
        logger.info(
            "comfyui: queued prompt %s [%s] (%dx%d sampled -> %dx%d upscaled%s)",
            prompt_id, spec.workflow, src_w, src_h, width, height,
            " + refine" if refine else "",
        )
    else:
        logger.info(
            "comfyui: queued prompt %s [%s] (%dx%d)",
            prompt_id, spec.workflow, width, height,
        )

    record = _await_history(base, prompt_id, timeout_s)
    error = _execution_error(record)
    if error:
        raise ComfyUIError(f"ComfyUI workflow failed: {error}")

    image_bytes, media_type = _fetch_image(base, _first_image_ref(record))
    elapsed = time.monotonic() - started
    logger.info(
        "comfyui: prompt %s produced %d bytes (%s) in %.1fs",
        prompt_id, len(image_bytes), media_type, elapsed,
    )
    return {
        "image_bytes": image_bytes,
        "media_type": media_type,
        "width": width,
        "height": height,
        "result_text": f"comfyui prompt {prompt_id} in {elapsed:.1f}s",
    }


def is_reachable(base_url: str, timeout: float = 1.5) -> bool:
    """Liveness probe for a ComfyUI backend.

    ComfyUI has neither ``/health`` nor ``/v1/models`` (the two endpoints
    ``backend_process.is_reachable`` tries for every other engine), but
    ``/system_stats`` answers 200 with a JSON device summary once the server is
    accepting work.
    """
    try:
        r = get_sync_client().get(
            f"{base_url.rstrip('/')}/system_stats", timeout=timeout,
        )
        return r.status_code == 200
    except Exception:  # noqa: BLE001 — any failure is "not up"
        return False
