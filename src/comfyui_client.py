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
from pathlib import Path
from typing import Any, Dict, Optional

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


def generate_image(
    prompt: str,
    *,
    base_url: str,
    ckpt_name: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    seed: Optional[int] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Dict[str, Any]:
    """Generate one image with FLUX on a running ComfyUI.

    Returns ``{"image_bytes", "media_type", "result_text"}`` — the same shape
    ``gemini_cli.call_gemini_image`` returns, so ``server_images`` handles both
    backends' results identically. Raises :class:`ComfyUIError` on any failure.

    The caller is responsible for making sure the backend is up
    (``on_demand.ensure_ready``); this function does not spawn anything.
    """
    base = base_url.rstrip("/")
    workflow = build_flux_workflow(
        prompt, ckpt_name, width=width, height=height,
        steps=steps, guidance=guidance, seed=seed,
    )

    started = time.monotonic()
    prompt_id = _post_prompt(base, workflow)
    logger.info(
        "comfyui: queued prompt %s (%dx%d, %d steps, guidance %.1f)",
        prompt_id, width, height, steps, guidance,
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
