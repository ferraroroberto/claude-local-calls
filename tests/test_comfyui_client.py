"""Unit tests for src/comfyui_client.py — ComfyUI's HTTP API is mocked (#492).

No ComfyUI process is started and no weights are read. These cover the two
things that are genuinely easy to get wrong and expensive to debug live: the
shape of the generated FLUX workflow graph, and the submit → poll → fetch
sequence's error handling.
"""

from __future__ import annotations

import pytest

from src import comfyui_client as cc
from src.comfyui_client import ComfyUIError

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
_CKPT = "flux1-dev-fp8.safetensors"
_BASE = "http://127.0.0.1:8188"


class _FakeResponse:
    def __init__(self, status_code=200, json_body=None, content=b"", headers=None):
        self.status_code = status_code
        self._json = json_body
        self.content = content
        self.headers = headers or {}
        self.text = str(json_body) if json_body is not None else ""

    def json(self):
        if self._json is None:
            raise ValueError("no json body")
        return self._json


class _FakeClient:
    """Minimal stand-in for the shared httpx client.

    ``history_sequence`` is consumed one entry per ``/history`` poll, so a test
    can model "still running, still running, done".
    """

    def __init__(self, *, prompt_response=None, history_sequence=None,
                 view_response=None, raise_on_post=None):
        self.prompt_response = prompt_response or _FakeResponse(
            json_body={"prompt_id": "pid-1"})
        self.history_sequence = list(history_sequence or [])
        self.view_response = view_response or _FakeResponse(
            content=_PNG, headers={"content-type": "image/png"})
        self.raise_on_post = raise_on_post
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        if self.raise_on_post:
            raise self.raise_on_post
        return self.prompt_response

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if "/history/" in url:
            if self.history_sequence:
                return _FakeResponse(json_body=self.history_sequence.pop(0))
            return _FakeResponse(json_body={})
        if "/view" in url:
            return self.view_response
        if "/system_stats" in url:
            return _FakeResponse(json_body={"devices": []})
        raise AssertionError(f"unexpected GET {url}")


def _install(monkeypatch, client):
    monkeypatch.setattr(cc, "get_sync_client", lambda: client)
    # Keep the poll loop from sleeping a real second between iterations.
    monkeypatch.setattr(cc, "POLL_INTERVAL_S", 0.001)
    return client


def _done_record(images=None):
    return {
        "pid-1": {
            "status": {"status_str": "success", "completed": True, "messages": []},
            "outputs": {"output": {"images": images if images is not None else [
                {"filename": "ComfyUI_00001_.png", "subfolder": "", "type": "temp"}
            ]}},
        }
    }


# --- workflow graph -------------------------------------------------------

def test_workflow_uses_sd3_latent_not_sd_latent():
    """FLUX needs a 16-channel latent. EmptyLatentImage emits 4 and the
    sampler produces noise — a silent, plausible-looking failure."""
    wf = cc.build_flux_workflow("a cat", _CKPT)
    class_types = {node["class_type"] for node in wf.values()}
    assert "EmptySD3LatentImage" in class_types
    assert "EmptyLatentImage" not in class_types


def test_workflow_uses_flux_guidance_and_cfg_one():
    """FLUX [dev] is guidance-distilled: real CFG is a no-op that doubles the
    work, and the guidance scale rides on the conditioning instead."""
    wf = cc.build_flux_workflow("a cat", _CKPT, guidance=3.5)
    sampler = next(n for n in wf.values() if n["class_type"] == "KSampler")
    assert sampler["inputs"]["cfg"] == 1.0
    guidance = next(n for n in wf.values() if n["class_type"] == "FluxGuidance")
    assert guidance["inputs"]["guidance"] == 3.5
    # positive prompt -> FluxGuidance -> sampler, not prompt -> sampler.
    assert sampler["inputs"]["positive"][0] == cc._GUIDANCE
    assert guidance["inputs"]["conditioning"][0] == cc._POSITIVE


def test_workflow_wires_checkpoint_outputs():
    """The all-in-one fp8 checkpoint yields MODEL/CLIP/VAE from one loader —
    outputs 0/1/2 respectively."""
    wf = cc.build_flux_workflow("a cat", _CKPT)
    assert wf[cc._CKPT]["inputs"]["ckpt_name"] == _CKPT
    assert wf[cc._SAMPLER]["inputs"]["model"] == [cc._CKPT, 0]
    assert wf[cc._POSITIVE]["inputs"]["clip"] == [cc._CKPT, 1]
    assert wf[cc._DECODE]["inputs"]["vae"] == [cc._CKPT, 2]


def test_workflow_honours_prompt_size_steps_and_seed():
    wf = cc.build_flux_workflow(
        "a duck", _CKPT, width=768, height=512, steps=8, seed=42)
    assert wf[cc._POSITIVE]["inputs"]["text"] == "a duck"
    assert wf[cc._LATENT]["inputs"]["width"] == 768
    assert wf[cc._LATENT]["inputs"]["height"] == 512
    assert wf[cc._SAMPLER]["inputs"]["steps"] == 8
    assert wf[cc._SAMPLER]["inputs"]["seed"] == 42


def test_workflow_seed_is_random_and_in_signed_64bit_range():
    """ComfyUI validates seed as a signed 64-bit int; a negative or oversized
    value is rejected at submit time."""
    seeds = {cc.build_flux_workflow("x", _CKPT)[cc._SAMPLER]["inputs"]["seed"]
             for _ in range(20)}
    assert len(seeds) > 1, "seed should vary between calls"
    assert all(0 <= s < 2**63 for s in seeds)


# --- checkpoint naming ----------------------------------------------------

def test_checkpoint_name_is_bare_filename():
    assert cc.checkpoint_name_for(
        "models/comfyui/checkpoints/flux1-dev-fp8.safetensors") == _CKPT


def test_checkpoint_name_requires_a_path():
    with pytest.raises(ComfyUIError, match="model_path"):
        cc.checkpoint_name_for(None)


# --- generate_image happy path -------------------------------------------

def test_generate_image_returns_bytes_and_media_type(monkeypatch):
    _install(monkeypatch, _FakeClient(history_sequence=[_done_record()]))
    out = cc.generate_image("a red apple", base_url=_BASE, ckpt_name=_CKPT)
    assert out["image_bytes"] == _PNG
    assert out["media_type"] == "image/png"
    assert "pid-1" in out["result_text"]


def test_generate_image_polls_until_history_populates(monkeypatch):
    """An empty {} from /history means *pending*, not missing — the client has
    to keep polling rather than treat it as a failure."""
    client = _install(monkeypatch, _FakeClient(
        history_sequence=[{}, {}, _done_record()]))
    out = cc.generate_image("x", base_url=_BASE, ckpt_name=_CKPT, timeout_s=5)
    assert out["image_bytes"] == _PNG
    history_polls = [c for c in client.calls if "/history/" in c[1]]
    assert len(history_polls) == 3


def test_generate_image_sends_workflow_to_prompt_endpoint(monkeypatch):
    client = _install(monkeypatch, _FakeClient(history_sequence=[_done_record()]))
    cc.generate_image("a red apple", base_url=_BASE, ckpt_name=_CKPT)
    method, url, kwargs = client.calls[0]
    assert method == "POST" and url.endswith("/prompt")
    body = kwargs["json"]
    assert body["prompt"][cc._POSITIVE]["inputs"]["text"] == "a red apple"
    assert body["client_id"]


def test_generate_image_falls_back_to_extension_for_media_type(monkeypatch):
    """ComfyUI's declared Content-Type is trusted when it's an image type;
    otherwise the filename decides."""
    _install(monkeypatch, _FakeClient(
        history_sequence=[_done_record([
            {"filename": "out.jpeg", "subfolder": "", "type": "temp"}])],
        view_response=_FakeResponse(
            content=_PNG, headers={"content-type": "application/octet-stream"}),
    ))
    out = cc.generate_image("x", base_url=_BASE, ckpt_name=_CKPT)
    assert out["media_type"] == "image/jpeg"


# --- failure modes --------------------------------------------------------

def test_unreachable_backend_raises(monkeypatch):
    _install(monkeypatch, _FakeClient(raise_on_post=OSError("connection refused")))
    with pytest.raises(ComfyUIError, match="unreachable"):
        cc.generate_image("x", base_url=_BASE, ckpt_name=_CKPT)


def test_rejected_workflow_surfaces_node_errors(monkeypatch):
    _install(monkeypatch, _FakeClient(prompt_response=_FakeResponse(
        status_code=400,
        json_body={"error": "invalid prompt",
                   "node_errors": {"checkpoint": "ckpt_name not in list"}},
    )))
    with pytest.raises(ComfyUIError, match="node_errors"):
        cc.generate_image("x", base_url=_BASE, ckpt_name="missing.safetensors")


def test_execution_error_is_reported_with_node_detail(monkeypatch):
    """An OOM or missing-checkpoint failure lands in the history record's
    status messages, not as an HTTP error — surface the node that raised."""
    _install(monkeypatch, _FakeClient(history_sequence=[{
        "pid-1": {
            "status": {"status_str": "error", "completed": False, "messages": [
                ["execution_error", {
                    "node_type": "KSampler",
                    "exception_type": "torch.cuda.OutOfMemoryError",
                    "exception_message": "CUDA out of memory",
                }],
            ]},
            "outputs": {},
        }
    }]))
    with pytest.raises(ComfyUIError, match="KSampler.*CUDA out of memory"):
        cc.generate_image("x", base_url=_BASE, ckpt_name=_CKPT)


def test_finished_with_no_image_raises(monkeypatch):
    _install(monkeypatch, _FakeClient(history_sequence=[_done_record(images=[])]))
    with pytest.raises(ComfyUIError, match="no image output"):
        cc.generate_image("x", base_url=_BASE, ckpt_name=_CKPT)


def test_timeout_names_the_deadline(monkeypatch):
    """A never-completing job must fail with a distinct, actionable message —
    not the same error as an unreachable backend."""
    _install(monkeypatch, _FakeClient(history_sequence=[]))
    with pytest.raises(ComfyUIError, match="did not finish"):
        cc.generate_image("x", base_url=_BASE, ckpt_name=_CKPT, timeout_s=0.05)


def test_view_failure_raises(monkeypatch):
    _install(monkeypatch, _FakeClient(
        history_sequence=[_done_record()],
        view_response=_FakeResponse(status_code=404, content=b""),
    ))
    with pytest.raises(ComfyUIError, match="could not fetch"):
        cc.generate_image("x", base_url=_BASE, ckpt_name=_CKPT)


# --- upscale + refine tail (#497) ----------------------------------------

def _classes(wf):
    return {n["class_type"] for n in wf.values()}


def test_upscaler_filename_matches_what_the_installer_provisions():
    """The client names the upscaler; the installer downloads it. When those
    two drifted, the 4K path failed on a fresh machine with a bare
    'value not in list' node error while working fine where the file had been
    placed by hand."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import install_comfyui  # type: ignore

    assert cc.DEFAULT_UPSCALE_MODEL == install_comfyui.UPSCALE_MODEL_FILE


def test_upscale_tail_adds_the_chain_and_repoints_output():
    wf = cc.build_flux_workflow("x", _CKPT, width=1920, height=1088)
    cc.add_upscale_tail(wf, 3840, 2160)
    assert {"UpscaleModelLoader", "ImageUpscaleWithModel", "ImageScale"} <= _classes(wf)
    # The output node must show the END of the chain, not the raw decode.
    assert wf[cc._OUTPUT]["inputs"]["images"] == [cc._RESIZE, 0]
    assert wf[cc._UPSCALE]["inputs"]["image"] == [cc._DECODE, 0]


def test_upscale_tail_scales_to_the_exact_requested_size():
    wf = cc.build_flux_workflow("x", _CKPT, width=1920, height=1088)
    cc.add_upscale_tail(wf, 3840, 2160)
    resize = wf[cc._RESIZE]["inputs"]
    assert (resize["width"], resize["height"]) == (3840, 2160)


def test_refine_appends_a_second_pass_and_repoints_output():
    wf = cc.build_flux_workflow("x", _CKPT, width=1920, height=1088)
    cc.add_upscale_tail(wf, 3840, 2160, refine=True)
    assert {"VAEEncode"} <= _classes(wf)
    # Refine encodes the RESIZED image, not the original latent.
    assert wf[cc._REFINE_ENCODE]["inputs"]["pixels"] == [cc._RESIZE, 0]
    assert wf[cc._OUTPUT]["inputs"]["images"] == [cc._REFINE_DECODE, 0]


def test_refine_uses_low_denoise():
    """A high denoise would let the second pass reinvent the composition it
    was handed rather than add detail to it."""
    wf = cc.build_flux_workflow("x", _CKPT, width=1920, height=1088)
    cc.add_upscale_tail(wf, 3840, 2160, refine=True)
    assert 0 < wf[cc._REFINE_SAMPLER]["inputs"]["denoise"] <= 0.35


def test_refine_seed_differs_from_the_base_pass():
    """Reusing the base seed re-applies the noise pattern the image already
    carries, biasing the refinement toward the original grain."""
    wf = cc.build_flux_workflow("x", _CKPT, width=1920, height=1088, seed=42)
    cc.add_upscale_tail(wf, 3840, 2160, refine=True, seed=42)
    assert wf[cc._REFINE_SAMPLER]["inputs"]["seed"] != 42


def test_no_refine_leaves_no_second_sampler():
    wf = cc.build_flux_workflow("x", _CKPT, width=1920, height=1088)
    cc.add_upscale_tail(wf, 3840, 2160, refine=False)
    assert cc._REFINE_SAMPLER not in wf
    assert cc._REFINE_ENCODE not in wf


def test_generate_samples_natively_then_upscales_for_4k(monkeypatch):
    """The whole point of the two-stage path: 4K must NOT be sampled natively.
    Sampling at 8.3 MP is a quality cliff, not just a slow one."""
    client = _install(monkeypatch, _FakeClient(history_sequence=[_done_record()]))
    out = cc.generate_image("x", base_url=_BASE, ckpt_name=_CKPT,
                            width=3840, height=2160)
    posted = client.calls[0][2]["json"]["prompt"]
    latent = posted[cc._LATENT]["inputs"]
    assert latent["width"] * latent["height"] <= 2_100_000
    assert (latent["width"], latent["height"]) != (3840, 2160)
    assert posted[cc._RESIZE]["inputs"]["width"] == 3840
    # The caller asked for 4K and is told it got 4K.
    assert (out["width"], out["height"]) == (3840, 2160)


def test_generate_keeps_native_sizes_single_stage(monkeypatch):
    client = _install(monkeypatch, _FakeClient(history_sequence=[_done_record()]))
    out = cc.generate_image("x", base_url=_BASE, ckpt_name=_CKPT,
                            width=1024, height=1024)
    posted = client.calls[0][2]["json"]["prompt"]
    assert cc._UPSCALE not in posted
    assert posted[cc._LATENT]["inputs"]["width"] == 1024
    assert (out["width"], out["height"]) == (1024, 1024)


def test_refine_is_ignored_for_a_native_size(monkeypatch):
    """Nothing to refine without an upscale — the flag must not silently add
    a second sampling pass and double the cost of an ordinary request."""
    client = _install(monkeypatch, _FakeClient(history_sequence=[_done_record()]))
    cc.generate_image("x", base_url=_BASE, ckpt_name=_CKPT,
                      width=1024, height=1024, refine=True)
    posted = client.calls[0][2]["json"]["prompt"]
    assert cc._REFINE_SAMPLER not in posted


# --- FLUX.2 split-loader graph (#498) ------------------------------------

_F2 = dict(unet_name="flux2-dev-Q4_K_M.gguf",
           clip_name="mistral_3_small_flux2_fp4_mixed.safetensors",
           vae_name="flux2-vae.safetensors")


def test_flux2_uses_its_own_latent_and_scheduler():
    """FLUX.2 is not FLUX.1 with different weights: it needs its own latent
    node and a *resolution-aware* scheduler. Reusing the FLUX.1 nodes is the
    kind of mistake that produces noise rather than an error."""
    wf, _ = cc.build_flux2_workflow("x", gguf=True, **_F2)
    classes = _classes(wf)
    assert "EmptyFlux2LatentImage" in classes
    assert "Flux2Scheduler" in classes
    assert "EmptySD3LatentImage" not in classes
    assert "KSampler" not in classes  # sampling goes through SamplerCustomAdvanced


def test_flux2_scheduler_gets_the_sampled_resolution():
    """Flux2Scheduler takes width/height — that is *why* it cannot be swapped
    for a plain scheduler."""
    wf, _ = cc.build_flux2_workflow("x", width=1216, height=832, steps=24, **_F2, gguf=True)
    sig = wf[cc._F2_SIGMAS]["inputs"]
    assert (sig["width"], sig["height"], sig["steps"]) == (1216, 832, 24)


def test_flux2_clip_loader_uses_the_flux2_type():
    """The FLUX.1 value would silently mis-tokenize the prompt rather than
    fail, so this is asserted explicitly."""
    wf, _ = cc.build_flux2_workflow("x", **_F2, gguf=True)
    assert wf[cc._F2_CLIP]["inputs"]["type"] == "flux2"


def test_flux2_gguf_flag_picks_the_loader():
    """Stock ComfyUI cannot load a .gguf transformer at all; the quantized dev
    model needs the custom node's loader."""
    gguf_wf, _ = cc.build_flux2_workflow("x", **_F2, gguf=True)
    assert gguf_wf[cc._F2_UNET]["class_type"] == "UnetLoaderGGUF"

    plain = dict(_F2, unet_name="flux-2-klein-4b-fp8.safetensors")
    safet_wf, _ = cc.build_flux2_workflow("x", **plain, gguf=False)
    assert safet_wf[cc._F2_UNET]["class_type"] == "UNETLoader"
    assert safet_wf[cc._F2_UNET]["inputs"]["weight_dtype"] == "fp8_e4m3fn"


def test_flux2_wires_three_separate_loaders():
    """Unlike FLUX.1's all-in-one checkpoint, MODEL/CLIP/VAE come from three
    different files."""
    wf, h = cc.build_flux2_workflow("x", **_F2, gguf=True)
    assert wf[cc._F2_POSITIVE]["inputs"]["clip"] == [cc._F2_CLIP, 0]
    assert wf[cc._F2_DECODE]["inputs"]["vae"] == [cc._F2_VAE, 0]
    assert h.model == [cc._F2_UNET, 0]
    assert h.vae == [cc._F2_VAE, 0]


def test_flux2_guider_carries_the_guidance_conditioning():
    wf, _ = cc.build_flux2_workflow("x", **_F2, gguf=True, guidance=4.0)
    assert wf[cc._F2_GUIDANCE]["inputs"]["guidance"] == 4.0
    assert wf[cc._F2_GUIDER]["inputs"]["conditioning"] == [cc._F2_GUIDANCE, 0]
    assert wf[cc._F2_GUIDER]["inputs"]["model"] == [cc._F2_UNET, 0]


def test_flux2_upscale_tail_wires_against_flux2_nodes():
    """The shared tail must follow the graph's own handles — hardcoding
    FLUX.1's checkpoint node would reference a node that does not exist here."""
    wf, h = cc.build_flux2_workflow("x", **_F2, gguf=True, width=1920, height=1088)
    cc.add_upscale_tail(wf, 3840, 2160, handles=h, refine=True, seed=7)
    assert wf[cc._UPSCALE]["inputs"]["image"] == [cc._F2_DECODE, 0]
    assert wf[cc._REFINE_ENCODE]["inputs"]["vae"] == [cc._F2_VAE, 0]
    assert wf[cc._REFINE_SAMPLER]["inputs"]["model"] == [cc._F2_UNET, 0]
    assert wf[cc._OUTPUT]["inputs"]["images"] == [cc._REFINE_DECODE, 0]
    assert cc._CKPT not in wf  # no FLUX.1 leakage


def test_flux1_graph_is_unchanged_by_the_refactor():
    """Regression guard for #492/#497: the split-loader work must not alter
    the all-in-one path."""
    wf = cc.build_flux_workflow("x", _CKPT, seed=1)
    assert wf[cc._CKPT]["class_type"] == "CheckpointLoaderSimple"
    assert wf[cc._LATENT]["class_type"] == "EmptySD3LatentImage"
    assert wf[cc._SAMPLER]["class_type"] == "KSampler"
    h = cc.flux1_handles()
    assert (h.model, h.vae, h.decode, h.output) == ([cc._CKPT, 0], [cc._CKPT, 2],
                                                    cc._DECODE, cc._OUTPUT)


def test_generate_routes_by_spec_workflow(monkeypatch):
    client = _install(monkeypatch, _FakeClient(history_sequence=[_done_record()]))
    cc.generate_image("x", base_url=_BASE,
                      spec=cc.ModelSpec(workflow="flux2", gguf=True, **_F2))
    posted = client.calls[0][2]["json"]["prompt"]
    assert posted[cc._F2_CLIP]["inputs"]["type"] == "flux2"


def test_generate_without_spec_still_uses_flux1(monkeypatch):
    """Back-compat for the pre-#498 call shape."""
    client = _install(monkeypatch, _FakeClient(history_sequence=[_done_record()]))
    cc.generate_image("x", base_url=_BASE, ckpt_name=_CKPT)
    posted = client.calls[0][2]["json"]["prompt"]
    assert posted[cc._CKPT]["inputs"]["ckpt_name"] == _CKPT


def test_flux2_gets_a_bigger_default_timeout_than_flux1(monkeypatch):
    """Regression: a 1024x1024 FLUX.2 [dev] image measured 600.5 s — half a
    second past FLUX.1's 600 s budget. Sharing one timeout means the very first
    real request fails."""
    assert cc.FLUX2_TIMEOUT_S > cc.DEFAULT_TIMEOUT_S
    seen = {}

    def fake_await(base, prompt_id, timeout_s):
        seen["timeout"] = timeout_s
        return _done_record()["pid-1"]

    _install(monkeypatch, _FakeClient(history_sequence=[_done_record()]))
    monkeypatch.setattr(cc, "_await_history", fake_await)

    cc.generate_image("x", base_url=_BASE,
                      spec=cc.ModelSpec(workflow="flux2", gguf=True, **_F2))
    assert seen["timeout"] == cc.FLUX2_TIMEOUT_S

    cc.generate_image("x", base_url=_BASE, ckpt_name=_CKPT)
    assert seen["timeout"] == cc.DEFAULT_TIMEOUT_S


def test_flux2_spec_missing_companion_weights_is_a_clear_error():
    with pytest.raises(ComfyUIError, match="extra_weights"):
        cc._build_graph(cc.ModelSpec(workflow="flux2", unet_name="a.gguf"),
                        "x", 1024, 1024, 1)


# --- liveness -------------------------------------------------------------

def test_is_reachable_uses_system_stats(monkeypatch):
    client = _install(monkeypatch, _FakeClient())
    assert cc.is_reachable(_BASE) is True
    assert any("/system_stats" in c[1] for c in client.calls)


def test_is_reachable_false_when_down(monkeypatch):
    class _Dead:
        def get(self, *a, **k):
            raise OSError("connection refused")

    monkeypatch.setattr(cc, "get_sync_client", lambda: _Dead())
    assert cc.is_reachable(_BASE) is False
