"""Unit tests for POST /v1/images/generations — both backends are mocked.

We monkeypatch `call_gemini_image` (no real Antigravity CLI / Imagen call) and
`generate_image` (no ComfyUI process, no 17 GB checkpoint); the tests assert the
OpenAI-shape contract, the per-backend dispatch, and the routing guards.
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from src import server as server_mod
from src import server_images as images_mod
from src.comfyui_client import ComfyUIError
from src.gemini_cli import GeminiCLIError

# A 1x1 PNG — enough to round-trip through base64 in the response.
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)


def test_images_generation_returns_b64(monkeypatch):
    seen = {}

    def fake_call(prompt, *, reference_image=None, timeout=None):
        seen["prompt"] = prompt
        return {
            "image_bytes": _PNG_BYTES,
            "media_type": "image/png",
            "result_text": "SAVED",
        }

    monkeypatch.setattr(images_mod, "call_gemini_image", fake_call)

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/images/generations",
        json={"model": "gemini_image", "prompt": "a red apple"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "created" in body
    assert len(body["data"]) == 1
    assert base64.b64decode(body["data"][0]["b64_json"]) == _PNG_BYTES
    assert seen["prompt"] == "a red apple"


def test_images_edit_returns_b64(monkeypatch):
    seen = {}

    def fake_call(prompt, *, reference_image=None, timeout=None):
        seen["prompt"] = prompt
        seen["has_ref"] = reference_image is not None
        return {
            "image_bytes": _PNG_BYTES,
            "media_type": "image/png",
            "result_text": "SAVED",
        }

    monkeypatch.setattr(images_mod, "call_gemini_image", fake_call)

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/images/edits",
        data={"model": "gemini_image", "prompt": "make it blue"},
        files={"image": ("duck.png", _PNG_BYTES, "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert base64.b64decode(body["data"][0]["b64_json"]) == _PNG_BYTES
    assert seen["prompt"] == "make it blue"
    assert seen["has_ref"] is True


def test_images_edit_rejects_non_image_model():
    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/images/edits",
        data={"model": "gemini_flash", "prompt": "x"},
        files={"image": ("duck.png", _PNG_BYTES, "image/png")},
    )
    assert r.status_code == 400


def test_images_generation_cli_error_returns_502(monkeypatch):
    def fake_call(prompt, *, size="1024x1024", timeout=300.0):
        raise GeminiCLIError("agy did not produce an image artifact")

    monkeypatch.setattr(images_mod, "call_gemini_image", fake_call)

    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/images/generations",
        json={"model": "gemini_image", "prompt": "a red apple"},
    )
    assert r.status_code == 502


def test_images_generation_rejects_non_image_model():
    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/images/generations",
        json={"model": "gemini_flash", "prompt": "a red apple"},
    )
    assert r.status_code == 400
    assert "image-generation" in r.json()["detail"]


def test_images_generation_rejects_n_gt_1(monkeypatch):
    monkeypatch.setattr(
        images_mod, "call_gemini_image",
        lambda *a, **k: {"image_bytes": _PNG_BYTES, "media_type": "image/png",
                         "result_text": "SAVED"},
    )
    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/images/generations",
        json={"model": "gemini_image", "prompt": "x", "n": 2},
    )
    assert r.status_code == 400


def test_list_models_includes_gemini_image():
    client = TestClient(server_mod.app)
    r = client.get("/v1/models")
    ids = {m["id"] for m in r.json()["data"]}
    assert "gemini_image" in ids


# --- ComfyUI / FLUX backend (#492) ---------------------------------------
# The second image backend. No ComfyUI process is spawned and no checkpoint is
# read: `generate_image` and the on-demand loader are both patched out.

@pytest.fixture
def flux_backend(monkeypatch):
    """Patch out the ComfyUI call + its on-demand spawn, recording both.

    ``ensure_ready`` is what would otherwise try to launch a real ComfyUI, so
    stubbing it is what keeps these tests hermetic — and asserting it was
    called is how we prove the route loads a cold backend before dispatching.
    """
    seen = {"ready": [], "prompt": None, "base_url": None, "ckpt": None,
            "spec": None, "width": None, "height": None, "refine": None}

    monkeypatch.setattr(images_mod, "remote_base_url", lambda m: None)
    monkeypatch.setattr(
        images_mod.on_demand, "ensure_ready",
        lambda model, *a, **k: seen["ready"].append(model.id),
    )

    def fake_generate(prompt, *, base_url, spec=None, width=None, height=None,
                      refine=False, **kwargs):
        seen["prompt"] = prompt
        seen["base_url"] = base_url
        seen["spec"] = spec
        # #498 moved the checkpoint name inside the workflow spec; keep the
        # old key populated so the size/refine assertions stay readable.
        seen["ckpt"] = getattr(spec, "ckpt_name", None)
        seen["width"] = width
        seen["height"] = height
        seen["refine"] = refine
        return {"image_bytes": _PNG_BYTES, "media_type": "image/png",
                "width": width, "height": height,
                "result_text": "comfyui prompt pid-1 in 4.2s"}

    monkeypatch.setattr(images_mod, "generate_image", fake_generate)
    return seen


def test_flux_generation_returns_b64(flux_backend):
    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/images/generations",
        json={"model": "flux1_local", "prompt": "a red apple on white"},
    )
    assert r.status_code == 200
    body = r.json()
    assert base64.b64decode(body["data"][0]["b64_json"]) == _PNG_BYTES
    assert flux_backend["prompt"] == "a red apple on white"


def test_flux_generation_loads_backend_on_demand_first(flux_backend):
    """A cold ComfyUI must be spawned and waited for before the workflow is
    submitted — otherwise the first request after an idle unload 502s."""
    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/images/generations", json={"model": "flux1_local", "prompt": "x"},
    )
    assert r.status_code == 200
    assert flux_backend["ready"] == ["flux1_local"]


def test_flux_generation_targets_the_rows_port_and_checkpoint(flux_backend):
    """The registry row is the single source of truth for both the port the
    client dials and the checkpoint name the workflow graph references."""
    client = TestClient(server_mod.app)
    client.post("/v1/images/generations",
                json={"model": "flux1_local", "prompt": "x"})
    assert flux_backend["base_url"] == "http://127.0.0.1:8188"
    assert flux_backend["ckpt"] == "flux1-dev-fp8.safetensors"


def test_flux_alias_routes_to_the_same_backend(flux_backend):
    client = TestClient(server_mod.app)
    r = client.post("/v1/images/generations",
                    json={"model": "flux", "prompt": "x"})
    assert r.status_code == 200
    assert flux_backend["ready"] == ["flux1_local"]


def test_flux_generation_error_returns_502(monkeypatch):
    monkeypatch.setattr(images_mod, "remote_base_url", lambda m: None)
    monkeypatch.setattr(images_mod.on_demand, "ensure_ready", lambda *a, **k: None)

    def boom(*a, **k):
        raise ComfyUIError("ComfyUI workflow failed: KSampler raised OOM")

    monkeypatch.setattr(images_mod, "generate_image", boom)
    client = TestClient(server_mod.app)
    r = client.post("/v1/images/generations",
                    json={"model": "flux1_local", "prompt": "x"})
    assert r.status_code == 502
    assert "KSampler" in r.json()["detail"]


def test_flux_backend_that_never_comes_up_returns_503(monkeypatch):
    """A failed on-demand load is a distinct condition from a failed
    generation, and must not be reported as a 502 backend error."""
    from src.on_demand import OnDemandNotReady

    monkeypatch.setattr(images_mod, "remote_base_url", lambda m: None)

    def never_ready(model, *a, **k):
        raise OnDemandNotReady("flux1_local did not become ready within 180s")

    monkeypatch.setattr(images_mod.on_demand, "ensure_ready", never_ready)
    client = TestClient(server_mod.app)
    r = client.post("/v1/images/generations",
                    json={"model": "flux1_local", "prompt": "x"})
    assert r.status_code == 503
    assert "did not become ready" in r.json()["detail"]


def test_flux_owned_by_another_host_returns_503(monkeypatch):
    """Image generation isn't proxied between hubs — say so, rather than
    dialling a loopback port nothing is listening on."""
    monkeypatch.setattr(images_mod, "remote_base_url", lambda m: "http://tower:8000")
    client = TestClient(server_mod.app)
    r = client.post("/v1/images/generations",
                    json={"model": "flux1_local", "prompt": "x"})
    assert r.status_code == 503
    assert "not proxied" in r.json()["detail"]


def test_flux_rejects_edits_with_a_capability_message():
    """flux1_local *is* an image model, so the generic 'not an
    image-generation model' rejection would be actively misleading."""
    client = TestClient(server_mod.app)
    r = client.post(
        "/v1/images/edits",
        data={"model": "flux1_local", "prompt": "make it blue"},
        files={"image": ("duck.png", _PNG_BYTES, "image/png")},
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "not edit them" in detail and "gemini_image" in detail


# --- size + refine (#497) -------------------------------------------------

def test_flux_size_preset_is_passed_through(flux_backend):
    client = TestClient(server_mod.app)
    r = client.post("/v1/images/generations",
                    json={"model": "flux1_local", "prompt": "x", "size": "widescreen"})
    assert r.status_code == 200
    assert (flux_backend["width"], flux_backend["height"]) == (1344, 768)


def test_flux_explicit_size_is_passed_through(flux_backend):
    client = TestClient(server_mod.app)
    r = client.post("/v1/images/generations",
                    json={"model": "flux1_local", "prompt": "x", "size": "1216x832"})
    assert r.status_code == 200
    assert (flux_backend["width"], flux_backend["height"]) == (1216, 832)


def test_flux_defaults_to_1024_square_when_size_omitted(flux_backend):
    client = TestClient(server_mod.app)
    client.post("/v1/images/generations", json={"model": "flux1_local", "prompt": "x"})
    assert (flux_backend["width"], flux_backend["height"]) == (1024, 1024)


def test_flux_refine_flag_is_forwarded(flux_backend):
    client = TestClient(server_mod.app)
    client.post("/v1/images/generations",
                json={"model": "flux1_local", "prompt": "x", "size": "4k", "refine": True})
    assert flux_backend["refine"] is True
    assert (flux_backend["width"], flux_backend["height"]) == (3840, 2160)


def test_off_grid_size_returns_400_with_a_suggestion(flux_backend):
    client = TestClient(server_mod.app)
    r = client.post("/v1/images/generations",
                    json={"model": "flux1_local", "prompt": "x", "size": "1000x1000"})
    assert r.status_code == 400
    assert "multiple of 16" in r.json()["detail"]


def test_1920x1080_is_rejected_pointing_at_1088(flux_backend):
    """The most likely thing a user types. Returning 1088 silently would be
    handing back something they didn't ask for."""
    client = TestClient(server_mod.app)
    r = client.post("/v1/images/generations",
                    json={"model": "flux1_local", "prompt": "x", "size": "1920x1080"})
    assert r.status_code == 400
    assert "1088" in r.json()["detail"]


def test_unknown_preset_returns_400(flux_backend):
    client = TestClient(server_mod.app)
    r = client.post("/v1/images/generations",
                    json={"model": "flux1_local", "prompt": "x", "size": "enormous"})
    assert r.status_code == 400


def test_response_reports_the_size_produced(flux_backend):
    client = TestClient(server_mod.app)
    r = client.post("/v1/images/generations",
                    json={"model": "flux1_local", "prompt": "x", "size": "portrait"})
    assert r.json()["size"] == "832x1216"


@pytest.fixture
def gemini_backend(monkeypatch):
    seen = {}

    def fake_call(prompt, *, reference_image=None, timeout=None):
        seen["called"] = True
        return {"image_bytes": _PNG_BYTES, "media_type": "image/png",
                "result_text": "SAVED"}

    monkeypatch.setattr(images_mod, "call_gemini_image", fake_call)
    return seen


def test_gemini_accepts_size_and_ignores_it(gemini_backend):
    """OpenAI clients send `size` routinely — 400-ing on it would break every
    existing Imagen caller. Imagen picks its own dimensions regardless."""
    client = TestClient(server_mod.app)
    r = client.post("/v1/images/generations",
                    json={"model": "gemini_image", "prompt": "x", "size": "4k"})
    assert r.status_code == 200
    assert gemini_backend["called"] is True
    # No size echoed back: Imagen never reports its own dimensions, and
    # guessing would be worse than omitting the field.
    assert "size" not in r.json()


def test_gemini_ignores_a_size_it_could_never_honour(gemini_backend):
    """A size that FLUX would reject must NOT 400 on Imagen: the 16-pixel grid
    is a FLUX constraint, and rejecting an Imagen request with a message about
    FLUX would be explaining a rule that doesn't apply to the model called.
    A field that is ignored must ignore malformed values too."""
    client = TestClient(server_mod.app)
    r = client.post("/v1/images/generations",
                    json={"model": "gemini_image", "prompt": "x", "size": "1920x1080"})
    assert r.status_code == 200
    assert gemini_backend["called"] is True


# --- FLUX.2 rows (#498) ---------------------------------------------------

def _row(model_id):
    from src.model_registry import enabled_models
    row = next((m for m in enabled_models() if m.id == model_id), None)
    if row is None:
        pytest.skip(f"{model_id} is not enabled on this host")
    return row


def test_flux1_row_maps_to_the_all_in_one_spec():
    spec = images_mod.model_spec_for(_row("flux1_local"))
    assert spec.workflow == "flux1"
    assert spec.ckpt_name == "flux1-dev-fp8.safetensors"
    assert spec.unet_name is None


def test_flux2_klein_row_maps_to_a_split_spec():
    """klein is a split-loader model: transformer, text encoder and VAE are
    three separate files, unlike flux1_local's all-in-one checkpoint."""
    spec = images_mod.model_spec_for(_row("flux2_klein"))
    assert spec.workflow == "flux2"
    assert spec.ckpt_name is None
    assert spec.unet_name.endswith(".safetensors")
    assert spec.clip_name and spec.vae_name


def test_klein_uses_the_qwen_encoder_not_mistral():
    """klein pairs with Qwen3-4B. The removed FLUX.2 [dev] row used
    Mistral-Small-24B, and the two are not interchangeable — swapping them does
    not fail at load, it fails deep in sampling with 'mat1 and mat2 shapes
    cannot be multiplied', which is miserable to debug from a config file."""
    spec = images_mod.model_spec_for(_row("flux2_klein"))
    assert "qwen" in spec.clip_name.lower()
    assert "mistral" not in spec.clip_name.lower()


def test_spec_basenames_the_registry_paths():
    """The registry stores repo-relative paths so the downloader knows where
    files belong; ComfyUI resolves by bare filename inside its search path."""
    spec = images_mod.model_spec_for(_row("flux2_klein"))
    for name in (spec.unet_name, spec.clip_name, spec.vae_name):
        assert "/" not in name and "\\" not in name


@pytest.mark.parametrize("model_id", ["flux2_klein"])
def test_flux2_rows_generate_through_the_route(monkeypatch, model_id):
    _row(model_id)
    seen = {}
    monkeypatch.setattr(images_mod, "remote_base_url", lambda m: None)
    monkeypatch.setattr(images_mod.on_demand, "ensure_ready", lambda *a, **k: None)

    def fake_generate(prompt, *, base_url, spec=None, width=None, height=None,
                      refine=False, **kw):
        seen["spec"] = spec
        seen["base_url"] = base_url
        return {"image_bytes": _PNG_BYTES, "media_type": "image/png",
                "width": width, "height": height, "result_text": "ok"}

    monkeypatch.setattr(images_mod, "generate_image", fake_generate)
    client = TestClient(server_mod.app)
    r = client.post("/v1/images/generations",
                    json={"model": model_id, "prompt": "x"})
    assert r.status_code == 200
    assert seen["spec"].workflow == "flux2"


def test_image_rows_have_their_own_ports():
    """One ComfyUI could serve every model, but this repo's process layer keys
    start/stop, inheritance and idle-unload by model id — a shared port would
    let one row's idle unload kill a server another row is mid-request on."""
    ports = {mid: _row(mid).port for mid in ("flux1_local", "flux2_klein")}
    assert len(set(ports.values())) == len(ports), ports


def test_list_models_includes_klein_and_not_the_removed_dev_row():
    """#501 removed flux2_local as unusable (~4 h/image). It must be gone from
    the served surface, not merely unlisted somewhere."""
    client = TestClient(server_mod.app)
    ids = {m["id"] for m in client.get("/v1/models").json()["data"]}
    assert "flux2_klein" in ids
    assert "flux2_local" not in ids


def test_list_models_includes_flux1_local():
    client = TestClient(server_mod.app)
    r = client.get("/v1/models")
    ids = {m["id"] for m in r.json()["data"]}
    assert "flux1_local" in ids


def test_collect_artifact_identifies_by_content_not_extension(tmp_path):
    """A .png file holding JPEG bytes must be reported as image/jpeg —
    `agy` was observed saving JPEG bytes under a .png name (issue #114)."""
    from src.gemini_cli import _collect_image_artifact

    jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 64
    (tmp_path / "generated.png").write_bytes(jpeg_bytes)
    (tmp_path / "notes.txt").write_text("not an image")

    data, media_type = _collect_image_artifact(tmp_path)
    assert data == jpeg_bytes
    assert media_type == "image/jpeg"


def test_collect_artifact_returns_none_when_no_image(tmp_path):
    from src.gemini_cli import _collect_image_artifact

    (tmp_path / "reply.txt").write_text("NO_IMAGE: cannot generate")
    data, media_type = _collect_image_artifact(tmp_path)
    assert data is None and media_type is None
