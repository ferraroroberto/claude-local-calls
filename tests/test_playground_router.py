"""Unit tests for app_web/routers/playground.py attachment handling.

The Playground route proxies in-process to the hub's /v1/messages. Here
we mock that httpx call and assert the route builds the right content
block for the uploaded file: a `document` block for PDFs, an `image`
block otherwise. The real document->agy path is covered end-to-end by
tests/test_image_blocks.py + the server extractor.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app_web.routers import playground as playground_router
from src import server as server_mod

# Tiny valid single-page PDF (no extractable text needed — we only assert
# the block shape the route hands upstream).
_TINY_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
    b"trailer<</Root 1 0 R>>\n%%EOF\n"
)
_PNG = b"\x89PNG\r\n\x1a\nfake-bytes"


def _mock_upstream(monkeypatch) -> dict:
    """Patch httpx.AsyncClient so the proxied payload is captured, not sent."""
    captured: dict = {}

    class _FakeResp:
        is_success = True
        status_code = 200
        text = ""

        def json(self):
            return {
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, **kw):
            captured["url"] = url
            captured["payload"] = json
            return _FakeResp()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    return captured


def _attachment_block(captured: dict) -> dict:
    content = captured["payload"]["messages"][0]["content"]
    media = [b for b in content if b.get("type") != "text"]
    assert len(media) == 1, content
    return media[0]


def test_playground_pdf_builds_document_block(monkeypatch):
    captured = _mock_upstream(monkeypatch)
    client = TestClient(server_mod.app)
    r = client.post(
        "/admin/api/playground/send",
        data={"model": "claude-haiku-4-5", "prompt": "summarize", "max_tokens": "64"},
        files={"attachment": ("report.pdf", _TINY_PDF, "application/pdf")},
    )
    assert r.status_code == 200, r.text
    block = _attachment_block(captured)
    assert block["type"] == "document"
    assert block["source"]["media_type"] == "application/pdf"
    assert block["source"]["type"] == "base64"


def test_playground_image_still_builds_image_block(monkeypatch):
    captured = _mock_upstream(monkeypatch)
    client = TestClient(server_mod.app)
    r = client.post(
        "/admin/api/playground/send",
        data={"model": "claude-haiku-4-5", "prompt": "describe", "max_tokens": "64"},
        files={"attachment": ("pic.png", _PNG, "image/png")},
    )
    assert r.status_code == 200, r.text
    block = _attachment_block(captured)
    assert block["type"] == "image"
    assert block["source"]["media_type"] == "image/png"


def test_playground_json_builds_document_block(monkeypatch):
    captured = _mock_upstream(monkeypatch)
    client = TestClient(server_mod.app)
    r = client.post(
        "/admin/api/playground/send",
        data={"model": "claude-haiku-4-5", "prompt": "what's in here?",
              "max_tokens": "64"},
        files={"attachment": ("data.json", b'{"k": 1}', "application/json")},
    )
    assert r.status_code == 200, r.text
    block = _attachment_block(captured)
    assert block["type"] == "document"
    assert block["source"]["media_type"] == "application/json"


def test_playground_no_attachment_is_text_only(monkeypatch):
    captured = _mock_upstream(monkeypatch)
    client = TestClient(server_mod.app)
    r = client.post(
        "/admin/api/playground/send",
        data={"model": "claude-haiku-4-5", "prompt": "hi", "max_tokens": "64"},
    )
    assert r.status_code == 200, r.text
    content = captured["payload"]["messages"][0]["content"]
    assert content == [{"type": "text", "text": "hi"}]


def _mock_stream_upstream(monkeypatch) -> dict:
    """Patch httpx.AsyncClient so the proxied /v1/audio/speech *stream* is
    captured and a fake chunked body is returned."""
    captured: dict = {}

    class _FakeStreamResp:
        is_success = True
        status_code = 200
        headers = {"content-type": "audio/L16", "x-sample-rate": "24000"}

        async def aiter_bytes(self):
            yield b"\x01\x00"
            yield b"\x02\x00"

        async def aread(self):
            return b""

    class _FakeStreamCM:
        async def __aenter__(self):
            return _FakeStreamResp()

        async def __aexit__(self, *a):
            return False

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def stream(self, method, url, json=None, **kw):
            captured["url"] = url
            captured["payload"] = json
            return _FakeStreamCM()

        async def aclose(self):
            pass

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    return captured


def test_playground_speak_streaming_forwards_chunks(monkeypatch):
    captured = _mock_stream_upstream(monkeypatch)
    client = TestClient(server_mod.app)
    r = client.post(
        "/admin/api/playground/speak",
        data={"model": "chatterbox-tts", "input": "hi", "stream": "true",
              "response_format": "pcm"},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/L16")
    assert r.headers["x-sample-rate"] == "24000"
    assert r.content == b"\x01\x00\x02\x00"
    # The hub-side streaming flag was forwarded upstream.
    assert captured["payload"]["stream_format"] == "audio"


def test_tts_models_expose_runtime_state_and_capabilities(monkeypatch):
    async def _runtime():
        return {"models": [
            {"id": "piper", "reachable": True},
            {"id": "kokoro", "reachable": False},
        ]}

    monkeypatch.setattr(playground_router, "list_models_for_admin", _runtime)
    client = TestClient(server_mod.app)
    response = client.get("/admin/api/playground/tts_models")
    assert response.status_code == 200, response.text
    by_id = {row["id"]: row for row in response.json()["models"]}
    assert by_id["piper"]["reachable"] is True
    assert by_id["kokoro"]["reachable"] is False
    assert by_id["piper"]["capabilities"]["default_voice"] == "amy"
    assert by_id["kokoro"]["capabilities"]["default_voice"] == "am_michael"
    spanish = [
        voice["id"]
        for voice in by_id["kokoro"]["capabilities"]["voices"]
        if voice["language"] == "es"
    ]
    assert spanish == ["ef_dora", "em_alex", "em_santa"]


def test_playground_speak_forwards_speed(monkeypatch):
    captured: dict = {}

    class _Response:
        is_success = True
        status_code = 200
        content = b"RIFFaudio"
        text = ""
        headers = {"content-type": "audio/wav"}

    class _Client:
        async def post(self, url, json=None, **kwargs):
            captured["payload"] = json
            return _Response()

    monkeypatch.setattr(playground_router, "get_async_client", lambda: _Client())
    client = TestClient(server_mod.app)
    response = client.post(
        "/admin/api/playground/speak",
        data={
            "model": "piper",
            "input": "Compatibility check",
            "voice": "amy",
            "speed": "1.25",
        },
    )
    assert response.status_code == 200, response.text
    assert captured["payload"]["model"] == "piper-tts"
    assert captured["payload"]["voice"] == "amy"
    assert captured["payload"]["speed"] == 1.25


# --- image card: model ordering + per-model cost copy (#498) ---------------

def _image_models():
    client = TestClient(server_mod.app)
    return client.get("/admin/api/playground/image_models").json()


def test_image_card_defaults_to_the_fastest_local_model():
    """The dropdown's first entry is what a user gets without choosing. It must
    be local (not the subscription path) and, among the local models, not the
    one that takes ten minutes — the three differ by a factor of ~24.
    """
    models = _image_models()["models"]
    ids = [m["id"] for m in models]
    if "flux2_klein" not in ids:
        return  # host without the FLUX.2 rows enabled
    assert ids[0] == "flux2_klein"
    assert ids.index("flux2_local") > ids.index("flux1_local")
    # Subscription rows sort after every size-honouring local one.
    if "gemini_image" in ids:
        assert ids.index("gemini_image") > ids.index("flux2_klein")


def test_image_models_carry_per_model_cost_copy():
    """A backend-level note cannot tell a user that one local model is 24x
    slower than another, so each row carries its own."""
    models = {m["id"]: m for m in _image_models()["models"]}
    if "flux2_local" not in models:
        return
    assert "MINUTE" in models["flux2_local"]["note"].upper()
    assert models["flux2_local"]["note"] != models["flux2_klein"]["note"]
    for row in models.values():
        assert row["note"].strip()
