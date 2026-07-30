"""Models-tab Ping is protocol-aware, and whisper reachability is correct.

Two coupled fixes (issue #46, also resolving the #39 reachability finding):

  * ``is_reachable`` used ``model.url.rstrip("/v1")`` — ``str.rstrip`` strips
    a *character set*, so a whisper port ending in ``1`` (``:8091``) got its
    last digit eaten, the probe hit a dead ``:809``, and the translate row's
    Ping icon stayed dimmed. The fix uses ``removesuffix``.
  * ``model_ping`` always sent a chat request, which whisper (an ASR backend)
    rejects with ``400``. Whisper rows now get a real audio-transcription
    probe; chat/llama-server/claude rows keep the chat probe.
"""

from __future__ import annotations

import os

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

import wave  # noqa: E402
from io import BytesIO  # noqa: E402

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src import backend_process as bp  # noqa: E402
from src.model_registry import Model  # noqa: E402


# --------------------------------------------------------------------------
# is_reachable: the port-mangling regression
# --------------------------------------------------------------------------

def test_is_reachable_whisper_probes_correct_port(monkeypatch):
    """:8091 must be probed as :8091, not the mangled :809."""
    captured = {}

    class _Resp:
        status_code = 200

    class _FakeClient:
        def get(self, url, timeout=None):
            captured["url"] = url
            return _Resp()

    # is_reachable routes through the shared pooled client (issue #450), not
    # a bare httpx.get — patch the client getter, not httpx itself.
    monkeypatch.setattr(bp, "get_sync_client", lambda: _FakeClient())

    m = Model(
        id="whisper_translate",
        display_name="whisper-medium-translate",
        backend="whisper",
        engine="whisper-server",
        port=8091,
    )
    assert bp.is_reachable(m, timeout=0.1) is True
    assert captured["url"] == "http://127.0.0.1:8091/"


def test_silent_wav_is_decodable():
    """The ping probe clip must be a valid mono 16-bit PCM WAV."""
    from app_web.routers.models import _silent_wav

    data = _silent_wav()
    with wave.open(BytesIO(data), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 16000
        assert w.getnframes() > 0


# --------------------------------------------------------------------------
# model_ping: protocol-aware probe routing
# --------------------------------------------------------------------------

class _FakePingResp:
    status_code = 200
    is_success = True
    text = "{}"

    def json(self):
        return {"text": ""}


def _patch_async_client(monkeypatch, calls: list):
    """Replace httpx.AsyncClient so the ping makes no real network call."""

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, **kwargs):
            calls.append({"url": url, "kwargs": kwargs})
            return _FakePingResp()

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)


def _admin_client() -> TestClient:
    from app_web.server import create_app

    return TestClient(create_app())


def test_whisper_ping_probes_audio_endpoint(monkeypatch):
    """A whisper row pings the audio transcription endpoint with a file."""
    calls: list = []
    _patch_async_client(monkeypatch, calls)

    resp = _admin_client().post("/api/models/whisper/ping")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    assert len(calls) == 1
    assert calls[0]["url"].endswith("/v1/audio/transcriptions")
    assert "files" in calls[0]["kwargs"]
    assert calls[0]["kwargs"]["data"]["model"] == "whisper-large-v3-turbo"


def test_chat_ping_probes_messages_endpoint(monkeypatch):
    """A chat row still pings /v1/messages with a 1-token prompt."""
    calls: list = []
    _patch_async_client(monkeypatch, calls)

    resp = _admin_client().post("/api/models/qwen35_4b/ping")
    assert resp.status_code == 200

    assert len(calls) == 1
    assert calls[0]["url"].endswith("/v1/messages")
    assert calls[0]["kwargs"]["json"]["max_tokens"] == 1
