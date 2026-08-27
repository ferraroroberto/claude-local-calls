"""On-demand lifecycle + default-language injection for the ASR dispatch (#530).

#530 retired the dedicated lazy-load whisper proxy (``src/whisper_translate_proxy.py``,
``_ChildSupervisor``) — a second, independent spawn-on-first-request /
unload-when-idle implementation running alongside ``src/on_demand.py``. Both of
that shim's genuinely unique behaviours moved into the hub's shared
``server_audio_asr`` dispatch:

* a ``startup: on_demand`` whisper row is now spawned (and idle-tracked) via
  the same ``on_demand``/``server_common.ensure_backend_ready_or_503`` path
  every other on-demand backend uses, in :func:`server_audio_asr._forward_to_candidate`;
* a row's configured default ``--language`` (``audio_proxy.default_language_from_args``)
  is injected into a transcribe request that omits one, in
  :func:`server_audio_asr._dispatch_audio`.

Reuses the fakes from ``test_audio_failover.py`` rather than re-declaring them.
"""

from __future__ import annotations

import asyncio
import io

from fastapi import HTTPException
from starlette.datastructures import FormData, UploadFile

from src import server_audio_asr, transcription_glossary
from tests.test_audio_failover import _FakeClient, _FakeReq, _FakeResp, _model_body


class _FakeFormReq(_FakeReq):
    """A fake request whose ``.form()`` returns real ``starlette`` FormData —
    needed once a candidate declares a default language and the transcribe
    branch has to actually parse the multipart body (#128 via #530)."""

    def __init__(self, *, data: dict, body: bytes):
        super().__init__(body=body)
        self._data = data

    async def form(self) -> FormData:
        upload = UploadFile(file=io.BytesIO(b"RIFF....WAVEfmt "), filename="clip.wav")
        items = list(self._data.items()) + [("file", upload)]
        return FormData(items)


def _proxy(req):
    return asyncio.run(server_audio_asr._proxy_audio(
        req, default_role="audio_transcribe", ctx_path="/v1/audio/transcriptions"))


def _on_demand_config(write_config, monkeypatch, *, transcribe: dict, extra_models: dict):
    write_config({
        "hub": {"port": 8000},
        "hosts": {"pc": {"platform": "win32", "default": True, "enabled": list(extra_models)}},
        "models": extra_models,
        "roles": {"audio": {"transcribe": transcribe}},
    })
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "pc")
    monkeypatch.setattr(transcription_glossary, "load_rules", lambda: [])


# --------------------------------------------------------------------------- #
# on-demand spawn + idle tracking (#422/#530)
# --------------------------------------------------------------------------- #
def test_on_demand_candidate_is_brought_up_before_dispatch(monkeypatch, write_config):
    _on_demand_config(
        write_config, monkeypatch,
        transcribe={"model_id": "wv"},
        extra_models={
            "wv": {"display_name": "whisper-vanilla", "backend": "whisper",
                   "engine": "whisper-server", "port": 9004,
                   "startup": "on_demand", "idle_unload_minutes": 5},
        },
    )
    calls = []
    monkeypatch.setattr(
        server_audio_asr, "ensure_backend_ready_or_503",
        lambda m: calls.append(m.id),
    )
    monkeypatch.setattr(
        server_audio_asr, "get_async_client",
        lambda: _FakeClient(lambda url, kwargs: _FakeResp(200, b'{"text":"ok"}')))

    resp = _proxy(_FakeReq())
    assert resp.status_code == 200
    assert calls == ["wv"]   # brought up exactly once, before the POST


def test_eager_candidate_still_dispatches_through_the_hook(monkeypatch, write_config):
    """An ordinary eager row (no ``startup: on_demand``) passes through the
    same ``ensure_backend_ready_or_503`` hook every local whisper candidate
    now does (#530) — ``on_demand.ensure_ready`` itself no-ops for it, so
    this must never block or fail the request."""
    _on_demand_config(
        write_config, monkeypatch,
        transcribe={"model_id": "wa"},
        extra_models={
            "wa": {"display_name": "whisper-a", "backend": "whisper",
                   "engine": "whisper-server", "port": 9001},
        },
    )
    calls = []
    monkeypatch.setattr(
        server_audio_asr, "ensure_backend_ready_or_503",
        lambda m: calls.append(m.id),
    )
    monkeypatch.setattr(
        server_audio_asr, "get_async_client",
        lambda: _FakeClient(lambda url, kwargs: _FakeResp(200, b'{"text":"ok"}')))

    resp = _proxy(_FakeReq())
    assert resp.status_code == 200
    assert calls == ["wa"]


def test_on_demand_spawn_failure_fails_over_to_next_candidate(monkeypatch, write_config):
    """A cold-load failure (503) is a backend-unavailable condition exactly
    like a dead port (#348) — it must fail over, not propagate raw."""
    _on_demand_config(
        write_config, monkeypatch,
        transcribe={"model_id": "wv", "fallback": ["wa"]},
        extra_models={
            "wv": {"display_name": "whisper-vanilla", "backend": "whisper",
                   "engine": "whisper-server", "port": 9004,
                   "startup": "on_demand", "idle_unload_minutes": 5},
            "wa": {"display_name": "whisper-a", "backend": "whisper",
                   "engine": "whisper-server", "port": 9001},
        },
    )

    def _ready(model):
        if model.id == "wv":
            raise HTTPException(status_code=503, detail="on-demand load timed out")

    monkeypatch.setattr(server_audio_asr, "ensure_backend_ready_or_503", _ready)

    calls = []

    def handler(url, kwargs):
        calls.append(url)
        return _FakeResp(200, b'{"text":"served by wa"}')

    monkeypatch.setattr(server_audio_asr, "get_async_client", lambda: _FakeClient(handler))
    resp = _proxy(_FakeReq())
    assert resp.status_code == 200
    assert b"served by wa" in resp.body
    assert len(calls) == 1 and ":9001" in calls[0]   # wv never reached the POST


# --------------------------------------------------------------------------- #
# default-language injection (#128, folded in from the shim by #530)
# --------------------------------------------------------------------------- #
def test_default_language_injected_when_caller_omits_one(monkeypatch, write_config):
    _on_demand_config(
        write_config, monkeypatch,
        transcribe={"model_id": "wv"},
        extra_models={
            "wv": {"display_name": "whisper-vanilla", "backend": "whisper",
                   "engine": "whisper-server", "port": 9004,
                   "args": ["--language", "auto"]},
        },
    )
    monkeypatch.setattr(server_audio_asr, "ensure_backend_ready_or_503", lambda m: None)
    calls = []

    def handler(url, kwargs):
        calls.append(kwargs)
        return _FakeResp(200, b'{"text":"ok"}')

    monkeypatch.setattr(server_audio_asr, "get_async_client", lambda: _FakeClient(handler))
    req = _FakeFormReq(data={}, body=_model_body("wv"))
    resp = _proxy(req)
    assert resp.status_code == 200
    assert calls[0]["data"]["language"] == "auto"
    assert "file" in calls[0]["files"]


def test_caller_supplied_language_is_never_overwritten(monkeypatch, write_config):
    _on_demand_config(
        write_config, monkeypatch,
        transcribe={"model_id": "wv"},
        extra_models={
            "wv": {"display_name": "whisper-vanilla", "backend": "whisper",
                   "engine": "whisper-server", "port": 9004,
                   "args": ["--language", "auto"]},
        },
    )
    monkeypatch.setattr(server_audio_asr, "ensure_backend_ready_or_503", lambda m: None)
    calls = []

    def handler(url, kwargs):
        calls.append(kwargs)
        return _FakeResp(200, b'{"text":"ok"}')

    monkeypatch.setattr(server_audio_asr, "get_async_client", lambda: _FakeClient(handler))
    req = _FakeFormReq(data={"language": "es"}, body=_model_body("wv"))
    resp = _proxy(req)
    assert resp.status_code == 200
    assert calls[0]["data"]["language"] == "es"


def test_no_default_language_stays_on_raw_bytes_path(monkeypatch, write_config):
    """A row without a configured default language must not pay the
    form-parse cost — this candidate's fake request has no ``.form()``,
    so hitting that path would AttributeError instead of silently passing."""
    _on_demand_config(
        write_config, monkeypatch,
        transcribe={"model_id": "wa"},
        extra_models={
            "wa": {"display_name": "whisper-a", "backend": "whisper",
                   "engine": "whisper-server", "port": 9001},
        },
    )
    monkeypatch.setattr(server_audio_asr, "ensure_backend_ready_or_503", lambda m: None)
    calls = []

    def handler(url, kwargs):
        calls.append(kwargs)
        return _FakeResp(200, b'{"text":"ok"}')

    monkeypatch.setattr(server_audio_asr, "get_async_client", lambda: _FakeClient(handler))
    resp = _proxy(_FakeReq())   # no .form() on this fake — would blow up if called
    assert resp.status_code == 200
    assert "content" in calls[0]
