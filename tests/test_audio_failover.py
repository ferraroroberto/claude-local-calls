"""Audio role failover (#348) and the strict-explicit-model contract (#412).

The audio proxy resolves the ``roles.audio.<role>`` chain (primary + fallback)
and, when a candidate's backend is *unavailable* (connection error / 502-503-504),
transparently retries the next model instead of erroring. An explicit concrete
``model=`` is honoured single-shot (no failover, preserving #128) and is never
answered by a *different* model (#412) — it 503s instead, and every request
records which model actually served alongside the one asked for.

The whisper worker binaries are platform/GPU-specific, so these fake the httpx
client and the config rather than driving a real backend.
"""

from __future__ import annotations

import asyncio
import types

import httpx
import pytest
from fastapi import HTTPException

from src import model_registry, server_audio_asr, transcription_glossary
from src.hub_observability import ObservabilityCtx


# --------------------------------------------------------------------------- #
# config helpers (temp models.yaml via conftest's `write_config` fixture)
# --------------------------------------------------------------------------- #
def _two_whisper_config(write_config, monkeypatch, *, transcribe: dict):
    write_config({
        "hub": {"port": 8000},
        "hosts": {
            "pc": {"platform": "win32", "default": True, "enabled": ["wa", "wb"]},
        },
        "models": {
            "wa": {"display_name": "whisper-a", "backend": "whisper",
                   "engine": "whisper-server", "port": 9001},
            "wb": {"display_name": "whisper-b", "backend": "whisper",
                   "engine": "whisper-server", "port": 9002},
        },
        "roles": {"audio": {"transcribe": transcribe}},
    })
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "pc")
    # keep glossary post-processing a no-op so assertions see the raw body
    monkeypatch.setattr(transcription_glossary, "load_rules", lambda: [])


# --------------------------------------------------------------------------- #
# fakes for the proxy round-trip
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, status, content=b"", headers=None):
        self.status_code = status
        self.content = content
        self.headers = headers or {"content-type": "application/json"}


class _FakeClient:
    def __init__(self, handler):
        self._h = handler

    async def post(self, url, **kwargs):
        return self._h(url, kwargs)  # handler returns _FakeResp or raises


class _FakeReq:
    def __init__(self, body=b"----x\r\n", headers=None):
        self._body = body
        self.headers = headers or {"content-type": "multipart/form-data; boundary=x"}
        self.state = types.SimpleNamespace()

    async def body(self):
        return self._body


def _proxy(req):
    return asyncio.run(server_audio_asr._proxy_audio(
        req, default_role="audio_transcribe", ctx_path="/v1/audio/transcriptions"))


# --------------------------------------------------------------------------- #
# audio_role_chain — config parsing
# --------------------------------------------------------------------------- #
def test_role_chain_primary_plus_fallback(monkeypatch, write_config):
    _two_whisper_config(write_config, monkeypatch, transcribe={"model_id": "wa", "fallback": ["wb"]})
    assert model_registry.audio_role_chain("transcribe") == ["wa", "wb"]


def test_role_chain_single_model(monkeypatch, write_config):
    _two_whisper_config(write_config, monkeypatch, transcribe={"model_id": "wa"})
    assert model_registry.audio_role_chain("transcribe") == ["wa"]


def test_role_chain_dedups_repeated_primary(monkeypatch, write_config):
    _two_whisper_config(write_config, monkeypatch, transcribe={"model_id": "wa", "fallback": ["wa", "wb"]})
    assert model_registry.audio_role_chain("transcribe") == ["wa", "wb"]


def test_role_chain_absent_role_is_empty(monkeypatch, write_config):
    _two_whisper_config(write_config, monkeypatch, transcribe={"model_id": "wa"})
    assert model_registry.audio_role_chain("speech") == []


# --------------------------------------------------------------------------- #
# _whisper_chain_for_request — resolution
# --------------------------------------------------------------------------- #
def test_chain_role_default_resolves_config(monkeypatch, write_config):
    _two_whisper_config(write_config, monkeypatch, transcribe={"model_id": "wa", "fallback": ["wb"]})
    chain = server_audio_asr._whisper_chain_for_request("", default_role="audio_transcribe")
    assert [m.id for m in chain] == ["wa", "wb"]


def test_chain_explicit_concrete_model_is_single(monkeypatch, write_config):
    _two_whisper_config(write_config, monkeypatch, transcribe={"model_id": "wa", "fallback": ["wb"]})
    chain = server_audio_asr._whisper_chain_for_request("wb", default_role="audio_transcribe")
    assert [m.id for m in chain] == ["wb"]  # explicit id → no failover chain


# --------------------------------------------------------------------------- #
# _proxy_audio — failover loop
# --------------------------------------------------------------------------- #
def test_failover_on_connection_error(monkeypatch, write_config):
    _two_whisper_config(write_config, monkeypatch, transcribe={"model_id": "wa", "fallback": ["wb"]})
    calls = []

    def handler(url, kwargs):
        calls.append(url)
        if ":9001" in url:
            raise httpx.ConnectError("connection refused")
        return _FakeResp(200, b'{"text":"served by wb"}')

    monkeypatch.setattr(server_audio_asr, "get_async_client", lambda: _FakeClient(handler))
    resp = _proxy(_FakeReq())
    assert resp.status_code == 200
    assert b"served by wb" in resp.body
    assert any(":9001" in u for u in calls) and any(":9002" in u for u in calls)


def test_failover_on_503(monkeypatch, write_config):
    _two_whisper_config(write_config, monkeypatch, transcribe={"model_id": "wa", "fallback": ["wb"]})

    def handler(url, kwargs):
        return _FakeResp(503) if ":9001" in url else _FakeResp(200, b'{"text":"wb"}')

    monkeypatch.setattr(server_audio_asr, "get_async_client", lambda: _FakeClient(handler))
    resp = _proxy(_FakeReq())
    assert resp.status_code == 200 and b"wb" in resp.body


def test_happy_path_no_second_call(monkeypatch, write_config):
    _two_whisper_config(write_config, monkeypatch, transcribe={"model_id": "wa", "fallback": ["wb"]})
    calls = []

    def handler(url, kwargs):
        calls.append(url)
        return _FakeResp(200, b'{"text":"wa"}')

    monkeypatch.setattr(server_audio_asr, "get_async_client", lambda: _FakeClient(handler))
    resp = _proxy(_FakeReq())
    assert resp.status_code == 200 and b"wa" in resp.body
    assert len(calls) == 1 and ":9001" in calls[0]  # primary served, no failover call


def test_all_down_raises_last_error(monkeypatch, write_config):
    _two_whisper_config(write_config, monkeypatch, transcribe={"model_id": "wa", "fallback": ["wb"]})

    def handler(url, kwargs):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(server_audio_asr, "get_async_client", lambda: _FakeClient(handler))
    with pytest.raises(HTTPException) as ei:
        _proxy(_FakeReq())
    assert ei.value.status_code == 503


def test_client_error_not_failed_over(monkeypatch, write_config):
    _two_whisper_config(write_config, monkeypatch, transcribe={"model_id": "wa", "fallback": ["wb"]})
    calls = []

    def handler(url, kwargs):
        calls.append(url)
        return _FakeResp(400, b'{"error":"bad audio"}')  # real client error on wa

    monkeypatch.setattr(server_audio_asr, "get_async_client", lambda: _FakeClient(handler))
    resp = _proxy(_FakeReq())
    assert resp.status_code == 400  # returned as-is
    assert len(calls) == 1  # wb never tried — 4xx is not an availability failure


def _model_body(model_id: str) -> bytes:
    return b'Content-Disposition: form-data; name="model"\r\n\r\n' + model_id.encode() + b"\r\n"


def test_explicit_model_down_does_not_fail_over(monkeypatch, write_config):
    _two_whisper_config(write_config, monkeypatch, transcribe={"model_id": "wa", "fallback": ["wb"]})
    calls = []

    def handler(url, kwargs):
        calls.append(url)
        raise httpx.ConnectError("down")

    monkeypatch.setattr(server_audio_asr, "get_async_client", lambda: _FakeClient(handler))
    req = _FakeReq(body=b'Content-Disposition: form-data; name="model"\r\n\r\nwb\r\n')
    with pytest.raises(HTTPException):
        _proxy(req)
    assert all(":9002" in u for u in calls) and all(":9001" not in u for u in calls)


# --------------------------------------------------------------------------- #
# strict explicit model — no silent substitution (#412)
#
# A role alias may fall back across models (that is #348). An explicit model id
# may not: the #405 drill saw `model=whisper` answered 200 by parakeet on a host
# with no whisper at all, logged as `model: whisper, backend: whisper`.
# --------------------------------------------------------------------------- #
def _strict_config(write_config, monkeypatch):
    """``wa`` is the serveable transcribe primary; ``wdown`` is a configured
    whisper row this host is *not* enabled for — whisper with its whole chain
    down on the host answering the request; ``chatty`` is a non-audio row."""
    write_config({
        "hub": {"port": 8000},
        "hosts": {"pc": {"platform": "win32", "default": True, "enabled": ["wa"]}},
        "models": {
            "wa": {"display_name": "whisper-a", "backend": "whisper",
                   "engine": "whisper-server", "port": 9001},
            "wdown": {"display_name": "whisper-down", "backend": "whisper",
                      "engine": "whisper-server", "port": 9003, "host": "elsewhere"},
            "chatty": {"display_name": "chat-model", "backend": "openai", "port": 9100},
        },
        "roles": {"audio": {"transcribe": {"model_id": "wa"}}},
    })
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "pc")
    monkeypatch.setattr(transcription_glossary, "load_rules", lambda: [])


def test_explicit_unavailable_model_is_503_never_another_model(monkeypatch, write_config):
    """The #412 repro: an explicit id whose chain is down must not be answered
    200 by the role's primary."""
    _strict_config(write_config, monkeypatch)
    calls = []

    def handler(url, kwargs):
        calls.append(url)
        return _FakeResp(200, b'{"text":"served by wa"}')

    monkeypatch.setattr(server_audio_asr, "get_async_client", lambda: _FakeClient(handler))
    with pytest.raises(HTTPException) as ei:
        _proxy(_FakeReq(body=_model_body("wdown")))
    assert ei.value.status_code == 503
    assert "wdown" in ei.value.detail          # names the model that was asked for
    assert not calls                           # wa was never asked to stand in


def test_explicit_non_audio_model_is_400(monkeypatch, write_config):
    _strict_config(write_config, monkeypatch)
    monkeypatch.setattr(
        server_audio_asr, "get_async_client",
        lambda: _FakeClient(lambda url, kwargs: _FakeResp(200, b'{"text":"wa"}')))
    with pytest.raises(HTTPException) as ei:
        _proxy(_FakeReq(body=_model_body("chatty")))
    assert ei.value.status_code == 400


def test_unknown_model_name_still_addresses_the_role(monkeypatch, write_config):
    """``whisper-1`` is the OpenAI SDK's default STT model name — not one of our
    ids, so it is a role request, not a strict model request."""
    _strict_config(write_config, monkeypatch)
    monkeypatch.setattr(
        server_audio_asr, "get_async_client",
        lambda: _FakeClient(lambda url, kwargs: _FakeResp(200, b'{"text":"wa"}')))
    resp = _proxy(_FakeReq(body=_model_body("whisper-1")))
    assert resp.status_code == 200 and b"wa" in resp.body


# --------------------------------------------------------------------------- #
# observability: requested vs served (#412)
# --------------------------------------------------------------------------- #
def _req_with_ctx(body=b"----x\r\n"):
    req = _FakeReq(body=body)
    req.state.obs_ctx = ObservabilityCtx()
    return req


def test_record_carries_requested_role_and_served_model(monkeypatch, write_config):
    _two_whisper_config(write_config, monkeypatch, transcribe={"model_id": "wa", "fallback": ["wb"]})

    def handler(url, kwargs):
        if ":9001" in url:
            raise httpx.ConnectError("down")
        return _FakeResp(200, b'{"text":"wb"}')

    monkeypatch.setattr(server_audio_asr, "get_async_client", lambda: _FakeClient(handler))
    req = _req_with_ctx()
    resp = _proxy(req)
    assert resp.status_code == 200
    assert req.state.obs_ctx.model == "audio_transcribe"   # what was requested
    assert req.state.obs_ctx.served_model == "wb"          # what actually served


def test_response_headers_name_requested_and_served(monkeypatch, write_config):
    _two_whisper_config(write_config, monkeypatch, transcribe={"model_id": "wa", "fallback": ["wb"]})

    def handler(url, kwargs):
        if ":9001" in url:
            raise httpx.ConnectError("down")
        return _FakeResp(200, b'{"text":"wb"}')

    monkeypatch.setattr(server_audio_asr, "get_async_client", lambda: _FakeClient(handler))
    resp = _proxy(_FakeReq())
    assert resp.headers["x-hub-requested-model"] == "audio_transcribe"
    assert resp.headers["x-hub-served-model"] == "wb"


def test_explicit_served_model_matches_requested(monkeypatch, write_config):
    """No substitution -> both names agree, so the UI shows a single model."""
    _two_whisper_config(write_config, monkeypatch, transcribe={"model_id": "wa", "fallback": ["wb"]})
    monkeypatch.setattr(
        server_audio_asr, "get_async_client",
        lambda: _FakeClient(lambda url, kwargs: _FakeResp(200, b'{"text":"wb"}')))
    req = _req_with_ctx(body=_model_body("wb"))
    _proxy(req)
    assert req.state.obs_ctx.model == "wb"
    assert req.state.obs_ctx.served_model == "wb"


# --------------------------------------------------------------------------- #
# a rejection must be the *most* observable outcome, not the least (#412 F1)
#
# The strict gate fires before any backend is contacted, so nothing used to
# stamp request.state.obs_ctx: the ring recorded model='' backend='' and the
# counter key 'unknown'. ObservatoryMiddleware cannot recover it — its fallback
# peeks a JSON body and a multipart request never parses.
# --------------------------------------------------------------------------- #
def test_strict_reject_is_recorded_not_blank(monkeypatch, write_config):
    _strict_config(write_config, monkeypatch)
    monkeypatch.setattr(
        server_audio_asr, "get_async_client",
        lambda: _FakeClient(lambda url, kwargs: _FakeResp(200, b'{"text":"wa"}')))
    req = _req_with_ctx(body=_model_body("wdown"))
    with pytest.raises(HTTPException) as ei:
        _proxy(req)

    ctx = req.state.obs_ctx
    assert ctx.model == "wdown"          # the ring row names what was asked for
    assert ctx.backend == "whisper"      # …and which family it addressed
    assert ctx.served_model == ""        # nothing served it
    assert ctx.served_host == ""
    assert "wdown" in ctx.error_detail   # …and why it failed
    # The same trio a 200 carries, so a client can read the outcome off the
    # response alone (README's promise).
    assert ei.value.headers["X-Hub-Requested-Model"] == "wdown"
    assert ei.value.headers["X-Hub-Served-Model"] == ""
    assert ei.value.headers["X-Hub-Served-Host"] == ""


def test_reject_before_chain_resolution_still_records_the_role(monkeypatch, write_config):
    """A role-addressed request that finds no backend at all records the role,
    not a blank row."""
    write_config({
        "hub": {"port": 8000},
        "hosts": {"pc": {"platform": "win32", "default": True, "enabled": []}},
        "models": {},
        "roles": {"audio": {}},
    })
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "pc")
    monkeypatch.setattr(transcription_glossary, "load_rules", lambda: [])
    req = _req_with_ctx()
    with pytest.raises(HTTPException) as ei:
        _proxy(req)
    assert ei.value.status_code == 503
    assert req.state.obs_ctx.model == "audio_transcribe"
    assert req.state.obs_ctx.error_detail
    assert ei.value.headers["X-Hub-Requested-Model"] == "audio_transcribe"


def test_whole_chain_down_does_not_claim_a_served_model(monkeypatch, write_config):
    """#412 F2: ``served_model`` is written only once a model has answered, so a
    503 is never charged to the last candidate's error counter."""
    _two_whisper_config(write_config, monkeypatch, transcribe={"model_id": "wa", "fallback": ["wb"]})

    def handler(url, kwargs):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(server_audio_asr, "get_async_client", lambda: _FakeClient(handler))
    req = _req_with_ctx()
    with pytest.raises(HTTPException):
        _proxy(req)
    assert req.state.obs_ctx.served_model == ""   # nothing served — not "wb"
    assert req.state.obs_ctx.served_host == ""
    assert req.state.obs_ctx.model == "audio_transcribe"


# --------------------------------------------------------------------------- #
# case-insensitive ownership test (#412 F3)
#
# `resolve` is exact-match, so `model=Whisper` used to look exactly like
# `whisper-1` to the gate: unknown -> role chain -> 200 from another model.
# --------------------------------------------------------------------------- #
def test_capitalized_owned_id_is_still_that_model(monkeypatch, write_config):
    _strict_config(write_config, monkeypatch)
    calls = []

    def handler(url, kwargs):
        calls.append(url)
        return _FakeResp(200, b'{"text":"wa"}')

    monkeypatch.setattr(server_audio_asr, "get_async_client", lambda: _FakeClient(handler))
    req = _req_with_ctx(body=_model_body("Whisper-A"))   # display_name, capitalized
    resp = _proxy(req)
    assert resp.status_code == 200
    assert req.state.obs_ctx.model == "Whisper-A"        # echoed as the caller typed it
    assert req.state.obs_ctx.served_model == "wa"        # …resolved to the real row
    assert len(calls) == 1 and ":9001" in calls[0]


def test_capitalized_unserveable_id_is_rejected_not_substituted(monkeypatch, write_config):
    """The #412 repro with a capital letter: `model=WDOWN` must 503, not fall
    through to the role chain and come back 200 off `wa`."""
    _strict_config(write_config, monkeypatch)
    calls = []

    def handler(url, kwargs):
        calls.append(url)
        return _FakeResp(200, b'{"text":"wa"}')

    monkeypatch.setattr(server_audio_asr, "get_async_client", lambda: _FakeClient(handler))
    with pytest.raises(HTTPException) as ei:
        _proxy(_FakeReq(body=_model_body("WDOWN")))
    assert ei.value.status_code == 503
    assert not calls


def test_role_alias_is_case_insensitive_too(monkeypatch, write_config):
    _strict_config(write_config, monkeypatch)
    monkeypatch.setattr(
        server_audio_asr, "get_async_client",
        lambda: _FakeClient(lambda url, kwargs: _FakeResp(200, b'{"text":"wa"}')))
    resp = _proxy(_FakeReq(body=_model_body("Audio_Transcribe")))
    assert resp.status_code == 200
    assert resp.headers["x-hub-served-model"] == "wa"


def test_resolve_any_is_case_insensitive_resolve_is_not(monkeypatch, write_config):
    """The asymmetry is the point: only the ownership test loosened."""
    _strict_config(write_config, monkeypatch)
    assert model_registry.resolve_any("WHISPER-A").id == "wa"
    assert model_registry.resolve_any("  Wa  ").id == "wa"
    assert model_registry.resolve("WHISPER-A") is None   # exact-match, untouched
    assert model_registry.resolve("wa").id == "wa"


# --------------------------------------------------------------------------- #
# distinct messages for distinct conditions (#412 F4)
# --------------------------------------------------------------------------- #
def test_not_enabled_here_does_not_claim_an_outage(monkeypatch, write_config):
    """The admin Models-tab ping sends `model=<display_name>`; a cross-listed
    row this host doesn't serve must not report an outage that isn't
    happening."""
    _strict_config(write_config, monkeypatch)
    monkeypatch.setattr(
        server_audio_asr, "get_async_client",
        lambda: _FakeClient(lambda url, kwargs: _FakeResp(200, b'{"text":"wa"}')))
    with pytest.raises(HTTPException) as ei:
        _proxy(_FakeReq(body=_model_body("wdown")))
    detail = ei.value.detail
    assert "not enabled on this host" in detail
    assert "elsewhere" in detail          # names the owner, so it is actionable
    assert "chain is down" not in detail  # the old, false claim


def test_backend_really_down_says_so(monkeypatch, write_config):
    """The other condition — the model *is* served here and its port is dead —
    keeps its own message naming the port (#147)."""
    _strict_config(write_config, monkeypatch)

    def handler(url, kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(server_audio_asr, "get_async_client", lambda: _FakeClient(handler))
    with pytest.raises(HTTPException) as ei:
        _proxy(_FakeReq(body=_model_body("wa")))
    assert ei.value.status_code == 503
    assert "not running on :9001" in ei.value.detail
    assert "not enabled on this host" not in ei.value.detail


def test_upstream_404_survives_the_error_funnel_unchanged(monkeypatch, write_config):
    """#412 wrapped `_dispatch_audio` in an `except HTTPException` funnel to
    stamp observability on strict rejects. An upstream status must not get
    caught by it and re-flavoured as a whisper outage: a 404 from the backend
    is *returned* as a Response, never raised, so it passes through intact."""
    _two_whisper_config(write_config, monkeypatch, transcribe={"model_id": "wa", "fallback": ["wb"]})
    calls = []

    def handler(url, kwargs):
        calls.append(url)
        return _FakeResp(404, b'{"error":"no such route"}')

    monkeypatch.setattr(server_audio_asr, "get_async_client", lambda: _FakeClient(handler))
    resp = _proxy(_FakeReq())
    assert resp.status_code == 404  # not 503, not re-worded
    assert b"no such route" in resp.body
    assert len(calls) == 1  # a 4xx is not an availability failure — no fallback


# --------------------------------------------------------------------------- #
# header sanitation (#412 F5)
# --------------------------------------------------------------------------- #
def test_control_bytes_are_stripped_from_the_echoed_header(monkeypatch, write_config):
    """A `model=` carrying a NUL passes the CRLF-only body regex; writing it
    verbatim made h11 raise *after* the transcription had already run."""
    _two_whisper_config(write_config, monkeypatch, transcribe={"model_id": "wa"})
    monkeypatch.setattr(
        server_audio_asr, "get_async_client",
        lambda: _FakeClient(lambda url, kwargs: _FakeResp(200, b'{"text":"wa"}')))
    resp = _proxy(_FakeReq(body=_model_body("wh\x00is\x7fper-1")))
    echoed = resp.headers["x-hub-requested-model"]
    assert echoed == "whisper-1"
    assert all(0x20 <= ord(c) <= 0x7e for c in echoed)


# --------------------------------------------------------------------------- #
# served *host* — the dimension #405 actually needed (#412 F6/F7)
# --------------------------------------------------------------------------- #
def _remote_peer_config(write_config, monkeypatch, *, transcribe: dict):
    """``wremote`` is owned by host ``peer`` and cross-listed here (the real
    whisper/parakeet arrangement); ``wa`` is served locally."""
    write_config({
        "hub": {"port": 8000},
        "hosts": {
            "pc": {"platform": "win32", "default": True, "enabled": ["wa", "wremote"]},
            "peer": {"platform": "darwin", "address": "10.0.0.9", "enabled": ["wremote"]},
        },
        "models": {
            "wa": {"display_name": "whisper-a", "backend": "whisper",
                   "engine": "whisper-server", "port": 9001},
            "wremote": {"display_name": "whisper-remote", "backend": "whisper",
                        "engine": "whisper-server", "port": 9005, "host": "peer"},
        },
        "roles": {"audio": {"transcribe": transcribe}},
    })
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "pc")
    monkeypatch.setattr(transcription_glossary, "load_rules", lambda: [])
    monkeypatch.setattr(
        server_audio_asr, "remote_base_url",
        lambda m: "http://10.0.0.9:8000" if m.id == "wremote" else None)


def test_served_host_is_the_active_host_when_served_locally(monkeypatch, write_config):
    _two_whisper_config(write_config, monkeypatch, transcribe={"model_id": "wa"})
    monkeypatch.setattr(
        server_audio_asr, "get_async_client",
        lambda: _FakeClient(lambda url, kwargs: _FakeResp(200, b'{"text":"wa"}')))
    req = _req_with_ctx()
    resp = _proxy(req)
    assert req.state.obs_ctx.served_host == "pc"
    assert resp.headers["x-hub-served-host"] == "pc"


def test_served_host_is_the_owner_on_a_remote_hop(monkeypatch, write_config):
    """The #405 observation explained: a 200 from a host with no whisper bound
    is a legitimate hop to the owner — now the record says which owner."""
    _remote_peer_config(write_config, monkeypatch, transcribe={"model_id": "wremote", "fallback": ["wa"]})
    urls = []

    def handler(url, kwargs):
        urls.append(url)
        return _FakeResp(200, b'{"text":"remote"}')   # older peer: no X-Hub-* headers

    monkeypatch.setattr(server_audio_asr, "get_async_client", lambda: _FakeClient(handler))
    req = _req_with_ctx()
    resp = _proxy(req)
    assert urls and urls[0].startswith("http://10.0.0.9:8000")
    assert req.state.obs_ctx.served_model == "wremote"
    assert req.state.obs_ctx.served_host == "peer"
    assert resp.headers["x-hub-served-host"] == "peer"


def test_peer_served_headers_win_over_local_guess(monkeypatch, write_config):
    """#412 F7: on the role path the peer hub re-resolves the role itself (the
    forwarded body carries no `model`), so its own answer is the honest one."""
    _remote_peer_config(write_config, monkeypatch, transcribe={"model_id": "wremote", "fallback": ["wa"]})

    def handler(url, kwargs):
        return _FakeResp(200, b'{"text":"parakeet"}', headers={
            "content-type": "application/json",
            "X-Hub-Requested-Model": "audio_transcribe",
            "X-Hub-Served-Model": "parakeet",
            "X-Hub-Served-Host": "gaming",
        })

    monkeypatch.setattr(server_audio_asr, "get_async_client", lambda: _FakeClient(handler))
    req = _req_with_ctx()
    resp = _proxy(req)
    assert req.state.obs_ctx.served_model == "parakeet"   # not "wremote"
    assert req.state.obs_ctx.served_host == "gaming"      # not "peer"
    assert resp.headers["x-hub-served-model"] == "parakeet"
    assert resp.headers["x-hub-served-host"] == "gaming"
    # …and the requested name stays *this* hub's view of what the caller asked.
    assert resp.headers["x-hub-requested-model"] == "audio_transcribe"
    assert len(resp.raw_headers) == len({k.lower() for k, _ in resp.raw_headers})


def test_local_backend_headers_are_not_trusted_as_peer_answers(monkeypatch, write_config):
    """A *local* whisper-server is not a hub; a stray X-Hub-* header from one
    must not override what this hub dispatched."""
    _two_whisper_config(write_config, monkeypatch, transcribe={"model_id": "wa"})

    def handler(url, kwargs):
        return _FakeResp(200, b'{"text":"wa"}', headers={
            "content-type": "application/json",
            "X-Hub-Served-Model": "impostor",
            "X-Hub-Served-Host": "impostor-host",
        })

    monkeypatch.setattr(server_audio_asr, "get_async_client", lambda: _FakeClient(handler))
    req = _req_with_ctx()
    resp = _proxy(req)
    assert req.state.obs_ctx.served_model == "wa"
    assert req.state.obs_ctx.served_host == "pc"
    assert resp.headers["x-hub-served-model"] == "wa"
