"""TTS shim (src/tts_server) behaviour — engine fully mocked, no torch/weights.

The heavy synthesis engine is replaced with a fake so these run in CI where
chatterbox-tts / torch are absent. Covers: request validation, the
loading/ready 503 gate, and a happy-path synthesis returning audio bytes.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient

from src import host_profile, model_registry, server as server_mod, server_audio_tts, tts_server
from src.tts_engines import KokoroEngine, OrpheusEngine, PiperEngine, SpeechRequest


def _config(tmp_path, monkeypatch):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(yaml.safe_dump({
        "hub": {"port": 8000},
        "hosts": {"tower": {"platform": "win32", "default": True, "enabled": ["chatterbox"]}},
        "models": {
            "chatterbox": {
                "display_name": "chatterbox-tts",
                "aliases": ["audio_speech"],
                "backend": "tts",
                "engine": "tts-server",
                "tts_engine": "chatterbox",
                "port": 8092,
                "args": ["--device", "cpu"],
            },
        },
    }), encoding="utf-8")
    monkeypatch.setattr(host_profile, "CONFIG_PATH", cfg)
    monkeypatch.setattr(model_registry, "CONFIG_PATH", cfg, raising=False)
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "tower")


class _FakeEngine:
    def __init__(self) -> None:
        self.sample_rate = 24000
        self.last_req = None
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def ready(self) -> bool:
        return self._loaded

    def validate_voice(self, voice: str) -> None:
        return

    def synthesize(self, req: SpeechRequest):
        self.last_req = req
        return [0.0] * 240  # placeholder samples; encode_audio is mocked

    def synthesize_stream(self, req: SpeechRequest):
        self.last_req = req
        yield [0.0] * 120
        yield [0.0] * 120

    def close(self) -> None:
        self._loaded = False


def _client(tmp_path, monkeypatch, fake):
    _config(tmp_path, monkeypatch)
    monkeypatch.setattr(tts_server, "build_engine", lambda model, device: fake)
    # Avoid numpy/soundfile in CI: stub the encoder + streaming byte helpers
    # to fixed bytes.
    monkeypatch.setattr(
        tts_server, "encode_audio",
        lambda samples, sr, fmt: (b"RIFFfake-wav-bytes", "audio/wav"),
    )
    monkeypatch.setattr(tts_server, "_pcm16_bytes", lambda samples: b"\x01\x00")
    monkeypatch.setattr(
        tts_server, "_streaming_wav_header", lambda sr, **kw: b"RIFFstreamhdr"
    )
    app = tts_server.build_app("chatterbox", device="cpu")
    return TestClient(app)


def _wait_ready(client, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.get("/health").json().get("ready"):
            return True
        time.sleep(0.02)
    return False


def test_backend_disables_tqdm_progress_bars():
    """Regression for #104: importing the TTS backend must disable tqdm.

    Chatterbox draws a "Sampling" progress bar to stdout during synthesis.
    When the backend is inherited across a hub restart its stdout is an
    orphaned pipe, and on Windows writing to it raises
    ``OSError: [Errno 22] Invalid argument`` — which 502'd every synthesis.
    Disabling tqdm removes that per-synthesis stdout write.
    """
    import os

    from src import tts_server  # noqa: F401 — import is the behaviour under test

    assert os.environ.get("TQDM_DISABLE") == "1"


def test_capabilities_preserve_existing_consumer_profiles():
    piper = PiperEngine.capabilities()
    orpheus = OrpheusEngine.capabilities()
    assert piper["default_voice"] == "amy"
    assert "amy" in {voice["id"] for voice in piper["voices"]}
    assert orpheus["default_voice"] == "tara"
    assert "tara" in {voice["id"] for voice in orpheus["voices"]}


def test_kokoro_capabilities_offer_spanish_female_and_male_voices():
    capabilities = KokoroEngine.capabilities()
    assert "es" in {language["id"] for language in capabilities["languages"]}
    spanish = {
        voice["id"]: voice["gender"]
        for voice in capabilities["voices"]
        if voice["language"] == "es"
    }
    assert spanish["ef_dora"] == "female"
    assert spanish["em_alex"] == "male"


def test_missing_input_is_400(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, _FakeEngine()) as client:
        r = client.post("/v1/audio/speech", json={"model": "chatterbox-tts"})
        assert r.status_code == 400
        assert "input" in r.json()["detail"]


def test_invalid_explicit_voice_is_400(tmp_path, monkeypatch):
    class _RejectingEngine(_FakeEngine):
        def validate_voice(self, voice: str) -> None:
            raise ValueError(f"unsupported test voice: {voice}")

    with _client(tmp_path, monkeypatch, _RejectingEngine()) as client:
        assert _wait_ready(client)
        r = client.post("/v1/audio/speech", json={
            "model": "chatterbox-tts",
            "input": "hello there",
            "voice": "not-a-real-voice",
        })
        assert r.status_code == 400
        assert r.json()["detail"] == "unsupported test voice: not-a-real-voice"


def test_speech_happy_path_returns_audio(tmp_path, monkeypatch):
    fake = _FakeEngine()
    with _client(tmp_path, monkeypatch, fake) as client:
        assert _wait_ready(client)
        r = client.post("/v1/audio/speech", json={
            "model": "chatterbox-tts",
            "input": "hello there",
            "voice": "default",
            "exaggeration": 0.8,
            "cfg_weight": 0.3,
            "response_format": "wav",
        })
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("audio/")
        assert r.content == b"RIFFfake-wav-bytes"
        # The tone dial reached the engine.
        assert fake.last_req.exaggeration == 0.8
        assert fake.last_req.cfg_weight == 0.3
        assert fake.last_req.text == "hello there"


def test_health_reports_engine_and_readiness(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, _FakeEngine()) as client:
        assert _wait_ready(client)
        body = client.get("/health").json()
        assert body["ok"] is True
        assert body["engine"] == "chatterbox"
        assert body["ready"] is True


def test_encode_audio_wav_roundtrip(tmp_path, monkeypatch):
    """encode_audio produces a real WAV header for the default format."""
    import pytest

    np = pytest.importorskip("numpy")
    samples = np.zeros(240, dtype=np.float32)
    audio, media = tts_server.encode_audio(samples, 24000, "wav")
    assert media == "audio/wav"
    assert audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"


def test_speech_streaming_returns_chunked_wav(tmp_path, monkeypatch):
    """stream_format=audio streams a WAV header followed by PCM16 frames."""
    fake = _FakeEngine()
    with _client(tmp_path, monkeypatch, fake) as client:
        assert _wait_ready(client)
        r = client.post("/v1/audio/speech", json={
            "model": "chatterbox-tts",
            "input": "hello there",
            "stream_format": "audio",
            "response_format": "wav",
        })
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("audio/wav")
        assert r.headers["x-sample-rate"] == "24000"
        # Header emitted once, then one stubbed PCM frame per yielded chunk.
        assert r.content == b"RIFFstreamhdr" + b"\x01\x00" + b"\x01\x00"


def test_speech_streaming_pcm_is_headerless(tmp_path, monkeypatch):
    fake = _FakeEngine()
    with _client(tmp_path, monkeypatch, fake) as client:
        assert _wait_ready(client)
        r = client.post("/v1/audio/speech", json={
            "model": "chatterbox-tts",
            "input": "hello there",
            "stream_format": "audio",
            "response_format": "pcm",
        })
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("audio/L16")
        assert r.content == b"\x01\x00" + b"\x01\x00"  # no WAV header


def test_speech_non_streaming_unchanged(tmp_path, monkeypatch):
    """Without stream_format the buffered single-Response path is unchanged."""
    fake = _FakeEngine()
    with _client(tmp_path, monkeypatch, fake) as client:
        assert _wait_ready(client)
        r = client.post("/v1/audio/speech", json={
            "model": "chatterbox-tts", "input": "hello there", "response_format": "wav",
        })
        assert r.status_code == 200
        assert r.content == b"RIFFfake-wav-bytes"


# ---- Orpheus incremental decode (no torch) ----

def _tok(code: int, pos: int) -> str:
    """Build the <custom_token_N> text that decodes to ``code`` at ``pos``."""
    from src.tts_engines import _SNAC_CODEBOOK

    return f"<custom_token_{code + 10 + (pos % 7) * _SNAC_CODEBOOK}>"


def test_iter_token_ids_reassembles_across_chunks():
    from src.tts_engines import OrpheusEngine

    full = _tok(5, 0) + _tok(6, 1) + _tok(7, 2)
    chunks = [full[:9], full[9:26], full[26:]]  # split mid-tag
    assert list(OrpheusEngine._iter_token_ids(chunks)) == [5, 6, 7]


def test_iter_token_ids_skips_control_without_shifting_frame():
    from src.tts_engines import OrpheusEngine

    # <custom_token_10> → tid 0 at pos 0: a control token, skipped without
    # advancing pos, so the next real token still decodes at pos 0.
    text = "<custom_token_10>" + _tok(5, 0)
    assert list(OrpheusEngine._iter_token_ids([text])) == [5]


def test_synthesize_stream_sliding_window_cadence(monkeypatch):
    import pytest
    from types import SimpleNamespace

    np = pytest.importorskip("numpy")
    from src.tts_engines import OrpheusEngine, SpeechRequest

    eng = OrpheusEngine(SimpleNamespace(internal_port=18093), device="cpu")
    monkeypatch.setattr(eng, "ready", lambda: True)
    # 35 tokens = 5 frames; each tid = 1 + frame index, always in range.
    text = "".join(_tok(1 + p // 7, p) for p in range(35))
    monkeypatch.setattr(eng, "_stream_completion", lambda prompt: [text])
    monkeypatch.setattr(eng, "_decode_window", lambda w: np.ones(2048, dtype=np.float32))

    out = list(eng.synthesize_stream(SpeechRequest(text="hi", voice="tara")))
    # Windows emit at len 28 and len 35 → two 2048-sample segments.
    assert len(out) == 2
    assert all(seg.shape == (2048,) for seg in out)


def test_synthesize_stream_short_input_falls_back_to_whole_clip(monkeypatch):
    import pytest
    from types import SimpleNamespace

    np = pytest.importorskip("numpy")
    from src.tts_engines import OrpheusEngine, SpeechRequest

    eng = OrpheusEngine(SimpleNamespace(internal_port=18093), device="cpu")
    monkeypatch.setattr(eng, "ready", lambda: True)
    text = "".join(_tok(1, p) for p in range(14))  # 2 frames, never reaches 28
    monkeypatch.setattr(eng, "_stream_completion", lambda prompt: [text])
    monkeypatch.setattr(eng, "_decode_snac", lambda codes: np.ones(4096, dtype=np.float32))

    out = list(eng.synthesize_stream(SpeechRequest(text="hi", voice="tara")))
    assert len(out) == 1 and out[0].shape == (4096,)


def test_default_synthesize_stream_yields_single_chunk():
    from src.tts_engines import SpeechRequest, TTSEngine

    class _Single(TTSEngine):
        def synthesize(self, req):
            return [0.1, 0.2, 0.3]

    out = list(_Single().synthesize_stream(SpeechRequest(text="x")))
    assert out == [[0.1, 0.2, 0.3]]


def test_kokoro_engine_voice_fallback_and_build():
    from types import SimpleNamespace

    from src.tts_engines import KokoroEngine, build_engine

    model = SimpleNamespace(
        id="kokoro",
        tts_engine="kokoro",
        model_path="models/kokoro/kokoro-v1.0.int8.onnx",
    )
    eng = build_engine(model, device="cpu")
    assert isinstance(eng, KokoroEngine)
    assert KokoroEngine._voice_for("") == "am_michael"
    assert KokoroEngine._voice_for("default") == "am_michael"
    assert KokoroEngine._voice_for("af_bella") == "af_bella"
    assert KokoroEngine._voice_for("ef_dora") == "ef_dora"
    assert KokoroEngine._voice_for("em_alex") == "em_alex"
    assert KokoroEngine._lang_for_voice("bm_george") == "en-gb"
    assert KokoroEngine._lang_for_voice("ef_dora") == "es"
    with pytest.raises(ValueError, match="unsupported Kokoro voice"):
        KokoroEngine._voice_for("tara")


def test_kokoro_spanish_synthesis_uses_exact_voice_and_language():
    # synthesize() reaches tts_engines/common.py's lazy numpy import; skip in CI
    # (numpy lives in requirements-tts.txt, not installed there) like siblings.
    pytest.importorskip("numpy")
    calls: list[dict] = []

    class _FakeKokoro:
        def create(self, text, **kwargs):
            calls.append({"text": text, **kwargs})
            return [0.0, 0.1], 24000

    eng = KokoroEngine(SimpleNamespace(model_path="unused.onnx"), device="cpu")
    eng.model = _FakeKokoro()
    eng.synthesize(SpeechRequest(
        text="Hola, esta es una prueba.",
        voice="ef_dora",
        speed=1.0,
    ))

    assert calls == [{
        "text": "Hola, esta es una prueba.",
        "voice": "ef_dora",
        "speed": 1.0,
        "lang": "es",
    }]


class _SpeechUpstreamResponse:
    status_code = 200
    content = b"RIFFspanish-audio"
    headers = {"content-type": "audio/wav"}


def test_hub_forwards_exact_spanish_speech_payload(monkeypatch):
    captured: dict = {}

    class _FakeClient:
        async def post(self, url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return _SpeechUpstreamResponse()

    monkeypatch.setattr(server_audio_tts, "get_async_client", lambda: _FakeClient())
    payload = {
        "model": "kokoro-tts",
        "input": "Hola, esta es una prueba.",
        "voice": "em_alex",
        "response_format": "wav",
    }
    client = TestClient(server_mod.app)
    response = client.post("/v1/audio/speech", json=payload)

    assert response.status_code == 200
    assert captured["url"] == "http://127.0.0.1:8095/v1/audio/speech"
    assert json.loads(captured["content"]) == payload


def test_hub_rejects_explicit_unknown_tts_model():
    client = TestClient(server_mod.app)
    response = client.post("/v1/audio/speech", json={
        "model": "not-a-real-tts-model",
        "input": "Hola.",
        "voice": "ef_dora",
    })

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "unknown or unsupported TTS model: not-a-real-tts-model"
    )


def test_piper_engine_voice_mapping_and_build():
    from types import SimpleNamespace

    from src.tts_engines import PiperEngine, build_engine

    model = SimpleNamespace(
        id="piper",
        tts_engine="piper",
        model_path="models/piper/en_US-ryan-medium.onnx",
    )
    eng = build_engine(model, device="cpu")
    assert isinstance(eng, PiperEngine)
    assert PiperEngine.VOICE_FILES["default"] == "en_US-amy-medium.onnx"
    assert PiperEngine.VOICE_FILES["amy"] == "en_US-amy-medium.onnx"
    assert PiperEngine.VOICE_FILES["ryan"] == "en_US-ryan-medium.onnx"
    assert PiperEngine.VOICE_FILES["ryan-high"] == "en_US-ryan-high.onnx"
    assert PiperEngine.VOICE_FILES["lessac"] == "en_US-lessac-medium.onnx"


def test_piper_engine_device_hardcoded_cpu_regardless_of_arg():
    """Piper is CPU-only by design (#371) — the standalone binary is light
    enough it never needed GPU. `device_arg` is accepted for interface parity
    with the other TTS engines (chatterbox/orpheus/kokoro) but never honored;
    `.device` reads "cpu" no matter what config's `--device` arg says."""
    from types import SimpleNamespace

    from src.tts_engines import PiperEngine

    model = SimpleNamespace(id="piper", tts_engine="piper", model_path="unused.onnx")
    eng = PiperEngine(model, device="cuda")
    assert eng.device == "cpu"
    assert eng.device_arg == "cuda"  # recorded, but load() below never reads it


# ---- Orpheus long-input chunking (#130) ----

class _FakeResp:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"content": self._content}


def _chunk_text_of(prompt: str) -> str:
    """Recover the chunk body from a ``<|audio|>voice: BODY<|eot_id|>`` prompt."""
    return prompt.split(": ", 1)[1].rsplit("<|eot_id|>", 1)[0]


def _capped_completion_content(chunk_text: str, n_predict: int) -> str:
    """Simulate llama-server: ~7 audio tokens per character, **capped** at
    ``n_predict`` (this is exactly where the #130 truncation bites a single
    request). Returns whole 7-token frames as ``<custom_token_N>`` text."""
    want = min(len(chunk_text) * 7, n_predict)
    want -= want % 7  # whole frames only
    return "".join(_tok(1 + p // 7, p) for p in range(want))


def _sentences(n: int) -> str:
    return " ".join(f"This is spoken sentence number {i} of many." for i in range(n))


def test_split_into_chunks_short_input_is_returned_unchanged():
    from src.tts_engines import OrpheusEngine

    # Single-chunk transparency: under the budget, the text is untouched
    # (whitespace preserved), so short synthesis is identical to pre-#130.
    assert OrpheusEngine._split_into_chunks("Hello there.") == ["Hello there."]
    assert OrpheusEngine._split_into_chunks("  odd  spacing  ") == ["  odd  spacing  "]
    assert OrpheusEngine._split_into_chunks("") == []
    assert OrpheusEngine._split_into_chunks("   ") == []


def test_split_into_chunks_long_input_packs_under_budget():
    from src.tts_engines import OrpheusEngine, _MAX_CHARS_PER_CHUNK

    text = _sentences(200)
    chunks = OrpheusEngine._split_into_chunks(text)
    assert len(chunks) > 1
    assert all(len(c) <= _MAX_CHARS_PER_CHUNK for c in chunks)
    # Order preserved — first sentence heads the first chunk, last the last.
    assert chunks[0].startswith("This is spoken sentence number 0 ")
    assert "number 199 " in chunks[-1]


def test_split_into_chunks_hard_wraps_oversized_sentence():
    from src.tts_engines import OrpheusEngine, _MAX_CHARS_PER_CHUNK

    giant = ("word " * 300).strip()  # one ~1500-char sentence, no terminator
    chunks = OrpheusEngine._split_into_chunks(giant)
    assert len(chunks) > 1
    assert all(len(c) <= _MAX_CHARS_PER_CHUNK for c in chunks)


def test_synthesize_long_input_is_not_truncated(monkeypatch):
    """A >49.6 s synthesis returns proportional, non-flatlined audio (#130).

    The fake backend caps every single /completion at n_predict tokens — the
    real ceiling that made two different long inputs return byte-identical
    output. With chunking, output length now tracks the input length.
    """
    import pytest
    from types import SimpleNamespace

    np = pytest.importorskip("numpy")
    from src import tts_engines
    from src.tts_engines import OrpheusEngine, SpeechRequest, _N_PREDICT

    eng = OrpheusEngine(SimpleNamespace(internal_port=18093), device="cpu")
    monkeypatch.setattr(eng, "ready", lambda: True)
    monkeypatch.setattr(
        eng, "_decode_snac",
        lambda codes: np.ones((len(codes) // 7) * 2048, dtype=np.float32),
    )

    calls: list = []

    def fake_post(url, json=None, timeout=None):
        body = _chunk_text_of(json["prompt"])
        calls.append(body)
        return _FakeResp(_capped_completion_content(body, json["n_predict"]))

    # Orpheus now posts through its own persistent client (self._client),
    # not the module-level httpx.post (#165); stub that client.
    monkeypatch.setattr(eng, "_client", SimpleNamespace(post=fake_post))

    # The single-request ceiling: 4096 tokens → 585 frames → 585*2048 samples.
    ceiling = (_N_PREDICT // 7) * 2048

    # Short input → exactly one /completion, audio under the ceiling, unchanged.
    short = eng.synthesize(SpeechRequest(text="Hello there.", voice="tara"))
    assert len(calls) == 1
    assert 0 < short.size < ceiling

    # Long input (>49.6 s) → multiple /completion calls, audio well past the
    # single-request ceiling instead of flatlining at it.
    calls.clear()
    long_a = _sentences(60)   # ~2.6k chars ≫ the ~900-char single-call limit
    out_a = eng.synthesize(SpeechRequest(text=long_a, voice="tara"))
    assert len(calls) > 1
    assert out_a.size > ceiling

    # A *longer* input yields *more* audio — proof it is no longer byte-identical
    # / capped (the original bug: 1,513 and 3,446 chars gave identical bytes).
    calls.clear()
    long_b = _sentences(120)  # ~2x long_a
    out_b = eng.synthesize(SpeechRequest(text=long_b, voice="tara"))
    assert out_b.size > out_a.size


def test_synthesize_stream_long_input_streams_past_cap(monkeypatch):
    """The streamed path also chunks: total emitted audio exceeds what a
    single capped /completion could produce (#130)."""
    import pytest
    from types import SimpleNamespace

    np = pytest.importorskip("numpy")
    from src.tts_engines import OrpheusEngine, SpeechRequest, _N_PREDICT

    eng = OrpheusEngine(SimpleNamespace(internal_port=18093), device="cpu")
    monkeypatch.setattr(eng, "ready", lambda: True)
    monkeypatch.setattr(eng, "_decode_window", lambda w: np.ones(2048, dtype=np.float32))

    def fake_stream(prompt):
        yield _capped_completion_content(_chunk_text_of(prompt), _N_PREDICT)

    monkeypatch.setattr(eng, "_stream_completion", fake_stream)

    out = list(eng.synthesize_stream(SpeechRequest(text=_sentences(80), voice="tara")))
    # A single capped chunk emits at most (4096/7 - 3) sliding windows; chunking
    # the long input streams strictly more segments than that ceiling.
    single_chunk_windows = (_N_PREDICT // 7) - 3
    assert len(out) > single_chunk_windows
