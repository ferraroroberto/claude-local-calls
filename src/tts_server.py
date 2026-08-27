"""OpenAI-shape text-to-speech server — the inverse of the whisper STT pair.

Binds the registry row's external ``port`` (piper :8096 / chatterbox :8092 /
orpheus :8093 / kokoro :8095) and exposes ``POST /v1/audio/speech`` accepting the OpenAI body
``{model, input, voice, response_format, speed}`` (plus Chatterbox's
``exaggeration`` / ``cfg_weight`` tone dial) and returning audio bytes.

Launched by ``backend_process.build_command`` for any ``engine: tts-server``
row as ``python -m src.tts_server --model-id <id>`` — the same in-repo-shim
pattern as ``parakeet_server``. The hub proxies ``/v1/audio/speech``
to this port so requests land in the observability ring (``src/server.py``).

The synthesis engine loads in a **background
thread** after startup, so the port answers ``GET /health`` immediately
(``ready`` flips true once the model is warm). Synthesis returns 503 while
loading and surfaces any load error verbatim.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import struct
import sys
import threading
import wave
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from .backend_process import resolve_model_for_engine
from .event_loop import LOOP_FACTORY
from .tts_engines import SpeechRequest, TTSEngine, build_engine

if TYPE_CHECKING:  # numpy is imported lazily at runtime (see encode_audio)
    import numpy as np

# Disable tqdm progress bars for the whole backend process (#104). Chatterbox's
# t3.inference draws a "Sampling" bar to stdout on every synthesis. This backend
# is spawned with stdout piped to the hub; when the hub restarts but the backend
# survives (tray.bat --restart deliberately leaves the TTS ports alone, so the
# new hub *inherits* the old process), that pipe's read end is closed. On Windows
# a write to a broken pipe surfaces as OSError: [Errno 22] Invalid argument — not
# BrokenPipeError — so the bar write blew up mid-generate and every synthesis
# 502'd. Silencing tqdm removes the only per-synthesis stdout write, so an
# orphaned stdout can no longer crash synthesis. setdefault leaves an explicit
# override in place. Must run before chatterbox/tqdm import (lazy in load()).
os.environ.setdefault("TQDM_DISABLE", "1")

log = logging.getLogger("tts_server")

DEFAULT_MODEL_ID = "piper"


class _State:
    def __init__(self) -> None:
        self.engine: Optional[TTSEngine] = None
        self.ready: bool = False
        self.loading: bool = True
        self.error: str = ""
        self.device: str = ""
        self.sample_rate: int = 24000


def _float(body: dict, key: str, default: float) -> float:
    try:
        v = body.get(key)
        return default if v is None else float(v)
    except (TypeError, ValueError):
        return default


def _pcm16_bytes(samples) -> bytes:
    """Mono float32 samples → little-endian signed 16-bit PCM bytes."""
    import numpy as np

    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (pcm * 32767.0).astype("<i2").tobytes()


def _streaming_wav_header(sample_rate: int, channels: int = 1, bits: int = 16) -> bytes:
    """A 44-byte canonical WAV header with open-ended (0xFFFFFFFF) RIFF/data
    sizes, so PCM16 frames can be appended as they synthesize. Browsers'
    ``<audio>`` and ffmpeg play such a stream incrementally."""
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )


def _wav_bytes(samples, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(_pcm16_bytes(samples))
    return buf.getvalue()


def encode_audio(samples: "np.ndarray", sample_rate: int, fmt: str) -> Tuple[bytes, str]:
    """Encode mono float32 samples to the requested ``response_format``.

    ``wav`` (default) and ``pcm`` are produced with the stdlib — no extra
    deps. ``flac`` / ``ogg`` / ``opus`` / ``mp3`` / ``aac`` are attempted via
    ``soundfile`` and fall back to wav (with a logged note) when the
    encoder isn't available, so a request never fails on format alone.
    """
    import numpy as np

    fmt = (fmt or "wav").strip().lower()
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    if fmt in ("wav", "wave", ""):
        return _wav_bytes(pcm, sample_rate), "audio/wav"
    if fmt == "pcm":
        return (pcm * 32767.0).astype("<i2").tobytes(), "audio/L16"

    sf_format = {"flac": "FLAC", "ogg": "OGG", "opus": "OGG", "mp3": "MP3", "aac": "MP3"}.get(fmt)
    media = {
        "flac": "audio/flac", "ogg": "audio/ogg", "opus": "audio/ogg",
        "mp3": "audio/mpeg", "aac": "audio/mpeg",
    }.get(fmt)
    if sf_format is not None:
        try:
            import soundfile as sf

            buf = io.BytesIO()
            sf.write(buf, pcm, int(sample_rate), format=sf_format)
            return buf.getvalue(), media or "application/octet-stream"
        except Exception as exc:  # noqa: BLE001
            log.warning("format %r unavailable (%s) — returning wav", fmt, exc)
    else:
        log.warning("unknown response_format %r — returning wav", fmt)
    return _wav_bytes(pcm, sample_rate), "audio/wav"


def build_app(model_id: str = DEFAULT_MODEL_ID, device: str = "auto") -> FastAPI:
    model = resolve_model_for_engine(model_id, "tts-server")
    state = _State()

    def _load() -> None:
        try:
            engine = build_engine(model, device)
            engine.load()
            state.engine = engine
            state.sample_rate = engine.sample_rate
            # Report the *resolved* device (cuda/cpu/mps) the engine chose,
            # not the "auto" arg — the admin UI needs to show GPU vs CPU.
            state.device = getattr(engine, "device", device) or device
            state.ready = True
            log.info("%s ready on :%s (engine=%s)", model.display_name, model.port, model.tts_engine)
        except Exception as exc:  # noqa: BLE001
            state.error = f"{type(exc).__name__}: {exc}"
            log.error("TTS engine load failed: %s", state.error)
        finally:
            state.loading = False

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state.device = device
        threading.Thread(target=_load, name="tts-load", daemon=True).start()
        try:
            yield
        finally:
            if state.engine is not None:
                try:
                    state.engine.close()
                except Exception:  # noqa: BLE001
                    pass

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {
            "ok": True,
            "model": model.display_name,
            "engine": model.tts_engine,
            "device": state.device,
            "ready": state.ready,
            "loading": state.loading,
            "error": state.error,
            "sample_rate": state.sample_rate,
        }

    @app.get("/")
    async def root() -> Response:
        status = "ready" if state.ready else ("loading" if state.loading else f"error: {state.error}")
        body = (
            f"tts_server: {model.display_name} ({model.tts_engine})\n"
            f"  port    : {model.port}\n"
            f"  device  : {state.device}\n"
            f"  status  : {status}\n"
            f"  POST /v1/audio/speech  {{model, input, voice, response_format, speed}}\n"
        )
        return Response(content=body, media_type="text/plain")

    @app.post("/v1/audio/speech")
    async def speech(request: Request) -> Response:
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}")
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")

        text = body.get("input") or body.get("text")
        if not text or not str(text).strip():
            raise HTTPException(status_code=400, detail="missing required field: input")

        if not state.ready:
            if state.error:
                raise HTTPException(status_code=503, detail=f"TTS engine unavailable: {state.error}")
            raise HTTPException(status_code=503, detail="TTS engine still loading — retry shortly")

        speed = _float(body, "speed", 1.0)
        if abs(speed - 1.0) > 1e-3:
            log.info("speed=%.2f requested; engines without rate control ignore it", speed)

        req = SpeechRequest(
            text=str(text),
            voice=str(body.get("voice") or ""),
            speed=speed,
            exaggeration=_float(body, "exaggeration", 0.5),
            cfg_weight=_float(body, "cfg_weight", 0.5),
        )
        assert state.engine is not None
        engine = state.engine
        sr = state.sample_rate
        try:
            engine.validate_voice(req.voice)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        fmt = str(body.get("response_format") or "wav").strip().lower()
        # OpenAI-native opt-in: stream_format="audio" returns raw chunked
        # bytes that play as they synthesize. Streaming supports wav (a
        # streaming WAV header + PCM16 frames) and pcm (headerless PCM16);
        # any other format falls back to the buffered response below.
        streaming = str(body.get("stream_format") or "").strip().lower() == "audio"
        if streaming and fmt in ("wav", "wave", "", "pcm"):
            is_pcm = fmt == "pcm"

            async def _byte_stream():
                first = True
                try:
                    async for chunk in iterate_in_threadpool(engine.synthesize_stream(req)):
                        data = _pcm16_bytes(chunk)
                        if not data:
                            continue
                        if first and not is_pcm:
                            yield _streaming_wav_header(sr)
                        first = False
                        yield data
                    if first and not is_pcm:
                        # No audio produced — still a valid (empty) WAV.
                        yield _streaming_wav_header(sr)
                except Exception as exc:  # noqa: BLE001 — headers already sent
                    log.error("streaming synthesis failed: %s", exc, exc_info=True)

            media_type = "audio/L16" if is_pcm else "audio/wav"
            return StreamingResponse(
                _byte_stream(), media_type=media_type, headers={"X-Sample-Rate": str(sr)}
            )

        if streaming:
            log.info("stream_format=audio with response_format=%r is not streamable — buffering", fmt)

        try:
            samples = await run_in_threadpool(engine.synthesize, req)
        except Exception as exc:  # noqa: BLE001
            # Log the full traceback — the 502 detail only carries the
            # exception string, which is too thin to diagnose engine faults.
            log.error("synthesis failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=502, detail=f"synthesis failed: {exc}")

        audio, media_type = encode_audio(samples, sr, fmt)
        return Response(content=audio, media_type=media_type)

    return app


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser(prog="python -m src.tts_server")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="registry id of the tts-server row")
    p.add_argument("--device", default="auto", help="auto|cuda|cpu|mps")
    args = p.parse_args(argv)

    model = resolve_model_for_engine(args.model_id, "tts-server")
    if not model.port:
        raise SystemExit(f"model {model.id!r} has no port configured")

    app = build_app(args.model_id, args.device)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=model.port,
        log_level="info",
        access_log=False,
        loop=LOOP_FACTORY,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
