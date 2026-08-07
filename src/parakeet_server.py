"""OpenAI-shape ASR server wrapping the FluidAudio Parakeet CoreML worker.

Binds the registry row's external ``port`` and exposes ``POST
/v1/audio/transcriptions`` accepting an OpenAI multipart upload
(``file``, optional ``model``), returning ``{"text": ...}``.

Launched by ``backend_process.build_command`` for any ``engine:
parakeet-server`` row as ``python -m src.parakeet_server --model-id
<id>`` — the same in-repo-shim pattern as ``tts_server``/
``whisper_translate_proxy``. The hub proxies ``/v1/audio/transcriptions``
to this port so requests land in the observability ring
(``src/server_audio_asr.py``).

darwin-only: keeps one long-lived ``mac/parakeet-worker`` Swift subprocess
warm (the CoreML model loads once) and serializes requests through its
stdin/stdout — see ``mac/parakeet-worker/Sources/ParakeetWorker/main.swift``
for the worker side of this protocol (one JSON result per line). Spike
origin + benchmark results: #138, docs/parakeet-asr-evaluation.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import queue
import subprocess
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .backend_process import resolve_model_for_engine
from .event_loop import LOOP_FACTORY
from .no_window import NO_WINDOW

log = logging.getLogger("parakeet_server")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKER_BIN = PROJECT_ROOT / "mac" / "parakeet-worker" / ".build" / "release" / "ParakeetWorker"

DEFAULT_MODEL_ID = "parakeet"
# Mirrors whisper_translate_proxy.STARTUP_DEADLINE_S / orpheus.LLAMA_READY_DEADLINE_S
# — this worker's sibling process-wrapper modules both bound their startup
# wait; _start_worker previously blocked forever on a wedged CoreML load
# (issue #297).
STARTUP_DEADLINE_S = 60.0
# Bound the per-request worker round-trip too. The startup wait was already
# bounded (#297); the request path was not — and a subtler bug made every
# request hang regardless: the startup ``_pump`` daemon keeps reading the
# worker's stdout for its whole lifetime, so a direct ``proc.stdout.readline()``
# in the request path raced that thread and blocked forever (the pump ate the
# reply). Requests now read the worker's replies from the same pump queue, with
# this deadline as a backstop so a genuinely wedged worker 504s, never hangs.
TRANSCRIBE_DEADLINE_S = 120.0


class _State:
    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.out_q: "Optional[queue.Queue[str]]" = None
        self.lock = asyncio.Lock()


def _start_worker() -> "tuple[subprocess.Popen, queue.Queue[str]]":
    if not WORKER_BIN.exists():
        raise RuntimeError(
            f"ParakeetWorker binary missing at {WORKER_BIN} — "
            f"run `swift build -c release` in mac/parakeet-worker/ "
            f"(the admin Health & install panel's Fix-all does this too)"
        )
    proc = subprocess.Popen(
        [str(WORKER_BIN)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        creationflags=NO_WINDOW,
    )

    # readline() on the worker's stdout blocks indefinitely, so it can't be
    # polled against a deadline directly. Pump lines into a queue from a
    # daemon thread instead, and bound the wait on *that* queue.
    lines: "queue.Queue[str]" = queue.Queue()

    def _pump() -> None:
        while True:
            chunk = proc.stdout.readline()
            lines.put(chunk)
            if not chunk:
                return

    threading.Thread(target=_pump, daemon=True).start()

    deadline = time.monotonic() + STARTUP_DEADLINE_S
    line = ""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            proc.terminate()
            proc.wait()
            raise RuntimeError(
                f"ParakeetWorker did not become ready within {STARTUP_DEADLINE_S:.0f}s"
            )
        try:
            line = lines.get(timeout=remaining)
        except queue.Empty:
            proc.terminate()
            proc.wait()
            raise RuntimeError(
                f"ParakeetWorker did not become ready within {STARTUP_DEADLINE_S:.0f}s"
            )
        if not line:
            err = proc.stderr.read()
            proc.wait()
            raise RuntimeError(f"ParakeetWorker failed to start: {err}")
        if line.strip() == "READY":
            break

    log.info("ParakeetWorker ready (pid=%s)", proc.pid)
    # Hand the pump's queue back so the request path reads replies from it
    # rather than contending with the still-running pump thread on stdout.
    return proc, lines


def _to_wav16k_mono(src: Path) -> Path:
    dst = src.with_suffix(".norm.wav")
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(src), str(dst)],
        check=True, capture_output=True, creationflags=NO_WINDOW,
    )
    return dst


def build_app(model_id: str = DEFAULT_MODEL_ID) -> FastAPI:
    model = resolve_model_for_engine(model_id, "parakeet-server")
    state = _State()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            state.proc, state.out_q = await asyncio.to_thread(_start_worker)
        except Exception as exc:  # noqa: BLE001
            log.error("ParakeetWorker startup failed: %s", exc)
        try:
            yield
        finally:
            if state.proc is not None and state.proc.poll() is None:
                state.proc.terminate()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        alive = state.proc is not None and state.proc.poll() is None
        return {"status": "ok" if alive else "down", "model": model.display_name}

    @app.post("/v1/audio/transcriptions")
    async def transcribe(file: UploadFile = File(...)) -> JSONResponse:
        async with state.lock:
            if state.proc is None or state.proc.poll() is not None:
                log.warning("worker not alive, restarting")
                state.proc, state.out_q = await asyncio.to_thread(_start_worker)
            proc = state.proc
            out_q = state.out_q

            with tempfile.TemporaryDirectory() as tmpdir:
                raw_path = Path(tmpdir) / (file.filename or "upload.bin")
                raw_path.write_bytes(await file.read())
                try:
                    wav_path = await asyncio.to_thread(_to_wav16k_mono, raw_path)
                except subprocess.CalledProcessError as exc:
                    raise HTTPException(
                        400, f"audio conversion failed: {exc.stderr.decode(errors='replace')}"
                    )

                def _roundtrip() -> str:
                    proc.stdin.write(str(wav_path) + "\n")
                    proc.stdin.flush()
                    # Read the reply from the pump queue, NOT proc.stdout
                    # directly — the startup pump thread owns stdout for the
                    # worker's lifetime, so a direct readline() here races it
                    # and hangs. Bounded so a wedged worker 504s, never hangs.
                    return out_q.get(timeout=TRANSCRIBE_DEADLINE_S)

                try:
                    line = await asyncio.to_thread(_roundtrip)
                except queue.Empty:
                    raise HTTPException(504, "parakeet worker timed out — no response")

            if not line:
                err = proc.stderr.read() if proc.stderr else ""
                raise HTTPException(500, f"worker died mid-request: {err}")

            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                raise HTTPException(500, f"worker returned non-JSON: {line!r}")

            if not resp.get("ok"):
                raise HTTPException(500, f"transcription failed: {resp.get('error')}")

            return JSONResponse({"text": resp["text"]})

    return app


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser(prog="python -m src.parakeet_server")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="registry id of the parakeet-server row")
    args = p.parse_args(argv)

    model = resolve_model_for_engine(args.model_id, "parakeet-server")
    if not model.port:
        raise SystemExit(f"model {model.id!r} has no port configured")

    app = build_app(args.model_id)
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
    raise SystemExit(main())
