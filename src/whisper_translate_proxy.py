"""Lazy-loading proxy for the translate-capable whisper instance.

Binds the registry's external port (default 8091) and exposes the same
OpenAI-compatible ``POST /v1/audio/transcriptions`` shape voice clients
already speak. The actual ``whisper-server`` child is *not* started on
boot — it is spawned the first time a request lands and torn down again
after the configured idle window.

Why a proxy at all:
  ``whisper-server`` itself has no shutdown-on-idle. The translate slot
  (medium model on CPU) is rare-use by design, so we run a tiny FastAPI
  shim that owns the child's lifecycle. The shim itself is cheap to keep
  resident; only the model bytes cycle.

Lifecycle:
  * proxy boots                → port 8091 LISTEN, child not running
  * first POST  /v1/audio/...  → spawn child on internal port, wait for
                                  readiness, proxy the request
  * any POST                   → reset idle timer
  * idle_seconds elapse        → SIGTERM child, free RAM
  * proxy shutdown / Ctrl+C    → terminate child cleanly

Configuration is read from the registry entry whose id is passed via
``--model-id`` (default ``whisper_translate``). Required fields on that
entry: ``port`` (external), ``internal_port``, ``idle_seconds``,
``model_path``, optional ``args``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response

from .audio_proxy import build_whisper_upstream_request
from .event_loop import LOOP_FACTORY
from .backend_process import (
    VENDOR_WHISPER,
    whisper_server_binary,
    resolve_model_for_engine,
)
from .http_client import aclose as _aclose_http, get_async_client
from .model_registry import Model
from .process_supervisor import ProcessSupervisor
from .server_process import WIN_NEW_GROUP

log = logging.getLogger("whisper_translate_proxy")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_ID = "whisper_translate"
DEFAULT_INTERNAL_PORT = 18091
DEFAULT_IDLE_SECONDS = 300
STARTUP_DEADLINE_S = 60.0  # cold-load on CPU for medium can take ~15-30s
SHUTDOWN_GRACE_S = 8.0


class _ChildSupervisor:
    """Owns the whisper-server child process: spawn, readiness, idle stop."""

    def __init__(self, model: Model, internal_port: int, idle_seconds: int) -> None:
        self.model = model
        self.internal_port = internal_port
        self.idle_seconds = idle_seconds
        self.proc: Optional[subprocess.Popen] = None
        self._lock = asyncio.Lock()
        self._last_request_at: float = 0.0
        self._reader_thread: Optional[threading.Thread] = None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _build_command(self) -> list[str]:
        bin_path = whisper_server_binary()
        if not bin_path.exists():
            raise RuntimeError(
                f"whisper-server not found at {bin_path} - "
                "run scripts/install_whisper_cpp.py"
            )
        if not self.model.model_path:
            raise RuntimeError(f"model {self.model.id} has no model_path")
        model_path = (PROJECT_ROOT / self.model.model_path).resolve()
        if not model_path.exists():
            raise RuntimeError(
                f"whisper model not found at {model_path} - "
                f"run scripts/download_models.py --only {self.model.id}"
            )
        cmd = [
            str(bin_path),
            "--host", "127.0.0.1",
            "--port", str(self.internal_port),
            "--model", str(model_path),
        ]
        cmd.extend(self.model.args or [])
        return cmd

    def _forward_stdout(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            sys.stdout.write(f"[whisper-child] {line}\n")
            sys.stdout.flush()

    async def _spawn(self) -> None:
        cmd = self._build_command()
        log.info("spawning whisper child: %s", " ".join(cmd))
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        if sys.platform == "win32":
            env["PATH"] = str(VENDOR_WHISPER) + os.pathsep + env.get("PATH", "")
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            creationflags=WIN_NEW_GROUP,
        )
        self.proc = proc
        t = threading.Thread(target=self._forward_stdout, args=(proc,), daemon=True)
        t.start()
        self._reader_thread = t
        await self._wait_ready()

    async def _wait_ready(self) -> None:
        deadline = time.monotonic() + STARTUP_DEADLINE_S
        url = f"http://127.0.0.1:{self.internal_port}/"
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.monotonic() < deadline:
                if not self.alive:
                    raise RuntimeError("whisper-server child exited during startup")
                try:
                    r = await client.get(url)
                    if r.status_code == 200:
                        log.info("whisper child ready on :%d", self.internal_port)
                        return
                except Exception:
                    pass
                await asyncio.sleep(0.4)
        # Timed out — kill and surface.
        await self.stop()
        raise RuntimeError(
            f"whisper-server child did not become ready within {STARTUP_DEADLINE_S:.0f}s"
        )

    async def ensure_running(self) -> None:
        async with self._lock:
            if self.alive:
                return
            await self._spawn()

    async def stop(self, reason: str = "shutdown") -> None:
        async with self._lock:
            await self._stop_locked(reason)

    async def _stop_locked(self, reason: str) -> None:
        proc = self.proc
        if proc is None or proc.poll() is not None:
            self.proc = None
            return
        log.info("stopping whisper child (%s, pid=%s)", reason, proc.pid)
        ok, msg = ProcessSupervisor.stop_popen(proc, terminate_timeout=SHUTDOWN_GRACE_S, kill_timeout=5.0)
        if not ok:
            log.warning("error stopping whisper child: %s", msg)
        self.proc = None

    def touch(self) -> None:
        self._last_request_at = time.monotonic()

    def idle_for(self) -> float:
        if self._last_request_at == 0.0:
            return 0.0
        return time.monotonic() - self._last_request_at


async def _idle_watchdog(sup: _ChildSupervisor) -> None:
    """Background task: stop the child after `idle_seconds` of no traffic."""
    while True:
        await asyncio.sleep(min(30, max(5, sup.idle_seconds // 4)))
        if not sup.alive:
            continue
        if sup.idle_for() >= sup.idle_seconds:
            try:
                await sup.stop(reason=f"idle>{sup.idle_seconds}s")
            except Exception as exc:
                log.warning("idle stop failed: %s", exc)


def build_app(model_id: str = DEFAULT_MODEL_ID) -> FastAPI:
    model = resolve_model_for_engine(model_id, "whisper-server-lazy")
    internal_port = model.internal_port or DEFAULT_INTERNAL_PORT
    idle_seconds = model.idle_seconds or DEFAULT_IDLE_SECONDS
    sup = _ChildSupervisor(model, internal_port, idle_seconds)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        watchdog = asyncio.create_task(_idle_watchdog(sup))
        log.info(
            "%s proxy ready on :%d (internal :%d, idle=%ds, model=%s)",
            model.display_name, model.port, internal_port, idle_seconds,
            model.model_path,
        )
        try:
            yield
        finally:
            watchdog.cancel()
            try:
                await watchdog
            except (asyncio.CancelledError, Exception):
                pass
            await _aclose_http()
            await sup.stop(reason="proxy shutdown")

    app = FastAPI(lifespan=lifespan)
    inference_path = _inference_path_from_args(model.args)
    default_language = _default_language_from_args(model.args)

    @app.get("/")
    async def root() -> Response:
        # Match whisper-server's contract: 200 means "I'm here". Whether
        # the child is actually loaded is hidden from callers — the next
        # POST will warm it.
        body = (
            f"whisper_translate_proxy: {model.display_name}\n"
            f"  external port : {model.port}\n"
            f"  internal port : {internal_port}\n"
            f"  idle window   : {idle_seconds}s\n"
            f"  child running : {sup.alive}\n"
        )
        return Response(content=body, media_type="text/plain")

    async def _proxy_audio(request: Request) -> Response:
        sup.touch()
        try:
            await sup.ensure_running()
        except Exception as exc:
            log.error("ensure_running failed: %s", exc)
            return Response(
                content=f'{{"error": "whisper-server unavailable: {exc}"}}',
                status_code=503,
                media_type="application/json",
            )
        sup.touch()

        # Parse the inbound multipart form, then bridge it to whisper-server's
        # upstream request shape via the shared helper (the hub's _proxy_audio
        # in src/server.py calls the same helper — issue #132): forward every
        # field, drop extra file parts, and map OpenAI's `task=translate` to
        # whisper.cpp's `translate=true` boolean.
        try:
            form = await request.form()
        except Exception as exc:
            log.warning("multipart parse failed: %s", exc)
            return Response(
                content=f'{{"error": "invalid multipart body: {exc}"}}',
                status_code=400,
                media_type="application/json",
            )

        upload, data, files = await build_whisper_upstream_request(form)

        # Apply the row's configured default language when the caller did
        # not specify one (#128). whisper-server otherwise forces `en` per
        # request regardless of the launch-level --language flag, so this
        # is the only effective lever for an auto-detect default. A caller
        # that sends its own `language` always wins (we never overwrite).
        if default_language and not data.get("language"):
            data["language"] = default_language

        if upload is None:
            return Response(
                content='{"error": "missing required form field: file"}',
                status_code=400,
                media_type="application/json",
            )

        url = f"http://127.0.0.1:{internal_port}{inference_path}"
        r = await get_async_client().post(
            url, files=files, data=data, timeout=httpx.Timeout(300.0)
        )
        sup.touch()
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type"),
        )

    # whisper-server's --inference-path is configurable; mount that exact
    # path so clients can hit the proxy with the same URL they'd use
    # against a non-lazy whisper-server.
    app.add_api_route(inference_path, _proxy_audio, methods=["POST"])
    return app


def _inference_path_from_args(args: list[str]) -> str:
    """Pull --inference-path X out of args; default matches whisper-server."""
    default = "/v1/audio/transcriptions"
    if not args:
        return default
    for i, a in enumerate(args):
        if a == "--inference-path" and i + 1 < len(args):
            return args[i + 1]
    return default


def _default_language_from_args(args: list[str]) -> Optional[str]:
    """Pull the configured spoken language (``-l`` / ``--language``) out of args.

    whisper-server takes ``--language`` at launch but, empirically (#128),
    its HTTP handler resets each request's language to ``en`` unless the
    request body carries one — the launch flag does *not* change the
    per-request default. So a row that wants a non-``en`` default (e.g.
    ``--language auto`` for unbiased detection) must have it injected into
    every request that omits ``language``. Rows without the flag (e.g.
    ``whisper_translate``) return ``None`` and are left untouched.
    """
    if not args:
        return None
    for i, a in enumerate(args):
        if a in ("-l", "--language") and i + 1 < len(args):
            return args[i + 1]
    return None


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser(prog="python -m src.whisper_translate_proxy")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                   help="registry id of the lazy whisper model")
    args = p.parse_args(argv)

    model = resolve_model_for_engine(args.model_id, "whisper-server-lazy")
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
    sys.exit(main())
