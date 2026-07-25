"""In-memory live-ops surface for the hub.

This is **not** the durable telemetry stack — that's the OpenTelemetry +
Langfuse export in :mod:`src.observability`. This module is the
complementary "is the kitchen on fire" pane that lives in-process,
in-memory, and answers "what's happening right now?". Each record here
carries the request's Langfuse ``trace_id`` (stashed by
``stash_trace_id_on_ctx`` in :mod:`src.server_common`) so the live-ops view
cross-links to the durable trace.

Holds:
  * a ring buffer of the last ~200 routed requests
  * a ring buffer of the last ~50 non-2xx responses
  * per-backend counters since hub start (req, err, latencies, tokens)
  * a 5-minute ring of system resource samples for sparklines
  * an asyncio fan-out for SSE subscribers
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from .async_fanout import AsyncFanout

logger = logging.getLogger(__name__)

REQUEST_RING_MAX = 200
ERROR_RING_MAX = 50
STATS_RING_MAX = 150  # 150 samples × 2s tick = 5 min


@dataclass
class RequestRecord:
    ts: float                 # wall-clock epoch seconds
    path: str                 # e.g. "/v1/messages"
    model: str = ""           # request model alias the client sent (what was *asked* for)
    # Model id that actually served the request, when the hub resolved the
    # requested name to something else — a role alias (``audio_transcribe``)
    # landing on a chain member, or a chain fallback taking over. Empty when
    # nothing resolved it (the ordinary chat paths) — never guess: a consumer
    # reads "requested=X served=Y" only when this is set (issue #412).
    served_model: str = ""
    # Host id that actually served the request — the missing dimension behind
    # the #405 false pass: a `model=whisper` answered 200 by a host with no
    # whisper bound was a legitimate remote hop to whisper's owner, and the
    # record had no field that said so (``client`` is the *caller*, not the
    # server). Set on the audio paths: the active host for a locally-served
    # request, the owning host for a remote-proxy hop, and whatever the peer
    # hub reports when it answers with its own ``X-Hub-Served-Host``. Empty
    # when nothing resolved it (issue #412).
    served_host: str = ""
    backend: str = ""         # "claude" | "gemini" | "openai" | "whisper" | ""
    status: int = 0           # HTTP status of the response
    latency_ms: float = 0.0
    in_tok: int = 0
    out_tok: int = 0
    cache_read_tok: int = 0
    cache_write_tok: int = 0
    stop_reason: str = ""
    client: str = ""          # client IP (best-effort)
    error_detail: str = ""    # filled when status >= 400
    trace_id: str = ""        # Langfuse trace id (set per routed request)


@dataclass
class BackendCounters:
    requests: int = 0
    errors: int = 0
    in_tok: int = 0
    out_tok: int = 0
    latencies_ms: List[float] = field(default_factory=list)


@dataclass
class StatSample:
    ts: float
    ram_percent: float
    cpu_percent: float = 0.0
    gpu0_vram_percent: Optional[float] = None
    gpu0_util_percent: Optional[float] = None


class ObservabilityCtx:
    """Per-request scratch space stashed on ``request.state.obs_ctx``.

    The middleware creates one of these; the route handler enriches it
    with model + token counts; the middleware finalises and pushes to
    the ring on response.
    """

    __slots__ = (
        "start_ns",
        "model",
        "served_model",
        "served_host",
        "backend",
        "in_tok",
        "out_tok",
        "cache_read_tok",
        "cache_write_tok",
        "stop_reason",
        "error_detail",
        "trace_id",
    )

    def __init__(self) -> None:
        self.start_ns: int = time.monotonic_ns()
        self.model: str = ""
        self.served_model: str = ""
        self.served_host: str = ""
        self.backend: str = ""
        self.in_tok: int = 0
        self.out_tok: int = 0
        self.cache_read_tok: int = 0
        self.cache_write_tok: int = 0
        self.stop_reason: str = ""
        self.error_detail: str = ""
        self.trace_id: str = ""


class Observatory:
    """Singleton holder for ring buffers, counters, and SSE subscribers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: Deque[RequestRecord] = deque(maxlen=REQUEST_RING_MAX)
        self._errors: Deque[RequestRecord] = deque(maxlen=ERROR_RING_MAX)
        self._counters: Dict[str, BackendCounters] = {}
        self._stats: Deque[StatSample] = deque(maxlen=STATS_RING_MAX)
        self._pubsub: AsyncFanout[RequestRecord] = AsyncFanout(maxsize=200)
        self._started_at = time.time()

    # ----------------------------------------------------------- writes
    def record_request(self, rec: RequestRecord) -> None:
        with self._lock:
            self._requests.append(rec)
            if rec.status >= 400 or rec.error_detail:
                self._errors.append(rec)
            # Count against the model that *served*: a role-addressed audio
            # request would otherwise pile every call onto the alias
            # ("audio_transcribe") and hide which backend actually carried the
            # traffic (issue #412). ``served_model`` is only ever set once a
            # model has really answered, so a request nothing served (a strict
            # rejection, or a chain where every candidate was down) falls back
            # to the requested name and is never charged to a model that never
            # ran.
            key = rec.served_model or rec.model or rec.backend or "unknown"
            c = self._counters.setdefault(key, BackendCounters())
            c.requests += 1
            if rec.status >= 400:
                c.errors += 1
            c.in_tok += rec.in_tok
            c.out_tok += rec.out_tok
            c.latencies_ms.append(rec.latency_ms)
            # Bound the latency vector so memory doesn't grow unbounded;
            # only the most-recent 1000 samples per backend feed p50/p95.
            if len(c.latencies_ms) > 1000:
                c.latencies_ms = c.latencies_ms[-1000:]
        self._pubsub.push(rec)

    def record_stat(self, sample: StatSample) -> None:
        with self._lock:
            self._stats.append(sample)

    # ------------------------------------------------------------ reads
    def recent_requests(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._requests)[-limit:]
        return [_rec_to_dict(r) for r in reversed(items)]

    def recent_errors(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._errors)[-limit:]
        return [_rec_to_dict(r) for r in reversed(items)]

    def counters_snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = []
            for key, c in sorted(self._counters.items()):
                p50 = _percentile(c.latencies_ms, 50.0)
                p95 = _percentile(c.latencies_ms, 95.0)
                rows.append(
                    {
                        "key": key,
                        "requests": c.requests,
                        "errors": c.errors,
                        "p50_ms": round(p50, 1),
                        "p95_ms": round(p95, 1),
                        "in_tok": c.in_tok,
                        "out_tok": c.out_tok,
                    }
                )
            return rows

    def stats_snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._stats)
        return [
            {
                "ts": s.ts,
                "ram_percent": s.ram_percent,
                "cpu_percent": s.cpu_percent,
                "gpu0_vram_percent": s.gpu0_vram_percent,
                "gpu0_util_percent": s.gpu0_util_percent,
            }
            for s in items
        ]

    def started_at(self) -> float:
        return self._started_at

    # --------------------------------------------------- SSE fan-out
    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the running event loop so non-async callers can fan
        out without grabbing the loop themselves."""
        self._pubsub.attach_loop(loop)

    def subscribe(self) -> "asyncio.Queue[RequestRecord]":
        return self._pubsub.subscribe()

    def unsubscribe(self, q: "asyncio.Queue[RequestRecord]") -> None:
        self._pubsub.unsubscribe(q)


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[int(k)]
    d0 = s[f] * (c - k)
    d1 = s[c] * (k - f)
    return d0 + d1


def _rec_to_dict(r: RequestRecord) -> Dict[str, Any]:
    return {
        "ts": r.ts,
        "path": r.path,
        "model": r.model,
        "served_model": r.served_model,
        "served_host": r.served_host,
        "backend": r.backend,
        "status": r.status,
        "latency_ms": round(r.latency_ms, 1),
        "in_tok": r.in_tok,
        "out_tok": r.out_tok,
        "cache_read_tok": r.cache_read_tok,
        "cache_write_tok": r.cache_write_tok,
        "stop_reason": r.stop_reason,
        "client": r.client,
        "error_detail": r.error_detail,
        "trace_id": r.trace_id,
    }


# Module-level singleton — imported by both the hub server (the writer)
# and the admin webapp (the reader).
OBS = Observatory()


# ---------------------------------------------------------------- middleware

from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402
from starlette.requests import Request  # noqa: E402

OBSERVABLE_PATHS = (
    "/v1/messages",
    "/v1/chat/completions",
    "/v1/images/generations",
    "/v1/images/edits",
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
)


class ObservatoryMiddleware(BaseHTTPMiddleware):
    """Stash an :class:`ObservabilityCtx` for routed requests, record on response.

    Only acts on the chat/messages routes — model listing and the admin
    sub-app are noise in the request ring.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not any(path.startswith(p) for p in OBSERVABLE_PATHS):
            return await call_next(request)

        ctx = ObservabilityCtx()
        request.state.obs_ctx = ctx
        client = request.client.host if request.client else ""

        # Pull the model name out of the JSON body without consuming the
        # request — starlette caches the bytes on first read.
        try:
            body_bytes = await request.body()
            if body_bytes:
                import json
                doc = json.loads(body_bytes)
                if isinstance(doc, dict):
                    ctx.model = str(doc.get("model") or "")
        except Exception:  # noqa: BLE001 — best-effort peek
            pass

        status = 500
        error_detail = ""
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        except Exception as exc:  # noqa: BLE001 — log + re-raise
            error_detail = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            latency_ms = (time.monotonic_ns() - ctx.start_ns) / 1e6
            OBS.record_request(
                RequestRecord(
                    ts=time.time(),
                    path=path,
                    model=ctx.model,
                    served_model=ctx.served_model,
                    served_host=ctx.served_host,
                    backend=ctx.backend,
                    status=status,
                    latency_ms=latency_ms,
                    in_tok=ctx.in_tok,
                    out_tok=ctx.out_tok,
                    cache_read_tok=ctx.cache_read_tok,
                    cache_write_tok=ctx.cache_write_tok,
                    stop_reason=ctx.stop_reason,
                    client=client,
                    error_detail=ctx.error_detail or error_detail,
                    trace_id=ctx.trace_id,
                )
            )
