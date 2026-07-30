"""OTLP metrics receiver — ``POST /v1/metrics`` (issue #68).

Split out like ``server_images.py``/``server_audio_asr.py`` so ``server.py``'s
routing core stays readable. Mounted directly on the main hub app (not the
``/admin`` sub-app), unauthenticated — same posture as the existing
``/v1/messages``/``/v1/chat/completions`` routes: this is a personal-
localhost hub, and any client that can already reach the hub's other
unauthenticated model-routing surface can reach this one too.

The only consumer today is Claude Code's own OTel export
(``CLAUDE_CODE_ENABLE_TELEMETRY=1`` + ``OTEL_EXPORTER_OTLP_METRICS_ENDPOINT``
pointed here, see ``docs/telemetry-langfuse.md``), but the route speaks
plain OTLP/HTTP so any OTLP-metrics exporter could push here.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import Response

from .claude_code_otel import ingest_export_request

logger = logging.getLogger(__name__)
router = APIRouter()

# An empty `ExportMetricsServiceResponse` protobuf message serializes to zero
# bytes — valid per the OTLP spec and all a well-behaved exporter needs to
# see a 200 and move on.
_EMPTY_OTLP_RESPONSE = b""


@router.post("/v1/metrics", include_in_schema=False)
async def receive_otlp_metrics(request: Request) -> Response:
    """Ingest an OTLP metrics export. Always returns 200.

    Parsing/persistence failures are logged server-side and swallowed —
    telemetry must never break the sender, mirroring this repo's existing
    "never break the request over telemetry" rule applied to the receive
    side.
    """
    try:
        raw = await request.body()
        n = ingest_export_request(raw)
        if n:
            logger.debug("ℹ️ ingested %d Claude Code OTel usage point(s)", n)
    except Exception:  # noqa: BLE001
        logger.warning("⚠️ /v1/metrics ingestion failed", exc_info=True)
    return Response(content=_EMPTY_OTLP_RESPONSE, media_type="application/x-protobuf")
