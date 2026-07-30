"""Helpers shared by the ``/v1/audio/*`` ASR and TTS route modules.

``server_audio_asr.py`` (the whisper transcription/translation proxy) and
``server_audio_tts.py`` (the speech-synthesis proxy) both need the same
small set of header-safety, remote-auth-header, and upstream-error helpers
below — the two used to be one file (``server_audio.py``) before #451 split
it apart for size. Rather than have one of the two import the other's
privates (the exact underscore-private cross-module problem #451 also fixes
in ``code_usage.py``), the shared handful lives here: a leaf module with no
dependency on the FastAPI ``app``, mirroring ``server_common.py``'s own
pattern but scoped to audio only (``server_common.py`` stays the
cross-*all*-route-module leaf — ``server.py``, ``server_audio_asr.py`` /
``server_audio_tts.py`` and ``server_images.py``).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from fastapi import HTTPException

from .model_registry import Model
from .remote_proxy import remote_auth_token_for_model

logger = logging.getLogger(__name__)

# Response headers naming the model/host pair behind an audio request (#412).
# Additive metadata — every OpenAI/Anthropic client ignores unknown headers —
# and the only way a caller can tell "whisper answered" from "the role chain
# fell through to parakeet" without reading the hub's admin API.
HEADER_REQUESTED_MODEL = "X-Hub-Requested-Model"
HEADER_SERVED_MODEL = "X-Hub-Served-Model"
HEADER_SERVED_HOST = "X-Hub-Served-Host"

# Everything outside printable ASCII, and the cap on an echoed header value.
_NON_PRINTABLE_ASCII = re.compile(r"[^\x20-\x7e]")
_HEADER_VALUE_MAX = 200


def _header_safe(value: str) -> str:
    """Make ``value`` safe to write into a response header.

    The requested-model name is peeked out of a *client-supplied* multipart
    body by regex. That pattern rejects CR/LF but nothing else, so a
    ``model=`` carrying a NUL (or any other control byte) used to reach h11
    verbatim and raise ``LocalProtocolError`` — after the upstream
    transcription had already run, turning a completed request into a 500
    (#412). Strip to printable ASCII and cap the length instead.
    """
    return _NON_PRINTABLE_ASCII.sub("", value or "")[:_HEADER_VALUE_MAX]


def _remote_audio_headers(model: Model) -> Optional[dict]:
    """``Authorization`` header for a remote-hub audio call, mirrors
    ``server._remote_headers`` — kept local to avoid a circular import
    (``server.py`` imports these modules' routers).
    """
    token = remote_auth_token_for_model(model)
    return {"Authorization": f"Bearer {token}"} if token else None


def _audio_upstream_error(exc: Exception, *, backend: str, port: int) -> HTTPException:
    """Map an httpx upstream failure to a *distinct* HTTPException.

    A connection failure — the backend port refuses the socket or never
    answers — is a categorically different condition from a transient
    mid-flight error: the backend is wholesale down (crashed, not started, or
    lost its mutex-shared port), not merely slow. Surface it as a ``503`` whose
    message names the port and says the backend isn't running, instead of the
    opaque ``502 "whisper upstream error: All connection attempts failed"`` that
    gave downstream consumers no way to tell "down" from "in flight past
    timeout" (issue #147). Every other upstream error stays a ``502``.
    """
    import httpx as _httpx

    if isinstance(exc, (_httpx.ConnectError, _httpx.ConnectTimeout)):
        logger.warning(
            "⚠️ %s not reachable on :%s — backend not running (connection refused)",
            backend, port,
        )
        return HTTPException(
            status_code=503,
            detail=(
                f"{backend} not running on :{port} — start the backend "
                f"(admin Models tab or its launcher) and retry"
            ),
        )
    return HTTPException(status_code=502, detail=f"{backend} upstream error: {exc}")
