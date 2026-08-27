"""Shared multipart bridging for the hub's whisper audio proxy paths.

``server_audio_asr.py``'s ``_dispatch_audio`` accepts an OpenAI-shaped
multipart audio request on both the transcribe and translate roles and must
hand ``whisper-server`` a byte-identical upstream request:

``whisper-server`` exposes a single inference path
(``/v1/audio/transcriptions``) and honors whisper.cpp's own ``translate=true``
boolean rather than OpenAI's ``task=translate`` string, and (#128) resets each
request's language to ``en`` unless the body carries one — a row configured
with a non-default ``-l``/``--language`` launch flag needs that value injected
into requests that omit ``language``. :func:`build_whisper_upstream_request`
picks the single ``file`` upload (dropping any extra file parts — whisper-server
takes exactly one), bridges ``task`` → ``translate``, and forwards every other
field untouched; :func:`default_language_from_args` reads a row's configured
default off its launch ``args`` so the caller can inject it. This module is
the single home for both so ``_dispatch_audio``'s translate and transcribe
branches (and, before #530, the now-retired lazy-load whisper-translate shim)
cannot silently diverge (issue #132).

Per-caller concerns stay at the call site: each parses the form with its own
error shape and owns its upstream URL, timeout, observability stashing and
error responses.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from starlette.datastructures import FormData, UploadFile

# httpx ``files=`` mapping: field name -> (filename, bytes, content-type).
WhisperFiles = Dict[str, Tuple[str, bytes, str]]


async def build_whisper_upstream_request(
    form: FormData,
) -> Tuple[Optional[UploadFile], Dict[str, str], Optional[WhisperFiles]]:
    """Bridge an OpenAI-shaped audio form into a whisper-server upstream request.

    Returns ``(upload, data, files)`` where:

    * ``upload`` is the single ``file`` part, or ``None`` if the caller sent
      none — each caller raises its own "missing file" error.
    * ``data`` is the non-file form fields, with ``task=translate`` rewritten to
      ``translate=true`` and ``task=transcribe`` (whisper-server's default)
      dropped; every other field is forwarded verbatim.
    * ``files`` is the httpx ``files=`` dict built from ``upload`` (its bytes are
      read here), or ``None`` when there is no upload.
    """
    upload: Optional[UploadFile] = None
    data: Dict[str, str] = {}
    for key, value in form.multi_items():
        if isinstance(value, UploadFile):
            if key == "file" and upload is None:
                upload = value
            # Drop any extra file parts — whisper-server takes exactly one.
            continue
        if key == "task":
            if value == "translate":
                data["translate"] = "true"
            # task=transcribe is whisper-server's default; drop silently.
            continue
        data[key] = value

    files: Optional[WhisperFiles] = None
    if upload is not None:
        file_bytes = await upload.read()
        files = {
            "file": (
                upload.filename or "audio",
                file_bytes,
                upload.content_type or "application/octet-stream",
            )
        }
    return upload, data, files


def default_language_from_args(args: Optional[List[str]]) -> Optional[str]:
    """Pull a whisper-server row's configured spoken-language launch flag
    (``-l``/``--language``) out of its ``args`` (#128).

    whisper-server takes ``--language`` at launch but its HTTP handler resets
    each request's language to ``en`` unless the request body carries one —
    the launch flag does *not* change the per-request default. So a row that
    wants a non-``en`` default (e.g. ``--language auto`` for unbiased
    detection, ``whisper_vanilla``) must have this value injected into every
    request that omits ``language``; a caller that sends its own ``language``
    always wins. Rows without the flag (e.g. ``whisper``, ``whisper_translate``)
    return ``None`` and are left untouched — the caller only pays the form-parse
    cost when this is non-``None``.
    """
    if not args:
        return None
    for i, a in enumerate(args):
        if a in ("-l", "--language") and i + 1 < len(args):
            return args[i + 1]
    return None
