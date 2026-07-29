"""Live integration test — Gemini attachment ingestion via the real `agy` CLI.

This test is **skipped by default** and runs only when BOTH:
  - ``shutil.which("agy")`` returns a path (agy is on PATH), AND
  - the environment variable ``HUB_LIVE_GEMINI=1`` is set.

It sends a real PNG and a real PDF (each containing a distinct random token
rendered in a large font) through ``src.gemini_cli.call_gemini`` N times each,
and asserts the model echoes back the exact token.  This is the repeatable form
of the manual probe used to verify the #63 fix: without ``--add-dir`` the `agy`
backend resolves ``@<basename>`` via a filesystem search, which is intermittent
and fails under load; with ``--add-dir`` the reference resolves in-workspace
every time.

Run locally:
    $env:HUB_LIVE_GEMINI = "1"
    .venv/Scripts/python.exe -m pytest tests/test_gemini_attachments_live.py -v

GitHub CI (windows-latest) has no authenticated `agy` / Gemini subscription —
the skip guard ensures this file never runs there.
"""

from __future__ import annotations

import os
import shutil
import string
import tempfile
import random
from pathlib import Path
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Skip guard — must satisfy BOTH conditions, evaluated at collection time.
# ---------------------------------------------------------------------------

_HAS_AGY: bool = shutil.which("agy") is not None
_OPT_IN: bool = bool(os.environ.get("HUB_LIVE_GEMINI"))

_SKIP_REASON = (
    "live Gemini test requires `agy` on PATH AND HUB_LIVE_GEMINI=1 — "
    "set both to run this test locally on the Windows reference box"
)

live_only = pytest.mark.skipif(
    not (_HAS_AGY and _OPT_IN),
    reason=_SKIP_REASON,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# Model label as `agy` displays it in the picker.
_MODEL = "Gemini 3.1 Pro"
# Number of repetitions per media type.
_N = 5
# Font for rendering the random token.
_FONT_PATH = "C:/Windows/Fonts/arialbd.ttf"
_FONT_SIZE = 72


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_token(length: int = 10) -> str:
    """Return an uppercase alphanumeric token unlikely to appear in boilerplate."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def _render_png(token: str, dest: Path) -> Path:
    """Render *token* in a large bold font onto a white PNG and save to *dest*."""
    from PIL import Image, ImageDraw, ImageFont  # Pillow is in requirements.txt

    img = Image.new("RGB", (600, 150), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(_FONT_PATH, _FONT_SIZE)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((20, 20), token, font=font, fill=(0, 0, 0))
    path = dest / f"token_{token}.png"
    img.save(str(path), "PNG")
    return path


def _render_pdf(token: str, dest: Path) -> Path:
    """Render *token* in a large bold font onto a white PDF page via Pillow."""
    from PIL import Image, ImageDraw, ImageFont  # Pillow is in requirements.txt

    img = Image.new("RGB", (600, 150), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(_FONT_PATH, _FONT_SIZE)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((20, 20), token, font=font, fill=(0, 0, 0))
    path = dest / f"token_{token}.pdf"
    img.save(str(path), "PDF", resolution=150)
    return path


def _call(attachment: Path, prompt: str) -> str:
    """Call call_gemini in-process with one attachment and return the reply text."""
    from src.gemini_cli import call_gemini

    result = call_gemini(
        prompt,
        model=_MODEL,
        attachments=[attachment],
        timeout=120.0,
    )
    assert result["is_error"] is False, f"call_gemini returned error: {result}"
    return result["result"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@live_only
def test_image_token_echoed_back(tmp_path: Path) -> None:
    """PNG attachment: model reads and echoes the embedded token N times."""
    failures: list[str] = []
    for i in range(_N):
        token = _random_token()
        png = _render_png(token, tmp_path)
        reply = _call(
            png,
            f"This image contains a single word or code printed in large text. "
            f"Reply with ONLY that exact word or code, nothing else.",
        )
        if token not in reply:
            failures.append(
                f"run {i+1}/{_N}: expected token {token!r} in reply, got: {reply!r}"
            )

    assert not failures, "PNG ingestion failed on some runs:\n" + "\n".join(failures)


@live_only
def test_pdf_token_echoed_back(tmp_path: Path) -> None:
    """PDF attachment: model reads and echoes the embedded token N times."""
    failures: list[str] = []
    for i in range(_N):
        token = _random_token()
        pdf = _render_pdf(token, tmp_path)
        reply = _call(
            pdf,
            f"This PDF contains a single word or code printed in large text. "
            f"Reply with ONLY that exact word or code, nothing else.",
        )
        if token not in reply:
            failures.append(
                f"run {i+1}/{_N}: expected token {token!r} in reply, got: {reply!r}"
            )

    assert not failures, "PDF ingestion failed on some runs:\n" + "\n".join(failures)
