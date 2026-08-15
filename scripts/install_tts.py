"""Install TTS deps and pre-warm the weights for enabled tts-server rows.

Mirrors ``install_whisper_cpp.py`` in spirit, but the TTS engines are
Python packages rather than a vendored binary:

  1. ``pip install -r requirements-tts.txt`` (Chatterbox's runtime deps +
     snac + soundfile — pulls torch), then ``chatterbox-tts`` itself with
     ``--no-deps``. Chatterbox's own dep set pins ``spacy-pkuseg`` (no
     Python-3.14 wheel → needs a C++ compiler) and ``gradio`` (a demo-UI
     framework); neither is needed for English server synthesis, so we
     install the package without letting pip pull them. Kept out of the base
     ``requirements.txt`` so non-TTS hosts stay lean.
  2. Pre-fetch the model weights so the first ``/v1/audio/speech`` request
     isn't a cold download:
       - Chatterbox + SNAC stream from the HF cache via ``from_pretrained``.
       - Orpheus is a GGUF — fetched into ``models/`` by ``download_models``.
       - Piper is a standalone binary plus ONNX voices.

Each step is best-effort: a failure is logged and the script continues, so a
flaky network fetch doesn't abort the whole install.

Usage:
    python scripts/install_tts.py
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import no_window_flags  # noqa: E402
from src.model_registry import enabled_models  # noqa: E402


# Installed --no-deps after the curated runtime set above — see module
# docstring (avoids spacy-pkuseg's compiler build + the gradio UI tree).
_CHATTERBOX_PIN = "chatterbox-tts==0.1.7"
_KOKORO_ASSETS = {
    "kokoro-v1.0.int8.onnx": (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.0/kokoro-v1.0.int8.onnx"
    ),
    "voices-v1.0.bin": (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.0/voices-v1.0.bin"
    ),
}
_KOKORO_SPANISH_VOICES = ("ef_dora", "em_alex")
_PIPER_RELEASE_URL = os.environ.get(
    "HUB_TTS_PIPER_URL",
    "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip",
)
_PIPER_VOICES = {
    "en_US-amy-medium.onnx": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
        "en/en_US/amy/medium/en_US-amy-medium.onnx?download=true"
    ),
    "en_US-amy-medium.onnx.json": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
        "en/en_US/amy/medium/en_US-amy-medium.onnx.json?download=true"
    ),
    "en_US-ryan-medium.onnx": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
        "en/en_US/ryan/medium/en_US-ryan-medium.onnx?download=true"
    ),
    "en_US-ryan-medium.onnx.json": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
        "en/en_US/ryan/medium/en_US-ryan-medium.onnx.json?download=true"
    ),
    "en_US-ryan-high.onnx": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
        "en/en_US/ryan/high/en_US-ryan-high.onnx?download=true"
    ),
    "en_US-ryan-high.onnx.json": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
        "en/en_US/ryan/high/en_US-ryan-high.onnx.json?download=true"
    ),
    "en_US-lessac-medium.onnx": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
        "en/en_US/lessac/medium/en_US-lessac-medium.onnx?download=true"
    ),
    "en_US-lessac-medium.onnx.json": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
        "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json?download=true"
    ),
}

# CUDA torch override. requirements-tts.txt pulls the CPU torch wheel from
# PyPI (the only Windows torch there), which runs synthesis at ~real-time on
# CPU — too slow. On an NVIDIA host we replace it with the CUDA build so the
# engines load on the GPU (~5x faster). The default index + pins are the pair
# validated on this repo's reference box (Python 3.14 / CUDA 13 / Blackwell:
# torch 2.11.0+cu130 — torchaudio has no 2.12 on cu130). Override per host via
# HUB_TTS_TORCH_INDEX / HUB_TTS_TORCH_SPEC if your Python/driver differ.
_CUDA_TORCH_INDEX = os.environ.get(
    "HUB_TTS_TORCH_INDEX", "https://download.pytorch.org/whl/cu130"
)
_CUDA_TORCH_SPEC = os.environ.get(
    "HUB_TTS_TORCH_SPEC", "torch==2.11.0+cu130 torchaudio==2.11.0+cu130"
).split()
_CUDA_ORT_SPEC = os.environ.get("HUB_TTS_ORT_SPEC", "onnxruntime-gpu==1.27.0")


def _pip_install_requirements() -> None:
    req = PROJECT_ROOT / "requirements-tts.txt"
    log.info("pip install -r %s", req)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req)],
                   check=True, creationflags=no_window_flags())
    log.info("pip install --no-deps %s", _CHATTERBOX_PIN)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", _CHATTERBOX_PIN],
        check=True,
        creationflags=no_window_flags(),
    )


def _install_cuda_torch() -> None:
    """Replace the CPU torch with a CUDA build on NVIDIA hosts (best-effort).

    Skipped silently when no NVIDIA GPU is detected (``nvidia-smi`` absent) —
    those hosts keep the CPU torch from requirements. A failure here is
    logged but non-fatal: synthesis still works on CPU, just slowly.
    """
    if not shutil.which("nvidia-smi"):
        log.info("no NVIDIA GPU detected (nvidia-smi absent) — keeping CPU torch")
        return
    log.info("NVIDIA GPU detected — installing CUDA torch (%s)", " ".join(_CUDA_TORCH_SPEC))
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade",
             "--index-url", _CUDA_TORCH_INDEX, *_CUDA_TORCH_SPEC],
            check=True,
            creationflags=no_window_flags(),
        )
        log.info("  CUDA torch installed")
    except subprocess.CalledProcessError as exc:
        log.warning(
            "CUDA torch install failed (%s) — falling back to CPU torch. "
            "Set HUB_TTS_TORCH_INDEX/HUB_TTS_TORCH_SPEC for your Python/driver.",
            exc,
        )


def _install_cuda_onnxruntime() -> None:
    """Install ONNX Runtime GPU on NVIDIA hosts for Kokoro."""
    if not shutil.which("nvidia-smi"):
        return
    log.info("NVIDIA GPU detected — installing %s for Kokoro", _CUDA_ORT_SPEC)
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", _CUDA_ORT_SPEC],
            check=True,
            creationflags=no_window_flags(),
        )
        log.info("  ONNX Runtime GPU installed")
    except subprocess.CalledProcessError as exc:
        log.warning(
            "ONNX Runtime GPU install failed (%s) — Kokoro will use CPU. "
            "Set HUB_TTS_ORT_SPEC for your Python/CUDA stack.",
            exc,
        )


def _warm_chatterbox() -> None:
    try:
        from chatterbox.tts import ChatterboxTTS  # type: ignore

        log.info("pre-fetching Chatterbox weights (HF cache) …")
        ChatterboxTTS.from_pretrained(device="cpu")
        log.info("  Chatterbox weights ready")
    except Exception as exc:  # noqa: BLE001
        log.warning("Chatterbox warm-up skipped: %s", exc)


def _warm_snac() -> None:
    try:
        from snac import SNAC  # type: ignore

        log.info("pre-fetching SNAC codec weights (HF cache) …")
        SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        log.info("  SNAC weights ready")
    except Exception as exc:  # noqa: BLE001
        log.warning("SNAC warm-up skipped: %s", exc)


def _download_orpheus_gguf() -> None:
    orpheus = [m for m in enabled_models()
               if m.backend == "tts" and m.tts_engine == "orpheus" and m.hf_repo]
    if not orpheus:
        return
    try:
        from scripts import download_models  # type: ignore

        for m in orpheus:
            log.info("downloading Orpheus GGUF for %s …", m.id)
            download_models.download_one(m.id)
    except Exception as exc:  # noqa: BLE001
        log.warning("Orpheus GGUF download skipped: %s — verify hf_repo/hf_pattern in models.yaml", exc)


def _download_kokoro_onnx() -> None:
    kokoro = [m for m in enabled_models()
              if m.backend == "tts" and m.tts_engine == "kokoro" and m.model_path]
    if not kokoro:
        return
    for m in kokoro:
        target = (PROJECT_ROOT / str(m.model_path)).resolve()
        out_dir = target.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        for filename, url in _KOKORO_ASSETS.items():
            out = out_dir / filename
            try:
                _download_file(url, out, "Kokoro asset")
            except Exception as exc:  # noqa: BLE001
                log.warning("Kokoro asset download skipped: %s", exc)


def _download_file(url: str, out: Path, label: str) -> None:
    if out.exists() and out.stat().st_size > 0:
        log.info("%s already present: %s", label, out)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    log.info("downloading %s …", label)
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(out)
        log.info("  -> %s", out)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _download_piper_assets() -> None:
    piper = [m for m in enabled_models()
             if m.backend == "tts" and m.tts_engine == "piper" and m.model_path]
    if not piper:
        return
    if sys.platform != "win32":
        log.warning("Piper standalone installer is currently wired for Windows only")
        return
    vendor_dir = PROJECT_ROOT / "vendor"
    piper_exe = vendor_dir / "piper" / "piper.exe"
    try:
        if not piper_exe.exists():
            zip_path = PROJECT_ROOT / "models" / "piper" / "piper_windows_amd64.zip"
            _download_file(_PIPER_RELEASE_URL, zip_path, "Piper Windows binary")
            log.info("extracting Piper binary to %s", vendor_dir / "piper")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(vendor_dir)
        else:
            log.info("Piper binary already present: %s", piper_exe)
    except Exception as exc:  # noqa: BLE001
        log.warning("Piper binary download skipped: %s", exc)

    out_dir = PROJECT_ROOT / "models" / "piper"
    for filename, url in _PIPER_VOICES.items():
        try:
            _download_file(url, out_dir / filename, f"Piper voice {filename}")
        except Exception as exc:  # noqa: BLE001
            log.warning("Piper voice download skipped: %s", exc)


def _warm_kokoro() -> None:
    try:
        from kokoro_onnx import Kokoro  # type: ignore

        rows = [m for m in enabled_models()
                if m.backend == "tts" and m.tts_engine == "kokoro" and m.model_path]
        if not rows:
            return
        model_path = (PROJECT_ROOT / str(rows[0].model_path)).resolve()
        voices_path = model_path.parent / "voices-v1.0.bin"
        if not model_path.exists() or not voices_path.exists():
            log.warning("Kokoro warm-up skipped: missing %s or %s", model_path, voices_path)
            return
        log.info("warming Kokoro ONNX with English and Spanish voices …")
        model = Kokoro(str(model_path), str(voices_path))
        model.create("Arming the perimeter.", voice="am_michael", speed=1.0, lang="en-us")
        spanish_text = "Hola, esta es una prueba de voz en español."
        for voice in _KOKORO_SPANISH_VOICES:
            model.create(spanish_text, voice=voice, speed=1.0, lang="es")
        log.info("  Kokoro ready (Spanish: %s)", ", ".join(_KOKORO_SPANISH_VOICES))
    except Exception as exc:  # noqa: BLE001
        log.warning("Kokoro warm-up skipped: %s", exc)


def _warm_piper() -> None:
    rows = [m for m in enabled_models()
            if m.backend == "tts" and m.tts_engine == "piper" and m.model_path]
    if not rows:
        return
    try:
        from src.tts_engines import PiperEngine, SpeechRequest  # type: ignore

        log.info("warming Piper with amy …")
        eng = PiperEngine(rows[0], device="cpu")
        eng.load()
        eng.synthesize(SpeechRequest(text="Arming the perimeter.", voice="amy"))
        log.info("  Piper ready")
    except Exception as exc:  # noqa: BLE001
        log.warning("Piper warm-up skipped: %s", exc)


def _tts_engines_enabled() -> set[str]:
    return {m.tts_engine for m in enabled_models()
            if m.backend == "tts" and m.tts_engine}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    engines = _tts_engines_enabled()
    if not engines:
        log.info("no tts-server row enabled on this host — nothing to install")
        return 0

    _pip_install_requirements()
    _install_cuda_torch()
    _install_cuda_onnxruntime()
    if "chatterbox" in engines:
        _warm_chatterbox()
    if "orpheus" in engines:
        _warm_snac()
        _download_orpheus_gguf()
    if "piper" in engines:
        _download_piper_assets()
        _warm_piper()
    if "kokoro" in engines:
        _download_kokoro_onnx()
        _warm_kokoro()
    log.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
