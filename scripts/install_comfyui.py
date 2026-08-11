"""Install ComfyUI into vendor/comfyui/ as the hub's local image-generation engine.

Unlike ``install_llama_cpp.py`` / ``install_whisper_cpp.py`` — which unpack a
prebuilt release archive into ``vendor/`` — ComfyUI is a Python application, so
this script clones it at a pinned tag and gives it **its own virtualenv**
(``vendor/comfyui/.venv``).

That isolation is deliberate. ComfyUI pins a wide dependency surface
(transformers, numpy, Pillow, pydantic, …) that overlaps the hub's own; letting
``pip`` resolve them into the hub's ``.venv`` would let an image-engine bump
silently downgrade a package the routing core depends on. A separate venv costs
disk (a second torch) and buys a blast radius of zero.

CUDA: the wheels come from PyTorch's ``cu130`` index. The RTX 5060 Ti this repo
targets is Blackwell (**sm_120**), which needs a CUDA 12.8-or-newer build — the
default PyPI ``torch`` often resolves an older CUDA build with no sm_120 kernels
and falls back to CPU *silently*. :func:`verify_cuda` fails loudly on that
rather than leaving a 20x-slower install to be discovered at generation time.

Usage:
    python scripts/install_comfyui.py            # idempotent; no-op if installed
    python scripts/install_comfyui.py --force    # wipe vendor/comfyui and redo
    python scripts/install_comfyui.py --verify   # just re-run the CUDA check
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import InstallError, no_window_flags  # noqa: E402

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor" / "comfyui"
VENV_DIR = VENDOR_DIR / ".venv"
# Weights live in the repo's shared models/ tree (gitignored, same as the
# GGUFs), NOT under vendor/. ComfyUI is pointed at them by the generated
# extra_model_paths.yaml, so a --force reinstall never touches 17 GB of
# checkpoints.
MODELS_DIR = PROJECT_ROOT / "models" / "comfyui"

# Records the outcome of the last successful verify_cuda() so routine health
# checks don't have to re-pay it. Importing torch and initializing CUDA in a
# subprocess costs ~4.5 s — 28x the next-slowest install probe — and
# src/install.py's report is polled by the admin SPA, so doing it live made
# every status call 4.5 s slower.
MARKER_PATH = VENDOR_DIR / ".hub-install.json"

COMFYUI_GIT_URL = "https://github.com/comfyanonymous/ComfyUI"
# Pinned to a vetted release tag rather than floating `master`, same policy as
# whisper.cpp's PINNED_TAG — ComfyUI's node schemas and API surface move fast
# and src/comfyui_client.py's workflow graph is written against this one.
PINNED_TAG = "v0.31.0"

# Blackwell (sm_120) needs a CUDA >= 12.8 build; cu130 is the current line and
# the index the hub's own .venv already resolves against. No torch version is
# pinned here on purpose — what matters is the *index*, and verify_cuda()
# asserts the resulting build actually carries sm_120 rather than trusting a
# version number that would go stale in this comment.
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu130"
REQUIRED_ARCH = "sm_120"


def venv_python() -> Path:
    """The ComfyUI venv's interpreter — what ``backend_process`` spawns."""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def main_script() -> Path:
    return VENDOR_DIR / "main.py"


def _run(cmd: List[str], *, cwd: Optional[Path] = None, timeout: int) -> None:
    log.info("$ %s", " ".join(cmd))
    r = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True,
        timeout=timeout, creationflags=no_window_flags(),
    )
    if r.returncode != 0:
        tail = "\n".join(
            (r.stdout or "").splitlines()[-25:] + (r.stderr or "").splitlines()[-25:]
        )
        raise InstallError(f"step failed ({' '.join(cmd)}):\n{tail}")


def already_installed() -> bool:
    """True when the clone, the venv, and an importable torch are all present."""
    if not (main_script().exists() and venv_python().exists()):
        return False
    try:
        r = subprocess.run(
            [str(venv_python()), "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True, timeout=120,
            creationflags=no_window_flags(),
        )
        return r.returncode == 0
    except Exception:  # noqa: BLE001 — a broken venv is simply "not installed"
        return False


def verify_cuda() -> str:
    """Assert the venv's torch can actually drive this GPU.

    Raises :class:`InstallError` when CUDA is unavailable or the build lacks
    ``sm_120``. A silent CPU fallback would still *generate* images — roughly
    20x slower — so this has to fail loudly at install time rather than be
    diagnosed later from a slow request.
    """
    probe = (
        "import torch,json;"
        "print(json.dumps({'version': torch.__version__,"
        "'cuda': torch.version.cuda,"
        "'available': torch.cuda.is_available(),"
        "'archs': torch.cuda.get_arch_list() if torch.cuda.is_available() else [],"
        "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
    )
    r = subprocess.run(
        [str(venv_python()), "-c", probe],
        capture_output=True, text=True, timeout=180,
        creationflags=no_window_flags(),
    )
    if r.returncode != 0:
        raise InstallError(f"could not import torch in {venv_python()}:\n{r.stderr.strip()}")

    info = json.loads((r.stdout or "").strip().splitlines()[-1])
    if not info["available"]:
        raise InstallError(
            f"torch {info['version']} in {VENV_DIR} reports CUDA unavailable — "
            "ComfyUI would fall back to CPU and generate ~20x slower. Reinstall "
            f"with --force (wheels come from {TORCH_INDEX_URL})."
        )
    if REQUIRED_ARCH not in info["archs"]:
        raise InstallError(
            f"torch {info['version']} (CUDA {info['cuda']}) was built for "
            f"{info['archs']} — no {REQUIRED_ARCH}. This host's "
            f"{info['device']} is Blackwell and needs a CUDA >= 12.8 build. "
            f"Reinstall with --force so wheels come from {TORCH_INDEX_URL}."
        )
    log.info(
        "CUDA OK: torch %s (CUDA %s) on %s, arch list includes %s",
        info["version"], info["cuda"], info["device"], REQUIRED_ARCH,
    )
    write_marker(info)
    return info["version"]


def write_marker(info: Dict[str, Any]) -> None:
    """Persist a passing :func:`verify_cuda` result for cheap health checks."""
    try:
        MARKER_PATH.write_text(json.dumps({
            "comfyui_tag": PINNED_TAG,
            "torch_version": info["version"],
            "cuda": info["cuda"],
            "device": info["device"],
        }, indent=2), encoding="utf-8")
    except OSError as exc:  # noqa: BLE001 — best-effort; never fails an install
        log.warning("could not write %s: %s", MARKER_PATH, exc)


def read_marker() -> Optional[Dict[str, Any]]:
    """The last verified install's details, or ``None`` if never verified.

    ``None`` means *unknown*, not *broken* — the caller must report it as its
    own state rather than folding it into either pass or fail.
    """
    try:
        data = json.loads(MARKER_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def write_extra_model_paths() -> Path:
    """Point ComfyUI at the repo's ``models/comfyui/`` tree.

    Keeps weights beside the GGUFs under the already-gitignored ``models/``
    rather than inside ``vendor/``, so ``--force`` can delete the engine
    without re-downloading 17 GB of checkpoints.
    """
    path = VENDOR_DIR / "extra_model_paths.yaml"
    base = MODELS_DIR.as_posix()
    path.write_text(
        "# Generated by scripts/install_comfyui.py — do not edit by hand.\n"
        "# Keeps model weights in the repo's shared models/ tree so a --force\n"
        "# reinstall of the engine never touches the checkpoints.\n"
        "local_llm_hub:\n"
        f"  base_path: {base}\n"
        "  checkpoints: checkpoints\n"
        "  vae: vae\n"
        "  loras: loras\n"
        "  text_encoders: text_encoders\n"
        "  diffusion_models: diffusion_models\n",
        encoding="utf-8",
    )
    log.info("wrote %s -> %s", path, base)
    return path


def _purge_vendor() -> None:
    if not VENDOR_DIR.exists():
        return
    log.info("--force: removing existing %s", VENDOR_DIR)
    try:
        shutil.rmtree(VENDOR_DIR)
    except (PermissionError, OSError) as exc:
        raise InstallError(
            f"could not remove {VENDOR_DIR} ({exc}). ComfyUI is likely still "
            "running and holding files in its venv. Stop it first (the admin "
            "Models tab, or `tray.bat --restart`), then re-run with --force."
        )


def install() -> None:
    if shutil.which("git") is None:
        raise InstallError("git not found on PATH — required to clone ComfyUI")

    VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
    log.info("cloning %s @ %s ...", COMFYUI_GIT_URL, PINNED_TAG)
    _run(
        ["git", "clone", "--branch", PINNED_TAG, "--depth", "1",
         COMFYUI_GIT_URL, str(VENDOR_DIR)],
        timeout=900,
    )

    log.info("creating venv at %s ...", VENV_DIR)
    _run([sys.executable, "-m", "venv", str(VENV_DIR)], timeout=300)

    pip = [str(venv_python()), "-m", "pip"]
    _run(pip + ["install", "--upgrade", "pip", "wheel"], timeout=600)

    # torch first, from the CUDA-specific index, so the requirements.txt pass
    # below finds it already satisfied and never pulls a default-index build
    # with no sm_120 kernels.
    log.info("installing torch from %s (several GB — this takes a while) ...", TORCH_INDEX_URL)
    _run(
        pip + ["install", "torch", "torchvision", "torchaudio",
               "--index-url", TORCH_INDEX_URL],
        timeout=5400,
    )

    log.info("installing ComfyUI requirements ...")
    _run(pip + ["install", "-r", str(VENDOR_DIR / "requirements.txt")], timeout=3600)

    for sub in ("checkpoints", "vae", "loras", "text_encoders", "diffusion_models"):
        (MODELS_DIR / sub).mkdir(parents=True, exist_ok=True)
    write_extra_model_paths()


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="wipe vendor/comfyui and reinstall (keeps models/)")
    p.add_argument("--verify", action="store_true",
                   help="only re-run the CUDA capability check")
    args = p.parse_args(argv)

    if args.verify:
        if not already_installed():
            raise InstallError(f"ComfyUI is not installed at {VENDOR_DIR}")
        verify_cuda()
        return 0

    if args.force:
        _purge_vendor()
    elif already_installed():
        log.info("ComfyUI already installed at %s", VENDOR_DIR)
        write_extra_model_paths()  # cheap; keeps the path file in sync
        verify_cuda()
        return 0

    install()

    if not already_installed():
        raise InstallError(f"install finished but {venv_python()} is not usable")
    verify_cuda()
    log.info("installed: %s (ComfyUI %s)", VENDOR_DIR, PINNED_TAG)
    log.info("weights go in %s", MODELS_DIR / "checkpoints")
    return 0


if __name__ == "__main__":
    sys.exit(main())
