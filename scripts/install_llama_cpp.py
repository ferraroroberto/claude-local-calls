"""Download and extract a prebuilt llama.cpp release for the current platform.

Picks a release asset that matches this machine:
  - Windows x64 + NVIDIA  -> llama-<tag>-bin-win-cuda-13.1-x64.zip
                             (plus cudart-llama-bin-win-cuda-13.1-x64.zip
                              for the CUDA runtime DLLs)
  - macOS arm64           -> llama-<tag>-bin-macos-arm64.tar.gz

Extracts into vendor/llama.cpp/ at the project root. Idempotent: if
llama-server[.exe] --version already works, it exits fast.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor" / "llama.cpp"
RELEASES_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

# Prefer CUDA 13.1 on Windows (matches current driver/toolkit line; Blackwell
# requires CUDA >=12.8 so the older 12.4 build is the fallback only).
WIN_CUDA_PREFS = ["cuda-13.1", "cuda-12.4"]


class InstallError(RuntimeError):
    pass


def _server_binary() -> Path:
    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    return VENDOR_DIR / name


def already_installed() -> bool:
    bin_path = _server_binary()
    if not bin_path.exists():
        return False
    try:
        r = subprocess.run([str(bin_path), "--version"],
                           capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


def _fetch_release() -> dict:
    print(f"querying {RELEASES_URL} ...")
    req = urllib.request.Request(RELEASES_URL, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _pick_assets(release: dict) -> List[dict]:
    assets = release.get("assets") or []
    names = [a["name"] for a in assets]

    def find(predicate) -> Optional[dict]:
        for a in assets:
            if predicate(a["name"].lower()):
                return a
        return None

    if sys.platform == "win32":
        picks: List[dict] = []
        for cuda in WIN_CUDA_PREFS:
            main = find(lambda n, c=cuda: n.startswith("llama-") and c in n and "win" in n and "x64" in n and n.endswith(".zip"))
            if main:
                picks.append(main)
                rt = find(lambda n, c=cuda: n.startswith("cudart-") and c in n and n.endswith(".zip"))
                if rt:
                    picks.append(rt)
                break
        if not picks:
            raise InstallError(
                f"no matching CUDA Windows asset in release {release.get('tag_name')}. "
                f"assets available: {names}"
            )
        return picks

    if sys.platform == "darwin":
        if platform.machine() != "arm64":
            raise InstallError(
                f"only darwin arm64 is supported; this is {platform.machine()}"
            )
        pick = find(lambda n: n.startswith("llama-") and "macos-arm64" in n and n.endswith((".zip", ".tar.gz")) and "kleidiai" not in n)
        if not pick:
            raise InstallError(
                f"no macOS arm64 asset in release {release.get('tag_name')}. assets: {names}"
            )
        return [pick]

    raise InstallError(f"unsupported platform: {sys.platform}")


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    print(f"       -> {dest}")
    with urllib.request.urlopen(url, timeout=120) as r:
        total = int(r.headers.get("Content-Length", 0))
        seen = 0
        next_report = 0
        with dest.open("wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                seen += len(chunk)
                if total and seen >= next_report:
                    pct = 100 * seen / total
                    print(f"  {seen/1_048_576:6.1f} / {total/1_048_576:6.1f} MB ({pct:5.1f}%)")
                    next_report = seen + total // 20
    print(f"  done: {seen/1_048_576:.1f} MB")


def _extract(archive: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"extracting {archive.name} -> {dest_dir}")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
    elif archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest_dir)
    else:
        raise InstallError(f"unknown archive type: {archive}")


def _flatten_if_nested(target: Path) -> None:
    """Some releases extract into a single subdir; collapse it into target."""
    entries = [p for p in target.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for child in inner.iterdir():
            shutil.move(str(child), str(target / child.name))
        inner.rmdir()


def main() -> int:
    if already_installed():
        print(f"llama.cpp already installed at {_server_binary()}")
        return 0

    release = _fetch_release()
    tag = release.get("tag_name", "?")
    assets = _pick_assets(release)
    print(f"release {tag}: picking {len(assets)} asset(s)")

    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    for a in assets:
        archive = VENDOR_DIR / a["name"]
        if not archive.exists():
            _download(a["browser_download_url"], archive)
        _extract(archive, VENDOR_DIR)
        archive.unlink(missing_ok=True)

    _flatten_if_nested(VENDOR_DIR)

    bin_path = _server_binary()
    if not bin_path.exists():
        # Some zips extract into a `build/bin/` or `bin/` subdirectory.
        for candidate in VENDOR_DIR.rglob(bin_path.name):
            # Move the entire bin directory up next to llama-server.exe.
            src_dir = candidate.parent
            if src_dir == VENDOR_DIR:
                break
            print(f"flattening {src_dir} -> {VENDOR_DIR}")
            for child in list(src_dir.iterdir()):
                target = VENDOR_DIR / child.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(child), str(target))
            break

    if not already_installed():
        raise InstallError(
            f"extracted archives but {_server_binary()} still missing or non-runnable"
        )

    print(f"installed: {_server_binary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
