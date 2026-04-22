"""Download and extract a prebuilt whisper.cpp release for the current platform.

Picks a release asset from ggerganov/whisper.cpp that matches this machine:
  - Windows x64 + NVIDIA  -> whisper-cublas-<cuda>-bin-x64.zip
  - macOS arm64           -> whisper-bin-arm64.zip   (Metal build)

Extracts into vendor/whisper.cpp/ at the project root, then renames the
primary binary to `whisper-server[.exe]` (upstream ships it as `server[.exe]`).
Idempotent: if the binary already runs, exits fast.
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
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor" / "whisper.cpp"
RELEASES_URL = "https://api.github.com/repos/ggerganov/whisper.cpp/releases/latest"

# Prefer the newest CUDA line upstream ships; fall back to older ones.
WIN_CUDA_PREFS = ["cublas-12.4.0", "cublas-12.2.0", "cublas-11.8.0"]


class InstallError(RuntimeError):
    pass


def _server_binary() -> Path:
    name = "whisper-server.exe" if sys.platform == "win32" else "whisper-server"
    return VENDOR_DIR / name


def _upstream_server_names() -> List[str]:
    # Upstream has shipped this binary under a couple of names over time.
    if sys.platform == "win32":
        return ["whisper-server.exe", "server.exe"]
    return ["whisper-server", "server"]


def already_installed() -> bool:
    bin_path = _server_binary()
    if not bin_path.exists():
        return False
    try:
        # whisper-server prints usage on --help and exits non-zero, so just
        # check that the binary can execute at all.
        r = subprocess.run([str(bin_path), "--help"],
                           capture_output=True, text=True, timeout=10)
        return r.returncode in (0, 1)
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
        for cuda in WIN_CUDA_PREFS:
            pick = find(lambda n, c=cuda: n.startswith("whisper-") and c in n and "x64" in n and n.endswith(".zip"))
            if pick:
                return [pick]
        raise InstallError(
            f"no CUDA Windows asset in release {release.get('tag_name')}. "
            f"assets available: {names}"
        )

    if sys.platform == "darwin":
        if platform.machine() != "arm64":
            raise InstallError(
                f"only darwin arm64 is supported; this is {platform.machine()}"
            )
        pick = find(lambda n: n.startswith("whisper-") and "arm64" in n and n.endswith((".zip", ".tar.gz")))
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


def _normalise_binary_name() -> None:
    """Upstream names the server binary `server[.exe]`; the manager expects
    `whisper-server[.exe]`. Rename it if we find the upstream name."""
    want = _server_binary()
    if want.exists():
        return
    for candidate_name in _upstream_server_names():
        for candidate in VENDOR_DIR.rglob(candidate_name):
            # Lift the entire bin directory up alongside the expected path,
            # so sibling DLLs (cudart, whisper.dll, ggml.dll, ...) travel with it.
            src_dir = candidate.parent
            if src_dir != VENDOR_DIR:
                print(f"flattening {src_dir} -> {VENDOR_DIR}")
                for child in list(src_dir.iterdir()):
                    target = VENDOR_DIR / child.name
                    if target.exists():
                        if target.is_dir():
                            shutil.rmtree(target)
                        else:
                            target.unlink()
                    shutil.move(str(child), str(target))
            src = VENDOR_DIR / candidate_name
            if src.exists() and src != want:
                print(f"renaming {src.name} -> {want.name}")
                src.rename(want)
            return


def main() -> int:
    if already_installed():
        print(f"whisper.cpp already installed at {_server_binary()}")
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
    _normalise_binary_name()

    if not already_installed():
        raise InstallError(
            f"extracted archives but {_server_binary()} still missing or non-runnable"
        )

    print(f"installed: {_server_binary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
