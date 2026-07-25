"""Download and extract a prebuilt whisper.cpp release for the current platform.

Picks a release asset from ggerganov/whisper.cpp that matches this machine:
  - Windows x64 + NVIDIA  -> whisper-cublas-<cuda>-bin-x64.zip
  - macOS arm64           -> built from source (Metal) — upstream ships no
    prebuilt macOS asset as of the pinned tag (#413); see
    ``_macos_build_from_source``.

Extracts into vendor/whisper.cpp/ at the project root, then renames the
primary binary to `whisper-server[.exe]` (upstream ships it as `server[.exe]`).
Idempotent: if the binary already runs, exits fast.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    InstallError,
    detect_cuda_arch,
    download,
    extract,
    flatten_if_nested,
    no_window_flags,
)

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor" / "whisper.cpp"

# Pinned to a known-good tag rather than floating "latest" so the feature
# set is deterministic (issue #91). v1.8.5 (2026-05-29, ggml-org/whisper.cpp
# #3781) added server-side `carry_initial_prompt`; v1.8.6 is the newest
# patch on that line. Bump this tag deliberately when a newer one is
# vetted. The repo moved ggerganov → ggml-org; the API redirects either way.
PINNED_TAG = "v1.8.6"
RELEASES_URL = (
    f"https://api.github.com/repos/ggml-org/whisper.cpp/releases/tags/{PINNED_TAG}"
)
WHISPER_CPP_GIT_URL = "https://github.com/ggml-org/whisper.cpp"

# Prefer the newest CUDA line upstream ships; fall back to older ones.
WIN_CUDA_PREFS = ["cublas-12.4.0", "cublas-12.2.0", "cublas-11.8.0"]


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
    # The first exec immediately after a --force extract can transiently
    # fail (Errno-ish / AV scan) while the OS finishes flushing the large
    # CUDA DLLs (cublasLt64_12.dll is ~450 MB) to disk. Retry a couple of
    # times before declaring the binary non-runnable.
    for attempt in range(3):
        try:
            # whisper-server prints usage on --help and exits non-zero, so just
            # check that the binary can execute at all.
            r = subprocess.run([str(bin_path), "--help"],
                               capture_output=True, text=True, timeout=10,
                               creationflags=no_window_flags())
            if r.returncode in (0, 1):
                return True
        except Exception:
            pass
        if attempt < 2:
            time.sleep(1.5)
    return False


def _linux_cuda_build_hint() -> str:
    """A reproducible from-source CUDA build recipe for a Linux satellite.

    Upstream ships no prebuilt Linux CUDA whisper-server; gaming's sm_61 build
    was compiled by hand (#368). Rather than an untested automated compile,
    surface the exact commands with the arch defaulted to this host's detected
    GPU (override via ``LOCAL_LLM_HUB_CUDA_ARCH``). Pinned tag matches the
    Windows/macOS vendored line. The automated build itself is a follow-up.
    """
    arch = os.environ.get("LOCAL_LLM_HUB_CUDA_ARCH") or detect_cuda_arch() or "61"
    return (
        "no prebuilt whisper.cpp asset for Linux — build from source with CUDA.\n"
        f"target GPU arch: sm_{arch} (override via LOCAL_LLM_HUB_CUDA_ARCH). "
        "Reproducible build (run on the satellite; not yet automated — #368):\n"
        f"  git clone --branch {PINNED_TAG} --depth 1 "
        "https://github.com/ggml-org/whisper.cpp /tmp/whisper.cpp\n"
        "  cmake -S /tmp/whisper.cpp -B /tmp/whisper.cpp/build "
        f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={arch} -DWHISPER_BUILD_SERVER=ON\n"
        "  cmake --build /tmp/whisper.cpp/build --config Release -j --target whisper-server\n"
        f"  cp /tmp/whisper.cpp/build/bin/whisper-server {VENDOR_DIR}/"
    )


def _macos_find_cmake() -> Optional[str]:
    """cmake's location on macOS, tolerant of a restricted non-interactive
    PATH (``/usr/bin:/bin:/usr/sbin:/sbin``) that a launchd/SSH-triggered
    "Fix" run gets even when an interactive shell's PATH has it via
    Homebrew (#413 — confirmed on mac-mini-m4: ``cmake`` is on
    ``/opt/homebrew/bin``, off that restricted PATH)."""
    found = shutil.which("cmake")
    if found:
        return found
    for candidate in ("/opt/homebrew/bin/cmake", "/usr/local/bin/cmake"):
        if Path(candidate).exists():
            return candidate
    return None


def _macos_build_prereqs() -> str:
    """Returns the cmake path if every from-source build prerequisite is
    present; raises a clear, actionable InstallError otherwise (#413 scope:
    "fail with a clear message when a build prerequisite is missing rather
    than an asset-list dump")."""
    cmake_path = _macos_find_cmake()
    if not cmake_path:
        raise InstallError(
            "no macOS arm64 whisper.cpp release asset, and cmake is missing "
            "so it cannot be built from source. Install it with "
            "`brew install cmake`, then re-run."
        )
    try:
        r = subprocess.run(["xcode-select", "-p"], capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        r = None
    if r is None or r.returncode != 0:
        raise InstallError(
            "no macOS arm64 whisper.cpp release asset, and the Xcode Command "
            "Line Tools are missing so it cannot be built from source. "
            "Install them with `xcode-select --install`, then re-run."
        )
    return cmake_path


def _macos_build_from_source() -> None:
    """Build whisper-server from source for this pinned tag (#413) — no
    prebuilt macOS asset exists as of ``PINNED_TAG``. Mirrors the recipe
    hand-verified on mac-mini-m4: cmake configure (Metal + Accelerate BLAS
    are auto-detected on Apple platforms), build the ``whisper-server``
    target, flatten the binary plus every produced ``.dylib`` into
    ``VENDOR_DIR``, then ``install_name_tool -add_rpath @loader_path`` each
    so the flattened tree is self-contained after the source tree is
    deleted.
    """
    cmake_path = _macos_build_prereqs()
    tmp_root = Path(tempfile.mkdtemp(prefix="whisper-cpp-build-"))
    try:
        src_dir = tmp_root / "whisper.cpp"
        build_dir = src_dir / "build"

        log.info("cloning %s @ %s ...", WHISPER_CPP_GIT_URL, PINNED_TAG)
        _run_build_step(
            ["git", "clone", "--branch", PINNED_TAG, "--depth", "1", WHISPER_CPP_GIT_URL, str(src_dir)],
            timeout=300,
        )

        log.info("configuring (cmake) ...")
        _run_build_step(
            [cmake_path, "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release", "-DWHISPER_BUILD_SERVER=ON"],
            cwd=src_dir, timeout=300,
        )

        log.info("building whisper-server (this can take a few minutes) ...")
        _run_build_step(
            [cmake_path, "--build", str(build_dir), "--config", "Release", "-j", "--target", "whisper-server"],
            cwd=src_dir, timeout=1800,
        )

        VENDOR_DIR.mkdir(parents=True, exist_ok=True)
        built_binary = build_dir / "bin" / "whisper-server"
        if not built_binary.exists():
            raise InstallError(f"build reported success but {built_binary} is missing")
        shutil.copy2(built_binary, _server_binary())

        dylibs = sorted(build_dir.rglob("*.dylib"))
        log.info("flattening %d dylib(s) into %s", len(dylibs), VENDOR_DIR)
        for dylib in dylibs:
            shutil.copy2(dylib, VENDOR_DIR / dylib.name, follow_symlinks=False)

        # Real (non-symlink) files only — a symlink shares its target's
        # inode, so re-running install_name_tool on it errors with
        # "would duplicate path" once the target has already been fixed.
        flattened = [_server_binary()] + [VENDOR_DIR / d.name for d in dylibs]
        for target in flattened:
            if target.is_symlink():
                continue
            r = subprocess.run(
                ["install_name_tool", "-add_rpath", "@loader_path", str(target)],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                raise InstallError(f"install_name_tool -add_rpath failed for {target}: {r.stderr.strip()}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _run_build_step(cmd: List[str], *, cwd: Optional[Path] = None, timeout: int) -> None:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        tail = "\n".join(r.stdout.splitlines()[-20:] + r.stderr.splitlines()[-20:])
        raise InstallError(f"build step failed: {' '.join(cmd)}\n{tail}")


def _fetch_release() -> dict:
    log.info("querying %s ...", RELEASES_URL)
    req = urllib.request.Request(RELEASES_URL, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _purge_vendor() -> None:
    """Remove the existing vendored tree so a forced reinstall lands clean.

    Used by ``--force`` to upgrade the pinned binary. On Windows the
    server .exe / DLLs are locked while whisper-server is running, so
    rmtree raises — surface that as a clear "stop the server first"
    message rather than a bare OSError.
    """
    if not VENDOR_DIR.exists():
        return
    log.info("--force: removing existing %s", VENDOR_DIR)
    try:
        shutil.rmtree(VENDOR_DIR)
    except (PermissionError, OSError) as exc:
        raise InstallError(
            f"could not remove {VENDOR_DIR} ({exc}). whisper-server is "
            "likely still running and holding the binary. Stop it first "
            "(coordinate with voice-transcriber on the shared :8090/:8091 "
            "mutex), then re-run with --force."
        )


def _pick_assets(release: dict) -> List[dict]:
    """Windows only — the darwin platform has no reliable prebuilt asset as
    of ``PINNED_TAG`` (#413) and is handled separately by ``_find_macos_asset``
    plus a from-source fallback."""
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

    raise InstallError(f"unsupported platform: {sys.platform}")


def _find_macos_asset(release: dict) -> Optional[dict]:
    """A prebuilt macOS arm64 asset, if the pinned tag happens to publish one
    (upstream did not as of v1.8.6 — #413). Checked first so a future
    ``PINNED_TAG`` bump onto a release that does ship one uses it instead of
    building from source."""
    assets = release.get("assets") or []
    for a in assets:
        n = a["name"].lower()
        if n.startswith("whisper-") and "arm64" in n and n.endswith((".zip", ".tar.gz")):
            return a
    return None


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
                log.info("flattening %s -> %s", src_dir, VENDOR_DIR)
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
                log.info("renaming %s -> %s", src.name, want.name)
                src.rename(want)
            return


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = list(sys.argv[1:] if argv is None else argv)
    force = "--force" in args

    if force:
        _purge_vendor()
    elif already_installed():
        log.info("whisper.cpp already installed at %s", _server_binary())
        return 0

    if sys.platform.startswith("linux"):
        # A hand-built binary already present is caught by already_installed()
        # above; reaching here means it's missing (or --force purged it) and
        # must be compiled from source.
        raise InstallError(_linux_cuda_build_hint())

    if sys.platform == "darwin":
        if platform.machine() != "arm64":
            raise InstallError(
                f"only darwin arm64 is supported; this is {platform.machine()}"
            )
        release = _fetch_release()
        tag = release.get("tag_name", "?")
        asset = _find_macos_asset(release)
        VENDOR_DIR.mkdir(parents=True, exist_ok=True)
        if asset:
            log.info("release %s: macOS arm64 asset found", tag)
            archive = VENDOR_DIR / asset["name"]
            if not archive.exists():
                download(asset["browser_download_url"], archive)
            extract(archive, VENDOR_DIR)
            archive.unlink(missing_ok=True)
            flatten_if_nested(VENDOR_DIR)
            _normalise_binary_name()
        else:
            log.info(
                "release %s: no macOS arm64 asset — building whisper-server "
                "from source (Metal)", tag,
            )
            _macos_build_from_source()

        if not already_installed():
            raise InstallError(
                f"installed but {_server_binary()} still missing or non-runnable"
            )
        log.info("installed: %s", _server_binary())
        return 0

    release = _fetch_release()
    tag = release.get("tag_name", "?")
    assets = _pick_assets(release)
    log.info("release %s: picking %d asset(s)", tag, len(assets))

    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    for a in assets:
        archive = VENDOR_DIR / a["name"]
        if not archive.exists():
            download(a["browser_download_url"], archive)
        extract(archive, VENDOR_DIR)
        archive.unlink(missing_ok=True)

    flatten_if_nested(VENDOR_DIR)
    _normalise_binary_name()

    if not already_installed():
        raise InstallError(
            f"extracted archives but {_server_binary()} still missing or non-runnable"
        )

    log.info("installed: %s", _server_binary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
