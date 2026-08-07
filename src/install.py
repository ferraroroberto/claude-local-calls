"""First-run and health checks shared by the CLI and the admin SPA.

Each check inspects state and returns a `Check` row with a status of
`ok` | `missing` | `warn` | `error`. Fix functions are separate and do
the actual installing/downloading when the user opts in.

Usage:
    python -m src.install             # print a table, exit 1 if any error/missing
    python -m src.install --fix       # run fix_fn for every fixable row
    python -m src.install --json      # machine-readable output
"""

from __future__ import annotations

import argparse
import getpass
import importlib
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

log = logging.getLogger(__name__)

from .backend_process import llama_server_binary, whisper_server_binary
from .host_profile import hub_port, resolve as resolve_host
from .model_registry import Model, local_models
from .no_window import NO_WINDOW

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STATUS_ORDER = {"ok": 0, "warn": 1, "missing": 2, "error": 3}


@dataclass
class Check:
    id: str
    label: str
    status: str = "ok"        # ok | warn | missing | error
    detail: str = ""
    fix_id: Optional[str] = None
    fix_label: Optional[str] = None


@dataclass
class Report:
    checks: List[Check] = field(default_factory=list)

    @property
    def worst_status(self) -> str:
        return max((c.status for c in self.checks), key=lambda s: STATUS_ORDER.get(s, 0), default="ok")

    @property
    def ok(self) -> bool:
        return self.worst_status in ("ok", "warn")


# ---------- individual checks ----------

def _check_python_venv() -> Check:
    ver = sys.version_info
    if ver < (3, 10):
        return Check("python", "Python >= 3.10", "error",
                     f"found {ver.major}.{ver.minor}.{ver.micro}")
    try:
        in_project_venv = Path(sys.prefix).resolve().is_relative_to((PROJECT_ROOT / ".venv").resolve())
    except (AttributeError, ValueError):
        in_project_venv = str(Path(sys.prefix).resolve()).startswith(str((PROJECT_ROOT / ".venv").resolve()))
    if not in_project_venv:
        return Check("python", "Python >= 3.10, running from .venv", "warn",
                     f"python={sys.version.split()[0]} prefix={sys.prefix} (not the project .venv)")
    return Check("python", "Python >= 3.10, running from .venv", "ok",
                 f"python={sys.version.split()[0]} prefix={sys.prefix}")


_REQUIRED_DEPS = [
    "fastapi", "uvicorn", "httpx", "anthropic",
    "yaml", "huggingface_hub", "pydantic", "python_multipart",
]


def _check_deps() -> Check:
    missing = []
    for mod in _REQUIRED_DEPS:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        return Check("deps", "Python deps installed", "missing",
                     f"missing: {', '.join(missing)}",
                     fix_id="deps", fix_label="pip install -r requirements.txt")
    return Check("deps", "Python deps installed", "ok",
                 f"{len(_REQUIRED_DEPS)} packages OK")


def _check_host_profile() -> Check:
    try:
        h = resolve_host()
    except Exception as e:
        return Check("host", "Host profile resolves", "error", str(e))
    return Check("host", "Host profile resolves", "ok",
                 f"host={h.id} ({h.source}); enabled local models: {h.enabled or '(none)'}")


def _probe_cli_version(
    *,
    check_id: str,
    label: str,
    exe: Optional[str],
    version_args: List[str],
    not_found_detail: str,
    not_found_status: str = "warn",
    fix_id: Optional[str] = None,
    fix_label: Optional[str] = None,
    ok_codes: tuple = (0,),
) -> Check:
    """Shared shape for the four "is this executable here and runnable?" checks.

    The caller resolves ``exe`` to a runnable path (``shutil.which(...)`` for
    PATH tools, or ``str(bin_path)`` when the vendored binary exists, else
    ``None``). We then run ``exe version_args`` with a short timeout and
    classify the exit code:

      * ``exe is None``      → ``not_found_status`` (warn for optional CLIs,
                               missing + fix for installable binaries).
      * returncode in ``ok_codes`` → ``ok``, detail = first output line.
      * any other returncode → ``warn`` ("<args> exited N").
      * subprocess raised    → ``warn`` (the exception text).
    """
    if not exe:
        return Check(check_id, label, not_found_status, not_found_detail,
                     fix_id=fix_id, fix_label=fix_label)
    try:
        r = subprocess.run([str(exe), *version_args],
                           capture_output=True, text=True, timeout=10,
                           creationflags=NO_WINDOW)
        if r.returncode in ok_codes:
            first = ((r.stdout or r.stderr).strip().splitlines() or ["ok"])[0]
            return Check(check_id, label, "ok", first)
        return Check(check_id, label, "warn",
                     f"{' '.join(version_args)} exited {r.returncode}")
    except Exception as e:  # noqa: BLE001
        return Check(check_id, label, "warn", str(e))


def _check_claude_cli() -> Check:
    return _probe_cli_version(
        check_id="claude_cli",
        label="`claude` CLI on PATH",
        exe=shutil.which("claude"),
        version_args=["--version"],
        not_found_detail="not found — the Claude backend won't work until Claude Code is installed",
    )


def _check_gemini_cli() -> Check:
    # The Gemini backend drives the Antigravity CLI (`agy`); Google's
    # standalone `gemini` CLI stops serving AI Pro subscribers 2026-06-18.
    return _probe_cli_version(
        check_id="gemini_cli",
        label="`agy` (Antigravity CLI) on PATH",
        exe=shutil.which("agy"),
        version_args=["--version"],
        not_found_detail=(
            "not found — install the Antigravity CLI from https://antigravity.google "
            "and sign in once to use the Gemini backend (Google AI Pro)"
        ),
    )


def _nvidia_smi_gpu_check() -> Check:
    """nvidia-smi name+VRAM probe — shared by the Windows and Linux branches
    (the tower and the CUDA satellites both read their GPU the same way)."""
    nv = shutil.which("nvidia-smi")
    if not nv:
        return Check("gpu", "GPU / accelerator detected", "warn", "nvidia-smi not found")
    try:
        r = subprocess.run(
            [nv, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
            creationflags=NO_WINDOW,
        )
        if r.returncode == 0 and r.stdout.strip():
            return Check("gpu", "GPU / accelerator detected", "ok", r.stdout.strip().splitlines()[0])
        return Check("gpu", "GPU / accelerator detected", "warn",
                     f"nvidia-smi exit {r.returncode}: {r.stderr[:200]}")
    except Exception as e:
        return Check("gpu", "GPU / accelerator detected", "warn", str(e))


def _check_gpu() -> Check:
    if sys.platform == "win32":
        return _nvidia_smi_gpu_check()
    if sys.platform == "darwin":
        mach = platform.machine()
        if mach == "arm64":
            return Check("gpu", "Apple Silicon GPU (Metal)", "ok", f"arch={mach}")
        return Check("gpu", "GPU / accelerator detected", "warn",
                     f"darwin but arch={mach} — MLX / Metal expect arm64")
    if sys.platform.startswith("linux"):
        # The CUDA satellites (gaming, later openclaw) read their NVIDIA GPU
        # exactly like the tower does.
        return _nvidia_smi_gpu_check()
    return Check("gpu", "GPU / accelerator detected", "warn",
                 f"unknown platform {sys.platform}")


def _check_llama_cpp() -> Check:
    bin_path = llama_server_binary()
    return _probe_cli_version(
        check_id="llama_cpp",
        label="llama.cpp binary installed",
        exe=str(bin_path) if bin_path.exists() else None,
        version_args=["--version"],
        not_found_detail=f"expected at {bin_path}",
        not_found_status="missing",
        fix_id="llama_cpp",
        fix_label="scripts/install_llama_cpp.py (downloads the platform-matching release)",
    )


def _check_models() -> List[Check]:
    rows: List[Check] = []
    for m in local_models():
        if m.backend not in ("openai", "whisper", "tts") or not m.model_path:
            continue
        path = (PROJECT_ROOT / m.model_path).resolve()
        label = f"Model present: {m.display_name}"
        if path.exists() and path.is_file():
            size_gb = path.stat().st_size / (1024 ** 3)
            rows.append(Check(f"model_{m.id}", label, "ok", f"{path.name} ({size_gb:.1f} GB)"))
        else:
            rows.append(Check(
                f"model_{m.id}", label, "missing",
                f"expected at {path}",
                fix_id=f"download_{m.id}",
                fix_label=f"scripts/download_models.py --only {m.id}",
            ))
    return rows


def _whisper_enabled() -> bool:
    # Engine-specific, not just `backend == "whisper"` — a whisper-*shaped*
    # backend (e.g. Parakeet's `engine: parakeet-server`, #138) doesn't need
    # the whisper.cpp binary this check is actually gating.
    return any(m.engine in ("whisper-server", "whisper-server-lazy") for m in local_models())


def _check_whisper_cpp() -> Check:
    bin_path = whisper_server_binary()
    # whisper-server prints usage on --help and may exit non-zero (0 or 1).
    return _probe_cli_version(
        check_id="whisper_cpp",
        label="whisper.cpp binary installed",
        exe=str(bin_path) if bin_path.exists() else None,
        version_args=["--help"],
        not_found_detail=f"expected at {bin_path}",
        not_found_status="missing",
        fix_id="whisper_cpp",
        fix_label="scripts/install_whisper_cpp.py (downloads the platform-matching release)",
        ok_codes=(0, 1),
    )


def _parakeet_enabled() -> bool:
    return any(m.engine == "parakeet-server" for m in local_models())


def _check_parakeet_worker() -> Check:
    """Is the FluidAudio Swift binary (mac/parakeet-worker) built? (#138)

    darwin-only in practice — gated on a `engine: parakeet-server` row
    being locally enabled, which only ever happens on mac-mini-m4. No
    CLI-version probe (`_probe_cli_version`): the worker isn't a real CLI,
    it blocks on stdin waiting for a request — existence is all we can
    check without actually spinning up the CoreML model.
    """
    bin_path = PROJECT_ROOT / "mac" / "parakeet-worker" / ".build" / "release" / "ParakeetWorker"
    if bin_path.exists():
        return Check("parakeet_worker", "ParakeetWorker binary built", "ok", str(bin_path))
    return Check(
        "parakeet_worker", "ParakeetWorker binary built", "missing",
        f"expected at {bin_path}",
        fix_id="parakeet_worker",
        fix_label="swift build -c release in mac/parakeet-worker/ (downloads FluidAudio + builds, first run also fetches the CoreML model on first request)",
    )


LAUNCHAGENT_LABEL = "com.ferraroroberto.local-llm-hub"


def _launchagent_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHAGENT_LABEL}.plist"


def _check_launchagent() -> Check:
    """Is the boot-time LaunchAgent installed and loaded? (#181, darwin-only)

    Boot autostart + crash respawn for the Mac Mini's otherwise-unsupervised
    hub process — the macOS analogue of the Windows tray's spawn-on-launch.
    Existence + ``launchctl print`` are all we check; installing registers
    and loads the job (``RunAtLoad`` fires it immediately if not already
    running).
    """
    plist_path = _launchagent_plist_path()
    if not plist_path.exists():
        return Check(
            "launchagent", "Boot-time LaunchAgent installed", "missing",
            f"expected at {plist_path}",
            fix_id="launchagent",
            fix_label="writes the plist + `launchctl bootstrap` (boot autostart + crash respawn)",
        )
    loaded = subprocess.run(
        ["launchctl", "print", f"gui/{os.getuid()}/{LAUNCHAGENT_LABEL}"],
        capture_output=True, text=True, creationflags=NO_WINDOW,
    ).returncode == 0
    if not loaded:
        return Check(
            "launchagent", "Boot-time LaunchAgent installed", "warn",
            f"plist exists at {plist_path} but is not loaded",
            fix_id="launchagent",
            fix_label="`launchctl bootstrap gui/<uid> <plist>` to (re)load it",
        )
    return Check("launchagent", "Boot-time LaunchAgent installed", "ok", str(plist_path))


SYSTEMD_UNIT_NAME = "local-llm-hub.service"


def _systemd_unit_path() -> Path:
    return Path("/etc/systemd/system") / SYSTEMD_UNIT_NAME


def _check_systemd_unit() -> Check:
    """Is the boot-time systemd unit installed and active? (#341/#368, linux-only)

    Boot autostart + crash respawn for a headless Linux satellite's otherwise
    unsupervised hub — the systemd analogue of the Mac's LaunchAgent and the
    Windows tray's spawn-on-launch. ``systemctl is-active``/``is-enabled`` are
    read-only (no sudo); the fix renders + installs the unit (that half needs
    passwordless sudo, see ``_fix_systemd_unit``).
    """
    unit_path = _systemd_unit_path()
    if not unit_path.exists():
        return Check(
            "systemd_unit", "Boot-time systemd unit installed", "missing",
            f"expected at {unit_path}",
            fix_id="systemd_unit",
            fix_label="renders linux/systemd/local-llm-hub.service + `systemctl enable --now` (boot autostart + crash respawn)",
        )
    active = subprocess.run(
        ["systemctl", "is-active", SYSTEMD_UNIT_NAME],
        capture_output=True, text=True, creationflags=NO_WINDOW,
    ).stdout.strip()
    enabled = subprocess.run(
        ["systemctl", "is-enabled", SYSTEMD_UNIT_NAME],
        capture_output=True, text=True, creationflags=NO_WINDOW,
    ).stdout.strip()
    if active != "active":
        return Check(
            "systemd_unit", "Boot-time systemd unit installed", "warn",
            f"unit at {unit_path} but is-active={active or '?'} (is-enabled={enabled or '?'})",
            fix_id="systemd_unit",
            fix_label="`systemctl enable --now local-llm-hub` to (re)start it",
        )
    return Check("systemd_unit", "Boot-time systemd unit installed", "ok",
                 f"{unit_path} (active, is-enabled={enabled or '?'})")


def _tts_enabled() -> bool:
    return any(m.backend == "tts" or m.engine == "tts-server"
               for m in local_models())


def _check_tts() -> Check:
    """Are the TTS Python deps present? (chatterbox/orpheus/kokoro/piper runtimes.)

    Gated on a tts-server row being enabled. The weights themselves stream
    from the HF cache on first start, so this only verifies the packages
    are importable — the heavy bit that ``requirements.txt`` deliberately
    omits (torch) so non-TTS hosts stay lean.
    """
    missing = []
    for mod in ("torch", "chatterbox", "snac", "kokoro_onnx"):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(
                "chatterbox-tts" if mod == "chatterbox"
                else "kokoro-onnx" if mod == "kokoro_onnx"
                else mod
            )
    if missing:
        return Check("tts", "TTS deps installed (chatterbox, snac, kokoro, piper)", "missing",
                     f"missing: {', '.join(missing)}",
                     fix_id="tts",
                     fix_label="scripts/install_tts.py (pip install -r requirements-tts.txt + warm weights)")
    piper_rows = [m for m in local_models()
                  if m.backend == "tts" and m.tts_engine == "piper" and m.model_path]
    if piper_rows:
        exe_name = "piper.exe" if sys.platform == "win32" else "piper"
        piper_exe = PROJECT_ROOT / "vendor" / "piper" / exe_name
        voice = PROJECT_ROOT / str(piper_rows[0].model_path)
        if not piper_exe.exists() or not voice.exists() or not Path(str(voice) + ".json").exists():
            return Check("tts", "TTS deps installed (chatterbox, snac, kokoro, piper)", "missing",
                         "missing Piper binary or voice assets",
                         fix_id="tts",
                         fix_label="scripts/install_tts.py (download Piper assets)")
    return Check("tts", "TTS deps installed (chatterbox, snac, kokoro, piper)", "ok",
                 "torch + chatterbox-tts + snac + kokoro-onnx importable; Piper assets present")


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True


def _check_ports() -> List[Check]:
    rows: List[Check] = []
    for label, port in [("hub", hub_port())] + [
        (m.display_name, m.port)
        for m in local_models()
        if m.backend in ("openai", "whisper", "tts") and m.port and not m.virtual
    ]:
        if _port_in_use(port):
            rows.append(Check(
                f"port_{port}", f"Port {port} free ({label})", "warn",
                f"port {port} already in use — may be our own process"
            ))
        else:
            rows.append(Check(f"port_{port}", f"Port {port} free ({label})", "ok", f"port {port} free"))
    return rows


# ---------- report + fixes ----------

# Brief cache for run_all_checks(use_cache=True) (issue #198). The battery
# shells out to `claude --version` / `nvidia-smi` / `llama-server --version`
# via blocking subprocess.run and "can pin the entire uvicorn worker for
# seconds" (see app_web/routers/hub.py's install_status docstring) — yet
# install_fix/install_fix_all only need it to *locate* one check by
# fix_id, and the admin UI always calls install_status (which populates
# this cache) moments before a user clicks a fix button. A short TTL means
# a fix click right after a status load reuses that report instead of
# re-running the whole battery; a stale/empty cache still falls back to a
# fresh run, so correctness never depends on the cache being warm.
_CACHE_TTL_S = 5.0
_cached_report: Optional[Report] = None
_cached_at: float = 0.0


def run_all_checks(*, use_cache: bool = False) -> Report:
    """Run every install check. Every call refreshes the brief cache used
    by ``use_cache=True`` callers; pass ``use_cache=True`` to reuse a
    report computed within the last ``_CACHE_TTL_S`` seconds instead of
    forcing a fresh (expensive) battery run.
    """
    global _cached_report, _cached_at
    if use_cache and _cached_report is not None and (time.monotonic() - _cached_at) < _CACHE_TTL_S:
        return _cached_report

    checks: List[Check] = [
        _check_python_venv(),
        _check_deps(),
        _check_host_profile(),
        _check_claude_cli(),
        _check_gemini_cli(),
        _check_gpu(),
        _check_llama_cpp(),
    ]
    if sys.platform == "darwin":
        checks.append(_check_launchagent())
    if sys.platform.startswith("linux"):
        checks.append(_check_systemd_unit())
    if _whisper_enabled():
        checks.append(_check_whisper_cpp())
    if _tts_enabled():
        checks.append(_check_tts())
    if _parakeet_enabled():
        checks.append(_check_parakeet_worker())
    checks.extend(_check_models())
    checks.extend(_check_ports())
    report = Report(checks=checks)
    _cached_report = report
    _cached_at = time.monotonic()
    return report


FixFn = Callable[[], None]


def _fix_deps() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")],
        check=True,
        creationflags=NO_WINDOW,
    )


def _fix_llama_cpp() -> None:
    from scripts import install_llama_cpp  # type: ignore
    install_llama_cpp.main()


def _fix_whisper_cpp() -> None:
    from scripts import install_whisper_cpp  # type: ignore
    install_whisper_cpp.main()


def _fix_tts() -> None:
    from scripts import install_tts  # type: ignore
    install_tts.main()


def _fix_parakeet_worker() -> None:
    swift = shutil.which("swift")
    if not swift:
        raise RuntimeError(
            "swift not found on PATH — install Xcode Command Line Tools "
            "(headless: `softwareupdate --install \"Command Line Tools for Xcode\"`)"
        )
    subprocess.run(
        [swift, "build", "-c", "release"],
        cwd=str(PROJECT_ROOT / "mac" / "parakeet-worker"),
        check=True,
        creationflags=NO_WINDOW,
    )


def _fix_launchagent() -> None:
    if sys.platform != "darwin":
        raise RuntimeError("LaunchAgent install only applies on macOS")
    template_path = PROJECT_ROOT / "mac" / "launchagent" / f"{LAUNCHAGENT_LABEL}.plist"
    rendered = (
        template_path.read_text(encoding="utf-8")
        .replace("__PYTHON__", sys.executable)
        .replace("__PROJECT_ROOT__", str(PROJECT_ROOT))
    )
    (PROJECT_ROOT / "data" / "logs").mkdir(parents=True, exist_ok=True)
    plist_path = _launchagent_plist_path()
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(rendered, encoding="utf-8")
    uid = os.getuid()
    # bootout first so a re-run picks up a changed plist (bootstrap alone
    # errors "already bootstrapped" on a still-loaded label); ignore its
    # failure when nothing was loaded yet. launchd needs a beat to fully
    # release the label after an actual bootout — an immediate bootstrap
    # can transiently fail with "Input/output error" (confirmed live) —
    # so retry a few times with a short pause rather than a single attempt.
    subprocess.run(["launchctl", "bootout", f"gui/{uid}/{LAUNCHAGENT_LABEL}"],
                   capture_output=True, creationflags=NO_WINDOW)
    result = None
    for _ in range(5):
        time.sleep(1)
        result = subprocess.run(
            ["launchctl", "bootstrap", f"gui/{uid}", str(plist_path)],
            capture_output=True, text=True, creationflags=NO_WINDOW,
        )
        if result.returncode == 0:
            break
    if result is None or result.returncode != 0:
        detail = result.stderr.strip() if result else "no attempt ran"
        raise RuntimeError(f"launchctl bootstrap failed after retries: {detail}")


def _fix_systemd_unit() -> None:
    """Render + install the boot-time systemd unit on a Linux satellite (#368).

    The Linux counterpart to ``_fix_launchagent``. Substitutes the two
    placeholders in ``linux/systemd/local-llm-hub.service`` with this box's
    concrete values, writes it into the root-owned system unit dir via
    ``sudo -n tee``, then ``daemon-reload`` + ``enable --now``. ``sudo -n`` so a
    missing passwordless-sudo drop-in (see docs/machines.md) fails fast with a
    clear message instead of hanging on a password prompt.
    """
    if not sys.platform.startswith("linux"):
        raise RuntimeError("systemd unit install only applies on Linux")
    template_path = PROJECT_ROOT / "linux" / "systemd" / SYSTEMD_UNIT_NAME
    rendered = (
        template_path.read_text(encoding="utf-8")
        .replace("__PROJECT_ROOT__", str(PROJECT_ROOT))
        .replace("__USER__", getpass.getuser())
    )
    unit_path = _systemd_unit_path()
    tee = subprocess.run(
        ["sudo", "-n", "tee", str(unit_path)],
        input=rendered, capture_output=True, text=True, creationflags=NO_WINDOW,
    )
    if tee.returncode != 0:
        raise RuntimeError(
            f"failed to write {unit_path} via `sudo -n tee`: "
            f"{tee.stderr.strip() or 'sudo -n denied — is passwordless sudo configured? (docs/machines.md)'}"
        )
    for args in (["daemon-reload"], ["enable", "--now", SYSTEMD_UNIT_NAME]):
        p = subprocess.run(["sudo", "-n", "systemctl", *args],
                           capture_output=True, text=True, creationflags=NO_WINDOW)
        if p.returncode != 0:
            raise RuntimeError(
                f"`sudo -n systemctl {' '.join(args)}` failed: "
                f"{p.stderr.strip() or 'sudo -n denied — passwordless sudo not configured?'}"
            )


def _fix_download(model_id: str) -> Callable[[], None]:
    def _fix() -> None:
        from scripts import download_models  # type: ignore
        download_models.download_one(model_id)
    return _fix


def fix_fn_for(check: Check) -> Optional[FixFn]:
    if check.fix_id == "deps":
        return _fix_deps
    if check.fix_id == "llama_cpp":
        return _fix_llama_cpp
    if check.fix_id == "whisper_cpp":
        return _fix_whisper_cpp
    if check.fix_id == "tts":
        return _fix_tts
    if check.fix_id == "parakeet_worker":
        return _fix_parakeet_worker
    if check.fix_id == "launchagent":
        return _fix_launchagent
    if check.fix_id == "systemd_unit":
        return _fix_systemd_unit
    if check.fix_id and check.fix_id.startswith("download_"):
        return _fix_download(check.fix_id[len("download_"):])
    return None


# ---------- CLI ----------

_STATUS_GLYPH = {"ok": "OK", "warn": "!!", "missing": "??", "error": "xx"}


def _print_report(report: Report) -> None:
    width = max(len(c.label) for c in report.checks) + 2
    for c in report.checks:
        glyph = _STATUS_GLYPH.get(c.status, "?")
        log.info("  %s [%7s] %s %s", glyph, c.status, c.label.ljust(width), c.detail)
    log.info("")
    if report.ok:
        log.info("overall: %s", report.worst_status)
    else:
        log.info("overall: %s - run with --fix to attempt repairs", report.worst_status)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="python -m src.install", description="First-run checks for local-llm-hub")
    p.add_argument("--fix", action="store_true", help="attempt to fix every fixable row")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args(argv)

    report = run_all_checks()

    if args.fix:
        for c in list(report.checks):
            if c.status in ("missing", "error"):
                fn = fix_fn_for(c)
                if fn is None:
                    continue
                log.info("-> fixing %s: %s", c.id, c.fix_label)
                try:
                    fn()
                except Exception as e:
                    log.error("   fix failed: %s", e)
        report = run_all_checks()

    if args.json:
        print(json.dumps([c.__dict__ for c in report.checks], indent=2))
    else:
        _print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
