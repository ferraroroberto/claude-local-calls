"""First-run and health checks shared by the CLI and the Streamlit UI.

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
import importlib
import json
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .host_profile import hub_port, resolve as resolve_host
from .model_registry import Model, enabled_models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_LLAMA = PROJECT_ROOT / "vendor" / "llama.cpp"
VENDOR_WHISPER = PROJECT_ROOT / "vendor" / "whisper.cpp"

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
    "fastapi", "uvicorn", "httpx", "anthropic", "streamlit",
    "yaml", "huggingface_hub", "pydantic",
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


def _check_claude_cli() -> Check:
    exe = shutil.which("claude")
    if not exe:
        return Check("claude_cli", "`claude` CLI on PATH", "warn",
                     "not found — the Claude backend won't work until Claude Code is installed")
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            ver = (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr) else "ok"
            return Check("claude_cli", "`claude` CLI on PATH", "ok", ver)
        return Check("claude_cli", "`claude` CLI on PATH", "warn",
                     f"--version exited {r.returncode}")
    except Exception as e:
        return Check("claude_cli", "`claude` CLI on PATH", "warn", str(e))


def _check_gpu() -> Check:
    if sys.platform == "win32":
        nv = shutil.which("nvidia-smi")
        if not nv:
            return Check("gpu", "GPU / accelerator detected", "warn", "nvidia-smi not found")
        try:
            r = subprocess.run(
                [nv, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                return Check("gpu", "GPU / accelerator detected", "ok", r.stdout.strip().splitlines()[0])
            return Check("gpu", "GPU / accelerator detected", "warn",
                         f"nvidia-smi exit {r.returncode}: {r.stderr[:200]}")
        except Exception as e:
            return Check("gpu", "GPU / accelerator detected", "warn", str(e))
    if sys.platform == "darwin":
        mach = platform.machine()
        if mach == "arm64":
            return Check("gpu", "Apple Silicon GPU (Metal)", "ok", f"arch={mach}")
        return Check("gpu", "GPU / accelerator detected", "warn",
                     f"darwin but arch={mach} — MLX / Metal expect arm64")
    return Check("gpu", "GPU / accelerator detected", "warn",
                 f"unknown platform {sys.platform}")


def _llama_server_binary() -> Path:
    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    return VENDOR_LLAMA / name


def _check_llama_cpp() -> Check:
    bin_path = _llama_server_binary()
    if not bin_path.exists():
        return Check("llama_cpp", "llama.cpp binary installed", "missing",
                     f"expected at {bin_path}",
                     fix_id="llama_cpp",
                     fix_label="scripts/install_llama_cpp.py (downloads the platform-matching release)")
    try:
        r = subprocess.run([str(bin_path), "--version"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return Check("llama_cpp", "llama.cpp binary installed", "ok",
                         (r.stdout or r.stderr).strip().splitlines()[0])
        return Check("llama_cpp", "llama.cpp binary installed", "warn",
                     f"--version exited {r.returncode}")
    except Exception as e:
        return Check("llama_cpp", "llama.cpp binary installed", "warn", str(e))


def _check_models() -> List[Check]:
    rows: List[Check] = []
    for m in enabled_models():
        if m.backend not in ("openai", "whisper") or not m.model_path:
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


def _whisper_server_binary() -> Path:
    name = "whisper-server.exe" if sys.platform == "win32" else "whisper-server"
    return VENDOR_WHISPER / name


def _whisper_enabled() -> bool:
    return any(m.engine == "whisper-server" or m.backend == "whisper"
               for m in enabled_models())


def _check_whisper_cpp() -> Check:
    bin_path = _whisper_server_binary()
    if not bin_path.exists():
        return Check("whisper_cpp", "whisper.cpp binary installed", "missing",
                     f"expected at {bin_path}",
                     fix_id="whisper_cpp",
                     fix_label="scripts/install_whisper_cpp.py (downloads the platform-matching release)")
    try:
        # whisper-server prints usage on --help and may exit non-zero.
        r = subprocess.run([str(bin_path), "--help"], capture_output=True, text=True, timeout=10)
        if r.returncode in (0, 1):
            first = ((r.stdout or r.stderr).strip().splitlines() or ["ok"])[0]
            return Check("whisper_cpp", "whisper.cpp binary installed", "ok", first)
        return Check("whisper_cpp", "whisper.cpp binary installed", "warn",
                     f"--help exited {r.returncode}")
    except Exception as e:
        return Check("whisper_cpp", "whisper.cpp binary installed", "warn", str(e))


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
        for m in enabled_models()
        if m.backend in ("openai", "whisper") and m.port
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

def run_all_checks() -> Report:
    checks: List[Check] = [
        _check_python_venv(),
        _check_deps(),
        _check_host_profile(),
        _check_claude_cli(),
        _check_gpu(),
        _check_llama_cpp(),
    ]
    if _whisper_enabled():
        checks.append(_check_whisper_cpp())
    checks.extend(_check_models())
    checks.extend(_check_ports())
    return Report(checks=checks)


FixFn = Callable[[], None]


def _fix_deps() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")],
        check=True,
    )


def _fix_llama_cpp() -> None:
    from scripts import install_llama_cpp  # type: ignore
    install_llama_cpp.main()


def _fix_whisper_cpp() -> None:
    from scripts import install_whisper_cpp  # type: ignore
    install_whisper_cpp.main()


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
    if check.fix_id and check.fix_id.startswith("download_"):
        return _fix_download(check.fix_id[len("download_"):])
    return None


# ---------- CLI ----------

_STATUS_GLYPH = {"ok": "OK", "warn": "!!", "missing": "??", "error": "xx"}


def _print_report(report: Report) -> None:
    width = max(len(c.label) for c in report.checks) + 2
    for c in report.checks:
        glyph = _STATUS_GLYPH.get(c.status, "?")
        print(f"  {glyph} [{c.status:7}] {c.label.ljust(width)} {c.detail}")
    print()
    if report.ok:
        print(f"overall: {report.worst_status}")
    else:
        print(f"overall: {report.worst_status} - run with --fix to attempt repairs")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.install", description="First-run checks for claude-local-calls")
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
                print(f"-> fixing {c.id}: {c.fix_label}")
                try:
                    fn()
                except Exception as e:
                    print(f"   fix failed: {e}")
        report = run_all_checks()

    if args.json:
        print(json.dumps([c.__dict__ for c in report.checks], indent=2))
    else:
        _print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
