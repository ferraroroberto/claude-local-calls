"""Detect host hardware and write config/machine_specs.yaml.

Run from the project root:

    .venv\\Scripts\\python.exe scripts\\detect_machine_specs.py
    ./.venv/bin/python scripts/detect_machine_specs.py

Use --print to dump to stdout without touching disk, --force to overwrite
an existing file.
"""

from __future__ import annotations

import argparse
import logging
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import psutil
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import no_window_flags  # noqa: E402

logger = logging.getLogger("detect_machine_specs")

# UTF-8 stdout so emoji log lines don't throw under a captured/redirected run
# on Windows (cp1252 fallback otherwise raises UnicodeEncodeError) — same
# guard as scripts/bench_voice.py and scripts/bench_orpheus.py.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "config" / "machine_specs.yaml"

# Absolute path — the bare "powershell" name is unpinned to a resolvable
# executable across machines/PATH states. See ~/.claude/CLAUDE.md,
# "Windows PowerShell in spawned commands".
POWERSHELL_EXE = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"


def _round_gb(num_bytes: int) -> float:
    return round(num_bytes / (1024 ** 3), 2)


def detect_os() -> dict[str, Any]:
    return {
        "name": f"{platform.system()} {platform.release()}".strip(),
        "version": platform.version(),
        "architecture": platform.machine(),
    }


def detect_cpu() -> dict[str, Any]:
    model = ""
    if sys.platform == "win32":
        model = _cpu_model_windows()
    elif sys.platform.startswith("linux"):
        model = _cpu_model_linux()
    if not model:
        model = platform.processor() or ""

    freq = psutil.cpu_freq()
    base_mhz: Optional[int] = None
    if freq is not None:
        base_mhz = int(freq.max or freq.current or 0) or None

    return {
        "model": model.strip() or "unknown",
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True),
        "base_clock_mhz": base_mhz,
    }


def _cpu_model_windows() -> str:
    try:
        out = subprocess.run(
            [POWERSHELL_EXE, "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"],
            capture_output=True, text=True, timeout=15, check=True,
            creationflags=no_window_flags(),
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return ""


def _cpu_model_linux() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def detect_memory() -> dict[str, Any]:
    return {"total_gb": _round_gb(psutil.virtual_memory().total)}


def detect_gpus() -> list[dict[str, Any]]:
    gpus = _gpus_via_nvidia_smi()
    if gpus:
        gpus[0]["primary"] = True
        return gpus
    if sys.platform == "win32":
        return _gpus_via_windows_cim()
    return []


def _gpus_via_nvidia_smi() -> list[dict[str, Any]]:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
            creationflags=no_window_flags(),
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return []

    gpus: list[dict[str, Any]] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        name, vram_mib, driver, compute_cap = parts[:4]
        try:
            vram_gb = round(int(vram_mib) / 1024, 2)
        except ValueError:
            vram_gb = None
        gpus.append({
            "name": name,
            "vram_gb": vram_gb,
            "driver_version": driver,
            "compute_capability": compute_cap,
            "cuda": True,
            "primary": False,
        })
    return gpus


def _gpus_via_windows_cim() -> list[dict[str, Any]]:
    try:
        out = subprocess.run(
            [POWERSHELL_EXE, "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | "
             "Select-Object Name,AdapterRAM,DriverVersion | "
             "ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=15, check=True,
            creationflags=no_window_flags(),
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return []

    import json
    try:
        data = json.loads(out.stdout or "[]")
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]

    gpus: list[dict[str, Any]] = []
    for entry in data:
        name = (entry.get("Name") or "").strip()
        if not name or "DisplayLink" in name:
            continue
        ram = entry.get("AdapterRAM")
        vram_gb = _round_gb(ram) if isinstance(ram, int) and ram > 0 else None
        gpus.append({
            "name": name,
            "vram_gb": vram_gb,
            "driver_version": entry.get("DriverVersion"),
            "compute_capability": None,
            "cuda": "NVIDIA" in name.upper(),
            "primary": False,
        })
    if gpus:
        gpus[0]["primary"] = True
    return gpus


def detect_storage() -> list[dict[str, Any]]:
    if sys.platform == "win32":
        return _storage_windows()
    if sys.platform.startswith("linux"):
        return _storage_linux()
    return []


def _storage_windows() -> list[dict[str, Any]]:
    try:
        out = subprocess.run(
            [POWERSHELL_EXE, "-NoProfile", "-Command",
             "Get-CimInstance Win32_DiskDrive | "
             "Select-Object Model,Size,MediaType | "
             "ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=15, check=True,
            creationflags=no_window_flags(),
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return []

    import json
    try:
        data = json.loads(out.stdout or "[]")
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]

    disks: list[dict[str, Any]] = []
    for entry in data:
        model = (entry.get("Model") or "").strip()
        size = entry.get("Size")
        size_gb = _round_gb(size) if isinstance(size, int) and size > 0 else None
        media = entry.get("MediaType") or ""
        disks.append({
            "model": model or "unknown",
            "size_gb": size_gb,
            "type": _classify_disk(model, media),
        })
    return disks


def _storage_linux() -> list[dict[str, Any]]:
    try:
        out = subprocess.run(
            ["lsblk", "-J", "-b", "-d", "-o", "NAME,SIZE,MODEL,ROTA"],
            capture_output=True, text=True, timeout=15, check=True,
            creationflags=no_window_flags(),
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return []

    import json
    try:
        data = json.loads(out.stdout or "{}")
    except json.JSONDecodeError:
        return []

    disks: list[dict[str, Any]] = []
    for entry in data.get("blockdevices", []):
        name = entry.get("name", "")
        if name.startswith(("loop", "sr")):
            continue
        size = entry.get("size")
        size_gb = _round_gb(int(size)) if size else None
        rota = str(entry.get("rota", "0")) == "1"
        if rota:
            disk_type = "HDD"
        elif name.startswith("nvme"):
            disk_type = "NVMe SSD"
        else:
            disk_type = "SATA SSD"
        disks.append({
            "model": (entry.get("model") or name).strip(),
            "size_gb": size_gb,
            "type": disk_type,
        })
    return disks


def _classify_disk(model: str, media: str) -> str:
    upper = (model + " " + media).upper()
    if "NVME" in upper or re.search(r"\bSN\d{3,}", upper):
        return "NVMe SSD"
    if "SSD" in upper:
        return "SATA SSD"
    return "HDD"


def build_specs() -> dict[str, Any]:
    return {
        "machine": {
            "name": platform.node() or "this-host",
            "role": "primary",
        },
        "os": detect_os(),
        "cpu": detect_cpu(),
        "memory": detect_memory(),
        "gpus": detect_gpus(),
        "storage": detect_storage(),
        "notes": (
            "Auto-generated by scripts/detect_machine_specs.py. "
            "Edit freely - the script will refuse to overwrite without --force."
        ),
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output YAML path (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--print", dest="to_stdout", action="store_true",
        help="Print YAML to stdout instead of writing to disk.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    logger.info("ℹ️  Detecting hardware...")
    specs = build_specs()
    yaml_text = yaml.safe_dump(
        specs, sort_keys=False, default_flow_style=False, indent=2, allow_unicode=True,
    )

    if args.to_stdout:
        sys.stdout.write(yaml_text)
        return 0

    out: Path = args.output
    if out.exists() and not args.force:
        logger.error(
            "❌ %s already exists. Pass --force to overwrite, or --print to preview.",
            out,
        )
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml_text, encoding="utf-8")
    logger.info("✅ Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
