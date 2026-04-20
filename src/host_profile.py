"""Resolve which host profile from config/models.yaml applies to this machine.

The registry keeps per-host settings (which models are enabled, etc.)
keyed by a short id. At runtime we pick the matching row based on
`sys.platform` and hostname, with `default: true` as a tiebreaker and
the `CLAUDE_LOCAL_CALLS_HOST` env var as an explicit override.
"""

from __future__ import annotations

import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "models.yaml"
ENV_OVERRIDE = "CLAUDE_LOCAL_CALLS_HOST"


@dataclass(frozen=True)
class HostProfile:
    id: str
    platform: str
    enabled: List[str]
    hostname: Optional[str] = None
    default: bool = False
    source: str = ""  # human-readable description of how we picked it


def _load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config file missing: {CONFIG_PATH}")
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


def _row_to_profile(host_id: str, row: Dict[str, Any], *, source: str) -> HostProfile:
    return HostProfile(
        id=host_id,
        platform=str(row.get("platform", "")),
        enabled=list(row.get("enabled", []) or []),
        hostname=row.get("hostname"),
        default=bool(row.get("default", False)),
        source=source,
    )


def resolve() -> HostProfile:
    """Pick the host profile for this machine.

    Precedence:
      1. `CLAUDE_LOCAL_CALLS_HOST` env var selects an exact id.
      2. Any host row whose `hostname` equals `socket.gethostname()`.
      3. Any host row matching `sys.platform` with `default: true`.
      4. Any host row matching `sys.platform`.
    """
    cfg = _load_config()
    hosts: Dict[str, Any] = cfg.get("hosts") or {}
    if not hosts:
        raise RuntimeError(f"no 'hosts' defined in {CONFIG_PATH}")

    override = os.environ.get(ENV_OVERRIDE)
    if override:
        if override not in hosts:
            raise RuntimeError(
                f"{ENV_OVERRIDE}={override!r} but {override!r} is not in "
                f"config hosts: {sorted(hosts.keys())}"
            )
        return _row_to_profile(override, hosts[override], source=f"env {ENV_OVERRIDE}")

    this_host = socket.gethostname().lower()
    this_platform = sys.platform

    for host_id, row in hosts.items():
        hn = row.get("hostname")
        if hn and str(hn).lower() == this_host:
            return _row_to_profile(host_id, row, source=f"hostname match {this_host}")

    for host_id, row in hosts.items():
        if row.get("platform") == this_platform and row.get("default"):
            return _row_to_profile(host_id, row, source=f"default for {this_platform}")

    for host_id, row in hosts.items():
        if row.get("platform") == this_platform:
            return _row_to_profile(host_id, row, source=f"first match for {this_platform}")

    raise RuntimeError(
        f"no host row matched platform={this_platform} "
        f"hostname={this_host} (available: {sorted(hosts.keys())})"
    )


def hub_port() -> int:
    cfg = _load_config()
    return int((cfg.get("hub") or {}).get("port", 8000))


def hub_bind_host() -> str:
    cfg = _load_config()
    return str((cfg.get("hub") or {}).get("bind_host", "0.0.0.0"))
