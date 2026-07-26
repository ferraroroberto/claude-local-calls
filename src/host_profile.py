"""Resolve which host profile from config/models.yaml applies to this machine.

The registry keeps per-host settings (which models are enabled, etc.)
keyed by a short id. At runtime we pick the matching row based on
`sys.platform` and hostname, with `default: true` as a tiebreaker and
the `LOCAL_LLM_HUB_HOST` env var as an explicit override.

``_load_config()`` below is also the single cached YAML loader for
``config/models.yaml`` — ``src/model_registry.py`` imports it directly
rather than keeping its own parallel cache of the same file.
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
ENV_OVERRIDE = "LOCAL_LLM_HUB_HOST"


@dataclass(frozen=True)
class HostProfile:
    id: str
    platform: str
    enabled: List[str]
    hostname: Optional[str] = None
    default: bool = False
    source: str = ""  # human-readable description of how we picked it
    # LAN address (IP or resolvable hostname) other hosts dial to reach this
    # machine's own hub — e.g. "192.168.0.14". Unset on hosts nothing ever
    # proxies to (today, that's fine; only a host that owns a remote-tagged
    # model row needs one). See src/remote_proxy.py.
    address: Optional[str] = None
    # SSH login for the remote-bootstrap/sync endpoints (#181) — not a
    # secret, just which account to log in as; the private key path lives
    # in .env (LOCAL_LLM_HUB_SSH_KEY), never in this committed file. Unset
    # on hosts nothing ever SSHes into. See src/remote_bootstrap.py.
    ssh_user: Optional[str] = None
    # --- Machines-tab console metadata (#309) — never touches model routing.
    # A human label + one-line role + Lucide glyph id for the machine card.
    display_name: Optional[str] = None
    role: Optional[str] = None
    icon: Optional[str] = None
    # A powered-down node: shown as a card but not live-probed on the LAN.
    dormant: bool = False
    # Tailscale magic-DNS hostname — peer reachability + off-LAN RDP target.
    tailscale: Optional[str] = None
    # Remote-Desktop launch target: {"address": ..., "user": ...}. Unset on
    # hosts with no RDP path (e.g. the local host, or SSH/VNC-only peers).
    rdp: Optional[Dict[str, str]] = None
    # Wired-NIC MAC address for Wake-on-LAN (#356), e.g. "aa:bb:cc:dd:ee:ff".
    # WiFi WOL is unsupported — unset on hosts with no wired NIC (e.g. a
    # laptop) or that nothing ever wakes remotely.
    mac: Optional[str] = None
    # Discrete-GPU VRAM ceiling in MB (#375), promoted from docs/machines.md's
    # prose hardware facts. The fleet placement grid sums a host's placed
    # models' ``est_vram_mb`` and warns (advisory only, never blocks) when the
    # total exceeds this. Unset on hosts with no discrete-VRAM notion to check
    # against — the Apple-silicon Mac Mini (unified memory) and managed-only
    # boxes that serve no models; a host with no ceiling never warns.
    vram_mb: Optional[int] = None
    # Total system RAM in MB (#431), promoted from docs/machines.md's prose
    # hardware facts like ``vram_mb`` above. Display-only context on the
    # Models tab's fleet summary — never feeds a warning or a routing
    # decision. Unset where the fact isn't documented.
    ram_mb: Optional[int] = None

    @property
    def can_ssh(self) -> bool:
        """True when this host has both an address and an ssh_user — the
        prerequisite for the SSH-driven actions (remote uptime, reboot,
        shutdown). The active host and any SSH-less peer return False."""
        return bool(self.address and self.ssh_user)


# Parsed-YAML cache, keyed by the resolved config path. The README's
# contract is "edit the YAML and restart the hub to pick up changes," so a
# single hub request shouldn't pay several YAML parses behind the scenes
# (resolve() + hub_port() + hub_bind_host() + model_registry's all_models()/
# desired_model_ids() all read the same file). Keying on the path means
# swapping ``CONFIG_PATH`` (as the tests do) transparently busts the cache.
_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_config() -> Dict[str, Any]:
    key = str(CONFIG_PATH)
    cached = _CONFIG_CACHE.get(key)
    if cached is not None:
        return cached
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config file missing: {CONFIG_PATH}")
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    _CONFIG_CACHE[key] = data
    return data


def _row_to_profile(host_id: str, row: Dict[str, Any], *, source: str) -> HostProfile:
    return HostProfile(
        id=host_id,
        platform=str(row.get("platform", "")),
        enabled=list(row.get("enabled", []) or []),
        hostname=row.get("hostname"),
        default=bool(row.get("default", False)),
        source=source,
        address=row.get("address"),
        ssh_user=row.get("ssh_user"),
        display_name=row.get("display_name"),
        role=row.get("role"),
        icon=row.get("icon"),
        dormant=bool(row.get("dormant", False)),
        tailscale=row.get("tailscale"),
        rdp=row.get("rdp"),
        mac=row.get("mac"),
        vram_mb=int(row["vram_mb"]) if row.get("vram_mb") is not None else None,
        ram_mb=int(row["ram_mb"]) if row.get("ram_mb") is not None else None,
    )


def resolve() -> HostProfile:
    """Pick the host profile for this machine.

    Precedence:
      1. `LOCAL_LLM_HUB_HOST` env var selects an exact id.
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


def get_host(host_id: str) -> Optional[HostProfile]:
    """Look up any declared host row by id — including hosts other than the
    one this process is running on. Used to resolve a remote model's owning
    host's ``address`` (see src/remote_proxy.py); ``resolve()`` above only
    ever returns the *active* host's profile, not an arbitrary one.
    """
    cfg = _load_config()
    hosts: Dict[str, Any] = cfg.get("hosts") or {}
    row = hosts.get(host_id)
    if row is None:
        return None
    return _row_to_profile(host_id, row, source=f"lookup {host_id!r}")


def all_hosts() -> List[HostProfile]:
    """Every declared host row as a profile, in config order — the machine
    console's inventory (#309). Unlike ``resolve()`` (the active host) or
    ``get_host()`` (one by id), this returns the whole fleet, including
    managed-only machines that serve no models (OpenClaw, gaming).
    """
    cfg = _load_config()
    hosts: Dict[str, Any] = cfg.get("hosts") or {}
    return [
        _row_to_profile(host_id, row, source="all_hosts")
        for host_id, row in hosts.items()
    ]


def hub_port() -> int:
    cfg = _load_config()
    return int((cfg.get("hub") or {}).get("port", 8000))


def hub_bind_host() -> str:
    cfg = _load_config()
    return str((cfg.get("hub") or {}).get("bind_host", "0.0.0.0"))
