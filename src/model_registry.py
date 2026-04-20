"""Load models.yaml and expose typed entries filtered by the active host.

One source of truth for: which models exist, how to launch them, which
port they listen on, and how the hub should route by model-name alias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .host_profile import CONFIG_PATH, HostProfile, resolve as resolve_host


@dataclass(frozen=True)
class Model:
    id: str                         # short key in YAML ("qwen", "glm", "claude")
    display_name: str               # name the client sends in the `model` field
    backend: str                    # "claude" | "openai"
    aliases: List[str] = field(default_factory=list)
    engine: Optional[str] = None
    port: Optional[int] = None
    hf_repo: Optional[str] = None
    hf_pattern: Optional[str] = None
    model_path: Optional[str] = None
    args: List[str] = field(default_factory=list)

    @property
    def all_names(self) -> List[str]:
        names = [self.display_name, *self.aliases]
        return [n for n in dict.fromkeys(names) if n]

    @property
    def url(self) -> Optional[str]:
        return f"http://127.0.0.1:{self.port}/v1" if self.port else None


def _load_config() -> Dict:
    return yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8")) or {}


def _row_to_model(model_id: str, row: Dict) -> Model:
    return Model(
        id=model_id,
        display_name=str(row.get("display_name") or model_id),
        backend=str(row.get("backend", "openai")),
        aliases=list(row.get("aliases", []) or []),
        engine=row.get("engine"),
        port=int(row["port"]) if row.get("port") is not None else None,
        hf_repo=row.get("hf_repo"),
        hf_pattern=row.get("hf_pattern"),
        model_path=row.get("model_path"),
        args=list(row.get("args", []) or []),
    )


def all_models() -> List[Model]:
    cfg = _load_config()
    rows: Dict = cfg.get("models") or {}
    return [_row_to_model(mid, row) for mid, row in rows.items()]


def enabled_models(host: Optional[HostProfile] = None) -> List[Model]:
    """Return models the active host is configured to serve.

    Claude is always enabled (the subscription path doesn't cost disk or
    VRAM) — openai-backed local models must appear in the host's
    `enabled` list.
    """
    profile = host or resolve_host()
    whitelist = set(profile.enabled)
    result: List[Model] = []
    for m in all_models():
        if m.backend == "claude" or m.id in whitelist:
            result.append(m)
    return result


def resolve(name: str, host: Optional[HostProfile] = None) -> Optional[Model]:
    """Look up a model by any of its names (display_name or aliases)."""
    name = name.strip()
    for m in enabled_models(host):
        if name == m.display_name or name in m.aliases:
            return m
    return None


def known_names(host: Optional[HostProfile] = None) -> List[str]:
    names: List[str] = []
    for m in enabled_models(host):
        names.extend(m.all_names)
    return sorted(dict.fromkeys(names))
