"""The hub's declarative "what should be up at launch" profile (issue #265).

Single source of truth for the *services* the hub brings up automatically on
every boot (tray, ``run_hub.bat``, or ``python -m src.run_backend hub``):

  * ``docker`` / ``langfuse`` — whether to run ``services.launch_stack()``
    (start Docker Desktop if down, then the Langfuse containers) at startup.
  * ``agentsview`` — whether to run ``services.launch_agentsview()`` (the
    optional AgentsView server feeding the Code tab's AGY vendor, #280).

Since #430 this file carries **service flags only**. The former ``models``
autostart list (and the fleet-wide ``config/fleet_placement.json`` it
mirrored) duplicated what ``config/models.yaml`` already declares per row —
``hosts:`` chains say *where* a model runs, ``startup: eager|on_demand``
(#422) says *whether* it runs eagerly — so the desired running set is now
derived from the registry (``model_registry.desired_model_ids``). A stale
``models`` key left in an existing gitignored live file is simply ignored on
load and dropped on the next save — no migration needed, same treatment as
the ``mac_mini_sync`` key retired in #374.

The live ``config/startup_profile.json`` is **gitignored** (issue #304): the
admin UI's Startup card rewrites it on every toggle, so tracking it would
dirty the tree on every flip. The committed
``config/startup_profile.example.json`` is the template and the fresh-clone
default — ``load_startup_profile`` falls back to it when the live file is
absent, so the example is the single source of default truth rather than
decorative. Same shape as ``config/machine_specs.yaml`` (real gitignored,
example committed); load/save mechanics still mirror
``config/transcription_glossary.json`` (atomic write, cache clear on save,
tolerant load that never raises).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE_PATH = PROJECT_ROOT / "config" / "startup_profile.json"
# Committed template + fresh-clone default (issue #304). Read only when the
# gitignored live profile above is absent — never written to.
EXAMPLE_PROFILE_PATH = PROJECT_ROOT / "config" / "startup_profile.example.json"


@dataclass(frozen=True)
class StartupProfile:
    docker: bool = True
    langfuse: bool = True
    # AgentsView server for the Code tab's AGY vendor (issue #280) — launch
    # soft-fails with a log line when the tool isn't installed.
    agentsview: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


_DEFAULT = StartupProfile()

# Parsed cache keyed by the resolved profile path (same shape as
# host_profile._CONFIG_CACHE) — keying on the path rather than relying on
# an lru_cache'd optional arg means swapping DEFAULT_PROFILE_PATH (as tests
# do) transparently busts the cache instead of returning a stale hit.
_PROFILE_CACHE: Dict[str, StartupProfile] = {}


def load_startup_profile(path: Optional[str] = None) -> StartupProfile:
    """Load the startup profile. Missing/unparseable file → the defaults.

    A broken or absent profile must never prevent the hub from starting —
    same tolerant-load contract as ``transcription_glossary.load_rules()``.
    Unknown keys (the retired ``models`` list, #430; ``mac_mini_sync``, #374)
    are ignored, so a pre-migration live file loads cleanly.

    When the live file is absent (fresh clone / first run) and no explicit
    ``path`` was given, the committed ``EXAMPLE_PROFILE_PATH`` template is read
    instead (issue #304), so the example seeds fresh-clone defaults. The cache
    still keys on the resolved live ``target`` so ``save_startup_profile``'s
    invalidation lands on the same slot once a real file is written.
    """
    target = Path(path) if path else DEFAULT_PROFILE_PATH
    key = str(target)
    cached = _PROFILE_CACHE.get(key)
    if cached is not None:
        return cached

    # Fall back to the committed template only for the default (live) path —
    # an explicit path is honoured verbatim so tests stay hermetic.
    source = target
    if not target.exists() and path is None and EXAMPLE_PROFILE_PATH.exists():
        source = EXAMPLE_PROFILE_PATH

    if not source.exists():
        result = _DEFAULT
    else:
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("⚠️ could not load startup profile %s: %s", source, exc)
            data = None
        if not isinstance(data, dict):
            result = _DEFAULT
        else:
            result = StartupProfile(
                docker=bool(data.get("docker", True)),
                langfuse=bool(data.get("langfuse", True)),
                agentsview=bool(data.get("agentsview", True)),
            )

    _PROFILE_CACHE[key] = result
    return result


def normalize_profile(data: Dict[str, Any]) -> StartupProfile:
    """Validate + clean an incoming profile payload for persistence.

    Service flags only (#430) — a lingering ``models`` key from an older
    caller is silently dropped rather than rejected, so a not-yet-updated
    peer PATching the old shape can never 400 the migration.
    """
    if not isinstance(data, dict):
        raise ValueError("startup profile must be a JSON object")
    return StartupProfile(
        docker=bool(data.get("docker", True)),
        langfuse=bool(data.get("langfuse", True)),
        agentsview=bool(data.get("agentsview", True)),
    )


def save_startup_profile(data: Dict[str, Any], path: Optional[str] = None) -> StartupProfile:
    """Validate, atomically write, and invalidate the load cache."""
    target = Path(path) if path else DEFAULT_PROFILE_PATH
    clean = normalize_profile(data)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(clean.as_dict(), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    _PROFILE_CACHE.pop(str(target), None)
    logger.info(
        "💾 Saved startup profile (docker=%s langfuse=%s agentsview=%s)",
        clean.docker, clean.langfuse, clean.agentsview,
    )
    return clean
