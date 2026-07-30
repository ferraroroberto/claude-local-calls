"""What commit is this process running? Single source of truth.

Shared by `/admin/api/version` (own build identity) and
`peer_health()` (peer build identity comparison, #181, generalized #372) so
both ask the same question the same way instead of computing it
independently twice.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict

from .no_window import NO_WINDOW

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_log = logging.getLogger(__name__)


def _resolve_git_sha() -> str:
    """Short git SHA, captured once at import. Falls back to ``"unknown"``."""
    cmd = ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"]
    kwargs: Dict[str, Any] = dict(
        capture_output=True,
        stdin=subprocess.DEVNULL,
        text=True,
        timeout=5,
        check=False,
        creationflags=NO_WINDOW,
    )
    try:
        result = subprocess.run(cmd, **kwargs)
    except (OSError, subprocess.SubprocessError) as exc:
        _log.warning("⚠️ build_info: git rev-parse raised %s: %s", type(exc).__name__, exc)
        return "unknown"
    sha = (result.stdout or "").strip()
    return sha or "unknown"


_GIT_SHA = _resolve_git_sha()


def git_sha() -> str:
    """This process's short git SHA, memoized at import time.

    A ``git pull`` on disk does not change this until the process restarts
    — intentional: it answers "what is *running*," not "what is on disk."
    """
    return _GIT_SHA
