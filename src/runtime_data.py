"""Where an always-on service's SQLite state lives (project-scaffolding#243).

A fleet app's runtime database has, historically, landed wherever the repo
happened to be checked out — ``<repo>/webapp/telemetry.sqlite3``,
``<repo>/data/tasks.db``. That is fine on one machine and wrong on the next:
on ``tower`` the checkouts live on a spinning 4TB HDD, and seven always-on
services each ``fsync``-ing small writes at a poll interval kept the drive
audibly clicking around the clock (the drive itself measured healthy; the
noise was pure aggregate write pressure).

This module resolves that path instead of hardcoding it, so the physical
drive is a machine-level decision rather than a consequence of where someone
cloned the repo. Every app gets its own subdirectory under one root, so two
apps can both own a ``telemetry.sqlite3`` without colliding.

Resolution order for a single database file, highest precedence first:

1. ``env_var`` — the app's own full-path override (``TELEMETRY_DB_PATH``,
   ``WR_DB_PATH``, ``TASKOS_DB_PATH``, …). Names a *file*, not a directory.
   Kept because unit and e2e harnesses already point it at a temp path; this
   module must not break them.
2. ``<APP>_DATA_DIR`` — a directory for this one app (slug upper-cased,
   ``-`` → ``_``; e.g. ``HOME_AUTOMATION_DATA_DIR``).
3. ``FLEET_DATA_ROOT`` — the fleet-wide root; the app's directory is
   ``<root>/<app>``.
4. The platform default root: ``C:\\sqlite`` on Windows (deliberately a
   short, top-level, obvious name — "what is in here" answers itself),
   ``$XDG_DATA_HOME/sqlite`` (or ``~/.local/share/sqlite``) elsewhere.

**Change the path, don't junction it.** A directory junction pointing
``webapp/`` at another drive would migrate the data with zero code change,
and that is exactly its problem: nothing in the tree records which drive is
really being written, a recursive delete walks the reparse point, and no test
can assert the outcome. A resolved path is greppable, overridable per
process, and asserted by ``tests/test_runtime_data.py``.

**Relocating leaves the git working tree, so it leaves git-derived backup.**
``fleet-config``'s ``backup_private.py`` selects files via ``git ls-files``;
anything under this root is invisible to it and needs the explicit
``C:\\sqlite`` backup source (``fleet-config#724``). An app that invents its
own root outside this one is unbacked.

Vendor-verbatim: adopters copy this file byte-identical into their own
``src/`` (like ``src/no_window.py``, ``src/pooled_http.py``). The app slug and
the legacy env-var name are call-site arguments, so the copy never forks.
Stdlib-only and side-effect-free at import, so a standalone script under a
stripped-down venv can import it safely.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

FLEET_DATA_ROOT_ENV = "FLEET_DATA_ROOT"

#: Windows default. Top-level and unabbreviated on purpose: someone finding
#: this directory on a strange machine should not have to guess what it holds.
WINDOWS_DEFAULT_ROOT = Path("C:/sqlite")


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def app_dir_env_var(app: str) -> str:
    """The per-app directory override name for ``app`` (``home-automation`` →
    ``HOME_AUTOMATION_DATA_DIR``)."""
    return f"{app.replace('-', '_').upper()}_DATA_DIR"


def fleet_data_root() -> Path:
    """The root every app's runtime-data directory sits under."""
    override = _env(FLEET_DATA_ROOT_ENV)
    if override:
        return Path(override)
    if sys.platform == "win32":
        return WINDOWS_DEFAULT_ROOT
    base = _env("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "sqlite"


def runtime_data_dir(app: str, *, create: bool = False) -> Path:
    """This app's runtime-data directory — ``<root>/<app>`` unless overridden.

    Pass ``create=True`` at connect time, never at import time: a root on an
    unwritable or absent drive should surface as an error from the call that
    needed it, not as a crash while the module graph is still loading.
    """
    override = _env(app_dir_env_var(app))
    path = Path(override) if override else fleet_data_root() / app
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_db_path(
    app: str,
    filename: str,
    *,
    env_var: str | None = None,
    create: bool = False,
) -> Path:
    """Resolve one database file for ``app``.

    ``env_var`` names the app's pre-existing full-path override and wins over
    everything else — keep passing it, or every harness that sets it starts
    writing to the real store.
    """
    if env_var:
        override = _env(env_var)
        if override:
            path = Path(override)
            if create:
                path.parent.mkdir(parents=True, exist_ok=True)
            return path
    return runtime_data_dir(app, create=create) / filename
