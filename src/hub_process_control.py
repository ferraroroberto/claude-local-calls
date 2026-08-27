"""Self-supervision for the running hub process — stop/restart signaling.

Split out of ``app_web/routers/hub.py`` (issue #533) — these helpers are
pure platform process-supervision (a self-signal thread, ``launchctl
bootout``/``kickstart``, ``sudo -n systemctl``, and a detached
respawn-watchdog spawn) with no FastAPI in them at all, unlike the
``@router`` endpoints in ``hub.py`` that call into them. Per
``app_web/admin_forward.py``'s own layering rule ("kept here in
``app_web``, not ``src`` — because it raises FastAPI ``HTTPException`` —
the non-web ``src`` layer must stay framework-free"), none of these
four qualify to live in ``app_web``.

Sibling to ``server_process.py`` (external Popen ownership, driven by the
tray) — this module instead is the process talking to *itself* (or to the
supervisor — launchd/systemd — that owns it) to stop or restart in place.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from src.server_process import WIN_NEW_GROUP

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _delayed_shutdown(delay: float = 0.4) -> None:
    """Signal ourselves to exit after ``delay`` seconds, so the HTTP
    response can flush first. Uvicorn handles SIGINT/SIGTERM as a clean
    shutdown on both Windows and POSIX."""

    def _runner() -> None:
        time.sleep(delay)
        try:
            if sys.platform == "win32":
                # signal.raise_signal arrived in 3.8 and works under
                # uvicorn's SIGINT handler.
                signal.raise_signal(signal.SIGINT)
            else:
                os.kill(os.getpid(), signal.SIGTERM)
        except Exception as exc:  # noqa: BLE001 — fall back
            logger.error("⚠️ shutdown signal failed: %s — using os._exit", exc)
            os._exit(0)

    threading.Thread(target=_runner, daemon=True).start()


def _delayed_darwin_bootout(label: str, delay: float = 0.4) -> None:
    """Unload the LaunchAgent job entirely, so a deliberate stop actually
    stays stopped (#181).

    Confirmed empirically on this machine: launchd's ``KeepAlive`` respawns
    the job after *any* signal-terminated exit — a plain self-SIGTERM
    (``_delayed_shutdown``) and even an explicit ``launchctl stop`` both got
    immediately relaunched. ``launchctl bootout`` is the only thing that
    actually removes the job from launchd's active registry, so nothing is
    left to respawn. Bringing it back requires ``launchctl bootstrap``
    again — the ``bootstrap`` action in ``mac/bin/hub-remote-ctl.sh`` and
    ``src/install.py``'s ``_fix_launchagent()`` both already do this.
    """

    def _runner() -> None:
        time.sleep(delay)
        subprocess.run(
            ["launchctl", "bootout", f"gui/{os.getuid()}/{label}"],
            capture_output=True,
        )

    threading.Thread(target=_runner, daemon=True).start()


def _under_systemd() -> bool:
    """Are we running as a systemd unit? (#341/#368)

    systemd sets ``INVOCATION_ID`` in every unit's environment — the standard
    "am I supervised by systemd" signal. Only then do the stop/restart
    endpoints drive ``systemctl`` (below): a dev running the hub by hand on
    Linux falls through to the plain self-signal path, which is correct there
    (nothing would respawn it). Gated on Linux so it never fires elsewhere.
    """
    return sys.platform.startswith("linux") and bool(os.environ.get("INVOCATION_ID"))


def _delayed_systemctl(verb: str, delay: float = 0.4) -> None:
    """Run ``sudo -n systemctl <verb> local-llm-hub`` after a short delay so the
    HTTP response flushes first (#368).

    ``stop``/``restart`` SIGTERM this very process (it lives in the unit's
    cgroup) — which is the point: unlike the plain self-SIGTERM path,
    ``Restart=always`` would immediately respawn a bare signal, so a *deliberate*
    stop must go through systemd itself. ``sudo -n`` never prompts; a missing
    passwordless-sudo rule is logged rather than hanging.
    """
    from src.install import SYSTEMD_UNIT_NAME

    def _runner() -> None:
        time.sleep(delay)
        r = subprocess.run(
            ["sudo", "-n", "systemctl", verb, SYSTEMD_UNIT_NAME],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            logger.error("⚠️ `sudo -n systemctl %s %s` failed: %s",
                         verb, SYSTEMD_UNIT_NAME, (r.stderr or "").strip())

    threading.Thread(target=_runner, daemon=True).start()


def _restart_log_path() -> Path:
    """File the detached watchdog redirects the relaunched server into.

    The respawn is detached and outlives this process, so its stdout has
    nowhere to go in-process — and under ``pythonw`` there is no console
    at all. We give it a real file so (a) ``src.server``'s import-time
    logging write doesn't crash a console-less child, and (b) a failed
    restart leaves a diagnostic trail instead of vanishing silently.
    """
    return PROJECT_ROOT / "data" / "logs" / "restart.log"


def _spawn_respawn_watchdog() -> None:
    """Spawn a detached Python that waits for our PID to die then re-launches us.

    The relaunch is made the way ``src/server_process.start()`` spawns the
    hub — never a bare ``pythonw`` with no stdout. The actual wait/relaunch
    logic lives in ``src/_respawn_watchdog.py`` (issue #198 — this used to
    be a ~60-line string literal built up line-by-line and fed to
    ``python -c``, invisible to lint/type-check and one quoting slip away
    from a silently-failed restart). That module is deliberately
    stdlib-only with no import from any other ``src.*`` module: it's the
    thing recovering *from* a broken deploy, so it can't assume the rest
    of the hub's package still imports cleanly — only its own module and
    the empty ``src/__init__.py`` need to load.
    """
    from src.host_profile import hub_port

    parent_pid = os.getpid()
    port = int(hub_port())
    log_path = _restart_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("🔄 respawn watchdog: relaunch log → %s", log_path)
    # Same flags src.server_process.start() uses for the hub itself —
    # DETACHED_PROCESS is deliberately omitted, it's mutually exclusive
    # with CREATE_NO_WINDOW per the Win32 CreateProcess docs (#282/#283).
    creationflags = WIN_NEW_GROUP
    # Capture the watchdog's own stdout/stderr to the same log so a
    # failure *before* it opens its own handle (e.g. a bad argv) is still
    # visible rather than swallowed by DEVNULL.
    wd_log = open(log_path, "a", encoding="utf-8", errors="replace")
    subprocess.Popen(
        [
            sys.executable, "-m", "src._respawn_watchdog",
            "--parent-pid", str(parent_pid),
            "--port", str(port),
            "--log-path", str(log_path),
            "--root", str(PROJECT_ROOT),
        ],
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=wd_log,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
