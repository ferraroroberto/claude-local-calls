"""Respawn watchdog for the hub's admin-triggered restart (issue #198).

Spawned by ``app_web/routers/hub.py``'s ``_spawn_respawn_watchdog()`` as a
detached ``python -m src._respawn_watchdog`` child right before the hub
signals its own shutdown. Waits for the parent PID to exit, waits for its
port to free, then relaunches ``python -m src.server`` and confirms it
comes back up on the same port, logging the outcome either way.

Deliberately stdlib-only — no import from any other ``src.*`` module.
This process is the thing *recovering from* a bad deploy (a broken import
introduced mid-restart by a ``git pull``, say), so it can't assume the
rest of the ``src`` package still imports cleanly; only the empty
``src/__init__.py`` and this module itself need to load.

Previously this logic was a ~60-line string literal built up line-by-line
and fed to ``python -c`` — unlintable, untestable, no type-checking, and
any quoting slip in an interpolated value was invisible until a live
restart silently failed. Now a real module: covered by this project's
byte-compile gate, and its pure helpers are unit-tested directly
(``tests/test_respawn_watchdog.py``).
"""

from __future__ import annotations

import argparse
import datetime
import os
import socket
import subprocess
import sys
import time
from typing import List, Optional


def is_alive(pid: int) -> bool:
    try:
        if sys.platform == "win32":
            r = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True,
                # Inline ternary, not src.no_window.NO_WINDOW — this module is
                # deliberately stdlib-only (see module docstring), it can't
                # assume any other src.* module still imports cleanly.
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            return str(pid) in r.stdout
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def port_is_free(port: int, timeout: float = 0.3) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def port_is_reachable(port: int, timeout: float = 0.3) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def relaunch_executable() -> str:
    """Normalise pythonw.exe -> python.exe: a console-less child crashes
    on src.server's import-time logging write."""
    exe = sys.executable
    if exe.lower().endswith("pythonw.exe"):
        cand = exe[: -len("pythonw.exe")] + "python.exe"
        if os.path.exists(cand):
            return cand
    return exe


def _stamp() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def run(parent_pid: int, port: int, log_path: str, root: str) -> None:
    deadline = time.time() + 30
    while time.time() < deadline and is_alive(parent_pid):
        time.sleep(0.3)

    # Wait briefly for the port to free.
    for _ in range(60):
        if port_is_free(port):
            break
        time.sleep(0.3)

    exe = relaunch_executable()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW

    with open(log_path, "a", encoding="utf-8", errors="replace") as logf:
        logf.write(f"{_stamp()} [respawn] relaunching {exe} -m src.server\n")
        logf.flush()
        child = subprocess.Popen(
            [exe, "-m", "src.server"],
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        ok = False
        for _ in range(100):  # ~30s
            time.sleep(0.3)
            if child.poll() is not None:
                break
            if port_is_reachable(port):
                ok = True
                break
        if ok:
            logf.write(f"{_stamp()} [respawn] hub back up on :{port} (pid={child.pid})\n")
        else:
            logf.write(
                f"{_stamp()} [respawn] FAILED to bring hub up on :{port} "
                f"(child rc={child.poll()})\n"
            )
        logf.flush()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src._respawn_watchdog")
    ap.add_argument("--parent-pid", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--log-path", required=True)
    ap.add_argument("--root", required=True)
    args = ap.parse_args(argv)
    run(args.parent_pid, args.port, args.log_path, args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
