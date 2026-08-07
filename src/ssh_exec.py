"""One-shot SSH command execution — the single ``ssh`` invocation builder (#470).

Two callers used to build the same command line independently:
``remote_stats._run_ssh`` (read-only stats snapshots over the hub user's own
passwordless channel) and ``remote_bootstrap._run_ssh`` (bootstrap/sync over
the forced-command key, and reboot/shutdown over the general channel). They
differed only in what they *returned* — raw stdout vs. an ``{ok, error}``
dict — so the transport lives here and each caller adapts the result at its
own call site.

The option set is deliberately fixed: ``BatchMode=yes`` (never prompt — these
run under a service with no tty), an explicit ``ConnectTimeout`` so a dead
peer fails fast instead of hanging the hub, and
``StrictHostKeyChecking=accept-new`` (trust-on-first-use for a fleet host that
was just reimaged, while still refusing a *changed* key).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Optional

from .no_window import NO_WINDOW

# How long ``ssh`` may spend establishing the connection. Both callers size
# their own overall ``timeout`` as this plus their command's expected runtime.
CONNECT_TIMEOUT_S = 5


@dataclass(frozen=True)
class SshResult:
    """Outcome of one ``ssh`` invocation.

    ``error`` is set only when the ssh process never produced an exit status
    (spawn failure, timeout) — a *connected* ssh that exits non-zero reports
    that through ``returncode``/``stderr`` instead. Callers must distinguish
    the two: "could not run ssh" and "ssh ran and the remote said no" are
    different conditions.
    """

    returncode: Optional[int]
    stdout: str
    stderr: str
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.returncode == 0


def run_ssh(
    user_at_host: str,
    command: str,
    *,
    key_path: Optional[str] = None,
    timeout: float,
    connect_timeout: int = CONNECT_TIMEOUT_S,
) -> SshResult:
    """Run ``command`` on ``user_at_host`` over SSH and return the outcome.

    ``key_path`` selects the forced-command channel (``-i <key>``); omit it
    for the hub user's own general passwordless channel. Never raises — a
    spawn failure or timeout comes back as an ``SshResult`` with ``error``
    set, because every caller here is a soft-failing status/action path.
    """
    cmd = ["ssh"]
    if key_path:
        cmd += ["-i", key_path]
    cmd += [
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={connect_timeout}",
        "-o", "StrictHostKeyChecking=accept-new",
        user_at_host,
        command,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            creationflags=NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return SshResult(
            returncode=None, stdout="", stderr="",
            error=f"{type(exc).__name__}: {exc}",
        )
    return SshResult(
        returncode=result.returncode,
        stdout=result.stdout or "",
        stderr=result.stderr or "",
    )
