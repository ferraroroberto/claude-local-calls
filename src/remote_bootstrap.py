"""SSH-triggered remote hub bootstrap/sync (#181) + machine power (#309/#311).

Two distinct SSH channels, deliberately kept apart:

  * **bootstrap / sync** (#181) — brings a *dead* remote host's hub back up,
    or syncs it to the latest ``main`` + restarts. These ride the
    forced-command-restricted key (``mac/bin/hub-remote-ctl.sh``), never a
    general shell: the verb sent over SSH (``bootstrap`` / ``sync``) becomes
    ``$SSH_ORIGINAL_COMMAND`` on the remote end and the forced command there —
    not anything sent here — decides what runs. Gated on
    ``LOCAL_LLM_HUB_SSH_KEY``.
  * **reboot / shutdown** (#309, transport decided in #311) — the Machines
    console's destructive power actions run over the hub user's **own general
    SSH**, the same passwordless channel ``remote_stats`` already uses for
    read-only snapshots (``ssh <user>@<addr> "<cmd>"``, no ``-i`` key). The
    peer's own passwordless-sudo sudoers drop-in — already required for the
    forced-command path — is the only prerequisite, so no per-peer key or
    forced-command script has to be deployed to a managed machine (e.g.
    OpenClaw). This means power actions do **not** depend on
    ``LOCAL_LLM_HUB_SSH_KEY`` being set.

Symmetric by construction: which host can dial which is entirely a
function of ``HostProfile.ssh_user``/``address`` (config) and, for
bootstrap/sync, whether ``LOCAL_LLM_HUB_SSH_KEY`` is set in this process's
``.env`` (today, only on ``tower``) — nothing here hardcodes which host is
the caller.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

import httpx

from . import remote_stats
from .host_profile import get_host, hub_port
from .ssh_exec import run_ssh

logger = logging.getLogger(__name__)

_SSH_KEY_ENV = "LOCAL_LLM_HUB_SSH_KEY"
_SSH_CONNECT_TIMEOUT_S = 5
_HEALTH_POLL_TIMEOUT_S = 30
_HEALTH_POLL_INTERVAL_S = 1.0


def _ssh_key_path() -> Optional[str]:
    return os.environ.get(_SSH_KEY_ENV)


def _resolve_dialable(host_id: str) -> tuple[Optional[Any], Optional[str], Optional[Dict[str, Any]]]:
    """Look up ``host_id`` and dial its address. Returns ``(owner, address,
    None)`` on success, or ``(None, None, <error dict>)`` when the host is
    unconfigured/unreachable-by-config — shared by the forced-command and
    general-SSH callers below, which otherwise duplicate this exact check."""
    owner = get_host(host_id)
    address = remote_stats.dial_address(owner, wait=True) if owner is not None else None
    if owner is None or not address or not owner.ssh_user:
        return None, None, {"ok": False, "error": f"host {host_id!r} has no address/ssh_user configured"}
    return owner, address, None


def _run_ssh(owner: Any, address: str, payload: str, *, use_key: bool) -> Dict[str, Any]:
    """Run one ``ssh <user>@<address> <payload>`` and normalize the result.

    ``use_key`` selects the forced-command channel (``-i`` key, restricted
    remote command) vs. the hub user's own general passwordless channel —
    the only two differences between ``bootstrap``/``sync`` and the
    reboot/shutdown power actions. The invocation itself is
    ``ssh_exec.run_ssh`` (shared with ``remote_stats``, #470); only the
    ``{ok, error}`` shape below is this layer's own."""
    key_path = None
    if use_key:
        key_path = _ssh_key_path()
        if not key_path:
            return {"ok": False, "error": f"{_SSH_KEY_ENV} is not set in .env"}
    result = run_ssh(
        f"{owner.ssh_user}@{address}",
        payload,
        key_path=key_path,
        timeout=_SSH_CONNECT_TIMEOUT_S + 10,
        connect_timeout=_SSH_CONNECT_TIMEOUT_S,
    )
    if result.error is not None:
        return {"ok": False, "error": result.error}
    if result.returncode != 0:
        return {"ok": False, "error": f"ssh exit {result.returncode}: {result.stderr.strip()}"}
    return {"ok": True}


def _run_remote_command(host_id: str, verb: str) -> Dict[str, Any]:
    key_path = _ssh_key_path()
    if not key_path:
        return {"ok": False, "error": f"{_SSH_KEY_ENV} is not set in .env"}
    owner, address, err = _resolve_dialable(host_id)
    if err is not None:
        return err
    return _run_ssh(owner, address, verb, use_key=True)


async def _poll_health(host_id: str) -> Dict[str, Any]:
    owner = get_host(host_id)
    address = await remote_stats.dial_address_async(owner) if owner is not None else None
    if owner is None or not address:
        return {"reachable": False, "error": f"host {host_id!r} has no address configured"}
    base = f"http://{address}:{hub_port()}"
    deadline = time.monotonic() + _HEALTH_POLL_TIMEOUT_S
    last_error = ""
    async with httpx.AsyncClient(timeout=3.0) as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(f"{base}/health")
                if r.status_code < 500:
                    return {"reachable": True, "address": base}
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(_HEALTH_POLL_INTERVAL_S)
    return {"reachable": False, "address": base, "error": last_error or "timed out"}


async def _trigger_and_poll(host_id: str, verb: str, emoji: str) -> Dict[str, Any]:
    """Send ``verb`` over the forced-command SSH channel, then poll
    ``/health`` for up to ~30s. ``ok`` is only true once the peer actually
    answers. Shared by ``bootstrap_host``/``sync_host`` — same shape apart
    from the verb and the log line's icon."""
    logger.info("%s %s_host(%r)", emoji, verb, host_id)
    ssh_result = await asyncio.to_thread(_run_remote_command, host_id, verb)
    if not ssh_result["ok"]:
        return {"ok": False, "detail": ssh_result["error"]}
    health = await _poll_health(host_id)
    return {"ok": health["reachable"], "detail": health}


async def bootstrap_host(host_id: str) -> Dict[str, Any]:
    """Trigger the remote host's ``bootstrap`` action, then poll ``/health``
    for up to ~30s. ``ok`` is only true once the peer actually answers."""
    return await _trigger_and_poll(host_id, "bootstrap", "🛎️")


async def sync_host(host_id: str) -> Dict[str, Any]:
    """Trigger the remote host's ``sync`` action (git pull + restart), then
    poll ``/health`` for up to ~30s."""
    return await _trigger_and_poll(host_id, "sync", "🔃")


def _run_power_command(host_id: str, flag: str) -> Dict[str, Any]:
    """Run reboot/shutdown over the hub user's **own** general SSH (#311).

    The same passwordless channel ``remote_stats`` uses for read-only
    snapshots — ``ssh <user>@<addr> "<cmd>"`` with no ``-i`` forced-command
    key — so a managed peer (e.g. OpenClaw) needs nothing deployed beyond the
    passwordless-sudo sudoers drop-in it already carries. ``flag`` is
    ``shutdown``'s ``-r`` (reboot) or ``-h`` (halt/power-off).

    The actual ``shutdown`` drops the SSH connection the instant the box goes
    down, which would race this command's own exit and surface as a spurious
    ssh failure. So — exactly as the old forced-command dispatcher did — we
    detach a short-delayed ``shutdown`` with ``nohup`` (survives the closing
    SSH channel) and let the remote command return cleanly; the box powers
    down/reboots ~2 s later."""
    owner, address, err = _resolve_dialable(host_id)
    if err is not None:
        return err
    remote = f"nohup sh -c 'sleep 2; sudo -n /sbin/shutdown {flag} now' >/dev/null 2>&1 &"
    return _run_ssh(owner, address, remote, use_key=False)


async def _trigger_power(host_id: str, flag: str, verb: str, emoji: str) -> Dict[str, Any]:
    """Send a ``reboot``/``shutdown`` over the general-SSH power channel (#311).

    Unlike ``bootstrap``/``sync`` there is **no** ``/health`` poll — the
    machine is going *down*, so "reachable" is the wrong success signal. A
    clean SSH exit (the detached power action was scheduled) is the
    confirmation; the box drops off the network a couple of seconds later.
    Refusing the destructive action to the local host is the caller's job (the
    router knows which host is active); this layer already can't reach a host
    with no ``ssh_user``."""
    logger.info("%s %s_host(%r)", emoji, verb, host_id)
    ssh_result = await asyncio.to_thread(_run_power_command, host_id, flag)
    if not ssh_result["ok"]:
        return {"ok": False, "detail": ssh_result["error"]}
    return {"ok": True, "detail": f"{verb} scheduled on {host_id}"}


async def reboot_host(host_id: str) -> Dict[str, Any]:
    """Reboot a peer over the general-SSH power channel (#309/#311). Destructive
    — the caller must exclude the active hub host before invoking this."""
    return await _trigger_power(host_id, "-r", "reboot", "♻️")


async def shutdown_host(host_id: str) -> Dict[str, Any]:
    """Power off a peer over the general-SSH power channel (#309/#311).
    Destructive — the caller must exclude the active hub host."""
    return await _trigger_power(host_id, "-h", "shutdown", "⏻")
