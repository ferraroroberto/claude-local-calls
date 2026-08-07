"""Tailscale MagicDNS fallback when a peer's LAN address dies (#396).

The 2026-07-23 network cleanup put every peer on a wired NIC with a fixed
DHCP reservation and kept Wi-Fi as an *unreserved* fallback — so a wired
failure moves the box to a pool address, and every hub touchpoint dialing the
raw LAN IP goes dark exactly when the fallback kicks in. ``remote_stats``
grew one central dial resolver — the LAN ``address:`` first, the host's
``tailscale:`` name on connect failure, with a short-TTL last-known-good per
host — and the peer-connect paths (model-proxy upstream, SSH ops, remote
stats/liveness) plus the Machines card's "via tailnet" badge all ride it.

Everything here drives the resolver through stubbed port probes (the global
conftest fixture already pins ``_probe_port`` to "nothing answers"), so no
test touches a real socket.
"""

from __future__ import annotations

import asyncio
import logging

from src import machine_console as mc
from src import remote_stats, ssh_exec
from src.host_profile import HostProfile, get_host


def _run(coro):
    """Run a coroutine on a fresh worker-thread loop (see test_machines_router)."""
    import threading

    bucket: dict = {}

    def _worker() -> None:
        loop = asyncio.new_event_loop()
        try:
            bucket["value"] = loop.run_until_complete(coro)
        except BaseException as exc:  # noqa: BLE001 — re-raised in caller
            bucket["error"] = exc
        finally:
            loop.close()

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    if "error" in bucket:
        raise bucket["error"]
    return bucket.get("value")


_LAN = "192.168.0.99"
_TS = "peer.tail1121fd.ts.net"


def _peer(**kwargs) -> HostProfile:
    return HostProfile(
        id="peer", platform="linux", enabled=[], address=_LAN, tailscale=_TS, **kwargs
    )


def _probe_stub(monkeypatch, alive: set):
    """Patch the port scan to answer only for addresses in ``alive``; returns
    the ordered list of addresses actually probed."""
    probed: list = []

    def fake_scan(address):
        probed.append(address)
        return address in alive

    monkeypatch.setattr(remote_stats, "_probe_liveness_ports", fake_scan)
    return probed


# ------------------------------------------------------------- dial resolution


def test_dial_without_tailscale_is_passthrough_with_zero_probes(monkeypatch):
    """A host with no ``tailscale:`` field behaves exactly as before #396 —
    its LAN address, and not a single probe paid for it."""
    probed = _probe_stub(monkeypatch, alive=set())
    host = HostProfile(id="plain", platform="linux", enabled=[], address=_LAN)
    assert remote_stats.dial_address(host, wait=True) == _LAN
    assert probed == []


def test_dial_without_any_address_is_none(monkeypatch):
    probed = _probe_stub(monkeypatch, alive=set())
    host = HostProfile(id="nowhere", platform="linux", enabled=[])
    assert remote_stats.dial_address(host) is None
    assert probed == []


def test_dial_prefers_lan_while_it_answers(monkeypatch):
    """On-LAN behavior unchanged when the wire is healthy: the LAN address
    wins and the tailnet name is never even probed (no WireGuard hop)."""
    probed = _probe_stub(monkeypatch, alive={_LAN})
    assert remote_stats.dial_address(_peer(), wait=True) == _LAN
    assert probed == [_LAN]  # tailnet candidate untouched


def test_dial_falls_back_to_tailnet_when_lan_dead(monkeypatch, caplog):
    """LAN dead → the tailscale name is tried and wins, with an info-level
    breadcrumb naming the host and both addresses."""
    _probe_stub(monkeypatch, alive={_TS})
    with caplog.at_level(logging.INFO, logger="src.remote_stats"):
        assert remote_stats.dial_address(_peer(), wait=True) == _TS
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "falling back to tailnet" in m and "peer" in m and _LAN in m and _TS in m
        for m in messages
    ), messages


def test_dial_both_dead_returns_lan_primary(monkeypatch, caplog):
    """Both paths dead → the LAN primary comes back so callers fail with
    exactly today's connect errors, and no failover is (mis)logged."""
    _probe_stub(monkeypatch, alive=set())
    with caplog.at_level(logging.INFO, logger="src.remote_stats"):
        assert remote_stats.dial_address(_peer(), wait=True) == _LAN
    assert not any("tailnet" in r.getMessage() for r in caplog.records)


def test_dial_caches_last_known_good_within_ttl(monkeypatch):
    """The winner is a last-known-good with a short TTL: within it, repeat
    dials are dict lookups (no re-probe); past it, the LAN path is re-tried
    so recovery back to the wire is automatic."""
    probed = _probe_stub(monkeypatch, alive={_TS})
    clock = {"now": 1000.0}
    monkeypatch.setattr(remote_stats.time, "monotonic", lambda: clock["now"])
    host = _peer()
    assert remote_stats.dial_address(host, wait=True) == _TS
    assert remote_stats.dial_address(host, wait=True) == _TS  # cache hit
    assert probed == [_LAN, _TS]  # exactly one probe pass paid
    clock["now"] += remote_stats._DIAL_TTL_S + 0.1
    assert remote_stats.dial_address(host, wait=True) == _TS
    assert probed == [_LAN, _TS, _LAN, _TS]  # expired — LAN re-tried first


def test_dial_recovery_back_to_lan_logs_once(monkeypatch, caplog):
    """When the wire comes back the resolver returns to the LAN address and
    logs the recovery — once per transition, not once per probe."""
    alive = {_TS}
    _probe_stub(monkeypatch, alive=alive)
    clock = {"now": 1000.0}
    monkeypatch.setattr(remote_stats.time, "monotonic", lambda: clock["now"])
    host = _peer()
    with caplog.at_level(logging.INFO, logger="src.remote_stats"):
        assert remote_stats.dial_address(host, wait=True) == _TS  # outage: on tailnet
        alive.add(_LAN)  # wire restored
        clock["now"] += remote_stats._DIAL_TTL_S + 0.1
        assert remote_stats.dial_address(host, wait=True) == _LAN
        clock["now"] += remote_stats._DIAL_TTL_S + 0.1
        assert remote_stats.dial_address(host, wait=True) == _LAN  # steady state
    recoveries = [r for r in caplog.records if "answers again" in r.getMessage()]
    assert len(recoveries) == 1


def test_dial_reuses_fresh_liveness_result(monkeypatch):
    """A fresh liveness probe already knows which address answers — the dial
    path reuses it instead of paying its own probe (the #369 cache and the
    #396 resolver cooperate, they don't fight)."""

    def boom(address):  # pragma: no cover — must never run
        raise AssertionError("dial must reuse the fresh liveness winner")

    monkeypatch.setattr(remote_stats, "_probe_liveness_ports", boom)
    now = remote_stats.time.monotonic()
    remote_stats._liveness_cache["peer"] = (now + 5.0, _TS)
    assert remote_stats.dial_address(_peer(), wait=True) == _TS


def test_dial_default_mode_never_probes_inline(monkeypatch):
    """``wait=False`` (the event-loop mode) must not block: a cold cache
    returns the LAN best-guess immediately and hands the probe to a background
    refresh, which pins the tailnet route for the *next* dial. This is the
    regression guard for the first cut of #396, whose inline probe of a dark
    address stalled every unrelated request on the loop."""
    probed = _probe_stub(monkeypatch, alive={_TS})
    kicked = []
    monkeypatch.setattr(remote_stats, "_kick_refresh", lambda h: kicked.append(h.id))
    host = _peer()
    assert remote_stats.dial_address(host) == _LAN  # instant best-guess
    assert probed == []                             # nothing probed inline
    assert kicked == ["peer"]                       # probe handed to the background
    remote_stats._refresh_route(host)               # what the background thread runs
    assert probed == [_LAN, _TS]
    assert remote_stats.dial_address(host) == _TS   # next dial rides the new route


# --------------------------------------------------------- liveness via tailnet


def test_locate_finds_tailnet_only_peer(monkeypatch):
    """A box whose wire died but whose tailnet is alive is still *on* — the
    liveness probe reports the tailscale name instead of masking the box as
    down."""
    _probe_stub(monkeypatch, alive={_TS})
    assert remote_stats.locate(_peer()) == _TS
    assert remote_stats.reachable(_peer()) is True


def test_locate_none_when_every_path_dead(monkeypatch):
    _probe_stub(monkeypatch, alive=set())
    monkeypatch.setattr(remote_stats.time, "sleep", lambda *_: None)
    assert remote_stats.locate(_peer()) is None
    assert remote_stats.reachable(_peer()) is False


# ------------------------------------------------- consumers ride the resolver


def test_remote_base_url_for_host_dials_the_winner(monkeypatch):
    """The model-proxy upstream resolves through the dial resolver — a peer on
    tailnet fallback gets proxied at its MagicDNS name."""
    from src import remote_proxy

    monkeypatch.setattr(
        remote_stats, "dial_address", lambda host, **kw: "mac-mini.tail1121fd.ts.net"
    )
    url = remote_proxy.remote_base_url_for_host("mac-mini-m4")
    assert url == "http://mac-mini.tail1121fd.ts.net:8000"


def test_run_ssh_targets_the_dial_winner(monkeypatch):
    """The remote-stats SSH snapshot dials the resolved address, not the raw
    LAN field."""
    captured = {}

    class _Done:
        returncode = 0
        stdout = "uptime 1\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _Done()

    monkeypatch.setattr(remote_stats, "dial_address", lambda host, **kw: _TS)
    monkeypatch.setattr(ssh_exec.subprocess, "run", fake_run)
    host = HostProfile(
        id="peer", platform="linux", enabled=[], address=_LAN, tailscale=_TS,
        ssh_user="pilot",
    )
    assert remote_stats._run_ssh(host, "echo hi") == "uptime 1\n"
    assert f"pilot@{_TS}" in captured["cmd"]


def test_power_command_dials_tailnet_when_lan_dead(monkeypatch):
    """SSH power actions survive a wired failure: with the resolver reporting
    the tailnet route, reboot/shutdown target ``user@<tailscale-name>``."""
    from src import remote_bootstrap

    captured = {}

    class _Done:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _Done()

    monkeypatch.setattr(
        remote_stats, "dial_address", lambda host, **kw: "mac-mini.tail1121fd.ts.net"
    )
    monkeypatch.setattr(ssh_exec.subprocess, "run", fake_run)
    result = remote_bootstrap._run_power_command("mac-mini-m4", "-r")
    assert result["ok"] is True
    assert "roberto@mac-mini.tail1121fd.ts.net" in captured["cmd"]


def test_probe_machine_badges_via_tailnet(monkeypatch):
    """The Machines card carries ``via_tailscale: True`` when the liveness
    winner is the tailscale name — the wired failure is visible, not masked."""
    host = get_host("gaming")

    async def _located(h):
        return h.tailscale

    async def _fake_collect(h):
        return {"uptime_seconds": 42}

    monkeypatch.setattr(mc.remote_stats, "located_address", _located)
    monkeypatch.setattr(mc.remote_stats, "collect", _fake_collect)
    card = _run(mc._probe_machine(host, "tower"))
    assert card["state"] == "up"
    assert card["via_tailscale"] is True
