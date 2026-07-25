"""Unit tests for src/fleet_reconcile.py (issue #353).

Covers the reconcile contract without any real network or process control:
reachable-remote starts every placed model, an unreachable can-ssh host is
woken, an already-running model is a benign no-op, the additive pass never
stops anything, and an explicit un-place stops + de-profiles.
"""

from __future__ import annotations

import asyncio
import os

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

from src import backend_process as bp  # noqa: E402
from src import fleet_maintenance, fleet_placement, fleet_reconcile as fr  # noqa: E402
from src import remote_bootstrap, services, startup_profile  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _stub_peer_transport(monkeypatch, calls):
    """Record start/stop/profile-writes instead of hitting a peer hub."""
    async def write_profile(host_id, base, models):
        calls.append(("profile", host_id, tuple(models)))
        return {"ok": True, "status": 200}

    async def model_action(host_id, base, model_id, action):
        calls.append((action, host_id, model_id))
        return {"ok": True, "status": 200}

    monkeypatch.setattr(fr, "_remote_write_profile", write_profile)
    monkeypatch.setattr(fr, "_remote_model_action", model_action)


def _stub_wol(monkeypatch, sent, *, fail=False):
    """Record magic packets instead of broadcasting real UDP (#364). Registry
    hosts now carry real MACs, so any unreachable-branch test would otherwise
    put actual wake packets on the LAN."""
    def fake_send_wake(mac):
        if fail:
            raise fr.WakeOnLanError(f"boom sending to {mac}")
        sent.append(mac)

    monkeypatch.setattr(fr, "send_wake", fake_send_wake)


# --------------------------------------------------------------------------- #
# reconcile_once — additive convergence
# --------------------------------------------------------------------------- #
def test_reachable_remote_starts_every_placed_model(monkeypatch, tmp_path):
    calls: list = []
    _stub_peer_transport(monkeypatch, calls)
    monkeypatch.setattr(fleet_placement, "load_fleet_placement",
                        lambda: {"mac-mini-m4": ["parakeet", "qwen"]})
    monkeypatch.setattr(services, "peer_health", _async_ret({"reachable": True}))

    results = _run(fr.reconcile_once())

    starts = [c for c in calls if c[0] == "start"]
    assert {c[2] for c in starts} == {"parakeet", "qwen"}
    assert ("profile", "mac-mini-m4", ("parakeet", "qwen")) in calls
    assert results["mac-mini-m4"]["reachable"] is True
    # additive: never a stop
    assert not [c for c in calls if c[0] == "stop"]


def test_unreachable_can_ssh_host_is_woken(monkeypatch, tmp_path):
    calls: list = []
    _stub_peer_transport(monkeypatch, calls)
    _stub_wol(monkeypatch, [])
    woke = {"woke": []}
    monkeypatch.setattr(fleet_placement, "load_fleet_placement",
                        lambda: {"mac-mini-m4": ["parakeet"]})
    monkeypatch.setattr(services, "peer_health", _async_ret({"reachable": False}))

    async def fake_bootstrap(host_id):
        woke["woke"].append(host_id)
        return {"ok": False}  # stayed down this pass

    monkeypatch.setattr(remote_bootstrap, "bootstrap_host", fake_bootstrap)

    results = _run(fr.reconcile_once())

    assert woke["woke"] == ["mac-mini-m4"]        # a wake was attempted
    assert results["mac-mini-m4"]["reachable"] is False
    assert not [c for c in calls if c[0] == "start"]  # no start while down


def test_woken_host_converges_in_same_pass(monkeypatch, tmp_path):
    calls: list = []
    _stub_peer_transport(monkeypatch, calls)
    _stub_wol(monkeypatch, [])
    monkeypatch.setattr(fleet_placement, "load_fleet_placement",
                        lambda: {"mac-mini-m4": ["parakeet"]})
    monkeypatch.setattr(services, "peer_health", _async_ret({"reachable": False}))
    monkeypatch.setattr(remote_bootstrap, "bootstrap_host", _async_ret({"ok": True}))

    _run(fr.reconcile_once())

    assert ("start", "mac-mini-m4", "parakeet") in calls  # started after wake


# --------------------------------------------------------------------------- #
# WOL before the SSH bootstrap (#364 — Phase 2 of #356)
# --------------------------------------------------------------------------- #
def test_unreachable_mac_host_gets_wol_then_bootstrap_same_pass(monkeypatch):
    calls: list = []
    _stub_peer_transport(monkeypatch, calls)
    sent: list = []
    _stub_wol(monkeypatch, sent)
    bootstraps: list = []
    monkeypatch.setattr(fleet_placement, "load_fleet_placement",
                        lambda: {"mac-mini-m4": ["parakeet"]})
    monkeypatch.setattr(services, "peer_health", _async_ret({"reachable": False}))

    async def fake_bootstrap(host_id):
        bootstraps.append(host_id)
        return {"ok": False}  # cold box: SSH can't reach it this pass

    monkeypatch.setattr(remote_bootstrap, "bootstrap_host", fake_bootstrap)

    results = _run(fr.reconcile_once())

    assert len(sent) == 1                                  # exactly one packet
    assert bootstraps == ["mac-mini-m4"]                   # bootstrap still tried
    assert results["mac-mini-m4"]["wol_sent"] is True
    assert results["mac-mini-m4"]["reachable"] is False    # fire-and-continue


def test_unreachable_macless_host_sends_no_wol(monkeypatch):
    calls: list = []
    _stub_peer_transport(monkeypatch, calls)
    sent: list = []
    _stub_wol(monkeypatch, sent)
    monkeypatch.setattr(fleet_placement, "load_fleet_placement",
                        lambda: {"openclaw": ["parakeet"]})  # no wired NIC, no mac
    monkeypatch.setattr(services, "peer_health", _async_ret({"reachable": False}))
    monkeypatch.setattr(remote_bootstrap, "bootstrap_host", _async_ret({"ok": False}))

    results = _run(fr.reconcile_once())

    assert sent == []                                      # nothing to send
    assert results["openclaw"]["wol_sent"] is False
    assert results["openclaw"]["reachable"] is False


def test_wol_send_failure_is_swallowed_and_pass_continues(monkeypatch):
    calls: list = []
    _stub_peer_transport(monkeypatch, calls)
    _stub_wol(monkeypatch, [], fail=True)
    bootstraps: list = []
    monkeypatch.setattr(fleet_placement, "load_fleet_placement",
                        lambda: {"mac-mini-m4": ["parakeet"]})
    monkeypatch.setattr(services, "peer_health", _async_ret({"reachable": False}))

    async def fake_bootstrap(host_id):
        bootstraps.append(host_id)
        return {"ok": True}  # box was actually just hub-dead, SSH worked

    monkeypatch.setattr(remote_bootstrap, "bootstrap_host", fake_bootstrap)

    results = _run(fr.reconcile_once())

    assert bootstraps == ["mac-mini-m4"]                   # failure didn't abort
    assert results["mac-mini-m4"]["wol_sent"] is False
    assert results["mac-mini-m4"]["reachable"] is True     # converged via SSH


def test_empty_placement_host_is_skipped(monkeypatch):
    calls: list = []
    _stub_peer_transport(monkeypatch, calls)
    probed = {"n": 0}

    async def health(host_id):
        probed["n"] += 1
        return {"reachable": True}

    monkeypatch.setattr(fleet_placement, "load_fleet_placement", lambda: {"mac-mini-m4": []})
    monkeypatch.setattr(services, "peer_health", health)

    results = _run(fr.reconcile_once())
    assert results == {}          # nothing placed → nothing converged
    assert probed["n"] == 0       # and no reason to even probe it


def test_local_already_running_is_noop_success(monkeypatch):
    monkeypatch.setattr(fleet_placement, "load_fleet_placement", lambda: {"tower": ["whisper"]})
    monkeypatch.setattr(bp, "start", lambda mid: (False, "backend already running"))
    stops: list = []
    monkeypatch.setattr(bp, "stop", lambda mid: stops.append(mid) or (True, "stopped"))

    results = _run(fr.reconcile_once())

    entry = results["tower"]["started"][0]
    assert entry["id"] == "whisper" and entry["ok"] is True  # already-running = ok
    assert stops == []  # additive pass never stops


# --------------------------------------------------------------------------- #
# Maintenance gate (#411) — reconcile skips a drained host entirely
# --------------------------------------------------------------------------- #
def test_maintained_host_is_skipped_entirely(monkeypatch):
    """Reproduction proof: before #411, a host under maintenance still
    converged normally — reconcile would wake/probe/start it, racing
    model_failover's fail_after_s window exactly as the issue describes."""
    calls: list = []
    _stub_peer_transport(monkeypatch, calls)
    sent: list = []
    _stub_wol(monkeypatch, sent)
    probed = {"n": 0}

    async def health(host_id):
        probed["n"] += 1
        return {"reachable": False}

    monkeypatch.setattr(fleet_placement, "load_fleet_placement",
                        lambda: {"mac-mini-m4": ["parakeet"]})
    monkeypatch.setattr(services, "peer_health", health)
    monkeypatch.setattr(remote_bootstrap, "bootstrap_host", _async_ret({"ok": True}))
    monkeypatch.setattr(fleet_maintenance, "is_under_maintenance", lambda host_id: True)

    results = _run(fr.reconcile_once())

    assert results["mac-mini-m4"] == {"maintenance": True, "reachable": None}
    assert probed["n"] == 0                                # never even probed
    assert sent == []                                      # no WOL
    assert not [c for c in calls if c[0] in ("start", "profile")]  # no convergence


def test_expired_maintenance_host_converges_normally(monkeypatch):
    calls: list = []
    _stub_peer_transport(monkeypatch, calls)
    monkeypatch.setattr(fleet_placement, "load_fleet_placement",
                        lambda: {"mac-mini-m4": ["parakeet"]})
    monkeypatch.setattr(services, "peer_health", _async_ret({"reachable": True}))
    # A real (non-monkeypatched) call against an empty maintenance file — the
    # host has no marker at all, i.e. the equivalent of an expired one.
    monkeypatch.setattr(fleet_maintenance, "load_fleet_maintenance", lambda path=None: {})

    results = _run(fr.reconcile_once())

    assert results["mac-mini-m4"]["reachable"] is True
    assert ("start", "mac-mini-m4", "parakeet") in calls


# --------------------------------------------------------------------------- #
# apply_placement_change — explicit un-place stops + de-profiles
# --------------------------------------------------------------------------- #
def test_unplace_local_stops_and_deprofiles(monkeypatch):
    stopped: list = []
    monkeypatch.setattr(bp, "stop", lambda mid: stopped.append(mid) or (True, "stopped"))
    monkeypatch.setattr(bp, "start", lambda mid: (True, "started"))

    profile_saves: list = []
    monkeypatch.setattr(startup_profile, "load_startup_profile",
                        lambda: startup_profile.StartupProfile(models=["whisper", "piper"]))
    monkeypatch.setattr(startup_profile, "save_startup_profile",
                        lambda data, path=None: profile_saves.append(data) or None)

    result = _run(fr.apply_placement_change("tower", ["whisper", "piper"], ["piper"], "tower"))

    assert stopped == ["whisper"]                       # removed model stopped
    assert profile_saves[0]["models"] == ["piper"]      # dropped from local profile
    assert result["stopped"][0]["id"] == "whisper"


def test_unplace_remote_stops_via_peer(monkeypatch):
    calls: list = []
    _stub_peer_transport(monkeypatch, calls)
    monkeypatch.setattr(services, "peer_health", _async_ret({"reachable": True}))

    _run(fr.apply_placement_change("mac-mini-m4", ["parakeet", "qwen"], ["qwen"], "tower"))

    assert ("stop", "mac-mini-m4", "parakeet") in calls   # removed stopped on peer
    assert ("start", "mac-mini-m4", "qwen") in calls      # survivor converged


def test_unplace_last_remote_model_deprofiles_peer(monkeypatch):
    # #360: emptying a remote host's placement must still write the profile
    # through (models: []) so the un-placed model can't resurrect on reboot.
    calls: list = []
    _stub_peer_transport(monkeypatch, calls)
    monkeypatch.setattr(services, "peer_health", _async_ret({"reachable": True}))

    result = _run(fr.apply_placement_change("mac-mini-m4", ["parakeet"], [], "tower"))

    assert ("stop", "mac-mini-m4", "parakeet") in calls
    assert ("profile", "mac-mini-m4", ()) in calls        # empty write-through sent
    assert result["converged"]["profile_written"] is True
    assert not [c for c in calls if c[0] == "start"]


def test_unplace_last_remote_model_soft_fails_when_peer_down(monkeypatch):
    # Unreachable peer: the un-place itself must not error; no profile PATCH,
    # no wake/bootstrap attempt — the stale entry waits for the peer's return.
    calls: list = []
    _stub_peer_transport(monkeypatch, calls)
    monkeypatch.setattr(services, "peer_health", _async_ret({"reachable": False}))

    result = _run(fr.apply_placement_change("mac-mini-m4", ["parakeet"], [], "tower"))

    assert not [c for c in calls if c[0] == "profile"]
    assert result["converged"] == {"reachable": False, "profile_written": False}


def _async_ret(value):
    async def _f(*args, **kwargs):
        return value
    return _f
