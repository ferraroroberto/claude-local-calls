"""Unit tests for src/fleet_reconcile.py (issues #353, #430).

Covers the reconcile contract without any real network or process control:
the desired state comes from the registry (``model_registry.desired_placement``
— monkeypatched here), a reachable remote starts every desired model, an
unreachable can-ssh host is woken, an already-running model is a benign
no-op, and the additive pass never stops anything. The derivation itself
(eager/on_demand/chain rules) is unit-tested in ``test_model_registry.py``.
"""

from __future__ import annotations

import asyncio
import os

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

from src import backend_process as bp  # noqa: E402
from src import fleet_maintenance, fleet_reconcile as fr, model_registry  # noqa: E402
from src import remote_bootstrap, services  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _stub_desired(monkeypatch, placement):
    """Pin the registry-derived desired placement (#430)."""
    monkeypatch.setattr(model_registry, "desired_placement", lambda: placement)


def _stub_peer_transport(monkeypatch, calls):
    """Record start/stop actions instead of hitting a peer hub."""
    async def model_action(host_id, base, model_id, action):
        calls.append((action, host_id, model_id))
        return {"ok": True, "status": 200}

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
def test_reachable_remote_starts_every_desired_model(monkeypatch):
    calls: list = []
    _stub_peer_transport(monkeypatch, calls)
    _stub_desired(monkeypatch, {"mac-mini-m4": ["parakeet", "qwen"]})
    monkeypatch.setattr(services, "peer_health", _async_ret({"reachable": True}))

    results = _run(fr.reconcile_once())

    starts = [c for c in calls if c[0] == "start"]
    assert {c[2] for c in starts} == {"parakeet", "qwen"}
    assert results["mac-mini-m4"]["reachable"] is True
    # additive: never a stop
    assert not [c for c in calls if c[0] == "stop"]


def test_unreachable_can_ssh_host_is_woken(monkeypatch):
    calls: list = []
    _stub_peer_transport(monkeypatch, calls)
    _stub_wol(monkeypatch, [])
    woke = {"woke": []}
    _stub_desired(monkeypatch, {"mac-mini-m4": ["parakeet"]})
    monkeypatch.setattr(services, "peer_health", _async_ret({"reachable": False}))

    async def fake_bootstrap(host_id):
        woke["woke"].append(host_id)
        return {"ok": False}  # stayed down this pass

    monkeypatch.setattr(remote_bootstrap, "bootstrap_host", fake_bootstrap)

    results = _run(fr.reconcile_once())

    assert woke["woke"] == ["mac-mini-m4"]        # a wake was attempted
    assert results["mac-mini-m4"]["reachable"] is False
    assert not [c for c in calls if c[0] == "start"]  # no start while down


def test_woken_host_converges_in_same_pass(monkeypatch):
    calls: list = []
    _stub_peer_transport(monkeypatch, calls)
    _stub_wol(monkeypatch, [])
    _stub_desired(monkeypatch, {"mac-mini-m4": ["parakeet"]})
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
    _stub_desired(monkeypatch, {"mac-mini-m4": ["parakeet"]})
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
    _stub_desired(monkeypatch, {"openclaw": ["parakeet"]})  # no wired NIC, no mac
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
    _stub_desired(monkeypatch, {"mac-mini-m4": ["parakeet"]})
    monkeypatch.setattr(services, "peer_health", _async_ret({"reachable": False}))

    async def fake_bootstrap(host_id):
        bootstraps.append(host_id)
        return {"ok": True}  # box was actually just hub-dead, SSH worked

    monkeypatch.setattr(remote_bootstrap, "bootstrap_host", fake_bootstrap)

    results = _run(fr.reconcile_once())

    assert bootstraps == ["mac-mini-m4"]                   # failure didn't abort
    assert results["mac-mini-m4"]["wol_sent"] is False
    assert results["mac-mini-m4"]["reachable"] is True     # converged via SSH


def test_empty_desired_host_is_skipped(monkeypatch):
    calls: list = []
    _stub_peer_transport(monkeypatch, calls)
    probed = {"n": 0}

    async def health(host_id):
        probed["n"] += 1
        return {"reachable": True}

    # desired_placement() omits empty hosts by construction; an explicit empty
    # entry must still be skipped without probing.
    _stub_desired(monkeypatch, {"mac-mini-m4": []})
    monkeypatch.setattr(services, "peer_health", health)

    results = _run(fr.reconcile_once())
    assert results == {}          # nothing desired → nothing converged
    assert probed["n"] == 0       # and no reason to even probe it


def test_local_already_running_is_noop_success(monkeypatch):
    _stub_desired(monkeypatch, {"tower": ["whisper"]})
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

    _stub_desired(monkeypatch, {"mac-mini-m4": ["parakeet"]})
    monkeypatch.setattr(services, "peer_health", health)
    monkeypatch.setattr(remote_bootstrap, "bootstrap_host", _async_ret({"ok": True}))
    monkeypatch.setattr(fleet_maintenance, "is_under_maintenance", lambda host_id: True)

    results = _run(fr.reconcile_once())

    assert results["mac-mini-m4"] == {"maintenance": True, "reachable": None}
    assert probed["n"] == 0                                # never even probed
    assert sent == []                                      # no WOL
    assert not [c for c in calls if c[0] == "start"]       # no convergence


def test_expired_maintenance_host_converges_normally(monkeypatch):
    calls: list = []
    _stub_peer_transport(monkeypatch, calls)
    _stub_desired(monkeypatch, {"mac-mini-m4": ["parakeet"]})
    monkeypatch.setattr(services, "peer_health", _async_ret({"reachable": True}))
    # A real (non-monkeypatched) call against an empty maintenance file — the
    # host has no marker at all, i.e. the equivalent of an expired one.
    monkeypatch.setattr(fleet_maintenance, "load_fleet_maintenance", lambda path=None: {})

    results = _run(fr.reconcile_once())

    assert results["mac-mini-m4"]["reachable"] is True
    assert ("start", "mac-mini-m4", "parakeet") in calls


def _async_ret(value):
    async def _f(*args, **kwargs):
        return value
    return _f
