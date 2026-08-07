"""Dynamic model fallback (#342): chain schema, ownership, hysteresis, actions.

Everything here runs against fakes — a temp models.yaml, a dict-driven
probe, recorder start/stop callables, and an injected clock. Nothing
touches a real peer, port, or process; the live failover drill is a
deliberate manual exercise (see the issue), not a unit concern.
"""

from __future__ import annotations

import asyncio

import pytest

from src import backend_process, model_failover, model_registry
from src.model_failover import (
    POLICY_STICKY,
    FailoverSettings,
    OwnershipTracker,
)


@pytest.fixture(autouse=True)
def _reset_failover_state():
    model_failover.TRACKER.reset()
    model_failover._ENGINE_STARTED.clear()
    yield
    model_failover.TRACKER.reset()
    model_failover._ENGINE_STARTED.clear()


def _chain_config(write_config, **overrides):
    """Three-host fleet with one multi-host model (whisper) and one bare-host
    model (qwen) — the canonical #342 scenario."""
    content = {
        "hub": {"port": 8000},
        "hosts": {
            "gaming": {"platform": "linux", "address": "10.0.0.16",
                       "enabled": ["whisper"]},
            "mac": {"platform": "darwin", "address": "10.0.0.14",
                    "enabled": ["whisper", "qwen"]},
            "tower": {"platform": "win32", "default": True, "address": "10.0.0.13",
                      "enabled": ["whisper", "qwen"]},
        },
        "models": {
            "whisper": {
                "display_name": "whisper-large-v3-turbo",
                "backend": "whisper", "engine": "whisper-server", "port": 8090,
                "model_path": "models/ggml-large-v3-turbo.bin",
                "hosts": ["gaming", "mac", {"id": "tower", "cpu": True}],
                "args": ["--threads", "4"],
            },
            "qwen": {
                "display_name": "qwen3.5-9b", "backend": "openai",
                "engine": "llama-server", "port": 8081, "host": "mac",
                "model_path": "models/q.gguf",
            },
        },
    }
    content.update(overrides)
    return write_config(content)


# --------------------------------------------------------------------------- #
# Schema: hosts chain parsing (registry layer)
# --------------------------------------------------------------------------- #
def test_bare_host_parses_as_one_element_chain(monkeypatch, write_config):
    _chain_config(write_config)
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "tower")
    qwen = model_registry.resolve("qwen3.5-9b")
    assert qwen.host == "mac"
    assert qwen.host_chain == ["mac"]
    assert qwen.cpu_hosts == []


def test_hosts_list_parses_ordered_chain_with_cpu_flag(monkeypatch, write_config):
    _chain_config(write_config)
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "mac")
    w = model_registry.resolve("whisper-large-v3-turbo")
    assert w.host_chain == ["gaming", "mac", "tower"]
    assert w.host == "gaming"          # preferred owner = chain head
    assert w.cpu_hosts == ["tower"]


def test_hosts_wins_over_bare_host_and_dedups(monkeypatch, write_config):
    write_config({
        "hub": {"port": 8000},
        "hosts": {"pc": {"platform": "win32", "default": True, "enabled": ["m"]}},
        "models": {
            "m": {"display_name": "m", "backend": "openai", "port": 8081,
                  "host": "ignored", "hosts": ["a", "b", "a"]},
        },
    })
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "pc")
    m = model_registry.all_models()[0]
    assert m.host_chain == ["a", "b"]
    assert m.host == "a"


def test_unowned_row_has_empty_chain(monkeypatch, write_config):
    write_config({
        "hub": {"port": 8000},
        "hosts": {"pc": {"platform": "win32", "default": True, "enabled": ["m"]}},
        "models": {"m": {"display_name": "m", "backend": "openai", "port": 8081}},
    })
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "pc")
    m = model_registry.resolve("m")
    assert m.host is None
    assert m.host_chain == []


# --------------------------------------------------------------------------- #
# CPU-offload degraded tier
# --------------------------------------------------------------------------- #
def test_cpu_offload_args_rewrites_per_engine():
    f = model_registry.cpu_offload_args
    assert f("llama-server", ["--jinja", "-ngl", "99", "-c", "4096"]) == \
        ["--jinja", "-ngl", "0", "-c", "4096"]
    assert f("llama-server", []) == ["-ngl", "0"]
    assert f("whisper-server", ["--threads", "4"]) == ["--threads", "4", "-ng"]
    assert f("whisper-server", ["-ng"]) == ["-ng"]           # already CPU-only
    assert f("whisper-server-lazy", []) == ["-ng"]
    assert f("tts-server", ["--device", "auto"]) == ["--device", "cpu"]
    assert f("tts-server", []) == ["--device", "cpu"]
    assert f("parakeet-server", ["x"]) == ["x"]              # no notion of it
    assert f(None, ["x"]) == ["x"]


def test_cpu_flag_bakes_args_only_on_flagged_host(monkeypatch, write_config):
    _chain_config(write_config)
    # tower is the cpu-flagged last resort: its whisper args gain -ng.
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "tower")
    w = model_registry.resolve("whisper-large-v3-turbo")
    assert "-ng" in w.args
    # gaming (preferred, GPU) keeps the row's args untouched.
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "gaming")
    w = model_registry.resolve("whisper-large-v3-turbo")
    assert "-ng" not in w.args


# --------------------------------------------------------------------------- #
# Chain membership: local_models + start guard
# --------------------------------------------------------------------------- #
def test_local_models_includes_every_chain_member(monkeypatch, write_config):
    _chain_config(write_config)
    for host_id in ("gaming", "mac", "tower"):
        monkeypatch.setenv("LOCAL_LLM_HUB_HOST", host_id)
        ids = {m.id for m in model_registry.local_models()}
        assert "whisper" in ids, host_id
    # qwen (bare host: mac) stays excluded off-owner — pre-#342 behavior.
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "tower")
    assert "qwen" not in {m.id for m in model_registry.local_models()}
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "mac")
    assert "qwen" in {m.id for m in model_registry.local_models()}


def test_start_guard_allows_chain_members_and_refuses_outsiders(monkeypatch, write_config):
    _chain_config(write_config)
    # tower is NOT in qwen's chain (bare host: mac) → refused.
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "tower")
    ok, msg = backend_process.start("qwen")
    assert ok is False and "owned by host(s)" in msg and "mac" in msg

    # mac is a *fallback* member of whisper's chain → the guard passes.
    # is_running is monkeypatched True so start() returns benignly right
    # after the guard, without touching ports or processes.
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "mac")
    monkeypatch.setattr(backend_process, "is_running", lambda mid: True)
    ok, msg = backend_process.start("whisper")
    assert (ok, msg) == (False, "already running")


# --------------------------------------------------------------------------- #
# effective_owner
# --------------------------------------------------------------------------- #
def test_single_host_model_never_consults_tracker(monkeypatch, write_config):
    _chain_config(write_config)
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "tower")

    def _boom(model_id):
        raise AssertionError("tracker consulted for a single-host row")

    monkeypatch.setattr(model_failover.TRACKER, "owner_for", _boom)
    qwen = model_registry.resolve("qwen3.5-9b")
    assert model_failover.effective_owner(qwen) == "mac"


def test_effective_owner_defaults_to_preferred_until_observed(monkeypatch, write_config):
    _chain_config(write_config)
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "tower")
    w = model_registry.resolve("whisper-large-v3-turbo")
    assert model_failover.effective_owner(w) == "gaming"


def test_eligible_chain_drops_unenabled_and_unknown_hosts(monkeypatch, write_config):
    write_config({
        "hub": {"port": 8000},
        "hosts": {
            "a": {"platform": "win32", "default": True, "enabled": ["m"]},
            "b": {"platform": "linux", "enabled": []},          # not enabled there
        },
        "models": {
            "m": {"display_name": "m", "backend": "openai", "port": 8081,
                  "hosts": ["a", "b", "ghost"]},                # ghost: no such host
        },
    })
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "a")
    m = model_registry.resolve("m")
    assert model_failover.eligible_chain(m) == ["a"]
    # A chain that collapses to one eligible candidate is static again.
    assert model_failover.effective_owner(m) == "a"


# --------------------------------------------------------------------------- #
# OwnershipTracker — the anti-flap state machine (pure, injected clock)
# --------------------------------------------------------------------------- #
SETTINGS = FailoverSettings(
    probe_interval_s=30, fail_after_s=90, failback_after_s=600, policy="auto"
)
CHAIN = ["gaming", "mac", "tower"]


def _tracker(policy: str = "auto") -> OwnershipTracker:
    s = FailoverSettings(
        probe_interval_s=30, fail_after_s=90, failback_after_s=600, policy=policy
    )
    return OwnershipTracker(s)


def _observe_all(trk, now, **states):
    for host, up in states.items():
        trk.observe_host(host, up, now)


def test_boot_pick_is_chain_head_without_observations():
    trk = _tracker()
    assert trk.decide("w", CHAIN, now=0) == "gaming"


def test_boot_pick_skips_known_down_head():
    trk = _tracker()
    _observe_all(trk, 0, gaming=False, mac=True, tower=True)
    assert trk.decide("w", CHAIN, now=0) == "mac"


def test_owner_down_briefly_does_not_fail_over():
    trk = _tracker()
    _observe_all(trk, 0, gaming=True, mac=True, tower=True)
    assert trk.decide("w", CHAIN, now=0) == "gaming"
    _observe_all(trk, 100, gaming=False, mac=True, tower=True)
    # 89 s down < fail_after_s=90 → still gaming.
    assert trk.decide("w", CHAIN, now=189) == "gaming"


def test_owner_down_past_window_fails_over_to_next_up():
    trk = _tracker()
    _observe_all(trk, 0, gaming=True, mac=True, tower=True)
    trk.decide("w", CHAIN, now=0)
    _observe_all(trk, 100, gaming=False)
    assert trk.decide("w", CHAIN, now=191) == "mac"


def test_failover_skips_dead_middle_candidate_to_last_resort():
    trk = _tracker()
    _observe_all(trk, 0, gaming=True, mac=False, tower=True)
    trk.decide("w", CHAIN, now=0)
    _observe_all(trk, 100, gaming=False)
    assert trk.decide("w", CHAIN, now=191) == "tower"


def test_no_candidate_up_keeps_current_owner():
    trk = _tracker()
    _observe_all(trk, 0, gaming=True, mac=True, tower=True)
    trk.decide("w", CHAIN, now=0)
    _observe_all(trk, 100, gaming=False, mac=False, tower=False)
    assert trk.decide("w", CHAIN, now=500) == "gaming"


def test_failback_waits_for_stability_window():
    trk = _tracker()
    # Failed over to mac.
    _observe_all(trk, 0, gaming=False, mac=True, tower=True)
    assert trk.decide("w", CHAIN, now=0) == "mac"
    # gaming returns at t=1000; at t=1500 it has been up 500 s < 600 s.
    _observe_all(trk, 1000, gaming=True)
    assert trk.decide("w", CHAIN, now=1500) == "mac"
    # At t=1601 the window is met → hand back.
    assert trk.decide("w", CHAIN, now=1601) == "gaming"


def test_repeatedly_rebooting_host_never_reclaims_ownership():
    """The issue's flap scenario: gaming bounces (up 300 s, down, up 300 s …)
    while failback needs 600 s of *continuous* uptime — ownership must stay
    on mac through every bounce."""
    trk = _tracker()
    _observe_all(trk, 0, gaming=False, mac=True, tower=True)
    assert trk.decide("w", CHAIN, now=0) == "mac"
    t = 1000
    for _ in range(10):                      # ten reboot cycles
        _observe_all(trk, t, gaming=True)    # comes up
        assert trk.decide("w", CHAIN, now=t + 300) == "mac"   # 300 < 600
        _observe_all(trk, t + 301, gaming=False)              # dies again
        assert trk.decide("w", CHAIN, now=t + 320) == "mac"
        t += 400
    # And a genuinely stable return does hand back.
    _observe_all(trk, t, gaming=True)
    assert trk.decide("w", CHAIN, now=t + 601) == "gaming"


def test_sticky_policy_never_hands_back_but_still_fails_over():
    trk = _tracker(policy=POLICY_STICKY)
    _observe_all(trk, 0, gaming=False, mac=True, tower=True)
    assert trk.decide("w", CHAIN, now=0) == "mac"
    # gaming stable for hours — sticky keeps mac.
    _observe_all(trk, 100, gaming=True)
    assert trk.decide("w", CHAIN, now=100_000) == "mac"
    # But when mac itself dies past the window, failover still runs (and the
    # first up candidate — gaming — wins the deterministic tie-break).
    _observe_all(trk, 100_000, mac=False)
    assert trk.decide("w", CHAIN, now=100_000 + 91) == "gaming"


def test_flapping_owner_probe_reset_restarts_down_window():
    """A single missed probe never fails over: the down-window restarts
    whenever the owner answers again."""
    trk = _tracker()
    _observe_all(trk, 0, gaming=True, mac=True, tower=True)
    trk.decide("w", CHAIN, now=0)
    _observe_all(trk, 100, gaming=False)          # missed one probe
    _observe_all(trk, 130, gaming=True)           # back
    _observe_all(trk, 160, gaming=False)          # missed again
    # 80 s after the *latest* down transition < 90 → no failover.
    assert trk.decide("w", CHAIN, now=240) == "gaming"


# --------------------------------------------------------------------------- #
# failover_pass — probe → decide → act, with fakes
# --------------------------------------------------------------------------- #
def _run_pass(**kwargs):
    return asyncio.run(model_failover.failover_pass(**kwargs))


def test_pass_noops_when_no_multi_host_chains(monkeypatch, write_config):
    write_config({
        "hub": {"port": 8000},
        "hosts": {"pc": {"platform": "win32", "default": True, "enabled": ["m"]}},
        "models": {"m": {"display_name": "m", "backend": "openai", "port": 8081,
                         "host": "pc"}},
    })
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "pc")

    async def _probe(host_id):  # pragma: no cover — must never be called
        raise AssertionError("probe called with no multi-host chains")

    assert _run_pass(probe=_probe) == {}


def test_pass_starts_model_locally_when_ownership_arrives(monkeypatch, write_config):
    _chain_config(write_config)
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "mac")
    monkeypatch.setattr(backend_process, "is_running", lambda mid: False)

    up = {"gaming": False, "mac": True, "tower": True}
    started, stopped = [], []

    async def _probe(h):
        return up[h]

    async def _start(mid):
        started.append(mid)

    async def _stop(mid):
        stopped.append(mid)

    # Pass 1 (t=0): gaming already observed down, boot pick lands on mac →
    # mac (this host) brings whisper up.
    res = _run_pass(now=0, probe=_probe, start_local=_start, stop_local=_stop)
    assert res["whisper"]["owner"] == "mac"
    assert started == ["whisper"]
    assert "whisper" in model_failover._ENGINE_STARTED

    # Pass 2 (t=1000): gaming back but not yet stable → still mac, no re-start
    # (is_running now reports True).
    monkeypatch.setattr(backend_process, "is_running", lambda mid: True)
    up["gaming"] = True
    res = _run_pass(now=1000, probe=_probe, start_local=_start, stop_local=_stop)
    assert res["whisper"]["owner"] == "mac"
    assert started == ["whisper"] and stopped == []

    # Pass 3 (t=1700): gaming stable ≥ 600 s → hand back; mac stops the
    # instance the engine itself started.
    res = _run_pass(now=1700, probe=_probe, start_local=_start, stop_local=_stop)
    assert res["whisper"]["owner"] == "gaming"
    assert stopped == ["whisper"]
    assert "whisper" not in model_failover._ENGINE_STARTED


def test_pass_never_stops_an_instance_it_did_not_start(monkeypatch, write_config):
    _chain_config(write_config)
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "mac")
    # whisper is running locally (hand-started / autostarted) but NOT
    # engine-started; owner resolves to gaming → no stop may happen.
    monkeypatch.setattr(backend_process, "is_running", lambda mid: True)

    stopped = []

    async def _probe(h):
        return True

    async def _stop(mid):  # pragma: no cover — must never be called
        stopped.append(mid)
        raise AssertionError("stopped a non-engine-started instance")

    res = _run_pass(now=0, probe=_probe, stop_local=_stop)
    assert res["whisper"]["owner"] == "gaming"
    assert stopped == []


def test_pass_probe_error_reads_as_down(monkeypatch, write_config):
    _chain_config(write_config)
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "tower")
    monkeypatch.setattr(backend_process, "is_running", lambda mid: True)

    async def _probe(h):
        if h == "gaming":
            raise RuntimeError("dial blew up")
        return h != "mac"

    res = _run_pass(now=0, probe=_probe)
    # gaming (error→down) and mac (down) both skipped → tower owns.
    assert res["whisper"]["owner"] == "tower"


# --------------------------------------------------------------------------- #
# Settings loader
# --------------------------------------------------------------------------- #
def test_load_settings_defaults_and_overrides(tmp_path, write_config):
    write_config({
        "hub": {"port": 8000},
        "hosts": {"pc": {"platform": "win32", "default": True, "enabled": []}},
        "models": {},
        "failover": {"probe_interval_s": 5, "fail_after_s": 10,
                     "failback_after_s": 20, "policy": "sticky"},
    })
    s = model_failover.load_settings()
    assert (s.probe_interval_s, s.fail_after_s, s.failback_after_s, s.policy) == \
        (5.0, 10.0, 20.0, "sticky")

    d2 = tmp_path / "d2"
    d2.mkdir()
    write_config({
        "hub": {"port": 8000},
        "hosts": {"pc": {"platform": "win32", "default": True, "enabled": []}},
        "models": {},
        "failover": {"policy": "bogus"},
    }, dirpath=d2)
    s = model_failover.load_settings()
    assert s.policy == "auto"                       # bogus policy falls back
    assert s.fail_after_s == 90.0                   # defaults fill the rest


# --------------------------------------------------------------------------- #
# On-demand rows (#422): the failover engine never spawns them.
# --------------------------------------------------------------------------- #
def test_ensure_running_local_skips_on_demand_models(monkeypatch):
    from src.model_registry import Model

    def _boom(*a, **kw):
        raise AssertionError("backend_process must not be touched")

    monkeypatch.setattr(backend_process, "is_running", _boom)
    monkeypatch.setattr(backend_process, "start", _boom)

    m = Model(id="orpheus_od", display_name="orpheus-od", backend="tts",
              port=8093, hosts=["tower", "gaming"], host="tower",
              startup="on_demand", idle_unload_minutes=30)
    result = asyncio.run(model_failover._ensure_running_local(m, None))
    assert result is None
