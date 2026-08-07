"""On-demand model lifecycle (#422): spawn-on-first-request + idle unload.

Pure-logic coverage — ``backend_process`` / host resolution / the remote
proxy are monkeypatched, no process is ever spawned. The idle sweep is
driven with injected ``now`` values so the window math is deterministic.
"""

from __future__ import annotations

import logging

import pytest

from src import backend_process as bp
from src import host_profile, model_registry, on_demand, remote_proxy
from src.model_registry import Model


def _model(**kw) -> Model:
    base = dict(
        id="gemma", display_name="gemma4-26b-a4b-it", backend="openai",
        port=8087, startup="on_demand", idle_unload_minutes=30,
        est_vram_mb=13400,
    )
    base.update(kw)
    return Model(**base)


@pytest.fixture(autouse=True)
def _clean_state():
    on_demand.reset()
    yield
    on_demand.reset()


# --------------------------------------------------------------------------- #
# ensure_ready
# --------------------------------------------------------------------------- #
class _Profile:
    def __init__(self, vram_mb, host_id="tower"):
        self.vram_mb = vram_mb
        self.id = host_id


def _forbid(monkeypatch, module, name):
    def _boom(*a, **kw):  # noqa: ANN002, ANN003
        raise AssertionError(f"{name} must not be called")
    monkeypatch.setattr(module, name, _boom)


def test_ensure_ready_noop_for_eager_rows(monkeypatch):
    _forbid(monkeypatch, bp, "is_reachable")
    _forbid(monkeypatch, bp, "start")
    on_demand.ensure_ready(_model(startup="eager"))


def test_ensure_ready_noop_for_remote_owned(monkeypatch):
    monkeypatch.setattr(remote_proxy, "remote_base_url", lambda m: "http://peer:8000")
    _forbid(monkeypatch, bp, "is_reachable")
    _forbid(monkeypatch, bp, "start")
    on_demand.ensure_ready(_model())


def test_ensure_ready_noop_when_already_reachable(monkeypatch):
    monkeypatch.setattr(remote_proxy, "remote_base_url", lambda m: None)
    monkeypatch.setattr(bp, "is_reachable", lambda m, timeout=1.5: True)
    _forbid(monkeypatch, bp, "start")
    on_demand.ensure_ready(_model())


def test_ensure_ready_spawns_and_waits_for_readiness(monkeypatch):
    monkeypatch.setattr(remote_proxy, "remote_base_url", lambda m: None)
    monkeypatch.setattr(on_demand, "READY_POLL_S", 0.01)
    monkeypatch.setattr(host_profile, "resolve", lambda: _Profile(None))

    probes = {"n": 0}

    def _reachable(m, timeout=1.5):
        probes["n"] += 1
        return probes["n"] > 3  # cold for the pre-checks, up on the 4th probe

    started: list[str] = []
    monkeypatch.setattr(bp, "is_reachable", _reachable)
    monkeypatch.setattr(bp, "running_backends", dict)
    monkeypatch.setattr(bp, "start", lambda mid: (started.append(mid), (True, "started"))[1])

    on_demand.ensure_ready(_model(), deadline_s=5.0)
    assert started == ["gemma"]


def test_ensure_ready_raises_on_deadline(monkeypatch):
    monkeypatch.setattr(remote_proxy, "remote_base_url", lambda m: None)
    monkeypatch.setattr(on_demand, "READY_POLL_S", 0.01)
    monkeypatch.setattr(host_profile, "resolve", lambda: _Profile(None))
    monkeypatch.setattr(bp, "is_reachable", lambda m, timeout=1.5: False)
    monkeypatch.setattr(bp, "running_backends", dict)
    monkeypatch.setattr(bp, "start", lambda mid: (True, "started"))

    with pytest.raises(on_demand.OnDemandNotReady):
        on_demand.ensure_ready(_model(), deadline_s=0.05)


def test_ensure_ready_raises_on_start_failure(monkeypatch):
    monkeypatch.setattr(remote_proxy, "remote_base_url", lambda m: None)
    monkeypatch.setattr(host_profile, "resolve", lambda: _Profile(None))
    monkeypatch.setattr(bp, "is_reachable", lambda m, timeout=1.5: False)
    monkeypatch.setattr(bp, "running_backends", dict)
    monkeypatch.setattr(bp, "start", lambda mid: (False, "GGUF not found"))

    with pytest.raises(on_demand.OnDemandNotReady, match="GGUF not found"):
        on_demand.ensure_ready(_model(), deadline_s=0.05)


# --------------------------------------------------------------------------- #
# VRAM budget warning (warning only, never a block)
# --------------------------------------------------------------------------- #
def test_vram_overcommit_warns_and_returns_projection(monkeypatch, caplog):
    monkeypatch.setattr(host_profile, "resolve", lambda: _Profile(16384))
    running = {
        "qwen35_4b": _model(id="qwen35_4b", startup="eager", est_vram_mb=2100),
        "orpheus": _model(id="orpheus", startup="eager", est_vram_mb=2200),
    }
    monkeypatch.setattr(bp, "running_backends", lambda: running)

    with caplog.at_level(logging.WARNING, logger="src.on_demand"):
        projected = on_demand._warn_on_vram_overcommit(_model(est_vram_mb=13400))
    assert projected == 2100 + 2200 + 13400
    assert any("VRAM overcommit" in r.message for r in caplog.records)


def test_vram_within_budget_stays_silent(monkeypatch, caplog):
    monkeypatch.setattr(host_profile, "resolve", lambda: _Profile(16384))
    monkeypatch.setattr(bp, "running_backends", dict)

    with caplog.at_level(logging.WARNING, logger="src.on_demand"):
        assert on_demand._warn_on_vram_overcommit(_model(est_vram_mb=13400)) is None
    assert not caplog.records


def test_vram_check_skips_ceilingless_hosts(monkeypatch):
    monkeypatch.setattr(host_profile, "resolve", lambda: _Profile(None))
    _forbid(monkeypatch, bp, "running_backends")
    assert on_demand._warn_on_vram_overcommit(_model()) is None


def test_vram_check_excludes_cpu_resident_rows(monkeypatch, caplog):
    """A running row that is CPU-resident on this host (piper — and whisper,
    whose chain flags tower as a degraded ``cpu: true`` tier) holds no GPU
    VRAM and must not eat headroom (#431). Here the GPU-resident set is
    2100 + 13400 = 15500 ≤ 16384 — counting the CPU rows' 5000 would have
    raised a false overcommit warning."""
    monkeypatch.setattr(host_profile, "resolve", lambda: _Profile(16384))
    running = {
        "qwen35_4b": _model(id="qwen35_4b", startup="eager", est_vram_mb=2100),
        "piper": _model(id="piper", startup="eager", est_vram_mb=2500),
        "whisper": _model(id="whisper", startup="eager", est_vram_mb=2500),
    }
    monkeypatch.setattr(bp, "running_backends", lambda: running)

    with caplog.at_level(logging.WARNING, logger="src.on_demand"):
        assert on_demand._warn_on_vram_overcommit(_model(est_vram_mb=13400)) is None
    assert not caplog.records


# --------------------------------------------------------------------------- #
# Idle-unload decision
# --------------------------------------------------------------------------- #
def _wire_sweep(monkeypatch, models, running=True):
    monkeypatch.setattr(model_registry, "local_models", lambda host=None: models)
    monkeypatch.setattr(bp, "is_running", lambda mid: running)
    stopped: list[str] = []
    monkeypatch.setattr(bp, "stop", lambda mid: (stopped.append(mid), (True, "stopped"))[1])
    return stopped


def test_idle_sweep_arms_then_unloads_after_window(monkeypatch):
    m = _model()  # 30-minute window
    stopped = _wire_sweep(monkeypatch, [m])

    # First sighting of an already-running instance arms the clock…
    assert on_demand.idle_unload_pass(now=1000.0) == {"gemma": "armed"}
    assert stopped == []
    # …still warm inside the window…
    assert on_demand.idle_unload_pass(now=1000.0 + 29 * 60) == {"gemma": "warm"}
    # …and unloads once the window has fully elapsed.
    assert on_demand.idle_unload_pass(now=1000.0 + 31 * 60) == {"gemma": "unloaded"}
    assert stopped == ["gemma"]


def test_idle_sweep_never_unloads_with_requests_in_flight(monkeypatch):
    m = _model()
    stopped = _wire_sweep(monkeypatch, [m])

    on_demand.request_started("gemma", now=1000.0)
    assert on_demand.idle_unload_pass(now=1000.0 + 31 * 60) == {"gemma": "busy"}
    assert stopped == []

    # The finish re-touches, so the window restarts from request end.
    on_demand.request_finished("gemma", now=1000.0 + 31 * 60)
    assert on_demand.idle_unload_pass(now=1000.0 + 32 * 60) == {"gemma": "warm"}
    assert on_demand.idle_unload_pass(now=1000.0 + 62 * 60) == {"gemma": "unloaded"}
    assert stopped == ["gemma"]


def test_idle_sweep_ignores_eager_and_windowless_rows(monkeypatch):
    eager = _model(id="qwen", startup="eager")
    no_window = _model(id="sticky", idle_unload_minutes=None)
    stopped = _wire_sweep(monkeypatch, [eager, no_window])

    assert on_demand.idle_unload_pass(now=1000.0) == {}
    assert stopped == []


def test_idle_sweep_skips_models_not_running(monkeypatch):
    m = _model()
    stopped = _wire_sweep(monkeypatch, [m], running=False)

    assert on_demand.idle_unload_pass(now=1000.0) == {}
    assert stopped == []


# --------------------------------------------------------------------------- #
# tracking() — the shared in-flight bookkeeping every dispatch site uses (#470)
# --------------------------------------------------------------------------- #
def _in_flight(model_id: str) -> int:
    return on_demand._IN_FLIGHT.get(model_id, 0)


def test_tracking_is_inactive_for_eager_rows_and_remote_hops():
    """Only a *locally served* on_demand row is counted: an eager row has no
    unload window, and a remote hop is the peer hub's to account for."""
    with on_demand.tracking(_model(startup="eager")):
        assert _in_flight("gemma") == 0
    with on_demand.tracking(_model(), "http://10.0.0.9:8000"):
        assert _in_flight("gemma") == 0
    assert _in_flight("gemma") == 0


def test_tracking_block_balances_on_return_and_on_raise():
    with on_demand.tracking(_model()):
        assert _in_flight("gemma") == 1
    assert _in_flight("gemma") == 0

    with pytest.raises(RuntimeError):
        with on_demand.tracking(_model()):
            assert _in_flight("gemma") == 1
            raise RuntimeError("upstream blew up")
    assert _in_flight("gemma") == 0


def test_detached_tracking_outlives_the_block_until_finish():
    """The streaming shape: the response is still being served after the
    dispatch function returns, so ``detach()`` hands the mark to the
    generator's ``finally`` instead of the ``with``."""
    with on_demand.tracking(_model()) as track:
        track.detach()
    assert _in_flight("gemma") == 1          # still in flight after the block
    track.finish()
    assert _in_flight("gemma") == 0


def test_finish_is_idempotent():
    """A belt-and-braces second finish (a detach racing a raise, a generator
    closed twice) must not decrement someone else's request."""
    track = on_demand.tracking(_model()).start()
    assert _in_flight("gemma") == 1
    track.finish()
    track.finish()
    assert _in_flight("gemma") == 0
