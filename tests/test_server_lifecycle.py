"""Unit tests for src/server_lifecycle.py's background-task lifecycle (#416).

``wire_observatory_loop`` spawns 8 perpetual/one-shot tasks on the running
loop (6 pre-#422 + the on-demand idle watchdog + the #424 config drift
loop). Before this fix nothing cancelled them on shutdown, so a torn-down
ASGI lifespan (every ``with TestClient(app) as c:`` exit in the test suite)
left them running against a dead loop -- ``_fleet_reconcile_loop`` /
``_model_failover_loop`` kept mutating the exact module-global state
``test_fleet_reconcile.py`` / ``test_model_failover.py`` monkeypatch and
assert on, the order-dependent full-suite pollution reported in #416.
"""

from __future__ import annotations

import asyncio

from src import backend_process, model_registry, server_lifecycle as sl, services


def _stub_real_ops(monkeypatch):
    """No real subprocess/network side effects from the lifecycle tasks."""
    monkeypatch.setattr(model_registry, "desired_model_ids", lambda *a, **kw: [])
    monkeypatch.setattr(backend_process, "inherit_running_backends", lambda: 0)

    async def _no_launch():
        return {"ok": True, "steps": []}

    monkeypatch.setattr(services, "launch_stack", _no_launch)
    monkeypatch.setattr(services, "launch_agentsview", _no_launch)


def test_wire_observatory_loop_tracks_its_tasks(monkeypatch):
    _stub_real_ops(monkeypatch)

    async def _run():
        await sl.wire_observatory_loop()
        assert len(sl._BACKGROUND_TASKS) == 8
        await sl.stop_background_tasks()

    asyncio.run(_run())
    assert sl._BACKGROUND_TASKS == []


def test_stop_background_tasks_cancels_the_perpetual_loops(monkeypatch):
    """Reproduces #416's leak: without cancellation, the boot-delay loops
    (fleet reconcile / model failover / resource sampler) are still pending
    when the lifespan tears down -- proven here by asserting each task is
    both cancelled and finished after stop_background_tasks()."""
    _stub_real_ops(monkeypatch)

    async def _run():
        await sl.wire_observatory_loop()
        tasks = list(sl._BACKGROUND_TASKS)
        # The perpetual loops (resource sampler, fleet reconcile, model
        # failover) are still sleeping through their boot delay -- not done.
        assert any(not t.done() for t in tasks)
        await sl.stop_background_tasks()
        for t in tasks:
            assert t.done()
            assert t.cancelled() or t.exception() is None

    asyncio.run(_run())


def test_stop_background_tasks_is_a_safe_noop_with_nothing_tracked():
    asyncio.run(sl.stop_background_tasks())
    assert sl._BACKGROUND_TASKS == []
