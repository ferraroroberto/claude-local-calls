"""Unit tests for ``tray.tray.HubProcess`` driving the shared
``ProcessSupervisor`` (#530) instead of a second hand-rolled start/stop.

These fake ``ProcessSupervisor`` itself (a class-level monkeypatch of the
name imported into ``tray.tray``) rather than spawning real subprocesses —
``HubProcess``'s own job is wiring ``already_running``/``reachable``/
``build_spawn_spec``/``set_process`` correctly and translating the
supervisor's ``(False, "already running")`` into the tray's own
``(True, ...)`` convention; the supervisor's internal escalation logic is
covered by ``tests/test_process_supervisor.py``.
"""

from __future__ import annotations

import os

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

from tray import tray as tray_mod  # noqa: E402


class _FakeSupervisor:
    """Records the kwargs ``HubProcess.start()`` wired it with and returns
    a scripted ``(ok, msg)`` from ``start()``."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        _FakeSupervisor.last_kwargs = kwargs

    def start(self):
        return _FakeSupervisor.result


def _fake_supervisor(result):
    _FakeSupervisor.result = result
    return _FakeSupervisor


class _FakeAliveProc:
    def poll(self):
        return None   # still running


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_start_translates_already_running_to_ok(monkeypatch):
    monkeypatch.setattr(tray_mod, "ProcessSupervisor", _fake_supervisor((False, "already running")))
    monkeypatch.setattr(tray_mod, "cross_process_lock", lambda name: _NullCtx())

    hub = tray_mod.HubProcess()
    ok, msg = hub.start()
    assert ok is True
    assert msg == "already running"


def test_start_passes_through_adopted_ok_true(monkeypatch):
    monkeypatch.setattr(tray_mod, "ProcessSupervisor", _fake_supervisor((True, "adopted external hub")))
    monkeypatch.setattr(tray_mod, "cross_process_lock", lambda name: _NullCtx())

    hub = tray_mod.HubProcess()
    ok, msg = hub.start()
    assert ok is True
    assert msg == "adopted external hub"


def test_start_passes_through_spawn_failure_unmodified(monkeypatch):
    monkeypatch.setattr(tray_mod, "ProcessSupervisor", _fake_supervisor((False, "failed to launch: boom")))
    monkeypatch.setattr(tray_mod, "cross_process_lock", lambda name: _NullCtx())

    hub = tray_mod.HubProcess()
    ok, msg = hub.start()
    assert ok is False
    assert msg == "failed to launch: boom"


def test_start_wires_the_hub_module_as_the_spawn_command(monkeypatch):
    monkeypatch.setattr(tray_mod, "ProcessSupervisor", _fake_supervisor((True, "started (pid=1)")))
    monkeypatch.setattr(tray_mod, "cross_process_lock", lambda name: _NullCtx())

    hub = tray_mod.HubProcess()
    hub.start()

    spec = _FakeSupervisor.last_kwargs["build_spawn_spec"]()
    assert spec.cmd[-2:] == ["-m", "src.server"]
    assert spec.creationflags == tray_mod.WIN_NEW_GROUP


def test_stop_delegates_to_shared_stop_popen(monkeypatch):
    calls = []
    monkeypatch.setattr(
        tray_mod.ProcessSupervisor, "stop_popen",
        staticmethod(lambda proc, **kw: (calls.append((proc, kw)), (True, "stopped"))[1]),
    )

    hub = tray_mod.HubProcess()
    fake_proc = _FakeAliveProc()
    hub.proc = fake_proc

    ok, msg = hub.stop()
    assert ok and msg == "stopped"
    assert calls and calls[0][0] is fake_proc
    assert hub.proc is None
