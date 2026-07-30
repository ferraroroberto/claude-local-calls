"""Selector event-loop shim (issue #222) -- root cause of hub-unresponsive wedges.

asyncio's default Windows proactor event loop closes its listening socket
on any aborted client connection (WinError 64); the selector loop's accept
path doesn't. These tests cover the wiring (all 4 of this repo's
``uvicorn.run()`` call sites pick the shim) and the actual accept-loop
resilience the shim buys.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from src.event_loop import LOOP_FACTORY, selector_loop_factory

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_selector_loop_factory_returns_selector_instance_on_win32(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    sentinel = object()
    monkeypatch.setattr(asyncio, "SelectorEventLoop", lambda: sentinel)
    assert selector_loop_factory() is sentinel


def test_selector_loop_factory_defers_on_other_platforms(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    sentinel = object()
    monkeypatch.setattr(asyncio, "new_event_loop", lambda: sentinel)
    assert selector_loop_factory() is sentinel


def test_selector_loop_factory_is_zero_arg_and_returns_an_instance():
    """Regression pin: uvicorn imports a *custom* loop= target and calls
    it as a bare Callable[[], AbstractEventLoop] -- no use_subprocess kwarg,
    and it must return an instantiated loop, not a loop class (app-launcher
    #388's original bug: returning the class left Runner calling unbound
    methods)."""
    loop = selector_loop_factory()
    try:
        assert isinstance(loop, asyncio.AbstractEventLoop)
    finally:
        loop.close()


def test_loop_factory_dotted_path_matches_module():
    """LOOP_FACTORY is a dotted-path string uvicorn imports fresh via
    importlib -- it must resolve to this module regardless of how the
    servers import the constant themselves."""
    assert LOOP_FACTORY == "src.event_loop:selector_loop_factory"


def _wires_loop_factory(relpath: str) -> None:
    src = (_REPO_ROOT / relpath).read_text(encoding="utf-8")
    assert "event_loop import LOOP_FACTORY" in src, f"{relpath} doesn't import LOOP_FACTORY"
    assert "loop=LOOP_FACTORY" in src, f"{relpath} doesn't pass loop=LOOP_FACTORY to uvicorn.run"


def test_hub_server_wires_loop_factory():
    _wires_loop_factory("src/server.py")


def test_whisper_translate_proxy_wires_loop_factory():
    _wires_loop_factory("src/whisper_translate_proxy.py")


def test_tts_server_wires_loop_factory():
    _wires_loop_factory("src/tts_server.py")


def test_parakeet_server_wires_loop_factory():
    _wires_loop_factory("src/parakeet_server.py")


def _run_event_loop_worker(mode: str) -> subprocess.CompletedProcess:
    """Run the real-socket abort bombardment in its own process (#441).

    The bombardment (``tests/_event_loop_worker.py``) deliberately corrupts
    OS-level asyncio/proactor state -- that's the whole point of the
    proactor-loop test below. #416's original investigation into the full
    suite's order-dependent ``asyncio.run()`` failures named this file as
    the prime suspect for leaking that corruption into later tests sharing
    the same pytest process; #417 fixed a different, confirmed leak but
    never actually ruled this one in or out. Running it out-of-process means
    whatever it perturbs dies with the subprocess instead."""
    return subprocess.run(
        [sys.executable, "-m", "tests._event_loop_worker", mode],
        cwd=_REPO_ROOT, capture_output=True, text=True, timeout=60,
    )


@pytest.mark.skipif(sys.platform != "win32", reason="proactor-loop bug is Windows-only")
def test_selector_loop_survives_aborted_connections():
    result = _run_event_loop_worker("selector")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="proactor-loop bug is Windows-only")
def test_proactor_loop_dies_on_aborted_connections():
    """Documents the bug this issue fixes -- the shim exists because this
    fails. If a future CPython/uvicorn release fixes the proactor loop
    itself, this test (not the shim) is what should be revisited."""
    result = _run_event_loop_worker("proactor")
    assert result.returncode == 0, result.stdout + result.stderr
