"""Unit tests for `parakeet_server._start_worker`'s bounded startup wait.

Regression for issue #297: `_start_worker` used to block forever on
`proc.stdout.readline()` waiting for "READY", with no wall-clock deadline
— unlike its sibling process wrapper (`tts_engines.orpheus._wait_llama_ready`),
which bounds the wait. The worker binary itself is darwin-only, so these
tests fake `subprocess.Popen` rather than driving the real Swift process.
"""

from __future__ import annotations

import queue as queue_mod

import pytest

from src import parakeet_server


class _FakeStdout:
    """A readline() source that blocks forever once its lines are exhausted
    — mirrors a real pipe with nothing more written to it."""

    def __init__(self, lines):
        self._q: "queue_mod.Queue[str]" = queue_mod.Queue()
        for line in lines:
            self._q.put(line)

    def readline(self) -> str:
        return self._q.get()

    def read(self) -> str:
        parts = []
        while not self._q.empty():
            parts.append(self._q.get())
        return "".join(parts)


class _FakeProc:
    def __init__(self, stdout_lines, stderr_text="boom"):
        self.stdout = _FakeStdout(stdout_lines)
        self.stderr = _FakeStdout([stderr_text])
        self.pid = 4242
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None) -> int:
        return 0

    def poll(self):
        return None if not self.terminated else 0


class _FakeWorkerBin:
    def exists(self) -> bool:
        return True

    def __str__(self) -> str:
        return "/fake/ParakeetWorker"


@pytest.fixture(autouse=True)
def _fake_worker_bin(monkeypatch):
    monkeypatch.setattr(parakeet_server, "WORKER_BIN", _FakeWorkerBin())


def test_start_worker_returns_once_ready(monkeypatch):
    fake = _FakeProc(["loading model...\n", "READY\n"])
    monkeypatch.setattr(parakeet_server.subprocess, "Popen", lambda *a, **k: fake)

    proc, out_q = parakeet_server._start_worker()

    assert proc is fake
    assert out_q is not None
    assert not fake.terminated


def test_start_worker_times_out_when_worker_never_prints_ready(monkeypatch):
    fake = _FakeProc([])  # worker starts but never writes anything to stdout
    monkeypatch.setattr(parakeet_server.subprocess, "Popen", lambda *a, **k: fake)
    monkeypatch.setattr(parakeet_server, "STARTUP_DEADLINE_S", 0.2)

    with pytest.raises(RuntimeError, match="did not become ready"):
        parakeet_server._start_worker()

    assert fake.terminated


def test_start_worker_raises_on_early_exit(monkeypatch):
    fake = _FakeProc([""], stderr_text="crashed on load")  # EOF before READY
    monkeypatch.setattr(parakeet_server.subprocess, "Popen", lambda *a, **k: fake)
    monkeypatch.setattr(parakeet_server, "STARTUP_DEADLINE_S", 5.0)

    with pytest.raises(RuntimeError, match="failed to start"):
        parakeet_server._start_worker()
