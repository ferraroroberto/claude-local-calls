"""Boot a hub instance on a free port and expose its URL as a fixture.

Each e2e session spawns one ``uvicorn src.server:app`` on a random free
port and tears it down at the end. The hub is a single ASGI process —
the /admin SPA, the routers, and the /v1 surface all share one
event loop, so a single boot covers every endpoint.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import httpx
import pytest

from src import win_job

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Bounded default Playwright timeout (#100): cap implicit auto-waiting actions
# at 15 s so a stuck click/goto names itself instead of silently stacking
# toward Playwright's opaque 30 s default.  Explicit per-call timeout= and
# expect() web-first assertions are unaffected.
_DEFAULT_TIMEOUT_MS = int(os.environ.get("E2E_DEFAULT_TIMEOUT_MS", "15000"))


@pytest.fixture(autouse=True)
def _bound_default_timeouts(page):
    """Set a bounded action + navigation timeout on every Playwright page (#100)."""
    page.set_default_timeout(_DEFAULT_TIMEOUT_MS)
    page.set_default_navigation_timeout(_DEFAULT_TIMEOUT_MS)


def _free_tcp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


@pytest.fixture(scope="session")
def hub_url() -> Iterator[str]:
    port = _free_tcp_port()
    url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # Disable OTel for the autobooted hub — otherwise it tries to push
    # spans to a non-existent localhost:4317 OTLP collector and logs
    # connect-refused noise into the test log on every routed request.
    env.setdefault("OTEL_SDK_DISABLED", "true")
    # The autostart sampler hits nvidia-smi every 2s. On a CI runner
    # without an NVIDIA GPU that's noisy but harmless.
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW
    log_path = PROJECT_ROOT / "tests" / "e2e" / "autoboot-hub.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.server:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=str(PROJECT_ROOT),
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        env=env,
        creationflags=creationflags,
    )
    # Contain the hub and anything it spawns (e.g. an on-demand TTS backend
    # loaded mid-test, #468) in one Job Object, so tearing this fixture down
    # reaps the whole tree instead of just the hub PID — a plain
    # ``proc.terminate()`` never reaches a grandchild, and an on-demand
    # backend is deliberately spawned in its own process group (so a
    # restart's CTRL_BREAK doesn't hit it), so it survives that alone.
    # ``None`` on non-Windows / on API failure — the ``if job`` guards below
    # then fall back to the previous hub-PID-only teardown.
    job = win_job.create_kill_on_close_job(f"local-llm-hub-e2e-{port}")
    if job:
        win_job.assign_pid(job, proc.pid)

    deadline = time.time() + 30.0
    last_err = "timed out"
    while time.time() < deadline:
        if proc.poll() is not None:
            log_fp.close()
            if job:
                win_job.terminate_and_close(job)
            tail = log_path.read_text(encoding="utf-8")[-1500:]
            pytest.fail(f"hub exited before becoming reachable. Log tail:\n{tail}")
        try:
            r = httpx.get(f"{url}/health", timeout=1.0)
            if r.status_code == 200:
                break
        except httpx.HTTPError as exc:
            last_err = repr(exc)
        time.sleep(0.3)
    else:
        proc.terminate()
        log_fp.close()
        if job:
            win_job.terminate_and_close(job)
        pytest.fail(f"hub never became reachable: {last_err}")

    try:
        yield url
    finally:
        if job:
            # Kills the hub *and* every backend it spawned in the meantime,
            # even one whose immediate parent (the hub) already exited —
            # the exact shape of #468's incident.
            win_job.terminate_and_close(job)
        else:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log_fp.close()


@pytest.fixture(scope="session")
def admin_url(hub_url: str) -> str:
    return f"{hub_url}/admin/"
