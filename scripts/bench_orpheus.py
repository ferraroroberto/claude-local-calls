"""Benchmark Orpheus token-generation throughput on the loopback llama-server.

Orpheus TTS total synthesis time is bounded by how fast the loopback
``llama-server`` child generates SNAC audio tokens (issue #105). This script
measures that floor in isolation so llama-server flags can be tuned without
bouncing the live hub for every trial, and re-verified after future model /
GPU swaps.

Two modes:

  Flag sweep (default) — spawn a *scratch* ``llama-server`` on ``--port``
  (default 18099) for each named flag set in ``SWEEP``, POST the canonical
  Orpheus prompt to ``/completion`` ``--reps`` times, and report the median
  generation rate (``tok/s``, from llama's own ``timings``) and median wall
  time. Each scratch server is torn down before the next. This is pure
  llama-server — it does not touch the running hub.

      .venv\\Scripts\\python scripts/bench_orpheus.py

  Hub end-to-end (``--hub-e2e``) — time the *live* hub's
  ``POST /v1/audio/speech`` (:8000) for the same phrase ``--reps`` times.
  Run it once before the flag change (baseline) and once after the winning
  flags are applied + the hub restarted (after) to get the end-to-end delta.

      .venv\\Scripts\\python scripts/bench_orpheus.py --hub-e2e

The phrase defaults to "this is a test" (the ~1.8 s clip the issue measured).
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# UTF-8 stdout so the table renders under captured/redirected runs on Windows
# (cp1252 fallback otherwise throws on the box-drawing chars).
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

from _lib import no_window_flags  # noqa: E402
from src.backend_process import VENDOR_LLAMA, llama_server_binary  # noqa: E402
from src.model_registry import resolve as resolve_model  # noqa: E402

DEFAULT_VOICE = "tara"
HUB_BASE = "http://127.0.0.1:8000"

# Named flag sets layered on top of the always-on base (``-ngl 99 --no-webui``
# + the scratch host/port/model). One variable changed at a time so the
# contribution of each is legible. "greedy" uses the production flags but
# disables sampling in the payload (see ``run_trial``) to isolate sampler cost.
SWEEP: List[Tuple[str, List[str], bool]] = [
    # (name, extra llama-server flags, greedy_sampler)
    ("baseline (-c 8192)", ["-c", "8192"], False),
    ("flash-attn", ["-c", "8192", "--flash-attn", "on"], False),
    ("flash-attn +no-mmap", ["-c", "8192", "--flash-attn", "on", "--no-mmap"], False),
    ("flash-attn +batch", ["-c", "8192", "--flash-attn", "on", "-b", "2048", "-ub", "512"], False),
    ("flash-attn +greedy", ["-c", "8192", "--flash-attn", "on"], True),
]


def _orpheus_prompt(text: str, voice: str) -> str:
    """The llama.cpp-route prompt convention used by OrpheusEngine._prompt_for."""
    return f"<|audio|>{voice}: {text}<|eot_id|>"


def _payload(prompt: str, greedy: bool) -> dict:
    """Mirror OrpheusEngine._completion_payload; ``greedy`` strips sampling."""
    body = {
        "prompt": prompt,
        "n_predict": 4096,
        "cache_prompt": True,
        "stream": False,
    }
    if greedy:
        body.update({"temperature": 0.0, "top_k": 1})
    else:
        body.update({"temperature": 0.6, "top_p": 0.9, "repeat_penalty": 1.1})
    return body


def _gguf_path() -> Path:
    model = resolve_model("orpheus")
    if model is None or not model.model_path:
        raise SystemExit("orpheus row not found / has no model_path in config/models.yaml")
    gguf = (PROJECT_ROOT / model.model_path).resolve()
    if not gguf.exists():
        raise SystemExit(
            f"Orpheus GGUF not found at {gguf} - "
            f"run scripts/download_models.py --only orpheus"
        )
    return gguf


def _spawn(port: int, extra_flags: List[str]) -> subprocess.Popen:
    bin_path = llama_server_binary()
    if not bin_path.exists():
        raise SystemExit(f"llama-server not found at {bin_path} - run scripts/install_llama_cpp.py")
    cmd = [
        str(bin_path),
        "-m", str(_gguf_path()),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-ngl", "99",
        "--no-webui",
        *extra_flags,
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if sys.platform == "win32":
        env["PATH"] = str(VENDOR_LLAMA) + os.pathsep + env.get("PATH", "")
    return subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        creationflags=no_window_flags(),
    )


def _wait_ready(port: int, proc: subprocess.Popen, deadline_s: float = 180.0) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit("scratch llama-server exited during startup")
        try:
            if httpx.get(url, timeout=2.0).status_code == 200:
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    raise SystemExit(f"scratch llama-server not ready within {deadline_s:.0f}s")


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_trial(port: int, text: str, voice: str, reps: int, greedy: bool) -> Dict[str, float]:
    """POST the prompt ``reps`` times; return median tok/s, wall ms, tokens."""
    url = f"http://127.0.0.1:{port}/completion"
    prompt = _orpheus_prompt(text, voice)
    payload = _payload(prompt, greedy)
    # One warmup (compiles CUDA graphs / fills prompt cache) — not measured.
    httpx.post(url, json=payload, timeout=300.0).raise_for_status()
    tok_s: List[float] = []
    walls: List[float] = []
    toks: List[int] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = httpx.post(url, json=payload, timeout=300.0)
        wall = (time.perf_counter() - t0) * 1000.0
        r.raise_for_status()
        timings = r.json().get("timings", {}) or {}
        tok_s.append(float(timings.get("predicted_per_second", 0.0)))
        toks.append(int(timings.get("predicted_n", 0)))
        walls.append(wall)
    return {
        "tok_s": statistics.median(tok_s),
        "wall_ms": statistics.median(walls),
        "tokens": statistics.median(toks),
    }


def run_sweep(port: int, text: str, voice: str, reps: int) -> None:
    log.info("Orpheus llama-server flag sweep — phrase=%r voice=%s reps=%d", text, voice, reps)
    rows: List[Tuple[str, Dict[str, float]]] = []
    for name, flags, greedy in SWEEP:
        log.info("… %s: spawning scratch llama-server on :%d …", name, port)
        proc = _spawn(port, flags)
        try:
            _wait_ready(port, proc)
            res = run_trial(port, text, voice, reps, greedy)
        finally:
            _stop(proc)
            time.sleep(1.0)  # let the OS release the port before the next spawn
        rows.append((name, res))
        log.info(
            "   → %7.1f tok/s | %8.1f ms wall | %d tok",
            res["tok_s"], res["wall_ms"], int(res["tokens"]),
        )

    base = rows[0][1]["tok_s"] if rows else 0.0
    log.info("=" * 64)
    log.info("%-26s%10s%11s%10s", "flag set", "tok/s", "wall ms", "vs base")
    log.info("-" * 64)
    for name, res in rows:
        speedup = (res["tok_s"] / base) if base else 0.0
        log.info("%-26s%10.1f%11.1f%9.2fx", name, res["tok_s"], res["wall_ms"], speedup)
    log.info("=" * 64)


def run_hub_e2e(text: str, voice: str, reps: int) -> None:
    """Time the live hub's POST /v1/audio/speech end-to-end."""
    url = f"{HUB_BASE}/v1/audio/speech"
    body = {"model": "orpheus", "input": text, "voice": voice, "response_format": "wav"}
    log.info("Hub end-to-end /v1/audio/speech — phrase=%r voice=%s reps=%d", text, voice, reps)
    # Warmup (loads engine / fills cache) — not measured.
    httpx.post(url, json=body, timeout=300.0).raise_for_status()
    walls: List[float] = []
    nbytes = 0
    for _ in range(reps):
        t0 = time.perf_counter()
        r = httpx.post(url, json=body, timeout=300.0)
        wall = (time.perf_counter() - t0) * 1000.0
        r.raise_for_status()
        nbytes = len(r.content)
        walls.append(wall)
        log.info("   → %8.1f ms (%d bytes)", wall, nbytes)
    log.info("=" * 48)
    log.info("median end-to-end: %.1f ms  (%d bytes wav)", statistics.median(walls), nbytes)
    log.info("=" * 48)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=18099, help="scratch llama-server port (default 18099)")
    ap.add_argument("--reps", type=int, default=5, help="measured repetitions per trial (default 5)")
    ap.add_argument("--text", default="this is a test", help="phrase to synthesize")
    ap.add_argument("--voice", default=DEFAULT_VOICE, help="Orpheus voice (default tara)")
    ap.add_argument("--hub-e2e", action="store_true", help="measure live hub :8000 end-to-end instead of the flag sweep")
    args = ap.parse_args(argv)

    if args.hub_e2e:
        run_hub_e2e(args.text, args.voice, args.reps)
    else:
        run_sweep(args.port, args.text, args.voice, args.reps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
