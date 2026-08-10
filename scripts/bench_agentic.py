"""Benchmark chat/agentic backends on any hub, for role-swap decisions.

The text-lane counterpart to ``scripts/bench_voice.py``: one reusable harness
for deciding which model should own ``agentic_light`` / ``agentic_heavy``, and
for re-verifying after a ``/swap-model`` or a host/GPU change. It targets a hub
by ``--base-url``, so the *same* tool measures any machine::

    # one arm
    python scripts/bench_agentic.py run --base-url http://127.0.0.1:8000 \
        --model qwen35_4b --out .scratch/nemotron-spike

    # a whole bake-off, arms measured back to back (issue #486)
    python scripts/bench_agentic.py run --base-url http://127.0.0.1:8000 \
        --model qwen35_4b --model qwen35_4b_nothink \
        --model nemotron4b --model nemotron4b_nothink \
        --out .scratch/nemotron-spike --blind

Requests go through the hub's OpenAI-shaped ``/v1/chat/completions`` — the path
real clients take — not a private backend port, so the numbers include the
hub's own overhead the way a caller experiences it.

**Why wall-clock, not just tok/s.** A reasoning model emits hundreds of
``<think>`` tokens before its answer. Generation tok/s can therefore *rise*
while time-to-a-usable-answer gets *worse*. This harness reports both and
treats ``total_s`` (request sent -> complete answer in hand) as the headline.
``completion_tokens`` and ``gen_tok_s`` are exact — read from the backend's
own ``usage`` and ``timings`` — and both *include* reasoning tokens, which is
precisely why the two numbers diverge.

The reasoning tax is measured as the **think-arm minus no-think-arm delta**
rather than estimated per response. That is what the 4-arm layout in issue #486
buys: the hub scrubs ``<think>`` blocks server-side
(``src/openai_upstream.ThinkStripper``), so the reasoning never appears in the
response text and cannot be counted from it — but running the same prompt with
thinking on and off isolates the cost exactly.

Note that server-side scrubbing also means the caller sees *nothing* until
reasoning finishes: streamed TTFT on a thinking arm was 1.005 s of a 1.571 s
total (64%) during #486. ``ttft_s`` is measured with a real streaming request
for that reason.

**A truncated answer is a failed answer.** Any response with
``finish_reason == "length"`` is recorded as ``ok=False`` and excluded from the
timing aggregates. Reasoning models blow small budgets *inside* the think block
and return nothing usable (observed at ``max_tokens=384`` on the work-PC spike,
issue #486); counting that as a fast response would invert the result.

**VRAM sampling is a validity check, not a metric.** ``gemma4_26b`` is
on-demand at ~13.4 GB on the tower. If a fleet client wakes it mid-run it
overcommits the card and every subsequent timing is garbage. Each prompt
records VRAM before/after; ``--vram-guard`` fails the run loudly when used
memory jumps beyond the threshold rather than silently reporting spoiled
numbers.

Prompts live in ``scripts/bench_agentic_prompts.json`` (committed, so a run is
reproducible) and are weighted to match ``/frontier-refresh``'s fixed quality
weights: agentic/tool 0.35, polish 0.25, multilingual 0.25, long context 0.15.
Long-context items declare ``filler_words`` and the haystack is synthesised
deterministically from a fixed seed — same bytes every run, on every machine.

With ``--blind``, each response is written to ``responses/<opaque>.md`` with the
label->arm mapping held in ``mapping.json``. Score the responses first, join
after. Judging one's own candidate is defensible only if the judging is blind.

**The blind guarantee covers the response files only.** ``results.json`` must
carry both ``model`` and ``blind_label`` — the timing analysis is per-arm and
would be impossible otherwise — so anyone who opens it has the mapping. The
discipline is procedural, not cryptographic: *do not open ``results.json`` or
``mapping.json`` until the content scores are written down.* When that has to be
violated (during #486 the truncation triage required per-arm token counts before
judging could start at all), say so in the write-up and stop calling the result
blind. An overstated method is worse than an honest single-judge one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("bench_agentic")

# UTF-8 stdout so the table renders under captured/redirected runs on Windows
# (cp1252 fallback otherwise throws on the box-drawing chars).
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

from _lib import no_window_flags  # noqa: E402

DEFAULT_PROMPTS = Path(__file__).resolve().parent / "bench_agentic_prompts.json"

# Deterministic filler for the long-context haystack. A fixed seed keeps the
# synthesised prompt byte-identical across runs and machines, so a long-context
# number from today is comparable with one from a month ago.
FILLER_SEED = 486
FILLER_VOCAB = [
    "the", "hub", "routes", "requests", "between", "local", "models", "and",
    "clients", "over", "loopback", "while", "the", "registry", "resolves",
    "each", "alias", "to", "a", "backend", "process", "that", "owns", "one",
    "port", "on", "this", "machine", "under", "a", "supervisor",
]

# Per-vendor sampling. NVIDIA documents temp 1.0 / top_p 1.0 for general chat
# and 0.6 / 0.95 for tool calling; Qwen's published guidance for the thinking
# path is 0.6 / 0.95. Running each model the way its vendor says to run it is
# the deployment-realistic comparison; --matched-temp overrides both to a
# single value as a control, to prove the ranking is not a sampling artifact.
VENDOR_SAMPLING: Dict[str, Dict[str, float]] = {
    "nemotron": {"temperature": 1.0, "top_p": 1.0},
    "nemotron_tools": {"temperature": 0.6, "top_p": 0.95},
    "qwen": {"temperature": 0.6, "top_p": 0.95},
    "qwen_tools": {"temperature": 0.6, "top_p": 0.95},
}


# --------------------------------------------------------------------------- #
# environment probes
# --------------------------------------------------------------------------- #
def vram_used_mb() -> Optional[int]:
    """Discrete-GPU memory in use, MB. None when nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=no_window_flags(),
        )
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def _sampling_for(model: str, is_tool_prompt: bool) -> Dict[str, float]:
    family = "nemotron" if "nemotron" in model.lower() else "qwen"
    key = f"{family}_tools" if is_tool_prompt else family
    return dict(VENDOR_SAMPLING[key])


# --------------------------------------------------------------------------- #
# prompt construction
# --------------------------------------------------------------------------- #
def _filler(n_words: int) -> str:
    rng = random.Random(FILLER_SEED)
    return " ".join(rng.choice(FILLER_VOCAB) for _ in range(n_words))


def build_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """Expand a prompt spec into chat messages, synthesising any haystack."""
    user = item["user"]
    filler_words = int(item.get("filler_words", 0))
    if filler_words:
        # Needle placed at a fixed fraction so recall is measured at depth, not
        # at the edges where every model trivially succeeds.
        needle = item["needle"]
        pre = _filler(int(filler_words * 0.6))
        post = _filler(int(filler_words * 0.4))
        user = f"{pre}\n\n{needle}\n\n{post}\n\n{user}"
    msgs: List[Dict[str, str]] = []
    if item.get("system"):
        msgs.append({"role": "system", "content": item["system"]})
    msgs.append({"role": "user", "content": user})
    return msgs


# --------------------------------------------------------------------------- #
# one measured call
# --------------------------------------------------------------------------- #
def _measure_ttft(base: str, model: str, body: Dict[str, Any],
                  timeout: float) -> Optional[float]:
    """Client-visible time to the first rendered character, via streaming.

    Worth measuring separately because the hub strips ``<think>`` blocks
    server-side: on a thinking arm the caller sees *nothing* until reasoning
    ends, so TTFT is dominated by reasoning time, not by prompt eval. Measured
    at 1.005 s of a 1.571 s total (64%) on nemotron4b during #486 — which is
    the whole argument for a no-think lane, and is invisible if you only look
    at tok/s.
    """
    stream_body = dict(body, stream=True)
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", f"{base}/v1/chat/completions",
                               json=stream_body) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        evt = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    for choice in evt.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content") or delta.get("tool_calls"):
                            return time.perf_counter() - t0
    except Exception:  # noqa: BLE001
        return None
    return None


def call_once(base: str, model: str, item: Dict[str, Any], max_tokens: int,
              timeout: float, matched_temp: Optional[float],
              with_ttft: bool = True) -> Dict[str, Any]:
    """Measure one completion end to end through the hub.

    Two requests: a non-streaming one for *exact* token counts and llama.cpp's
    own ``timings`` block (the hub does not emit ``usage`` on streamed
    responses, and a chars/4 estimate is not good enough to base a role swap
    on), plus an optional streaming one purely to time the first visible
    character. ``total_s`` from the non-streaming call is the headline.
    """
    msgs = build_messages(item)
    is_tool = bool(item.get("tools"))
    sampling = _sampling_for(model, is_tool)
    if matched_temp is not None:
        sampling = {"temperature": matched_temp, "top_p": 0.95}

    body: Dict[str, Any] = {
        "model": model,
        "messages": msgs,
        "max_tokens": max_tokens,
        **sampling,
    }
    if is_tool:
        body["tools"] = item["tools"]

    vram_before = vram_used_mb()
    error: Optional[str] = None
    data: Dict[str, Any] = {}

    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(f"{base}/v1/chat/completions", json=body)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    total_s = time.perf_counter() - t0
    vram_after = vram_used_mb()

    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    answer = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    finish_reason = choice.get("finish_reason")
    usage = data.get("usage") or {}
    timings = data.get("timings") or {}

    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = usage.get("prompt_tokens")

    ttft = _measure_ttft(base, model, body, timeout) if (with_ttft and not error) else None

    truncated = finish_reason == "length"
    ok = error is None and not truncated and (bool(answer.strip()) or bool(tool_calls))

    return {
        "id": item["id"],
        "category": item["category"],
        "model": model,
        "ok": ok,
        "error": error,
        "truncated": truncated,
        "finish_reason": finish_reason,
        # Client-visible latency to the first rendered character (streaming).
        "ttft_s": round(ttft, 4) if ttft is not None else None,
        # Request sent -> complete answer in hand. The headline number.
        "total_s": round(total_s, 4),
        # Exact, from the backend — not estimated.
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "gen_tok_s": round(timings.get("predicted_per_second"), 2)
        if timings.get("predicted_per_second") else None,
        "prompt_tok_s": round(timings.get("prompt_per_second"), 2)
        if timings.get("prompt_per_second") else None,
        "prompt_eval_s": round(timings.get("prompt_ms", 0) / 1000.0, 4) or None,
        "predicted_s": round(timings.get("predicted_ms", 0) / 1000.0, 4) or None,
        "answer_chars": len(answer),
        "sampling": sampling,
        "vram_before_mb": vram_before,
        "vram_after_mb": vram_after,
        "tool_calls": tool_calls,
        "answer": answer,
    }


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(base: str, models: List[str], items: List[Dict[str, Any]], reps: int,
        max_tokens: int, timeout: float, out_dir: Path, blind: bool,
        matched_temp: Optional[float], vram_guard: Optional[int]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    resp_dir = out_dir / "responses"
    resp_dir.mkdir(exist_ok=True)

    baseline_vram = vram_used_mb()
    log.info("baseline VRAM: %s MB", baseline_vram)

    rows: List[Dict[str, Any]] = []
    mapping: Dict[str, Dict[str, str]] = {}
    spoiled: List[str] = []

    for model in models:
        log.info("")
        log.info("=== %s ===", model)
        for item in items:
            best: Optional[Dict[str, Any]] = None
            for rep in range(reps):
                res = call_once(base, model, item, max_tokens, timeout, matched_temp)
                # Median-of-reps is overkill for a 1-2 rep run; keep the fastest
                # successful attempt, which is the warm steady state a caller
                # would see, and fall back to the last attempt if all failed.
                if best is None or (res["ok"] and not best["ok"]) or (
                        res["ok"] and best["ok"] and res["total_s"] < best["total_s"]):
                    best = res
            assert best is not None

            if vram_guard and baseline_vram is not None:
                after = best.get("vram_after_mb")
                if after is not None and after - baseline_vram > vram_guard:
                    spoiled.append(f"{model}/{item['id']}")

            flag = "ok " if best["ok"] else ("TRUNC" if best["truncated"] else "ERR")
            log.info("  %-5s %-24s %6.2fs  ttft=%-7s gen=%-7s tok=%-5s %s",
                     flag, item["id"], best["total_s"],
                     best["ttft_s"], best["gen_tok_s"], best["completion_tokens"],
                     f"reason={best['finish_reason']}" if not best["ok"] else "")

            # Blind label: stable hash so a re-run overwrites the same file, but
            # opaque enough that the arm is not guessable from the filename.
            key = f"{model}|{item['id']}"
            label = hashlib.sha256(key.encode()).hexdigest()[:10] if blind else key.replace("|", "__")
            mapping[label] = {"model": model, "prompt_id": item["id"],
                              "category": item["category"]}
            (resp_dir / f"{label}.md").write_text(
                f"# {label}\n\n## Prompt ({item['category']})\n\n{item['user']}\n\n"
                f"## Response\n\n{best['answer']}\n\n"
                + (f"## Tool calls\n\n```json\n{json.dumps(best['tool_calls'], indent=2)}\n```\n"
                   if best["tool_calls"] else ""),
                encoding="utf-8")

            row = {k: v for k, v in best.items() if k != "answer"}
            row["blind_label"] = label
            rows.append(row)

    (out_dir / "results.json").write_text(
        json.dumps({"base_url": base, "models": models, "reps": reps,
                    "max_tokens": max_tokens, "matched_temp": matched_temp,
                    "baseline_vram_mb": baseline_vram, "rows": rows},
                   indent=2), encoding="utf-8")
    (out_dir / "mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    _summary(rows, models)

    if spoiled:
        log.error("")
        log.error("VRAM GUARD TRIPPED on %d sample(s): %s", len(spoiled), ", ".join(spoiled))
        log.error("Another model likely loaded mid-run (gemma4_26b is on-demand at "
                  "~13.4 GB). Those timings are not trustworthy — re-run them.")
        return 2
    return 0


def _median(vals: List[Any]) -> Optional[float]:
    clean = [v for v in vals if v is not None]
    return statistics.median(clean) if clean else None


def _fmt(v: Optional[float], places: int = 2) -> str:
    return "-" if v is None else f"{v:.{places}f}"


def _summary(rows: List[Dict[str, Any]], models: List[str]) -> None:
    log.info("")
    log.info("%-26s %7s %6s %9s %9s %9s %9s",
             "model", "ok/n", "trunc", "total_s", "ttft_s", "gen_tok/s", "out_tok")
    for model in models:
        mr = [r for r in rows if r["model"] == model]
        good = [r for r in mr if r["ok"]]
        trunc = sum(1 for r in mr if r["truncated"])
        log.info("%-26s %7s %6d %9s %9s %9s %9s",
                 model, f"{len(good)}/{len(mr)}", trunc,
                 _fmt(_median([r["total_s"] for r in good])),
                 _fmt(_median([r["ttft_s"] for r in good])),
                 _fmt(_median([r["gen_tok_s"] for r in good])),
                 _fmt(_median([r["completion_tokens"] for r in good]), 0))
    log.info("")
    log.info("All medians over successful samples. total_s is the headline: "
             "request sent -> complete answer in hand.")
    log.info("gen_tok/s and out_tok are exact (backend usage/timings) and "
             "INCLUDE reasoning tokens — which is why a thinking arm can post "
             "a higher tok/s and still be slower to a usable answer.")
    log.info("The reasoning tax is the think-arm vs no-think-arm delta, "
             "measured directly rather than estimated per response.")


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # httpx logs every request at INFO; one line per call would bury the table.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="measure one or more model arms")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", action="append", required=True,
                   help="model id or alias; repeatable, one arm each")
    p.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    p.add_argument("--out", type=Path, required=True, help="output dir (use .scratch/)")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="1024 default: reasoning models exhaust smaller budgets "
                        "inside <think> and return finish_reason=length")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--blind", action="store_true",
                   help="opaque response filenames; mapping held in mapping.json")
    p.add_argument("--matched-temp", type=float, default=None,
                   help="override vendor sampling with one temperature (control pass)")
    p.add_argument("--vram-guard", type=int, default=4000,
                   help="fail if VRAM rises this many MB above baseline mid-run")
    p.add_argument("--category", action="append",
                   help="only run these categories; repeatable")

    args = ap.parse_args(argv)

    spec = json.loads(args.prompts.read_text(encoding="utf-8"))
    items = spec["prompts"]
    if args.category:
        items = [i for i in items if i["category"] in args.category]
    if not items:
        log.error("no prompts selected")
        return 1

    log.info("%d prompt(s), %d model(s), reps=%d, max_tokens=%d",
             len(items), len(args.model), args.reps, args.max_tokens)
    return run(args.base_url.rstrip("/"), args.model, items, args.reps,
               args.max_tokens, args.timeout, args.out, args.blind,
               args.matched_temp, args.vram_guard)


if __name__ == "__main__":
    raise SystemExit(main())
