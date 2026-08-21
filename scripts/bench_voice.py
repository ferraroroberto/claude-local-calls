"""Benchmark voice backends (STT + TTS) on any hub, for placement decisions.

One reusable harness the fleet uses to decide where each voice model should
live (issue #343) and to re-verify after a ``/swap-model`` or a host/GPU swap.
It targets a hub by ``--base-url``, so the *same* tool measures any machine:

    # STT — whisper on the tower vs the gaming satellite (same clip set)
    python scripts/bench_voice.py stt --base-url http://127.0.0.1:8000  --model whisper  --clips-dir .scratch/voice-bench
    python scripts/bench_voice.py stt --base-url http://192.168.1.13:8000 --model whisper  --clips-dir .scratch/voice-bench
    # STT — parakeet on the Mac Mini
    python scripts/bench_voice.py stt --base-url http://192.168.1.11:8000 --model parakeet --clips-dir .scratch/voice-bench

    # TTS — orpheus on the tower vs gaming
    python scripts/bench_voice.py tts --base-url http://127.0.0.1:8000  --model orpheus
    python scripts/bench_voice.py tts --base-url http://192.168.1.13:8000 --model orpheus

Both modes hit the hub's OpenAI-shaped endpoints (``/v1/audio/transcriptions``
and ``/v1/audio/speech``) so the numbers are end-to-end through the hub — the
same path real clients take — not a private backend port.

**STT metrics.** Per clip: wall-clock (median of ``--reps`` warm requests),
RTFx (audio_seconds / processing_seconds — higher is faster than real time),
WER vs the archived reference transcript, and domain-jargon survival.

    WER is *comparative, not absolute*: the reference is the daily-driver
    whisper transcript archived next to each clip, which structurally favours
    whisper. Normalisation lowercases, strips punctuation, and drops filler
    words; it does NOT normalise number words ("64" vs "sixty-four"), so a few
    points of WER are formatting, not error. The RTFx / latency numbers are the
    objective ones. Same caveat as docs/parakeet-asr-evaluation.md.

**TTS metrics.** Per sentence: wall-clock (median of ``--reps`` warm requests),
synthesized audio duration, and RTF (audio / wall — >1 is faster than real
time). Generalises scripts/bench_orpheus.py --hub-e2e to any model/host.

Clip WAVs are personal dictation audio — keep them under a gitignored dir
(.scratch/); this script never writes them into the tree.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import statistics
import sys
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

log = logging.getLogger("bench_voice")

# UTF-8 stdout so the table renders under captured/redirected runs on Windows
# (cp1252 fallback otherwise throws on the box-drawing chars).
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

# Domain terms the hub's #90/#91 boosting line of work cares about — tracked so
# a placement swap can't silently regress jargon recognition. Matched as
# normalised substrings; only terms present in a clip's reference are scored.
DEFAULT_JARGON = [
    "claude code", "yolo", "orpheus", "sonnet", "chatterbox", "kokoro",
    "design md", "issue add", "shift tab", "langfuse", "whisper",
]

# Two sentences mirroring docs/parakeet-asr-evaluation.md's TTS methodology: a
# tiny clip (cold-path floor) and a ~15-word utterance (steady-state).
DEFAULT_TTS_SENTENCES = [
    "This is a test.",
    "Arming the perimeter now; the energy summary shows a nominal load across every connected device.",
]

_FILLERS = {"uh", "um", "uhh", "umm", "hmm", "mm", "mhm", "eh", "ah"}


# --------------------------------------------------------------------------- #
# text normalisation + WER
# --------------------------------------------------------------------------- #
def _normalise(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)          # drop punctuation
    words = [w for w in text.split() if w and w not in _FILLERS]
    return words


def _wer(ref: str, hyp: str) -> float:
    """Word error rate = word-level Levenshtein / max(len(ref), 1)."""
    r = _normalise(ref)
    h = _normalise(hyp)
    if not r:
        return 0.0 if not h else 1.0
    # DP edit distance over word lists.
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cost = 0 if rw == hw else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(h)] / len(r)


def _jargon_hits(ref: str, hyp: str, terms: List[str]) -> Tuple[int, int, List[str]]:
    """(survived, present_in_ref, missed_terms) — only terms in the reference
    are scored, so a clip that never says 'YOLO' doesn't count against a model."""
    rn = " ".join(_normalise(ref))
    hn = " ".join(_normalise(hyp))
    present = [t for t in terms if t in rn]
    survived = [t for t in present if t in hn]
    missed = [t for t in present if t not in hn]
    return len(survived), len(present), missed


def _wav_duration_seconds(raw: bytes) -> Optional[float]:
    try:
        with wave.open(io.BytesIO(raw)) as w:
            frames = w.getnframes()
            rate = w.getframerate()
            return frames / rate if rate else None
    except (wave.Error, EOFError, OSError):
        return None


# --------------------------------------------------------------------------- #
# STT
# --------------------------------------------------------------------------- #
def _transcribe(base: str, model: str, wav_path: Path, timeout: float) -> str:
    url = f"{base}/v1/audio/transcriptions"
    with wav_path.open("rb") as fh:
        files = {"file": (wav_path.name, fh, "audio/wav")}
        data = {"model": model, "response_format": "text"}
        r = httpx.post(url, files=files, data=data, timeout=timeout)
    r.raise_for_status()
    body = r.text.strip()
    # response_format=text should be raw text, but some servers answer JSON.
    if body.startswith("{"):
        try:
            return str(json.loads(body).get("text", body)).strip()
        except json.JSONDecodeError:
            return body
    return body


def run_stt(base: str, model: str, clips_dir: Path, reps: int,
            jargon: List[str], timeout: float) -> Dict:
    clips = sorted(clips_dir.glob("*.wav"))
    if not clips:
        raise SystemExit(f"no .wav clips in {clips_dir}")
    log.info("STT %s @ %s — %d clips, reps=%d", model, base, len(clips), reps)

    rows: List[Dict] = []
    for wav in clips:
        ref_path = wav.with_suffix(".txt")
        ref = ref_path.read_text(encoding="utf-8").strip() if ref_path.exists() else ""
        audio_s = _wav_duration_seconds(wav.read_bytes()) or 0.0

        # One warmup (not measured) then `reps` measured; keep the last text.
        text = ""
        walls: List[float] = []
        try:
            _transcribe(base, model, wav, timeout)
            for _ in range(reps):
                t0 = time.perf_counter()
                text = _transcribe(base, model, wav, timeout)
                walls.append(time.perf_counter() - t0)
        except httpx.HTTPError as exc:
            log.warning("   %s FAILED: %s", wav.name, exc)
            rows.append({"clip": wav.name, "error": str(exc), "audio_s": audio_s})
            continue

        proc = statistics.median(walls)
        wer = _wer(ref, text) if ref else None
        surv, present, missed = _jargon_hits(ref, text, jargon)
        rtfx = (audio_s / proc) if proc > 0 else 0.0
        rows.append({
            "clip": wav.name, "audio_s": round(audio_s, 1),
            "proc_s": round(proc, 3), "rtfx": round(rtfx, 1),
            "wer": round(wer, 3) if wer is not None else None,
            "jargon_survived": surv, "jargon_present": present,
            "jargon_missed": missed, "text": text,
        })
        log.info("   %-20s %5.1fs audio | %6.3fs | %6.1fx RTF | WER %s | jargon %d/%d%s",
                 wav.name, audio_s, proc, rtfx,
                 f"{wer:.1%}" if wer is not None else "n/a",
                 surv, present, f"  missed={missed}" if missed else "")

    ok = [r for r in rows if "error" not in r]
    total_audio = sum(r["audio_s"] for r in ok)
    total_proc = sum(r["proc_s"] for r in ok)
    wers = [r["wer"] for r in ok if r["wer"] is not None]
    js = sum(r["jargon_survived"] for r in ok)
    jp = sum(r["jargon_present"] for r in ok)
    summary = {
        "mode": "stt", "base_url": base, "model": model, "clips": len(clips),
        "reps": reps, "total_audio_s": round(total_audio, 1),
        "total_proc_s": round(total_proc, 3),
        "overall_rtfx": round(total_audio / total_proc, 1) if total_proc else 0.0,
        "mean_wer": round(statistics.mean(wers), 3) if wers else None,
        "jargon_survival": f"{js}/{jp}",
        "rows": rows,
    }
    log.info("=" * 72)
    log.info("STT %s @ %s: overall %.1fx RTF | mean WER %s | jargon %d/%d",
             model, base, summary["overall_rtfx"],
             f"{summary['mean_wer']:.1%}" if summary["mean_wer"] is not None else "n/a",
             js, jp)
    log.info("=" * 72)
    return summary


# --------------------------------------------------------------------------- #
# TTS
# --------------------------------------------------------------------------- #
def _speak(base: str, model: str, text: str, voice: str, timeout: float) -> bytes:
    url = f"{base}/v1/audio/speech"
    body = {"model": model, "input": text, "voice": voice, "response_format": "wav"}
    r = httpx.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    return r.content


def run_tts(base: str, model: str, sentences: List[str], reps: int,
            voice: str, timeout: float) -> Dict:
    log.info("TTS %s @ %s — %d sentences, reps=%d, voice=%s",
             model, base, len(sentences), reps, voice)
    rows: List[Dict] = []
    for text in sentences:
        raw = b""
        walls: List[float] = []
        try:
            raw = _speak(base, model, text, voice, timeout)  # warmup
            for _ in range(reps):
                t0 = time.perf_counter()
                raw = _speak(base, model, text, voice, timeout)
                walls.append(time.perf_counter() - t0)
        except httpx.HTTPError as exc:
            log.warning("   FAILED (%r): %s", text[:30], exc)
            rows.append({"text": text, "error": str(exc)})
            continue
        proc = statistics.median(walls)
        audio_s = _wav_duration_seconds(raw) or 0.0
        rtf = (audio_s / proc) if proc > 0 else 0.0
        rows.append({
            "chars": len(text), "audio_s": round(audio_s, 2),
            "proc_s": round(proc, 3), "rtf": round(rtf, 2),
            "bytes": len(raw), "text": text,
        })
        log.info("   %3d chars | %5.2fs audio | %6.3fs synth | %5.2fx RT | %d bytes",
                 len(text), audio_s, proc, rtf, len(raw))

    ok = [r for r in rows if "error" not in r]
    summary = {
        "mode": "tts", "base_url": base, "model": model, "voice": voice,
        "reps": reps,
        "median_synth_s": round(statistics.median([r["proc_s"] for r in ok]), 3) if ok else None,
        "rows": rows,
    }
    log.info("=" * 72)
    log.info("TTS %s @ %s: median synth %s",
             model, base,
             f"{summary['median_synth_s']:.3f}s" if summary["median_synth_s"] is not None else "n/a")
    log.info("=" * 72)
    return summary


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    p_stt = sub.add_parser("stt", help="benchmark speech-to-text")
    p_stt.add_argument("--base-url", required=True)
    p_stt.add_argument("--model", required=True)
    p_stt.add_argument("--clips-dir", required=True, type=Path)
    p_stt.add_argument("--reps", type=int, default=2)
    p_stt.add_argument("--timeout", type=float, default=300.0)
    p_stt.add_argument("--json", type=Path, help="write the summary JSON here")

    p_tts = sub.add_parser("tts", help="benchmark text-to-speech")
    p_tts.add_argument("--base-url", required=True)
    p_tts.add_argument("--model", required=True)
    p_tts.add_argument("--voice", default="tara")
    p_tts.add_argument("--text", action="append", help="sentence(s); repeatable")
    p_tts.add_argument("--reps", type=int, default=3)
    p_tts.add_argument("--timeout", type=float, default=300.0)
    p_tts.add_argument("--json", type=Path, help="write the summary JSON here")

    args = ap.parse_args(argv)

    if args.mode == "stt":
        summary = run_stt(args.base_url.rstrip("/"), args.model, args.clips_dir,
                          args.reps, DEFAULT_JARGON, args.timeout)
    else:
        sentences = args.text or DEFAULT_TTS_SENTENCES
        summary = run_tts(args.base_url.rstrip("/"), args.model, sentences,
                          args.reps, args.voice, args.timeout)

    if args.json:
        args.json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        log.info("wrote %s", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
