"""Orpheus TTS engine — Orpheus-3B run as a GGUF on a loopback
``llama-server`` child (reusing the vendored binary) whose emitted audio
tokens are decoded with the SNAC neural codec in-process.

The most expressive local voice, but heavier. Orpheus's reference runtime
(vLLM) has no usable Windows build, hence the llama.cpp + SNAC route. Heavy
deps (torch, snac) are imported **lazily inside** ``load``/``synthesize`` so
this module imports cleanly under pytest/CI where they are absent.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional

import httpx

from ..model_registry import Model
from ..process_supervisor import ProcessSupervisor
from ..win_job import assign_pid, create_kill_on_close_job, terminate_and_close
from .common import (
    PROJECT_ROOT,
    SpeechRequest,
    TTSEngine,
    TTS_LANGUAGE_LABELS,
    TTS_SAMPLE_TEXT,
    _wrap_on_words,
    resolve_device,
    voice_option,
)

log = logging.getLogger(__name__)

# SNAC code book size — each audio token, after offset removal, lands in
# [0, 4096). Frames are 7 tokens that fan out into SNAC's 3 hierarchical
# layers (1 / 2 / 4 codes). Orpheus token ids carry a +10 base offset and a
# per-position +4096*(i%7) stride (canopyai/Orpheus convention).
_SNAC_CODEBOOK = 4096
_ORPHEUS_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")

# A single llama-server ``/completion`` is capped at ``_N_PREDICT`` generated
# audio tokens. 4096 SNAC tokens ≈ 49.6 s of speech (4096 ÷ 7 codes/frame ×
# 2048 samples/frame ÷ 24 kHz). Synthesising longer than that in one request
# silently truncates the audio (issue #130), so long input is split into
# chunks that each comfortably fit under the cap and then concatenated.
#
# Orpheus emits roughly 18 characters of text per second of speech, so the
# ~49.6 s ceiling is ~900 characters. ``_MAX_CHARS_PER_CHUNK`` budgets each
# chunk at ~27 s — generous headroom against rate variance, and small enough
# that single generations stay coherent (very long ones also degrade quality).
_N_PREDICT = 4096
_MAX_CHARS_PER_CHUNK = 480
# Split on whitespace that follows sentence-ending punctuation, keeping the
# punctuation with its sentence.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class OrpheusEngine(TTSEngine):
    """Orpheus-3B: GGUF on a loopback llama-server child + SNAC decode."""

    AVAILABLE_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
    DEFAULT_VOICE = "tara"

    @classmethod
    def capabilities(cls) -> Dict[str, Any]:
        female = {"tara", "leah", "jess", "mia", "zoe"}
        return {
            "languages": [{"id": "en-US", "label": TTS_LANGUAGE_LABELS["en-US"]}],
            "voices": [
                voice_option(voice, voice.title(), "en-US", "female" if voice in female else "male")
                for voice in cls.AVAILABLE_VOICES
            ],
            "default_voice": cls.DEFAULT_VOICE,
            "default_language": "en-US",
            "sample_text": {"en-US": TTS_SAMPLE_TEXT["en-US"]},
            "controls": {"speed": False, "stream": True, "exaggeration": False, "cfg_weight": False},
        }
    LLAMA_READY_DEADLINE_S = 180.0  # 3B cold-load on first start

    def __init__(self, model: Model, device: str = "auto") -> None:
        self.model_row = model
        self.device_arg = device
        self.device = "cpu"
        self.snac = None
        self.proc: Optional[subprocess.Popen] = None
        self._job = None  # Windows Job handle: kills the llama child when we die
        self.internal_port = int(model.internal_port or 18093)
        self.sample_rate = 24000
        # Persistent client for the loopback /completion calls to our own
        # llama-server child. Constructing an httpx client is ~0.26s on Windows
        # (#165), so a per-request client would tax every synthesis; one
        # reused client costs ~1ms. This engine runs in the tts_server process,
        # not the hub, so it can't use the hub's shared client.
        self._client: Optional[httpx.Client] = None

    # ---- lifecycle ----

    def load(self) -> None:
        import torch  # noqa: F401  (presence check; SNAC needs it)
        from snac import SNAC

        self.device = resolve_device(self.device_arg)
        log.info("loading SNAC codec on %s …", self.device)
        self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(self.device)
        self._client = httpx.Client(timeout=300.0)
        self._spawn_llama()
        self._wait_llama_ready()
        log.info("Orpheus ready (llama-server :%d, SNAC on %s)", self.internal_port, self.device)

    def _reclaim_internal_port(self) -> None:
        """Kill any stale listener on our internal port before spawning.

        Defends against a llama-server orphaned by a previous session that
        died before the job object could reap it — without this, the fresh
        spawn can't bind ``internal_port``. The port is hub-private, so
        killing whatever holds it is safe.
        """
        from ..server_process import kill_pid, snapshot_listening_pids

        for pid in snapshot_listening_pids().get(self.internal_port, []) or []:
            log.warning("reclaiming stale process %s on internal port %s", pid, self.internal_port)
            kill_pid(int(pid))

    def _spawn_llama(self) -> None:
        from ..backend_process import llama_server_binary, VENDOR_LLAMA
        from ..server_process import WIN_NEW_GROUP

        self._reclaim_internal_port()
        bin_path = llama_server_binary()
        if not bin_path.exists():
            raise RuntimeError(
                f"llama-server not found at {bin_path} - run scripts/install_llama_cpp.py"
            )
        if not self.model_row.model_path:
            raise RuntimeError("orpheus row has no model_path (GGUF)")
        gguf = (PROJECT_ROOT / self.model_row.model_path).resolve()
        if not gguf.exists():
            raise RuntimeError(
                f"Orpheus GGUF not found at {gguf} - "
                f"run scripts/download_models.py --only {self.model_row.id}"
            )
        # Throughput note (issue #105): on the reference GPU this 3B Q4_K_M
        # generates ~150 tok/s, which sets the total synthesis time. That rate
        # is memory-bandwidth bound, NOT a missing flag — the model is fully
        # offloaded (-ngl 99) and llama.cpp already auto-enables flash
        # attention, so --flash-attn / -b/-ub batch / --no-mmap leave the rate
        # unchanged (measured: scripts/bench_orpheus.py). ~150 tok/s is ~65% of
        # this card's bandwidth ceiling for a ~1.94 GB resident model; the only
        # faster route is a lower quant, which would regress SNAC audio quality.
        # Perceived latency is handled by streaming (#102). Full analysis +
        # before/after numbers: docs/orpheus-throughput.md.
        #
        # -c 8192 (not smaller): KV is sized by n_ctx, not per slot, so a
        # shorter context would not free meaningful VRAM, and 8192 is needed to
        # hold longer inputs (Orpheus emits ~107 audio tokens per second of
        # speech, so 8192 ≈ 76 s of audio headroom).
        cmd = [
            str(bin_path),
            "-m", str(gguf),
            "--host", "127.0.0.1",
            "--port", str(self.internal_port),
            "-c", "8192",
            "-ngl", "99",
            "--no-webui",
        ]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        if sys.platform == "win32":
            env["PATH"] = str(VENDOR_LLAMA) + os.pathsep + env.get("PATH", "")
        log.info("spawning Orpheus llama-server: %s", " ".join(cmd))
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            creationflags=WIN_NEW_GROUP,
        )
        # Tie the child's lifetime to ours: when this tts_server process
        # dies (even via the hub's TerminateProcess), the OS closes the job
        # handle and reaps the llama-server, freeing its VRAM and port.
        self._job = create_kill_on_close_job()
        if not assign_pid(self._job, self.proc.pid) and self._job is not None:
            log.warning("could not assign llama-server to job object; relying on close()")
        t = threading.Thread(target=self._forward_stdout, args=(self.proc,), daemon=True)
        t.start()

    def _forward_stdout(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            sys.stdout.write(f"[orpheus-llama] {raw.rstrip()}\n")
            sys.stdout.flush()

    def _wait_llama_ready(self) -> None:
        deadline = time.monotonic() + self.LLAMA_READY_DEADLINE_S
        url = f"http://127.0.0.1:{self.internal_port}/health"
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError("Orpheus llama-server child exited during startup")
            try:
                # self._client is a persistent client set in load() — see its
                # comment for why this can't be a per-call httpx.get (#450).
                r = self._client.get(url, timeout=2.0)  # type: ignore[union-attr]
                if r.status_code == 200:
                    return
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)
        self.close()
        raise RuntimeError(
            f"Orpheus llama-server did not become ready within {self.LLAMA_READY_DEADLINE_S:.0f}s"
        )

    def ready(self) -> bool:
        return self.snac is not None and self.proc is not None and self.proc.poll() is None

    # ---- synthesis ----

    @classmethod
    def _voice_for(cls, voice: str) -> str:
        requested = (voice or "").strip()
        if not requested or requested.lower() in ("default", "none"):
            return cls.DEFAULT_VOICE
        if requested not in cls.AVAILABLE_VOICES:
            raise ValueError(f"unsupported Orpheus voice: {requested}")
        return requested

    def validate_voice(self, voice: str) -> None:
        self._voice_for(voice)

    def _prompt_for(self, req: SpeechRequest, text: Optional[str] = None) -> str:
        voice = self._voice_for(req.voice)
        body = req.text if text is None else text
        # Orpheus-FastAPI prompt convention for the llama.cpp route: the
        # end marker is the model's <|eot_id|> special token (not <|eot|>).
        return f"<|audio|>{voice}: {body}<|eot_id|>"

    @staticmethod
    def _split_into_chunks(text: str, budget: int = _MAX_CHARS_PER_CHUNK) -> List[str]:
        """Split ``text`` into chunks no longer than ``budget`` characters.

        Input that already fits in a single chunk is returned **unchanged**
        (``[text]``) so short synthesis is byte-for-byte identical to the
        pre-chunking behaviour. Longer input is broken on sentence
        boundaries and greedily packed; a single sentence over budget is
        hard-wrapped on word boundaries so no chunk can exceed ``budget``.
        """
        if not text or not text.strip():
            return []
        if len(text) <= budget:
            return [text]
        chunks: List[str] = []
        current = ""
        for sentence in (s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip())):
            if not sentence:
                continue
            pieces = [sentence] if len(sentence) <= budget else _wrap_on_words(sentence, budget)
            for piece in pieces:
                if not current:
                    current = piece
                elif len(current) + 1 + len(piece) <= budget:
                    current += " " + piece
                else:
                    chunks.append(current)
                    current = piece
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _completion_payload(prompt: str, stream: bool) -> dict:
        return {
            "prompt": prompt,
            "n_predict": _N_PREDICT,
            "temperature": 0.6,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "cache_prompt": True,
            "stream": stream,
        }

    def synthesize(self, req: SpeechRequest):
        import numpy as np

        if not self.ready():
            raise RuntimeError("Orpheus not loaded")
        # Long input is split into per-chunk /completion calls (each under the
        # n_predict cap) and the decoded PCM segments concatenated in order;
        # short input is a single chunk, so this is identical to before (#130).
        segments = [self._synthesize_chunk(req, chunk) for chunk in self._split_into_chunks(req.text)]
        segments = [seg for seg in segments if seg.size]
        if not segments:
            log.warning("Orpheus emitted no audio tokens for input")
            return np.zeros(0, dtype=np.float32)
        return segments[0] if len(segments) == 1 else np.concatenate(segments)

    def _synthesize_chunk(self, req: SpeechRequest, chunk: str):
        """Synthesise one text chunk through a single buffered /completion."""
        import numpy as np

        prompt = self._prompt_for(req, chunk)
        url = f"http://127.0.0.1:{self.internal_port}/completion"
        assert self._client is not None
        r = self._client.post(
            url, json=self._completion_payload(prompt, stream=False), timeout=300.0
        )
        r.raise_for_status()
        content = r.json().get("content", "")
        codes = self._parse_tokens(content)
        if not codes:
            return np.zeros(0, dtype=np.float32)
        return self._decode_snac(codes)

    # ---- streaming synthesis ----

    def synthesize_stream(self, req: SpeechRequest):
        """Decode SNAC frames incrementally as the llama-server streams them.

        Mirrors the canopyai/Orpheus-FastAPI ``speechpipe`` sliding window:
        on every completed 7-token frame past a 4-frame warmup, decode the
        newest 28-token window and emit its artefact-free ``[2048:4096]``
        segment (~85 ms at 24 kHz). Short inputs that never reach the window
        fall back to a single whole-clip decode so they still produce audio.
        """
        if not self.ready():
            raise RuntimeError("Orpheus not loaded")
        # Stream chunk after chunk back-to-back: long input is split so each
        # chunk stays under the n_predict cap (#130), while a short single
        # chunk streams exactly as before — time-to-first-audio is unchanged.
        for chunk in self._split_into_chunks(req.text):
            yield from self._stream_chunk(req, chunk)

    def _stream_chunk(self, req: SpeechRequest, chunk: str):
        """Stream one text chunk's PCM via the sliding-window SNAC decode."""
        prompt = self._prompt_for(req, chunk)
        buffer: List[int] = []
        emitted = False
        for tid in self._iter_token_ids(self._stream_completion(prompt)):
            buffer.append(tid)
            if len(buffer) % 7 == 0 and len(buffer) >= 28:
                seg = self._decode_window(buffer[-28:])
                if seg.size:
                    emitted = True
                    yield seg
        if not emitted and buffer:
            audio = self._decode_snac(buffer)
            if audio.size:
                yield audio

    def _stream_completion(self, prompt: str) -> Iterator[str]:
        """Stream the llama-server ``/completion`` SSE, yielding each delta's
        ``content`` text as it arrives."""
        url = f"http://127.0.0.1:{self.internal_port}/completion"
        assert self._client is not None
        with self._client.stream(
            "POST", url, json=self._completion_payload(prompt, stream=True), timeout=300.0
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                if line.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(line)
                except (ValueError, TypeError):
                    continue
                piece = obj.get("content", "")
                if piece:
                    yield piece
                if obj.get("stop"):
                    break

    @staticmethod
    def _iter_token_ids(text_chunks: Iterable[str]) -> Iterator[int]:
        """Stream SNAC code ids from a sequence of completion text deltas.

        Buffers a partial ``<custom_token_N>`` tail across chunk boundaries
        (a token may be split mid-tag), applies the +10 / per-position
        stride offset, and skips control tokens (id <= 0) **without
        advancing the frame position** — the same validated semantics as
        :meth:`_parse_tokens`, made incremental.
        """
        carry = ""
        pos = 0
        for chunk in text_chunks:
            carry += chunk
            last_end = 0
            for m in _ORPHEUS_TOKEN_RE.finditer(carry):
                tid = int(m.group(1)) - 10 - ((pos % 7) * _SNAC_CODEBOOK)
                if tid > 0:
                    yield tid
                    pos += 1
                last_end = m.end()
            # Retain the unconsumed tail (a possible partial token) for the
            # next chunk; a closing '>' only appears at a token's true end,
            # so splitting mid-number can never match prematurely.
            carry = carry[last_end:]

    def _decode_window(self, window: List[int]):
        """Decode a 28-token (4-frame) sliding window and return its newest
        2048-sample segment. Empty array if any code is out of range."""
        import numpy as np

        frames = [window[7 * j: 7 * j + 7] for j in range(len(window) // 7)]
        if not frames or any(not all(0 <= c < _SNAC_CODEBOOK for c in f) for f in frames):
            return np.zeros(0, dtype=np.float32)
        return self._snac_decode(frames)[2048:4096]

    @staticmethod
    def _parse_tokens(text: str) -> List[int]:
        """Parse ``<custom_token_N>`` strings into SNAC code ids.

        Removes the +10 base offset and the per-position +4096*(pos%7)
        stride so each returned id lands in [0, 4096). The model prefixes
        the audio stream with a few small control tokens (e.g. 4, 5, 1);
        those decode to a non-positive id, so — matching the canopyai /
        Orpheus-FastAPI ``speechpipe`` decoder — they are skipped *without*
        advancing the frame position. That re-aligns the 7-token framing to
        the first real audio token; indexing every token instead would shift
        every frame and make all codes fall out of range (silent output).
        """
        ids: List[int] = []
        pos = 0
        for m in _ORPHEUS_TOKEN_RE.finditer(text):
            tid = int(m.group(1)) - 10 - ((pos % 7) * _SNAC_CODEBOOK)
            if tid > 0:
                ids.append(tid)
                pos += 1
        return ids

    def _decode_snac(self, code_list: List[int]):
        import numpy as np

        # Whole 7-token frames only; drop any frame with an out-of-range code.
        n_frames = len(code_list) // 7
        frames = [code_list[7 * j: 7 * j + 7] for j in range(n_frames)]
        frames = [f for f in frames if all(0 <= c < _SNAC_CODEBOOK for c in f)]
        if not frames:
            return np.zeros(0, dtype=np.float32)
        return self._snac_decode(frames)

    def _snac_decode(self, frames: List[List[int]]):
        """Run the SNAC codec on whole 7-token frames → mono float32 audio.

        Each frame fans out into SNAC's three hierarchical layers as
        ``[f0] / [f1,f4] / [f2,f3,f5,f6]``.
        """
        import torch

        layer_1: List[int] = []
        layer_2: List[int] = []
        layer_3: List[int] = []
        for f in frames:
            layer_1.append(f[0])
            layer_2.append(f[1]); layer_2.append(f[4])
            layer_3.append(f[2]); layer_3.append(f[3]); layer_3.append(f[5]); layer_3.append(f[6])

        dev = self.device
        codes = [
            torch.tensor(layer_1, device=dev).unsqueeze(0),
            torch.tensor(layer_2, device=dev).unsqueeze(0),
            torch.tensor(layer_3, device=dev).unsqueeze(0),
        ]
        with torch.inference_mode():
            audio = self.snac.decode(codes)
        return audio.detach().cpu().float().numpy().reshape(-1)

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
        proc = self.proc
        self.proc = None
        if proc is not None and proc.poll() is None:
            ok, msg = ProcessSupervisor.stop_popen(proc, terminate_timeout=8.0, kill_timeout=3.0)
            if not ok:
                log.warning("error stopping Orpheus llama-server: %s", msg)
        job = self._job
        self._job = None
        terminate_and_close(job)
        self.snac = None
