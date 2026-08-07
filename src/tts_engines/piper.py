"""Piper TTS engine — the standalone Piper binary with local ONNX voices.

Keeps a pool of **resident** ``piper.exe`` processes (one per (voice, speed)
combination, lazily spawned) so the ONNX voice is loaded once and reused,
instead of spawning — and re-loading — a fresh binary per request. A resident
process whose synthesis fails for any reason falls back to a single one-shot
subprocess for that request and is recycled.
"""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..model_registry import Model
from ..no_window import NO_WINDOW
from ..win_job import assign_pid, create_kill_on_close_job, terminate_and_close
from .common import PROJECT_ROOT, SpeechRequest, TTSEngine, TTS_LANGUAGE_LABELS, TTS_SAMPLE_TEXT, voice_option

log = logging.getLogger(__name__)


class _PiperProc:
    """One resident ``piper.exe`` for a fixed (voice model, length_scale).

    The whole point of resident mode: the ONNX voice is loaded **once** at
    spawn and reused across requests, instead of paying a fresh model load on
    every call (measured ~0.40 s → ~0.07 s per short utterance — the inference
    itself is ~0.06 s; the rest was startup tax). Utterances are fed as
    ``--json-input`` lines; piper prints each finished WAV's path on stdout
    (emitted even under ``--quiet``) as the per-utterance completion signal.

    A single piper process synthesises one utterance at a time, so
    :meth:`synth` serialises access with a lock — concurrent hub requests for
    the *same* (voice, speed) queue here, while a different combination runs on
    its own resident process. Death is detected via a sentinel pushed by the
    reader thread, so a crashed child is recycled on the next call.
    """

    def __init__(self, cmd: List[str], out_dir: Path, job) -> None:
        self._cmd = cmd
        self._out_dir = out_dir
        self._lock = threading.Lock()
        self._done: "queue.Queue[Optional[str]]" = queue.Queue()
        self._counter = 0
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=str(PROJECT_ROOT),
            creationflags=NO_WINDOW,
        )
        assign_pid(job, self._proc.pid)
        threading.Thread(target=self._reader, args=(self._proc,), daemon=True).start()

    def _reader(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if line:
                self._done.put(line)
        self._done.put(None)  # EOF — the child exited

    def alive(self) -> bool:
        return self._proc.poll() is None

    def synth(self, text: str, timeout: float = 60.0) -> bytes:
        """Synthesise one utterance and return its WAV bytes. Raises on any
        failure (dead child, broken pipe, timeout) so the caller can fall back
        to a one-shot subprocess and recycle this process."""
        with self._lock:
            if self._proc.poll() is not None:
                raise RuntimeError("piper resident process is not running")
            # Drain any stale completion lines (shouldn't happen under the lock,
            # but keeps one synth strictly paired with one stdout line).
            try:
                while True:
                    self._done.get_nowait()
            except queue.Empty:
                pass
            self._counter += 1
            out_path = self._out_dir / f"u{self._counter}.wav"
            line = json.dumps({"text": text.strip(), "output_file": str(out_path)}) + "\n"
            assert self._proc.stdin is not None
            try:
                self._proc.stdin.write(line.encode("utf-8"))
                self._proc.stdin.flush()
            except OSError as exc:
                raise RuntimeError(f"piper resident stdin write failed: {exc}")
            signal_line = self._done.get(timeout=timeout)
            if signal_line is None:
                raise RuntimeError("piper resident process exited mid-synthesis")
            try:
                return out_path.read_bytes()
            finally:
                try:
                    out_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def close(self) -> None:
        proc = self._proc
        if proc.poll() is not None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:  # noqa: BLE001
            pass


class PiperEngine(TTSEngine):
    """Piper standalone binary with local ONNX voices.

    Keeps a pool of **resident** ``piper.exe`` processes (one per (voice,
    speed) combination, lazily spawned) so the ONNX voice is loaded once and
    reused, instead of spawning — and re-loading — a fresh binary per request.
    A resident process whose synthesis fails for any reason falls back to a
    single one-shot subprocess for that request and is recycled.
    """

    DEFAULT_VOICE = "amy"

    @classmethod
    def capabilities(cls) -> Dict[str, Any]:
        return {
            "languages": [{"id": "en-US", "label": TTS_LANGUAGE_LABELS["en-US"]}],
            "voices": [
                voice_option("amy", "Amy", "en-US", "female"),
                voice_option("ryan", "Ryan", "en-US", "male"),
                voice_option("ryan-high", "Ryan (high quality)", "en-US", "male"),
                voice_option("lessac", "Lessac", "en-US", "female"),
            ],
            "default_voice": cls.DEFAULT_VOICE,
            "default_language": "en-US",
            "sample_text": {"en-US": TTS_SAMPLE_TEXT["en-US"]},
            "controls": {"speed": True, "stream": False, "exaggeration": False, "cfg_weight": False},
        }
    VOICE_FILES = {
        "default": "en_US-amy-medium.onnx",
        "amy": "en_US-amy-medium.onnx",
        "amy-medium": "en_US-amy-medium.onnx",
        "en_us-amy-medium": "en_US-amy-medium.onnx",
        "en_us_amy_medium": "en_US-amy-medium.onnx",
        "ryan": "en_US-ryan-medium.onnx",
        "ryan-medium": "en_US-ryan-medium.onnx",
        "en_us-ryan-medium": "en_US-ryan-medium.onnx",
        "en_us_ryan_medium": "en_US-ryan-medium.onnx",
        "ryan-high": "en_US-ryan-high.onnx",
        "en_us-ryan-high": "en_US-ryan-high.onnx",
        "en_us_ryan_high": "en_US-ryan-high.onnx",
        "lessac": "en_US-lessac-medium.onnx",
        "lessac-medium": "en_US-lessac-medium.onnx",
        "en_us-lessac-medium": "en_US-lessac-medium.onnx",
        "en_us_lessac_medium": "en_US-lessac-medium.onnx",
    }

    def __init__(self, model: Model, device: str = "auto") -> None:
        self.model_row = model
        # `device` (config's `--device` arg, if any) is accepted for interface
        # parity with the other TTS engines but deliberately ignored: piper's
        # standalone VITS binary is CPU-only by design — light enough it never
        # needed GPU, unlike chatterbox/orpheus/kokoro's resolve_device() path
        # (see .common.resolve_device). Config carries no `--device` arg for
        # this row (#371) since there is nothing here for it to steer.
        self.device_arg = device
        self.device = "cpu"
        self.binary = self._default_binary()
        self.default_model: Optional[Path] = None
        self.sample_rate = 22050
        # Resident piper.exe pool, keyed by (model_path_str, length_scale).
        self._procs: dict = {}
        self._procs_lock = threading.Lock()
        self._out_dir: Optional[Path] = None
        self._job = None  # Windows kill-on-close job for the resident children

    @staticmethod
    def _default_binary() -> Path:
        name = "piper.exe" if sys.platform == "win32" else "piper"
        return PROJECT_ROOT / "vendor" / "piper" / name

    @staticmethod
    def _config_path(model_path: Path) -> Path:
        return Path(str(model_path) + ".json")

    @staticmethod
    def _read_sample_rate(config_path: Path) -> int:
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            return int(data.get("audio", {}).get("sample_rate") or 22050)
        except Exception:  # noqa: BLE001
            return 22050

    def load(self) -> None:
        if not self.model_row.model_path:
            raise RuntimeError("piper row has no model_path (ONNX voice)")
        model_path = (PROJECT_ROOT / self.model_row.model_path).resolve()
        config_path = self._config_path(model_path)
        if not self.binary.exists():
            raise RuntimeError(f"Piper binary not found at {self.binary} - run scripts/install_tts.py")
        if not model_path.exists() or not config_path.exists():
            raise RuntimeError(
                f"Piper voice files not found at {model_path.parent} - run scripts/install_tts.py"
            )
        self.default_model = model_path
        self.sample_rate = self._read_sample_rate(config_path)
        self.device = "cpu"  # always — see the __init__ note on device_arg
        self._out_dir = Path(tempfile.mkdtemp(prefix="piper-resident-"))
        # Tie resident children to a kill-on-close job so a TerminateProcess on
        # this backend (the hub's stop path) can't leak piper.exe processes.
        self._job = create_kill_on_close_job()
        log.info(
            "Piper ready (resident mode, binary=%s, voice=%s, sr=%d)",
            self.binary,
            model_path.name,
            self.sample_rate,
        )
        # Pre-warm the default voice so the first real request is already warm
        # (the cold model load happens here, in the background load thread,
        # rather than on the first user-facing synthesis).
        try:
            self.synthesize(SpeechRequest(text="warming up", voice=self.DEFAULT_VOICE))
        except Exception as exc:  # noqa: BLE001
            log.warning("Piper pre-warm failed (non-fatal): %s", exc)

    def ready(self) -> bool:
        return self.default_model is not None

    def validate_voice(self, voice: str) -> None:
        self._model_for_voice(voice)

    def _model_for_voice(self, voice: str) -> Path:
        assert self.default_model is not None
        raw = (voice or "").strip()
        key = raw.lower().replace(" ", "-")
        if not key or key == "none":
            return self.default_model
        direct = Path(raw)
        if direct.is_file():
            return direct.resolve()
        filename = self.VOICE_FILES.get(key)
        if filename:
            candidate = self.default_model.parent / filename
            if candidate.exists():
                return candidate
        candidate = self.default_model.parent / f"{raw}.onnx"
        if candidate.exists():
            return candidate
        raise ValueError(f"unsupported Piper voice: {raw}")

    def _resident_cmd(self, model_path: Path, config_path: Path, length_scale: float) -> List[str]:
        assert self._out_dir is not None
        espeak_dir = self.binary.parent / "espeak-ng-data"
        cmd = [
            str(self.binary),
            "--model", str(model_path),
            "--config", str(config_path),
            "--length_scale", f"{length_scale:.4f}",
            "--sentence_silence", "0.05",
            "--quiet",
            "--json-input",
            "--output_dir", str(self._out_dir),
        ]
        if espeak_dir.is_dir():
            cmd.extend(["--espeak_data", str(espeak_dir)])
        return cmd

    def _get_proc(self, model_path: Path, config_path: Path, length_scale: float) -> _PiperProc:
        # length_scale (not voice+speed) keys speed; rounded so float jitter
        # doesn't fragment the pool. The common case (default voice, 1.0) maps
        # to a single warm process.
        key = (str(model_path), round(length_scale, 4))
        with self._procs_lock:
            proc = self._procs.get(key)
            if proc is not None and proc.alive():
                return proc
            if proc is not None:  # dead — recycle
                proc.close()
            cmd = self._resident_cmd(model_path, config_path, length_scale)
            proc = _PiperProc(cmd, self._out_dir, self._job)
            self._procs[key] = proc
            return proc

    def _wav_bytes_to_samples(self, data: bytes):
        import io

        import numpy as np

        if not data or len(data) <= 44:
            return np.zeros(0, dtype=np.float32)
        with wave.open(io.BytesIO(data), "rb") as wav:
            self.sample_rate = int(wav.getframerate())
            channels = int(wav.getnchannels())
            frames = wav.readframes(wav.getnframes())
        pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32)
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1)
        return pcm / 32768.0

    def synthesize(self, req: SpeechRequest):
        if self.default_model is None:
            raise RuntimeError("Piper not loaded")
        model_path = self._model_for_voice(req.voice)
        config_path = self._config_path(model_path)
        self.sample_rate = self._read_sample_rate(config_path)
        # Piper's length_scale is inverse speed: lower scale speaks faster.
        speed = max(0.5, min(2.0, float(req.speed or 1.0)))
        length_scale = 1.0 / speed
        try:
            proc = self._get_proc(model_path, config_path, length_scale)
            return self._wav_bytes_to_samples(proc.synth(req.text))
        except Exception as exc:  # noqa: BLE001
            # Resident path wedged or the child died — never fail the request:
            # drop that process and synthesise this one with a fresh one-shot.
            log.warning("resident Piper synth failed (%s); falling back to one-shot", exc)
            with self._procs_lock:
                stale = self._procs.pop((str(model_path), round(length_scale, 4)), None)
            if stale is not None:
                stale.close()
            return self._synthesize_oneshot(model_path, config_path, length_scale, req.text)

    def _synthesize_oneshot(self, model_path: Path, config_path: Path, length_scale: float, text: str):
        """Fallback: a single fresh piper.exe per call (the pre-resident path).
        Used only when the resident process fails, so synthesis is resilient."""
        import numpy as np

        espeak_dir = self.binary.parent / "espeak-ng-data"
        cmd = [
            str(self.binary),
            "--model", str(model_path),
            "--config", str(config_path),
            "--length_scale", f"{length_scale:.4f}",
            "--sentence_silence", "0.05",
            "--quiet",
        ]
        if espeak_dir.is_dir():
            cmd.extend(["--espeak_data", str(espeak_dir)])
        with tempfile.NamedTemporaryFile(prefix="piper-", suffix=".wav", delete=False) as tmp:
            out_path = Path(tmp.name)
        try:
            proc = subprocess.run(
                [*cmd, "--output_file", str(out_path)],
                input=(text.strip() + "\n").encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(PROJECT_ROOT),
                timeout=60,
                check=False,
                creationflags=NO_WINDOW,
            )
            if proc.returncode != 0:
                err = proc.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"Piper exited {proc.returncode}: {err}")
            if not out_path.exists():
                return np.zeros(0, dtype=np.float32)
            return self._wav_bytes_to_samples(out_path.read_bytes())
        finally:
            try:
                out_path.unlink(missing_ok=True)
            except OSError:
                pass

    def close(self) -> None:
        with self._procs_lock:
            for proc in self._procs.values():
                proc.close()
            self._procs.clear()
        terminate_and_close(self._job)
        self._job = None
        if self._out_dir is not None:
            import shutil

            shutil.rmtree(self._out_dir, ignore_errors=True)
            self._out_dir = None
