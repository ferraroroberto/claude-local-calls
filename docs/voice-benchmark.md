# Voice-model benchmark across the fleet — 2026-07-22 (STT + TTS placement)

Consolidated speech-to-text and text-to-speech numbers across the three-machine
fleet (tower / mac-mini-m4 / gaming), taken to decide **where each voice model
should live** so the tower's GPU can be reserved for the agentic lanes
(`agentic_heavy` / `agentic_light`). Issue
[#343](https://github.com/ferraroroberto/local-llm-hub/issues/343). Supersedes
the scattered numbers in `docs/parakeet-asr-evaluation.md` (STT accuracy detail
there is still the reference) and prior ad-hoc session measurements.

Static single-host placement is the target here; the dynamic-fallback superset
(a model that follows an ordered host chain and never goes down) is tracked
separately in [#342](https://github.com/ferraroroberto/local-llm-hub/issues/342).

## How it was measured

All numbers come from the reusable harness `scripts/bench_voice.py`, which hits
each hub's OpenAI-shaped endpoints end-to-end (`/v1/audio/transcriptions`,
`/v1/audio/speech`) — the same path real clients take, not a private backend
port. A hub is selected by `--base-url`, so the same tool measures any machine
and is reused after every future `/swap-model` or host/GPU change.

- **STT** — 7 real dictation clips (6 s–108 s, 319 s total) pulled from the
  voice-transcriber archive, spanning short/long and jargon-heavy content
  including the "Claude Code" wake phrase. Per clip: wall-clock (median of 2
  warm reps), **RTFx** (audio_seconds / processing_seconds — higher is faster
  than real time), WER vs the archived reference transcript, and domain-jargon
  survival.
- **TTS** — a 15-char and a 96-char sentence, median of 3 warm reps. Per
  sentence: synth wall-clock, synthesized audio duration, and **RT factor**
  (audio / wall — >1 is faster than real time).

**WER is comparative, not absolute.** I cannot listen to the audio; the
reference is the daily-driver whisper transcript archived next to each clip,
which shares whisper's phrasing conventions and so structurally favours whisper.
Normalisation lowercases, strips punctuation, and drops filler words; it does
**not** normalise number words, so a few WER points are formatting, not error.
The **RTFx / latency** figures are the objective ones and are what actually
decide the GPU-freeing question. Same caveat and methodology as
`docs/parakeet-asr-evaluation.md`.

Clip WAVs are personal dictation audio — kept under the gitignored `.scratch/`
and never committed. The set is reproducible from these archive session-ids
under `E:\automation\voice-transcriber\archive\2026\06\21\`: `15-49-47-2db373ed`
(6 s), `17-21-26-6f441e13` (12 s), `16-56-37-24ccb6f3` (22 s, wake phrase),
`16-51-12-5bec3e3f` (32 s), `17-13-22-f2d22946` (44 s), `16-47-13-dfe89b79`
(94 s), `15-50-17-abb7647b` (108 s).

## STT results

| Backend | Host (GPU) | RTFx overall | RTFx range | WER | Notes |
| --- | --- | --- | --- | --- | --- |
| whisper-large-v3-turbo (boosted) | **tower** — RTX, CUDA | **40×** | 21–56× | ~3.6% † | daily-driver dictation default |
| whisper-large-v3-turbo | **gaming** — GTX 1070, CUDA `sm_61` | **19.3×** | 6.5–24× | 1.9% | ~2× slower than tower; identical model → identical accuracy; **stable through the full run** |
| parakeet-tdt-0.6b-v3 | **mac** — Apple Neural Engine | **65.8×** | 11.7–113× | ~22% ‡ | **fastest STT by far** — sub-second even on the 108 s clip; higher/variable WER on this domain and still **drops the "Claude Code" wake phrase**; see `parakeet-asr-evaluation.md`. The backend was wedged on arrival (a real bug, **fixed in this work** — see finding 3); numbers are post-fix |

† Tower's mean WER as-measured was skewed by one 6 s profane clip that
whisper collapsed to "Okay." (a short-clip hallucination; gaming happened to
match the reference on the same clip). Excluding that degenerate clip, tower and
gaming both land ~2–4 % — they run the same `ggml-large-v3-turbo` weights, so
accuracy is equivalent and the only real axis is speed.

‡ Parakeet's WER is comparative only — the reference is whisper-family text, so
parakeet's differently-styled output is penalised for formatting it isn't
actually wrong about (some clips land 0–6%, others 20–26%). Speed, not WER, is
the reason to reach for it. The backend arrived **wedged** (ready-but-hangs) and
was root-caused and fixed as part of this work — see finding 3; the numbers are
from the fixed backend.

**Read:** gaming hosts the whisper family perfectly well for accuracy — same
model, ~2 % WER — at ~half the tower throughput. Even so, 19× real time means a
30 s dictation clip transcribes in ~1.5 s, imperceptible for human-in-the-loop
use. The GTX 1070 is the bottleneck only for latency-critical batch, which this
workload is not.

## TTS results

Median synth wall-clock for a short (15-char) and long (96-char) sentence:

| Backend | Host | short | long | RT factor | GPU? | Notes |
| --- | --- | --- | --- | --- | --- |
| **piper** | tower | **0.34 s** | **0.56 s** | 4–11× RT | **CPU** | fastest by far; **zero GPU cost** |
| orpheus | tower | 1.56 s | 4.38 s | 1.3–1.5× RT | GPU | expressive; faster than real time |
| orpheus | gaming | 3.88 s | 12.58 s | 0.46–0.54× RT | GPU | **slower than real time** on the GTX 1070 |
| kokoro | tower | 2.18 s | 6.40 s | 0.6–1.0× RT | GPU (ONNX) | slow — the Windows ONNX path, as the config note warns |

**Read:** the decisive fact is that **piper is CPU** — it costs the tower *no*
GPU and is 3–10× faster than every GPU option, so the premise "move TTS off the
tower to free the GPU" only applies to the GPU engines (orpheus / kokoro /
chatterbox), not piper. Orpheus on the GTX 1070 runs at ~0.5× real time
(a 7 s utterance takes ~12.6 s), acceptable for on-demand expressive speech but
not for anything interactive.

## Placement recommendation

Data-backed, superseding the pre-benchmark guess. Static `host:` targets:

| Model | Recommended host | Rationale |
| --- | --- | --- |
| `agentic_heavy` / `agentic_light` | **tower** | GPU-hungry; the box being reserved |
| **piper** (`audio_speech`, HA voice) | **tower** (stay) | CPU — zero GPU cost, fastest option; moving it to an edge buys nothing until the mac/Linux piper installer exists, and even then only for locality |
| **whisper-vanilla + whisper-translate** | **gaming** | low-frequency, non-latency-critical (Spanish notes, ES→EN); the ~2× slowdown is irrelevant here and it frees tower VRAM |
| **whisper-turbo** (accurate dictation) | **gaming** | **No tower VRAM headroom** — agentic-heavy (~13 GB) + agentic-light fill the 16 GB card, so a GPU whisper can't co-reside on the tower. It lives on gaming (19× RT — imperceptible for dictation); interim reliability rides gaming stability + the #342 fallback (a last-resort CPU-whisper on the tower, which costs no VRAM). Migrate in earnest once gaming is multi-day stable |
| **orpheus** (expressive TTS) | **gaming** (stay) | GPU engine, so moving it off tower is what actually frees GPU; 0.5× real time is fine for on-demand expressive speech |
| **parakeet** | **mac** — selectable, not default | ANE speed is excellent (65× RT, sub-second) and suits HA voice commands; the dropped wake phrase + higher domain WER keep it a fast opt-in, not the accurate default. The ready-but-hangs wedge that blocked it is now fixed |
| kokoro / chatterbox | tower, on-demand | low-priority comparison options; kokoro's Windows ONNX path is slow |

Net effect: **tower keeps only the agentic GPU load + CPU piper**; the GPU-TTS
(orpheus) and the low-frequency STT (vanilla/translate) move to gaming; parakeet
stays a mac-side opt-in. This frees the tower GPU for the agentic lanes without
regressing the fast HA-voice path (piper stays put and stays instant).

## Findings / caveats

1. **Gaming stability — cautiously positive, not yet proven.** An earlier
   gaming run was interrupted by a full reboot; that reboot was the **life-os
   agent** deliberately rebooting the box to apply an `nvidia-drm.modeset=0`
   GRUB fix for the PC freezes (Server mode), **not** an inference-load crash —
   so no "whisper crashed gaming" conclusion is drawn. After that fix the box
   ran a full whisper STT matrix **and** an orpheus TTS run without crashing
   (uptime climbed monotonically). Encouraging, but not multi-day proof. Note
   whisper-turbo **cannot** fall back to the tower's GPU in the interim —
   agentic fills the 16 GB card — so its reliability during the migration rests
   on gaming's uptime plus the #342 fallback chain (a CPU-whisper on the tower
   costs no VRAM), not on keeping a GPU copy warm on the tower.
2. **The pruned CUDA toolkit did not break inference.** The life-os agent
   removed `nvidia-cuda-toolkit` + nsight (~4 GB, build-time only) while
   `apt-mark manual`-protecting the runtime libs (`libcudart12` / `libcublas12`
   / `libcublaslt12`) the pre-built `llama-server` / `whisper-server` link
   against — whisper-server stayed up and served 200 throughout, so **nothing
   needed reinstalling**.
3. **Mac parakeet was wedged — root-caused and fixed here.** It reported "ready"
   (uvicorn on :8098, worker alive) but hung on every transcription, direct to
   :8098 and through the hub alike. Root cause: `parakeet_server._start_worker`'s
   startup `_pump` daemon (added in #297 to bound the CoreML load) keeps reading
   the worker's stdout for its whole lifetime, so the request path's direct
   `proc.stdout.readline()` raced it and blocked forever — parakeet transcription
   had been broken since #297. Fix: the request path now reads replies from the
   pump queue with a `TRANSCRIBE_DEADLINE_S` backstop (so a genuinely wedged
   worker 504s instead of hanging). Post-fix, parakeet transcribes sub-second
   (65× RT overall). Tracked separately from the benchmark.
4. **Edge Piper/whisper-Metal are un-runnable today** — `install_tts.py`'s piper
   installer is Windows-only and `install_{llama,whisper}_cpp.py` have no Linux
   path, so piper-on-mac/gaming and whisper-Metal-on-mac could not be measured.
   That is the Linux/mac install-path gap (its own issue), and it is why the
   piper placement above rests on piper being CPU-portable rather than on an
   edge measurement.

## Reproduce

```bash
# STT — same clip set, any host
python scripts/bench_voice.py stt --base-url http://127.0.0.1:8000  --model whisper  --clips-dir .scratch/voice-bench --json .scratch/stt-tower.json
python scripts/bench_voice.py stt --base-url http://192.168.0.16:8000 --model whisper  --clips-dir .scratch/voice-bench --json .scratch/stt-gaming.json
python scripts/bench_voice.py stt --base-url http://192.168.0.14:8000 --model parakeet --clips-dir .scratch/voice-bench

# TTS — any model/host
python scripts/bench_voice.py tts --base-url http://127.0.0.1:8000  --model piper   --voice amy
python scripts/bench_voice.py tts --base-url http://127.0.0.1:8000  --model orpheus --voice tara
python scripts/bench_voice.py tts --base-url http://192.168.0.16:8000 --model orpheus --voice tara
```
