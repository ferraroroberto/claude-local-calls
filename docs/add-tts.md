# Adding a TTS backend at `/v1/audio/speech`

Companion to [whisper-asr.md](whisper-asr.md). Where whisper gave
the hub **speech → text**, this adds the inverse — **text → speech** — at
the canonical OpenAI route `POST /v1/audio/speech`, proxied through the hub
so it lands in the observability ring exactly like the transcription proxy.

Driving consumer: `app-launcher`'s eyes-free "read the last reply aloud"
button (ferraroroberto/app-launcher#190). The hub side just has to expose a
consistent, observable, role-based voice that any LAN client can call.

## Why a Python shim instead of a vendored binary

whisper.cpp and llama.cpp both ship a native HTTP server `.exe`, so those
backends are "drop a binary on a port". The good local TTS engines don't:

- **Piper** (default) is a standalone local VITS TTS binary with ONNX voices.
  It is the fast path for short assistant replies and Home Assistant.
- **Chatterbox** (Resemble AI, on demand) is a `chatterbox-tts` PyPI package
  (torch). Small (~0.5 B), has an emotion/"tone" dial (`exaggeration` +
  `cfg_weight`) and optional zero-shot voice cloning.
- **Orpheus-3B** (on demand) is an LLM-based TTS that emits SNAC audio
  tokens. Its reference runtime is vLLM, which has **no usable Windows
  build** — so we run its **GGUF on the already-vendored `llama-server`**
  (loopback) and decode the audio tokens with the **SNAC** codec in-process.
- **Kokoro-82M** (on demand) is the small ONNX path and the multilingual
  option. Its pinned voice pack includes `ef_dora` (Spanish female) and
  `em_alex` (Spanish male), both synthesized with the `es` language code.

So instead of a binary we run a thin in-repo FastAPI shim
[src/tts_server.py](../src/tts_server.py) launched as
`python -m src.tts_server --model-id <id>` — the same pattern as
[whisper_translate_proxy.py](../src/whisper_translate_proxy.py). The engine
implementations live behind one common interface
([src/tts_engines/common.py](../src/tts_engines/common.py)) in the
[src/tts_engines/](../src/tts_engines/) package, one module per engine
(`chatterbox.py`, `kokoro.py`, `orpheus.py`, `piper.py`) plus a shared
`process.py` for the Windows job-object process-lifecycle helpers.

## The TTS registry rows ([config/models.yaml](../config/models.yaml))

```yaml
piper:                          # auto-loaded default (startup: eager on tower, #430)
  display_name: piper-tts
  aliases: ["audio_speech"]     # the role alias clients should address
  backend: tts
  engine: tts-server
  tts_engine: piper
  port: 8096
  model_path: "models/piper/en_US-amy-medium.onnx"
  args: ["--device", "cpu"]

orpheus:                        # on demand — expressive
  display_name: orpheus-tts
  backend: tts
  engine: tts-server
  tts_engine: orpheus
  port: 8093
  internal_port: 18093          # llama-server child (loopback) for the GGUF
  hf_repo: "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF"
  hf_pattern: "*q4_k_m*.gguf"
  model_path: "models/orpheus-3b-0.1-ft-q4_k_m.gguf"
  args: ["--device", "auto"]    # cuda if the GPU has room, else cpu

chatterbox:                     # on demand — NOT autostarted
  display_name: chatterbox-tts
  backend: tts
  engine: tts-server
  tts_engine: chatterbox
  port: 8092
  args: ["--device", "auto"]

kokoro:                         # on demand — low-footprint comparison model
  display_name: kokoro-tts
  backend: tts
  engine: tts-server
  tts_engine: kokoro
  port: 8095
  model_path: "models/kokoro/kokoro-v1.0.int8.onnx"
  args: ["--device", "auto"]
```

All TTS rows are enabled on `tower` only; the role lives at
`roles.audio.speech.model_id: piper`. Piper is the auto-loaded default because
it measures sub-second for short assistant phrases. Orpheus is the expressive
on-demand alternate, Chatterbox is kept for its tone dial / voice cloning, and
Kokoro is the tiny ONNX comparison model whose current Windows ONNX path is not
the fast option.

> **Orpheus GGUF caveat.** The exact upstream repo / filename for the
> community Q4_K_M build pairs with the llama.cpp route but drifts over
> time. If `python scripts/download_models.py --only orpheus` can't match a
> file, adjust `hf_repo` / `hf_pattern` / `model_path` to a current GGUF
> (e.g. `lex-au/Orpheus-3b-FT-Q4_K_M.gguf`, `QuantFactory/...`).

## What got wired (mirrors the whisper change shape)

| area | change |
| ---- | ------ |
| `config/models.yaml` | TTS rows (`piper`, `chatterbox`, `orpheus`, `kokoro`) on `tower`; `roles.audio.speech` points at `piper` |
| `src/model_registry.py` | `+tts_engine` field |
| `src/tts_engines/` | Piper + Chatterbox + Orpheus + Kokoro engines, one module each, behind one common interface (`common.py`) |
| `src/tts_server.py` | new — FastAPI shim (`/v1/audio/speech`, `/health`) |
| `src/backend_process.py` | `build_command` branch for `engine: tts-server`; widen filters to `tts` |
| `src/run_backend.py` | widen spawnable backends to `tts` |
| `src/server.py` | hub `POST /v1/audio/speech` proxy + a `tts` 400 on the chat routes |
| `src/install.py` | `_check_tts` / `_fix_tts`; widen model + port filters |
| `scripts/download_models.py` | widen to `tts` rows with `hf_repo` (Orpheus GGUF) |
| `scripts/install_tts.py` | pip install + warm Chatterbox/SNAC/Piper/Kokoro + fetch Piper assets, Orpheus GGUF, and Kokoro ONNX assets |
| `scripts/smoke_test.py` | synth probe when a TTS port is reachable |
| `app_web/...` | Models-tab 🔊 tile + a synthesis-based ping |
| `launchers/run_model.{bat,sh} <id>` | parameterized per-model launcher (#448); `run_all.*` includes all enabled TTS rows |
| `requirements-tts.txt` | `chatterbox-tts`, `snac`, `kokoro-onnx`, `soundfile` (torch transitively); Piper is a downloaded binary |

## Dependencies — kept off the base install

`chatterbox-tts` pulls torch (~2 GB on CUDA). Putting the TTS set in
`requirements.txt` would force torch onto the Mac mini (which enables no TTS
role). So TTS deps live in [requirements-tts.txt](../requirements-tts.txt),
installed only on TTS-enabled hosts:

```bat
.venv\Scripts\python -m pip install -r requirements-tts.txt
```

…or let the installer do it (it also downloads Piper's binary/voices and
Kokoro's ONNX assets, then pre-warms the weights so the first request isn't a
cold download). The Kokoro warm-up exercises `am_michael`, `ef_dora`, and
`em_alex`, so a clean install verifies both Spanish profiles before use:

```bat
.venv\Scripts\python -m src.install --fix      :: runs scripts/install_tts.py
```

`src.install` shows a **TTS deps installed** row whenever a `tts-server`
row is enabled.

### GPU vs CPU torch

`requirements-tts.txt` lists `torch`/`torchaudio` as lower bounds, and on
Windows PyPI only serves the **CPU** wheel — on which Orpheus synthesis runs
at roughly real-time (RTF ≈ 1), too slow to feel responsive. So
`scripts/install_tts.py` follows the requirements install with a CUDA
override: if `nvidia-smi` is present it reinstalls torch from the PyTorch
CUDA index, after which the torch-based engines load on the GPU (~5× faster;
Orpheus reaches RTF ≈ 0.85 on the 16 GB reference box). The default pins target this
repo's reference box (Python 3.14 / CUDA 13 / Blackwell — `torch
2.11.0+cu130`, the newest with a matching `torchaudio` on cu130). Different
Python or driver? Override the index/spec without touching the script:

```bat
set HUB_TTS_TORCH_INDEX=https://download.pytorch.org/whl/cu128
set HUB_TTS_TORCH_SPEC=torch==2.11.0+cu128 torchaudio==2.11.0+cu128
.venv\Scripts\python scripts\install_tts.py
```

The installer also installs `onnxruntime-gpu` on NVIDIA hosts so Kokoro can
request `CUDAExecutionProvider`. On the reference box the backend health does
report `device: "cuda"`, but warm latency stayed around 2.2 s directly to
`:8095` and 2.5 s through the hub for the short sample phrase; the bottleneck
is the current `kokoro-onnx` ONNX path, not phonemization or hub encoding.
Hosts without an NVIDIA GPU keep the CPU torch / ONNX Runtime path
automatically. `--device` in each row's `args` stays `auto` (CUDA when
available, else CPU).

## The request shape

```bash
# through the hub (observable) — or directly to :8096 to skip the ring
curl -s -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"audio_speech","input":"Hey, listen to this.","voice":"ryan","response_format":"wav"}' \
  --output reply.wav
```

Spanish uses the explicit Kokoro model and voice profile; it never routes
through the English `audio_speech` default:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro-tts","input":"Hola, esta es una prueba de voz en español.","voice":"ef_dora","response_format":"wav"}' \
  --output spanish-female.wav

# Male profile: same payload with "voice":"em_alex"
```

```python
from openai import OpenAI
client = OpenAI(api_key="local-dummy", base_url="http://127.0.0.1:8000/v1")
audio = client.audio.speech.create(model="audio_speech", voice="ryan", input="Hey, listen to this.")
audio.stream_to_file("reply.wav")
```

Body fields: `model` (registry id / display_name / the `audio_speech`
alias), `input` (required), `voice`, `response_format`, `speed`, plus
Chatterbox's `exaggeration` + `cfg_weight`. Defaults / notes:

- **`response_format`** — `wav` (default) and `pcm` are produced with the
  stdlib (no extra deps); `flac`/`ogg`/`opus`/`mp3`/`aac` go through
  `soundfile` and **fall back to wav** (with a logged note) when the encoder
  isn't available, so a request never fails on format alone.
- **`voice`** — `default`/empty uses the engine's built-in voice. For Piper,
  `default`/`amy` maps to `en_US-amy-medium`, `ryan` is the standard Piper
  male voice (`ryan-high` trades a little latency for quality), and `lessac`
  is the standard Piper female voice. For
  Chatterbox, any other name maps to a reference clip
  `config/tts_voices/<voice>.wav` (gitignored) → zero-shot cloning. For
  Orpheus, `voice` selects a preset (`tara`, `leah`, `jess`, `leo`, `dan`,
  `mia`, `zac`, `zoe`). For Kokoro, `voice` selects a
  Kokoro voice id (`am_michael`, `af_bella`, `am_fenrir`, `bm_george`, …);
  Spanish uses `ef_dora` (female) or `em_alex` (male), which select `lang=es`.
  `default`/empty still uses the engine default; an explicit unknown model or
  voice returns HTTP 400 instead of silently substituting an English model or
  voice.
- **`exaggeration` / `cfg_weight`** — Chatterbox's tone dial. Ignored by
  Orpheus (which expresses emotion through inline text instead).
- **`speed`** — Piper and Kokoro honor `0.5`–`2.0`; Chatterbox/Orpheus accept
  it as a documented no-op.
- **`stream_format`** — `"audio"` opts into **streaming** delivery (raw
  chunked bytes that play as they synthesize); absent / any other value keeps
  the current buffered single response. See [Streaming](#streaming) below.

## Streaming

Long inputs feel laggy when the whole clip has to synthesize before the
first byte returns (perceived latency is bounded by *total* synth time, not
time-to-first-audio). Setting `stream_format: "audio"` flips the endpoint to
incremental delivery — audio starts flowing as soon as the first frames
decode, so time-to-first-audio drops to a fraction of a second regardless of
length.

```bash
curl -N -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"audio_speech","input":"A long paragraph…","stream_format":"audio"}' \
  --output reply.wav
```

What streams:

- **Orpheus** (default) streams natively. Its llama-server child emits SNAC
  audio tokens incrementally; the engine switches that `/completion` call to
  `stream: true` and decodes a **sliding 28-token (4-frame) window**,
  emitting each window's artefact-free `[2048:4096]` segment (~85 ms at
  24 kHz) — the canopyai/Orpheus-FastAPI `speechpipe` pattern. Very short
  inputs that never fill the window fall back to one whole-clip decode.
- **Chatterbox** can't stream (its flow-matching vocoder needs the full
  token sequence), so it **cleanly falls back to a single final chunk** via
  the default `TTSEngine.synthesize_stream`. The request still succeeds; it
  just isn't incremental.
- **Kokoro** also falls back to one final chunk on the current ONNX path.
  Use buffered mode for latency comparisons; streaming is mainly an Orpheus
  perceived-latency feature.

Body format of the stream:

- `response_format: "wav"` (default) → a **streaming WAV** header
  (open-ended `0xFFFFFFFF` RIFF/`data` sizes) followed by PCM16 frames; plays
  incrementally in a browser `<audio>` element and ffmpeg with no MediaSource
  glue.
- `response_format: "pcm"` → headerless little-endian PCM16 frames; the
  sample rate is on the `X-Sample-Rate` response header.
- Any other format (`mp3`/`flac`/…) can't be encoded frame-by-frame with the
  stdlib/soundfile, so a streaming request for one **falls back to the
  buffered response** in that format (logged) — never an error.

The hub proxy (`POST /v1/audio/speech` on :8000) forwards the streamed body
through `httpx`'s streaming client while still recording the request in the
observability ring, exactly like the streamed chat path — so streamed synth
stays observable. Clients that POST directly to the backend port (:8096 /
:8093 / :8092 / :8095) use the same backend route, just without the ring
entry.

Non-streaming requests are unaffected — omit `stream_format` (or send any
other value) and the endpoint returns the same single buffered response as
before.

### Live validation (streaming)

`pytest` covers the incremental SNAC decode with **mocked tokens** (no torch)
and the shim's streaming response shape (mocked engine). The real perceived
latency can only be felt on the TTS box:

```bat
tray.bat --restart      :: piper auto-loads on :8096
:: long paragraph, streamed — first audio should arrive in ~1 s
curl -N -X POST http://127.0.0.1:8000/v1/audio/speech ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"audio_speech\",\"input\":\"<a few sentences>\",\"stream_format\":\"audio\"}" ^
  --output reply.wav
```

Measured on the 16 GB reference GPU (Orpheus), a ~19 s paragraph: **time-to-first-audio ≈ 1.1 s streamed vs ≈ 13.5 s buffered** (~12× faster to first sound; total synth time is unchanged). The floor is set by the 4-frame (28-token) decode warmup plus prompt processing, so very long prompts add a little. Confirm a request *without* `stream_format` still returns a complete buffered clip (back-compat).

> **Restarting a TTS backend to pick up code changes.** `tray.bat --restart` reclaims `:8000` (the hub) but deliberately leaves the model/backend ports alone, so it does **not** reload `tts_server`/`tts_engines` changes. Cycle the backend itself from the Models tab (stop → start) or `POST /admin/api/models/<id>/{stop,start}` (loopback-exempt), then restart the hub if you also changed `src/server.py`.

## Choosing the engine / model (downstream)

The hub exposes every enabled TTS model on the same `/v1/audio/speech`
route, addressed by `model`:

- `model="audio_speech"` (or `model="piper-tts"`) → Piper on :8096
  (the autostarted default). Default voice is `amy`. `piper.exe` runs
  resident (one process per voice+speed, ONNX voice loaded once), so short
  phrases skip the per-request model load: measured on the reference Windows
  box ~0.06 s direct to :8096 and ~0.06 s through the observable hub route for
  `Arming the perimeter.` (warm, connection reused; #163). The pre-resident
  path re-loaded the voice every call (~0.79 s / ~1.10 s).
- `model="orpheus-tts"` → Orpheus on :8093 (start it first from the Models tab
  or `launchers/run_model.bat orpheus`).
- `model="chatterbox-tts"` → Chatterbox on :8092 (start it first from the
  Models tab or `launchers/run_model.bat chatterbox`).
- `model="kokoro-tts"` → Kokoro on :8095 (start it first from the Models tab
  or `launchers/run_model.bat kokoro`). Default voice is `am_michael`.
  Spanish profiles are `ef_dora` (female) and `em_alex` (male).

The admin Playground discovers this same registry metadata and exposes only
valid model → language → voice combinations. Stopped models remain visible but
disabled, and controls appear only for engines that honor them: speed for
Piper/Kokoro, streaming for Orpheus, and tone controls for Chatterbox. Language
is a Playground filter; the submitted OpenAI-compatible payload remains
`model` + `voice`, so existing clients do not need a new field.

## VRAM note

On the 16 GB reference GPU, dropping the heavy Gemma model out of tray
autostart leaves room for Orpheus (~3–4 GB: 3B Q4 GGUF + SNAC) to be the
auto-loaded default alongside qwen + whisper. Chatterbox (~1–2 GB) is the
on-demand alternate, competing for VRAM only when you start it.

## Live validation (local only — CI has no torch/GPU)

`pytest` covers the shim with a **mocked engine** (no torch). The real
synthesis path can only be exercised on a TTS-enabled box:

```bat
.venv\Scripts\python -m src.install --fix
tray.bat --restart                         :: chatterbox auto-loads on :8092
.venv\Scripts\python scripts\smoke_test.py :: synth probe for reachable TTS rows
```

For Orpheus, also `python scripts\download_models.py --only orpheus`, start
it, and confirm `/v1/audio/speech` returns playable audio.

### Orpheus decode notes (validated)

The GGUF emits its audio stream as `<custom_token_N>` text. Two details make
or break it:

- **Prompt end marker is `<|eot_id|>`** (the model's special token), not
  `<|eot|>` — the latter generates nothing usable.
- The stream **leads with a few small control tokens** (e.g. 4, 5, 1) before
  the first real audio frame. The decoder skips any token that resolves to a
  non-positive SNAC id *without advancing the 7-token frame position*
  (matching the canopyai/Orpheus-FastAPI `speechpipe` decoder). Indexing
  every token instead shifts every frame out of range → silent output.

The token id math is `int(N) - 10 - ((pos % 7) * 4096)`; 7-token frames fan
out into SNAC's three layers as `[f0] / [f1,f4] / [f2,f3,f5,f6]`.

### Child-process lifetime

The Orpheus engine spawns a `llama-server` **grandchild** (loopback
`internal_port`). The hub stops a backend with `TerminateProcess`, which
runs no cleanup, so that grandchild is assigned to a Windows **Job Object**
with kill-on-close — when the `tts_server` parent dies (any reason) the OS
reaps the llama child, freeing its VRAM and port. As a belt-and-braces
guard, the engine also reclaims a stale `internal_port` listener before
spawning. (No-op on non-Windows, where the explicit `close()` handles it.)
