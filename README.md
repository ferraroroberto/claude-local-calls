# Local LLM Hub

A tiny local HTTP hub that routes `POST /v1/messages` (Anthropic shape) and
`POST /v1/chat/completions` (OpenAI shape) to several backends by `model` name,
plus a local whisper.cpp ASR pair and a text-to-speech pair, both reachable
through the hub's `/v1/audio/*` proxy (observable) or directly on their own
ports. Audio runs both directions: speech→text via `/v1/audio/transcriptions`
and text→speech via `/v1/audio/speech`. Image **generation** is available
OpenAI-shape at `POST /v1/images/generations` (Google Imagen via the
Antigravity CLI).

## Active rotation

Subscription-backed cloud routes (no GPU, no API keys, no Cloud project):

- **`claude-*`** — forwarded to the **`claude -p`** CLI on your machine,
  using your local Claude Code auth (your subscription) instead of an
  API key. Four rows: `claude-haiku-4-5` (alias `claude_haiku`),
  `claude-sonnet-4-6` (`claude_sonnet`), `claude-opus-4-8`
  (`claude_opus`), `claude-fable-5` (`claude_fable`). The short aliases
  are version-free, so when a new Claude release lands only the row's
  `display_name` needs updating and downstream callers keep working
  unchanged.
- **`gemini-*`** — forwarded to the **Antigravity CLI** (`agy`), using
  your Google sign-in (no API key required). Three rows: `Gemini 3.1
  Pro` (alias `gemini_pro`), `Gemini 3.6 Flash` (alias
  `gemini_flash`), `Gemini 3.5 Flash` (alias `gemini_lite`).
  `agy` replaces the standalone `gemini` CLI, which Google deprecates
  for AI Pro / Ultra subscribers on 2026-06-18. `agy` has no per-call
  model flag, so the hub switches its globally-selected model through
  the `/model` picker before each request — see
  [src/gemini_cli.py](src/gemini_cli.py). Quotas follow your Google
  AI Pro / Ultra plan.
- **`gemini_image`** — image **generation** via `agy`'s built-in Google
  **Imagen** tool, exposed OpenAI-shape at `POST /v1/images/generations`
  (returns `data[].b64_json`). `agy` ships no Nano Banana picker model;
  Imagen is its only image backend, hosted inside a Flash text session.
  The hub captures the generated artifact and returns it; the call lands
  in the observability ring like other traffic. Editing is also available
  (`POST /v1/images/edits`, multipart image+prompt) but is slow and
  procedural — see [docs/image-generation.md](docs/image-generation.md).
  Both are testable from the admin Playground's image card.

Self-hosted entries in active use as of the May 2026 frontier reading. A row
runs on whichever machine `config/models.yaml` gives it — not necessarily the
one you are reading this on; `127.0.0.1:<port>` below means "on the owning
host", and every one of them is reachable through this hub's own `:8000`
regardless (see [Multi-host: the Mac Mini](#multi-host-the-mac-mini)):

- **`qwen3.5-4b`** — local `llama-server` running
  [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)
  on `127.0.0.1:8088` (4 B hybrid Gated DeltaNet + sparse MoE, full
  GPU offload, Apache 2.0, 262 k native context). Fills the
  `agentic_light` role: OpenClaw fast lane, classification, edge.
  Also addressable as `model="agentic_light"` — clients that hit the
  role alias survive future `/swap-model` rotations unchanged.
  A **virtual no-think alias** `qwen3.5-4b-nothink` (role alias
  `agentic_light_nothink`) shares this same `:8088` backend — no second
  process, no extra VRAM — and makes the hub inject
  `chat_template_kwargs={enable_thinking:false}` into every request, so
  clients that can't send that field themselves (e.g. Home Assistant's
  `extended_openai_conversation`) still reach Qwen's fast, no-reasoning
  path. Plain `qwen3.5-4b` / `agentic_light` stay thinking-capable; a
  caller that sends its own `chat_template_kwargs` always wins.
- **`gemma4-26b-a4b-it`** — local `llama-server` running
  [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)
  on `127.0.0.1:8087` (25 B / 3.8 B-active MoE, IQ4_XS i-matrix quant
  — whole model on GPU in 16 GB VRAM). Fills the `agentic_heavy` role:
  deep agentic, transcript polishing, document work, EN↔ES↔CA. Also
  addressable as `model="agentic_heavy"` for the same reason. **On-demand
  since #422** (`startup: on_demand`, `idle_unload_minutes: 30`): the first
  request loads it (~tens of seconds), 30 idle minutes unload it — freeing
  ~13.4 GB of tower's VRAM for the voice path between uses.
- **`whisper-large-v3-turbo`** — `whisper-server`
  ([ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp))
  running [ggml-large-v3-turbo.bin](https://huggingface.co/ggerganov/whisper.cpp),
  **owned by the `gaming` satellite** since #323 — the tower runs no whisper
  backend of its own in the normal case. Reach it identically from any
  machine: POST to *this* hub's `:8000/v1/audio/transcriptions` and the
  request is served locally or proxied to the owning host, landing in the
  observability ring either way. It carries the repo's first production
  failover chain (`hosts: [gaming, mac-mini-m4, {id: tower, cpu: true}]`), so
  the tower is the degraded CPU last rung rather than the daily owner. Fills
  the `audio_transcribe` role — as its *fallback*; parakeet on the Mac Mini is
  the primary. Ownership, ports, and the chain live in
  [`config/models.yaml`](config/models.yaml); the topology is walked through
  under [Multi-host: the Mac Mini](#multi-host-the-mac-mini) below. On a host
  that actually runs the backend you can also POST straight to its `:8090` for
  lower overhead (skipping the ring); that port is a shared mutual-exclusion
  lock with `E:\automation\automation\audio\transcribe_voice`.
- **`whisper-medium-translate`** — sibling whisper-server running
  `ggml-medium.bin` on CPU, **also owned by `gaming`** (moved off the tower in
  #370). Same OpenAI-compatible `/v1/audio/transcriptions` shape; supports
  `task=translate` (turbo is transcription-only — its decoder distill
  drops translation). Eager-loaded (~1.5 GB RAM, 0 MB VRAM, always ready).
  Fills the `audio_translate` role; address it through this hub's
  `:8000/v1/audio/translations`. A lazy-load mode is also available — see
  [src/whisper_translate_proxy.py](src/whisper_translate_proxy.py) — for
  hosts that need to reclaim RAM when translate is rare.
- **`whisper-vanilla`** — the same `ggml-large-v3-turbo.bin` as the turbo
  row, configured for **unbiased language auto-detection**. The escape
  hatch for callers transcribing general multilingual audio (e.g. Spanish
  voice notes) that the English tech-dictation glossary would otherwise
  force-Englishize. Select it with `model="whisper-vanilla"` on the
  standard `:8000/v1/audio/transcriptions` request — no `language` needed.
  Two things make detection unbiased, **both required** (proven in #128):
  it carries **no** dictation glossary (`--carry-initial-prompt`/`--prompt`
  bias detection toward English), **and** the lazy proxy injects
  `language=auto` into every request that omits one — because
  whisper-server otherwise forces `language=en` per request regardless of
  its launch-level `--language` flag, so dropping the glossary alone is not
  enough. A caller that sends its own `language` always wins. Lazy-loaded
  (spawn-on-request via
  [src/whisper_translate_proxy.py](src/whisper_translate_proxy.py),
  idle-unload after 300 s) so it costs no VRAM when idle, and **owned by
  `gaming`** alongside the other two whisper slots (#370). See issue #128.
- **`piper-tts`** — fast local text-to-speech (the inverse of whisper),
  served by the in-repo FastAPI shim [src/tts_server.py](src/tts_server.py)
  on `127.0.0.1:8096`. OpenAI-compatible `POST /v1/audio/speech`. POST to
  the hub's proxy at `:8000/v1/audio/speech` with `model="audio_speech"`
  (captured in the observability ring). Uses the standalone Piper binary plus
  ONNX voices in `models/piper/`; default voice is `amy`
  (`en_US-amy-medium`; `ryan`, `ryan-high`, `lessac` remain selectable).
  `piper.exe` runs **resident** (one process per
  voice+speed, ONNX voice loaded once and reused) so short phrases skip the
  per-request model-load tax: integrated latency for `Arming the perimeter.`
  is ~0.06 s direct to `:8096` and ~0.06 s through the hub (warm, connection
  reused; #163). Fills the `audio_speech` role and is auto-loaded by the tray.
- **`orpheus-tts`** — expressive local text-to-speech, served
  by the in-repo FastAPI shim [src/tts_server.py](src/tts_server.py) on
  `127.0.0.1:8093`. OpenAI-compatible `POST /v1/audio/speech`. POST to the
  hub's proxy at `:8000/v1/audio/speech` (captured in the observability ring)
  or directly to `:8093` for lower overhead. Orpheus-3B is LLM-based, the
  most natural/expressive local voice and faster than real-time on GPU; its
  reference runtime (vLLM) has no usable Windows build, so the shim runs the
  GGUF on the vendored `llama-server` (loopback `:18093`) and decodes its
  audio tokens with the SNAC codec **on CPU** (`--device cpu`, #422 — measured
  faster than GPU SNAC on the 5060 Ti and ~1 GB VRAM cheaper; the llama child
  stays fully on GPU). Owned by **tower** with the gaming satellite as
  degraded fallback (`hosts: [tower, gaming]`, #422 — the 1070 renders at
  0.71x real-time, tower sustains 1.84x). Address explicitly as
  `model="orpheus-tts"` when expressiveness matters more than latency.
- **`kokoro-tts`** — low-footprint Kokoro-82M TTS on `127.0.0.1:8095`, served
  by the same [src/tts_server.py](src/tts_server.py) OpenAI-compatible
  `/v1/audio/speech` shim. It uses `kokoro-onnx` with the int8 ONNX model and
  packed voice styles in `models/kokoro/`. Start it from the Models tab or
  `launchers/run_model.bat kokoro`, then call the hub with `model="kokoro-tts"`.
  Default voice is `am_michael`, chosen as the closest built-in starting point
  for a Jarvis-like assistant voice. Spanish is available explicitly as
  `ef_dora` (female) or `em_alex` (male); those profiles select Spanish
  phonemization rather than the English default. ONNX Runtime CUDA is used when available,
  but the current Windows path measures roughly 2.2 s direct / 2.5 s through
  the hub for a short phrase, so it is kept as an option rather than the
  `audio_speech` role default until you intentionally repoint that role.
- **`chatterbox-tts`** — second TTS engine on `127.0.0.1:8092`, **on demand**
  (not autostarted). Resemble AI's Chatterbox (~0.5 B, torch) with an
  emotion/"tone" dial (`exaggeration` + `cfg_weight`) and optional zero-shot
  voice cloning. Start it from the Models tab or
  `launchers/run_model.bat chatterbox`. See
  [docs/add-tts.md](docs/add-tts.md) for the engine choice, request shape,
  and the Orpheus GGUF caveat.

**Role vs explicit model on `/v1/audio/*` (#412).** The two are different
contracts. Sending **no `model`** (or the role alias `audio_transcribe` /
`audio_translate`) addresses the *role*, and the hub is free to fall back
across the models in `roles.audio.<role>` when the primary's backend is down —
that is the point of the chain. Sending an **explicit model id** (`whisper`,
`parakeet`, `whisper-vanilla`) is **strict**: it is served by that model or it
fails. Matching is case-insensitive, so `Whisper` is the same strict request as
`whisper`. The rejections are three distinct conditions with three distinct
messages: `400` when the id names something that isn't a transcription backend,
`503 "not enabled on this host"` when it names a transcription model this hub
doesn't serve (a routing fact — nothing is down), and `503 "whisper-server not
running on :<port>"` when the model *is* served here and its backend is
genuinely dead. It is never quietly answered by a different model. (Host-level
failover for the *same* id — the #342 chain — is not a substitution and still
applies.) A `model=` the registry doesn't know, e.g. the OpenAI SDK's default
`whisper-1`, isn't a model request at all and still addresses the role.

Either way the substitution is **observable**. Every `/v1/audio/transcriptions`
and `/v1/audio/translations` response — including a strict rejection — carries
`X-Hub-Requested-Model`, `X-Hub-Served-Model`, and `X-Hub-Served-Host` (additive
— clients ignore them; the served pair is empty when the request was rejected
before any model ran, and the trio is not emitted on `/v1/audio/speech`, which
has no chain to substitute across). The request ring records `served_model` and
`served_host` alongside `model` (`GET /admin/api/hub/requests/recent`), and the
admin Hub/Telemetry rows render `requested → served @host`. `served_host` is the
dimension a cross-host drill actually needs: a `model=whisper` answered by a
machine with no whisper bound is a legitimate proxy hop to whisper's owner, and
until #412 no field said so. Use all of it, not a bare 200, when verifying a
failover drill — see [docs/fleet-maintenance.md](docs/fleet-maintenance.md).

> **Semantic change for role-addressed audio (#412).** For a request that
> addresses the *role* (no `model`, or `audio_transcribe` / `audio_translate`),
> the record's `model` field now holds the **role alias** rather than the
> concrete model that ran — the concrete one moved to `served_model`. The same
> value feeds the OTel `gen_ai.request.model` attribute (with the served id in
> `gen_ai.response.model` and the host in `hub.served_host`), so any dashboard
> grouping audio traffic by *request* model will re-bucket from `parakeet` to
> `audio_transcribe`. The hub's own per-model counters are unaffected: they key
> off `served_model` when one exists. Group by `served_model` /
> `gen_ai.response.model` to restore the old buckets.
>
> The counter key also canonicalises for an **explicit** model request: it is
> now the registry row id (`whisper`) rather than whatever alias the client
> typed (`whisper-large-v3-turbo`), because `served_model` records the row that
> actually ran. More consistent, but a dashboard keyed on a display name will
> re-bucket once.

**Transcription glossary.** Requests that go through the hub's audio
proxy (`:8000/v1/audio/*`) get a deterministic post-processing pass that
fixes persistent domain-term misspellings (e.g. "cloud code" → "Claude
Code"). The rules live in [config/transcription_glossary.json](config/transcription_glossary.json)
— an ordered list of literal `{"from","to"}` replacements
(case-insensitive, word-boundary, longest-phrase-first) plus a
`boost_terms` vocabulary. Edit it from the **📖 button on any whisper row
in the Models tab** (an inline editor; replacement edits apply without a
restart, boost-term edits on the next whisper start), or hand-edit the
JSON. **✨ Suggest from transcripts** mines the last *N* days of real
dictation from voice-transcriber's session API and proposes additions to
review. Direct hits to `:8090`/`:8091` bypass the glossary (and the
observability ring). See
[docs/whisper-asr.md](docs/whisper-asr.md) for the schema, the
in-app editor + miner, and the companion recognition-boosting mechanism.
NVIDIA Parakeet on Windows+CUDA (`parakeet.cpp`) was evaluated as a
*replacement* for this role and rejected — ~4× worse WER and no boosting
lever on this jargon-heavy workload. Parakeet running on the **Mac Mini's
Apple Neural Engine** (via FluidAudio/CoreML) is a different story: it's
enrolled as a selectable, non-default alternative — see
[Multi-host: the Mac Mini](#multi-host-the-mac-mini) below and
[docs/parakeet-asr-evaluation.md](docs/parakeet-asr-evaluation.md) for the
full trade-off writeup.

## Demoted candidates (kept defined, not in active rotation)

`glm-4.5-air` is **defined in `config/models.yaml`** but not in any
host's `enabled:` list anymore. Bring it up ad-hoc via
`launchers/run_model.bat glm`. Demoted on 2026-05-10 per
the May 2026 frontier reading — see
[docs/frontier-workflow.md](docs/frontier-workflow.md)
for the reasoning.

`qwen3.5-9b` was demoted the same day, but is **active again as of the
Mac Mini multi-host work** — it now runs on `mac-mini-m4` instead of
`tower` and is reachable through the Windows hub's own `base_url` like
any other model. See [Multi-host: the Mac Mini](#multi-host-the-mac-mini).

GLM **5.2** (the newer flagship) was evaluated for the local coding lane
and rejected — it is a single 744B-A40B MoE with no Air/Flash variant,
and even its smallest quant needs ~245 GB RAM+VRAM vs. this box's
~144 GB, so it does not load. Revisit if a GLM-5.2-Air/Flash ships; see
[docs/glm-5.2-evaluation.md](docs/glm-5.2-evaluation.md).

`gemma4-e4b-it` is the previous `agentic_light` role-holder, replaced
by `qwen3.5-4b` on 2026-05-10 via `/swap-model`. It is **kept in
`enabled:`** on the reference host for ad-hoc bring-up via
`launchers/run_model.bat gemma4_e4b`, but no longer autostarted.

## Roles & bi-weekly refresh

The four active local roles live in `config/models.yaml` → `roles:`:

| Role | Model | Why |
|---|---|---|
| `agentic_light` | `qwen35_4b` | OpenClaw fast lane / classify / edge |
| `agentic_heavy` | `gemma4_26b` | Deep agentic, transcripts, docs, ES↔EN↔CA |
| `audio_transcribe` | `whisper` | EN/ES audio → text |
| `audio_translate` | `whisper_translate` | ES audio → English (eager CPU sibling) |
| `audio_speech` | `piper` | text → speech (Piper fast default; Orpheus/Kokoro/Chatterbox on demand) |

Two Claude Code entry points drive the refresh:

- **`/frontier-refresh`** — the research skill
  (`.claude/skills/frontier-refresh/SKILL.md`, the single owner of the
  brief, cadence, and output contract). Regenerates
  `docs/frontier/runs/<today>/{report.md,frontier.json,frontier.html}`,
  repoints `LATEST`, and posts the per-role verdict as a comment on the
  always-open **frontier ledger issue
  [#272](https://github.com/ferraroroberto/local-llm-hub/issues/272)** —
  its last comment is always the current state. **Read-only on the
  registry** — produces artifacts only, never rewires anything. Runs
  unattended **bi-weekly** (the skill's `run-weekly.bat`, registered in
  app-launcher's Jobs tab weekly FRI 02:30, self-skipping alternate
  weeks) and on demand any time.
- **`/swap-model`** — interactive role swap. Reads the latest run +
  current roles, asks one question at a time (which role, which target,
  hf_repo if not registered, download now?), shows the planned diff,
  then edits `config/models.yaml` + writes a launcher pair + (optionally)
  shells out to `scripts/download_models.py`.

To browse a run interactively, open
`docs/frontier/runs/LATEST/frontier.html` in a browser — it's a
standalone interactive chart, no admin UI involved. To act on a run,
run `/swap-model` from Claude Code.

Side-by-side technical specs + docs links for all active models live in
[docs/model-comparison.md](docs/model-comparison.md). The latest research
brief and run sit under [docs/frontier/](docs/frontier/).

Point any client — the official `anthropic` or `openai` SDKs, openClaw,
a curl one-liner — at `http://127.0.0.1:8000` and swap backends by
changing the `model` string. Claude requests bill your subscription;
local model requests never leave the machine.

Inspired by the `_call_claude` pattern in
`E:\automation\inspiration-system\src\enrichment.py`.

## Latest-only policy

This repo intentionally ships **one model per role**. When a newer
release in the same family covers the same use case on the reference
hardware (e.g. Gemma 4 superseding Gemma 3), the older entry is
removed — registry, launchers, weights, and docs all go. The current
lineup and what each model is for live in
[docs/model-comparison.md](docs/model-comparison.md). Older entries
survive only in `git log` for historical context.

**Exception — one model per role, not per family.** The whisper
backend has two slots — `whisper` (turbo, 8090, transcription) and
`whisper_translate` (medium, 8091, translation). Turbo is
distill-decoded and was *not* trained on the translate task, so we
keep medium in a sibling slot for the rare cases when translation is
needed. Medium runs eager on CPU (~1.5 GB RAM) so the first translate
call is instant. The single-slot rule still applies *per role* —
there is one transcription model and one translation model.

## Scope & usage policy

This is a **personal playground** for running your own experiments
against your own Claude Code and Google AI Pro subscriptions and your
own local GPU on devices you personally own. It is **not** a hosted
service, a multi-tenant proxy, or a way to share subscription access.

To stay clearly within Anthropic's and Google's terms, please use it
only as intended:

- ✅ **Do** use it locally to call Claude or Gemini from your own
  scripts, agents, and tools on devices you personally own.
- ✅ **Do** use it on a trusted LAN to reach your own second machine
  or VM (e.g. a local agent runtime).
- ✅ **Do** route non-cloud traffic to the local qwen/gemma backends
  as much as you like — those are your own weights on your own silicon.
- ❌ **Don't** share the endpoint with other people — for Claude or
  Gemini, that would be sharing subscription access, which neither
  Anthropic's [Consumer Terms](https://www.anthropic.com/legal/consumer-terms)
  nor Google's [Additional Terms](https://policies.google.com/terms/generative-ai)
  allow.
- ❌ **Don't** port-forward it to the public internet or host it
  behind a domain.
- ❌ **Don't** build a product, commercial service, or large automated
  pipeline on top of the Claude or Gemini paths — for anything beyond
  personal experimentation use the paid Anthropic API or Vertex AI /
  Gemini API, which their respective usage policies and commercial
  terms are designed for.
- ❌ **Don't** hammer `claude -p` or the `agy` CLI in tight loops; keep
  volume at human-in-the-loop speeds so you don't abuse the service or
  get rate-limited. The Antigravity CLI quota follows your Google AI
  Pro / Ultra plan and is shared with the Antigravity IDE, so heavy hub
  use can also starve your IDE assistant. The local backends are
  rate-limited only by your GPU.

If your use case goes beyond "me, tinkering on my own machine,"
switch to the Anthropic API for Claude. When in doubt, check
[anthropic.com/legal](https://www.anthropic.com/legal/) or email
`support@anthropic.com`. This repo is provided as-is, with no
guarantee that it complies with Anthropic's terms for any particular
use.

## Architecture at a glance

```
openClaw / anthropic SDK / openai SDK / curl
                    │
                    ▼  http://<lan>:8000
   ┌────────────── FastAPI hub (src/server.py) ───────────────┐
   │  route by `model`:                                       │
   │    claude-*               → call_claude()   (claude -p subprocess)  │
   │    gemini-* / gemini_*    → call_gemini()   (agy CLI via ConPTY)    │
   │    POST /v1/images/generations (gemini_image) → call_gemini_image() │
   │      (agy Imagen tool; returns data[].b64_json, observable)         │
   │    POST /v1/images/edits (multipart image+prompt) → edit (slow)     │
   │    qwen3.5-4b             → llama-server 127.0.0.1:8088             │
   │    gemma4-26b-a4b-it      → llama-server 127.0.0.1:8087             │
   │    whisper-* (via chat shape) → 400 "use /v1/audio/* or direct"    │
   │    POST /v1/audio/transcriptions → whisper / parakeet (role chain) │
   │      (model=whisper-vanilla → glossary-free turbo, lazy)           │
   │    POST /v1/audio/translations   → whisper-medium (translate)      │
   │    POST /v1/audio/speech         → proxy to tts shim :8092/:8093/:8095/:8096 │
   │      (the audio proxy lands requests in the observability ring)    │
   │    GET  /v1/audio/health         → probe whisper/tts; 503 if down   │
   │      (preflight liveness — never sends a doomed transcription)      │
   └──────────────────────────────────────────────────────────┘

audio clients  ──►  hub 127.0.0.1:8000 /v1/audio/*  ──►  whisper-server / tts shim  (proxied, observable)
audio clients  ──►  tts shim       127.0.0.1:8096   (piper, text→speech, auto-loaded; fast CPU)
audio clients  ──►  tts shim       127.0.0.1:8093   (orpheus, text→speech, on demand; llama-server :18093 + SNAC)
audio clients  ──►  tts shim       127.0.0.1:8095   (kokoro, text→speech, on demand; ONNX Runtime)
audio clients  ──►  tts shim       127.0.0.1:8092   (chatterbox, text→speech, on demand; direct)
                          (whisper speaks /v1/audio/transcriptions, the tts shim /v1/audio/speech;
                           POST via the hub proxy for observability, or direct to the port — on the
                           host that owns the backend — to skip it)

STT is NOT provisioned on this box. The whisper trio moved to the gaming
satellite (#323/#370) and parakeet lives on the Mac Mini; the tower is only
whisper's degraded `cpu: true` last rung. Callers still POST to this hub's
:8000 and it proxies to the owner — but weights, builds, and `enabled:` rows
belong on the owning machine:
  whisper / whisper_translate / whisper_vanilla  → gaming      (192.168.0.16)
  parakeet (transcribe primary)                  → mac-mini-m4 (192.168.0.14)

Demoted (defined in config/models.yaml, not in any host's enabled list):
  glm-4.5-air — bring up via launchers/run_model.bat glm
Replaced as agentic_light on 2026-05-10 (still enabled on tower for fallback):
  gemma4-e4b-it — bring up via launchers/run_model.bat gemma4_e4b
Mac Mini (mac-mini-m4), proxied through this hub's own base_url — see below:
  qwen3.5-9b  → this hub 127.0.0.1:8000  → mac hub 192.168.0.14:8000 → llama-server :8081
  parakeet    → this hub 127.0.0.1:8000  → mac hub 192.168.0.14:8000 → parakeet-server :8098
```

Which machine owns which row is [`config/models.yaml`](config/models.yaml)'s
answer alone — the block above sketches request flow, not topology.
[Multi-host: the Mac Mini](#multi-host-the-mac-mini) below walks the current
tower / mac-mini-m4 / gaming split.

See [docs/architecture.mmd](docs/architecture.mmd) for the full component
diagram (the one structural map under a same-PR upkeep contract),
[docs/project-structure.md](docs/project-structure.md) for the per-backend
request-lifecycle sequences and the LLM key-facts briefing,
[docs/whisper-asr.md](docs/whisper-asr.md) for the whisper ASR backend
(glossary, boosting, tuning), and
[docs/add-tts.md](docs/add-tts.md) for the text-to-speech backend
(`/v1/audio/speech`).

## Multi-host: the Mac Mini

`local-llm-hub` runs as **one full install per machine**, but a model can
be *owned* by one host and made reachable through any other host's own
`base_url` — a client never needs to know or care which machine actually
runs a given model. Each `hosts:` entry in `config/models.yaml` gets an
`address:` (LAN IP), and each `models:` row gets an optional `host:` (which
host owns it — omitted means "whichever host resolves this config", i.e.
every existing single-host model is unaffected). A model listed in a
*non-owning* host's `enabled:` is transparently proxied: the request lands
on that hub exactly like any other, gets resolved, sees `host` doesn't
match the active machine, and is forwarded verbatim to the owning host's
own `:8000` (not its raw backend port) — so the proxied call still lands in
the owning hub's own observability ring. This is symmetric: the Mac Mini's
own hub can equally proxy to a Windows-owned model.

Today this powers the `mac-mini-m4` host (`192.168.0.14`, Apple M4):

- **`qwen3.5-9b`** — moved here from `tower` (see
  [Demoted candidates](#demoted-candidates-kept-defined-not-in-active-rotation)
  above); same `llama-server`, just running on the Mac.
- **`parakeet-tdt-0.6b-v3`** — NVIDIA Parakeet TDT 0.6B v3 on the Apple
  Neural Engine via [FluidAudio](https://github.com/FluidInference/FluidAudio)
  (CoreML), served by the vendored Swift worker in `mac/parakeet-worker/`
  + `src/parakeet_server.py`. A **selectable, non-default**
  `audio_transcribe` alternative (`model="parakeet"`) — faster than
  whisper-turbo but drops the "Claude Code" wake phrase and mangles
  "YOLO", so it's opt-in for latency-sensitive callers (e.g. Home
  Assistant voice commands) rather than the role default. Full
  measurement + trade-off writeup:
  [docs/parakeet-asr-evaluation.md](docs/parakeet-asr-evaluation.md).

The same pattern powers the **`gaming`** satellite (`192.168.0.16`, Ryzen 9
5900X, GTX 1070 8 GB, headless Ubuntu — #323), which owns the whisper STT
trio moved off the tower, so tower carries no whisper backends at all:

- **`whisper-large-v3-turbo`** — the transcribe role's *fallback* (parakeet on
  the Mac stays primary); CUDA-built whisper-server for `sm_61`, ~9.1 RTFx
  there vs ~33.6 on the tower's 5060 Ti — slower, but a failover path, not the
  daily-dictation primary.
- **`whisper-medium-translate`** and **`whisper-vanilla`** — moved to gaming
  in #370, alongside `whisper`'s earlier #323 move. `translate`
  stays CPU-only (0 MB VRAM); `vanilla` is GPU/lazy (~2000 MB when resident).
  voice-transcriber retains its own local `:8090` escape-hatch spawner for
  transport failures.
- **`orpheus-tts`** — moved here in #323, moved **back to tower** in #422: the
  1070 synthesizes at 0.71x real-time with the GPU pegged (measured
  2026-07-25), starving every streamed playback. Gaming stays second in
  orpheus's `hosts: [tower, gaming]` chain as the degraded-but-up fallback,
  so it keeps the weights staged and the id in `enabled:`.

Gaming's estimated VRAM footprint with the whisper trio resident: whisper
2000 + whisper_translate 0 + whisper_vanilla 2000 = 4000 MB (plus orpheus
2200 MB only while acting as failover tenant), against an 8192 MB ceiling
(`vram_mb` in `config/models.yaml`, #375) — comfortably under.

The Windows hub's admin UI Services card shows a live reachability pill for
every other hub-running peer (mac-mini-m4, gaming — any future satellite
with a non-empty `enabled:` list appears automatically, #372) alongside
Docker/Langfuse. Cross-host auth reuses `extra_allowlist` in
`config/webapp_config.json` (per-machine, not committed) — each host's LAN
IP is allowlisted on the other, the same bypass the bearer-token middleware
already grants loopback callers.

### Mac Mini lifecycle: autostart, remote bootstrap, sync (#181)

The Mac Mini's hub has no tray-equivalent process supervisor — a
`~/Library/LaunchAgents/com.ferraroroberto.local-llm-hub.plist`
LaunchAgent fills that role instead (`RunAtLoad` + `KeepAlive`, installed
by `python -m src.install --fix` on the Mac itself — see
`mac/launchagent/`). A **deliberate** stop (`POST /admin/api/hub/stop`)
`launchctl bootout`s the job so it stays down; a restart
(`POST /admin/api/hub/restart`) uses `launchctl kickstart -k`. (macOS
detail worth knowing if you touch this: launchd respawns a job under
`KeepAlive` after *any* signal-terminated exit — a plain self-SIGTERM and
even `launchctl stop` both get relaunched — `bootout` is the only thing
that actually unloads it.)

For the case where the Mac's hub is fully dead (crashed before
`RunAtLoad` fires, or manually killed), Windows can bring it back over a
**dedicated, forced-command-restricted** SSH key
(`~/.ssh/local-llm-hub-remote-ctl`, path in `.env`'s
`LOCAL_LLM_HUB_SSH_KEY`) — the Mac's `authorized_keys` restricts that key
to `mac/bin/hub-remote-ctl.sh`, which only allows two verbs
(`bootstrap` / `sync`), no general shell. The Services card renders one row
per hub-running peer (#372 generalized this from a hardcoded Mac-Mini-only
row); each gets a **Wake** button (visible when unreachable →
`POST /admin/api/hosts/<host-id>/bootstrap`) and a **Sync** button (visible
when reachable → `.../sync`, which `git pull --ff-only`s that peer's
checkout before restarting it — the Linux dispatcher, `linux/bin/hub-remote-ctl.sh`,
gives `gaming` the same two verbs since #368). An **out-of-sync** badge
appears on a peer's pill when the two hubs' `git_sha` (from
`/admin/api/version`) differ — `sync` is the fix.

Automatic peer wake/sync on the tower's own boot is no longer a separate
per-service toggle (the old `mac_mini_sync` flag was retired in #374). It is
owned entirely by the **fleet reconcile loop** (see the next section): on boot,
the loop wakes + syncs + starts every peer that carries fleet placement. A peer
with **no** placement is deliberately left asleep (nothing to run means no
reason to wake it) — the manual **Wake**/**Sync** buttons above stay available
for that case, or place a model on it to have the loop take over. `git_sha`
currency of an already-running peer is surfaced by the out-of-sync badge and
fixed with the manual **Sync** button; the reconcile loop's own sync happens
along the wake/bootstrap path when a peer is brought up cold.

The Models tab tags every remote-owned tile with a small `on <host-id>`
badge (e.g. `qwen3.5-9b` / `parakeet-tdt-0.6b-v3` both show `on
mac-mini-m4`) so a displayed PID is never mistaken for a local process.

**Placement cards (#423, lightened #434).** Each process-backed row in the
Models tab also renders its declared placement *intent* straight from
`config/models.yaml`: the host chain in priority order (effective owner
highlighted, cpu-resident tiers marked — the flag is the *effective* device
per host from `model_registry.cpu_resident_map`, so an always-CPU row like
`whisper-medium-translate`'s `-ng` is tagged on its owner too, matching the
Fleet summary) and a startup-policy badge that reads the live state honestly
(`eager`, `on-demand · loaded`, `on-demand · idle-unloaded`). No per-card
size chip and no host budget bar (#434): capacity is the Fleet summary
card's job, and the bar repeated the same machine fact on every card. The
data rides `GET /admin/api/models` as a per-row `placement` object (the
top-level `host_budgets` map was retired with the bar).

**Editable placement — the UI writes through to git (#424).** On the single
write host (`hub.config_write_host` in `models.yaml` — the tower) each
placement card gains an edit button that opens an inline editor: reorder /
add / remove chain hosts from the fleet host list, toggle a host's degraded
`cpu` tier, flip `eager` ↔ `on_demand`, set `idle_unload_minutes`. Saving
`PUT`s `/admin/api/models/<id>/placement`, which **validates first** —
schema (known hosts, no duplicates, every chain member cross-lists the model
in its `enabled:`) plus the #375 VRAM budget as a *hard* gate: an edit whose
eager, non-`cpu` chain members overcommit a host's `vram_mb` ceiling is
rejected with the arithmetic inline, before any file change. An accepted
edit mutates `config/models.yaml` comment-preservingly (ruamel.yaml
round-trip — the file's comments survive byte-for-byte), commits as
`config: <model> placement via admin UI` under the `local-llm-hub
config-bot` author name, and pushes to `origin main`. The write refuses a
dirty tracked file or a non-`main` checkout (409), and a failed push rolls
the commit *and* the file back (502) — never a silent local-only divergence.
Satellites converge automatically: a successful push fires the #181 sync
(git pull + hub restart) at every hub peer, and a periodic drift loop on the
write host re-syncs any satellite whose `models.yaml` sha lags (e.g. it was
powered off during the push; it also catches up on next boot). The Models
card header shows the config version (`cfg <sha>` — the models.yaml HEAD
sha, also on `/admin/api/version` as `config_sha`), so drift between hubs is
visible by comparing their `/admin` pages. Non-write hosts render the cards
read-only and 403 the write endpoint.

### Fleet placement: registry-derived desired state (#353, #430)

The per-machine service autostart above (`startup_profile.json`) answers
*"which services should this box bring up when it boots?"*. **Fleet
placement** answers the fleet-wide model question — *"what should run on
which machine, kept true even as boxes reboot or models die?"* — and since
#430 it is **derived from `config/models.yaml` itself**: a `startup: eager`
row's desired host is the first member of its `hosts:` chain (or its bare
`host:`) that enables it; `startup: on_demand` rows are never in the desired
set (the first request loads them, `src/on_demand.py` unloads them idle).
There is no separate placement file to keep in sync — the old
`config/fleet_placement.json` and the `models` list inside
`startup_profile.json` were retired because they duplicated (and drifted
from) exactly this registry truth. Moving a model is a `models.yaml` edit
(the Models-tab placement cards / swap-model), which the #424 config-write
pipeline commits and syncs to every satellite.

A background **reconcile loop** (`src/fleet_reconcile.py`, wired in
`server_lifecycle.py`) converges the fleet to the derived state on boot and
every ~5 min (`LOCAL_LLM_HUB_FLEET_RECONCILE_INTERVAL_S`). For each host with
desired models it: wakes an unreachable `can_ssh` satellite (same
forced-command `bootstrap` as above) and starts any desired model not already
running. A satellite that was powered off applies its own set on power-up
with no tower involvement — its hub derives the same desired set from its own
synced `models.yaml` (`model_registry.desired_model_ids`) at startup. The
loop leans entirely on existing idempotency — `backend_process.start` adopts
a live port and a forwarded `/start` returns a benign `409` — so the periodic
pass is safe to repeat forever and is strictly **additive**: it never stops a
model you started by hand.

The **Fleet summary** card in the Models tab (#354, reworked #431, polished
#434) renders a read-only per-machine summary over **every** configured
fleet host: the models **running** there right now (live state), the models
**loadable** but not running (that host's chain members, including
`on_demand` rows) — both lists one model per line, right-aligned at one
muted caption size, keeping the per-model device tag (#434) — and a
**homogeneous capacity line** (`GPU <used> / <total> · RAM <used> / <total>
GB`, the same shape on every machine). The capacity figures are **live-first**
(#436): GPU and RAM used/total come from the same probes the Machines tab
reads (local psutil/nvidia-smi, the cached SSH stats probe on peers), at the
same 1-decimal format, so the two tabs can never disagree while both are
live; the `~`-prefixed `est_vram_mb` sum vs the declared ceiling appears
only as a fallback where no live GPU figure exists (host offline, or a
unified-memory host with no discrete-GPU metric). It
carries **zero controls** beyond the collapse — placement is edited per model
in the Models card (#424) or `config/models.yaml`. Liveness is the same
hub-independent TCP probe the Machines tab uses (*is the box on?*), so a
**managed-only satellite that runs no hub** (`openclaw` — driven directly
over SSH, no models registered) reads *online* honestly with a note instead
of an empty list. A live backend the hub merely *adopted* on a mutex-shared
port (voice-transcriber's own whisper-server on `:8090`) is labelled
`· external` rather than claimed as hub-run.

**Capacity awareness (#375):** each host row may declare a `vram_mb` GPU-VRAM
ceiling (and, display-only, a `ram_mb` total) and each GPU model a rough
`est_vram_mb` footprint (all in `config/models.yaml`). The card sums a host's
desired + running models' `est_vram_mb` and shows an **advisory** *"Over VRAM
capacity"* warning on that host's row when the total exceeds its ceiling — a
heads-up that a placement change overcommits the box (e.g. `gaming`'s 8 GB
GTX 1070), replacing the by-hand `nvidia-smi` glance. Rows resident on
**CPU on that host** — piper's CPU-only engine, `-ng` whisper rows, a chain's
degraded `cpu: true` tier — hold no GPU VRAM and are **excluded from the sum**
(`model_registry.cpu_resident_map()`, #431); the same exclusion applies to
the model cards' effective-device pill tags and the on-demand loader's
headroom check, so every consumer inherits it. Hosts with no `vram_mb`
(Apple-silicon unified memory, managed-only boxes) never warn — on the
capacity line their GPU denominator is the system-RAM total (on unified
memory the GPU pool *is* system RAM; display context only, never a warning
input). Each per-host status object in the `GET` response below therefore
also carries `vram_mb` (ceiling or `null`), `est_vram_mb` (the summed,
CPU-excluded desired/running footprint), `capacity_warning` (bool), `ram_mb`
(declared total or `null`), `ram` (#434 — the live `{used_gb, total_gb,
percent}` snapshot: local psutil on the hub host, the Machines tab's cached
SSH stats probe on reachable peers, `null` where no live figure exists, in
which case the UI shows the declared total), and `gpu` (#436 — the live
`{used_mb, total_mb}` snapshot from the same plumbing: local nvidia-smi,
the cached SSH probe on peers, `null` where the host reports no GPU metric,
in which case the UI falls back to the `~` estimate vs the ceiling). The
`est_vram_mb` figures remain static engineering approximations feeding the
advisory warning and the no-live-figure fallback only. Each **model card's
meta line** ends with the same `· ~X GB` static footprint (#436) — shown
only for rows that actually load a model process; virtual aliases sharing
another row's process and CPU/ANE rows show nothing.

Or drive the same API directly:

```bash
# See the derived placement + live per-host status (eligible / reachable / running)
curl -s http://127.0.0.1:8000/admin/api/fleet-placement | jq

# Force a convergence pass on demand (the loop already does this periodically)
curl -s -X POST http://127.0.0.1:8000/admin/api/fleet-placement/reconcile
```

Since #374 the reconcile loop owns all peer wake/sync uniformly (mac-mini,
gaming, and any future satellite); since #430 the registry it converges on is
the single cross-host source of placement truth — a model desired on a
machine that's powered *off* is simply started when the box next reports in
(or by the box itself, from its own registry, when it boots).

### Dynamic model fallback: ordered host chains (#342)

Fleet placement above answers *"what should run where"*; **dynamic
fallback** answers *"and what if that host is down?"*. A `models:` row may
replace its single `host:` with an ordered preference chain:

```yaml
whisper:
  hosts: [gaming, mac-mini-m4, {id: tower, cpu: true}]
```

A bare `host:` stays valid and is exactly a one-element chain — every
existing row behaves as before, and with no multi-host chain configured
the engine adds zero probes and zero background work.

- **Ownership resolution.** The model's *effective owner* is the first
  chain candidate whose hub is reachable and that lists the model in its
  `enabled:`. Every dispatch path (chat, audio, admin start/stop/log)
  resolves against the effective owner and proxies there over the existing
  #178 cross-host path — chain order is the deterministic tie-break, so
  two live candidates always agree on who owns routing.
- **Failover.** A background loop in each hub probes the chain hosts
  (reusing the Machines-tab `peer_health` prober — no second prober,
  #396 tailnet fallback included) every `probe_interval_s` (30 s). When
  the owner stays *continuously* unreachable past `fail_after_s` (90 s),
  ownership moves to the next reachable candidate — and that candidate's
  own hub starts the model locally (each host acts only on itself; no
  master required, so failover works even when the control node is the
  host that died).
- **Degraded CPU tier.** A chain entry flagged `cpu: true` runs the model
  CPU-offloaded on that host — the hub rewrites its launch args
  (llama-server `-ngl 0`, whisper `-ng`, tts `--device cpu`) so the last
  resort is "up but slower," not "must match GPU perf."
- **Failback with hysteresis (anti-flap).** When a more-preferred host
  returns it must stay up *continuously* for `failback_after_s` (10 min)
  before ownership hands back; a repeatedly-rebooting host never
  accumulates the window, so ownership cannot bounce. The fallback host
  then stops only the instance the engine itself started (hand-started
  processes are never touched — same additive contract as the reconcile
  loop). `policy: sticky` disables automatic hand-back entirely.
- **Tunables.** Top-level `failover:` block in `config/models.yaml`
  (`probe_interval_s` / `fail_after_s` / `failback_after_s` / `policy`);
  defaults documented inline there.
- **Maintenance gate (#411).** The fleet reconcile loop's own always-on
  convergence races this engine's `fail_after_s` window — reconcile can
  SSH-resurrect a deliberately-stopped peer within seconds, before failover
  ever observes a continuous outage. `src/fleet_maintenance.py` gives the
  tower a host-scoped drain marker reconcile honours
  (`GET/POST/DELETE /admin/api/fleet-maintenance/{host_id}`), for running a
  drill without editing `.env`. See `docs/fleet-maintenance.md`.

**Enabling a chain:** every chain member must cross-list the model in its
`enabled:` **and** pre-stage the weights (`python -m src.install --fix` on
that host — the installer treats chain members as local candidates). The
Models tab tags a model served off-preference with a
`failover (prefers <host>)` note on its tile.

**The live chains (#405, #422).** `whisper` was the first production row to
use one:

```yaml
  hosts: [gaming, mac-mini-m4, {id: tower, cpu: true}]
```

It was chosen as the pilot because it is the transcribe role's *fallback*
(`parakeet` on the Mac is primary, via the #348 role chain), so a wrong
failover decision degrades a backup path rather than taking fleet STT down.
`tower` is flagged `cpu: true` because its 16 GB is already ~97% committed to
the agentic lanes — the degraded-but-up CPU tier is the honest last rung
there, not a compromise. `orpheus` joined in #422 with
`hosts: [tower, gaming]` — tower primary (1.84x real-time), gaming the
degraded fallback (0.71x — audible but slower than real-time). Every other
row still carries a bare `host:`.

### On-demand model lifecycle: `startup` + `idle_unload_minutes` (#422)

A `models:` row may declare a lifecycle policy:

```yaml
gemma4_26b:
  startup: on_demand        # default: eager (always-on, pre-#422 behavior)
  idle_unload_minutes: 30   # only meaningful for on_demand; omit to disable
```

An **on-demand** model is never started eagerly — not by hub autostart, not
by the fleet reconcile loop, not by the failover engine. The **first request**
that routes to it (chat, `/v1/messages`, or `/v1/audio/speech`) spawns the
backend via `src/on_demand.py` and waits for readiness — the hub-level
generalization of the `whisper-server-lazy` proxy pattern — so the first
call pays the load (tens of seconds for a big GGUF) and everything after is
warm. After `idle_unload_minutes` with no requests (in-flight requests hold
the window open), the idle watchdog stops the backend, and it **stays down**
until the next request — a hand-stopped or idle-unloaded on-demand model is
never resurrected by the supervisor loops. Before an on-demand load the hub
checks the #375 budget math (running `est_vram_mb` sum + the candidate vs
the host's `vram_mb` ceiling) and logs a loud warning on overcommit —
warning only, never a block: WDDM overcommit degrades transiently and idle
unload recovers it. `gemma4_26b` (the single heaviest local model, rarely
called) was the first on-demand row — which is what freed tower's VRAM for
orpheus's #422 move back. Since #430 the eager/on-demand split is also the
fleet's desired-state source (see *Fleet placement* above), and the
out-of-rotation rows (`gemma4_e4b`, `chatterbox`, `kokoro`, `glm`) are
explicitly `on_demand` so the derived desired set matches what actually runs.

### Linux satellite lifecycle: systemd (#323, #368)

A headless Linux satellite (`gaming`, later `openclaw`) has no
tray-equivalent and no LaunchAgent — a **systemd unit** fills that role,
the counterpart to `tray.bat` on Windows and the LaunchAgent on macOS. The
template lives at `linux/systemd/local-llm-hub.service`: it runs the
existing `run_hub.sh` under `Restart=always` and enables at boot with no
login required (`WantedBy=multi-user.target`).

Since #368 the Linux peer has **full lifecycle parity** with the Mac:

- **`install.py` fix** — `python -m src.install` on a Linux box checks the
  systemd unit (`is-active`/`is-enabled`) and the GPU (`nvidia-smi`); `--fix`
  renders the template's two placeholders, writes it to
  `/etc/systemd/system/` via `sudo -n tee`, then `daemon-reload` +
  `enable --now` (the systemd analogue of `_fix_launchagent`). Passwordless
  sudo is a prerequisite (see `docs/machines.md`); `sudo -n` fails fast with a
  clear message if it is missing rather than hanging on a prompt.
- **Remote bootstrap/sync** — `linux/bin/hub-remote-ctl.sh` is the systemd
  counterpart to `mac/bin/hub-remote-ctl.sh`, wired to the **same
  forced-command SSH key** (below). `sync` = `git pull --ff-only` +
  `./.venv/bin/python -m pip install -q -r requirements.txt` +
  `systemctl restart`; `bootstrap` = the same, tolerant of a
  dead/never-started unit (`restart`, falling back to `start`). So
  `POST /admin/api/hosts/gaming/{bootstrap,sync}` and the reconcile loop's
  wake→bootstrap chain (#364) work on any systemd satellite, not just the Mac.
- **Admin stop/restart** — `POST /admin/api/hub/{stop,restart}` drive
  `systemctl stop`/`restart` when the hub detects it is running under systemd
  (`INVOCATION_ID` set). This is necessary because `Restart=always` would
  respawn a bare self-SIGTERM — a *deliberate* stop must go through systemd,
  the same reason the Mac's stop goes through `launchctl bootout`.

**Forced-command `authorized_keys` line (Linux peer).** The dedicated
automation key's entry in the satellite's `~/.ssh/authorized_keys` pins the
command exactly as the Mac's does — the key can only run the dispatcher, never
a shell:

```
command="/home/<user>/local-llm-hub/linux/bin/hub-remote-ctl.sh",no-port-forwarding,no-x11-forwarding,no-agent-forwarding,no-pty ssh-ed25519 AAAA... local-llm-hub-remote-ctl
```

(On `gaming` the checkout is at `/home/gaming/local-llm-hub`, so the path is
`/home/gaming/local-llm-hub/linux/bin/hub-remote-ctl.sh`.) The script assumes
the repo lives at `~/local-llm-hub` — the same convention the Mac dispatcher
uses.

### Machines console (#309)

The **Machines** tab turns the hub into a fleet machine console — one place
to see the health of every box and act on it. It reads the host inventory
from `config/models.yaml` `hosts:` (now enrolling two managed-only machines
alongside `tower` and `mac-mini-m4`: **OpenClaw**, a Linux laptop, and
**gaming**, a Ryzen Linux inference satellite — #323) and renders a card per machine:

Every reachable machine shows the **same** snapshot — CPU / RAM / GPU / disk
(as uniform horizontal gauges) + uptime:

- **This machine** reads it locally from `src/system_stats.py`.
- **Peers** are probed two ways, both independent of whether the hub runs
  there — the card answers *is the box on?*, not *is the hub up?*
  (`src/remote_stats.py`): a **hub-independent TCP liveness probe** for
  up/down, and the same CPU/RAM/GPU/disk/uptime snapshot collected over the
  hub user's **own** passwordless SSH (a read-only one-liner, per-OS). A node
  flagged `dormant` is shown but not live-probed (none at present). All peer actions —
  read-only observability *and* reboot/shutdown — go over that general SSH
  (plus TCP for liveness); the forced-command key is reserved for the
  hub-lifecycle `bootstrap`/`sync` (#181/#368) on the model-serving peers
  (Mac Mini and `gaming`, each running its own dispatcher script).

**Tailscale fallback (#396).** Every peer-connect path — the model-proxy
upstream, SSH ops, and the remote stats/liveness probes — dials the peer's
wired LAN `address:` first and falls back to its `tailscale:` magic-DNS name
when the LAN path stops answering (`remote_stats.dial_address`, a short-TTL
last-known-good resolver). The wired NICs carry fixed DHCP reservations while
Wi-Fi is an *unreserved* fallback, so a wired failure moves a box to a pool
address — only the NIC-independent tailnet name survives that. The failover is
logged at info level (`falling back to tailnet …`) and the Machines card shows
a **via tailnet** badge while a peer is reached that way; hosts with no
`tailscale:` recorded behave exactly as before, and a healthy LAN path is
always preferred (no WireGuard hop while the wire works).

**Connection status (#397, #408).** Every card — including the active hub
host's own — now shows a fixed meta structure: Uptime, then **IP** (the
config `address:`, or a live `lan_ip()` lookup for the host's own row) and
**MAC** together, then a dedicated connection row. That row shows *live*
connection detail read over the same SSH round-trip as the CPU/RAM/GPU/disk
snapshot — no extra probe: **Wired** vs **Wi-Fi · SSID · signal (dBm)** for
whichever interface currently owns the box's outbound default route, with
Tailscale status right-aligned on the same line, plus a **Flaky link** flag
when the liveness probe that produced the card needed its #333 warm-up retry
to answer. Every signal degrades independently (no `iw` on Linux still
reports Wired/Wi-Fi, just without SSID/signal; a down peer shows none of it)
— see `docs/machines.md` for the per-platform availability matrix.

**Reboot / shutdown (destructive, peers only).** Any peer with an SSH channel
(`address` + `ssh_user`) offers **Reboot** and **Shut down** actions; the
active hub host is always excluded (powering it off would take the console
down with it). These run over the hub
user's **own general SSH** (issue #311) — the same passwordless channel the
stats snapshot uses — as `ssh <user>@<host> "sudo -n /sbin/shutdown -r|-h
now"`, detached with `nohup` so the SSH command returns cleanly before the box
drops off. The only prerequisite is the peer's passwordless-sudo sudoers
drop-in (already in place on the Mac Mini and OpenClaw); **nothing has to be
deployed to a managed machine** — no per-peer key, no forced-command script.
This is why OpenClaw's power buttons work the moment it is reachable.

**Wake (non-destructive, #356).** A machine that is **down or dormant** and
has a wired-NIC `mac:` recorded in its `config/models.yaml` host row offers a
**Wake** button — a fire-and-forget Wake-on-LAN magic packet
(`src/wake_on_lan.py`, UDP broadcast to `255.255.255.255:9`, stdlib socket
only) via `POST /admin/api/machines/{id}/wake`. No confirmation loop: the
card flips to *Rechecking…* and the next status poll reports whether the box
came up. The hub can't wake its own host (that's BIOS AC-restore territory),
and a host without a `mac:` simply has no Wake action. Per-machine WOL state
and caveats (Apple-silicon wakes from sleep only; openclaw has no wired NIC)
live in `docs/machines.md`. home-automation's UPS orchestration consumes this
endpoint over loopback for deliberate remote power-on.

**Remote Desktop.** A per-machine **Remote Desktop** action serves a
generated `.rdp` launcher (built from the machine's configured `rdp`
`{address, user}` target) that the viewing device downloads and opens — no
web RDP client, no dependency on any out-of-repo launcher file.

**On-demand diagnostics (#315).** The *this machine* card carries a **🔬
Diagnostics** row — a state chip showing the last health verdict (or live
capture progress) that opens a drill-in dialog. From there you can take a
**one-shot snapshot** or run a **timed capture** (15 min → 8 h, sampling every
5–60 s) that records system CPU/RAM/swap/disk/net/GPU **plus a full per-process
inventory and the listening-port map** into `data/diagnostics.db`.

The point is interpretation, not just recording: processes are attributed to the
app that owns them (`app-launcher: 3 procs / 800 MB`, not `python.exe ×14`) via
the committed `config/diagnostics_apps.json`; a finished run gets a persisted
`healthy`/`warning`/`critical` verdict from the tunable thresholds in
`config/diagnostics_rules.json`; and any run can be marked a **baseline** so
later runs report drift ("+2 resident apps, +3.1 GB idle RAM, new listener
:8099"). Each run also records **what it could and couldn't measure** (#322):
where the hub lacks privilege — macOS denies socket enumeration and ~40% of
per-process memory/CPU — the report says "not collected" and the verdict reads
`HEALTHY · ⚠ partial coverage` rather than letting an unmeasured signal pass as a
clean bill of health. Deep analysis happens outside the UI — export a run as
JSON, download an LLM-ready markdown health report, or query the SQLite file
directly.

**It adds no resident process**: the sampler is an asyncio task inside the
already-running hub, so nothing exists when no capture is active. An **opt-in
daily snapshot** (default off) keeps multi-week trends alive without adding one
either. Pure `psutil` + stdlib `sqlite3`, so the identical capture runs on the
Mac Mini and OpenClaw.

**Hub-less machines are covered too (#316).** A box that runs no hub (`openclaw`,
a dormant `tower`) is measured with a **zero-install** path: `scripts/portable_capture.py`
is a standalone `psutil`-only sampler delivered over SSH — `ssh host "python3 -
--duration-s 3600" < scripts/portable_capture.py > out.json` — whose raw output
is replayed into this store with `python -m src.diagnostics.ingest out.json`
(also `POST /admin/api/diagnostics/ingest`). The portable script interprets
nothing; attribution, coverage, and the verdict all run centrally at ingest
against the *source* machine's OS, so an ingested `openclaw` run is
indistinguishable from a local one and editing the attribution config
re-attributes it with no change on the peer. Machines that run their own hub use
the native path; both converge on the same store. Full reference:
[docs/diagnostics.md](docs/diagnostics.md).

**In-browser SSH terminal.** The **Terminal** action opens an xterm SSH
session by **reusing app-launcher's session-host** (its loopback ConPTY/WS
engine) rather than rebuilding a PTY stack here. This needs a small
companion change in app-launcher to register an `ssh` agent
([app-launcher#558](https://github.com/ferraroroberto/app-launcher/issues/558));
until that lands the terminal degrades gracefully with an actionable
"unavailable" state. Everything on this tab rides the existing bearer-token /
loopback-bypass middleware — no new auth scheme; the access model stays
loopback / Tailscale only.

## Layout

```
local-llm-hub/
├── .venv/                    # local virtualenv (gitignored)
├── .claude/
│   ├── commands/             # Claude Code slash commands (committed)
│   │   ├── swap-model.md         # interactive role swap (yaml + launcher + download)
│   │   └── system-specs.md       # collect Windows hardware specs
│   └── skills/
│       └── frontier-refresh/     # bi-weekly frontier research skill
│           ├── SKILL.md          #   brief + output contract + ledger (single owner)
│           └── run-weekly.bat    #   headless runner (app-launcher job, self-skips alternate weeks)
├── requirements.txt
├── requirements-dev.txt      # e2e + passkey deps (Playwright, pytest-playwright, webauthn)
├── requirements-tts.txt      # TTS deps (chatterbox-tts, snac, kokoro-onnx, soundfile — torch); Piper is a downloaded binary
├── tray.bat                  # Windows-only system-tray launcher (silent)
├── run_hub.bat / .sh         # start the FastAPI hub on :8000
├── launchers/                # per-model backends (.bat + .sh)
│   ├── run_model.* <id>          # parameterized launcher (#448) — title/port/banner
│   │                              #   pulled live from config/models.yaml; covers every
│   │                              #   id below (qwen, glm, qwen35_4b, gemma4_e4b,
│   │                              #   gemma4_26b, whisper, whisper_translate, piper,
│   │                              #   orpheus, kokoro, chatterbox)
│   └── run_all.*                # start everything enabled on this host
├── config/
│   ├── models.yaml                   # hosts + models + roles — placement truth (hosts chains + startup policy, #430)
│   ├── diagnostics_apps.json         # process -> fleet-app attribution rules (#315, committed)
│   ├── diagnostics_rules.json        # health-verdict thresholds (#315, committed)
│   ├── diagnostics_settings.json     # retention + scheduled snapshot (#315, gitignored)
│   ├── startup_profile.example.json  # template + fresh-clone default for service autostart (#265, services-only since #430)
│   ├── startup_profile.json          # live service-autostart profile, rewritten by the admin UI (gitignored, #304)
│   ├── fleet_maintenance.json        # live {host: {until, reason}} reconcile drain markers (gitignored, #411)
│   ├── webapp_config.json            # admin auth: bearer token, optional password, webauthn rp, CORS origins (gitignored)
│   ├── webapp_config.sample.json     # committed template + fresh-clone default for the above
│   ├── machine_specs_example.yaml    # template for scripts/detect_machine_specs.py's real output
│   ├── claude_pricing.json / gemini_pricing.json / openai_pricing.json  # per-model $/token
│   │                                  #   tables for the Code-tab usage/cost parsers
│   ├── dictionary_miner.json         # glossary-miner state (issue #94)
│   └── transcription_glossary.json / transcription_glossary.local.sample.json  # whisper
│                                      #   replacement-rules dictionary (issue #90)
├── webapp/                   # runtime data dir written by the /admin webapp
│   ├── cloudflared.sample.yml  # sample named-tunnel config (copy to cloudflared.yml)
│   ├── cloudflared.yml         # your own tunnel config — gitignored
│   └── auth.log                # /admin/api/login attempts (gitignored)
├── data/                     # runtime artefacts (gitignored)
│   ├── logs/                    # per-backend stdout/stderr: backend-<id>.log (+ one .log.1 backup)
│   └── diagnostics.db           # SQLite capture store (#315)
├── src/
│   ├── server.py             # FastAPI hub (both shapes) + /admin sub-app mount
│   ├── server_lifecycle.py   # startup/shutdown wiring + background resource sampler (#198)
│   ├── server_otel_receiver.py  # POST /v1/metrics OTLP receiver (#68)
│   ├── chat_translation.py   # request/response schemas, content-block extraction,
│   │                         #   prompt flattening, per-backend dispatch (issue #245)
│   ├── token_counting.py     # POST /v1/messages/count_tokens — per-backend input_tokens,
│   │                         #   exact on llama-server, flagged approximate on the CLIs (#463)
│   ├── anthropic_errors.py   # Anthropic {"type":"error",…} envelope for /v1/messages
│   │                         #   errors — path-scoped so /v1/chat/completions is
│   │                         #   untouched (issue #460)
│   ├── server_common.py      # model-resolution + OTel span helpers shared by
│   │                         #   server.py / server_audio_asr.py / server_audio_tts.py / server_images.py
│   ├── server_audio_asr.py   # /v1/audio/{transcriptions,translations,health} — whisper proxy + failover (#451)
│   ├── server_audio_tts.py   # /v1/audio/speech — TTS proxy, routes onto server_audio_asr's router (#451)
│   ├── server_audio_common.py  # header-safety + upstream-error helpers shared by the two above (#451)
│   ├── audio_proxy.py        # shared multipart bridging for the whisper translate-proxy paths
│   ├── server_images.py      # /v1/images/* handlers (generations, edits)
│   ├── claude_cli.py         # subprocess wrapper around `claude -p`
│   ├── gemini_cli.py         # Antigravity CLI (`agy`) wrapper via ConPTY (Google AI Pro)
│   ├── openai_upstream.py    # httpx client + SSE think-strip pipeline
│   ├── http_client.py        # shared pooled httpx Client/AsyncClient (skip a fresh SSL ctx per call)
│   ├── model_registry.py     # YAML loader (resolves display_name + aliases)
│   ├── model_failover.py     # dynamic fallback across an ordered host chain (#342)
│   ├── config_write.py       # git-backed models.yaml placement writes: validate → ruamel
│   │                         #   edit → config-bot commit → push + drift sync (#424)
│   ├── startup_profile.py    # config/startup_profile.json load/save — service flags (#265, #430)
│   ├── fleet_reconcile.py    # additive reconcile loop: wake + start registry-desired models (#353, #430)
│   ├── fleet_maintenance.py  # config/fleet_maintenance.json load/save — reconcile drain markers (#411)
│   ├── on_demand.py          # spawn-on-first-request / unload-when-idle model lifecycle (#422)
│   ├── host_profile.py       # pick active host row
│   ├── remote_proxy.py       # resolve a non-local model row's owning host + forward verbatim
│   ├── remote_bootstrap.py   # SSH-triggered remote hub bootstrap/sync (#181) + machine power (#309/#311)
│   ├── remote_stats.py       # remote machine liveness + stats for the Machines console (#309)
│   ├── machine_console.py    # Machines-tab data layer: per-machine CPU/RAM/GPU/disk/uptime (#309)
│   ├── ssh_terminal.py       # in-browser SSH terminal, reusing app-launcher's session-host PTY (#309)
│   ├── ssh_exec.py           # the one `ssh <user>@<host> <cmd>` builder (remote_stats + remote_bootstrap, #470)
│   ├── wake_on_lan.py        # Wake-on-LAN magic-packet builder/sender (#356)
│   ├── system_stats.py       # live RAM/CPU/GPU readings (consumed by Hub tab sparklines)
│   ├── observability.py      # OpenTelemetry bootstrap: tracing + metrics → local Langfuse (issue #4)
│   ├── async_fanout.py       # shared thread-safe asyncio pub/sub fan-out for SSE subscribers
│   ├── trace_id_middleware.py  # X-Trace-Id contract + request-id / anthropic-* response headers (#461)
│   ├── anthropic_headers.py  # anthropic-version / anthropic-beta parity decisions (#461)
│   ├── cors_policy.py        # CORS policy — loopback-by-default origins, SDK request headers,
│   │                         #   exposed response headers; installed outermost (#462)
│   ├── event_loop.py         # Windows proactor-loop shim so uvicorn survives client aborts (#222)
│   ├── build_info.py         # single source of truth for "what commit is this process running"
│   ├── services.py           # host-side service helpers: Docker engine + Langfuse stack (#27)
│   ├── code_usage.py         # host-side Claude Code usage parser + get_summary aggregation API
│   ├── usage_common.py       # shared vendor substrate: UsageRecord/FileStats + parsing helpers (#451)
│   ├── usage_pricing.py      # per-vendor $/Mtok pricing tables + record_costs (#451)
│   ├── usage_charts.py       # time-series bucketing + period-over-period comparison (#451)
│   ├── code_usage_history.py # durable daily rollups of Code-tab usage (outlives transcript pruning)
│   ├── claude_code_otel.py   # Claude Code OTel token/cost metrics receiver — data layer (#68)
│   ├── codex_usage.py        # host-side Codex (OpenAI) usage parser (~/.codex/sessions/*.jsonl)
│   ├── copilot_usage.py      # host-side GitHub Copilot usage parser (#231)
│   ├── copilot_billing.py    # GitHub Copilot billing-API daily credits poller (#231)
│   ├── agentsview_usage.py   # usage records from the optional external AgentsView service
│   ├── dictionary_miner.py   # mine recent dictation transcripts for glossary suggestions (#94)
│   ├── transcription_glossary.py  # post-process whisper transcripts through a committed glossary
│   ├── diagnostics/          # on-demand machine diagnostics (#315) — no resident process
│   │   ├── sampler.py            #   in-hub asyncio capture loop + opt-in scheduled snapshot
│   │   ├── store.py              #   SQLite store (data/diagnostics.db), migrations, retention
│   │   ├── attribution.py        #   process -> fleet-app mapping + listening-port scan
│   │   ├── rules.py              #   health-verdict engine over stored rows
│   │   ├── coverage.py           #   per-collector coverage — measured vs blind (#322)
│   │   ├── report.py             #   summary digest, baseline drift, markdown report
│   │   ├── ingest.py             #   ingest a portable foreign capture as a run (#316)
│   │   └── settings.py           #   retention + scheduled-snapshot settings
│   ├── install.py            # first-run checks + --fix
│   ├── run_backend.py        # hub|qwen35_4b|gemma4_26b|whisper|… dispatcher
│   ├── process_supervisor.py # shared subprocess start/stop lifecycle workflow (hub + model backends)
│   ├── server_process.py     # hub Popen + ownership / adopt-or-spawn (used by the tray)
│   ├── backend_process.py    # per-model Popen (llama-server + whisper-server);
│   │                         #   stdout/stderr → data/logs/backend-<id>.log (child-owned)
│   ├── parakeet_server.py    # OpenAI-shape ASR server wrapping the FluidAudio Parakeet CoreML worker
│   ├── whisper_translate_proxy.py  # FastAPI shim for optional lazy-load mode
│   ├── tts_server.py            # FastAPI shim for /v1/audio/speech (engine: tts-server)
│   ├── tts_engines/             # TTS engines: piper + chatterbox + orpheus + kokoro
│   │   ├── common.py                #   shared TTSEngine interface, SpeechRequest, audio helpers
│   │   ├── chatterbox.py, kokoro.py, orpheus.py, piper.py  #   one module per engine
│   │   └── __init__.py              #   build_engine() dispatch + re-exports
│   ├── webapp_config.py      # admin webapp config loader (bearer token, webauthn, allowlist, CORS origins)
│   ├── webauthn_gate.py      # passkey gate (optional — needs `webauthn` package)
│   ├── static_versioning.py  # ?v=<hash> stamping for /admin/static assets
│   ├── hub_log.py            # in-memory log ring buffer (admin Hub tab streams it)
│   ├── hub_observability.py  # live request ring, per-backend counters, SSE fan-out
│   └── _respawn_watchdog.py  # detached watchdog for the hub's admin-triggered restart (#198)
├── app_web/                  # FastAPI sub-app at /admin (HTML/JS SPA — no bundler)
│   ├── server.py             #   create_app() — middleware, routers, static mount
│   ├── middleware.py         #   bearer-token gate (loopback bypasses)
│   ├── admin_forward.py      #   forward a request into the /admin sub-app in-process
│   ├── routers/              #   misc / version / auth / webauthn / hub / models /
│   │                         #   startup_profile / fleet_placement / fleet_maintenance /
│   │                         #   roles / playground / services / telemetry / code_usage /
│   │                         #   glossary / hosts / machines / diagnostics
│   └── static/               #   index.html + main.js + state.js + tabs.js + api.js +
│                             #   hub.js + models.js + startup.js + fleet_placement.js +
│                             #   playground.js + code_usage.js/.css + diagnostics.js +
│                             #   glossary.js + machines.js/.css + machines_terminal.js +
│                             #   roles_card.js + telemetry.js/.css + styles.css +
│                             #   manifest.webmanifest + icon-*.png/favicon.ico (generated
│                             #   by scripts/gen_icons.py, committed)
│       └── _vendored/        #   project-scaffolding components: button / card /
│                             #   disclosure / empty-state / icons (Lucide sprite) /
│                             #   modal / nav / switch / xterm — SPA UI glyphs +
│                             #   primitives per design.md; chartjs (Chart.js
│                             #   4.4.7 UMD, byte-for-byte, #451) — trend charts
├── tray/                     # Windows system-tray launcher (silent pythonw)
│   ├── tray.py               #   single-file pystray + hub lifecycle owner
│   ├── icon.py               #   Lucide hub glyph (share-2), rendered via resvg,
│   │                         #   tinted live by health state — see app-launcher#65
│   ├── single_instance.py    #   .tray.pid lock validated with psutil
│   └── __main__.py           #   `python -m tray` entry, writes one-shot crash log
├── scripts/
│   ├── _lib.py                # shared download/extract/flatten helpers for the two
│   │                         #   vendor-binary install scripts (#195)
│   ├── smoke_test.py
│   ├── gen_icons.py          # thin caller onto project-scaffolding's shared brand_gen.py (hub master)
│   ├── bench_orpheus.py      # measure Orpheus llama-server throughput (tok/s, e2e)
│   ├── bench_voice.py        # STT+TTS bench on any hub (RTFx/WER/latency) — placement decisions (#343)
│   ├── download_models.py    # huggingface_hub → models/
│   ├── detect_machine_specs.py   # populate config/machine_specs.yaml
│   ├── install_llama_cpp.py      # CUDA-Windows / Metal-macOS release
│   ├── install_whisper_cpp.py    # whisper.cpp CUDA-Windows release; macOS builds Metal from source (#413) → vendor/whisper.cpp/
│   ├── install_tts.py           # pip -r requirements-tts.txt + Piper/Kokoro assets + warm TTS
│   ├── portable_capture.py      # standalone psutil sampler, SSH-delivered to hub-less machines (#316)
│   └── verify-before-ship.ps1    # byte-compile + pytest + Playwright on Chromium
├── assets/                   # generated by scripts/gen_icons.py, committed
│   └── stream-deck/local-llm-hub-144.png  # Elgato Stream Deck button
├── tests/                    # ~70 pytest unit/integration modules — one `test_*.py`
│   │                         #   per router/service (test_server, test_router,
│   │                         #   test_model_registry, test_install, test_streaming, …)
│   └── e2e/                  # Playwright smoke tests (Chromium): tab-level coverage
│                             #   for Code Usage, Fleet Placement, Machines, Roles, Telemetry
├── .github/workflows/
│   └── e2e.yml               # CI: unit tests + e2e gate on windows-latest
├── vendor/
│   ├── llama.cpp/            # prebuilt llama-server binary (gitignored)
│   └── whisper.cpp/          # prebuilt whisper-server binary (gitignored)
├── models/                   # downloaded GGUFs (gitignored):
│                             #   Qwen3.5-4B (Q4_K_M), gemma-4-26B-A4B-it (IQ4_XS),
│                             #   gemma-4-E4B-it (fallback, still enabled),
│                             #   ggml-large-v3-turbo.bin (whisper turbo, transcribe),
│                             #   ggml-medium.bin (whisper medium, translate),
│                             #   plus any demoted candidates if brought up ad-hoc
└── docs/
    ├── project-structure.md
    ├── architecture.mmd          # hand-authored internal-structure diagram (CLAUDE.md-bound, same-PR upkeep)
    ├── model-comparison.md       # per-model specs, quantisation, docs links
    ├── diagnostics.md            # on-demand machine diagnostics (#315)
    ├── machines.md               # fleet machine inventory + Tailscale identities (#309/#323)
    ├── whisper-asr.md            # whisper STT backend: glossary, boosting, tuning
    ├── whisper-turbo-vs-large-v3.md  # whisper model-size decision rationale
    ├── voice-benchmark.md        # cross-host STT+TTS placement benchmark (#343)
    ├── parakeet-asr-evaluation.md    # Parakeet CoreML ASR spikes (#123/#138)
    ├── add-tts.md                # how the TTS backend (/v1/audio/speech) slotted in
    ├── image-generation.md       # Imagen via agy → /v1/images/generations
    ├── gemini-agy-backend.md     # agy (Antigravity CLI) backend reference
    ├── glm-performance-assessment.md / glm-5.2-evaluation.md  # local-model benchmarks
    ├── orpheus-throughput.md     # Orpheus llama-server throughput bench
    ├── code-usage-agentsview.md  # Code-tab usage parsing + the optional AgentsView source
    ├── clients-telemetry-contract.md  # X-Trace-Id / OTel contract clients must follow
    ├── telemetry-langfuse.md     # Langfuse OTel stack setup
    ├── fleet-maintenance.md      # reconcile drain markers (#411)
    ├── ci-e2e-decision.md        # why CI runs pytest + Playwright on windows-latest
    ├── frontier-workflow.md      # bi-weekly refresh + /swap-model workflow
    ├── webapp-architecture-notes.md / webapp-design-language.md  # admin SPA structure + design notes
    ├── playbook-cli-backend-migration.md  # reusable method when a vendor CLI changes
    ├── system-specs/             # detect_machine_specs.py output template (gitignored real output)
    │   └── system-specs.example.md
    └── frontier/                 # bi-weekly efficient-frontier research (brief lives in the skill)
        ├── local-findings.md     #   durable cross-run findings that survive a run's own report
        └── runs/
            ├── LATEST            #   flat file containing the latest run date
            └── <YYYY-MM-DD>/     #   one dir per run
                ├── report.md     #   didactic markdown report
                ├── frontier.json #   machine-readable run data
                └── frontier.html #   standalone interactive chart
```

## Setup

One command does everything — deps, llama.cpp binary, GGUF downloads
for the models enabled for this host:

```bat
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m src.install --fix
```

The installer reads [config/models.yaml](config/models.yaml), figures
out which host row you are (by `LOCAL_LLM_HUB_HOST` env var, else
hostname match, else `default: true`), and only downloads what that
host's `enabled` list asks for. On the reference Windows PC that's
the active rotation: Qwen 3.5 4B (~2.6 GB), Gemma 4 26B-A4B IQ4_XS
(~13.4 GB), whisper-large-v3-turbo (~1.62 GB), whisper-medium for
translate (~1.5 GB), plus Gemma 4 E4B (~5 GB) kept as the
agentic_light fallback, plus the llama.cpp + whisper.cpp CUDA
binaries under `vendor/`. On the Mac mini it's Qwen only.

The demoted candidates (`qwen3.5-9b`, `glm-4.5-air`) are in the
registry but **not** in any host's `enabled:` list, so the installer
ignores them. To bring one up ad-hoc, add it to `enabled:` and re-run
`--fix`, or just download manually with
`python scripts/download_models.py --only qwen` and launch via
`launchers/run_model.bat qwen`.

On a host with a TTS role enabled, `--fix` also pip-installs
[requirements-tts.txt](requirements-tts.txt) (`chatterbox-tts`, `snac`,
`kokoro-onnx`, `soundfile` — the full set pulls torch, ~2 GB, which is why
it's kept out of the base `requirements.txt` so non-TTS hosts like the Mac
mini stay lean), installs CUDA torch / ONNX Runtime GPU on NVIDIA hosts,
downloads Piper's binary/voices and Kokoro's ONNX assets, and pre-warms the
Chatterbox / SNAC / Piper / Kokoro weights so the first
`/v1/audio/speech` request isn't a cold download. To do it by hand:

```bat
.venv\Scripts\python -m pip install -r requirements-tts.txt
```

Plain check (no changes):

```bat
.venv\Scripts\python -m src.install
```

Or open the **🩺 Health & install** panel on the admin webapp's
**🛰 Hub** tab (`http://127.0.0.1:8000/admin/`) — same checks, same
fixes, one button per row plus a **Fix-all**.

Requires the `claude` CLI on `PATH` (Claude Code) if any `claude-*`
model is enabled for your host.

Requires the **Antigravity CLI** (`agy`) on `PATH` if you want to use
any `gemini-*` model — install it from
[antigravity.google](https://antigravity.google) and sign in once with
your Google account. `agy` replaces the standalone `gemini` CLI, which
Google deprecates for AI Pro / Ultra subscribers on 2026-06-18. On
Windows the Gemini path also needs `pywinpty` (in `requirements.txt`):
`agy`'s print mode renders to a console, so the hub drives it under a
ConPTY. Without `agy`, requests targeting `gemini-*` return 502 with a
clear "CLI not found" message — the rest of the hub keeps working.

### Machine specs (optional)

[config/machine_specs_example.yaml](config/machine_specs_example.yaml)
documents the hardware schema this hub uses to reason about local
model fit (VRAM, system RAM, GPU compute capability). To populate the
real file for your host, run the detection script:

```bat
.venv\Scripts\python scripts\detect_machine_specs.py
```

```bash
./.venv/bin/python scripts/detect_machine_specs.py
```

Or copy the example manually and edit it:

```bat
copy config\machine_specs_example.yaml config\machine_specs.yaml
```

`config/machine_specs.yaml` is gitignored. AI coding agents working in
this repo will read it (when present) to recommend model sizes and
quantizations that actually fit your hardware. Optional — the hub
itself runs fine without it.

## Run

```bat
run_hub.bat                      :: FastAPI hub on :8000 (the admin
                                 :: webapp lives inside it at /admin)

:: Active rotation — one parameterized launcher, id from config/models.yaml (#448)
launchers\run_model.bat qwen35_4b          :: agentic_light  on :8088
launchers\run_model.bat gemma4_26b         :: agentic_heavy  on :8087
launchers\run_model.bat whisper            :: audio_transcribe on :8090
launchers\run_model.bat whisper_translate  :: audio_translate on :8091 (eager CPU)
launchers\run_model.bat piper              :: audio_speech (piper) on :8096
launchers\run_model.bat orpheus            :: orpheus TTS on :8093 (on demand)
launchers\run_model.bat kokoro             :: kokoro TTS on :8095 (on demand)
launchers\run_model.bat chatterbox         :: chatterbox TTS on :8092 (on demand)
launchers\run_all.bat                      :: start every backend in `enabled:` for this host

:: Fallback / ad-hoc (still in `enabled:` on tower, not autostarted)
launchers\run_model.bat gemma4_e4b         :: previous agentic_light on :8086

:: Demoted candidates — present but not in `enabled:` by default
launchers\run_model.bat qwen               :: llama-server for Qwen on :8081
launchers\run_model.bat glm                :: llama-server for GLM on :8082
```

(macOS / Linux: `./run_hub.sh`, `./launchers/run_all.sh`, etc. For
boot-time start + keep-alive on a headless Linux satellite, install the
systemd unit at `linux/systemd/local-llm-hub.service` instead — see "Linux
satellite lifecycle" above.)

Once the hub is running, open `http://127.0.0.1:8000/admin/` for the
admin webapp — six tabs (Hub / Models / Play / OTel / Code / Machines)
covering every operational concern. Going to `http://127.0.0.1:8000/`
redirects there.

### Tray launcher (Windows)

```bat
tray.bat
```

Starts a resident system-tray icon (silent — no terminal window) that:

- Auto-starts the hub on :8000. Hub startup then brings up, for every launch
  surface (tray, `run_hub.bat`, or `python -m src.run_backend hub`):
  - the **registry-derived model set** (#430) — every `startup: eager` row in
    `config/models.yaml` whose preferred chain host is this machine
    (`model_registry.desired_model_ids`; on the tower today that's
    `qwen35_4b`, `piper`, and `orpheus`); `startup: on_demand` rows load on
    first request instead;
  - the **services** configured in **`config/startup_profile.json`**
    (gitignored live file; see the committed
    **[template](config/startup_profile.example.json)**, issues #265/#304 —
    services-only since #430): Docker + Langfuse if `docker` / `langfuse` are
    `true` (via the same `launch_stack()` the Services card's manual launch
    button uses), and AgentsView if `agentsview` is `true`. Toggle any service
    off from the admin SPA's Models tab **Service startup** card, or hand-edit the
    JSON. Peer wake/sync is *not* a startup-profile toggle (the
    `mac_mini_sync` flag was retired in #374) — it is owned by the fleet
    reconcile loop, driven by the registry-derived desired state.
- Lets you toggle any other enabled local model on/off from the
  **🧠 Models** submenu (multiple may run concurrently).
- Surfaces the admin webapp via **🚀 Open admin** — same `:8000/admin/`
  URL, opened in your default browser. Live logs, per-backend counters,
  live request stream, sparklines, and a 🩺 health/install panel all
  live there now.
- Surfaces the **local URL**, the **LAN URL**, and (when configured)
  the **Cloudflare tunnel URL** as one-tap clipboard copies — the
  Cloudflare one comes with `?token=<bearer>` appended, so a phone
  loading a fresh tab can hand it back to the SPA without typing.

Drop a shortcut to `tray.bat` in the Windows Startup folder
(`shell:startup`) so the box behaves as an always-on local-LLM
endpoint after login. Routine tray activity is silent; if the tray
ever crashes, a single-shot `tray-crash.log` is written at the repo
root with the traceback (delete it any time — it's only recreated on
the next crash).

### Cloudflare tunnel (optional)

For phone-side access from outside the LAN, run cloudflared with a
named tunnel. The repo ships `webapp/cloudflared.sample.yml` — copy it
to `webapp/cloudflared.yml`, fill in your tunnel UUID + hostname, and
the tray will pick the hostname up automatically and surface
**📋 Copy Cloudflare URL** in its menu. The hub itself doesn't spawn
cloudflared (you own its lifecycle); the tray only *reads* the config.

The admin webapp's bearer token is generated on first tray boot and
persisted to `config/webapp_config.json`. Loopback callers bypass it;
anyone reaching the hub over the tunnel must present
`Authorization: Bearer <token>` or `?token=…` on the URL.

### Passkey (WebAuthn) gate — parked, server-side only (not planned: #247)

`src/webauthn_gate.py` and `app_web/routers/webauthn.py` implement a
tested, working passkey (WebAuthn) second factor for `/admin` —
registration/authentication ceremonies, a device whitelist, session
tokens — but it is **deliberately unwired**: no enrollment button or
ceremony call anywhere in `app_web/static/`, and the session token
`finish_authentication` mints isn't checked by any request path.
[#247](https://github.com/ferraroroberto/local-llm-hub/issues/247)
scoped building that frontend piece and was closed **not planned**:
`/admin` is only ever reached over loopback or Tailscale, and both
already bypass the bearer-token gate as fully trusted (loopback
outright, Tailscale via `extra_allowlist`) — there's no Cloudflare
tunnel exposure of `/admin` in practice, so a passkey second factor
has no remaining trust boundary left to protect. The code stays in
the tree untouched as a reference implementation in case that trust
model changes later; it isn't a live security feature today.

To poke at it directly anyway: set `webauthn_rp_id` / `webauthn_origin`
in `config/webapp_config.json` (needs the `webauthn` package from
`requirements.txt`; its absence just makes
`GET /admin/api/webauthn/status` report `available: false`), then
drive `POST /admin/api/webauthn/enroll/window` (loopback-only, opens
a 5-minute window) followed by the `/enroll/begin` →
`navigator.credentials.create()` → `/enroll/finish` ceremony from a
WebAuthn-capable browser tab — there's no built-in page for this, so
script it or drive it from devtools. Even fully enrolled, it won't
gate anything.

### Server adoption between launchers

The hub on :8000 (and each per-model port :808x) is single-owner — TCP
allows only one process to bind a port. To make `tray.bat`,
`run_hub.bat`, the parameterized `launchers/run_model.bat <id>`, and the
admin SPA's Hub/Models tabs coexist, every launcher follows the same
**adopt-or-spawn** rule:

- If the port is already reachable, the launcher *adopts* the running
  process (no second spawn, no error) and treats it as up.
- Each launcher only stops what it spawned itself. Closing the tray
  doesn't stop a hub that `run_hub.bat` started, and vice versa.
- The admin webapp's **🧠 Models** tab distinguishes managed vs.
  adopted processes and surfaces the foreign PID so you can decide
  whether to take over.

The admin webapp's **🛰 Hub** tab streams the hub log over SSE — even
for an adopted hub, since the log lines come from the *current*
process's in-memory ring rather than a captured stdout. The
Streamlit-era caveat about adopted processes having no log tail no
longer applies.

Equivalent Python entrypoints (run from the project root):

```bat
.venv\Scripts\python -m src.run_backend hub
.venv\Scripts\python -m src.run_backend qwen35_4b
.venv\Scripts\python -m src.run_backend gemma4_26b
.venv\Scripts\python -m src.run_backend whisper
.venv\Scripts\python -m src.run_backend whisper_translate
.venv\Scripts\python -m src.run_backend piper
.venv\Scripts\python -m src.run_backend chatterbox
.venv\Scripts\python -m src.run_backend orpheus

:: Fallback (still enabled, not autostarted)
.venv\Scripts\python -m src.run_backend gemma4_e4b

:: Demoted (ad-hoc only; not in tray autostart, not auto-installed)
.venv\Scripts\python -m src.run_backend qwen
.venv\Scripts\python -m src.run_backend glm
```

The hub binds on `0.0.0.0:8000`, so other machines on your LAN can
also reach it. The llama-server backends bind on loopback — they're
only reachable through the hub.

## LAN access

The hub binds on `0.0.0.0:8000`, so any machine on the same network
(another laptop, a VM, an agent like openclaw running next to you) can
use it.

1. **Start the hub** (either `run_hub.bat` / `.sh` at the repo root,
   or `tray.bat` on Windows — the tray autostarts the hub). Start any
   local backends you need from `launchers/run_model.bat <id>` or the
   admin webapp's **🧠 Models** tab.
2. **Find your LAN IP.** The admin webapp's **🛰 Hub** tab shows it
   as a clickable **LAN** link. From a terminal:

   ```bat
   ipconfig | findstr IPv4
   ```

3. **First run on Windows:** the firewall will prompt to allow Python
   through. Accept on **Private** networks only — never Public.
4. **Point the remote client at the LAN URL**, with the bearer token
   from `config/webapp_config.json` (loopback callers skip this; every
   non-loopback caller — including a LAN client — must present it):

   ```python
   from anthropic import Anthropic
   client = Anthropic(
       api_key="local-dummy",
       base_url="http://192.168.1.42:8000",   # your LAN IP here
       default_headers={"Authorization": "Bearer <token from config/webapp_config.json>"},
   )
   ```

**Security caveats.** The hub is **not** unauthenticated: loopback
callers bypass auth, but `ParentBearerTokenMiddleware`
(`app_web/middleware.py`) gates every non-loopback caller — including
`/v1/messages` and `/v1/chat/completions`, not just `/admin` — behind
the bearer token generated on first tray boot and persisted to
`config/webapp_config.json` (see "Cloudflare tunnel" above), unless the
caller's IP matches `extra_allowlist` in that same config (e.g. a
Tailscale peer). Anyone on the LAN who doesn't have the token still
can't reach it. That said, still only run this on trusted networks (home
LAN, office LAN you own): the token lives in a plaintext local file, and
a compromised LAN peer that reads it can spend your Claude quota and
burn your GPU. Do **not** port-forward the hub to the public internet,
and do not accept the firewall prompt on Public networks (cafés,
airports, hotel Wi-Fi).

## Use it from Python

Anthropic SDK, any backend:

```python
from anthropic import Anthropic

client = Anthropic(api_key="local-dummy", base_url="http://127.0.0.1:8000")

# Claude via subscription — use the version-free alias, not the dated display_name
msg = client.messages.create(
    model="claude_haiku",
    max_tokens=128,
    messages=[{"role": "user", "content": "Hello"}],
)

# agentic_light role — Qwen 3.5 4B (hybrid Gated DeltaNet + sparse MoE, full GPU)
msg = client.messages.create(
    model="qwen3.5-4b",
    max_tokens=128,
    messages=[{"role": "user", "content": "Hello"}],
)

# agentic_heavy role — Gemma 4 26B MoE (25 B / 3.8 B-active on GPU)
msg = client.messages.create(
    model="gemma4-26b-a4b-it",
    max_tokens=128,
    messages=[{"role": "user", "content": "Hello"}],
)
print(msg.content[0].text)

# Or address the role directly — survives future /swap-model rotations
# unchanged. `agentic_light` and `agentic_heavy` both work the same way.
msg = client.messages.create(
    model="agentic_light",
    max_tokens=128,
    messages=[{"role": "user", "content": "Hello"}],
)

# Gemini 3.1 Pro via your Google AI Pro subscription (Antigravity CLI)
msg = client.messages.create(
    model="gemini_pro",   # alias for "Gemini 3.1 Pro"
    max_tokens=128,
    messages=[{"role": "user", "content": "Hello"}],
)

# Image content blocks work on both subscription paths (claude-* and gemini-*).
# The admin SPA's Play tab grows a file uploader automatically when the
# selected model resolves to a claude/gemini backend.
import base64
with open("photo.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

msg = client.messages.create(
    model="gemini_pro",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": b64,
            }},
        ],
    }],
)
print(msg.content[0].text)
```

> Demoted candidates (`qwen3.5-9b`, `glm-4.5-air`) work the same way
> if you've brought them up ad-hoc — pass their model name as
> `model=`. The hub will return 400 if their backend isn't reachable.

OpenAI SDK (get native tool calls via `llama-server --jinja`):

```python
from openai import OpenAI
client = OpenAI(api_key="local-dummy", base_url="http://127.0.0.1:8000/v1")
msg = client.chat.completions.create(
    model="gemma4-26b-a4b-it",
    messages=[{"role": "user", "content": "Hello"}],
)
print(msg.choices[0].message.content)
```

Raw HTTP:

```bash
curl -s http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4-26b-a4b-it","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}'
```

List enabled models:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

**Error responses.** `/v1/messages` answers errors in the Anthropic
envelope, so the SDK's typed exceptions (`BadRequestError`,
`RateLimitError`, …) and its retry logic behave here as they do against
the real API (issue #460):

```json
{"type": "error", "error": {"type": "invalid_request_error", "message": "messages must not be empty"}}
```

`error.type` follows the status: 400/422 `invalid_request_error`, 401
`authentication_error`, 403 `permission_error`, 404 `not_found_error`,
429 `rate_limit_error`, 503 `overloaded_error`, other 5xx `api_error`.
`/v1/chat/completions` and the `/admin` API deliberately keep FastAPI's
`{"detail": …}` shape — OpenAI-shape callers parse a different envelope.

### Counting input tokens (issue #463)

`POST /v1/messages/count_tokens` takes the same body as `/v1/messages`
(`model` + `messages` + optional `system`; `max_tokens` is ignored) and
answers with Anthropic's `input_tokens`, so a caller can size a prompt
against a model's context — or its budget — before paying to find out:

```bash
curl -s http://127.0.0.1:8000/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-4b","system":"You are terse.","messages":[{"role":"user","content":"Count to three."}]}'
```

```json
{"input_tokens": 23, "model": "qwen3.5-4b", "backend": "openai",
 "method": "llama_server_tokenizer", "exact": true}
```

**Read `exact` before you trust the number.** The honest count differs per
backend family, so the hub resolves it per backend instead of applying one
global guess — and when it cannot measure, it says so rather than dressing
an estimate up as a measurement:

| Backend | `method` | `exact` | How |
| --- | --- | --- | --- |
| `openai` (llama-server: qwen, gemma, glm) | `llama_server_tokenizer` | ✅ yes | `POST /apply-template` renders the messages through the model's own chat template, `POST /tokenize` counts the result (`add_special: true`, matching what llama-server does for a real completion) |
| `openai`, pre-`/apply-template` llama-server | `llama_server_tokenizer_untemplated` | ❌ no | Real vocabulary, but the per-turn template tokens are missing, so the true prompt is *larger*. Upgrade llama-server for an exact count |
| `claude` (`claude -p` CLI) | `character_heuristic` | ❌ no | `len(text) / 4` over the caller's own system + message text |
| `gemini` (`agy` CLI) | `character_heuristic` | ❌ no | Same heuristic — `agy` exposes no tokenizer either |
| remote-owned rows (`host:`/`hosts:` elsewhere) | *whatever the owner reports* | *passed through* | Proxied to the owning host's hub, which repeats this decision against the backend actually in front of it |
| `whisper` / `tts` | — | — | 400, same rejection `/v1/messages` gives ("POST audio to … instead") |

Every non-exact response also carries a `warning` string naming the method
and its blind spots (e.g. image/document blocks, which are not counted at
all). An unknown model 400s exactly as `/v1/messages` does, in the same
Anthropic error envelope.

**Why the CLI backends can't be exact.** `claude-*` and `gemini-*` are
driven through a subscription CLI, not an API key, so there is no
`count_tokens` call to forward to and no published offline tokenizer for
the current model generation. The only exact number available would come
from running the generation — which is the cost this endpoint exists to
avoid.

**Measured delta on the claude path** (2026-08-01, `claude-haiku-4-5`,
`system: "You are terse."` + one 15-char user turn): the hub returned
`input_tokens: 8`; the identical body through `/v1/messages` reported
`input_tokens: 10` with `cache_creation_input_tokens: 33590`. So the
heuristic is within ~20% of the caller's *own* content — but the call's
real input was ~33.6 k tokens, because `claude -p` prepends Claude Code's
own system prompt and tool definitions and they land in the cache-write
bucket. Budget against `input_tokens` from this endpoint as "how big is my
prompt", never as "what will this call cost".

**On-demand models.** Counting against a `startup: on_demand` row
(`gemma4-26b-a4b-it`, `glm-4.5-air`) loads it, because its tokenizer is the
only thing that can answer exactly — same spawn-and-wait a real request
does, and the same idle watchdog unloads it afterwards.

Generate an image (Google Imagen via `agy`) — OpenAI Images shape,
returns `data[].b64_json`:

```python
import base64
from openai import OpenAI
client = OpenAI(api_key="local-dummy", base_url="http://127.0.0.1:8000/v1")
r = client.images.generate(model="gemini_image", prompt="a red apple on white")
open("apple.png", "wb").write(base64.b64decode(r.data[0].b64_json))
```

```bash
curl -s http://127.0.0.1:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini_image","prompt":"a red apple on white"}'
```

Transcribe audio via whisper — through the hub's proxy at
`:8000/v1/audio/transcriptions`, which works from any machine (the hub
resolves the owning host and lands the call in the observability ring):

```bash
# through the hub proxy (portable — works whether or not this box owns whisper)
curl -s -F file=@clip.wav -F response_format=json \
  http://127.0.0.1:8000/v1/audio/transcriptions

# only on the host that actually runs the backend (gaming today): direct to
# whisper-server, lower overhead, skips the observability ring
curl -s -F file=@clip.wav -F response_format=json \
  http://127.0.0.1:8090/v1/audio/transcriptions
```

Translate non-English audio to English via the translate slot (medium runs
eager on CPU, ~1.5 GB RAM, on the owning host):

```bash
# portable: through the hub proxy, which bridges task=translate → translate=true
curl -s -F file=@spanish.wav -F task=translate \
  http://127.0.0.1:8000/v1/audio/translations

# on the owning host only: whisper-server honors translate=true (not OpenAI's
# task=translate) on the direct-to-port path
curl -s -F file=@spanish.wav -F translate=true \
  http://127.0.0.1:8091/v1/audio/transcriptions
```

Or with the OpenAI SDK:

```python
from openai import OpenAI
asr = OpenAI(api_key="local-dummy", base_url="http://127.0.0.1:8000/v1")
with open("clip.wav", "rb") as f:
    r = asr.audio.transcriptions.create(model="whisper-large-v3-turbo", file=f)
print(r.text)
```

Synthesize speech (text → audio) via the TTS backend — through the hub's
proxy at `:8000/v1/audio/speech` (captured in the observability ring) or
directly to the backend port (lower overhead):

```bash
# through the hub proxy (lands in the observability ring)
# audio_speech → Piper; voice picks amy (default), ryan, ryan-high, or lessac
curl -s -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"audio_speech","input":"Hey, listen to this.","voice":"amy","response_format":"wav"}' \
  --output reply.wav

# kokoro-tts → Kokoro-82M; empty/default voice uses am_michael
curl -s -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro-tts","input":"Arming the perimeter.","voice":"am_michael","response_format":"wav"}' \
  --output kokoro-reply.wav

# Spanish female voice; use em_alex for the male profile
curl -s -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro-tts","input":"Hola, esta es una prueba de voz en español.","voice":"ef_dora","response_format":"wav"}' \
  --output kokoro-spanish.wav
```

```python
from openai import OpenAI
tts = OpenAI(api_key="local-dummy", base_url="http://127.0.0.1:8000/v1")
# model="audio_speech" → Piper (auto-loaded); model="orpheus-tts" → Orpheus.
audio = tts.audio.speech.create(model="audio_speech", voice="amy", input="Hey, listen to this.")
audio.stream_to_file("reply.wav")
```

`exaggeration` / `cfg_weight` are Chatterbox's emotion/"tone" dial; `voice`
selects a Piper voice (`amy`, `ryan`, `ryan-high`, `lessac`), an Orpheus preset
(`tara`, `leah`, …), a Kokoro voice id (`am_michael`, `af_bella`,
`am_fenrir`, `ef_dora` (Spanish female), `em_alex` (Spanish male), …), or a Chatterbox cloning clip at
`config/tts_voices/<voice>.wav`. Piper and Kokoro honor `speed` in the
0.5–2.0 range; Chatterbox/Orpheus accept it for API compatibility. Add
`"stream_format":"audio"` to **stream** audio incrementally when the engine
supports it; otherwise the backend returns a single final chunk. The hub
exposes every enabled TTS model on the same route, so a client switches
engines just by changing `model`. Defaults, formats, streaming, voice
cloning, and the Orpheus GGUF caveat are in [docs/add-tts.md](docs/add-tts.md).
An explicit unknown model or voice returns HTTP 400; only omitted fields use
the configured defaults.

The admin Playground lists every configured voice model, marks stopped models,
and filters language and voice choices to the selected engine. It also shows
only controls the engine supports. The language selector is UI metadata: calls
still use the same `model` and `voice` fields as Home Automation, App Launcher,
and WhatsApp Radar.

## API headers (issue #461)

What the hub reads off a request and what it writes back, so an
`anthropic`-SDK client sees the header vocabulary it expects.

**Accepted on the way in:**

| Header | Effect |
| --- | --- |
| `Authorization: Bearer <token>` | Authenticates a non-loopback caller. |
| `x-api-key: <token>` | Same thing — the Anthropic SDK's own credential header, so `Anthropic(api_key="<hub token>", base_url=…)` authenticates with no `default_headers` special-casing. A wrong value is rejected exactly as a wrong bearer token is. |
| `?token=<token>` | Same thing, on the URL (bookmarked/shared admin links). |
| `anthropic-version` | Echoed back. The hub does not vary its wire shape by version, so this is an acknowledgement, not a negotiation. |
| `anthropic-beta` | Echoed back, **never honoured** — see the caveat below. |
| `X-Trace-Id` | Bridged into a W3C `traceparent`, so repeat calls land in one Langfuse trace. |

Loopback callers still bypass authentication entirely — none of the above
changes that.

**Emitted on the way out:**

| Header | Meaning |
| --- | --- |
| `request-id` | Correlation ID for the request, on every response except `/admin/static/*`. Aliased to the trace ID, so it equals `X-Trace-Id` and is the same value the hub logs (`request-id=… POST /v1/messages -> 200`) and Langfuse traces. When telemetry is off (`OTEL_SDK_DISABLED=true`) there is no trace ID, so the hub mints a random one and no `X-Trace-Id` is emitted. |
| `X-Trace-Id` | Unchanged — the existing contract, emitted whenever a span is live. |
| `anthropic-version` | The version the caller sent, or `2023-06-01` when it sent none. |
| `anthropic-beta` | The betas the caller asked for, verbatim. |
| `Warning: 299 local-llm-hub "…"` | Accompanies any echoed `anthropic-beta`. |

**`anthropic-beta` caveat — read this before relying on one.** The hub
implements **no** Anthropic beta feature. It accepts the header (rejecting
it would 400 an SDK caller who set a beta the hub has no opinion about) and
echoes it back as a receipt, but every requested beta that is not
implemented also gets an RFC 7234 advisory on the same response:

```
anthropic-beta: context-1m-2025-08-07
Warning: 299 local-llm-hub "anthropic-beta received but not implemented: context-1m-2025-08-07"
```

The echo means "received", never "honoured". The implemented set is
`IMPLEMENTED_BETAS` in `src/anthropic_headers.py` — empty, and a value only
goes in there when the beta is genuinely wired up.

Check it from loopback:

```bash
curl -is -X POST http://127.0.0.1:8000/v1/messages \
  -H "content-type: application/json" \
  -H "anthropic-beta: context-1m-2025-08-07" \
  -d '{"model":"claude_haiku","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}' \
  | findstr /i "request-id x-trace-id anthropic- warning"
```

## CORS — calling the hub from a browser (issue #462)

Page JavaScript can call the hub directly. A sister webapp on this
machine does **not** need to proxy `/v1/messages` through its own backend
just to satisfy the browser.

**What is allowed out of the box:** every loopback origin — `http` or
`https`, `localhost` / `127.x.x.x` / `[::1]`, any port. Nothing else.

That default is not arbitrary: loopback is *already* the hub's trust
boundary (a loopback caller skips the bearer token entirely), so letting a
loopback origin read the response grants nothing it did not already have.
A page served from somewhere else still makes its request from
`127.0.0.1`, so it is the **origin** check — not the client IP — that
stops it reading the answer.

**Adding an origin:** name it in `config/webapp_config.json` (template:
`config/webapp_config.sample.json`).

```json
{
  "cors_allow_origins": ["https://llm.example.com"]
}
```

Exact origins only — scheme + host + port, no path, no trailing slash.
This is read once at startup, so restart the hub (`tray.bat --restart`)
after changing it; that is unlike `auth_token`, which is re-read per
request.

**No wildcard ships, and `"*"` is not a way to get one.** A `"*"` entry is
dropped with a warning rather than honoured. `allow_origins: ["*"]` is
precisely the shape that turns a loopback-trusting service into one any
page in any tab can drive, and the hub declines to offer it. For the same
reason `allow_credentials` is off: the hub sets no cookies and uses no
HTTP-auth realm — it authenticates from `Authorization` / `x-api-key` /
`?token=`, all of which a plain cross-origin `fetch` carries fine — so the
credentialed-CORS risk class is declined outright instead of being
contained by the origin list.

**CORS is not an auth bypass.** The middleware sits outermost so a
preflight `OPTIONS` — which a browser sends with no credentials, by
design — is answered instead of 401'd. The real request behind it travels
the full stack and meets the bearer gate exactly as before.

**Readable response headers.** A header a browser cannot read is the same
as not sending it, so the hub's own contracts are on `expose_headers`:
`request-id`, `X-Trace-Id`, `anthropic-version`, `anthropic-beta`,
`Warning`, `WWW-Authenticate`. Request headers are a named list (not `*`)
covering both SDKs' vocabulary — `authorization`, `content-type`,
`x-api-key`, `anthropic-version`, `anthropic-beta`, the OpenAI
`openai-*` headers, and the `x-stainless-*` telemetry headers both
generated SDKs attach unconditionally. The full policy, and why each
piece is the way it is, is `src/cors_policy.py`.

From a page on any loopback origin:

```js
const r = await fetch("http://127.0.0.1:8000/v1/models");
console.log(r.headers.get("X-Trace-Id"), await r.json());
```

Check the preflight by hand — allowed, then refused:

```bash
curl -is -X OPTIONS http://127.0.0.1:8000/v1/messages \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: content-type, x-api-key"

curl -is -X OPTIONS http://127.0.0.1:8000/v1/messages \
  -H "Origin: https://evil.example" \
  -H "Access-Control-Request-Method: POST"
```

## Observability (issue #4)

The hub emits OpenTelemetry traces + metrics via OTLP/gRPC into a local
Langfuse stack started by `start_langfuse.bat`. The admin SPA's
**OTel** tab shows stack health, a per-model leaderboard, and a
live trace feed with deep-links into the Langfuse UI for inspection.

Default mode captures raw prompts and completions — fine on a personal
localhost hub, but flip `OTEL_HASH_PROMPTS=true` (in `.env`) any time
the hub binds beyond loopback. `OTEL_SDK_DISABLED=true` turns
telemetry off entirely.

## Self-hosted SearXNG (home-automation#321, #438)

A self-hosted [SearXNG](https://docs.searxng.org/) meta-search instance,
started by `start_searxng.bat` / stopped by `stop_searxng.bat`
(`docker compose -f docker/searxng/docker-compose.yml`), serving on
`:8085`. It backs a `rest`-type function on the Home Assistant voice
assistant's Tier-3 freeform LLM brain, giving it real web-search access —
see home-automation#321 for the decision record. No API key, no
per-query cost, and — like the hub's own `:8000` — bound to `0.0.0.0`
for LAN reachability but never meant to be port-forwarded to the public
internet. Not otherwise integrated with this repo's Python/webapp code.
`docker/searxng/config/` (the container-generated `settings.yml`,
including a fresh random secret key) is gitignored — JSON output is off by
default in SearXNG, so `start_searxng.bat` runs
`docker/searxng/ensure_json_format.py` after every start to idempotently
patch `search.formats` on the generated file (never touching `secret_key`),
restarting the container only the one time it actually changes something.

## Coding agent usage (issues #20, #71, #231, #280)

The **Code** tab is a passive, multi-vendor view of host-side coding-agent
activity.  It parses each agent's local session logs server-side — zero
subprocesses, no wrapper around any binary, no impact on the running CLIs:

- **Claude Code** (`vendor="claude"`) — the JSONL transcripts at
  `~/.claude/projects/<encoded-path>/*.jsonl` (`src/code_usage.py`).
- **Codex / OpenAI** (`vendor="codex"`) — the rollout JSONL files at
  `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` (`src/codex_usage.py`).  One
  record per `token_count` event, using the per-turn delta `last_token_usage`
  (never the cumulative `total_token_usage`, which would double-count).
- **GitHub Copilot** (`vendor="copilot"`, `src/copilot_usage.py`) — two local
  sources, both carrying **exact** billed AI Credits (not a rate-table
  estimate):
  - The **Copilot CLI**'s `~/.copilot/session-state/<uuid>/events.jsonl`,
    written only for a *clean-shutdown* session — one record per model used
    in that session (session-granular; the CLI doesn't expose a per-turn
    breakdown), priced from the `session.shutdown` event's `totalNanoAiu`
    (`credits = totalNanoAiu / 1e9`).
  - **VS Code Copilot Chat**'s per-session
    `%APPDATA%\Code\User\workspaceStorage\<hash>\chatSessions\<uuid>.jsonl`
    event log (macOS: `~/Library/Application Support/Code/...`), which
    carries an exact `copilotCredits` float per request plus the resolved
    model — parsed via a minimal replay of the file's patch stream (not a
    general JSON-patch engine, just the couple of fields usage needs).
  Both sources only see sessions that reached this specific machine and
  wrote a complete log — sessions that crashed mid-flight, or ran
  elsewhere, are invisible to them. A separate **"Copilot credits
  (official)" card** (only shown on the Copilot vendor tab) fills that gap
  with the *authoritative* GitHub billing API — per-day × per-model spend,
  no session/project attribution, requires a `GITHUB_COPILOT_BILLING_PAT`
  fine-grained PAT (`.env`, "Plan: read-only" permission) or the card shows
  "not configured" rather than erroring (`src/copilot_billing.py`,
  `GET /admin/api/code/copilot/billing`).
- **AGY / Antigravity** (`vendor="agy"`, `src/agentsview_usage.py`) — sourced
  from the optional external
  [AgentsView](https://github.com/kenn-io/agentsview) service, which indexes
  `agy`'s local session storage the hub declined to reverse-engineer itself
  (#72/#279). Merges AgentsView's `gemini` (hub-routed calls) and
  `antigravity-cli` (interactive) slugs into one AGY vendor; other agents
  AgentsView knows about are deliberately not surfaced (curated map in
  `agentsview_usage.py`). Polled over HTTP, never from raw files; the hub
  runs fully without it. AgentsView appears in the Hub tab's **Services**
  card and the Models tab's **Service startup** toggles — the hub launches it at
  startup when toggled on (exe from `.venv-agentsview/`, `AGENTSVIEW_EXE`,
  or PATH). Setup and behaviour:
  [docs/code-usage-agentsview.md](docs/code-usage-agentsview.md).

A **vendor selector** (All / Claude / Codex / Copilot / AGY) sits above the
period toggle.
**All** sums every vendor into the headline counters and ≈ $ costs and shows a
**Per-vendor** breakdown table; picking a single vendor scopes the whole panel
to it.  The requests tile also shows the grand-total ≈ $ cost.  Counters and
per-model / per-project breakdowns also toggle between **Day / Week / Month /
All**, each with a delta badge (green ↑ / red ↓) vs. the previous comparable
period — or a "new" badge when a metric had no activity in the prior window
(e.g. a vendor used for the first time this period).

In the per-project table, projects under the `automation` workspace are shown by
folder name (the `automation-` prefix is stripped; the workspace root itself
stays `automation`), and long names are truncated with “…” to keep the table
readable on mobile — hover for the full path.

The input, output, and cache-read tiles show an **≈ $… equivalent API cost**
(issue #52) — what those tokens would have billed on the metered API, priced
per model from `config/claude_pricing.json` (Anthropic) and
`config/openai_pricing.json` (OpenAI); refresh those files when a provider
changes prices.  Codex usage is subscription-metered, so the ≈ $ figure is an
*estimate* against OpenAI list prices.  **Copilot and AgentsView-sourced
vendors are the exceptions:** Copilot's cost figure is the session/VS-Code
log's own exact `credits_usd`, and AgentsView vendors carry the cost
*as reported by AgentsView* (its own estimate for subscription agents) —
neither is re-priced against the hub's rate tables, even when the underlying
model resolves to a Claude model.

Two cross-vendor token semantics to keep in mind: for **Codex**, `cached_input`
tokens are a *subset* of input (the cost path prices the non-cached remainder at
the input rate and the cached portion at the cheaper cached rate), and
`reasoning_output_tokens` are a *subset* of output — surfaced as an "incl. …
reasoning" sub-note under the output tile, never added on top.  Claude's
`cache_read` tokens, by contrast, are reported separately/additively.  The
OpenAI >272K long-context surcharge (2× input / 1.5× output) is not modelled.

A **Usage trend** card (issue #50) sits below the breakdowns with four stacked
area charts — input tokens, output tokens, requests, and cache reads — with one
coloured series per model: the Claude families (Haiku / Sonnet / Opus) keep
fixed colours, and every other model (e.g. Codex `GPT-5.5`) gets its own series
colour; only an unattributable model id falls into a grey "Other" band.  Day →
last 14 days; Week → last 12 weeks; Month → last 12 months; All → charts hidden.  A "Recent sessions" list shows the last 15
sessions across every project on this host.

> **⚠️ The local JSONL source undercounts Claude two ways (verified 2026-07-12, #280).** (a) Sessions bridged through **claude.ai/code** (web/desktop remote-control) write `mode`/`permission` lines but **no assistant/usage records** to the local transcripts — their tokens are only visible via the **OTel tab**'s "Claude Code (host CLI)" panel (issue #68): the hub runs its own OTLP-metrics receiver (`POST /v1/metrics`) that Claude Code's telemetry export can point at; see [docs/telemetry-langfuse.md](docs/telemetry-langfuse.md#claude-code-otel-metrics-receiver-issue-68) for the host env vars. (b) Claude Code **prunes transcripts after ~30 days** (`cleanupPeriodDays`) — to survive that, the hub snapshots daily rollups per (date, vendor, model, project) into `data/code_usage_history.json` (`src/code_usage_history.py`: max-merge on write, per-vendor cutoff on read so no day is double-counted) and feeds them back into the "All" period after the source files are gone. History accumulates from 2026-07-12 forward; days already pruned before then are unrecoverable from disk. On the plus side, newer Claude Code also writes per-session directories with **sub-agent transcripts** (`projects/<proj>/<session>/subagents/agent-*.jsonl`), which the parser now includes — the old sub-agent blind spot is captured where those exist; finer per-subagent attribution is tracked upstream in [anthropics/claude-code#22625](https://github.com/anthropics/claude-code/issues/22625).

> **Why no host OTEL → Langfuse bridge?**  Issue #20 originally shipped an
> opt-in path to forward host Claude Code traces into the hub's Langfuse
> instance.  In practice the JSONL counters above already give the
> community-standard tokens-in / tokens-out view, and the trace-graph view
> wasn't worth the wiring cost — Langfuse's OTLP receiver only ingests
> traces (not metrics/logs), Claude Code traces are still a beta signal,
> and the per-signal exporter config is fiddly.  We removed the env-var
> snippet rather than ship something half-working.  Full rationale and
> diagnostics in [#22](https://github.com/ferraroroberto/local-llm-hub/issues/22) —
> revisit if Anthropic stabilises tracing or Langfuse adds metrics
> ingestion.

See [docs/telemetry-langfuse.md](docs/telemetry-langfuse.md) for the
full architecture, what's captured, the X-Trace-Id contract, and
limitations. For client-side trace correlation + feedback posting see
[docs/clients-telemetry-contract.md](docs/clients-telemetry-contract.md).

## Test

Unit tests (fast, no real `claude` / GPU calls):

```bat
.venv\Scripts\python -m pytest -q
```

End-to-end smoke test (requires hub + whichever backends you care
about running):

```bat
.venv\Scripts\python scripts\smoke_test.py
```

It iterates every enabled model from the registry, skips backends
whose port isn't reachable, and reports per-model pass/fail.

## Limitations (intentional — lightweight)

- **Partial streaming.** `POST /v1/chat/completions` with
  `stream: true` is fully supported for local backends — the hub
  proxies llama-server's SSE through, scrubbing `<think>...</think>`
  blocks from reasoning models (qwen / glm) so OpenAI-shape clients
  see only the final answer. The Anthropic-shape `POST /v1/messages`
  still returns a single JSON object when `stream: true` (Anthropic
  event translation is on the backlog below).
- Multi-turn chats are flattened into a single prompt for `claude -p`.
  (The local backends handle multi-turn natively through llama-server.)
- Tool-use translation across Anthropic ↔ OpenAI shapes is not
  implemented for qwen/glm. OpenAI-shape callers get native tool calls
  from llama-server's `--jinja` templates; Anthropic-shape callers to
  qwen/glm are text-only for now. Claude tool use passes through
  unchanged.
- **Image and document content blocks are supported on the `claude-*`
  and `gemini-*` subscription paths** — the hub base64-decodes each
  `image` / `document` block to a per-request temp dir, adds that dir to
  the CLI workspace (`claude --add-dir` / `agy --add-dir`) and references
  the files inline (`@<basename>` for `agy`). Adding the dir to the
  workspace is what makes `agy` resolve the reference deterministically
  instead of searching the filesystem (which read attachments only
  intermittently — issue #63). `document`
  blocks accept any file the CLI can read: PDF, plus text/data/code
  files (JSON, CSV, Markdown, …); the `media_type` picks the temp-file
  extension and unknown types fall back to `.bin` (still read as bytes).
  Local `llama-server` backends (`qwen3.5-*`, `gemma4-*`) are text-only
  and return 400 with a hint to retry on a subscription model. URL
  sources degrade to a text reference (not fetched). Extended-thinking
  blocks are still dropped at the shape boundary. **Media only travels on
  the Anthropic shape:** `POST /v1/chat/completions` flattens a
  conversation to a single text prompt for the `claude-*` / `gemini-*`
  CLI dispatch, so an OpenAI-shape `image_url` (or any other non-`text`
  content part) is refused with a 400 pointing at `/v1/messages` — it
  used to be dropped silently and answered as if the text were the whole
  question (issue #474).
- Token counts reflect what each backend reports in its response. The
  `agy` CLI does not surface token counts, so usage on the `gemini-*`
  path is reported as zero.
- **Gemini calls are serialized.** `agy` selects its model from global
  persisted state, so the hub holds a lock across the model switch and
  the print call. Concurrent `gemini-*` requests run one at a time;
  switching model between calls adds a one-time interactive step.
- **The `agy` attachment path can only be verified locally.** GitHub CI (`windows-latest`) has no authenticated `agy` / Gemini subscription. `tests/test_gemini_attachments_live.py` is the live regression guard for the `--add-dir` fix from #63 — it is skipped by default and must be run manually on the Windows reference box after any change to `src/gemini_cli.py`'s attachment handling: `$env:HUB_LIVE_GEMINI = "1"; .venv/Scripts/python.exe -m pytest tests/test_gemini_attachments_live.py -v`.

## Backlog for improvement

Remaining API-parity and developer-experience gaps are tracked as a
GitHub issue, not restated here —
[`#453`](https://github.com/ferraroroberto/local-llm-hub/issues/453)
("Backlog: API-parity and DX improvements") carries the full, ordered
list and stays current as entries ship.

## License

[MIT](LICENSE). Use it, fork it, break it — just keep the copyright
notice. Note that the license covers *this code* only; your use of the
underlying `claude` CLI is still governed by Anthropic's terms (see
[Scope & usage policy](#scope--usage-policy) above) and the model
weights follow their own licenses
([Gemma terms](https://ai.google.dev/gemma/terms),
[Whisper / OpenAI MIT](https://github.com/openai/whisper/blob/main/LICENSE),
plus [Qwen](https://huggingface.co/Qwen) /
[GLM](https://huggingface.co/zai-org) for the demoted candidates).
