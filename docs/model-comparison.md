# Model comparison

Side-by-side technical snapshot of every model the hub currently
knows about. Designed to grow — one row per model. Add a row when
adding a new backend (registry entry + launcher + test + docs all
live elsewhere; this page is just the reference table).

Columns are deliberately short so the table renders legibly in
markdown. Where a precise number isn't meaningful for a backend
(e.g. Claude runs off-device), the cell says so directly.

## Local llama.cpp backends + Claude subscription

| Model (hub id) | Family | Params | Quant | GGUF size | Context | VRAM fit (16 GB) | Hub port | Typical tok/s* | References (official · card · benchmarks) |
|---|---|---|---|---|---|---|---|---|---|
| `claude-haiku-4-5` (alias `claude_haiku`) | Claude (Anthropic) | n/a (cloud via `claude -p`) | n/a | n/a | per model — see docs | n/a | 8000 (hub) | n/a (subscription) | [anthropic.com/claude](https://docs.anthropic.com/en/docs/about-claude/models) · [llm-stats](https://llm-stats.com/models?provider=anthropic) |
| `claude-sonnet-4-6` (alias `claude_sonnet`) | Claude (Anthropic) | n/a (cloud via `claude -p`) | n/a | n/a | per model — see docs | n/a | 8000 (hub) | n/a (subscription) | [anthropic.com/claude](https://docs.anthropic.com/en/docs/about-claude/models) |
| `claude-opus-4-8` (alias `claude_opus`) | Claude (Anthropic) | n/a (cloud via `claude -p`) | n/a | n/a | per model — see docs | n/a | 8000 (hub) | n/a (subscription) | [anthropic.com/claude](https://docs.anthropic.com/en/docs/about-claude/models) |
| `claude-fable-5` (alias `claude_fable`) | Claude (Anthropic) | n/a (cloud via `claude -p`) | n/a | n/a | 1 M tokens | n/a | 8000 (hub) | n/a (subscription) | [anthropic.com/claude](https://docs.anthropic.com/en/docs/about-claude/models) |
| `gemini_pro` (Gemini 3.1 Pro) | Gemini 3.1 (Google) — preview | n/a (cloud via `agy` CLI) | n/a | n/a | 1 M tokens | n/a | 8000 (hub) | n/a (AI Pro/Ultra required since 2026-03-25) | [Gemini 3.1 Pro post](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/) · [Gemini CLI](https://geminicli.com/) · [DeepMind Pro](https://deepmind.google/models/gemini/pro/) |
| `gemini_flash` (Gemini 3.6 Flash) | Gemini 3.6 (Google) | n/a (cloud via `agy` CLI) | n/a | n/a | per model — see docs | n/a | 8000 (hub) | n/a (subscription, Code Assist free tier) | [Gemini 3 Flash in CLI](https://developers.googleblog.com/gemini-3-flash-is-now-available-in-gemini-cli/) · [Gemini CLI quotas](https://geminicli.com/docs/resources/quota-and-pricing/) |
| `gemini_flash_lite` (alias `gemini_lite`, Gemini 3.5 Flash) | Gemini 3.5 (Google) — previous gen | n/a (cloud via `agy` CLI) | n/a | n/a | per model — see docs | n/a | 8000 (hub) | n/a (subscription, Code Assist free tier) | [Flash-Lite GA](https://cloud.google.com/blog/products/ai-machine-learning/gemini-3-1-flash-lite-is-now-generally-available) |
| `qwen3.5-4b` | Qwen 3.5 (Alibaba) — hybrid Gated DeltaNet + sparse MoE | 4 B base (sparse-MoE active set) | Q4_K_M | 2.6 GB | 65 536 (262 144 native) | full GPU (`-ngl 99`) + `--flash-attn on` | 8088 | ~110 | [Qwen 3.5 announcement](https://qwen.ai/blog/qwen3.5/) · [official org](https://huggingface.co/Qwen) · [GGUF we ship](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) |
| `gemma4-e4b-it` (fallback) | Gemma 4 (Google, edge / multimodal) | 8 B dense (text-only here) | Q4_K_M | 4.7 GB | 16 384 | full GPU (`-ngl 99`) | 8086 | ~92 | [Gemma 4 page](https://deepmind.google/models/gemma/gemma-4/) · [official card](https://huggingface.co/google/gemma-4-E4B-it) · [GGUF we ship](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) |
| `gemma4-26b-a4b-it` | Gemma 4 MoE (Google) | 25.2 B / 3.8 B active MoE | IQ4_XS (i-matrix) | 13.0 GB | 8 192 | full GPU (`-ngl 99`) + `--flash-attn on` | 8087 | ~91 | [Gemma 4 page](https://deepmind.google/models/gemma/gemma-4/) · [official card](https://huggingface.co/google/gemma-4-26B-A4B-it) · [GGUF we ship](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) |
| `whisper-large-v3-turbo` | whisper.cpp (OpenAI) | 809 M (ASR, not chat; distilled large-v3 w/ 4 decoder layers) | ggml f16 | 1.62 GB | audio (30 s chunks) | ~2 GB | 8090 | realtime-factor ~4–8× (GPU) | [whisper.cpp](https://github.com/ggerganov/whisper.cpp) · [ggml models](https://huggingface.co/ggerganov/whisper.cpp) · [turbo vs large-v3](whisper-turbo-vs-large-v3.md) · [OpenAI paper](https://arxiv.org/abs/2212.04356) |
| `whisper-medium-translate` | whisper.cpp (OpenAI) — eager CPU sibling | 769 M (medium) | ggml f16 | ~1.5 GB | audio (30 s chunks) | CPU-only | 8091 | ~realtime on 7800X3D | [official medium](https://huggingface.co/openai/whisper-medium) · [ggml models](https://huggingface.co/ggerganov/whisper.cpp) |
| `whisper-vanilla` | whisper.cpp (OpenAI) — lazy GPU, glossary-free | 809 M (same turbo model) | ggml f16 | 1.62 GB | audio (30 s chunks) | ~2 GB (only while loaded) | 8094 | realtime-factor ~4–8× (GPU) | [whisper.cpp](https://github.com/ggerganov/whisper.cpp) · [#128](https://github.com/ferraroroberto/local-llm-hub/issues/128) |
| `parakeet-tdt-0.6b-v3` | NVIDIA Parakeet TDT (NeMo) — CoreML via FluidAudio | 600 M (ASR, not chat) | n/a — `parakeet-tdt-0.6b-v3-coreml` bundle, precision not documented | n/a (no GGUF on this path) | audio | n/a — Apple Neural Engine, not GPU (`est_vram_mb: 0`) | 8098 | faster than whisper-turbo; see evaluation | [FluidAudio](https://github.com/FluidInference/FluidAudio) · [evaluation + trade-offs](parakeet-asr-evaluation.md) · [#138](https://github.com/ferraroroberto/local-llm-hub/issues/138) |

## Local text-to-speech backends

All rows speak the same OpenAI-compatible `POST /v1/audio/speech` shape through the hub on `:8000`; direct backend ports are listed for lower-overhead testing. The measured line is hub end-to-end on the Windows reference box for `input="Arming the perimeter."`, `response_format="wav"`, warm reps after backend load, **connection reused**. The hub proxies through a shared pooled httpx client (#165), so the hub route now adds **under ~0.1 s** over the direct port — it previously added ~0.26 s of per-request client construction.

| Model (hub id) | Engine | Size / runtime | Hub route | Direct port | Default voice | Measured latency | Notes |
|---|---|---|---|---|---|---|---|
| `piper-tts` (`audio_speech`) | Piper VITS via standalone binary | ONNX voice (~60 MB) + local binary | `model="audio_speech"` or `model="piper-tts"` | 8096 | `amy` (`en_US-amy-medium`) | direct median ~0.06 s; hub median ~0.06 s | Fast default for assistant replies and Home Assistant wiring. `piper.exe` runs **resident** (one process per voice+speed, ONNX voice loaded once), so short phrases skip the per-request model load (#163) — VITS inference itself is ~0.06 s (RTF ~0.05). Pre-resident it re-loaded the voice every call (~0.79 s direct / ~1.10 s hub). Voices: `amy`, `ryan`, `ryan-high`, `lessac`. |
| `orpheus-tts` | Orpheus-3B GGUF + SNAC | ~3 B, Q4_K_M GGUF + llama-server child | `model="orpheus-tts"` | 8093 | `tara` | direct median ~1.29 s; hub median ~1.36 s | Most expressive local voice; latency is **generation-bound** (LLM emits SNAC tokens at ~150 tok/s), so it scales with output length. Supports Orpheus presets (`tara`, `leah`, `jess`, `leo`, `dan`, `mia`, `zac`, `zoe`) and streaming time-to-first-audio (~0.5 s). Runs a failover chain — see `config/models.yaml` for current placement (#422). Measured: an RTX 5060 Ti sustains 1.84x real-time with SNAC on CPU (`--device cpu` — faster than GPU SNAC and ~1 GB VRAM cheaper); a GTX 1070 manages only 0.71x, compute-bound, which is why it ranks as the degraded tier. |
| `kokoro-tts` | Kokoro-82M ONNX Runtime | 82 M, int8 ONNX (~88 MB) + packed voices | `model="kokoro-tts"` | 8095 | `am_michael` | direct median ~1.99 s; hub median ~2.03 s | Low-footprint multilingual alternate. Spanish profiles: `ef_dora` (female), `em_alex` (male), both with `lang=es`. ONNX Runtime CUDA is enabled on the reference box, but the current `kokoro-onnx` int8 path is the **slowest** of the three for this short sample — slower than Orpheus despite the tiny model; keep it as a pronunciation-testing option, not for latency. |
| `chatterbox-tts` | Resemble Chatterbox | ~0.5 B torch package | `model="chatterbox-tts"` | 8092 | built-in / clip path | not remeasured for #156 | On-demand alternate with `exaggeration` / `cfg_weight` tone controls and optional reference-clip voice cloning. |

## Image-generation backends

Both local rows serve `POST /v1/images/generations` (OpenAI Images shape) through the hub on `:8000`; `POST /v1/images/edits` is **gemini-only** — editing on the ComfyUI path needs a different workflow graph (img2img/inpaint) and is deliberately out of scope. The two local rows are `startup: on_demand` with a 15-minute idle unload, so neither holds VRAM between requests. Latencies are on this repo's reference GPU at 1024²; the full controlled comparison, the upscale/refine path, and the Imagen spike findings are in [image-generation.md](image-generation.md).

| Model (hub id) | Engine | Weights | Direct port | Editing | Typical latency | Notes |
|---|---|---|---|---|---|---|
| `flux2_klein` (alias `klein`) | FLUX.2 [klein] 4B fp8 via ComfyUI | fp8 diffusion model + separate Qwen3-4B text encoder + FLUX.2 VAE (`est_vram_mb: 9400`) | 8190 | no | **~14 s** | The default local pick — ~3× faster than FLUX.1 and better at texture and typography. |
| `flux1_local` (alias `flux`) | FLUX.1 [dev] fp8 via ComfyUI | single fp8 checkpoint, `Comfy-Org/flux1-dev` (`est_vram_mb: 5700`) | 8188 | no | ~42 s (~96 s at 4K) | Reach for it when the prompt carries **numbers or spatial relations**, or when you want 4K — the only row whose upscale path is exercised end to end. |
| `gemini_image` (display `imagen`) | Google Imagen via the `agy` CLI | none — cloud, on the AI Pro/Ultra subscription | n/a | yes (procedural, slow) | seconds | The only honest image id on the `agy` path: its picker exposes no image model, and the selected *text* model doesn't change the image backend, so a `_pro` sibling would just be a second name for Imagen (#114). |

> **Demoted candidates** (kept defined in `config/models.yaml` but
> **not in the active rotation** — see `enabled:` for the active host):
> `qwen3.5-9b`, `glm-4.5-air`. Bring up ad-hoc with
> `launchers/run_model.bat qwen` / `run_model.bat glm` if you need them.

\* Single-stream generation on an RTX 5060 Ti 16 GB, short
prompts (~100 input tokens). Ranges are indicative, not a
benchmark — prompt prefill, long contexts, and concurrent
traffic move the numbers. Measure on your workload before
committing.

**Gemma 4 reasoning note:** Both `gemma4-*` models emit a
`message.reasoning_content` field alongside `message.content` — the
model thinks step-by-step before answering, and llama-server splits
the chain-of-thought from the final reply. Set `max_tokens` generous
enough (≥ ~150 for short answers, more for complex tasks) or the
budget runs out mid-reasoning and `content` comes back empty. Clients
that only read `content` will see "no reply" without realising the
model was still thinking.

## Roles at a glance

| Role | Model | Why |
|---|---|---|
| **agentic_light** (OpenClaw fast lane) | `qwen3.5-4b` | 4 B hybrid Gated DeltaNet + sparse MoE on full GPU (~3 GB). Apache 2.0, 262 k native context. The May 2026 frontier Tier A pick — strong instruction-following at ~110 t/s for routing, classification, JSON-schema work, and first-pass triage. `gemma4-e4b-it` stays in `enabled:` as fallback. |
| **agentic_heavy** (deep lane / transcripts / docs) | `gemma4-26b-a4b-it` | 25 B-total MoE with only 3.8 B active per token. IQ4_XS keeps the whole model on GPU; quality approaches a dense 27B at much higher tok/s thanks to MoE sparsity. Strong multilingual incl. Catalan. **On-demand since #422** (`startup: on_demand`, `idle_unload_minutes: 30`): the first request pays the ~13.4 GB load, 30 idle minutes unload it — the VRAM freed between uses is what lets orpheus live on tower. |
| **audio_transcribe** | `parakeet-tdt-0.6b-v3`, falling back to `whisper-large-v3-turbo` | Parakeet on the Apple Neural Engine is the role primary — faster than whisper-turbo, but it drops the "Claude Code" wake phrase and mangles "YOLO", so the chain falls back to whisper for general dictation. Whisper-turbo serves `/v1/audio/transcriptions` from whisper.cpp on :8090, whose port is a shared mutual-exclusion lock with the `transcribe_voice` project; distilled large-v3 (4 decoder layers) — ~2× faster than large-v3 at near-identical WER on Spanish/English. It carries the English tech-dictation glossary as its initial prompt, which biases language detection toward English; callers transcribing general multilingual audio should select `model=whisper-vanilla` (glossary-free, lazy, :8094 — #128) for unbiased auto-detect. Chain order and fallbacks live in `config/models.yaml` → `roles:`. |
| **audio_translate** | `whisper-medium-translate` | whisper.cpp medium on CPU, eager-loaded on :8091 (~1.5 GB RAM, always ready). Turbo's distilled decoder doesn't translate, so this slot fills the gap. A lazy-load mode (proxy + idle-unload) lives in `src/whisper_translate_proxy.py` for hosts that need to reclaim RAM. |
| **audio_speech** | `piper-tts` | Piper is the fast English default for short assistant replies. Orpheus remains available as `model="orpheus-tts"` for expressive English speech; Spanish uses explicit `model="kokoro-tts"` with `ef_dora` or `em_alex`. |
| **Cloud parity (Anthropic)** | `claude_haiku` / `claude_sonnet` / `claude_opus` / `claude_fable` | Off-device baseline via `claude -p`; same hub, just swap the alias. The aliases stay stable across version bumps — when Anthropic ships `claude-haiku-5`, only the row's `display_name` changes. Not a local role — never touched by `/swap-model`. |
| **gemini_pro (Google)** | `gemini_pro` | Subscription cloud route via the `agy` (Antigravity) CLI. AI Pro required for 3.1 Pro (paid since 2026-03-25). 1 M-token context. Image content blocks supported. |
| **gemini_lite (Google)** | `gemini_flash_lite` (alias `gemini_lite`) | Subscription cloud route via the `agy` (Antigravity) CLI. GA on 2026-05-07. `agy` >=1.1.8 dropped the picker's per-model High/Medium effort rows in favor of a single Flash generation split (#442) — Lite now means "previous-gen Flash" (`Gemini 3.5 Flash`) rather than "Medium effort"; still the cheaper/simpler pick relative to `gemini_flash`'s newest-gen row. Quotas share with Gemini Code Assist. |

> Roles are declared in `config/models.yaml` → `roles:`. Update them
> via `/swap-model` in Claude Code (interactive, edits the yaml +
> writes a launcher + optionally downloads weights).

## How to add a new row to this table

1. Add the model to [config/models.yaml](../config/models.yaml)
   and the relevant host's `enabled` list (see the existing
   plan docs for the pattern).
2. Add a line for it in `launchers/run_all.*` (the parameterized
   `launchers/run_model.bat` / `.sh <id>` already covers ad-hoc bring-up
   for any registry id — #448 — with nothing further to write).
3. Extend the registry test.
4. Run `python -m src.install --fix` and verify the smoke test
   shows a pass row.
5. Append a row to the table above. Keep the columns tight —
   link out to docs rather than inlining prose.

## Intentional exclusions

- **Gemma 3 family** (`gemma3-12b-it`, `gemma3-27b-it`,
  `gemma3n-e4b-it`) — superseded by the Gemma 4 entries above.
  Per the [latest-only policy](../README.md#latest-only-policy),
  older entries in the same family are removed once a comparable
  newer one ships.
- **Gemma PaliGemma / vision variants** — no image inputs in
  our workloads.
- **FLUX.2 [dev] 32B** (`flux2_dev`) — added in #498 alongside
  `flux2_klein` and removed in #501 as unusable on this hardware.
  The klein 4B row above covers the FLUX.2 generation instead.
- **Hugging Face Inference API, OpenRouter, etc.** — the
  project is local-first; cloud models go through Claude's
  subscription path, not arbitrary remote APIs.
- **GLM-5.2** (`zai-org/GLM-5.2`) — evaluated for the local coding
  lane and rejected: it is a single 744B-A40B MoE with no Air/Flash
  variant, and its smallest usable quant (UD-IQ2_M) needs ~245 GB
  RAM+VRAM vs. this box's ~144 GB — it does not load at any quant.
  Excellent coder, wrong size for the hardware. Revisit if a
  GLM-5.2-Air/Flash (~80–120 B) ships. Full analysis:
  [glm-5.2-evaluation.md](glm-5.2-evaluation.md) · [#141](https://github.com/ferraroroberto/local-llm-hub/issues/141).
