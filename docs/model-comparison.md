# Model comparison

Side-by-side technical snapshot of every model the hub currently
knows about. Designed to grow — one row per model. Add a row when
adding a new backend (registry entry + launcher + test + docs all
live elsewhere; this page is just the reference table).

Columns are deliberately short so the table renders legibly in
markdown. Where a precise number isn't meaningful for a backend
(e.g. Claude runs off-device), the cell says so directly.

## Local llama.cpp backends + Claude subscription

| Model (hub id) | Family | Params | Quant | GGUF size | Context | VRAM fit (16 GB) | Hub port | Typical tok/s* | Official docs |
|---|---|---|---|---|---|---|---|---|---|
| `claude-haiku-4-5` (aliases: `claude-sonnet-4-6`, `claude-opus-4-7`) | Claude (Anthropic) | n/a (cloud via `claude -p`) | n/a | n/a | per model — see docs | n/a | 8000 (hub) | n/a (subscription) | [anthropic.com/claude](https://docs.anthropic.com/en/docs/about-claude/models) |
| `qwen3.5-9b` | Qwen 3.5 (Alibaba) | 9 B dense | Q4_K_M | 5.3 GB | 16 384 | full GPU (`-ngl 99`) | 8081 | ~80–110 | [Qwen org on HF](https://huggingface.co/Qwen) |
| `glm-4.5-air` | GLM-4.5-Air (Zhipu / zai-org) | 106 B / 12 B active MoE | Q4_K_M | 46.6 + 21.4 GB (2 shards) | 16 384 | attention on GPU, experts on RAM (`-ot .ffn_.*_exps.=CPU`) | 8082 | ~6–10 | [zai-org on HF](https://huggingface.co/zai-org) |
| `gemma3-12b-it` | Gemma 3 (Google) | 12 B dense | Q4_K_M | 6.8 GB | 16 384 | full GPU (`-ngl 99`) | 8083 | ~70–90 | [Gemma docs](https://ai.google.dev/gemma/docs) · [card](https://huggingface.co/google/gemma-3-12b-it) |
| `gemma3-27b-it` | Gemma 3 QAT (Google) | 27 B dense | Q4_0 (QAT) | 14.5 GB | 4 096 | partial GPU (`-ngl 50`) + `--flash-attn on` | 8084 | ~15–25 | [QAT announcement](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/) · [card](https://huggingface.co/google/gemma-3-27b-it-qat-q4_0-gguf) |
| `gemma3n-e4b-it` | Gemma 3n (Google, edge) | ~4 B effective | Q4_K_M | 4.2 GB | 16 384 | full GPU (`-ngl 99`) | 8085 | ~120–160 | [Gemma 3n docs](https://ai.google.dev/gemma/docs/gemma-3n) · [card](https://huggingface.co/google/gemma-3n-E4B-it) |
| `whisper-small` | whisper.cpp (OpenAI) | 244 M (ASR, not chat) | ggml f16 | 466 MB | audio (30 s chunks) | tiny (~1 GB) | 8090 | realtime-factor ~6–10× (GPU) | [whisper.cpp](https://github.com/ggerganov/whisper.cpp) · [models](https://huggingface.co/ggerganov/whisper.cpp) |

\* Single-stream generation on an RTX 5060 Ti 16 GB, short
prompts (~100 input tokens). Ranges are indicative, not a
benchmark — prompt prefill, long contexts, and concurrent
traffic move the numbers. Measure on your workload before
committing.

## Roles at a glance

| Role | Model | Why |
|---|---|---|
| **Default agentic / coding** | `glm-4.5-air` | 106 B MoE quality; the MoE CPU-offload keeps it viable on 16 GB VRAM. Slow but strong. |
| **Fast dense all-rounder** | `qwen3.5-9b` | The fastest "smart enough" option. Tool-call-capable via `--jinja`. |
| **Classifier verifier (default)** | `gemma3-12b-it` | Strict instruction-following + JSON-schema adherence. See [20260420-add-gemma-for-action-item-classification.md](20260420-add-gemma-for-action-item-classification.md). |
| **Classifier quality ceiling** | `gemma3-27b-it` | QAT 4-bit, near-BF16 quality. Benchmark the 12B against it before shipping. |
| **Edge / ultra-fast probe** | `gemma3n-e4b-it` | Smallest footprint, highest tok/s. Use for first-pass triage or where latency > quality. |
| **Cloud parity check** | `claude-haiku-4-5` / `sonnet-4-6` / `opus-4-7` | Off-device baseline via `claude -p`; same hub, just swap the `model` string. |
| **Speech-to-text (ASR)** | `whisper-small` | whisper.cpp on :8090. OpenAI-compatible `/v1/audio/transcriptions`. Port is a shared mutual-exclusion lock with the `transcribe_voice` project. Switch size by editing [config/models.yaml](../config/models.yaml) (`ggml-<size>.bin`) and re-running `python -m src.install --fix`. |

## How to add a new row to this table

1. Add the model to [config/models.yaml](../config/models.yaml)
   and the relevant host's `enabled` list (see the existing
   plan docs for the pattern).
2. Add a launcher pair (`run_<id>.bat` / `.sh`) and a line
   in `run_all.*`.
3. Extend the registry test.
4. Run `python -m src.install --fix` and verify the smoke test
   shows a pass row.
5. Append a row to the table above. Keep the columns tight —
   link out to docs rather than inlining prose.

## Intentional exclusions

- **Gemma 3 4B**, **Gemma 3n E2B** — dominated by the 12B at the
  classifier task on the reference hardware (plan doc flags
  them as regressions); E4B is kept as the small/fast option.
- **Gemma PaliGemma / vision variants** — no image inputs in
  our workloads.
- **Hugging Face Inference API, OpenRouter, etc.** — the
  project is local-first; cloud models go through Claude's
  subscription path, not arbitrary remote APIs.
