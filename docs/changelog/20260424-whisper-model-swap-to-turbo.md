# What we did — 2026-04-24 (evening)

Swapped the whisper backend from `ggml-small.bin` to
`ggml-large-v3-turbo.bin`. Single-model repo by design — this is a
substitution, not an additional slot. Pairs with
[20260422-add-whisper-asr.md](20260422-add-whisper-asr.md) (how whisper
slotted into the hub in the first place) and
[20260424-whisper-turbo-vs-large-v3.md](20260424-whisper-turbo-vs-large-v3.md)
(the turbo-vs-large-v3-vs-quant rationale).

## Why now

The `small` tier was the cheapest thing that worked when we wired
whisper up on 2026-04-22. It's usable but not great on Spanish — the
`transcribe_voice` dictation use case noticeably benefits from the
jump to large-v3-class quality. Turbo is the right pick instead of
full large-v3: same encoder, decoder pruned from 32 → 4 layers,
~2× faster, near-identical WER on Spanish/English. On the 16 GB
reference GPU we have headroom for ~2 GB of VRAM without squeezing
the llama backends. See the rationale doc for numbers.

## What changed

Mechanical rename + file swap. No new code paths, no CLI flags, no
secondary registry slot. The backend key stays `whisper`; only the
weights filename and the `display_name` move.

| file                                         | change |
| -------------------------------------------- | ------ |
| [config/models.yaml](../../config/models.yaml)                                | `display_name`, `hf_pattern`, `model_path` → turbo; comment lists turbo in the size options |
| [launchers/run_whisper.bat](../../launchers/run_whisper.bat)                  | banner + title now say `whisper-large-v3-turbo` |
| [tests/test_model_registry.py](../../tests/test_model_registry.py)            | fixture + assertions switched to the new display name & filename |
| [README.md](../../README.md)                                                  | bullet, layout tree, Run table, Setup size numbers, OpenAI SDK example all reference the new model |
| [docs/model-comparison.md](../model-comparison.md)                            | whisper row updated; new link to the turbo-vs-large-v3 doc |
| [docs/changelog/20260422-add-whisper-asr.md](20260422-add-whisper-asr.md)     | "Addendum 2026-04-24 — model upgrade" paragraph at the bottom; historical body left intact |
| [docs/changelog/20260424-whisper-turbo-vs-large-v3.md](20260424-whisper-turbo-vs-large-v3.md) | new — explains the turbo pick, size/VRAM table, when large-v3 would still be preferred, q5_0 alternative |

`launchers/run_whisper.sh`, `src/backend_process.py`,
`src/run_backend.py`, `scripts/download_models.py`,
`scripts/smoke_test.py` did **not** need changes — they were already
fully data-driven off the registry.

## Sister-project changes

`E:\automation\automation\audio\transcribe_voice` shares the :8090
port as a mutual-exclusion lock and reads the same weights file from
this repo's `models/` directory. Three files tracked in that repo
needed the path swap so the two sides stay in lockstep:

- `whisper_server/whisper_server.yaml` — `model.path` now points at
  `ggml-large-v3-turbo.bin` with a dated comment explaining the swap.
- `whisper_server/manager.py` — the default path fallback in
  `load_config()` updated (only relevant if the YAML is missing).
- `README.md` — example yaml block and first-run checklist updated.

## Verification (what we actually ran)

1. `python -m pytest -q` → **17 passed** (registry test keeps asserting
   backend/engine/port on the new display name).
2. `python -m src.install` → reports `missing` for
   `ggml-large-v3-turbo.bin`, no mention of `ggml-small.bin`.
3. `python scripts/download_models.py --only whisper` → pulled
   1.62 GB from `ggerganov/whisper.cpp` into
   `models/ggml-large-v3-turbo.bin`.
4. `python -m src.run_backend whisper` → server log confirms the
   right model loaded:
   ```
   whisper_init_from_file_with_params_no_state: loading model from
     'E:\automation\claude-local-calls\models\ggml-large-v3-turbo.bin'
   whisper_model_load: n_text_layer  = 4      # distilled decoder
   whisper_model_load: type          = 5 (large v3)
   whisper_model_load:        CUDA0 total size =  1623.92 MB
   ```
5. `curl -F file=@silence.wav http://127.0.0.1:8090/v1/audio/transcriptions`
   → HTTP 200 with JSON body. (Silence transcribes to "Thank you." —
   a well-known whisper hallucination across sizes, not a regression;
   confirms the model is actually inferencing.)
6. **Mutex cross-check from transcribe_voice** —
   `launcher.py server status` reported
   `✅ running (external — started elsewhere) @ http://127.0.0.1:8090 [external]`.
   Exactly the same behaviour as before the swap.
7. `TranscriptionClient.transcribe_file(...)` against the running
   server returned `'Thank you.'` — proves the sister-project client
   path works unchanged against the new weights.
8. `grep -r "whisper-small\|ggml-small" .` → hits only in historical
   changelog docs (`20260422-add-whisper-asr.md`,
   `20260424-launchers-and-docs-reorg.md`, and the rationale doc's
   size-comparison table, all intentional).

## Non-goals (carried over from the task brief)

- No model-selection CLI flag. One model. One filename.
- No parallel `whisper_small` / `whisper_large_v3` rows — single slot.
- No change to the :8090 mutex contract with `transcribe_voice`.
- No hub passthrough for audio endpoints.
- No streaming, no VAD.
