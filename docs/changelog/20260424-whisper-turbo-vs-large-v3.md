# Whisper turbo vs large-v3 — why turbo is the default — 2026-04-24

Short rationale for shipping `ggml-large-v3-turbo.bin` as the single
whisper model in this repo, instead of `ggml-large-v3.bin` or the
previous `ggml-small.bin`. Pairs with
[20260422-add-whisper-asr.md](20260422-add-whisper-asr.md) — that doc
covers *how* whisper slots into the hub; this one covers *which
weights*.

## Why turbo, not large-v3

Whisper `large-v3-turbo` is a distilled variant of `large-v3` released
with the OpenAI turbo update. The encoder is identical; the decoder is
pruned from 32 layers down to 4. That single change carries the whole
trade-off:

- **Roughly 2× faster at inference** than `large-v3` for the same audio,
  because decoding (not encoding) dominates wall-clock on GPU once the
  mel spectrogram is in VRAM.
- **Near-identical WER** on well-resourced languages — OpenAI's own
  turbo release notes call out English and the larger European
  languages explicitly, and the whisper.cpp README mirrors the same
  claim. Spanish and English — the only two languages
  `transcribe_voice` actually uses — are both comfortably in that
  "well-resourced" bucket.

For this project's dictation use case (Spanish and English short
utterances, clean close-mic audio), there is no measurable quality gap
worth paying 2× latency for.

## Size and VRAM

| weights                        | disk    | VRAM (approx, fp16) |
| ------------------------------ | ------- | ------------------- |
| `ggml-small.bin` (previous)    | 466 MB  | ~1 GB               |
| `ggml-large-v3.bin`            | 3.09 GB | ~3.5 GB             |
| `ggml-large-v3-turbo.bin`      | 1.62 GB | ~2 GB               |
| `ggml-large-v3-turbo-q5_0.bin` | 547 MB  | ~1 GB               |

Turbo F16 is 1.62 GB on disk and sits comfortably under 2 GB of VRAM on
the reference RTX 5060 Ti 16 GB. That leaves 14 GB+ headroom for the
llama.cpp backends — qwen3.5-9b + gemma3-12b-it + gemma3n-e4b-it all
happily fit alongside it, and gemma3-27b-it QAT still fits when
whisper is the only other resident model.

## When large-v3 would still be preferred

Cases where keeping the full 32-layer decoder matters:

- **Very low-resource languages.** The distillation training set is
  biased toward the high-resource languages, so the turbo WER gap
  widens on languages with thin representation in the pretraining
  corpus.
- **Heavy code-switching or noisy audio.** Deeper decoders do better
  at disambiguation when the acoustic signal is weak or the lexical
  context flips languages mid-utterance.
- **Transcription of formal content where every word matters** (legal,
  medical) and latency is not the bottleneck.

None of these describe this project. If any of them ever do, swap
`hf_pattern` + `model_path` in
[config/models.yaml](../../config/models.yaml) to `ggml-large-v3.bin`
and re-run `python -m src.install --fix` — the registry slot, port,
and launcher name don't change.

## Alternative: `q5_0` quant

`ggml-large-v3-turbo-q5_0.bin` (~547 MB, ~1 GB VRAM) is a 5-bit
quantised turbo. On a smaller GPU (8 GB) or a machine where VRAM
budget is tight, it's the natural choice: the accuracy drop vs f16
turbo is small, and it fits alongside a 9B-class llama model with
headroom to spare.

We're not using it here because the 16 GB reference GPU has the room
for f16, and on-device ASR latency is already comfortably realtime —
no reason to take the quality hit.

## Accuracy numbers, briefly

Rather than invent per-language WER numbers, the authoritative sources
are:

- **OpenAI turbo release notes** (Oct 2024) — positions turbo as
  "faster than large-v3 with comparable quality on well-resourced
  languages." See the announcement on the
  [openai/whisper GitHub discussion](https://github.com/openai/whisper/discussions/2363).
- **whisper.cpp README** — publishes a model comparison table
  including `large-v3-turbo` next to `large-v3`, `small`, etc., with
  per-size size/speed numbers for the ggml ports.
- **OpenAI Whisper paper** — [arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356)
  for the base-model WER tables across 99 languages; the turbo
  addendum is best read via the release notes above.

These are the sources to pin in any future benchmark write-up. Don't
hand-edit WER numbers into this repo — link out.

## Non-goals (explicit)

- **No model-selection CLI flag or runtime switch.** One weights file,
  one registry row. Switching is a YAML edit + reinstall, same as it
  was for `small`.
- **No parallel `whisper_large_v3` / `whisper_small` rows.** Only one
  whisper backend exists in the registry; only one occupies :8090.
  The mutex with `transcribe_voice` depends on that.
- **No streaming, no VAD.** Out of scope; the backend stays the
  OpenAI-compatible request/response shape that `TranscriptionClient`
  already targets.
