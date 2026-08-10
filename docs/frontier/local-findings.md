# Frontier local findings — candidates tested on this box

Deterministic run-over-run memory for `/frontier-refresh` (#277). Published
numbers said one thing, a local measurement said another — this file is where
that learning survives, so the same candidate is never re-proposed and
re-disproven in a loop.

**Contract with the skill** (`.claude/skills/frontier-refresh/SKILL.md`):

- The skill reads this file in step 1, before computing verdicts.
- A role whose best alternative matches an **unresolved** entry here gets
  verdict `watch` with reason `disproven locally <date> (#N)` — never
  `upgrade` / `runtime_upgrade` — unless the entry's **re-open trigger** is
  demonstrably met (cite the evidence in the report if so).
- `watch` is not actionable, so step 8 files no issue for it: no repeat work.
- Whoever disproves (or re-proves) a candidate locally appends/updates the
  entry **in the same PR** as the disproof — same anti-staleness contract as
  `.fleet.toml`.

Entries are append-only; when a re-open trigger fires and the candidate is
re-tested, update its **Status** line instead of deleting history.

---

## 2026-07-12 — faster-whisper (CTranslate2) for `audio_transcribe` — DISPROVEN

- **Candidate:** faster-whisper 1.2.1 / CTranslate2 4.8.1, `large-v3-turbo`
  CT2 weights, int8_float16 and float16, RTX 5060 Ti.
- **Verdict it disproves:** `runtime_upgrade` (carried 2026-05-10 and
  2026-07-12, report §7.3 — "~2× RTFx, lower VRAM, same quality").
- **Measured:** speedup is **1.0×**, not 2× — aggregate RTFx 33.4 vs 33.6
  (int8) and 32.6 vs 33.3 (fp16) over 556 s of real dictation audio;
  whisper.cpp v1.8.6 cuBLAS is already ~33× real-time on this GPU. WER
  parity-to-worse (3.9 % vs 4.4 % int8; 3.7 % vs 5.4 % fp16). **Drops the
  leading "Claude Code" wake phrase 0/2 vs whisper.cpp's 2/2** across an
  18-attempt decode sweep (beam/greedy, patience, VAD, hotwords,
  initial_prompt, thresholds × both compute types); audio-head loss ruled
  out. Same decoder-side failure that kept Parakeet off the default role
  (#138).
- **Record:** [#274 closing comment](https://github.com/ferraroroberto/local-llm-hub/issues/274#issuecomment-4949098008)
  (full method + per-clip table); corpus: `.scratch/parakeet-bench/` clips +
  refs, harness `.scratch/fw-bench/`.
- **Re-open trigger:** a faster-whisper/CTranslate2 release that demonstrably
  fixes leading-phrase recall, or hardware where whisper.cpp no longer holds
  speed parity — then re-measure on the same corpus before any verdict
  upgrade.
- **Status:** unresolved (verdict stays `watch`).

---

## 2026-07-24 — FluidAudio `CustomVocabularyContext` rescorer for `audio_transcribe` — DISPROVEN

- **Candidate:** FluidAudio 0.15.4's CTC custom-vocabulary pipeline
  (`CustomVocabularyContext.loadWithCtcTokens` → `CtcKeywordSpotter.spotKeywordsWithLogProbs`
  → `VocabularyRescorer.ctcTokenRescore`) wired into the Mac Parakeet worker
  — the mechanism the 2026-07-24 frontier run (`report.md` §0/§7.2.1/§9)
  recommended as a `runtime_upgrade` to close the "Claude Code"/"YOLO"
  wake-phrase and jargon regression accepted by the 2026-07-22
  parakeet-primary switch (#350).
- **Verdict it disproves:** `runtime_upgrade` (proposed 2026-07-24, report
  §0 — "parakeet + FluidAudio CustomVocabularyContext").
- **Measured:** built end-to-end and tested against the #138/#343 jargon
  clips (issue #401). A post-hoc rescorer can only **swap** an existing
  transcript word for an acoustically/string-similar vocabulary term — it
  cannot **insert** a phrase the TDT decoder dropped entirely. "Claude Code"
  heard as "Yes"/"Yeah" is structurally unrecoverable (string similarity
  ≈ 0), and "YOLO" (4 chars) is blocked by the short-word ≥0.80 similarity
  guard against "yellow". A four-setting threshold sweep shows a hard
  precision/recall wall: every false-positive-free setting is a no-op on
  both targets, while every setting that recovers a target simultaneously
  corrupts correct words (`mention`→`Notion`, `tab`→`Tailscale`,
  `left, right`→`Playwright`, …). For daily-driver dictation, silently
  corrupting correct words is worse than a known-absent wake phrase, so no
  setting is shippable.
- **Record:** [issue #401](https://github.com/ferraroroberto/local-llm-hub/issues/401)
  (closed not-planned 2026-07-24, full before/after sweep + root-cause
  analysis in the closing comment); merged into
  `docs/parakeet-asr-evaluation.md` → "Update 2026-07-24: custom-vocabulary
  rescorer wired and DISPROVEN (#401)" via #407 (commit 160a280). No
  `audio_transcribe` role change resulted — whisper-turbo stays the
  jargon-safe accuracy leader (see `local-findings.md`'s own contract note:
  this entry was not appended in #407 itself; backfilled here by the
  2026-08-06 frontier run, which found the gap while reading this file per
  its step 1).
- **Re-open trigger:** a FluidAudio release that adds decode-time TDT
  vocabulary *biasing* (not post-hoc CTC rescoring — i.e. a hook that can
  insert a dropped phrase, not just swap an emitted one), or an alternative
  Parakeet checkpoint/architecture exposing such a hook — then re-measure
  on the same #138/#343 corpus before any verdict upgrade.
- **Status:** unresolved (verdict stays `watch`).

---

## 2026-08-10 — Nemotron 3 Nano 4B for `agentic_light` — DISPROVEN (quality)

- **Candidate:** NVIDIA Nemotron 3 Nano 4B, `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF`
  → `NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf` (2.84 GB, the only quant NVIDIA
  publishes). Hybrid Mamba-2 + MoE, 4 attention layers, 260 K native context,
  **NVIDIA Open Model License** (vs Qwen's Apache 2.0). Ran on `:8089`,
  `--jinja -ngl 99 -c 65536 --flash-attn on --reasoning-format none`.
- **Verdict it disproves:** the standing `keep` reason for `agentic_light`
  carried by four consecutive frontier runs — *"no new 4B-class entrant in
  this window"*. There **was** an entrant; it had simply never been sourced.
  Nemotron appears in zero of the 2026-05-10 / 07-12 / 07-24 / 08-06 runs.
- **Why it was tested here:** a spike on the sibling `local-llm-hub-lite`
  (work PC, Quadro P1000 4 GB) measured Nemotron as faster than Qwen with
  quality "comparable on classification, extraction, policy JSON, bullets,
  rewriting and restart steps". That prompt set covers roughly the
  agentic + polish half of this project's weighting and **no multilingual at
  all** — 25% of the composite, and precisely where the model fails.

- **Measured (RTX 5060 Ti, one arm at a time, 23 role-shaped prompts weighted
  to the skill's fixed 0.35 / 0.25 / 0.25 / 0.15, `max_tokens=1024`):**

  | arm | composite | agentic | polish | multiling | long ctx | total_s | ttft_s | tok/s | out_tok | trunc |
  |---|---|---|---|---|---|---|---|---|---|---|
  | `qwen35_4b_nothink` | **92.4** | 92.5 | 83.3 | **96.7** | 100.0 | 0.85 | 0.38 | 103.0 | 42 | 0/23 |
  | `nemotron4b` | 80.8 | 90.0 | 86.7 | 66.7 | 73.3 | 2.06 | 1.72 | 120.1 | 177 | 0/23 |
  | `nemotron4b_nothink` | 72.6 | 85.0 | 70.0 | 53.3 | 80.0 | **0.77** | 0.42 | **121.3** | 38 | 0/23 |
  | `qwen35_4b` (thinking) | n/a | — | — | — | — | 2.42 | 1.46 | 95.0 | 192 | **17/23** |

  **Nemotron is genuinely faster** — ~17% higher tok/s on both lanes and the
  fastest wall-clock of any arm — and its reasoning is far more economical
  (peak 564 tokens; Qwen's thinking arm exhausted 1024 on 17/23 prompts and
  needed ≥4096 to complete). **It still loses the role on quality by 11.6–19.8
  composite points, an order of magnitude outside the skill's ~3% tie band.**

- **Where it fails — Catalan, systematically.** Both Nemotron arms
  independently produced the same false friend, reading Catalan
  `si arriba el paquet` ("if the package *arrives*") as "if the package is
  **above**" — Spanish `arriba` bleeding into Catalan — and both emitted
  non-Catalan words (`conflicts`, `stress`, `Favoritza`, `reducint`).
  `qwen35_4b_nothink` scored 5/5 on both Catalan items. Two further failures:
  `nemotron4b_nothink` answered an ops question with `rm your_module.py` plus a
  fabricated `fastapi build` command, and on the buried-directive long-context
  test both Nemotron arms obeyed the directive then stopped without answering,
  where Qwen obeyed *and* answered.

- **Sampling ruled out.** The primary run used vendor-recommended sampling
  (NVIDIA 1.0/1.0, Qwen 0.6/0.95). Because the result went against the
  candidate, a matched-temperature control (0.6/0.95) was run on the decisive
  multilingual category: Spanish improves (`nemotron4b_nothink` 53.3% → 70.0%;
  the `las pesos` gender error and a unit-vs-total price error both resolve)
  but **Catalan does not move** — the same false friend reproduces at both
  temperatures. Best Nemotron multilingual at matched temp is 70.0% vs 96.7%.
  The gap is a model property, not a sampling artifact.

- **Positive results worth keeping.** Thinking suppression works: llama.cpp's
  `chat_template_kwargs {"enable_thinking": false}` genuinely suppresses
  Nemotron's `<think>` block (49 → 3 tokens on the same prompt), delivered
  through the hub's existing `inject_extra` with **no code change** — so
  ggml-org/llama.cpp#20182 does not affect this template, and an
  `agentic_light_nothink` equivalent was never the blocker. Nemotron also tied
  Qwen on tool calling (4/4 correct selections, correct refusal when no tool
  matched) and beat it on `polish_summarise`.

- **Record:** [issue #486](https://github.com/ferraroroberto/local-llm-hub/issues/486).
  Harness committed as `scripts/bench_agentic.py` + `scripts/bench_agentic_prompts.json`
  (reusable for any future `agentic_*` swap; targets any hub by `--base-url`).
  Candidate rows were removed after the run per the latest-only policy; restore
  them from #486's PR diff to re-test.

- **Re-open trigger:** a Nemotron 3 Nano release (or a community fine-tune)
  with demonstrated Catalan competence, **or** a decision to drop multilingual
  from `agentic_light`'s job — the role's own `notes:` currently cover ES/CA
  work. Absent either, re-measure only the multilingual category first: it is
  the whole gap, and it is cheap to re-check with
  `--category multilingual`.

- **Status:** resolved — `agentic_light` stays `qwen35_4b`. Future runs should
  still **source** Nemotron as a candidate (the sourcing gap was real) but can
  cite this entry rather than re-measuring the full set.
