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
