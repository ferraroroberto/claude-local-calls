# Local LLM + ASR Efficient Frontier — Results

**Run date:** 2026-07-24
**Hardware:** RTX 5060 Ti 16 GB · Ryzen 7 7800X3D · 128 GB DDR5
**Workloads:** OpenClaw agentic (fast + deep lanes), transcript polishing, document processing, EN↔ES↔CA translation, **audio transcription EN/ES, audio translation ES → EN, transcript disfluency removal**. No coding.

---

## 0. Verdict

| role | incumbent | verdict | best alternative | gap | reason |
|------|-----------|---------|-------------------|-----|--------|
| `agentic_light` | `qwen35_4b` (qwen3.5-4b) | keep | Gemma 3 4B (Catalan niche) | — | No new 4B-class entrant since Feb 2026; Kimi K3, GLM-5.2, and Mistral's "fat but sparse" MoE are all the wrong size class entirely |
| `agentic_heavy` | `gemma4_26b` (gemma4-26b-a4b-it) | keep | Qwen 3.6 35B-A3B | tie (≤3%) | Tie persists; Kimi K3 (2.8T, Jul 16) and GLM-5.2 (744B) both confirmed hardware NO-GOs, Mistral's "fat but sparse" MoE is still early-access only — no open weights |
| `audio_transcribe` | `parakeet` (parakeet-tdt-0.6b-v3, Mac ANE) + whisper fallback | runtime_upgrade | parakeet + FluidAudio `CustomVocabularyContext` rescorer | — | Incumbent **changed out-of-band on 2026-07-22** (#350, voice-transcriber#149) — a latency/placement swap, not an accuracy one. First-party testing shows it drops the "Claude Code" wake phrase and mangles jargon; FluidAudio ships an unintegrated custom-vocabulary rescorer that is the one concrete, already-identified fix |
| `audio_translate` | `whisper_translate` (whisper-medium) | watch | two-stage: Turbo → Gemma 4 26B MoE | — | Unchanged; two-stage stays the default, this slot stays a lazy fallback |

**Diff vs previous run (2026-07-12):** the headline change isn't a new external model — it's that `audio_transcribe`'s **incumbent itself changed between runs**: on 2026-07-22 (#350, voice-transcriber#149), Parakeet TDT v3 on the Mac Mini's Apple Neural Engine became the role's primary backend with whisper-large-v3-turbo as automatic failover, a fleet-placement/latency decision (#343's cross-host benchmark) made independently of this skill's own recommendation cycle. This run's job is catching the ledger up to that reality: the verdict flips from the twice-carried, now-moot faster-whisper-CT2 runtime upgrade (disproven locally, #274) to a **new** runtime_upgrade target — wiring FluidAudio's built-in `CustomVocabularyContext` rescorer into the Mac worker to close the wake-phrase/jargon regression the speed swap knowingly accepted. `agentic_light`, `agentic_heavy`, and `audio_translate` are unchanged in substance; no externally-released model displaces any incumbent in this 12-day window.

*Why this is the core artifact:* everything below exists to justify these six columns. If you read nothing else, this table plus the diff line is the run.

---

## 1. Objective

The "efficient frontier" of local LLMs is the set of models where, for a given level of quality, no other model is faster (or, for a given speed, no other model is more accurate). Everything off the frontier is **dominated** — a strictly better choice exists on at least one axis without giving up the other.

The frontier is **always hardware- and workload-specific**: a 70B that dominates on a 5090 falls off the 5060 Ti's frontier into CPU-offload territory, and a coding-specialist that wins SWE-bench is irrelevant here because coding carries 0% weight. This report identifies the frontier for *this* box and *these* workloads, as of July 2026.

**What changed since 2026-07-12:** on the text side, essentially nothing displaces the standing picks in a 12-day window — **Kimi K3** (Moonshot, announced Jul 16, 2.8T total / 16 active per token across 896 experts, weights shipping Jul 27) is the loudest release of the period but needs "64 or more accelerators" per Moonshot's own guidance; even the most optimistic consumer estimate puts it at sub-1 tok/s on this box, so it's out on size exactly like GLM-5.2. **GLM-5.2** stays a confirmed NO-GO (>1 TB VRAM in BF16, `docs/glm-5.2-evaluation.md`, #141). **Mistral's "fat but sparse" MoE** (teased Jul 8 by CEO Arthur Mensch, larger total than the 675B/41B-active Mistral Large 3) remains in partner early access — no open weights, still a watch item. **Qwen 3.7** is confirmed API-only (Max/Plus tiers); no open-weight 3.7 variant has shipped, so Qwen 3.6 (April) remains the newest locally-runnable Qwen.

On the audio side, the real news landed *between* runs, not during the survey window: the fleet's own `audio_transcribe` role swapped its primary backend from whisper to Parakeet TDT v3 on 2026-07-22 (#350), and the user's daily-driver dictation client started routing through the hub the same day (voice-transcriber#149) rather than hitting whisper directly. Details in §7.

---

## 2. System & workloads

| | |
|---|---|
| **GPU** | NVIDIA RTX 5060 Ti, 16 GB VRAM, 448 GB/s memory bandwidth (Blackwell, FP4-capable) |
| **CPU** | AMD Ryzen 7 7800X3D, 8c/16t, 96 MB L3 — the large cache makes CPU inference unusually viable |
| **RAM** | 128 GB DDR5 (running at 3600 MT/s on a 6400 kit) — huge offload headroom |
| **Storage** | 2 TB NVMe (WD_BLACK SN850X) for hot model weights |
| **OS** | Windows 11 Pro |
| **Runtimes** | llama.cpp / GGUF (primary, via this repo's launchers); Ollama, LM Studio, vLLM-under-WSL2 as references |

Composite quality-score weights (fixed by the skill brief):

- **35%** agentic / function calling — BFCL v3/v4, τ-bench, IFEval
- **25%** instruction following & writing polish — Arena-Hard, RewardBench
- **25%** multilingual quality — FLORES-200 EN↔ES, EN↔CA where measured (Catalan coverage is uneven)
- **15%** long-context document handling — needle-in-haystack, RULER

Audio workloads (transcription EN/ES, audio translation ES→EN, disfluency removal) are evaluated separately in §7. Coding benchmarks carry **0% weight**.

---

## 3. Methodology

1. Read `docs/frontier/runs/LATEST` (2026-07-12), that run's `report.md` + `frontier.json`, and `docs/frontier/local-findings.md` (#277) — the faster-whisper CTranslate2 disproof from 2026-07-12 is carried forward as unresolved.
2. Re-read `config/models.yaml` → `roles:` for the current incumbents, which surfaced the out-of-band change: `audio_transcribe` now points at `parakeet` (fallback `[whisper]`), not `whisper` — a swap made 2026-07-22 (#350) that predates this run but postdates the last one.
3. Read the fleet's own first-party evidence for that swap — `docs/parakeet-asr-evaluation.md` (the #123/#138 spikes) and `docs/voice-benchmark.md` (#343, the cross-host placement benchmark) — rather than relying on published leaderboard numbers alone, since the project's own jargon-heavy dictation domain diverges sharply from generic ASR benchmarks.
4. Surveyed the external landscape via web search for the 2026-07-12 → 2026-07-24 window: Kimi K3, GLM-5.2 follow-up coverage, Mistral's "fat but sparse" MoE status, Qwen 3.7 open-weight status, and published Parakeet TDT v3 vs Whisper Large v3 Spanish WER (MLS/FLEURS).
5. Computed VRAM with the standing rule of thumb **Q4_K_M ≈ 4.5 bits/param** plus KV-cache; carried figures from the previous run where nothing changed, flagged as unchanged.
6. Applied the honesty rules: date-stamped claims, ≤3% composite = tie (the Gemma 4 26B / Qwen 3.6 35B-A3B tie stands), licenses surfaced, and the local-findings override applied to the faster-whisper CT2 entry (still `watch`, not re-proposed).

---

## 4. How to read the chart

- **X axis** — estimated single-stream tokens/second on the 5060 Ti at the recommended quant.
- **Y axis** — composite quality score for *these* workloads (0–100, normalized).
- **Bubble size** — VRAM at recommended quant. **Color** — tier (A fast / B balanced / C quality).
- **Filled border** — on the Pareto frontier. **Hollow** — dominated.
- **Toggle** — show only models that fit fully in 16 GB VRAM, or include CPU-offload models.

### Worked memory example (so the math isn't a black box)

For **Kimi K3 at its smallest plausible quant** — this run's cautionary tale:

```
2.8T total params, 16 active of 896 experts per token
even an aggressive 2-bit quant ≈ ~700 GB on disk for the full expert set
this box: 16 GB VRAM + 128 GB RAM ≈ 144 GB total
shortfall ≈ ~550 GB → doesn't load at any quality worth using, let alone run interactively
```

Compare **Gemma 4 26B MoE** (the incumbent): ~14 GB at native W4, 4B active params during decode → bandwidth pressure set by 4B, not 26B → 99 t/s fully GPU-resident. **MoE with a small active set remains the single biggest reason the frontier looks the way it does on consumer GPUs** — and "total parameters must physically fit somewhere" remains the hard gate that keeps the trillion-plus wave off this box entirely, no matter how sparse the routing.

---

## 5. Results — shortlist by tier

### Tier A — Fast lane (OpenClaw routing, classification, simple tool calls)

| model | params | quant | VRAM | tok/s | quality | ctx | license | on frontier |
|-------|--------|-------|------|-------|---------|-----|---------|-------------|
| ★ Qwen 3.5 4B *(incumbent)* | 4B hybrid MoE | Q4_K_M | ~3 GB | ~110 | 65 | 262k | Apache 2.0 | yes |
| ☆ Gemma 3 4B | 4B dense | Q4_K_M | ~3 GB | ~100 | 60 | 128k | Gemma | yes |
| Phi-4 Mini | 3.8B dense | Q4_K_M | ~2.5 GB | ~120 | 49 | 16k | MIT | yes (speed end) |
| Granite 4.1 8B | 8B dense | Q4_K_M | ~5 GB | ~60 est | 64 est | 128k | Apache 2.0 | no |
| Llama 3.2 3B | 3B dense | Q4_K_M | ~2 GB | ~120 | 43 | 128k | Llama 3.2 | no |

No change this run. The incumbent keeps the tier — nothing in the July window ships a new 4B-class open model.

### Tier B — Balanced (the workhorse) — **TIED, unchanged**

| model | params | quant | VRAM | tok/s | quality | ctx | license | on frontier |
|-------|--------|-------|------|-------|---------|-----|---------|-------------|
| ★★ Gemma 4 26B MoE *(incumbent)* | 26B / 4B active | native W4 | ~14 GB | 99 | 83 | 256k | Gemma | yes |
| ★★ Qwen 3.6 35B-A3B | 35B / 3B active | Q4_K_M | ~13.5 GB | 98 | 84 | 262k | Apache 2.0 | yes |
| ☆ GPT-OSS 20B | 21B / 3.6B active | MXFP4 | ~12 GB | ~100 | 72 | 131k | Apache 2.0 | yes |
| Gemma 4 12B Unified | 12B dense | Q4_K_M | ~7 GB | ~45 est | 76 est | 256k | Apache 2.0 (verify) | no |
| Ministral 3 14B | 14B dense | Q4_K_M | ~8.5 GB | ~35 est | 70 est | 256k | Apache 2.0 | no |
| Mistral Small 3.2 | ~22B dense | Q4_K_M | ~13 GB | ~30 | 74 | 128k | Apache 2.0 | no |

The tie at the top persists. GPT-OSS 20B is worth one added data point this run: its MMMLU Spanish score (79.7 medium-reasoning) confirms decent — not leading — ES coverage, still short of what would displace either tied leader; Catalan remains unmeasured for it.

### Tier C — Quality (slow, CPU-offload, batch / non-interactive)

| model | params | quant | VRAM | tok/s | quality | ctx | license | on frontier |
|-------|--------|-------|------|-------|---------|-----|---------|-------------|
| ★ Qwen3 32B dense | 32B | Q4_K_M | ~19.5 GB (spill) | ~11 | 84 | 128k | Apache 2.0 | yes |
| Llama 3.3 70B | 70B | Q4_K_M | ~40 GB (offload) | ~4 | 82 | 128k | Llama 3.3 | no |
| Mistral Medium 3.5 | 128B dense | Q4_K_M | ~75 GB (offload) | ~2 | 86 | 256k | Modified MIT | no |

No change.

### Models considered and dropped this run

- **Kimi K3 (2.8T total / 16B active, announced Jul 16, weights Jul 27)** — Moonshot's own guidance calls for "64 or more accelerators"; even a 2-bit dynamic quant is roughly ~700 GB, an order of magnitude past this box's ~144 GB combined VRAM+RAM. Out on size, decisively — see the worked example in §4.
- **GLM-5.2 (744B / 40B active)** — standing NO-GO carried from the last two runs (`docs/glm-5.2-evaluation.md`, #141); >1 TB VRAM in BF16, no quant fits.
- **Mistral's "fat but sparse" MoE (teased Jul 8, early access)** — total parameter count reported larger than Mistral Large 3 (675B/41B active); weights not public. Watch next run.
- **Qwen 3.7 Max / Plus (API, May 20)** — confirmed still closed-weights as of this run; no GGUF, no Ollama, no HF weights for any 3.7 variant. Qwen 3.6 remains the newest open Qwen.
- Standing drops carried from July: Qwen 3.6 27B dense, Gemma 3 27B, Qwen3.5-35B-A3B, Mistral Small 3.2, Llama 3.2 3B, Phi-4 Mini (for ES/CA), Qwen3 8B/9B class, MiniMax M3 (still academic at 2-bit).

---

## 6. Concurrency plan

Unchanged from the previous run — the four recipes still describe the practical envelope:

1. **Two lanes (default):** Qwen 3.5 4B (GPU ~3 GB) + Gemma 4 26B MoE (GPU ~14 GB). Both near-peak; ~17 GB with graceful shared-memory spill.
2. **Qwen 3.6 stack (all-Apache):** Qwen 3.5 4B + Qwen 3.6 35B-A3B (~13.5 GB). Speed parity; license clarity.
3. **Quality batch:** Qwen 3.5 4B + Qwen3 32B dense (~3.5 GB CPU spill, ~10 t/s) for overnight reprocessing.
4. **Three concurrent:** Qwen 3.5 4B + GPT-OSS 20B (GPU) + Gemma 3 4B (CPU, ~10 t/s on the 7800X3D) as a Catalan specialist.

One addition this run: the tower's GPU budget for these recipes is now **exclusively agentic** — the 2026-07-22 fleet placement work (#343) moved GPU-hungry TTS (orpheus) and low-frequency STT (whisper-vanilla/translate) off the tower onto the gaming satellite, and moved accurate whisper-turbo dictation to gaming as well (agentic-heavy + agentic-light already fill the 16 GB card, leaving no headroom for a co-resident GPU whisper). None of this changes the LLM concurrency math above — it's audio placement, not agentic placement — but it's why whisper-turbo no longer runs on the same box as these recipes.

---

## 7. Audio (ASR) annex — workloads F, G, H

### 7.1 The landscape in July 2026 — and the fleet's own out-of-band move

Two things happened in this window, and only one of them was an external release. Externally: no new comparative Spanish WER landed on the public leaderboard cut for Qwen3-ASR or Parakeet v3 beyond what was already known, though this run did surface published MLS/FLEURS numbers for Parakeet TDT v3 vs Whisper Large v3 (§7.2) that weren't cited in the previous report.

Internally — and this is the actual news — **the fleet's own `audio_transcribe` role changed backends on 2026-07-22** (#350, activating the #348 failover mechanism), independent of this skill's cycle: `parakeet` (TDT v3 on the Mac Mini's Apple Neural Engine) became the primary, with `whisper` as automatic failover. The same day, voice-transcriber#149 routed the user's actual daily-driver dictation client through the hub for the first time (it previously hit whisper's `:8090` directly), so this isn't just a benchmark-table change — it's live on real dictation traffic now. The motivation was **fleet placement and latency** (#343's cross-host benchmark: parakeet-on-ANE measured 65.8× RTFx vs the tower's whisper at 40× and gaming's whisper at 19.3×), not an accuracy win — the project's own first-party testing (`docs/parakeet-asr-evaluation.md`) had twice concluded NO-GO on accuracy grounds before this placement-driven flip.

### 7.2 ASR candidate comparison (EN + ES)

| Variant | Params | VRAM | RTFx (measured/est.) | EN | ES | Translates → EN? | Notes |
|---------|--------|------|----------------------|----|----|-------------------|-------|
| **★ Parakeet TDT v3** (Mac ANE, current primary) | 0.6B | ~2 GB (CoreML) | 65.8× measured (ANE), sub-second even on 108 s clips | ✅ | ✅ (25 langs) | ❌ | New incumbent as of 2026-07-22. Fastest STT in the fleet by far. Published MLS WER 4.39% (vs Whisper's 4.89%) and FLEURS 3.41% (vs 4.32%) for generic Spanish — but the project's own jargon-heavy dictation domain measures **worse**, not better, than whisper (see 7.2.1). |
| **◆ Whisper Large v3 Turbo** (automatic failover) | 809M | ~1.6 GB | 40× tower / 19.3× gaming (measured, boosted) | ✅ | ✅ | ❌ | Was the sole default through 2026-07-12; now the failover leg, still the accuracy leader on this domain thanks to the `--carry-initial-prompt` boosting glossary (#91). |
| faster-whisper Turbo (CT2) | 809M (CT2) | ~1.0 GB INT8 | measured 1.0× vs whisper.cpp — **disproven** | ✅ | ✅ | ❌ | Carried `watch` per `docs/frontier/local-findings.md` (#277); applies to the failover leg's engine now, not the primary. |
| Whisper Large v3 (faster-whisper) | 1.55B | ~2 GB | 30–50× | ✅✅ | ✅✅ | ✅ | Workload-G single-model fallback, unchanged. |
| Granite Speech 3.3 8B | 8B | ~5 GB | 15–30× | ✅✅✅ | ✅✅ | ✅ (X↔EN) | Accuracy tier; only if the failover's own errors start to matter. |
| Qwen3-ASR 1.7B | 1.7B | ~1.5 GB | TBD | ✅ | ✅ | ❌ | Still `watch` — technical report out, but no comparative dictation-domain WER exists (same gap that made Parakeet's generic leaderboard numbers misleading here). |

#### 7.2.1 Published benchmarks vs this project's own domain — why they disagree

Generic ASR leaderboards (MLS, FLEURS) say Parakeet TDT v3 modestly **beats** Whisper Large v3 on Spanish WER. The fleet's own first-party testing on its actual jargon-heavy, code-switched dictation domain (`docs/parakeet-asr-evaluation.md`, `docs/voice-benchmark.md`) says the opposite: Parakeet's WER on this domain runs **2–4× worse** than whisper's, and it structurally lacks a recognition-boosting lever (no `initial_prompt`/`hotwords` equivalent) to close the gap the way whisper does with #91's glossary boost. Two concrete, reproduced failures: it drops the "Claude Code" wake phrase entirely (0/2 across two independent test rounds, #123 and #138/#343), and it mangles "YOLO" → "yellow" every time. This is exactly the honesty-rules case the skill brief calls for — a published-benchmark verdict that doesn't survive contact with the actual workload — except this time the divergence was discovered by the fleet's own prior spikes (#123/#138), not by this run.

**FluidAudio ships an answer that isn't wired in.** The Mac worker's underlying `FluidAudio` library exposes `CustomVocabularyContext`/`CustomVocabularyTerm` — a post-processing CTC rescorer that re-scores transcript words against per-frame log-probabilities and swaps in a boosted term when it has stronger acoustic evidence. This is a plausible, already-identified fix for exactly the wake-phrase and jargon failures above, confirmed to exist in the source (`VocabularyRescorer`, `CtcKeywordSpotter`) but never integrated — it requires loading a second CTC model and the token-timing rescoring pipeline, work that was explicitly deferred in the #138 spike. No open GitHub issue currently tracks this integration (checked: search for "vocabulary"/"FluidAudio" across the repo's issue history turns up only the evaluation spikes, not a build-it ticket) — that's this run's actionable item.

### 7.3 Workload F — transcribe EN/ES

**Verdict: `runtime_upgrade`.** Keep the model (Parakeet TDT v3, already the placement-driven choice), close the gap that made it a latency-over-accuracy trade in the first place: wire FluidAudio's `CustomVocabularyContext` into `src/parakeet_server.py`'s worker so the boosting glossary (#90/#91) that whisper already benefits from also protects the primary path's wake phrase and jargon. The whisper failover leg's own pending `faster-whisper` (CTranslate2) engine swap stays `watch` per the local-findings override (#277) — it was empirically disproven (1.0× speedup, not 2×) and now only matters for the failover leg, not the live primary.

### 7.4 Workload G — ES audio → English

**Verdict unchanged: two-stage default.** faster-whisper Turbo transcribes ES, Gemma 4 26B MoE translates + polishes + de-disfluences in one call (~15 GB total). Single-model faster-whisper Large v3 `task=translate` stays the fallback when the LLM slot is busy — implemented today by the `whisper_translate` role slot (whisper-medium, lazy CPU), hence its `watch` verdict rather than retirement. The 2026-07-22 fleet placement work didn't touch this role.

### 7.5 Workload H — disfluency / filler removal

**Verdict unchanged: folded into the LLM polishing pass.** Specialized disfluency models remain research-grade with sparse Spanish coverage; the tier-B LLM already does this work in the polish prompt (the canonical prompt lives in `frontier.json` → `disfluency_verdict.prompt`).

### 7.6 Concurrency footprint

Materially changed since the last run, not by model choice but by **host**: whisper-turbo (accurate dictation) and whisper-vanilla/translate now run on the gaming satellite rather than the tower, and orpheus (expressive TTS) joined it — freeing the tower's GPU for agentic_heavy + agentic_light exclusively (#343). Parakeet runs on the Mac Mini's ANE, its own dedicated accelerator, so it adds zero contention with either box. Piper (fast TTS) stays on the tower CPU — zero GPU cost either way.

### 7.7 Dropped

Carried from prior runs: Canary-Qwen (EN-only), Parakeet v1/v2 (superseded by v3), Distil-Whisper (EN-only), Phi-4-Multimodal (footprint/tooling), Seamless M4T v2 (heavier, weaker tooling than two-stage), specialized disfluency models (worse ES coverage than the LLM pass).

---

## 8. Progression

Cumulative run-over-run history — one row per role per run; this table only grows.

| run date | role | incumbent | verdict | best alternative |
|----------|------|-----------|---------|-------------------|
| 2026-05-10 | agentic_light | gemma4_e4b (gemma4-e4b-it) | upgrade | Qwen 3.5 4B |
| 2026-05-10 | agentic_heavy | gemma4_26b (gemma4-26b-a4b-it) | keep | Qwen 3.6 35B-A3B (tie) |
| 2026-05-10 | audio_transcribe | whisper (large-v3-turbo) | runtime_upgrade | faster-whisper Turbo |
| 2026-05-10 | audio_translate | whisper_translate (whisper-medium) | watch | two-stage Turbo → Gemma 4 26B |
| 2026-07-12 | agentic_light | qwen35_4b (qwen3.5-4b) | keep | Gemma 3 4B (Catalan niche) |
| 2026-07-12 | agentic_heavy | gemma4_26b (gemma4-26b-a4b-it) | keep | Qwen 3.6 35B-A3B (tie) |
| 2026-07-12 | audio_transcribe | whisper (large-v3-turbo) | runtime_upgrade | faster-whisper Turbo |
| 2026-07-12 | audio_translate | whisper_translate (whisper-medium) | watch | two-stage Turbo → Gemma 4 26B |
| 2026-07-24 | agentic_light | qwen35_4b (qwen3.5-4b) | keep | Gemma 3 4B (Catalan niche) |
| 2026-07-24 | agentic_heavy | gemma4_26b (gemma4-26b-a4b-it) | keep | Qwen 3.6 35B-A3B (tie) |
| 2026-07-24 | audio_transcribe | parakeet (parakeet-tdt-0.6b-v3) + whisper fallback | runtime_upgrade | parakeet + FluidAudio CustomVocabularyContext |
| 2026-07-24 | audio_translate | whisper_translate (whisper-medium) | watch | two-stage Turbo → Gemma 4 26B |

Reading the progression: the `audio_transcribe` row is the one that actually moved substance between runs, but not through this skill — the incumbent itself changed underneath it (#350) between the 2026-07-12 and 2026-07-24 runs. The `runtime_upgrade` verdict label stays constant across all three runs, but what it points at has changed twice now: first the (disproven) whisper→faster-whisper engine swap, now the parakeet→parakeet-with-custom-vocabulary integration. `agentic_light`/`agentic_heavy`/`audio_translate` have been stable since 2026-05-10.

---

## 9. Open questions / uncertainty

- **FluidAudio `CustomVocabularyContext` integration** — the concrete, already-identified fix for the wake-phrase/jargon regression accepted by the 2026-07-22 parakeet switch. Not validated even as a spike; requires a second CTC model and the token-timing rescoring pipeline. This run's actionable item (§7.2.1, §0).
- **The wake-phrase drop is now live on real dictation, not just a benchmark.** voice-transcriber#149 (closed 2026-07-22) routes the user's actual daily-driver dictation through the hub, so parakeet's documented "Claude Code" drop is a real-world risk now, not a spike finding sitting in a doc. Worth watching for actual missed-trigger reports.
- **Published vs domain-specific ASR benchmarks keep diverging for this workload.** Generic Spanish WER leaderboards say Parakeet ≥ Whisper; this project's own jargon-heavy, code-switched dictation domain says the reverse by 2–4×. Treat any future "leaderboard says X" claim about ASR on this fleet as provisional until checked against `docs/parakeet-asr-evaluation.md`-style domain testing.
- **Gemma 4 12B Unified license** — still says Apache 2.0 per launch coverage; still unverified on the model card.
- **Catalan on Qwen 3.6 35B-A3B** — still the only meaningful delta vs Gemma 4 26B; still needs a local smoke test.
- **MiniMax M3 at UD-Q2** — still technically loadable (~138 GB), still academic given single-digit t/s and 2-bit quality loss.
- **Mistral's "fat but sparse" MoE** — in partner early access as of Jul 8; watch for open weights next run.
- **Kimi K3 weights ship 2026-07-27** — three days after this run. Confirmed out on size regardless (§4, §5), but flag the date in case any surprise smaller variant follows.

---

## 10. Current decisions (live, edited by `/swap-model`)

The decisions below mirror `config/models.yaml` → `roles:` at the time
this section was last updated. `/swap-model` rewrites both this section
and the yaml together, so the two stay in sync — **except `audio_transcribe`
below, which was changed directly via #350, not `/swap-model`**; this run's
refresh is what catches this section up to that reality.

| Role | Model | Decided | Why |
|---|---|---|---|
| **agentic_light** | `qwen35_4b` (qwen3.5-4b) | 2026-05-10 | Upgraded from gemma4_e4b via `/swap-model`. Tier A top pick — hybrid Gated DeltaNet + sparse MoE on a 4B base, Q4_K_M ~3 GB, 262k native ctx, 201 languages, Apache 2.0. gemma4_e4b retained in `enabled:` for ad-hoc fallback. |
| **agentic_heavy** | `gemma4_26b` (gemma4-26b-a4b-it) | 2026-05-10 | Tier B top pick. 99 t/s, 256k ctx, strong multilingual including Catalan. Tied with Qwen 3.6 35B-A3B (Apache 2.0) — Gemma stays default on Catalan track record. |
| **audio_transcribe** | `parakeet` (parakeet-tdt-0.6b-v3, Mac ANE) with `fallback: [whisper]` | 2026-07-22 | Changed directly via #350 (fleet placement/latency, #343 benchmark), not `/swap-model`. A speed-over-accuracy trade with known regressions (dropped wake phrase, jargon mangling) — see §7. FluidAudio custom-vocabulary integration is the identified next step to close the gap. |
| **audio_translate** | `whisper_translate` (whisper-medium, lazy CPU) | 2026-05-10 | Strict frontier reading recommends `watch` — the two-stage path (Turbo → Gemma 4 26B) is the default, leaving this slot as a fallback only. Keep defined and lazy-loaded; no active maintenance. |

---

*Generated by the `/frontier-refresh` skill (`.claude/skills/frontier-refresh/SKILL.md`), which owns the research brief and this report's output contract. This is the July 24, 2026 snapshot.*
