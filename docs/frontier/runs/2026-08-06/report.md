# Local LLM + ASR Efficient Frontier — Results

**Run date:** 2026-08-06
**Hardware:** RTX 5060 Ti 16 GB · Ryzen 7 7800X3D · 128 GB DDR5
**Workloads:** OpenClaw agentic (fast + deep lanes), transcript polishing, document processing, EN↔ES↔CA translation, **audio transcription EN/ES, audio translation ES → EN, transcript disfluency removal**. No coding.

---

## 0. Verdict

| role | incumbent | verdict | best alternative | gap | reason |
|------|-----------|---------|-------------------|-----|--------|
| `agentic_light` | `qwen35_4b` (qwen3.5-4b) | keep | Gemma 3 4B (Catalan niche) | — | No new 4B-class entrant in this window; a rumored Qwen 3.8-27B open-weight companion has not shipped |
| `agentic_heavy` | `gemma4_26b` (gemma4-26b-a4b-it) | keep | Qwen 3.6 35B-A3B | tie (≤3%) | Tie persists; DeepSeek V4 Flash (284B/13B active, MIT, shipped 2026-07-31) is a new entrant this window but its footprint and non-coding composite quality don't clear the bar — see §5 |
| `audio_transcribe` | `parakeet` (parakeet-tdt-0.6b-v3, Mac ANE) + whisper fallback | watch | — | — | Last run's identified fix (FluidAudio `CustomVocabularyContext`) was built and **disproven locally** on 2026-07-24 (#401) — structurally cannot insert a decoder-dropped phrase. No other candidate fix or alternative surfaced this window |
| `audio_translate` | `whisper_translate` (whisper-medium) | watch | two-stage: Turbo → Gemma 4 26B MoE | — | Unchanged; two-stage stays the default, this slot stays a lazy fallback |

**Diff vs previous run (2026-07-24):** the one role that actually moves is `audio_transcribe` — but backwards, not forwards. The previous run recommended wiring FluidAudio's `CustomVocabularyContext` rescorer into the Mac Parakeet worker to recover the "Claude Code" wake phrase and jargon terms dropped by the 2026-07-22 parakeet-primary switch. That work was done the same day the recommendation was published (issue #401) and **disproven**: a post-hoc CTC rescorer can only swap an emitted word for a similar vocabulary term, it cannot insert a phrase the TDT decoder dropped entirely, so no threshold setting both recovers the targets and avoids corrupting unrelated words. The verdict flips from `runtime_upgrade` to `watch`, and this run backfills that disproof into `docs/frontier/local-findings.md` (it was never added in #407, the closing PR — a process gap this run closes). `agentic_light`, `agentic_heavy`, and `audio_translate` are unchanged in substance; the one new external entrant surfaced this window, DeepSeek V4 Flash, doesn't clear the bar for `agentic_heavy` (§5).

*Why this is the core artifact:* everything below exists to justify these six columns. If you read nothing else, this table plus the diff line is the run.

---

## 1. Objective

The "efficient frontier" of local LLMs is the set of models where, for a given level of quality, no other model is faster (or, for a given speed, no other model is more accurate). Everything off the frontier is **dominated** — a strictly better choice exists on at least one axis without giving up the other.

The frontier is **always hardware- and workload-specific**: a 70B that dominates on a 5090 falls off the 5060 Ti's frontier into CPU-offload territory, and a coding-specialist that wins SWE-bench is irrelevant here because coding carries 0% weight. This report identifies the frontier for *this* box and *these* workloads, as of August 2026.

**What changed since 2026-07-24:** on the text side, the loudest release of the window is **DeepSeek V4 Flash 0731** (284B total / 13B active, MoE, MIT license), which shipped open weights 2026-07-31 — three days inside this run's window. It's genuinely new and genuinely a MoE with a small active-parameter count, the exact shape that makes Gemma 4 26B MoE and Qwen 3.6 35B-A3B win on this hardware. It does not clear the bar here: at Q4_K_M its ~284B total parameters need ≈160 GB just for weights (see the worked example in §4), over this box's ~144 GB combined VRAM+RAM budget, so a usable local quant means IQ3-class (~117 GB, real quality loss) with no first-party evidence of composite quality on this stack's actual weighting (agentic/writing/multilingual — DeepSeek's historical strength is math/coding, which carries 0% weight here). Estimated decode speed at that quant (~11 t/s, RAM-bandwidth-bound on the 13B active set) lands roughly at Qwen3 32B's speed with a worse quant and an unverified quality edge — not a demonstrated win, so it stays a flagged watch item rather than a recommendation (honesty rules: don't recommend what you can't justify against the alternative). **Kimi K3**'s open weights shipped on schedule (2026-07-27, MXFP4, ~594 GB) — confirms, rather than changes, last run's decisive size-based NO-GO. **GLM-5.2** (744B/40B active, MIT) is unchanged and still a standing NO-GO; a **GLM-5.5** flagship is rumored for August per a JPMorgan research note but has no model card, benchmark, or endpoint as of this run — watch next cycle. **Mistral's "fat but sparse" MoE** remains in partner early access with no parameter count, benchmark, or license disclosed — unchanged watch. **Qwen 3.7** stays confirmed API-only; unconfirmed chatter about a further "Qwen 3.8" (with a rumored smaller 27B open-weight companion) exists but nothing has shipped — not actionable, flagged for next run.

On the audio side, the real news is the disproof covered in §0/§7: the fix this skill recommended last run was built and killed within the same day, and this run's job is catching `docs/frontier/local-findings.md` up to that reality — the same "ledger catch-up" role the 2026-07-24 run played for the out-of-band parakeet switch itself.

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

1. Read `docs/frontier/runs/LATEST` (2026-07-24), that run's `report.md` + `frontier.json`, and `docs/frontier/local-findings.md` (#277) — the faster-whisper CTranslate2 disproof (2026-07-12) carries forward unresolved.
2. Re-read `config/models.yaml` → `roles:` for the current incumbents — no out-of-band role change since the last run; `audio_transcribe` still reads `parakeet` with `fallback: [whisper]`.
3. Cross-checked GitHub issues for anything the previous run's actionable verdict should have produced: found issue #401 ("parakeet: wire FluidAudio CustomVocabularyContext rescorer for wake phrase/jargon"), filed same day as the 2026-07-24 run and **closed not-planned the same day** with a full disproof in the closing comment, referencing a doc update in #407. Confirmed `docs/frontier/local-findings.md` was never updated with that disproof — a gap in the standing "same PR as the disproof" contract — and backfilled it as step 1 of this run, before computing verdicts, per the skill's own override rule.
4. Surveyed the external landscape via web search for the 2026-07-24 → 2026-08-06 window: DeepSeek V4 (Pro/Flash) open-weight status, Kimi K3's actual shipped weights, GLM-5.x roadmap, Qwen 3.7/3.8 open-weight status, Mistral's "fat but sparse" MoE status, Gemma/GPT-OSS/Phi/Granite for any new releases, and llama.cpp release notes for perf-relevant changes (CUDA/Blackwell sparse-attention and flash-attention kernel work landed in this window — informational, doesn't change any model-level verdict).
5. Computed VRAM with the standing rule of thumb **Q4_K_M ≈ 4.5 bits/param** plus KV-cache; carried figures from the previous run where nothing changed, flagged as unchanged; worked the DeepSeek V4 Flash math fresh (§4/§5) since it's a genuine new entrant.
6. Applied the honesty rules: date-stamped claims, ≤3% composite = tie (the Gemma 4 26B / Qwen 3.6 35B-A3B tie stands), licenses surfaced, DeepSeek V4 Flash's numbers marked estimated where no first-party measurement exists, and the local-findings override applied to both the faster-whisper CT2 entry and the newly-backfilled FluidAudio entry (both `watch`, neither re-proposed).

---

## 4. How to read the chart

- **X axis** — estimated single-stream tokens/second on the 5060 Ti at the recommended quant.
- **Y axis** — composite quality score for *these* workloads (0–100, normalized).
- **Bubble size** — VRAM at recommended quant. **Color** — tier (A fast / B balanced / C quality).
- **Filled border** — on the Pareto frontier. **Hollow** — dominated.
- **Toggle** — show only models that fit fully in 16 GB VRAM, or include CPU-offload models.

### Worked memory example (so the math isn't a black box)

This run's cautionary tale is **DeepSeek V4 Flash** — genuinely the right *shape* (MoE, small active set) but the wrong *size*:

```
284B total params, 13B active per token
Q4_K_M (~4.5 bits/param): 284e9 × 4.5/8 ≈ 160 GB just for weights
this box: 16 GB VRAM + 128 GB RAM ≈ 144 GB total
shortfall ≈ 16 GB at Q4 → doesn't fit; next viable quant is IQ3-class
IQ3 (~3.3 bits/param): 284e9 × 3.3/8 ≈ 117 GB → fits, but with real quality loss
decode speed at IQ3, CPU-offloaded 13B active set:
  13e9 × 3.3/8 ≈ 5.4 GB moved per token
  ÷ ~57.6 GB/s dual-channel DDR5-3600 bandwidth ≈ 11 t/s
```

Compare **Gemma 4 26B MoE** (the incumbent): ~14 GB at native W4, 4B active params, fully GPU-resident → bandwidth pressure set by GDDR7's ~448 GB/s, not system RAM's ~58 GB/s → 99 t/s. **The gap between "fits at all" and "fits fast" is exactly RAM vs VRAM bandwidth, an order of magnitude apart on this box** — a CPU-offloaded MoE with 3× Gemma's active-parameter count and none of its GPU residency lands at roughly a ninth of the speed, which is why DeepSeek V4 Flash reads as a Tier C consideration at best, not a Tier B contender, even before weighing its unverified quality edge on non-coding workloads.

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

No change this run. The incumbent keeps the tier — nothing in the window ships a new 4B-class open model; the rumored Qwen 3.8-27B open-weight companion has not shipped and is the wrong size class regardless.

### Tier B — Balanced (the workhorse) — **TIED, unchanged**

| model | params | quant | VRAM | tok/s | quality | ctx | license | on frontier |
|-------|--------|-------|------|-------|---------|-----|---------|-------------|
| ★★ Gemma 4 26B MoE *(incumbent)* | 26B / 4B active | native W4 | ~14 GB | 99 | 83 | 256k | Gemma | yes |
| ★★ Qwen 3.6 35B-A3B | 35B / 3B active | Q4_K_M | ~13.5 GB | 98 | 84 | 262k | Apache 2.0 | yes |
| ☆ GPT-OSS 20B | 21B / 3.6B active | MXFP4 | ~12 GB | ~100 | 72 | 131k | Apache 2.0 | yes |
| Gemma 4 12B Unified | 12B dense | Q4_K_M | ~7 GB | ~45 est | 76 est | 256k | Apache 2.0 (verify) | no |
| Ministral 3 14B | 14B dense | Q4_K_M | ~8.5 GB | ~35 est | 70 est | 256k | Apache 2.0 | no |
| Mistral Small 3.2 | ~22B dense | Q4_K_M | ~13 GB | ~30 | 74 | 128k | Apache 2.0 | no |

The tie at the top persists. No entrant this run challenges either leader within the 16 GB-fits envelope.

### Tier C — Quality (slow, CPU-offload, batch / non-interactive)

| model | params | quant | VRAM | tok/s | quality | ctx | license | on frontier |
|-------|--------|-------|------|-------|---------|-----|---------|-------------|
| ★ Qwen3 32B dense | 32B | Q4_K_M | ~19.5 GB (spill) | ~11 | 84 | 128k | Apache 2.0 | yes |
| Llama 3.3 70B | 70B | Q4_K_M | ~40 GB (offload) | ~4 | 82 | 128k | Llama 3.3 | no |
| Mistral Medium 3.5 | 128B dense | Q4_K_M | ~75 GB (offload) | ~2 | 86 | 256k | Modified MIT | no |
| DeepSeek V4 Flash *(new, est.)* | 284B / 13B active | IQ3-class | ~117 GB (offload) | ~11 est | unverified | 1M (reported) | MIT | no |

No change to the frontier. **DeepSeek V4 Flash added as a considered-and-dropped entry** — see below.

### Models considered and dropped this run

- **DeepSeek V4 Flash 0731 (284B total / 13B active, shipped 2026-07-31, MIT)** — the window's genuine new entrant. Worked math in §4: doesn't fit at Q4 (~160 GB vs ~144 GB budget), needs an IQ3-class quant (~117 GB) with real quality loss to fit at all, and lands at an estimated ~11 t/s — Qwen3 32B's speed, at a worse quant, with no first-party or leaderboard evidence of a composite-quality edge on this stack's non-coding weighting (DeepSeek's historical strength — math/code — carries 0% weight here). Not recommended; flagged for re-evaluation if a smaller/better-fitting quant or a non-coding benchmark comparison surfaces.
- **DeepSeek V4 Pro (1.6T / 49B active)** — still preview-only as of 2026-08-04, not a stable open-weight release; out on both availability and size (49B active alone implies a footprint far past this box's budget).
- **Kimi K3 (2.8T total / 16B active, weights shipped 2026-07-27 as MXFP4, ~594 GB)** — shipped on schedule; confirms rather than changes the prior decisive NO-GO (§4's math scales the same way).
- **GLM-5.2 (744B / 40B active)** — standing NO-GO carried from four runs now (`docs/glm-5.2-evaluation.md`, #141); >1 TB VRAM in BF16, no quant fits.
- **GLM-5.5** — rumored for an August 2026 release per a JPMorgan research note relayed by Reuters (2026-06-25); no model card, benchmark, or endpoint published as of this run. Watch next cycle.
- **Mistral's "fat but sparse" MoE (teased Jul 8, early access)** — unchanged: no parameter count, benchmark, or license disclosed; still no public weights.
- **Qwen 3.7 Max / Plus (API, May 20)** — confirmed still closed-weights. Unconfirmed community chatter about a further "Qwen 3.8" (with a rumored smaller open-weight 27B companion) exists but nothing has shipped — not actionable this run.
- Standing drops carried from prior runs: Qwen 3.6 27B dense, Gemma 3 27B, Qwen3.5-35B-A3B, Mistral Small 3.2, Llama 3.2 3B, Phi-4 Mini (for ES/CA), Qwen3 8B/9B class, MiniMax M3 (still academic at 2-bit).

---

## 6. Concurrency plan

Unchanged from the previous run — the four recipes still describe the practical envelope:

1. **Two lanes (default):** Qwen 3.5 4B (GPU ~3 GB) + Gemma 4 26B MoE (GPU ~14 GB). Both near-peak; ~17 GB with graceful shared-memory spill.
2. **Qwen 3.6 stack (all-Apache):** Qwen 3.5 4B + Qwen 3.6 35B-A3B (~13.5 GB). Speed parity; license clarity.
3. **Quality batch:** Qwen 3.5 4B + Qwen3 32B dense (~3.5 GB CPU spill, ~10 t/s) for overnight reprocessing.
4. **Three concurrent:** Qwen 3.5 4B + GPT-OSS 20B (GPU) + Gemma 3 4B (CPU, ~10 t/s on the 7800X3D) as a Catalan specialist.

No placement changes this window. llama.cpp landed CUDA sparse-attention and flash-attention kernel tuning work in July 2026 (weekly GitHub activity reports); this is a potential free speed uplift for the existing GPU-resident picks on Blackwell but hasn't been locally re-benchmarked — noted in §9, not assumed.

---

## 7. Audio (ASR) annex — workloads F, G, H

### 7.1 The landscape in August 2026 — a recommendation built and killed in one day

No new external ASR release displaces anything in this window. The story is entirely internal: last run recommended wiring FluidAudio's `CustomVocabularyContext` rescorer into the Mac Parakeet worker to recover the "Claude Code" wake phrase and jargon terms the 2026-07-22 parakeet-primary placement swap (#350) knowingly dropped. That work happened the same day the recommendation was published — issue #401, filed and closed not-planned on 2026-07-24 — and the fix does not work: see §7.2.1 for the root cause. `docs/frontier/local-findings.md` now carries this as an unresolved entry (backfilled by this run — see §3), so future runs won't re-propose the same dead-end.

### 7.2 ASR candidate comparison (EN + ES)

| Variant | Params | VRAM | RTFx (measured/est.) | EN | ES | Translates → EN? | Notes |
|---------|--------|------|----------------------|----|----|-------------------|-------|
| **★ Parakeet TDT v3** (Mac ANE, current primary) | 0.6B | ~2 GB (CoreML) | 65.8× measured (ANE), sub-second even on 108 s clips | ✅ | ✅ (25 langs) | ❌ | Unchanged incumbent since 2026-07-22. Fastest STT in the fleet by far. Published generic Spanish WER modestly beats Whisper — but this project's own jargon-heavy dictation domain measures worse (§7.2.1), and the identified fix for that gap is now disproven. |
| **◆ Whisper Large v3 Turbo** (automatic failover) | 809M | ~1.6 GB | 40× tower / 19.3× gaming (measured, boosted) | ✅ | ✅ | ❌ | Unchanged. Still the accuracy leader on this domain thanks to the `--carry-initial-prompt` boosting glossary (#91) — a lever Parakeet structurally lacks. |
| ✗ Parakeet + FluidAudio CustomVocabularyContext | 0.6B + ~97 MB CTC model | ~2 GB + rescorer | n/a — disproven, not shipped | — | — | ❌ | **New this run.** Built and killed same-day (#401): post-hoc rescoring can swap a word, not insert a decoder-dropped phrase. Carried `watch` per `local-findings.md` (backfilled this run). |
| faster-whisper Turbo (CT2) | 809M (CT2) | ~1.0 GB INT8 | measured 1.0× vs whisper.cpp — **disproven** | ✅ | ✅ | ❌ | Carried `watch` per `docs/frontier/local-findings.md` (#277); applies to the failover leg's engine, not the primary. |
| Whisper Large v3 (faster-whisper) | 1.55B | ~2 GB | 30–50× | ✅✅ | ✅✅ | ✅ | Workload-G single-model fallback, unchanged. |
| Granite Speech 3.3 8B | 8B | ~5 GB | 15–30× | ✅✅✅ | ✅✅ | ✅ (X↔EN) | Accuracy tier; only if the failover's own errors start to matter. |
| Qwen3-ASR 1.7B | 1.7B | ~1.5 GB | TBD | ✅ | ✅ | ❌ | Still `watch` — no domain-specific (jargon-heavy dictation) WER comparison found this run either, the same gap that made Parakeet's generic numbers misleading here. |

#### 7.2.1 Why the identified fix failed — and why published benchmarks still don't transfer

FluidAudio 0.15.4 ships exactly one vocabulary mechanism for the Parakeet TDT 0.6B v3 model: a **post-hoc CTC rescorer** (`CustomVocabularyContext`/`VocabularyRescorer`) that re-scores already-emitted transcript words against per-frame log-probabilities and swaps in a boosted term when it has stronger acoustic evidence. There is no decode-time biasing hook for the TDT architecture — nothing that can make the decoder consider emitting a phrase it didn't already emit. That's the root cause #401 found: "Claude Code" isn't mis-transcribed as something similar, it's dropped entirely (heard as "Yes"/"Yeah"), so there's no candidate word for the rescorer to swap **from**. "YOLO" is a 4-character token, short enough that the rescorer's similarity guard (≥0.80, to avoid false-positive corrections) blocks the correction it would otherwise make against "yellow". A four-point threshold sweep confirmed a hard wall: every setting that avoids false positives is a no-op on both targets, and every setting that fixes either target simultaneously corrupts unrelated correct words elsewhere in the transcript (`mention`→`Notion`, `tab`→`Tailscale`, `left, right`→`Playwright`). This is the same honesty-rules lesson as before — a published-benchmark verdict (Parakeet ≥ Whisper on generic Spanish WER) that doesn't survive contact with this project's actual domain — but this time the divergence was closed off at the *implementation* level, not just observed at the *benchmark* level: the fix that would have reconciled the two doesn't exist yet in the tooling.

### 7.3 Workload F — transcribe EN/ES

**Verdict: `watch`.** No role change — `parakeet` stays primary with `whisper` as automatic failover, both unchanged since 2026-07-22 (#350/#348). **Explicit answer to the brief's required question: no strict upgrade over the incumbent transcribe model exists for EN/ES this run.** The one concretely identified fix for Parakeet's wake-phrase/jargon gap (FluidAudio custom vocabulary) was built and disproven (§7.2.1); Qwen3-ASR remains an unverified `watch` with no domain-specific comparison; no new ASR model release this window changes either leg. The whisper failover's own `faster-whisper` (CTranslate2) engine swap stays disproven/`watch` per `local-findings.md` (#277), unrelated to this verdict.

### 7.4 Workload G — ES audio → English

**Verdict unchanged: two-stage default.** faster-whisper Turbo transcribes ES, Gemma 4 26B MoE translates + polishes + de-disfluences in one call (~15 GB total). Single-model faster-whisper Large v3 `task=translate` stays the fallback when the LLM slot is busy — implemented today by the `whisper_translate` role slot (whisper-medium, lazy CPU), hence its `watch` verdict rather than retirement. No change this window.

### 7.5 Workload H — disfluency / filler removal

**Verdict unchanged: folded into the LLM polishing pass.** Specialized disfluency models remain research-grade with sparse Spanish coverage; the tier-B LLM already does this work in the polish prompt (the canonical prompt lives in `frontier.json` → `disfluency_verdict.prompt`).

### 7.6 Concurrency footprint

Unchanged from the previous run: whisper-turbo (accurate dictation) and whisper-vanilla/translate run on the gaming satellite, orpheus (expressive TTS) is back on the tower, freeing the tower's GPU for agentic_heavy + agentic_light exclusively (#343/#422). Parakeet runs on the Mac Mini's ANE, its own dedicated accelerator, adding zero contention with either box. Piper (fast TTS) stays on the tower CPU.

### 7.7 Dropped

Carried from prior runs: Canary-Qwen (EN-only), Parakeet v1/v2 (superseded by v3), Distil-Whisper (EN-only), Phi-4-Multimodal (footprint/tooling), Seamless M4T v2 (heavier, weaker tooling than two-stage), specialized disfluency models (worse ES coverage than the LLM pass), FluidAudio CustomVocabularyContext (disproven this run, §7.2.1).

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
| 2026-08-06 | agentic_light | qwen35_4b (qwen3.5-4b) | keep | Gemma 3 4B (Catalan niche) |
| 2026-08-06 | agentic_heavy | gemma4_26b (gemma4-26b-a4b-it) | keep | Qwen 3.6 35B-A3B (tie) |
| 2026-08-06 | audio_transcribe | parakeet (parakeet-tdt-0.6b-v3) + whisper fallback | watch | — (FluidAudio fix disproven #401) |
| 2026-08-06 | audio_translate | whisper_translate (whisper-medium) | watch | two-stage Turbo → Gemma 4 26B |

Reading the progression: `audio_transcribe`'s `runtime_upgrade` row from 2026-07-24 is now closed out — built, tested, and disproven within the same day it was proposed, the fastest propose→disprove cycle in this report's history. This is exactly the loop `docs/frontier/local-findings.md` (#277) exists to stop: the verdict drops to `watch` and stays there until a re-open trigger (§ local-findings.md) is met, rather than getting re-proposed next cycle. `agentic_light`/`agentic_heavy`/`audio_translate` remain stable since 2026-05-10.

---

## 9. Open questions / uncertainty

- **No fix currently exists for Parakeet's wake-phrase/jargon gap.** FluidAudio's only vocabulary mechanism (post-hoc CTC rescoring) is structurally incapable of it (§7.2.1). The re-open trigger is a FluidAudio release with decode-time TDT biasing, or an alternative checkpoint exposing one — worth a periodic upstream changelog check, not a re-test of the same approach.
- **DeepSeek V4 Flash's composite quality on non-coding workloads is unverified.** No first-party or leaderboard number was found for FLORES-style multilingual or agentic/tool-use benchmarks isolated from its coding/math strengths; the §4/§5 speed estimate is also unverified locally. Worth a smoke test only if a smaller/better-fitting quant appears.
- **GLM-5.5's August release is a rumor (JPMorgan research note, 2026-06-25), not a confirmed date.** Watch for an actual model card next cycle.
- **Qwen 3.8's rumored smaller open-weight companion (27B) has not shipped.** Track alongside the still-unshipped Qwen 3.7 open weights.
- **llama.cpp's July CUDA sparse-attention / flash-attention kernel work may speed up the existing GPU-resident Tier A/B picks on Blackwell.** Not locally re-benchmarked this run — the tok/s figures in §5 are still the prior run's numbers, flagged unchanged rather than re-measured.
- Standing carries: Gemma 4 12B Unified license still unverified on the model card; Catalan on Qwen 3.6 35B-A3B still needs a local smoke test; MiniMax M3 at UD-Q2 still academic.

---

## 10. Current decisions (live, edited by `/swap-model`)

The decisions below mirror `config/models.yaml` → `roles:` at the time
this section was last updated. `/swap-model` rewrites both this section
and the yaml together, so the two stay in sync.

| Role | Model | Decided | Why |
|---|---|---|---|
| **agentic_light** | `qwen35_4b` (qwen3.5-4b) | 2026-05-10 | Upgraded from gemma4_e4b via `/swap-model`. Tier A top pick — hybrid Gated DeltaNet + sparse MoE on a 4B base, Q4_K_M ~3 GB, 262k native ctx, 201 languages, Apache 2.0. gemma4_e4b retained in `enabled:` for ad-hoc fallback. |
| **agentic_heavy** | `gemma4_26b` (gemma4-26b-a4b-it) | 2026-05-10 | Tier B top pick. 99 t/s, 256k ctx, strong multilingual including Catalan. Tied with Qwen 3.6 35B-A3B (Apache 2.0) — Gemma stays default on Catalan track record. |
| **audio_transcribe** | `parakeet` (parakeet-tdt-0.6b-v3, Mac ANE) with `fallback: [whisper]` | 2026-07-22 | Changed directly via #350 (fleet placement/latency, #343 benchmark), not `/swap-model`. A speed-over-accuracy trade with known regressions (dropped wake phrase, jargon mangling). The identified remediation (FluidAudio custom-vocabulary) was tried and disproven (#401) — no current fix path; see §7. |
| **audio_translate** | `whisper_translate` (whisper-medium, lazy CPU) | 2026-05-10 | Strict frontier reading recommends `watch` — the two-stage path (Turbo → Gemma 4 26B) is the default, leaving this slot as a fallback only. Keep defined and lazy-loaded; no active maintenance. |

---

*Generated by the `/frontier-refresh` skill (`.claude/skills/frontier-refresh/SKILL.md`), which owns the research brief and this report's output contract. This is the August 6, 2026 snapshot.*
