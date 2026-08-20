# Local LLM + ASR Efficient Frontier — Results

**Run date:** 2026-08-20
**Hardware:** RTX 5060 Ti 16 GB · Ryzen 7 7800X3D · 128 GB DDR5
**Workloads:** OpenClaw agentic (fast + deep lanes), transcript polishing, document processing, EN↔ES↔CA translation, **audio transcription EN/ES, audio translation ES → EN, transcript disfluency removal**. No coding.

---

## 0. Verdict

| role | incumbent | verdict | best alternative | gap | reason |
|------|-----------|---------|-------------------|-----|--------|
| `agentic_light` | `qwen35_4b_nothink` (qwen3.5-4b-nothink) | keep | Gemma 3 4B (Catalan niche) | — | No new 4B-class entrant. Qwen's next flagship line (Qwen 3.8) shipped its smallest open checkpoint at 27B, not a 4B-class companion — last run's flagged rumor is now resolved as a Tier B entrant, not a Tier A one |
| `agentic_heavy` | `gemma4_26b` (gemma4-26b-a4b-it) | keep | Qwen 3.6 35B-A3B | tie (≤3%) | Tie persists. Qwen3.8-27B (dense, Apache 2.0, shipped 2026-08-14) is the window's headline entrant but is dominated on speed by both MoE leaders at a similar VRAM footprint — the same pattern that dropped Qwen 3.6 27B dense last quarter. DeepSeek V4 Pro reached GA 2026-08-13 (1.6T/49B active, MIT) and V4 Flash's real GGUF footprint is smaller than last run's estimate, but neither clears the bar — see §5 |
| `audio_transcribe` | `parakeet` (parakeet-tdt-0.6b-v3, Mac ANE) + whisper fallback | watch | — | — | No fix surfaced this window. Qwen3-ASR's general-benchmark WER improved (MLX 1.7B build now beats WhisperKit Turbo on a generic corpus), but FluidAudio removed its experimental Qwen3-ASR backend (upstream #676) and no domain-specific (jargon-heavy dictation) comparison exists — same generic-vs-domain gap that made Parakeet's own numbers misleading (§7.2.1) |
| `audio_translate` | `whisper_translate` (whisper-medium) | watch | two-stage: Turbo → Gemma 4 26B MoE | — | Unchanged; two-stage stays the default, this slot stays a lazy fallback |

**Diff vs previous run (2026-08-06):** no role changes — all four verdicts hold. The window's real news is on the text side: Alibaba shipped the Qwen 3.8 generation (Qwen3.8-Max API on 2026-08-02, a 2.4T/95B-active open flagship on 2026-08-12, and **Qwen3.8-27B** — dense, Apache 2.0, native vision-language, 262K ctx — on 2026-08-14), DeepSeek V4 reached GA as DeepSeek-V4-Pro-0813 (1.6T/49B active, MIT), and Z.ai shipped GLM-5.3 (same 743B/40B-active footprint as GLM-5.2, reused base, no architecture change). None of it changes a verdict: Qwen3.8-27B is dense and loses the speed race to both Tier B MoE incumbents at a comparable VRAM footprint (§5); DeepSeek V4 Pro and the Qwen 3.8 flagship are both far outside this box's budget; GLM-5.3 is the same standing NO-GO as GLM-5.2 under a new benchmark suite. DeepSeek V4 Flash's real GGUF sizes (162 GB lossless 8-bit, 103 GB at 3-bit) are now known rather than estimated — the 3-bit build fits this box's ~144 GB combined budget for the first time, a genuine change from "doesn't fit at all," but with no verified non-coding composite quality and only a speculative "DSpark" 2× decode claim, it stays a flagged watch item, not a recommendation (§4/§5). `agentic_light`, `audio_transcribe`, and `audio_translate` are unchanged in substance. Note: `config/models.yaml`'s `agentic_light` role pointer moved from `qwen35_4b` to `qwen35_4b_nothink` on 2026-08-12 (#489/#490) — an internal thinking-mode routing default, not a frontier verdict change; both ids share the same underlying Qwen 3.5 4B weights.

*Why this is the core artifact:* everything below exists to justify these six columns. If you read nothing else, this table plus the diff line is the run.

---

## 1. Objective

The "efficient frontier" of local LLMs is the set of models where, for a given level of quality, no other model is faster (or, for a given speed, no other model is more accurate). Everything off the frontier is **dominated** — a strictly better choice exists on at least one axis without giving up the other.

The frontier is **always hardware- and workload-specific**: a 70B that dominates on a 5090 falls off the 5060 Ti's frontier into CPU-offload territory, and a coding-specialist that wins SWE-bench is irrelevant here because coding carries 0% weight. This report identifies the frontier for *this* box and *these* workloads, as of August 2026.

**What changed since 2026-08-06:** Alibaba's Qwen 3.8 generation dominated the window. Qwen3.8-Max (closed, API-only) launched 2026-08-02; a 2.4T-total/95B-active open-weight flagship — the first-ever open release of a Qwen-Max-class model — followed 2026-08-12; and **Qwen3.8-27B** (dense, 27.78B params, Apache 2.0, native text/image/video input, 262K native context extensible to 1M via YaRN, 3:1 ratio of linear Gated-DeltaNet to full-attention layers) shipped 2026-08-14. Its benchmark gains are real but almost entirely coding/agentic-coding (Terminal-Bench 2.1 63.4→73.0, DeepSWE 1.1 13.3→42.2, SWE-Bench Pro 61.7%) — 0% weight here — and being **dense** means it loses the decode-speed race badly to this box's MoE incumbents: community RTX 4090 measurements put it at ~49 t/s (matching its Qwen 3.6-27B-dense predecessor), and Gemma 4 26B MoE hits 99–128 t/s on comparable hardware because it only activates ~4B of its 26B parameters per token. Same conclusion as every prior dense-27B entrant this report has evaluated: dominated by the Tier B MoE leaders at a similar VRAM footprint. The 2.4T/95B-active flagship and DeepSeek V4 Pro (GA 2026-08-13, 1.6T/49B active, MIT) are both an order of magnitude past this box's ~144 GB combined VRAM+RAM budget — not evaluated further. GLM-5.3 (Z.ai, 2026-08-14) reuses the GLM-5.2 base exactly — same 743B/40B-active footprint, no re-pretrain — so the standing NO-GO carries forward under a refreshed (and largely cybersecurity/coding-flavored) benchmark suite. **GLM-5.5** stays an unconfirmed rumor (JPMorgan research note) with no model card, benchmark, or endpoint as of this run.

The one genuinely new data point: DeepSeek V4 Flash's **actual** GGUF footprint is now published rather than hand-estimated — a lossless 8-bit build at 162 GB and a community 3-bit build at 103 GB, using the model's native mixed FP4/FP8 quantization (routed experts, ~96% of params, ship at ~4.25 bits/weight). The 3-bit build fits this box's combined budget for the first time (previous run's Q4_K_M-rule-of-thumb estimate put it at ~117 GB and outside budget by ~16 GB at Q4, requiring an even more aggressive quant). This changes the "doesn't fit at all" verdict to "fits, but everything else about it is still unverified" — see §4/§5 for the honesty-rules treatment.

On the audio side: no external ASR release displaces the incumbent. Qwen3-ASR's generic-benchmark WER improved (an MLX 1.7B 5-bit build reportedly beats WhisperKit Large-v3 Turbo on a general corpus), but FluidAudio — the Swift runtime this project's own Mac Parakeet worker is built on — **removed** its experimental Qwen3-ASR backend upstream (#676), and no domain-specific (jargon-heavy dictation) comparison exists for either model. This is the same generic-vs-domain gap that made Parakeet's own leaderboard numbers misleading here (§7.2.1) — a reason for caution about Qwen3-ASR's numbers, not a reason to act on them.

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

1. Read `docs/frontier/runs/LATEST` (2026-08-06), that run's `report.md` + `frontier.json`, and `docs/frontier/local-findings.md` (#277) — three unresolved entries carry forward: the faster-whisper CTranslate2 disproof (2026-07-12), the FluidAudio `CustomVocabularyContext` disproof (2026-07-24, backfilled 2026-08-06), and the Nemotron 3 Nano 4B quality disproof (2026-08-10, filed between runs via issue #486). None of this run's candidates match any of the three, so the override rule wasn't triggered.
2. Re-read `config/models.yaml` → `roles:` for the current incumbents. Confirmed one out-of-band change since the last run: `agentic_light` now resolves to `qwen35_4b_nothink` (was `qwen35_4b`) since 2026-08-12 (#489/#490) — a thinking-mode default flip on the same underlying weights, not a model swap. `agentic_heavy`, `audio.transcribe`, and `audio.translate` are unchanged.
3. Surveyed the external landscape for the 2026-08-06 → 2026-08-20 window: Qwen 3.8 generation (Max, the 2.4T/95B-active open flagship, and Qwen3.8-27B), DeepSeek V4 GA status and V4 Flash's actual GGUF sizes, GLM-5.3's parameters/architecture vs GLM-5.2, GLM-5.5's rumor status, Kimi K3/K4 status, Mistral's "fat but sparse" MoE status, Gemma/GPT-OSS/Phi/Granite for any new releases, and llama.cpp weekly activity + a Blackwell/Gemma-4-specific performance report for anything that would change this repo's own measured tok/s.
4. Cross-checked the Blackwell/Gemma-4 performance report against this repo's actual `gemma4_26b` launch args (`config/models.yaml`): the community fix (drop a copied `--swa-full` flag; use IQ4_XS or IQ3_XXS) matches what this row already does — no `--swa-full` is set, and IQ4_XS is already the active quant — so the reported regression doesn't apply to this box's configuration. No runtime change indicated; noted in §9 as a low-priority optimization idea (IQ3_XXS reportedly ~99 t/s vs IQ4_XS's ~85 t/s in that report) worth a future local A/B, not an action this run.
5. Computed VRAM with the standing rule of thumb **Q4_K_M ≈ 4.5 bits/param** plus KV-cache where no published GGUF size exists; used DeepSeek V4 Flash's actual published GGUF sizes (162 GB / 103 GB) instead of the rule of thumb now that they exist. Worked the Qwen3.8-27B dense-vs-MoE math fresh (§4) since it's the window's genuine new entrant.
6. Applied the honesty rules: date-stamped claims, ≤3% composite = tie (the Gemma 4 26B / Qwen 3.6 35B-A3B tie stands), licenses surfaced, DeepSeek V4 Flash's quality and the "DSpark" 2× decode claim marked estimated/unverified, Qwen3.8-27B's composite quality estimated from partial evidence (strong instruction-following, but its headline gains are coding-specific) and flagged as such.

---

## 4. How to read the chart

- **X axis** — estimated single-stream tokens/second on the 5060 Ti at the recommended quant.
- **Y axis** — composite quality score for *these* workloads (0–100, normalized).
- **Bubble size** — VRAM at recommended quant. **Color** — tier (A fast / B balanced / C quality).
- **Filled border** — on the Pareto frontier. **Hollow** — dominated.
- **Toggle** — show only models that fit fully in 16 GB VRAM, or include CPU-offload models.

### Worked memory example (so the math isn't a black box)

This run's cautionary tale is **Qwen3.8-27B** — the right *quality class* but the wrong *architecture* for this hardware:

```
27.78B params, dense (every parameter active on every token)
Q4_K_M (~4.5 bits/param): 27.78e9 × 4.5/8 ≈ 15.6 GB weights — barely fits 16 GB,
  leaves ~0.4 GB for KV cache before spilling
Community RTX 4090 measurement: ~49 t/s decode (matches its Qwen 3.6-27B-dense
  predecessor almost exactly — same architecture class, same bandwidth wall)
This box's 5060 Ti has ~448 GB/s vs the 4090's ~1008 GB/s — roughly 44% of the
  bandwidth, so a naive scale-down lands near ~22-30 t/s (consistent with the
  Qwen 3.6 27B dense entry's own measured/estimated 31 t/s on this exact box)
```

Compare **Gemma 4 26B MoE** (the incumbent, same VRAM class): ~14 GB at native W4, only ~4B of its 26B parameters active per token → the GPU moves ~4B params' worth of bytes per token instead of ~28B, so it hits 99–128 t/s on comparable hardware — **3-4× Qwen3.8-27B's speed at a smaller footprint**, because the *architecture* (MoE vs dense), not the parameter count, is what the memory-bandwidth wall actually taxes. This is the same lesson the 2026-08-06 report drew from DeepSeek V4 Flash's shape being right and its size being wrong; here the axes are reversed — Qwen3.8-27B's *size* is fine for this box, but its *architecture* (dense) is the wrong shape for a bandwidth-bound GPU, which is exactly what keeps dominating every dense ~27-32B entrant off this hardware's Tier B frontier run after run.

DeepSeek V4 Flash's own math also moved this run — its real (not estimated) GGUF sizes:

```
Native format: routed experts (~96% of params) at ~4.25 bits/weight (mixed FP4/FP8)
Published GGUF builds: lossless 8-bit ≈ 162 GB, community 3-bit ≈ 103 GB
This box: 16 GB VRAM + 128 GB RAM ≈ 144 GB combined
103 GB (3-bit) now fits within the ~144 GB budget — a change from last run's
  Q4-rule-of-thumb estimate (~117 GB, which did NOT fit)
Decode speed at 3-bit, CPU-offloaded 13B active set (baseline, no acceleration):
  13e9 × 3.3/8 ≈ 5.4 GB moved per token ÷ ~57.6 GB/s DDR5-3600 ≈ 10 t/s
"DSpark" (vendor-claimed decode acceleration for this GGUF path) claims up to 2×
  — unverified locally, so treat ~10-20 t/s as the honest range, not a point estimate
```

Fitting is necessary but not sufficient: DeepSeek V4 Flash has no verified non-coding composite quality number, and its best-case speed (even with the DSpark claim taken at face value) is still well below Qwen3 32B dense's already-verified ~11 t/s at a much smaller footprint. "Now fits" moved it from impossible to merely unattractive — not onto the frontier.

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

No change this run. The incumbent keeps the tier. The window's headline Qwen release (3.8) shipped no 4B-class model at all — its smallest open checkpoint is the 27B in Tier B — so last run's flagged rumor is now resolved: there is no small-companion challenger, at least not yet.

### Tier B — Balanced (the workhorse) — **TIED, unchanged**

| model | params | quant | VRAM | tok/s | quality | ctx | license | on frontier |
|-------|--------|-------|------|-------|---------|-----|---------|-------------|
| ★★ Gemma 4 26B MoE *(incumbent)* | 26B / 4B active | native W4 | ~14 GB | 99 | 83 | 256k | Gemma | yes |
| ★★ Qwen 3.6 35B-A3B | 35B / 3B active | Q4_K_M | ~13.5 GB | 98 | 84 | 262k | Apache 2.0 | yes |
| ☆ GPT-OSS 20B | 21B / 3.6B active | MXFP4 | ~12 GB | ~100 | 72 | 131k | Apache 2.0 | yes |
| Qwen3.8-27B *(new, dense)* | 27.78B dense | Q4_K_M | ~15.6 GB | ~28 est | 79 est | 262k (1M YaRN) | Apache 2.0 | no |
| Gemma 4 12B Unified | 12B dense | Q4_K_M | ~7 GB | ~45 est | 76 est | 256k | Apache 2.0 (verify) | no |
| Ministral 3 14B | 14B dense | Q4_K_M | ~8.5 GB | ~35 est | 70 est | 256k | Apache 2.0 | no |
| Mistral Small 3.2 | ~22B dense | Q4_K_M | ~13 GB | ~30 | 74 | 128k | Apache 2.0 | no |

The tie at the top persists. **Qwen3.8-27B added** — the window's genuine new entrant, but dominated on speed by both leaders at a comparable VRAM footprint (§4's worked example); its quality estimate (79) leans on strong reported instruction-following but is not composite-verified for this stack's non-coding weighting, and even generously scored it wouldn't clear the ~3-4× speed gap.

### Tier C — Quality (slow, CPU-offload, batch / non-interactive)

| model | params | quant | VRAM | tok/s | quality | ctx | license | on frontier |
|-------|--------|-------|------|-------|---------|-----|---------|-------------|
| ★ Qwen3 32B dense | 32B | Q4_K_M | ~19.5 GB (spill) | ~11 | 84 | 128k | Apache 2.0 | yes |
| Llama 3.3 70B | 70B | Q4_K_M | ~40 GB (offload) | ~4 | 82 | 128k | Llama 3.3 | no |
| Mistral Medium 3.5 | 128B dense | Q4_K_M | ~75 GB (offload) | ~2 | 86 | 256k | Modified MIT | no |
| DeepSeek V4 Flash *(updated)* | 284B / 13B active | GGUF 3-bit (native FP4/FP8) | ~103 GB (offload) | ~10-20 est | unverified | 1M (reported) | MIT | no |

No change to the frontier. **DeepSeek V4 Flash's footprint updated** (now-fits, still-unrecommended) — see below; DeepSeek V4 Pro and the Qwen 3.8 2.4T flagship are new entries in the dropped list, both far outside budget.

### Models considered and dropped this run

- **Qwen3.8-27B (27.78B dense, Apache 2.0, shipped 2026-08-14)** — the window's genuine new entrant, evaluated in the Tier B table above. Dominated on speed (~3-4×) by both Gemma 4 26B MoE and Qwen 3.6 35B-A3B at a comparable VRAM footprint — dense architecture is the wrong shape for a bandwidth-bound 16 GB GPU, the same conclusion drawn for every dense ~27-32B entrant this report has evaluated. Its reported gains (Terminal-Bench, DeepSWE, SWE-Bench Pro) are coding/agentic-coding benchmarks carrying 0% weight here.
- **Qwen3.8 2.4T/95B-active open flagship (shipped 2026-08-12)** — the first open release of a Qwen-Max-class model, but 95B active params alone implies a footprint far past this box's ~144 GB budget even at aggressive quantization. NO-GO on hardware grounds.
- **DeepSeek V4 Pro (GA 2026-08-13, 1.6T total / 49B active, MIT)** — reached general availability this window (was preview-only last run), but 49B active parameters is still an order of magnitude past this box's budget. NO-GO on hardware grounds, unchanged from the preview assessment.
- **DeepSeek V4 Flash (284B / 13B active, MIT)** — footprint math updated this run (§4): real published GGUF sizes (162 GB lossless 8-bit / 103 GB community 3-bit) replace last run's Q4-rule-of-thumb estimate, and the 3-bit build now fits this box's ~144 GB combined budget for the first time. Still not recommended: no first-party or leaderboard evidence of non-coding composite quality, and even the vendor's own "DSpark" 2× decode acceleration claim (unverified locally) would land it around Qwen3 32B dense's already-verified speed at a much larger footprint and an unverified quality edge.
- **GLM-5.3 (743B / 40B active, MIT, shipped 2026-08-14)** — reuses the GLM-5.2 base exactly (no re-pretrain, no architecture change per Z.ai's own release notes), so it carries forward the standing NO-GO (`docs/glm-5.2-evaluation.md`, #141) unchanged: >1 TB VRAM in BF16, no quant fits this box even with full RAM offload.
- **GLM-5.5** — still rumored for an August 2026 release (JPMorgan research note, 2026-06-25, unchanged from the last three runs); no model card, benchmark, or endpoint published as of this run. Watch next cycle.
- **Kimi K3 (2.8T total / 16B active)** — no update this window; standing NO-GO carried unchanged.
- **Mistral's "fat but sparse" MoE (teased July 4, early access)** — no update this window: no parameter count, benchmark, or license disclosed; still no public weights. The "fat but sparse" phrasing itself remains unconfirmed secondary framing, not an on-record Mistral spec.
- Standing drops carried from prior runs: Qwen 3.6 27B dense, Gemma 3 27B, Qwen3.5-35B-A3B, Mistral Small 3.2, Llama 3.2 3B, Phi-4 Mini (for ES/CA), Qwen3 8B/9B class, MiniMax M3 (still academic at 2-bit), Nemotron 3 Nano 4B (quality-disproven locally, #486 — see `local-findings.md`).

---

## 6. Concurrency plan

Unchanged from the previous run — the four recipes still describe the practical envelope:

1. **Two lanes (default):** Qwen 3.5 4B (GPU ~3 GB) + Gemma 4 26B MoE (GPU ~14 GB). Both near-peak; ~17 GB with graceful shared-memory spill.
2. **Qwen 3.6 stack (all-Apache):** Qwen 3.5 4B + Qwen 3.6 35B-A3B (~13.5 GB). Speed parity; license clarity.
3. **Quality batch:** Qwen 3.5 4B + Qwen3 32B dense (~3.5 GB CPU spill, ~10 t/s) for overnight reprocessing.
4. **Three concurrent:** Qwen 3.5 4B + GPT-OSS 20B (GPU) + Gemma 3 4B (CPU, ~10 t/s on the 7800X3D) as a Catalan specialist.

No placement changes this window. A community report flagged abnormally low `gemma4_26b`-class token generation on RTX 5060 Ti Blackwell hardware tied to a copied `--swa-full` flag and a coarser quant — checked against this repo's actual launch args (§3 step 4) and neither applies here, so no action taken; a future IQ3_XXS vs IQ4_XS local A/B is noted in §9 as a low-priority speed idea, not a verdict change.

---

## 7. Audio (ASR) annex — workloads F, G, H

### 7.1 The landscape in August 2026 — no fix, but a caution about generic benchmarks

No external ASR release displaces anything in this window's `audio_transcribe`/`audio_translate` roles. The one relevant development: **Qwen3-ASR's own reported accuracy improved** — an MLX 1.7B 5-bit build is described as beating WhisperKit Large-v3 Turbo on a generic multilingual corpus (1.32% WER vs 1.71%) — but **FluidAudio, the Swift runtime this project's Mac Parakeet worker is built on, removed its own experimental Qwen3-ASR backend** (upstream `FluidInference/fluidaudio-rs` #676). That removal doesn't disprove Qwen3-ASR's quality; it removes the most direct integration path this project would have used to test it, and it's a reminder — per the same lesson §7.2.1 already drew from Parakeet's own generic-vs-domain gap — that a strong generic-corpus WER number is not evidence about this project's jargon-heavy dictation domain. No domain-specific comparison exists for Qwen3-ASR here, so the `watch` status (carried since the 2026-08-06 run) is unchanged, not upgraded.

### 7.2 ASR candidate comparison (EN + ES)

| Variant | Params | VRAM | RTFx (measured/est.) | EN | ES | Translates → EN? | Notes |
|---------|--------|------|----------------------|----|----|-------------------|-------|
| **★ Parakeet TDT v3** (Mac ANE, current primary) | 0.6B | ~2 GB (CoreML) | 65.8× measured (ANE), sub-second even on 108 s clips | ✅ | ✅ (25 langs) | ❌ | Unchanged incumbent since 2026-07-22. Published generic Spanish WER modestly beats Whisper — but this project's own jargon-heavy dictation domain measures worse (§7.2.1), and the identified fix for that gap is disproven (#401). |
| **◆ Whisper Large v3 Turbo** (automatic failover) | 809M | ~1.6 GB | 40× tower / 19.3× gaming (measured, boosted) | ✅ | ✅ | ❌ | Unchanged. Still the accuracy leader on this domain thanks to the `--carry-initial-prompt` boosting glossary (#91) — a lever Parakeet structurally lacks. |
| ✗ Parakeet + FluidAudio CustomVocabularyContext | 0.6B + ~97 MB CTC model | ~2 GB + rescorer | n/a — disproven, not shipped | — | — | ❌ | Disproven #401 (2026-07-24). Carried `watch` per `local-findings.md`. |
| faster-whisper Turbo (CT2) | 809M (CT2) | ~1.0 GB INT8 | measured 1.0× vs whisper.cpp — **disproven** | ✅ | ✅ | ❌ | Carried `watch` per `docs/frontier/local-findings.md` (#277); applies to the failover leg's engine, not the primary. |
| Whisper Large v3 (faster-whisper) | 1.55B | ~2 GB | 30–50× | ✅✅ | ✅✅ | ✅ | Workload-G single-model fallback, unchanged. |
| Granite Speech 3.3 8B | 8B | ~5 GB | 15–30× | ✅✅✅ | ✅✅ | ✅ (X↔EN) | Accuracy tier; only if the failover's own errors start to matter. |
| Qwen3-ASR (1.7B / MLX build) | 1.7B | ~1.5 GB | TBD (domain) | ✅ | ✅ | ❌ | `watch`, unchanged status. Generic WER improved this window (1.32% MLX build vs WhisperKit Turbo's 1.71%) — but FluidAudio dropped its experimental backend (#676) and no jargon-domain comparison exists. Same generic-vs-domain caution as Parakeet. |

#### 7.2.1 Why the identified fix failed — and why published benchmarks still don't transfer

Unchanged from the 2026-08-06 run. FluidAudio 0.15.4 ships exactly one vocabulary mechanism for the Parakeet TDT 0.6B v3 model: a **post-hoc CTC rescorer** (`CustomVocabularyContext`/`VocabularyRescorer`) that re-scores already-emitted transcript words against per-frame log-probabilities and swaps in a boosted term when it has stronger acoustic evidence. There is no decode-time biasing hook for the TDT architecture — nothing that can make the decoder consider emitting a phrase it didn't already emit. That's the root cause #401 found: "Claude Code" isn't mis-transcribed as something similar, it's dropped entirely (heard as "Yes"/"Yeah"), so there's no candidate word for the rescorer to swap **from**. A four-point threshold sweep confirmed a hard wall: every setting that avoids false positives is a no-op on both targets, and every setting that fixes either target simultaneously corrupts unrelated correct words elsewhere in the transcript. This is the same honesty-rules lesson this run applies to Qwen3-ASR's improved generic WER: a published-benchmark verdict that doesn't survive contact with this project's actual domain.

### 7.3 Workload F — transcribe EN/ES

**Verdict: `watch`.** No role change — `parakeet` stays primary with `whisper` as automatic failover, both unchanged since 2026-07-22 (#350/#348). **Explicit answer to the brief's required question: no strict upgrade over the incumbent transcribe model exists for EN/ES this run.** Qwen3-ASR's generic-benchmark accuracy improved, but the same generic-vs-domain gap that undermined Parakeet's own leaderboard numbers applies, and its most direct integration path (FluidAudio's Swift backend) was removed upstream this window. The whisper failover's own `faster-whisper` (CTranslate2) engine swap stays disproven/`watch` per `local-findings.md` (#277), unrelated to this verdict.

### 7.4 Workload G — ES audio → English

**Verdict unchanged: two-stage default.** faster-whisper Turbo transcribes ES, Gemma 4 26B MoE translates + polishes + de-disfluences in one call (~15 GB total). Single-model faster-whisper Large v3 `task=translate` stays the fallback when the LLM slot is busy — implemented today by the `whisper_translate` role slot (whisper-medium, lazy CPU), hence its `watch` verdict rather than retirement. No change this window.

### 7.5 Workload H — disfluency / filler removal

**Verdict unchanged: folded into the LLM polishing pass.** Specialized disfluency models remain research-grade with sparse Spanish coverage; the tier-B LLM already does this work in the polish prompt (the canonical prompt lives in `frontier.json` → `disfluency_verdict.prompt`).

### 7.6 Concurrency footprint

Unchanged from the previous run: whisper-turbo (accurate dictation) and whisper-vanilla/translate run on the gaming satellite, orpheus (expressive TTS) is on the tower, freeing the tower's GPU for agentic_heavy + agentic_light exclusively (#343/#422). Parakeet runs on the Mac Mini's ANE, its own dedicated accelerator, adding zero contention with either box. Piper (fast TTS) stays on the tower CPU.

### 7.7 Dropped

Carried from prior runs: Canary-Qwen (EN-only), Parakeet v1/v2 (superseded by v3), Distil-Whisper (EN-only), Phi-4-Multimodal (footprint/tooling), Seamless M4T v2 (heavier, weaker tooling than two-stage), specialized disfluency models (worse ES coverage than the LLM pass), FluidAudio CustomVocabularyContext (disproven 2026-07-24, §7.2.1).

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
| 2026-08-20 | agentic_light | qwen35_4b_nothink (qwen3.5-4b-nothink) | keep | Gemma 3 4B (Catalan niche) |
| 2026-08-20 | agentic_heavy | gemma4_26b (gemma4-26b-a4b-it) | keep | Qwen 3.6 35B-A3B (tie) |
| 2026-08-20 | audio_transcribe | parakeet (parakeet-tdt-0.6b-v3) + whisper fallback | watch | — (no domain-tested alternative) |
| 2026-08-20 | audio_translate | whisper_translate (whisper-medium) | watch | two-stage Turbo → Gemma 4 26B |

Reading the progression: five consecutive runs now (2026-05-10 through 2026-08-20) have kept `agentic_heavy` on the Gemma 4 26B / Qwen 3.6 35B-A3B tie despite four genuinely new entrants across that span (Qwen 3.6 35B-A3B itself, DeepSeek V4 Flash, Qwen3.8-27B, GLM-5.3) — every one dominated by architecture (MoE beats dense at this bandwidth) or by size (nothing else fits). `agentic_light`'s incumbent id changed this run (`qwen35_4b` → `qwen35_4b_nothink`) but that's a routing default flip (#489), not a new frontier pick — the underlying model is unchanged since 2026-05-10. `audio_transcribe` stays `watch` for the second consecutive run with no candidate fix in sight; `audio_translate` has been stable since 2026-05-10.

---

## 9. Open questions / uncertainty

- **No fix currently exists for Parakeet's wake-phrase/jargon gap.** FluidAudio's only vocabulary mechanism (post-hoc CTC rescoring) is structurally incapable of it (§7.2.1). The re-open trigger is a FluidAudio release with decode-time TDT biasing, or an alternative checkpoint exposing one.
- **Qwen3-ASR has no domain-specific (jargon-heavy dictation) WER comparison**, and its most obvious local integration path (FluidAudio's Swift backend) was removed upstream this window (#676). If a future FluidAudio release re-adds it, or a separate MLX-based runner becomes practical on the Mac Mini, worth a smoke test against the #138/#343 corpus before trusting any generic-benchmark number.
- **DeepSeek V4 Flash's composite quality on non-coding workloads is still unverified**, and its vendor-claimed "DSpark" 2× decode acceleration is unverified locally. Now that the 3-bit GGUF fits this box's combined budget, a smoke test is cheap enough to be worth doing next window if idle capacity allows — not urgent, since even the optimistic case doesn't clearly beat Qwen3 32B dense on the frontier.
- **GLM-5.5's August release is still a rumor** (JPMorgan research note, 2026-06-25) — four consecutive runs now with no model card, benchmark, or endpoint. Watch for an actual release next cycle.
- **A community report of low Gemma-4-class token generation on RTX 5060 Ti Blackwell doesn't apply to this box's own launch config** (§3 step 4/§6) — but a local IQ3_XXS vs IQ4_XS A/B for `gemma4_26b` (reported ~99 vs ~85 t/s in that community writeup) is a cheap speed idea worth testing, independent of any frontier verdict.
- Standing carries: Gemma 4 12B Unified license still unverified on the model card; Catalan on Qwen 3.6 35B-A3B still needs a local smoke test; MiniMax M3 at UD-Q2 still academic.

---

## 10. Current decisions (live, edited by `/swap-model`)

The decisions below mirror `config/models.yaml` → `roles:` at the time
this section was last updated. `/swap-model` rewrites both this section
and the yaml together, so the two stay in sync.

| Role | Model | Decided | Why |
|---|---|---|---|
| **agentic_light** | `qwen35_4b_nothink` (qwen3.5-4b-nothink) | 2026-05-10 (model); 2026-08-12 (routing default, #489) | Tier A top pick since 2026-05-10 — hybrid Gated DeltaNet + sparse MoE on a 4B base, Q4_K_M ~3 GB, 262k native ctx, 201 languages, Apache 2.0. The role pointer moved to the no-think virtual alias on 2026-08-12 (same underlying weights); the thinking variant stays one alias away as `agentic_light_think`. |
| **agentic_heavy** | `gemma4_26b` (gemma4-26b-a4b-it) | 2026-05-10 | Tier B top pick. 99 t/s, 256k ctx, strong multilingual including Catalan. Tied with Qwen 3.6 35B-A3B (Apache 2.0) — Gemma stays default on Catalan track record. |
| **audio_transcribe** | `parakeet` (parakeet-tdt-0.6b-v3, Mac ANE) with `fallback: [whisper]` | 2026-07-22 | Changed directly via #350 (fleet placement/latency, #343 benchmark), not `/swap-model`. A speed-over-accuracy trade with known regressions (dropped wake phrase, jargon mangling). The identified remediation (FluidAudio custom-vocabulary) was tried and disproven (#401) — no current fix path; see §7. |
| **audio_translate** | `whisper_translate` (whisper-medium, lazy CPU) | 2026-05-10 | Strict frontier reading recommends `watch` — the two-stage path (Turbo → Gemma 4 26B) is the default, leaving this slot as a fallback only. Keep defined and lazy-loaded; no active maintenance. |

---

*Generated by the `/frontier-refresh` skill (`.claude/skills/frontier-refresh/SKILL.md`), which owns the research brief and this report's output contract. This is the August 20, 2026 snapshot.*
