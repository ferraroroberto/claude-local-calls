# Local LLM + ASR Efficient Frontier — Results

**Run date:** 2026-09-03
**Hardware:** RTX 5060 Ti 16 GB · Ryzen 7 7800X3D · 128 GB DDR5
**Workloads:** OpenClaw agentic (fast + deep lanes), transcript polishing, document processing, EN↔ES↔CA translation, **audio transcription EN/ES, audio translation ES → EN, transcript disfluency removal**. No coding.

---

## 0. Verdict

| role | incumbent | verdict | best alternative | gap | reason |
|------|-----------|---------|-------------------|-----|--------|
| `agentic_light` | `qwen35_4b_nothink` (qwen3.5-4b-nothink) | keep | Gemma 3 4B (Catalan niche) | — | No new 4B-class entrant this window either. Nemotron and Qwen's own releases this window both landed in Tier B/C, not Tier A |
| `agentic_heavy` | `gemma4_26b` (gemma4-26b-a4b-it) | keep | Qwen 3.6 35B-A3B | tie (≤3%) | Tie persists. Two genuine new entrants this window — Qwen3.8-Flash-Next (180B/6B active, shipped 2026-08-26) and NVIDIA Nemotron 3.5 Lightning (30B/3B active, shipped 2026-08-11, missed by the last run) — are both evaluated and dominated: Flash-Next's SSD-paged n-gram table caps it at dense-27B-class speed despite the MoE label, and Lightning doesn't fit 16 GB VRAM cleanly, has no measured tok/s on comparable hardware, and inherits the Nemotron family's unverified-to-risky Catalan coverage (#486) — see §5 |
| `audio_transcribe` | `parakeet` (parakeet-tdt-0.6b-v3, Mac ANE) + whisper fallback | watch | Granite Speech 4.1 2B (candidate, untested) | — | No verified fix this window, but the most promising lead in months surfaced: Granite Speech 4.1 2B's keyword-list biasing operates at decode time via the LLM's prompt interface — structurally different from the post-hoc CTC rescoring that failed in #401. Unverified on this project's jargon corpus; recommend a local smoke test before any verdict change — see §7 |
| `audio_translate` | `whisper_translate` (whisper-medium) | watch | two-stage: Turbo → Gemma 4 26B MoE | — | Unchanged; two-stage stays the default, this slot stays a lazy fallback. Granite Speech 4.1 2B's native ES↔EN speech translation is a candidate for the single-model fallback leg (architecture i) if ever promoted, but no change to the default this run |

**Diff vs previous run (2026-08-20):** no role changes — all four verdicts hold. Two things moved this window (2026-08-20 → 2026-09-03). First, on the text side: Alibaba shipped **Qwen3.8-Flash-Next** (2026-08-26, 180B total / 6B active, previewing the Qwen4 architecture) and this run also caught **NVIDIA Nemotron 3.5 Lightning** (30B/3B active hybrid Mamba-2+MoE, shipped 2026-08-11) — a genuine sourcing gap in the previous run, the same kind of miss #486 already flagged for the smaller Nemotron 3 Nano 4B. Both are evaluated in §5 and neither clears the bar: Flash-Next's headline MoE efficiency is undercut by its 51B-parameter n-gram table paging from SSD, landing its real decode speed in dense-27B territory (~17-23 t/s) despite 6B active parameters; Lightning doesn't fit 16 GB VRAM at any published quant, has no tok/s measurement on comparable 16 GB-class hardware, ships under a non-Apache license (OpenMDW-1.1), and — most importantly for this stack's 25%-weighted multilingual axis — its documented language list (EN/ES/FR/DE/IT/JA) omits Catalan the same way the disproven Nemotron 3 Nano 4B did, an unverified but real risk given that family's track record (#486). Second, and more consequential for the audio side: **Granite Speech 4.1 2B** (IBM, actually released 2026-04-30 but newly surfaced this run — see §9 sourcing note) now leads the HF Open ASR Leaderboard's public track at 5.33% WER and, critically, implements keyword-list biasing as a **decode-time** mechanism through its LLM prompt interface — not the post-hoc CTC rescoring that structurally cannot insert a dropped phrase and that sank the FluidAudio fix in #401. It has not been tested against this project's own jargon corpus, so `audio_transcribe` stays `watch`, not `upgrade` — but it is the first candidate since Parakeet's own switch that plausibly satisfies the local-findings re-open trigger, and a local smoke test is the recommended next step (§7.2.1, §9). `agentic_light` and `audio_translate` are unchanged in substance.

*Why this is the core artifact:* everything below exists to justify these six columns. If you read nothing else, this table plus the diff line is the run.

---

## 1. Objective

The "efficient frontier" of local LLMs is the set of models where, for a given level of quality, no other model is faster (or, for a given speed, no other model is more accurate). Everything off the frontier is **dominated** — a strictly better choice exists on at least one axis without giving up the other.

The frontier is **always hardware- and workload-specific**: a 70B that dominates on a 5090 falls off the 5060 Ti's frontier into CPU-offload territory, and a coding-specialist that wins SWE-bench is irrelevant here because coding carries 0% weight. This report identifies the frontier for *this* box and *these* workloads, as of September 2026.

**What changed since 2026-08-20:** Two text-side entrants, both dominated. Qwen3.8-Flash-Next (2026-08-26) is Alibaba's preview of the next-generation Qwen4 architecture: 125B language-model parameters plus a 51B n-gram embedding table and a 4B multi-token-prediction layer, of which only 6B activate per token. On paper that is an extremely sparse, extremely fast MoE. In practice, the 51B n-gram table is too large to keep resident and pages from SSD on every request, so measured decode speed lands in the same 17-23 t/s range as a dense 27B model — the efficiency the architecture promises never reaches the token stream. GGUF builds run 84.9-110.5 GB, which fits this box's ~144 GB combined budget, but at that speed it is dominated outright by both Tier B MoE incumbents (Gemma 4 26B MoE, Qwen 3.6 35B-A3B), which decode 4-5x faster at a third the footprint. License is Qwen Community License 1.0, not Apache 2.0 — more permissive than a research-only license but a step down from the incumbents' license clarity.

NVIDIA's Nemotron 3.5 Lightning (2026-08-11) is architecturally more interesting — a genuine hybrid of Mamba-2 state-space layers, MoE layers, and a handful of full-attention layers (23/23/6 of 52 total), 30B total / 3B active, explicitly built for the "agent execution layer" and trained against OpenClaw and Hermes Agent harnesses. It missed the 2026-08-20 report entirely — the same sourcing gap #486 already called out for the Nemotron 3 Nano 4B, now repeating one size class up. Worked out in §4, it doesn't clear the bar: no published GGUF quant fits under 16 GB VRAM without CPU-offloading the MoE experts (a pattern this repo already uses for `glm`), no tok/s measurement exists on 16 GB-class hardware (the one public benchmark used an RTX 5090 with headroom this box doesn't have), it ships under NVIDIA's OpenMDW-1.1 license rather than Apache 2.0, and its documented language support (EN/ES/FR/DE/IT/JA) does not list Catalan — the same gap that sank its smaller Nemotron 3 Nano 4B sibling on this exact axis (#486, `local-findings.md`). That disproof was Catalan-specific, not Lightning-specific, so this is a flagged risk rather than a re-triggered `local-findings.md` override, but it is reason enough not to recommend a swap without a local test.

GLM-5.5 remains an unconfirmed rumor (still no model card, benchmark, or endpoint five consecutive runs running). DeepSeek V4, Kimi K3, and Mistral's "fat but sparse" MoE carry forward unchanged — no updates surfaced this window.

On the audio side, the real news: **Granite Speech 4.1 2B** climbed to the top of the HF Open ASR Leaderboard's public track (5.33% mean WER, beating the 8B-class Granite Speech 3.3 this report previously carried at 5.85%) while shrinking from 8B to 2B parameters, and — the part that matters for this project's specific, previously-intractable problem — its keyword-list biasing is implemented as prompt-conditioned decode-time steering through its LLM decoder, not the post-hoc CTC rescoring that FluidAudio's `CustomVocabularyContext` uses and that #401 proved structurally incapable of inserting a phrase the acoustic model dropped entirely. This is evaluated in full in §7.

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

1. Read `docs/frontier/runs/LATEST` (2026-08-20), that run's `report.md` + `frontier.json`, and `docs/frontier/local-findings.md` (#277) — three unresolved entries carry forward: the faster-whisper CTranslate2 disproof (2026-07-12), the FluidAudio `CustomVocabularyContext` disproof (2026-07-24), and the Nemotron 3 Nano 4B quality disproof (2026-08-10, #486). None of this run's new candidates is the literal disproven candidate (Nemotron 3.5 Lightning is a different model from Nemotron 3 Nano 4B; Granite Speech 4.1 2B's biasing mechanism is architecturally distinct from FluidAudio's rescorer), so the override rule wasn't triggered for either — but both disproofs' underlying lessons (Nemotron family Catalan risk; post-hoc-vs-decode-time biasing) directly shaped how this run's candidates were scored.
2. Re-read `config/models.yaml` → `roles:` for the current incumbents. No out-of-band changes since the last run: `agentic_light` (qwen35_4b_nothink), `agentic_heavy` (gemma4_26b), `audio.transcribe` (parakeet + whisper fallback), and `audio.translate` (whisper_translate) are all unchanged.
3. Surveyed the external landscape for the 2026-08-20 → 2026-09-03 window: Qwen3.8-Flash-Next's architecture and measured speed, Nemotron 3.5 Lightning's architecture/license/GGUF footprint/multilingual coverage/llama.cpp support status (a sourcing-gap catch for a 2026-08-11 release), GLM-5.5's rumor status, Mistral's late-August releases (Medium 3.5 unchanged, Leanstral 1.5 and OCR 4.1 both out of scope), and Granite/Falcon/OLMo/Hermes/GPT-OSS/Phi/Gemma for anything new (none found this window).
4. On the audio side: checked FluidAudio's releases since 2026-08-20 (v0.15.5/0.15.6 added "custom vocabulary controls" — per-term CTC thresholds and opt-in knobs, still built on the same post-hoc CTC rescoring mechanism disproven in #401, not a decode-time biasing hook — the local-findings re-open trigger is not met), the HF Open ASR Leaderboard's current public-track ranking (Granite Speech 4.1 2B now leads at 5.33% WER), and Granite Speech 4.1 2B's biasing architecture specifically (confirmed as prompt-conditioned decode-time steering through the LLM decoder, distinct from CTC rescoring).
5. Computed VRAM with the standing rule of thumb **Q4_K_M ≈ 4.5 bits/param** plus KV-cache where no published GGUF size exists; used Nemotron 3.5 Lightning's actual published GGUF sizes (Q4_0 ≈ 18.9 GB + 1.16 GB MTP, Q4_K_M ≈ 25.5 GB) instead of the rule of thumb now that they exist. Worked the MoE-CPU-offload math fresh for Lightning (§4) since it is the window's architecturally novel entrant.
6. Applied the honesty rules: date-stamped claims, ≤3% composite = tie (the Gemma 4 26B / Qwen 3.6 35B-A3B tie stands), licenses surfaced (Qwen Community License 1.0 for Flash-Next, OpenMDW-1.1 for Lightning — both non-Apache), Lightning's Catalan gap and Flash-Next's real-world speed both flagged as risks rather than folded silently into a quality estimate, Granite Speech 4.1 2B's biasing mechanism explicitly *not* upgraded to a verdict change without a local domain test, consistent with this project's two prior ASR-benchmark disproofs.

---

## 4. How to read the chart

- **X axis** — estimated single-stream tokens/second on the 5060 Ti at the recommended quant.
- **Y axis** — composite quality score for *these* workloads (0–100, normalized).
- **Bubble size** — VRAM at recommended quant. **Color** — tier (A fast / B balanced / C quality).
- **Filled border** — on the Pareto frontier. **Hollow** — dominated.
- **Toggle** — show only models that fit fully in 16 GB VRAM, or include CPU-offload models.

### Worked memory example (so the math isn't a black box)

This run's cautionary tale is **NVIDIA Nemotron 3.5 Lightning 30B-A3B** — architecturally the closest thing to a genuine Tier B challenger this year, undone by fit and verification, not concept:

```
30B total params, 3B active (hybrid: 23 Mamba-2 + 23 MoE + 6 attention layers, 52 total)
Published GGUF: Q4_0 ≈ 18.9 GB + 1.16 GB MTP ≈ 20.1 GB; Q4_K_M ≈ 25.5 GB
This box: 16 GB VRAM ceiling — neither quant fits as a monolithic GPU load
MoE-CPU-offload option (same `-ot .ffn_.*_exps.=CPU` pattern this repo already
  uses for `glm`): attention + Mamba layers stay on GPU, MoE experts spill to
  system RAM, only the ~3B active experts' worth of bytes move per token —
  but no published or community measurement exists for this exact
  offload split on any 16 GB-class card
One public benchmark exists: 123 t/s on an RTX 5090 (32 GB) at 40K context —
  a card with 2x this box's VRAM and no need to offload experts at all, so
  the number doesn't transfer
Multilingual: NVIDIA's own post-training recipe explicitly lists EN, ES, FR,
  DE, IT, JA — Catalan is absent, the same axis where the smaller Nemotron 3
  Nano 4B sibling was measured and DISPROVEN for this exact role family (#486)
```

Compare **Gemma 4 26B MoE** (the incumbent, same active-parameter class): ~14 GB fits as a monolithic GPU load with no offload tuning required, 99 t/s measured, Catalan-verified across five consecutive runs. Lightning's architecture (hybrid Mamba+MoE, agent-tuned) is a genuinely interesting shape — but "interesting shape" without a fitting quant, a same-class speed measurement, or a language-coverage answer isn't a frontier entry, it's a research lead. That's the same conclusion the 2026-08-06 report drew from DeepSeek V4 Flash and the 2026-08-20 report drew from Qwen3.8-27B, arrived at from a third distinct direction: right active-parameter budget, wrong verification state.

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

No change this run. Both of the window's new text entrants (Qwen3.8-Flash-Next, Nemotron 3.5 Lightning) landed in Tier B/C by size class — no 4B-class challenger shipped.

### Tier B — Balanced (the workhorse) — **TIED, unchanged**

| model | params | quant | VRAM | tok/s | quality | ctx | license | on frontier |
|-------|--------|-------|------|-------|---------|-----|---------|-------------|
| ★★ Gemma 4 26B MoE *(incumbent)* | 26B / 4B active | native W4 | ~14 GB | 99 | 83 | 256k | Gemma | yes |
| ★★ Qwen 3.6 35B-A3B | 35B / 3B active | Q4_K_M | ~13.5 GB | 98 | 84 | 262k | Apache 2.0 | yes |
| ☆ GPT-OSS 20B | 21B / 3.6B active | MXFP4 | ~12 GB | ~100 | 72 | 131k | Apache 2.0 | yes |
| Nemotron 3.5 Lightning *(new)* | 30B / 3B active | Q4_0 (+MTP) | ~20.1 GB | ~40 est | 75 est | 128k | OpenMDW-1.1 | no |
| Qwen3.8-27B | 27.78B dense | Q4_K_M | ~15.6 GB | ~28 est | 79 est | 262k (1M YaRN) | Apache 2.0 | no |
| Gemma 4 12B Unified | 12B dense | Q4_K_M | ~7 GB | ~45 est | 76 est | 256k | Apache 2.0 (verify) | no |
| Ministral 3 14B | 14B dense | Q4_K_M | ~8.5 GB | ~35 est | 70 est | 256k | Apache 2.0 | no |
| Mistral Small 3.2 | ~22B dense | Q4_K_M | ~13 GB | ~30 | 74 | 128k | Apache 2.0 | no |

The tie at the top persists. **Nemotron 3.5 Lightning added** — the window's genuinely novel-architecture entrant (a sourcing-gap catch from 2026-08-11, see §1), but doesn't fit 16 GB VRAM at any published quant without an unverified MoE-CPU-offload split, has no tok/s measurement on comparable hardware (§4's worked example), ships under a non-Apache license, and carries an unverified Catalan-coverage risk inherited from its disproven Nemotron 3 Nano 4B sibling (#486). Quality estimate (75) is conservative pending any local test. **Qwen3.8-27B** (carried from last run) remains dominated on the same speed/architecture grounds.

### Tier C — Quality (slow, CPU-offload, batch / non-interactive)

| model | params | quant | VRAM | tok/s | quality | ctx | license | on frontier |
|-------|--------|-------|------|-------|---------|-----|---------|-------------|
| ★ Qwen3 32B dense | 32B | Q4_K_M | ~19.5 GB (spill) | ~11 | 84 | 128k | Apache 2.0 | yes |
| Qwen3.8-Flash-Next *(new)* | 180B / 6B active | GGUF (smallest) | ~85 GB | ~20 est | 78 est | 262k (1M YaRN) | Qwen Community 1.0 | no |
| Llama 3.3 70B | 70B | Q4_K_M | ~40 GB (offload) | ~4 | 82 | 128k | Llama 3.3 | no |
| Mistral Medium 3.5 | 128B dense | Q4_K_M | ~75 GB (offload) | ~2 | 86 | 256k | Modified MIT | no |
| DeepSeek V4 Flash | 284B / 13B active | GGUF 3-bit (native FP4/FP8) | ~103 GB (offload) | ~10-20 est | unverified | 1M (reported) | MIT | no |

No change to the frontier. **Qwen3.8-Flash-Next added** — its headline MoE efficiency (6B active of 180B total) is undercut in practice by a 51B-parameter n-gram embedding table that pages from SSD on every request, so measured real-world decode speed (17-23 t/s in community testing) lands in dense-27B territory despite the sparse-MoE label — dominated by Qwen3 32B dense on speed at a much larger footprint, and by both Tier B MoE leaders by a wide margin.

### Models considered and dropped this run

- **Qwen3.8-Flash-Next (180B total [125B LM + 51B n-gram + 4B MTP] / 6B active, Qwen Community License 1.0, shipped 2026-08-26)** — previewing the Qwen4 architecture. The n-gram table's SSD paging caps real decode speed at dense-27B-class levels (~17-23 t/s) despite the 6B-active MoE design — dominated on speed by every Tier B/C entry that actually fits its own footprint, and by Qwen3 32B dense at a fraction of the size. Evaluated in Tier C above.
- **NVIDIA Nemotron 3.5 Lightning (30B/3B active, hybrid Mamba-2+MoE, OpenMDW-1.1, shipped 2026-08-11)** — a genuine sourcing-gap catch (missed the 2026-08-20 run). Doesn't fit 16 GB VRAM at any published quant, no tok/s measurement exists on comparable 16 GB-class hardware, non-Apache license, and its documented language list (EN/ES/FR/DE/IT/JA) omits Catalan — the same axis where the smaller Nemotron 3 Nano 4B sibling was locally disproven (#486). Evaluated in Tier B above and in §4's worked example. Worth a cheap local smoke test given the architectural interest, but not recommended without one.
- **GLM-5.5** — still rumored (JPMorgan research note, 2026-06-25, unchanged from the last four runs); no model card, benchmark, or endpoint published as of this run. Watch next cycle.
- **DeepSeek V4 Pro / DeepSeek V4 Flash / Qwen3.8 2.4T-A95B / Kimi K3 / GLM-5.2+5.3** — no updates this window; standing NO-GOs carried unchanged.
- **Mistral's late-August releases (Medium 3.5 unchanged, Leanstral 1.5, OCR 4.1)** — Leanstral is a Lean-4 formal-proof specialist (0% weight, coding-adjacent) and OCR 4.1 is a document-OCR model, not a general LLM; neither is a Tier A/B/C candidate for this stack's roles.
- Standing drops carried from prior runs: Qwen 3.6 27B dense, Gemma 3 27B, Qwen3.5-35B-A3B, Mistral Small 3.2, Llama 3.2 3B, Phi-4 Mini (for ES/CA), Qwen3 8B/9B class, MiniMax M3 (still academic at 2-bit), Nemotron 3 Nano 4B (quality-disproven locally, #486 — see `local-findings.md`).

---

## 6. Concurrency plan

Unchanged from the previous run — the four recipes still describe the practical envelope:

1. **Two lanes (default):** Qwen 3.5 4B (GPU ~3 GB) + Gemma 4 26B MoE (GPU ~14 GB). Both near-peak; ~17 GB with graceful shared-memory spill.
2. **Qwen 3.6 stack (all-Apache):** Qwen 3.5 4B + Qwen 3.6 35B-A3B (~13.5 GB). Speed parity; license clarity.
3. **Quality batch:** Qwen 3.5 4B + Qwen3 32B dense (~3.5 GB CPU spill, ~10 t/s) for overnight reprocessing.
4. **Three concurrent:** Qwen 3.5 4B + GPT-OSS 20B (GPU) + Gemma 3 4B (CPU, ~10 t/s on the 7800X3D) as a Catalan specialist.

No placement changes this window. Nemotron 3.5 Lightning's MoE-CPU-offload shape (§4) would, if ever locally verified, slot into a fifth recipe alongside the fast lane — noted in §9 as a future idea, not adopted this run.

---

## 7. Audio (ASR) annex — workloads F, G, H

### 7.1 The landscape in September 2026 — the first structurally-different biasing candidate since Parakeet's own switch

FluidAudio shipped v0.15.5/v0.15.6 this window, adding "custom vocabulary controls" (upstream issues #647/#702/#724) — per-term CTC thresholds and opt-in knobs to curb short-term over-firing. This is a refinement of the **same** post-hoc CTC rescoring mechanism `CustomVocabularyContext` already uses, not a new decode-time biasing hook — the local-findings re-open trigger (a release that adds decode-time TDT vocabulary *biasing*, not post-hoc rescoring) is **not met**. The #401 disproof stands unresolved.

The genuinely new development is external to FluidAudio entirely: **Granite Speech 4.1 2B** (IBM, Apache 2.0) now leads the HF Open ASR Leaderboard's public track at 5.33% mean WER — beating the 8B-class Granite Speech 3.3 this report previously carried at 5.85%, while shrinking the model by 4x. Its architecture is a Conformer-CTC speech encoder feeding a Q-Former projector into an LLM decoder, and its keyword-list biasing (pass names/acronyms/jargon terms in the prompt) is implemented as **prompt-conditioned decode-time steering through that LLM decoder** — evaluated by IBM with-vs-without keyword biasing at inference time, not via a post-training fine-tune. This is structurally the opposite failure mode from what sank the FluidAudio fix in #401: an LLM decoder conditioned on a keyword list can, in principle, *generate* a token sequence it would not otherwise have produced, rather than only *rescoring* tokens the acoustic model already emitted. Whether that principle survives contact with this project's specific failure modes ("Claude Code" heard as "Yes"/"Yeah", "YOLO" heard as "yellow") is untested — no comparison against the #138/#343 jargon corpus exists yet. **Verdict stays `watch`, not `upgrade`** — this project has been burned twice on published-benchmark ASR claims that didn't survive local domain testing (faster-whisper's 2x claim, #274; Parakeet's own leaderboard numbers, §7.2.1) and a third promise-without-measurement isn't grounds for a role change. But it is the first candidate in this report's history whose *mechanism* is the right shape for the problem, not just its benchmark score — see §9 for the recommended next step.

### 7.2 ASR candidate comparison (EN + ES)

| Variant | Params | VRAM | RTFx (measured/est.) | EN | ES | Translates → EN? | Notes |
|---------|--------|------|----------------------|----|----|-------------------|-------|
| **★ Parakeet TDT v3** (Mac ANE, current primary) | 0.6B | ~2 GB (CoreML) | 65.8× measured (ANE), sub-second even on 108 s clips | ✅ | ✅ (25 langs) | ❌ | Unchanged incumbent since 2026-07-22. Published generic Spanish WER modestly beats Whisper — but this project's own jargon-heavy dictation domain measures worse (§7.2.1), and the identified fix for that gap is disproven (#401). |
| **◆ Whisper Large v3 Turbo** (automatic failover) | 809M | ~1.6 GB | 40× tower / 19.3× gaming (measured, boosted) | ✅ | ✅ | ❌ | Unchanged. Still the accuracy leader on this domain thanks to the `--carry-initial-prompt` boosting glossary (#91) — a lever Parakeet structurally lacks. |
| Granite Speech 4.1 2B *(new lead, untested)* | 2B | ~2 GB est | TBD (domain) | ✅ | ✅ | ✅ (bidirectional AST, 6 langs) | Apache 2.0. Now leads the public HF Open ASR Leaderboard (5.33% WER), replacing the 8B-class 3.3 generation this report previously carried. Decode-time prompt-conditioned keyword biasing — architecturally distinct from Parakeet's post-hoc CTC rescorer. Untested against the #138/#343 jargon corpus; recommended next-step smoke test (§9). |
| ✗ Parakeet + FluidAudio CustomVocabularyContext | 0.6B + ~97 MB CTC model | ~2 GB + rescorer | n/a — disproven, not shipped | — | — | ❌ | Disproven #401 (2026-07-24). Carried `watch` per `local-findings.md`. FluidAudio's 2026-09 "custom vocabulary controls" (v0.15.5) refine the same post-hoc mechanism — re-open trigger not met. |
| faster-whisper Turbo (CT2) | 809M (CT2) | ~1.0 GB INT8 | measured 1.0× vs whisper.cpp — **disproven** | ✅ | ✅ | ❌ | Carried `watch` per `docs/frontier/local-findings.md` (#277); applies to the failover leg's engine, not the primary. |
| Whisper Large v3 (faster-whisper) | 1.55B | ~2 GB | 30–50× | ✅✅ | ✅✅ | ✅ | Workload-G single-model fallback, unchanged. |
| Qwen3-ASR (1.7B / MLX build) | 1.7B | ~1.5 GB | TBD (domain) | ✅ | ✅ | ❌ | `watch`, unchanged status. FluidAudio still has no Qwen3-ASR backend (#676, removed) and no jargon-domain comparison exists. |

#### 7.2.1 Why the identified fix failed — and why the new lead might not repeat that failure

Unchanged root-cause analysis from prior runs: FluidAudio's only vocabulary mechanism for Parakeet TDT is a **post-hoc CTC rescorer** — it re-scores already-emitted transcript words against per-frame log-probabilities and swaps in a boosted term when it has stronger acoustic evidence. There is no decode-time biasing hook for the TDT architecture — nothing that can make the decoder consider emitting a phrase it didn't already emit. That's why "Claude Code" isn't mis-transcribed as something similar, it's dropped entirely (heard as "Yes"/"Yeah") — there's no candidate word for the rescorer to swap **from**. Granite Speech 4.1 2B's mechanism is different in kind, not just degree: it's an LLM decoder whose output distribution is conditioned on a keyword list *before* generation starts, the same category of technique (prompt-conditioned generation) already proven to work in this repo's own whisper.cpp `--carry-initial-prompt` boosting (#91) — except delivered through a full LLM decoder rather than a carried text prompt fed into a CTC/attention hybrid. Whether Granite's specific implementation actually recovers dropped short words or wake phrases as well as whisper.cpp's boosting does is the open, untested question — the mechanism being the right *shape* is necessary, not sufficient.

### 7.3 Workload F — transcribe EN/ES

**Verdict: `watch`.** No role change — `parakeet` stays primary with `whisper` as automatic failover, unchanged since 2026-07-22 (#350/#348). **Explicit answer to the brief's required question: no strict, verified upgrade over the incumbent transcribe model exists for EN/ES this run.** Granite Speech 4.1 2B is a structurally promising *candidate* (§7.1/§7.2.1) but is untested on this project's domain — recommending it now would repeat the exact mistake #274 and #401 already taught this project not to make. FluidAudio's v0.15.5/0.15.6 vocabulary-control refinements don't meet the local-findings re-open trigger (still post-hoc rescoring). Qwen3-ASR and the whisper failover's own `faster-whisper` engine swap stay `watch`/disproven per `local-findings.md`, unrelated to this verdict.

### 7.4 Workload G — ES audio → English

**Verdict unchanged: two-stage default.** faster-whisper Turbo transcribes ES, Gemma 4 26B MoE translates + polishes + de-disfluences in one call (~15 GB total). Single-model faster-whisper Large v3 `task=translate` stays the fallback when the LLM slot is busy — implemented today by the `whisper_translate` role slot (whisper-medium, lazy CPU), hence its `watch` verdict rather than retirement. Granite Speech 4.1 2B's native bidirectional ES↔EN speech translation (Apache 2.0, ~2 GB) is a candidate to eventually replace the single-model fallback leg's engine — smaller and higher-leaderboard-rank than the current `whisper_translate` (whisper-medium) — but no change to the default this run; note added to §9.

### 7.5 Workload H — disfluency / filler removal

**Verdict unchanged: folded into the LLM polishing pass.** Specialized disfluency models remain research-grade with sparse Spanish coverage; the tier-B LLM already does this work in the polish prompt (the canonical prompt lives in `frontier.json` → `disfluency_verdict.prompt`).

### 7.6 Concurrency footprint

Unchanged from the previous run: whisper-turbo (accurate dictation) and whisper-vanilla/translate run on the gaming satellite, orpheus (expressive TTS) is on the tower, freeing the tower's GPU for agentic_heavy + agentic_light exclusively (#343/#422). Parakeet runs on the Mac Mini's ANE, its own dedicated accelerator, adding zero contention with either box. Piper (fast TTS) stays on the tower CPU. Granite Speech 4.1 2B, if tested, would need its own GPU/CPU allocation — not yet placed anywhere.

### 7.7 Dropped

Carried from prior runs: Canary-Qwen (EN-only), Parakeet v1/v2 (superseded by v3), Distil-Whisper (EN-only), Phi-4-Multimodal (footprint/tooling), Seamless M4T v2 (heavier, weaker tooling than two-stage), specialized disfluency models (worse ES coverage than the LLM pass), FluidAudio CustomVocabularyContext (disproven 2026-07-24, §7.2.1), Granite Speech 3.3 8B (superseded within its own family by 4.1 2B — smaller and higher-ranked, see §7.2).

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
| 2026-09-03 | agentic_light | qwen35_4b_nothink (qwen3.5-4b-nothink) | keep | Gemma 3 4B (Catalan niche) |
| 2026-09-03 | agentic_heavy | gemma4_26b (gemma4-26b-a4b-it) | keep | Qwen 3.6 35B-A3B (tie) |
| 2026-09-03 | audio_transcribe | parakeet (parakeet-tdt-0.6b-v3) + whisper fallback | watch | Granite Speech 4.1 2B (candidate, untested) |
| 2026-09-03 | audio_translate | whisper_translate (whisper-medium) | watch | two-stage Turbo → Gemma 4 26B |

Reading the progression: six consecutive runs now (2026-05-10 through 2026-09-03) have kept `agentic_heavy` on the Gemma 4 26B / Qwen 3.6 35B-A3B tie despite six genuinely new entrants across that span (Qwen 3.6 35B-A3B itself, DeepSeek V4 Flash, Qwen3.8-27B, GLM-5.3, Qwen3.8-Flash-Next, Nemotron 3.5 Lightning) — every one dominated by architecture (MoE beats dense, or MoE-in-name-only loses to MoE-in-practice at this bandwidth) or by size/fit (nothing else clears 16 GB cleanly). `agentic_light` has been unchanged in substance since 2026-05-10 across seven runs. `audio_transcribe` has held `watch` for the third consecutive run, but this run's `best alternative` column changes for the first time since 2026-07-24 — not because anything was proven, but because a structurally-different candidate (decode-time biasing vs. post-hoc rescoring) finally surfaced. `audio_translate` has been stable since 2026-05-10.

---

## 9. Open questions / uncertainty

- **Granite Speech 4.1 2B is the recommended next local smoke test for `audio_transcribe`.** Test its keyword-list biasing against the #138/#343 jargon corpus (the "Claude Code" wake-phrase and "YOLO"/"yellow" cases specifically) before any verdict change — its decode-time mechanism is the right shape, but this project has twice recommended-then-disproven ASR claims that didn't survive contact with this exact corpus (#274, #401), so no verdict moves until it's measured here.
- **Nemotron 3.5 Lightning is architecturally the most interesting `agentic_heavy` challenger in months but has three unresolved gaps**: no fitting quant under 16 GB VRAM without an unverified MoE-CPU-offload split, no tok/s measurement on comparable hardware, and an undocumented (likely absent) Catalan capability inherited as a risk from its disproven 4B sibling (#486). If idle capacity allows, a cheap first test is the `-ot .ffn_.*_exps.=CPU` offload split (already proven for `glm`) plus a Catalan smoke test using #486's own prompt set — cite `--category multilingual` if re-running that harness.
- **No fix currently exists for Parakeet's wake-phrase/jargon gap via FluidAudio itself.** Its only vocabulary mechanism (post-hoc CTC rescoring, refined but not architecturally changed in v0.15.5/0.15.6) remains structurally incapable of it (§7.2.1). The re-open trigger is a FluidAudio release with decode-time TDT biasing, or an alternative Parakeet checkpoint/architecture exposing one — Granite Speech 4.1 2B is the closest thing so far to the latter, though it is a different model family entirely, not a Parakeet checkpoint.
- **Qwen3-ASR still has no domain-specific (jargon-heavy dictation) WER comparison**, and FluidAudio's experimental backend remains removed (#676). Unchanged status.
- **DeepSeek V4 Flash's composite quality on non-coding workloads is still unverified** — carried unchanged from the last two runs; not urgent given it still doesn't clearly beat Qwen3 32B dense's already-verified profile.
- **GLM-5.5's release is still a rumor** (JPMorgan research note, 2026-06-25) — five consecutive runs now with no model card, benchmark, or endpoint.
- **Sourcing-gap pattern worth noting for future runs:** this is the second consecutive run to catch a relevant NVIDIA Nemotron release that had already shipped before the *previous* run's cutoff (Nemotron 3 Nano 4B was caught between runs via #486; Nemotron 3.5 Lightning was caught this run, nine days after it actually shipped). Worth explicitly searching "Nemotron" by name each run rather than relying on general "new open-weights model" queries to surface it.
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
| **audio_transcribe** | `parakeet` (parakeet-tdt-0.6b-v3, Mac ANE) with `fallback: [whisper]` | 2026-07-22 | Changed directly via #350 (fleet placement/latency, #343 benchmark), not `/swap-model`. A speed-over-accuracy trade with known regressions (dropped wake phrase, jargon mangling). The identified remediation (FluidAudio custom-vocabulary) was tried and disproven (#401) — no current fix path; Granite Speech 4.1 2B is a promising untested lead — see §7. |
| **audio_translate** | `whisper_translate` (whisper-medium, lazy CPU) | 2026-05-10 | Strict frontier reading recommends `watch` — the two-stage path (Turbo → Gemma 4 26B) is the default, leaving this slot as a fallback only. Keep defined and lazy-loaded; no active maintenance. |

---

*Generated by the `/frontier-refresh` skill (`.claude/skills/frontier-refresh/SKILL.md`), which owns the research brief and this report's output contract. This is the September 3, 2026 snapshot.*
