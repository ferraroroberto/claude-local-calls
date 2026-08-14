---
name: frontier-refresh
description: Recurring (bi-weekly) local-AI efficient-frontier research run — regenerate report.md + frontier.html + frontier.json under docs/frontier/runs/<today>/, repoint LATEST, post the verdict to the frontier ledger issue (#272). Read-only on config/models.yaml — to act on the new run, use /swap-model.
---

You are running the **local-AI frontier refresh** for this repo. You produce
artifacts and a ledger comment only — never modify `config/models.yaml`,
`launchers/`, `models/`, or `tray/`. Acting on the result is a separate,
human-in-the-loop step driven by `/swap-model`. This file is the **single
owner** of the whole process (research brief, cadence, quality weights, output
contract, scheduling) — there is no separate research-prompt document.

## Cadence & scheduling

**Bi-weekly** (every two weeks), stated here and nowhere else.

- Unattended: `run-weekly.bat` (next to this file) is registered as a weekly
  job in app-launcher's Jobs tab and **self-skips odd ISO weeks**, so the
  skill effectively fires on even ISO weeks only.
- On demand: run `/frontier-refresh` any time — it is read-only and
  re-runnable; a notable model launch is a good reason not to wait.

**Unattended constraint (fleet-config#314):** every step below must run
**synchronously** within the turn — a headless `claude -p` session has no
wake-up mechanism. Never background a tool call (web research, artifact
writing, `gh issue comment`, git push) and end the turn expecting to be
resumed; poll a long step to completion before moving on.

## Research brief

### Goal

Produce the **efficient frontier (Pareto front)** of currently-available
open-weights LLMs and ASR models for **local inference on the system below**,
optimized for *these* workloads — explicitly **no coding**.

### System

- **GPU:** NVIDIA RTX 5060 Ti, **16 GB VRAM** (Blackwell, FP4-capable)
- **CPU:** AMD Ryzen 7 7800X3D (8c/16t, 96 MB L3 — strong CPU-inference candidate)
- **RAM:** **128 GB DDR5** (huge CPU-offload headroom; running at 3600 MT/s)
- **Storage:** 2 TB NVMe (WD_BLACK SN850X) + 11 TB HDD
- **OS:** Windows 11 Pro
- **Runtimes:** llama.cpp / GGUF (primary, via this repo's llama-server
  launchers); Ollama / LM Studio / vLLM-under-WSL2 as references
- Also read `config/machine_specs.yaml` if present.

### Workloads (priority order)

Text (LLM):

| # | Use case | Latency tolerance | Quality bar |
|---|----------|-------------------|-------------|
| A | Fast lane — easy agentic steps, routing, simple tool calls, classification | very low | medium |
| B | Deep lane — hard agentic reasoning, multi-step tool use, planning | medium | high |
| C | Transcript cleanup & conciseness polishing (post-ASR, EN/ES/CA) | medium | high (style) |
| D | Document processing — extraction, summarization, restructuring | medium | high |
| E | Translation — primarily EN↔ES, EN↔CA | low–medium | high |

Audio (ASR):

| # | Use case | Latency tolerance | Quality bar |
|---|----------|-------------------|-------------|
| F | Transcription of EN, ES audio → text in source language | medium (batch OK) | high (low WER, accent-robust) |
| G | Audio translation: ES → EN | medium | high |
| H | Transcript polishing — filler/disfluency removal, conciseness | medium | medium |

For workload G evaluate both architectures: (i) single-model
speech-to-text-translation (Whisper-class `task=translate`), and (ii)
two-stage ASR → tier-B LLM translate+polish. For workload H briefly assess
dedicated disfluency models vs. the LLM polishing pass (nice-to-know, not
blocking). Always answer explicitly whether a **strict upgrade over the
incumbent transcribe model** exists for EN/ES.

**Explicitly out of scope:** code generation/editing/reasoning. Do **not**
weight coding benchmarks (HumanEval, MBPP, SWE-bench, LiveCodeBench, …).

### Concurrency assumption

2–3 models run simultaneously; CPU offload acceptable (128 GB RAM). Slot 1:
always-hot GPU-resident tier-A model. Slot 2: GPU or partial-offload
tier-B/C/D model. Slot 3: occasional CPU/spill quality model for batch runs.

### Quality weights

Composite quality score weighted by workload mix — **fixed**, changing them
is out of scope for a run:

- agentic / tool use: **0.35** (IFEval, BFCL v3+, τ-bench, MT-Bench, Arena-Hard)
- polish / writing / summarization: **0.25** (Arena writing/longform, RewardBench, human-eval reports)
- multilingual / translation: **0.25** (FLORES-200 EN↔ES; FLORES + community reports for EN↔CA — Catalan coverage is uneven, flag it)
- long context: **0.15** (native window and usable-context reports)

### Survey scope

- SOTA open-weights LLMs released or majorly updated in the last 90 days:
  Qwen, Llama, Mistral/Magistral/Ministral, Gemma, Phi, DeepSeek (non-coder),
  Command, Hermes/Nous, Yi, Mixtral, GLM, InternLM, Falcon, OLMo, GPT-OSS,
  Granite, plus newcomers.
- Per model: params (total/active for MoE), architecture, license (flag
  non-commercial/AUP restrictions), native context, common quantizations
  (GGUF Q4_K_M…Q8_0, AWQ, EXL2, FP8/FP4 if Blackwell-supported), release date.
- Memory budget on 16 GB VRAM + 128 GB RAM: VRAM at quantization, KV-cache
  at 8k/32k/native, verdict *fully-in-VRAM / partial offload / CPU-only*.
  Show the math at least once in the report.
- Speed (tok/s) on RTX 5060 Ti — triangulate from RTX 4070 / 4060 Ti 16 GB
  if 5060 Ti numbers are scarce; mark estimates.
- ASR: Whisper family (incl. faster-whisper/CT2, Distil, WhisperX), NVIDIA
  NeMo (Canary, Canary-Qwen, Parakeet TDT), IBM Granite Speech, Phi-4
  Multimodal, Qwen3-ASR, Moonshine, Seamless M4T; per model capture params,
  VRAM, RTFx, EN/ES coverage, translate capability, license. Source: HF Open
  ASR Leaderboard.
- Sanity-check against community consensus: recent r/LocalLLaMA threads, HF
  trending (30 days), llama.cpp release notes for perf regressions/wins.

### Honesty rules

- **Date-stamp** every claim that depends on "current SOTA".
- No published number for a specific quant → mark it "estimated" and show reasoning.
- Don't recommend a model you can't justify against an alternative on the frontier.
- Surface licenses — some "open" models forbid commercial use.
- Two models within ~3% composite → call it a **tie**, don't force a pick.

## Output contract (deterministic)

Three artifacts under `docs/frontier/runs/<today>/`, plus the `LATEST`
pointer and the ledger comment. Table shapes and section order are **fixed**
so consecutive runs diff mechanically; keep prose to a short didactic
paragraph per section explaining *why* the tables say what they say — tables
carry the facts.

### `report.md` — sections, in this exact order

- `# Local LLM + ASR Efficient Frontier — Results` (title, run date, hardware one-liner)
- `## 0. Verdict` — **the core artifact.** The per-role verdict table, exactly
  these columns, one row per role (`agentic_light`, `agentic_heavy`,
  `audio_transcribe`, `audio_translate`):

  | role | incumbent | verdict | best alternative | gap | reason |
  |------|-----------|---------|-------------------|-----|--------|

  `verdict` ∈ `keep` \| `upgrade` \| `runtime_upgrade` \| `retire` \| `watch`.
  `gap` = composite-quality gap vs. the best alternative (`tie (≤3%)`, `+5%
  alt`, `—` when the incumbent is the pick). `reason` is one line.
  Below the table, one bold **Diff vs. previous run (<date>):** line — a
  single sentence; "no change" is a valid and expected result.
- `## 1. Objective` — what an efficient frontier means here.
- `## 2. System & workloads` — restate the box and workloads.
- `## 3. Methodology` — the steps actually taken, reproducible by a colleague.
- `## 4. How to read the chart` — Pareto dominance, tiers, quantization
  tradeoffs; include one worked memory-budget example.
- `## 5. Results — shortlist by tier` — per tier (A fast / B balanced /
  C quality) a fixed-column table:

  | model | params | quant | VRAM | tok/s | quality | ctx | license | on frontier |
  |-------|--------|-------|------|-------|---------|-----|---------|-------------|

  plus a "dominated models" list with one-line *why dropped* each.
- `## 6. Concurrency plan` — 2-model and 3-model recipes.
- `## 7. Audio (ASR) annex` — workloads F/G/H: candidate table (same shape
  as the previous run's §7.2: model, params, VRAM, RTFx, EN/ES, translate,
  license, role), the F recommendation vs. the incumbent, the G
  single-model-vs-two-stage decision, the H verdict.
- `## 8. Progression` — the **cumulative run-over-run history**: copy the
  previous run's §8 table verbatim and append this run's rows (one per
  role). Exactly these columns:

  | run date | role | incumbent | verdict | best alternative |
  |----------|------|-----------|---------|-------------------|

  Never rewrite old rows — this table only grows.
- `## 9. Open questions / uncertainty` — explicit unknowns.
- `## 10. Current decisions (live, edited by /swap-model)` — mirrors
  `config/models.yaml` → `roles:` **as it stands right now** (the incumbent
  mapping — recommendations live in §0/§5, never here). `/swap-model`
  appends to this table when a swap happens; keep its shape identical to
  the previous run's §10.

### `frontier.json` — schema unchanged, backward-compatible

Machine-readable run data. Keys: `run_date`, `hardware`, `weights`,
`tier_info`, `models[]`, `asr_models[]`, `asr_translation_paths[]`,
`disfluency_verdict`, `concurrency_recipes[]`, `top_picks[]`, and
`shortlist{}` keyed by role (`agentic_light`, `agentic_heavy`,
`audio_transcribe`, `audio_translate`) with fields `current`, `current_id`,
`recommended`, `recommended_id`, `recommended_action`,
`recommended_runtime` (optional), `reason`. **`/swap-model` reads
`shortlist{}` — never rename or restructure it.** Match the previous run's
field shapes for everything else.

### `frontier.html` — standalone interactive chart

React 18 + Recharts via CDN, single `<script type="text/babel">` block, all
data inlined via `<script id="frontier-data" type="application/json">`, no
external API calls, works offline. Match the previous run's design (scatter:
x = tok/s, y = composite quality, bubble = VRAM, color = tier; VRAM-fit
filter; hover cards; shortlist + audio panels; prominent date stamp). The
admin SPA footer's 📈 Frontier link serves the latest run's copy.

### Ledger comment — issue #272

Post one comment per run (`gh issue comment 272`) so the ledger's last
comment always answers what runs in each role and what the best alternative
is. Exact shape:

```markdown
## Frontier run <YYYY-MM-DD>

<the §0 per-role verdict table, verbatim>

**Diff vs. previous run (<date>):** <the §0 diff line>

Full report: [report.md](https://github.com/ferraroroberto/local-llm-hub/blob/main/docs/frontier/runs/<date>/report.md) · [frontier.html](https://github.com/ferraroroberto/local-llm-hub/blob/main/docs/frontier/runs/<date>/frontier.html)
```

Single long lines, no hard wraps (rendered markdown).

## Workflow

1. **Read inputs:** `docs/frontier/runs/LATEST` → previous run's `report.md`
   + `frontier.json` (diff context, §8 progression table to carry forward);
   `config/models.yaml` → `models:` and `roles:` (current incumbents);
   `config/machine_specs.yaml` if present;
   **`docs/frontier/local-findings.md`** — locally-tested candidates whose
   published numbers didn't hold up on this box (#277). It is the
   deterministic run-over-run memory; ledger comments are NOT re-read, so a
   disproof reaches you only through this file.
2. **Pick today's date** (`YYYY-MM-DD`). If today's run dir already exists,
   overwrite it — same-day re-runs are idempotent.
3. **Run the research** per the brief above (web search; date-stamp claims).
4. **Compute the per-role verdicts** using the quality weights and workload
   mix; apply the honesty rules (ties, estimate flags, licenses).
   **Local-findings override (#277):** if a role's best alternative matches
   an *unresolved* entry in `docs/frontier/local-findings.md`, the verdict is
   `watch` with reason `disproven locally <date> (#N)` — never `upgrade` /
   `runtime_upgrade` — regardless of published numbers, unless that entry's
   stated re-open trigger is demonstrably met (cite the evidence in §7/§9 if
   you claim it is). `watch` is not actionable, so step 8 files nothing:
   this is what stops the propose→disprove→re-propose loop.
5. **Write the three artifacts** under `docs/frontier/runs/<today>/` per the
   output contract.
6. **Repoint `docs/frontier/runs/LATEST`** to contain just `<today>\n`.
7. **Post the ledger comment** on #272 (shape above).
8. **File issues for actionable verdicts.** For each role whose verdict is
   `upgrade`, `runtime_upgrade`, or `retire` — **not** `keep` or `watch` —
   spawn one subagent (general-purpose, **model sonnet**) whose task is to
   invoke the `/issue-add` skill with this payload:

   > Frontier run <YYYY-MM-DD> verdict for role `<role>`: incumbent
   > `<incumbent>`, verdict `<verdict>`, best alternative `<best
   > alternative>`, reason: <reason>. Details in
   > `docs/frontier/runs/<date>/report.md` §0/§7. File the issue for the
   > work this verdict calls for.

   One subagent per actionable verdict; `/issue-add`'s own duplicate check
   (it scans open issues first) keeps repeat runs from re-filing while the
   issue is still open. **Await every subagent to completion within this
   turn** — in a headless scheduled run nothing resumes you
   (fleet-config#314); never fire-and-forget. Include the filed (or
   deduped) issue numbers in the step-10 summary.
9. **Commit the run** — scoped to the artifacts only:
   ```
   git add docs/frontier/runs/
   git commit -m "docs: frontier refresh run (<YYYY-MM-DD>)"
   ```
   If the current branch is `main` (the scheduled unattended case), also
   `git push`. On a feature branch, leave pushing to the normal
   PR/`issue-finish` flow.
10. **Stop.** Print a one-paragraph summary of the four role verdicts, the
    diff vs. last run, and any issues filed in step 8. Do **not** edit
    `config/models.yaml`, write launchers, download weights, or touch
    `models/`. When ready to act: `/swap-model`.

## What success looks like

- `docs/frontier/runs/<today>/{report.md,frontier.json,frontier.html}` all
  present, self-consistent, following the output contract (§0 verdict table
  + §8 progression table with the exact columns)
- `docs/frontier/runs/LATEST` resolves to today
- Ledger issue #272 has this run's comment
- Every actionable verdict (`upgrade` / `runtime_upgrade` / `retire`) has an
  open issue — filed this run via the sonnet `/issue-add` subagents, or
  already open from a previous run
- No verdict contradicts an unresolved `docs/frontier/local-findings.md`
  entry (such roles show `watch` / "disproven locally", not a re-proposal)
- Zero changes to `config/models.yaml`, `launchers/`, `models/`, `tray/`,
  `src/`, `tests/`
