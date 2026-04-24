# Plan — add Gemma to the hub for action-item extraction — 2026-04-20

Sibling of [20260420-hub-with-qwen-and-glm.md](20260420-hub-with-qwen-and-glm.md).
That doc is the post-mortem of how the three-backend hub (Claude +
Qwen + GLM) got built; this one is the forward-looking plan for adding
a fourth row — **Gemma 3** — so a small two-tier text classifier can
run end-to-end on the RTX 5060 Ti 16 GB gaming PC with no network hop
to a cloud API.

The mechanics slot in exactly like Qwen did: one new block in
`config/models.yaml`, one download entry, one launcher pair, one
Streamlit card for free via the registry-driven UI. No code paths
change.

---

## Why Gemma, and why now

The target didactic workload is a **two-tier action-item extractor**
over short lines of free text (meeting transcripts, chat logs, notes):

- **Tier A (strict)** — trigger when a line contains **verb + object +
  owner**. Example schema: verb ∈ {`send`, `review`, `deploy`,
  `update`, `schedule`, …}; object ∈ {`report`, `deck`, `doc`, `PR`,
  `release notes`, `ticket`, …}; owner ∈ a list of participants or
  team handles extracted from the document header.
- **Tier B (looser)** — trigger when a line contains **verb + object**
  only (no explicit owner). Higher recall, lower precision — captures
  unassigned intent ("we should deploy the patch") that might still be
  a task.

Both tiers run first through a **deterministic matcher** (regex /
lemma lookup) to cut the overwhelming majority of traffic for free.
Every positive from that stage is then **verified by an LLM** — a
cheap human-in-the-loop substitute that confirms whether the
deterministic hit is a genuine action item or a coincidental co-
occurrence ("I'll *send* my *regards*" looks like verb+object but is
not a task). False positives dominate the cost; precision on this
verify step is what matters.

Today the verifier can run against a cloud API; the goal here is a
local-only path for the same pipeline:

> Find the strongest open-weight verifier that a local container can
> host, with no external API dependency, and validate it first on the
> gaming PC (16 GB VRAM).

Gemma 3 matches the shape of the problem (strict instruction
following, JSON-out, short prompts, classification with binary verify)
and has Google-published **QAT** (quantization-aware-trained) GGUFs
that preserve near-BF16 quality at 4-bit — exactly what we need to fit
the 27B into 16 GB VRAM.

## Recommendation

Two Gemma entries, both via llama.cpp/`llama-server` like Qwen and
GLM. No Hugging Face Inference API — not a game-changer for this
workload (strict-schema classification is latency-bound per call, and
the whole point is to end up on a locally-hosted container).

### Tier-2 (default fast verifier) — `gemma3-12b-it`

- **Repo:** `unsloth/gemma-3-12b-it-GGUF` (Q4_K_M).
- **File size:** ~7.3 GB GGUF. Fits entirely in 16 GB VRAM with
  plenty of headroom for KV cache at 16 k context.
- **Why this one first.** Gemma 3 12B IT is the instruction-tuned
  mid-tier. In Google's published evals it lands close to Gemini
  1.5-Flash quality on instruction-following and JSON-schema
  adherence; on the verifier task (short prompt, binary answer, hard
  negatives) that is almost always sufficient. Runs at ~70–90 tok/s
  on a 5060 Ti at Q4 — a good fraction of per-call time stays in
  prompt prefill, not generation.
- **Expected role.** The verifier the pipeline actually calls. Same
  slot Qwen occupies in the current hub.

### Tier-1 (quality ceiling for eval) — `gemma3-27b-it-qat`

- **Repo:** `google/gemma-3-27b-it-qat-q4_0-gguf` (Google's official
  QAT release; Unsloth mirrors it as a fallback).
- **File size:** ~15.6 GB GGUF at `Q4_0` QAT.
- **Fit on 16 GB VRAM.** Weights nearly fill VRAM. Two viable
  configurations:
  1. **Partial GPU offload** — `-ngl ~50` (of 62 layers), `-c 4096`.
     ~3 GB spills to RAM. Expect 15–25 tok/s. Simpler.
  2. **Tight fit full-GPU** — `-ngl 99`, `-c 2048`, `--no-mmap`,
     `--flash-attn`. Risk: one long prompt tips over into OOM. Skip
     unless (1) is a blocker.
- **Why QAT specifically.** Gemma 3 27B at vanilla Q4_K_M is ~17 GB
  (doesn't fit). The Google QAT release is the same architecture
  re-trained to be robust to 4-bit quantization — it ships at Q4_0
  (~15.6 GB) and loses noticeably less quality than a
  post-training-quantized Q4. This is the single reason 27B is on
  the table for 16 GB VRAM at all.
- **Expected role.** The quality benchmark — run the full corpus
  through both 12B and 27B, measure false-positive rate, decide
  which one to ship. The 27B may win on hard-negative disambiguation
  ("send my regards" vs. "send the report") that the 12B gets wrong.

### What we are explicitly not picking (and why)

- **Gemma 3n (2B / 4B).** Designed for mobile/edge. Too small to
  suppress false positives on a verifier task at the quality bar a
  cloud Flash-class baseline sets — would be a regression.
- **Gemma 3 4B.** Same reasoning. 12B dominates it cleanly at the
  classifier task and costs ~60 % more VRAM we have to spare.
- **Gemma 3 PaliGemma / vision variants.** No images in the input;
  overhead with no payoff.
- **Hugging Face Inference API.** Rejected per the brief — the
  whole point is to end up hosted locally with no external traffic.
  No quality gap big enough to justify the network dependency for
  pre-deployment testing.

### How this maps to the existing tier model in the repo

The repo already has a soft two-tier concept:

| Existing tier | Slot | What we're adding |
|---|---|---|
| Tier 1 — heavy MoE (GLM-4.5-Air) | 106 B / 12 B active, agent/coding | — |
| Tier 2 — small dense (Qwen3.5-9B) | 9 B, fast all-rounder | `gemma3-12b-it` parallel to Qwen — classifier-optimised dense |
| — | — | `gemma3-27b-it-qat` as a quality ceiling / eval baseline |

So Gemma doesn't replace anything; it adds a *classifier-tuned*
dense tier alongside Qwen and a *single-GPU-fits* 27 B alongside
GLM's MoE path.

## Implementation plan

Exactly parallel to the Qwen / GLM build. Every step below is a
concrete file touch — no new modules, no routing changes, no UI
work. The registry-driven surfaces (installer, Models tab,
Playground picker, smoke test) pick up the new rows automatically.

### 1. `config/models.yaml` — two new blocks, one host edit

Add to `models:` (after the `glm:` block):

```yaml
  # --- Local Gemma 3 12B IT via llama.cpp (fast classifier tier) ---
  gemma3_12b:
    display_name: gemma3-12b-it
    backend: openai
    engine: llama-server
    port: 8083
    hf_repo: "unsloth/gemma-3-12b-it-GGUF"
    hf_pattern: "gemma-3-12b-it-Q4_K_M.gguf"
    model_path: "models/gemma-3-12b-it-Q4_K_M.gguf"
    args:
      - "--jinja"
      - "-ngl"
      - "99"
      - "-c"
      - "16384"
      - "--alias"
      - "gemma3-12b-it"

  # --- Local Gemma 3 27B IT QAT via llama.cpp (quality tier; fits 16 GB VRAM) ---
  gemma3_27b:
    display_name: gemma3-27b-it
    backend: openai
    engine: llama-server
    port: 8084
    hf_repo: "google/gemma-3-27b-it-qat-q4_0-gguf"
    hf_pattern: "gemma-3-27b-it-q4_0.gguf"
    model_path: "models/gemma-3-27b-it-q4_0.gguf"
    args:
      - "--jinja"
      - "-ngl"
      - "50"
      - "-c"
      - "4096"
      - "--flash-attn"
      - "--alias"
      - "gemma3-27b-it"
```

Extend the `pc-cuda` host's `enabled` list:

```yaml
  pc-cuda:
    platform: win32
    default: true
    enabled: [qwen, glm, gemma3_12b, gemma3_27b]
```

Leave `mac-mini-m4.enabled` as `[qwen]` — the 12B Gemma would fit
in 16 GB unified memory but we don't need it there (this whole
feature is about the PC path).

**Port assignments.** 8081/8082 are taken by Qwen/GLM. Allocate
**8083 → gemma3-12b** and **8084 → gemma3-27b**. `_check_ports()`
in `src/install.py` picks these up from the registry; no code
change.

### 2. `scripts/download_models.py` — no change

Already iterates every enabled `openai`-backed model with an
`hf_repo`. The two new blocks will be pulled automatically by:

```bat
.venv\Scripts\python -m src.install --fix
.venv\Scripts\python scripts\download_models.py --only gemma3_12b
.venv\Scripts\python scripts\download_models.py --only gemma3_27b
```

Rough disk budget: 7.3 GB + 15.6 GB = **~23 GB** on top of the
existing Qwen (6.6 GB) + GLM (55 GB). Confirm free space before
running `--fix`.

### 3. `src/run_backend.py` / launchers — add two thin scripts

`src/run_backend.py` already dispatches `hub | <model_id>` by id.
No change needed there.

Add four new launcher files following the existing pattern:

- `run_gemma3_12b.bat` — mirror of `run_qwen.bat`, calls
  `python -m src.run_backend gemma3_12b`.
- `run_gemma3_12b.sh` — mirror of `run_qwen.sh`.
- `run_gemma3_27b.bat` — mirror, with banner copy flagged as
  the quality tier.
- `run_gemma3_27b.sh` — mirror.

Extend `run_all.bat` and `run_all.sh` with `start gemma3_12b` and
`start gemma3_27b` entries so the "start every enabled backend"
path brings up all four local models on the PC.

### 4. `src/install.py` — no change

`_check_models()` and `_check_ports()` already iterate
`enabled_models()` and build `Check` rows + `fix_download_<id>`
dispatch for every row. Running `python -m src.install` after step
1 shows two new `missing` rows; `--fix` calls
`download_models.download_one("gemma3_12b")` and likewise for
`gemma3_27b`. The `-ngl 50` partial-offload for 27B is plain
llama-server args; no new code path.

### 5. `src/llama_process.py` — no change

`build_command()` reads `model.args` verbatim. The 27B's
`--flash-attn` and `-ngl 50` drop straight through. Per-model
process singletons work as-is; the Models tab renders two new
cards for free.

### 6. Tests — extend fixtures only

- `tests/test_model_registry.py` — add an assertion that both
  `gemma3-12b-it` and `gemma3-27b-it` resolve on `pc-cuda` and do
  *not* resolve on `mac-mini-m4`. Keeps the per-host filtering
  guarantee honest once we have four local models instead of two.
- `tests/test_router.py` — the existing parametrised tests already
  cover every `openai`-backend entry; the two new rows get coverage
  for free once the fixture config in the test includes them.
- No new test file. No change to `test_server.py`,
  `test_install.py`.

### 7. Smoke test — no change

`scripts/smoke_test.py` iterates `enabled_models()` and prints
pass/skip/fail per display name. Expected new output after both
backends are up:

```
passed : 5 — claude-haiku-4-5, qwen3.5-9b, glm-4.5-air,
             gemma3-12b-it, gemma3-27b-it
```

### 8. Streamlit UI — no change

`app/views/models.py`'s `_render_llama_card()` is
model-id-agnostic. Two new cards appear automatically; Start /
Stop / log tail / launch-args expander all work identically.

### 9. Docs

- `docs/project-structure.md` — add `gemma3-12b-it` and
  `gemma3-27b-it` to the backend box in the component mermaid, plus
  two more `models/<gguf>` entries in the filesystem tree. Update
  the "Purpose" and "Key facts" bullets to mention four backends.
- `README.md` — add both rows to the bullet list at the top, add
  the two new launcher scripts to the Run table, note the
  `gemma3-*` model strings in the Python example.

### 10. Classifier eval harness (new, lives outside the hub)

The deliverable is "does this match the cloud baseline on our
false-positive rate?" — that is an **eval task**, not a hub feature.
Keep it out of `src/` so it can't break the hub. Suggested shape,
co-located with `scripts/smoke_test.py`:

```
scripts/
  classifier_eval.py        # new; reads a CSV of labelled examples,
                            # calls /v1/messages with each candidate
                            # model, computes precision/recall/F1 +
                            # confusion matrix; optional --compare
                            # flag lines up a baseline JSONL.
  fixtures/
    action_items.csv        # labelled: text, tier, expected_label
                            # e.g. "Alice will send the deck by Fri",
                            #      tier=A, label=1
                            #      "Thanks for sending your regards",
                            #      tier=A, label=0  (hard negative)
```

`classifier_eval.py` takes `--model gemma3-12b-it` (or a
comma-separated list) and writes a report to
`eval_runs/<timestamp>-<model>.json`. No hub changes; uses the hub
just like any other client. Baseline numbers land in the same
JSON schema by running the same script with a `--baseline-file`
flag that reads pre-computed labels instead of calling an LLM.

## Risks / open questions

- **27B OOM headroom.** `-ngl 50 -c 4096` is conservative; the real
  limit depends on KV-cache dtype and on whether `--flash-attn` is
  fully supported for Gemma 3 in the llama.cpp build we vendored. If
  launch logs show `ggml_cuda_host_malloc: failed to allocate`, drop
  `-ngl` to 46 or `-c` to 2048. Tune live — we have instrumentation
  via the Models-tab log tail.
- **Prompt template differences.** Gemma 3 IT uses a specific
  `<start_of_turn>` template. `--jinja` + the bundled chat template
  in the GGUF handles this; confirm on first boot by sending a
  trivial message through the hub and checking the Models-tab log
  doesn't show "template not found" warnings.
- **QAT repo availability.** If Google pulls the official QAT repo,
  fall back to `unsloth/gemma-3-27b-it-qat-GGUF` or Bartowski's
  mirror. The `hf_repo` field is the only place this needs to
  change.
- **Linux-CUDA parity.** The vendored llama.cpp binary here is
  Windows-CUDA; a Linux-CUDA build exists and works the same way.
  Not a blocker — the model selection and args are identical across
  builds; only `scripts/install_llama_cpp.py` needs a Linux release
  URL if we ever want the same installer path on Linux.

## Deferred (not in this plan)

- Swapping the verifier from an LLM-verify step to a small fine-
  tuned Gemma 3 4B classifier. Interesting if the eval shows 12B is
  overkill, but that's a training task, not a hub task.
- Adding a second engine (vLLM) for the 27B to claw back tok/s.
  Same decision we made for Qwen/GLM — one engine across hosts;
  revisit if latency becomes the bottleneck.
- Tool-use round-trips on the Anthropic shape for Gemma. Same
  deferral as Qwen/GLM — OpenAI-shape callers get native tool
  calls from `llama-server --jinja`; the classifier doesn't need
  tool calls anyway.

---

## Execution log — 2026-04-20

Everything in the plan above except the "classifier eval harness"
(step 10) was applied. Two real-world deviations were needed;
both were risks the plan already flagged.

### What was done

1. **`config/models.yaml`** — added `gemma3_12b` and `gemma3_27b`
   blocks as specified, and extended `hosts.pc-cuda.enabled` to
   `[qwen, glm, gemma3_12b, gemma3_27b]`. `mac-mini-m4.enabled`
   left at `[qwen]`.
2. **Launchers** — created
   [run_gemma3_12b.bat](../../launchers/run_gemma3_12b.bat) /
   [run_gemma3_12b.sh](../../launchers/run_gemma3_12b.sh) and
   [run_gemma3_27b.bat](../../launchers/run_gemma3_27b.bat) /
   [run_gemma3_27b.sh](../../launchers/run_gemma3_27b.sh), mirrors of
   the Qwen / GLM ones. Extended
   [run_all.bat](../../launchers/run_all.bat) and
   [run_all.sh](../../launchers/run_all.sh) to start both.
   *(2026-04-24: launchers were later moved from the repo root into
   `launchers/`; links above reflect the current path.)*
3. **Tests** — added `test_gemma_per_host_filtering` in
   [tests/test_model_registry.py](../../tests/test_model_registry.py)
   that asserts both gemma ids resolve on `pc-cuda` and do *not*
   resolve on `mac-mini-m4`, and that their ports are 8083 / 8084.
   No other test file changed (as the plan predicted).
   Full suite: **16 passed**.
4. **Downloads** — `gemma3-12b-it` pulled from
   `unsloth/gemma-3-12b-it-GGUF` cleanly (6.8 GB on disk).
   `gemma3-27b-it` on the first try failed against
   `google/gemma-3-27b-it-qat-q4_0-gguf` with
   `GatedRepoError: 401 … Access to model is restricted`. The
   plan's **Risks / open questions** section named exactly this
   outcome ("If Google pulls the official QAT repo, fall back to
   `unsloth/gemma-3-27b-it-qat-GGUF`"). Switched the `hf_repo` +
   `hf_pattern` + `model_path` to that mirror
   (`gemma-3-27b-it-qat-Q4_0.gguf`, 14.5 GB on disk, same QAT
   weights).
5. **Smoke test** (hub + gemma3-12b + gemma3-27b running):

   ```
   passed : 3 — claude-haiku-4-5, gemma3-12b-it, gemma3-27b-it
   skipped: 2 — qwen3.5-9b, glm-4.5-air
   failed : 0
   ```

   Qwen and GLM were skipped because their backends weren't
   started for this session — the smoke test skips unreachable
   ports by design. Both Gemma backends returned `"pong"` in
   ~3 output tokens through `/v1/messages`.
6. **Docs** — updated
   [project-structure.md](../project-structure.md) (component +
   module mermaids, request-lifecycle header, key-facts
   bullets) and the [README](../../README.md) (top bullet list,
   ASCII architecture box, Layout tree, Setup disk budget,
   Run table, Python example).

### Deviations from the plan

- **27B GGUF source.** Repo switched from
  `google/gemma-3-27b-it-qat-q4_0-gguf` (gated, 401) to
  `unsloth/gemma-3-27b-it-qat-GGUF` file
  `gemma-3-27b-it-qat-Q4_0.gguf`. `model_path` became
  `models/gemma-3-27b-it-qat-Q4_0.gguf`. A short comment was
  added to the yaml block noting why.
- **`--flash-attn` arg.** The vendored llama.cpp build (CUDA
  Windows) rejects the bare flag:
  `error: unknown value for --flash-attn: '--alias'`. It
  requires a value (`on|off|auto`) in this version. Changed
  the 27B args from `--flash-attn` to
  `--flash-attn` + `on`. The plan flagged this exact
  uncertainty ("whether `--flash-attn` is fully supported for
  Gemma 3 in the llama.cpp build we vendored").
- **Classifier eval harness (step 10)** — deliberately *not*
  built yet. That step is described in the plan as sitting
  outside the hub and requires a labelled CSV that does not
  exist yet. The hub-integration piece of this plan is
  complete; the eval work is the separate deliverable.

### State at close

- All four local models download OK on `pc-cuda`:
  Qwen (5.3 GB), GLM (46.6 GB), Gemma 3 12B (6.8 GB),
  Gemma 3 27B QAT (14.5 GB).
- `python -m src.install` returns all rows **ok** with both
  Gemmas present and ports 8083 / 8084 free.
- 12B runs with full GPU offload (`-ngl 99`, `-c 16384`);
  27B runs with partial offload (`-ngl 50`, `-c 4096`,
  `--flash-attn on`). Both answer through the hub.
- Four launchers + `run_all` + Streamlit Models-tab cards
  appear for free via the registry.

### Known follow-ups

- The 27B's `-ngl 50 -c 4096` numbers are conservative. If
  VRAM has headroom during a sustained load, try `-ngl 62`
  (all layers) with `-c 2048` or `-c 4096 --no-mmap`. Tune
  live from the Models-tab log tail.
- Build the classifier eval harness (step 10) once the
  labelled `action_items.csv` fixture is ready.
