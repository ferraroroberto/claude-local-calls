# Running a Local LLM for the openclaw Agent — Hardware & Model Research

**Date:** 2026-04-19
**Goal:** Replace / complement `claude-local-calls` with a genuinely local
LLM backend that can drive the **openclaw** agent (tool-calling, long-ish
context, runs 24/7). Scope: what fits on the two machines the user owns,
which Qwen (and adjacent) models are best in 2026, and the realistic
tokens/sec + ergonomics of each choice.

---

## 1. The user's hardware

| Machine | Spec summary | Role it can play |
|---|---|---|
| **Mac mini (base M4)** | 10-core CPU · 10-core GPU · **16 GB unified memory** · 256 GB SSD · ~120 GB/s mem bandwidth · idle ≈ 5 W, burst ≈ 30–65 W | Always-on small-model server. Cheap to leave on 24/7 (~US$15–20/yr electricity). Memory is the hard ceiling. |
| **Windows PC** | ZOTAC RTX **5060 Ti AMP 16 GB GDDR7** (Blackwell, 4608 CUDA, 448 GB/s, 180 W TDP) + **128 GB system RAM** | The serious inference box. 16 GB VRAM fits dense ≤14B at Q4; the 128 GB of RAM unlocks large **MoE** models (Qwen3-Coder-Next, Qwen3-Next) via CPU-offload with tolerable speed thanks to low active-param counts. |

Key takeaway up front: **the Mac mini's 16 GB unified memory is the
binding constraint** — it caps you at ~7–9B dense models. The RTX 5060 Ti
PC is the one that can actually run agent-grade models; the Mac mini's
right role is a 24/7 lightweight endpoint or a utility model (embeddings,
reranker, small helper).

---

## 2. The Qwen lineup in April 2026 — what's current and what runs locally

The Qwen family expanded a lot over the past year. Current (April 2026)
releases relevant to local inference:

### 2.1 Dense text models
- **Qwen3.5 dense series**: 0.8 B, 2 B, 4 B, 9 B, 27 B
  - The 9 B fits in ~6.6 GB via Ollama Q4 and is the most commonly cited
    "good on a laptop" pick; beats models 3× its size on some reasoning
    benchmarks.
  - 27 B Q4 is the upper end of what a 16 GB single-GPU or 32 GB Mac can
    run; **too big for the 16 GB Mac mini**.

### 2.2 MoE text models (total params / active params)
- **Qwen3-Coder-Next** — 80 B / **3 B active** · 256 k context · explicitly
  trained for **tool-use + long-horizon agent loops** · works out of the
  box with Claude Code / Qwen Code / Cline / OpenCode. Unsloth recommends
  **≥45 GB memory for 4-bit, ≥30 GB for 2-bit-XL**.
- **Qwen3-Next-80B-A3B** — 80 B / 3 B active · hybrid attention, very
  efficient.
- **Qwen3.5-35B-A3B / 122B-A10B / 397B-A17B** — larger MoEs; only the
  **35B-A3B (~22 GB at Q4)** is realistic on a 16 GB GPU with CPU offload.
- **Qwen3.6-35B-A3B** (April 2026) — newest; same memory profile.

### 2.3 Multimodal / specialty
- **Qwen3-VL** — dense 2 B / 4 B / 8 B / 32 B; MoE 30 B-A3 B and
  235 B-A22 B. Handles text + image + video, 256 k context. The 8 B-VL
  is the sweet spot for a 16 GB GPU if you need vision.
- **Qwen3-Omni** — audio + vision + text; research-heavy, less battle-
  tested locally.
- **Qwen3-Coder (non-Next)** — earlier coder series; superseded by Coder-
  Next for agent work, keep the older one in mind only if a specific
  build works with your tooling.

### 2.4 Non-Qwen models worth knowing for comparison
- **gpt-oss-20B (MXFP4)** — runs fantastically on the 5060 Ti
  (~488 TPS for short-context API workloads, according to hardware-corner
  benchmarks). Strong agent model.
- **Gemma 3 / Phi-4** — small models (≤9 B) that compete with Qwen3.5-9B
  on the Mac.
- **Llama 3.1 8B** — on the 5060 Ti ≈ 71 tok/s; good baseline.

---

## 3. Inference engines — which to run where

| Engine | Best on | Why |
|---|---|---|
| **MLX / MLX-LM** | Mac mini | Native Metal. ~230 tok/s on optimized 7 B. 30–50 % faster than llama.cpp on Apple Silicon. |
| **Ollama** (w/ MLX backend on Apple) | Mac mini, PC (CUDA) | Easiest UX, OpenAI-compatible endpoint, good model library. Now uses MLX under the hood on Apple Silicon. |
| **llama.cpp (`llama-server`)** | PC, Mac mini | OpenAI-compatible `/v1/chat/completions`, **`--jinja` flag enables native function calling** including Hermes-style templates used by Qwen3. Best raw control. |
| **LM Studio** | Either, GUI users | Good for model browsing and comparing quants; wraps llama.cpp. |
| **vLLM** | PC only (CUDA) | Production-grade serving, paged attention, best throughput for Qwen3 / Qwen3-VL. Heavier to set up. Requires `vllm>=0.11.0` for Qwen3-VL. |
| **Qwen-Agent** | Any | Agent framework built on top of Qwen ≥ 3.0; function-calling, MCP, code interpreter, RAG — useful scaffolding. |

For an agent workload, the decisive feature is **reliable OpenAI-style
tool calling**. As of 2025-08 there were rough edges in `llama.cpp`'s
Qwen3-Coder tool-call parser; by early 2026 the Hermes template path is
the recommended one (already wired into Qwen3's `tokenizer_config.json`).
vLLM has the cleanest tool-calling story but only runs on the NVIDIA box.

---

## 4. What actually fits — quick capacity table

Rule of thumb: `VRAM_needed ≈ params × bytes_per_param + KV-cache`.
Q4_K_M ≈ 0.5 B/param plus ~10 % overhead; expect a bit more with
larger contexts.

### 4.1 Mac mini, 16 GB unified

| Model | Quant | Fits? | Expected tok/s | Notes |
|---|---|---|---|---|
| Qwen3.5-4B | Q4_K_M | ✅ (~3 GB) | 60–80+ via MLX | Fastest, useful as drafter / small helper. |
| Qwen3.5-9B | Q4_K_M | ✅ (~6.6 GB) | **25–35 (Ollama), up to ~50 (MLX)** | Best daily-driver on this box. |
| Qwen2.5-7B | Q4_K_M | ✅ | ~32–35 | Mature, great tool-calling. |
| Qwen3-VL-8B | Q4 | ⚠️ tight (~10 GB + vision tower) | low 20s | Works but leaves little headroom for context. |
| Qwen3.5-27B | Q4 | ❌ (>15 GB + ctx) | — | Don't try on 16 GB. |
| Qwen3-Coder-Next (80B-A3B) | any | ❌ | — | Needs ≥30 GB. |

### 4.2 PC, RTX 5060 Ti 16 GB + 128 GB DDR5

| Model | Quant | Fits VRAM? | Expected tok/s | Notes |
|---|---|---|---|---|
| Llama 3.1 8B | Q4 | ✅ | ~71 | Baseline. |
| Qwen3.5-9B | Q4 | ✅ | 60–80 | Strong small-model pick. |
| Qwen3 14B | Q4_K @ 16k ctx | ✅ (~9 GB) | **32.9 measured** | Reference benchmark. |
| Qwen2.5 14B | Q4 | ✅ | ~40 | Sanity check. |
| Qwen3-Coder 30B | Q4 | ❌ VRAM · ✅ w/ offload | 10–20 (offload) | Works, limited by DDR5 bandwidth. |
| **Qwen3-Coder-Next 80B-A3B** | **Q4 / 2-bit-XL** | ❌ VRAM alone · ✅ **with offload onto the 128 GB DDR5** | **~20–40 expected** (only 3 B active params!) | **Best agent model for this box.** MoE with tiny active set masks offload penalty. |
| Qwen3.5-35B-A3B | Q4 (~22 GB) | ❌ VRAM · ✅ offload | ~25–35 | Good all-rounder MoE. |
| gpt-oss-20B MXFP4 | native | ✅ | **488 on short ctx** | Blazing fast, strong agent. |
| Qwen3-VL-32B | Q4 | ❌ VRAM alone · partial offload | slow | Use 8B-VL unless you really need the 32B. |

The 128 GB of system RAM is the quiet superpower here: DDR5 at
40–60 GB/s is 7–10× slower than VRAM, but for an **MoE that activates
only ~3 B of 80 B parameters per token**, most of the cold experts
sit in RAM and the 5060 Ti only has to stream hot experts per token.
That's the specific regime where CPU offload actually works.

---

## 5. Tool calling / agent readiness

- **Qwen3-Coder-Next** was trained specifically for agentic loops —
  long-horizon reasoning, complex tool use, recovery from tool errors.
  That's the exact shape of openclaw's workload. It ships with Hermes-
  style function-calling baked into the chat template.
- **llama.cpp** exposes OpenAI-compatible `/v1/chat/completions` with
  `tools` and `tool_choice` when launched with `--jinja`. Confirmed
  working with Qwen3 Hermes templates; recent fixes (late 2025) made
  Qwen3-Coder parsing reliable.
- **vLLM** has first-class `--tool-call-parser` support for Qwen3 and
  is the most robust for heavy agent traffic.
- **Ollama**'s `/api/chat` and its OpenAI shim both support `tools` with
  Qwen3 variants.
- **Qwen-Agent** (QwenLM/Qwen-Agent) gives you function calling + MCP +
  code interpreter wrappers if you'd rather not roll your own.

openclaw (per the project README at README.md:88-89 it's referenced as
"an agent like openclaw running next to you") should talk to any of
these via an OpenAI- or Anthropic-compatible base URL. On the
Anthropic-shape side you'd still need a translation layer — either keep
this repo and swap the backend, or run the agent against the native
OpenAI-compatible endpoint if openclaw supports it.

---

## 6. Recommended setups

Ordered by how well they fit the user's stated goals.

### 🥇 Option A — "PC does the thinking, Mac does always-on routing"
- **PC (on-demand):** `llama.cpp` server or vLLM running
  **Qwen3-Coder-Next (80B-A3B)** at Q4_K_M or Q2_K_XL, with
  `--n-gpu-layers` tuned so hot experts + attention live in the
  16 GB VRAM and cold experts spill to the 128 GB DDR5. Expose an
  OpenAI-compatible endpoint on the LAN.
- **Mac mini (24/7):** keep the existing `claude-local-calls` server
  running for Claude fallbacks, **plus** an MLX-served **Qwen3.5-9B**
  for cheap/fast requests (classification, routing, short completions).
- openclaw points at the Mac as the default entry point; the Mac
  routes heavyweight agent turns over the LAN to the PC.
- Wakeonlan / auto-sleep on the PC keeps idle power realistic.

**Why this wins:** matches each machine to what it's good at, gives
openclaw the only Qwen model explicitly tuned for agents, and keeps the
light stuff instant on the Mac.

### 🥈 Option B — "Single-box on the PC"
- PC runs **vLLM** with Qwen3-Coder-Next (or gpt-oss-20B if you want
  pure speed over agent fidelity). Everything hits that one endpoint.
- Mac mini is only for the Claude proxy (or turned off).
- Simpler, but you lose the 24/7 always-on piece unless the PC runs
  24/7 — which costs ~10× more power than the Mac.

### 🥉 Option C — "Mac-only, small model"
- Mac mini runs **Qwen3.5-9B via MLX/Ollama** (or **Qwen2.5-7B** for
  maximum tool-calling reliability) as the sole backend.
- Good for basic agent work, not enough headroom for 256 k context or
  complex multi-tool loops. Power-cheap and dead simple.
- Skip this if openclaw does anything substantial with code or long
  contexts.

### Not recommended
- Running Qwen3-Coder-Next or any 30B+ model on the **Mac mini 16 GB**
  — it simply won't fit, and swap-based hacks are painful.
- Renting a 5060 Ti tier GPU is cheaper per-token than buying electricity
  if your usage is <2 h/day — worth running the numbers if power is
  expensive where you are.

---

## 7. Concrete next steps (if you decide to build this)

1. **Install Ollama on the Mac mini**, pull `qwen3.5:9b` (or
   `qwen2.5:7b-instruct`), confirm `ollama run` answers with
   tools. Leave running as a service.
2. **On the PC**, pick one of:
   - `llama.cpp` with `llama-server --jinja --reasoning-format deepseek
     -hf unsloth/Qwen3-Coder-Next-GGUF:Q4_K_M -ngl 99 -c 65536` (tune
     `-ngl` for your VRAM; offload will kick in automatically).
   - `vLLM` with Qwen3-Coder-Next if you want max throughput.
3. Benchmark with the `scripts/smoke_test.py` pattern in this repo —
   swap the `base_url` to `http://<pc-lan-ip>:8080/v1/` and drive a few
   tool-call turns through the openclaw prompt.
4. **Decide protocol shape:** if openclaw speaks Anthropic Messages, keep
   using this repo but swap the backend from `claude -p` to an HTTP call
   into the local OpenAI-compatible server, adding a thin translator
   (OpenAI tool-call → Anthropic `tool_use` blocks). That's the real
   value-add over just pointing openclaw at Ollama directly.

---

## 8. Cost comparison — local vs Gemini 3.1 Flash Lite vs Claude Sonnet 4.6

### 8.1 Published API prices (April 2026, per 1 M tokens)

| Model | Input | Output | Notes |
|---|---|---|---|
| **Gemini 3.1 Flash Lite Preview** | $0.25 | $1.50 | 1 M ctx, multimodal, thinking tokens billed as output. |
| **Claude Sonnet 4.6** | $3.00 | $15.00 | 1 M ctx at standard price. |
| Sonnet 4.6 (Batch API) | $1.50 | $7.50 | 50 % off, async only. |
| Sonnet 4.6 (prompt cache hit) | $0.30 | $15.00 | Cache hit = 10 % of input price; huge for agent loops that replay context. |

### 8.2 Effective cost on a typical agent turn

Agents are input-heavy (system prompt + tool results fed back in).
Assuming a **5 : 1 input : output** ratio:

| Option | Blended $/M tokens | Relative to Flash Lite |
|---|---|---|
| Gemini 3.1 Flash Lite | **$0.46** | 1× |
| Sonnet 4.6 (no cache) | **$5.00** | ~11× |
| Sonnet 4.6 (batch) | $2.50 | ~5.4× |
| Sonnet 4.6 (80 % cache hit) | ~$3.20 | ~7× |
| **Local (electricity only)** | **$0.002 – $0.01** | ~50–500× cheaper |

### 8.3 Electricity on the user's hardware

- **Mac mini M4 (16 GB), 24/7:** idle ≈ 5 W, inference burst 30–65 W.
  All-in ≈ **US$15 – 20 / year** at typical rates — basically free.
- **PC (RTX 5060 Ti 180 W + CPU/RAM, ~250 W loaded):** ~**$3 – 8 / month**
  if running ~4 h/day under load; ~$15 – 25 / month if on 24/7. Sleep or
  Wake-on-LAN keeps this realistic.
- **Per-token electricity:** literature consistently lands on
  **$0.001 – $0.01 per 1 M tokens** for consumer GPUs like the 5060 Ti on
  API-style (short-context) workloads; RAG-32k workloads push that to
  $0.14 – $0.22 / MTok because of prompt-processing overhead.

### 8.4 Break-even math (hardware treated as sunk cost — you already own both)

| Cloud alternative | $ saved per 1 M tokens by going local | Tokens/day to save $1 |
|---|---|---|
| Sonnet 4.6 (no cache) | ~$5.00 | 200 k |
| Sonnet 4.6 (cached / batch) | ~$2.50 | 400 k |
| Flash Lite | ~$0.46 | 2.2 M |

Translation: against **Sonnet**, any serious agent use pays back
immediately — you save roughly $5 per million tokens you don't send.
Against **Flash Lite**, you need sustained volume (several million tokens
a day) before electricity savings dominate; below that it's essentially
free to just call Google.

If you amortize the 5060 Ti alone (~$430, 3-year life = ~$143/yr) the
break-evens don't move much: you cover its depreciation by displacing
~30 M tokens of Sonnet or ~310 M tokens of Flash Lite per year.

### 8.5 What electricity doesn't price in

1. **Quality gap.** For agent work the ranking is roughly
   **Sonnet 4.6 > Qwen3-Coder-Next (local) > Gemini 3.1 Flash Lite**.
   Flash Lite is built for speed and cost, not long-horizon tool loops;
   don't assume same-tier performance just because the price looks
   attractive. Sonnet's lead on complex multi-tool flows is the main
   reason to pay its rate.
2. **Latency / throughput.** 5060 Ti does ~30 tok/s on 14 B Q4, ~20–40
   tok/s on Qwen3-Coder-Next via offload. Sonnet and Flash Lite don't
   care if you fire 50 parallel requests — local does.
3. **Ops cost.** Model swaps, GGUF updates, driver pain, VRAM-tuning,
   template bugs for tool-calling — none of this shows up on the power
   bill but it's the real cost of local.
4. **Privacy / air-gap.** Local wins by definition when the data can't
   leave the box.

### 8.6 Practical recommendation by traffic tier

| Daily volume | Best fit |
|---|---|
| < 500 k tokens/day | **Flash Lite**. Ops-free, ~$0.25/day ceiling. Local adds complexity for no meaningful saving. |
| 500 k – 5 M / day | **Hybrid**: cheap turns on Flash Lite or local Qwen3.5-9B (Mac), hard turns on Sonnet. |
| > 5 M / day, steady | **Local Qwen3-Coder-Next on the PC** as the default; Sonnet as a scalpel for the hardest turns. The electricity math becomes obviously dominant. |
| Privacy-sensitive | Local regardless of volume. |

---

## 9. The wider Chinese open-source landscape — is there something *better* than Qwen3-Coder-Next for this box?

Short answer: **yes, two serious contenders — MiniMax M2.x and GLM-4.5-Air — and one honorable mention (DeepSeek R1 distill 32B).** Each is worth trying before settling on Qwen.

Context for why this matters: the user tried openclaw with Gemini 3.1 Flash Lite and found it disappointing. That's expected — Flash Lite is tuned for speed/cost, not long-horizon multi-tool loops. The models below are *explicitly* trained for agent/coding workloads, which is exactly what openclaw needs.

### 9.1 GLM-4.5-Air — the realistic agent pick (Zhipu / z.ai)
- **Size:** 106 B total / ~12 B active (MoE). GLM-4.6 (357 B) is too big; GLM-5 (744 B) is way too big. **Air is the variant that fits consumer hardware.**
- **Designed for:** "Agentic, Reasoning, Coding" (ARC). Works out of the box with Claude Code, Cline, Roo Code, Kilo Code, OpenCode.
- **Fit on the PC:** Q4 MoE quant ≈ 55–60 GB. With `-ot ".ffn_.*_exps.=CPU"` offload of MoE layers to the 128 GB DDR5 it runs end-to-end; expect **~8–15 tok/s** based on 3090+128GB reports.
- **Context:** 128 K. Plenty for agent loops.
- **Verdict:** strongest agent-optimized model that comfortably fits your RAM budget. If you want one local model to replace Flash Lite in openclaw, this is the first thing to try. GLM-4.6 (full) is better but painful on 16 GB VRAM.

### 9.2 MiniMax M2 / M2.5 / M2.7 — the SWE-bench champion
- **Size:** 230 B total / **10 B active** (MoE). Purpose-built for "max coding and agentic workflows."
- **Benchmarks:** MiniMax M2.1 scored **74 % on SWE-bench Verified** — higher than most open models, closes in on Sonnet-class.
- **Fit on the PC:** **tight.** The smallest quants need **≥121 GB RAM**; your 128 GB is the bare minimum and you'll be swapping nothing else. Expect **~14 tok/s at Q6** on consumer HW per independent testing.
- **Engine:** SGLang is the recommended server; vLLM works too (`vllm>=...` recent). Native API runs ~100 tok/s so you'll feel the local gap.
- **Verdict:** if pure agent/coding quality is the goal and you can live with 128 GB being nearly full, this is the upgrade path above GLM-Air. Start with M2.7 (latest, NVIDIA-optimized). If MiniMax open-weights M2.5 or M2.7 at a smaller Air-style size, that'll become the obvious pick.

### 9.3 DeepSeek V3.2 (and R1 distills) — the reasoning specialist
- **DeepSeek V3.2 full:** 671 B total, ~37 B active. Too big even for 128 GB at reasonable quality — on RTX 4090 (24 GB) you get 18–25 tok/s on the active set, on a 5060 Ti it'll be worse because of the tighter VRAM and offload bandwidth.
- **DeepSeek R1 distilled models** (8 B / 14 B / **32 B**): these are what you actually want. The **32 B distill** fits the 5060 Ti at Q4 with partial offload and gives you reasoning-tuned behavior without paying 670 B model tax. Not as agent-tuned as GLM or MiniMax, but excellent on hard problem-solving turns.
- **Verdict:** keep R1-32B-distill in the hybrid setup as a "reasoner" endpoint the agent can call for planning turns.

### 9.4 Kimi K2 / Moonshot — skip for local
- Trillion-param, runs at **~8.5 tok/s at Q3** even on a 3090+128GB class machine. Its party trick is 200–300 sequential tool calls without context rot, which is genuinely impressive — but not at 8 tok/s. **Use via API if you ever need its specific strength.**

### 9.5 Summary table — what actually fits your PC (5060 Ti 16 GB + 128 GB DDR5)

| Model | Total / active | Agent tuning | Realistic quant | Expected tok/s | Notes |
|---|---|---|---|---|---|
| **Qwen3-Coder-Next** | 80 B / 3 B | ★★★★★ (explicit) | Q4_K_M or Q2_K_XL | 20–40 | Already in doc. Strong all-rounder. |
| **GLM-4.5-Air** | 106 B / 12 B | ★★★★★ (ARC) | Q4 MoE + offload | 8–15 | Best "comfortable fit" agent model. |
| **MiniMax M2.7** | 230 B / 10 B | ★★★★★ (SWE-bench 74 %) | Q4 w/ full offload | ~14 | Top quality, tight on 128 GB RAM. |
| **DeepSeek R1 distill 32B** | 32 B (dense) | ★★★ (reasoning) | Q4_K_M + offload | 18–25 | Great reasoner, not agent-specialist. |
| **gpt-oss-20B MXFP4** | 20 B MoE | ★★★★ | native MXFP4 | **~488 short-ctx** | Fast fallback. Non-Qwen template. |
| Qwen3.5-27B (dense) | 27 B | ★★★ | Q4 + offload | ~10 | Not really agent-tuned; skip. |
| DeepSeek V3.2 full | 671 B / 37 B | ★★★★ | Q2 + heavy offload | ~5–8 | Too painful locally; use API. |
| Kimi K2 | ~1 T MoE | ★★★★★ (tool chains) | Q3 + offload | ~8 | API only in practice. |

### 9.6 Why did Flash Lite disappoint and what actually fixes it?

Three causes, ordered by likelihood:
1. **Model capability mismatch.** Flash Lite is sub-Sonnet-4.6 class; openclaw's tool-call reasoning sits near the ceiling of what Flash Lite can handle. **Fix:** try a model trained for agents — GLM-4.5-Air, MiniMax M2, or Qwen3-Coder-Next — or jump to Sonnet 4.6 for the hard turns.
2. **Tool-call parsing quirks.** Gemini's function-call format is not identical to Anthropic's `tool_use` blocks; adapter bugs sometimes silently drop arguments. **Fix:** check openclaw's Gemini adapter, enable verbose logging on tool calls, compare to OpenAI-compatible endpoints where the format is better-trodden.
3. **Prompt style.** Agent prompts written for Claude/GPT-4 class models often over-rely on implicit reasoning that smaller models miss. **Fix:** add explicit step/plan scaffolding in the system prompt if you stick with smaller models.

### 9.7 Revised recommendation

Updated ordering for your specific hardware, agent-oriented workload, and the Flash Lite disappointment:

1. **Try GLM-4.5-Air locally first** — best balance of agent quality, hardware fit, and setup complexity. Serve via llama.cpp with `--jinja` or vLLM. This is the most likely win for openclaw.
2. **A/B it against Qwen3-Coder-Next** — different strengths; Qwen is lighter, GLM is arguably smarter at long-horizon tool use. Pick per task.
3. **Keep Sonnet 4.6 as the fallback API** — not Flash Lite. The quality gap is where your openclaw problem actually lives. If cost matters, use Sonnet only for turns that fail locally.
4. **Don't bother with Flash Lite for agents.** If you want a cheap API backstop, **z.ai's GLM API** is the same GLM family at low rates, or **DeepSeek's API** for V3.2 — both are cheaper than Sonnet and much more agent-capable than Flash Lite.
5. **MiniMax M2.7 is your upgrade target** if GLM-Air isn't enough. Plan for a RAM upgrade or comfortable 128 GB headroom before attempting it.

---

## 10. The recommended 3-tier stack (final answer)

This is the concrete setup: **1st choice / 2nd choice / 3rd choice** for
each tier, so you can pick a default and have fallbacks lined up before
you start downloading weights.

### Tier 1 — The PC (primary brain for openclaw)

Heavy agent turns land here. The PC has 16 GB VRAM + 128 GB DDR5 —
enough for a mid-sized MoE model with CPU offload.

| Rank | Model | Size (total / active) | Why | Expected tok/s | Serve with |
|---|---|---|---|---|---|
| 🥇 **1st** | **GLM-4.5-Air** | 106 B / 12 B MoE | Explicitly trained for agent + coding workflows (ARC). Comfortable RAM fit (~55–60 GB at Q4). Works out-of-the-box with Claude Code / Cline / openclaw-class scaffolds. Best quality-to-fit ratio on your hardware. | 8–15 | `llama.cpp --jinja` or vLLM |
| 🥈 **2nd** | **Qwen3-Coder-Next** | 80 B / 3 B MoE | Lighter active param count → faster. 256 K context. Very mature tool-calling via Hermes template. Use when GLM-Air is too slow or for longer contexts. | 20–40 | `llama.cpp --jinja`, vLLM, or Ollama |
| 🥉 **3rd** | **DeepSeek R1 distill 32B** | 32 B dense | Keep as a "reasoning" endpoint for planning-heavy turns. Not agent-tuned, but strong at analytical thinking. Fits VRAM at Q4 with partial offload. | 18–25 | `llama.cpp` or Ollama |

**Default openclaw endpoint:** GLM-4.5-Air. Swap to Qwen3-Coder-Next if
you hit tool-parsing bugs or need faster throughput.

### Tier 2 — The Mac mini (always-on light endpoint)

Runs 24/7 at ~5 W idle. Role: fast small-model for routing, classification,
simple tool calls, and as a fallback when the PC is asleep. Hard-capped
at ~9 B dense models by the 16 GB unified memory.

| Rank | Model | Size | Why | Expected tok/s | Serve with |
|---|---|---|---|---|---|
| 🥇 **1st** | **Qwen3.5-9B** | 9 B dense | Beats models 3× its size on reasoning. Solid tool-calling. Fits in ~6.6 GB Q4 → leaves headroom for context. Native MLX build is fastest on Apple Silicon. | 40–50 (MLX), 25–35 (Ollama) | **MLX-LM** or Ollama |
| 🥈 **2nd** | **Qwen2.5-7B-Instruct** | 7 B dense | Older but battle-tested. Most reliable tool-calling behavior in the small-model tier. Use if Qwen3.5-9B shows edge-case bugs. | 35–45 | MLX-LM or Ollama |
| 🥉 **3rd** | **Phi-4 / Gemma 3 (9B)** | ~9 B dense | Non-Qwen alternatives; useful if you want diversity in ensemble routing or hit a license concern. Gemma 3 is strong at multilingual. | 30–45 | MLX-LM or Ollama |

**Default Mac role:** Qwen3.5-9B served via MLX. Mac also keeps the
existing `claude-local-calls` proxy running so you can always reach
Claude through your subscription.

### Tier 3 — API fallback (for the hardest turns)

When local models fail a turn, or on production-critical runs, bounce
the request to a remote model. Explicitly **not Flash Lite** — that's
what disappointed you; skip the trap of "cheaper Gemini" as the fallback.

| Rank | Model | Price (blended 5:1) | Why | Use when |
|---|---|---|---|---|
| 🥇 **1st** | **Claude Sonnet 4.6** | ~$5.00 / MTok (or ~$3.20 with cache) | Highest-quality agent model in the lineup. With 1 M ctx and prompt caching, fits any openclaw workload. Worth the price for hard turns. | Any turn local models fail, or when quality matters more than cost. |
| 🥈 **2nd** | **z.ai GLM-5.1 / GLM-4.6 API** | Dramatically cheaper than Sonnet (~$0.50–1.50 / MTok blended, varies) | Same GLM family as your Tier-1 local pick, so prompt portability is high. Much more agent-capable than Flash Lite. | Routine agent fallbacks when cost matters more than top-end quality. |
| 🥉 **3rd** | **DeepSeek V3.2 API** | Cheap per MTok; strong reasoning | Excellent fallback for reasoning / planning turns specifically. Less agent-tuned than GLM or Sonnet. | Hard analytical turns; second-opinion check on locally-produced plans. |

**Skip Gemini 3.1 Flash Lite as a fallback.** It's the model class that
failed you originally — cheaper isn't better if quality is the problem.
If you want Gemini specifically, step up to **Gemini 3 Flash** or
**Gemini 3 Pro** tier where the agent reasoning actually holds.

### 10.4 Routing logic (how the three tiers work together)

A simple rule-of-thumb router, from cheapest to most expensive:

1. **Classify / route / short completions** → **Mac mini (Qwen3.5-9B)**.
   The Mac is always-on; no spin-up cost.
2. **Default agent turns (tool calls, code edits, multi-step reasoning)**
   → **PC (GLM-4.5-Air)**. Wake-on-LAN if asleep; keep it warm when
   openclaw is active.
3. **Failed turn, critical turn, or quality check** → **Sonnet 4.6 API**.
   Budget this as "scalpel only" — it's 500× the electricity cost, so
   use it deliberately.

The `claude-local-calls` proxy on the Mac remains the single entry
point: it can fan out to the PC or the API based on request metadata
(e.g. model name, a header, or message length).

### 10.5 Shopping list to actually get started

| What | Where | Notes |
|---|---|---|
| GLM-4.5-Air GGUF | `unsloth/GLM-4.5-Air-GGUF` on HF | Grab Q4_K_M first; fall back to Q2_K_XL if RAM pressure shows. |
| Qwen3-Coder-Next GGUF | `unsloth/Qwen3-Coder-Next-GGUF` on HF | Q4_K_M or Q2_K_XL. |
| Qwen3.5-9B (MLX) | `mlx-community/Qwen3.5-9B-Instruct-4bit` on HF | Or `ollama pull qwen3.5:9b`. |
| DeepSeek R1 32B distill | `unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF` on HF | Optional Tier-1 #3. |
| llama.cpp | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | Build with CUDA for the PC, Metal for the Mac. |
| Ollama | ollama.com | Fast path if you don't want to compile anything. |
| Anthropic API key | console.anthropic.com | For the Tier-3 fallback. |
| z.ai API key | z.ai | Optional cheaper Tier-3. |

---

## 11. Bottom line

- **1st choice, PC:** **GLM-4.5-Air** — the agent-tuned MoE that fits.
- **2nd choice, Mac mini:** **Qwen3.5-9B via MLX** — always-on, cheap, fast.
- **3rd choice, API fallback:** **Claude Sonnet 4.6** (with prompt caching).
- **Don't use:** Gemini 3.1 Flash Lite for agent work (that's what
  failed you), or anything >35 B dense on the Mac (won't fit).
- **Mac mini's honest role:** 24/7 always-on light endpoint + routing
  layer, not the agent brain.
- **PC's honest role:** the actual LLM workhorse. 128 GB RAM is what
  unlocks 100 B+ MoE models on a 16 GB GPU.
- **Tool calling works today** in llama.cpp (Hermes template, `--jinja`),
  vLLM, and Ollama — pick the one that matches your comfort level.

---

## Sources

- [QwenLM/Qwen3 (GitHub)](https://github.com/QwenLM/Qwen3)
- [QwenLM/Qwen3.6 (GitHub)](https://github.com/QwenLM/Qwen3.6)
- [QwenLM/Qwen3-VL (GitHub)](https://github.com/QwenLM/Qwen3-VL)
- [QwenLM/Qwen3-Coder (GitHub)](https://github.com/QwenLM/Qwen3-Coder)
- [QwenLM/Qwen-Agent (GitHub)](https://github.com/QwenLM/Qwen-Agent)
- [Qwen/Qwen3-Coder-Next (Hugging Face)](https://huggingface.co/Qwen/Qwen3-Coder-Next)
- [Qwen3-Coder-Next: Pushing Small Hybrid Models — qwen.ai blog](https://qwen.ai/blog?id=qwen3-coder-next)
- [Qwen3-Coder-Next: The Complete 2026 Guide (DEV)](https://dev.to/sienna/qwen3-coder-next-the-complete-2026-guide-to-running-powerful-ai-coding-agents-locally-1k95)
- [Qwen3.5 — How to Run Locally (Unsloth docs)](https://unsloth.ai/docs/models/qwen3.5)
- [Qwen3-Coder-Next — How to Run Locally (Unsloth docs)](https://unsloth.ai/docs/models/qwen3-coder-next)
- [unsloth/Qwen3-Coder-Next-GGUF (Hugging Face)](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF)
- [Function Calling — Qwen docs](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [llama.cpp function-calling docs (GitHub)](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)
- [vLLM Tool Calling docs](https://docs.vllm.ai/en/stable/features/tool_calling/)
- [Qwen3-VL Usage Guide — vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
- [Ollama is now powered by MLX on Apple Silicon](https://ollama.com/blog/mlx)
- [RTX 5060 Ti 16GB: Overlooked Sweet Spot — CraftRigs](https://craftrigs.com/articles/rtx-5060-ti-16gb-budget-local-llm/)
- [RTX 5060 Ti 16GB Benchmarks — Hardware Corner](https://www.hardware-corner.net/gpu-llm-benchmarks/rtx-5060-ti-16gb/)
- [Best Local LLMs for Every RTX 50 Series GPU — apxml](https://apxml.com/posts/best-local-llms-for-every-nvidia-rtx-50-series-gpu)
- [Qwen3-Coder 30B Hardware Requirements — Arsturn](https://www.arsturn.com/blog/running-qwen3-coder-30b-at-full-context-memory-requirements-performance-tips)
- [Running Qwen 3.5 35B-A3B on 5060 Ti — NJannasch.Dev](https://njannasch.dev/blog/running-qwen-3-5-35b-a3b-on-5060-ti/)
- [Mac Mini M4 for AI 2026 — Compute Market](https://www.compute-market.com/blog/mac-mini-m4-for-ai-apple-silicon-2026)
- [Mac Mini M4 16GB Local LLM Benchmarks — Like2Byte](https://like2byte.com/mac-mini-m4-16gb-local-llm-benchmarks-roi/)
- [Best LLM for Mac Mini M4 16GB (2026) — ModelFit](https://modelfit.io/blog/best-llm-mac-mini-m4-16gb/)
- [Installing Qwen 3.5 on Apple Silicon Using MLX (DEV)](https://dev.to/thefalkonguy/installing-qwen-35-on-apple-silicon-using-mlx-for-2x-performance-37ma)
- [Best Small AI Models to Run with Ollama (2026) — Local AI Master](https://localaimaster.com/blog/small-language-models-guide-2026)
- [Best Mac Mini for Running Local LLMs and OpenClaw — Starmorph](https://blog.starmorph.com/blog/best-mac-mini-for-local-llms)
- [Mac Mini OpenClaw Setup Guide — startwithopenclaw.com](https://startwithopenclaw.com/mac-mini/)
- [I run local LLMs in one of the world's priciest energy markets — XDA](https://www.xda-developers.com/run-local-llms-one-worlds-priciest-energy-markets/)
- [Gemini 3.1 Flash Lite — Google blog](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-flash-lite/)
- [Gemini Developer API pricing — Google AI](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini 3.1 Flash Lite Preview — OpenRouter](https://openrouter.ai/google/gemini-3.1-flash-lite-preview)
- [Claude API pricing — Anthropic docs](https://platform.claude.com/docs/en/about-claude/pricing)
- [Claude Sonnet 4.6 — OpenRouter](https://openrouter.ai/anthropic/claude-sonnet-4.6)
- [Anthropic API pricing guide 2026 — Finout](https://www.finout.io/blog/anthropic-api-pricing)
- [Local LLMs vs Cloud APIs TCO 2026 — SitePoint](https://www.sitepoint.com/local-llms-vs-cloud-api-cost-analysis-2026/)
- [Private LLM Inference on Consumer Blackwell GPUs (arXiv)](https://arxiv.org/html/2601.09527v1)
- [zai-org/GLM-4.6 (Hugging Face)](https://huggingface.co/zai-org/GLM-4.6)
- [zai-org/GLM-4.5 (GitHub)](https://github.com/zai-org/GLM-4.5)
- [zai-org/GLM-5 (GitHub)](https://github.com/zai-org/GLM-5)
- [GLM-4.6 Run Locally Guide — Unsloth](https://docs.unsloth.ai/models/glm-4.6-how-to-run-locally)
- [GLM-4.6: An Open-Source AI for Coding — IntuitionLabs](https://intuitionlabs.ai/articles/glm-4-6-open-source-coding-model)
- [MiniMax-AI/MiniMax-M2 (GitHub)](https://github.com/MiniMax-AI/MiniMax-M2)
- [MiniMaxAI/MiniMax-M2 (Hugging Face)](https://huggingface.co/MiniMaxAI/MiniMax-M2)
- [MiniMax M2.7 on NVIDIA Platforms — NVIDIA blog](https://developer.nvidia.com/blog/minimax-m2-7-advances-scalable-agentic-workflows-on-nvidia-platforms-for-complex-ai-applications/)
- [MiniMax M2 hardware requirements — Hardware Corner](https://www.hardware-corner.net/what-hardware-minimax-m2-7/)
- [MiniMax-M2 vLLM Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/MiniMax/MiniMax-M2.html)
- [How to Run DeepSeek V3.2 Locally — Pulse Mark](https://pulsemark.ai/run-deepseek-v3-2-locally-setup-guide-2026/)
- [DeepSeek R1 Local Setup — Local AI Master](https://localaimaster.com/blog/deepseek-r1-local-setup-guide)
- [Kimi K2 Thinking vs MiniMax M2 — kimi-k2.org](https://kimi-k2.org/blog/17-kimi-k2-thinking-vs-minimax-m2-en)
- [Best Open Source LLM for Agent Workflow 2026 — SiliconFlow](https://www.siliconflow.com/articles/en/best-open-source-LLM-for-Agent-Workflow)
- [Top Chinese Open-Source LLMs 2026 — Second Talent](https://www.secondtalent.com/resources/chinese-open-source-llms-ai-leaders/)
