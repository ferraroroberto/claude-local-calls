# GLM-4.5-Air performance assessment — 2026-04-20

Follow-up to [20260420-hub-with-qwen-and-glm.md](20260420-hub-with-qwen-and-glm.md).
Morning test: ~400 input tokens / ~1000 output tokens in ~120 s on the
Windows PC (5060 Ti 16 GB + 128 GB RAM), CPU pinned at 100 %, GPU
barely loaded. Concern: is this fast enough for openclaw, and is the
CPU/GPU balance a misconfiguration?

## TL;DR

- **The measured ~10 tok/s is not a misconfiguration — it is the
  expected floor** for GLM-4.5-Air with the current MoE-full-offload
  config. The research doc predicted 8–15 tok/s for exactly this
  setup; we are at the low end of that range.
- **CPU at 100 % / GPU at ~15 % is by design** for the current
  `-ot ".ffn_.*_exps.=CPU"` flag. All expert tensors (the bulk of
  compute for MoE) run on CPU. The GPU only holds attention + KV
  cache.
- **Headroom exists.** The current launch args leave 7–9 GB of VRAM
  unused, no FlashAttention, no KV quantisation, no thread pinning,
  no mlock. Realistic target after tuning: **18–25 tok/s** on GLM,
  which turns a 120 s response into 50–65 s.
- **Openclaw usability verdict.** Even at the optimistic 25 tok/s, GLM
  will feel slow for tight agent loops (each tool turn pays the
  latency again). Route most openclaw turns to **Qwen3.5-9B (~65
  tok/s)**, and reserve GLM for tasks that genuinely need the bigger
  model. This is what the research already recommended and still
  holds.

## Where the 120 s went (reconstruction)

We did not log timings on the run itself. But we can bound it.

| Phase | Work | Likely time |
|---|---|---|
| Prompt processing (`pp`) | ~400 input tokens | 1–3 s (GPU-bound; fast) |
| Token generation (`tg`) | ~1000 output tokens @ ~10 tok/s | **~100 s** (CPU-bound; the dominant cost) |
| Subprocess + routing overhead | hub → llama-server httpx roundtrip | <0.5 s |
| Model load (if cold) | ~55 GB GGUF from disk via mmap | **up to 20 s on first call** if the OS page cache is cold |

The dominant term is generation. A cold first call also eats an extra
10–20 s of disk-to-page-cache warmup that won't repeat on subsequent
calls (or on any call after `--mlock`).

**Action**: turn on llama-server's built-in timing log so we can see
`prompt eval time`, `eval time`, and `tokens per second` separately
on every request. llama-server prints them by default; make sure
stdout is being captured. Our ring buffer already captures it
([`src/llama_process.py`](../../src/llama_process.py)); the
`Logs` view in the Models tab will show them.

## Why CPU is at 100 % and GPU is idle

Current args (`config/models.yaml`):

```yaml
args:
  - "--jinja"
  - "-ngl"
  - "99"
  - "-ot"
  - ".ffn_.*_exps.=CPU"
  - "-c"
  - "16384"
```

`-ngl 99` says "put every layer on GPU". `-ot ".ffn_.*_exps.=CPU"`
then *overrides* that just for expert tensors — and for an MoE model
the experts are 80–90 % of total parameters. Net effect: only
attention weights, the router, embeddings, and the KV cache sit on
GPU. Every generated token has to move hidden states from GPU to CPU,
run the experts on CPU, and move the result back. CPU does the hot
inner loop, so CPU is pegged and GPU is nearly idle. This is the
intended trade-off — the alternative is "cannot load at all on 16 GB".

The actual question is: **are we leaving GPU capacity on the table?**
And the answer is yes. The install verification showed:

- `CUDA0 model buffer size = 3990.17 MiB` (≈ 4 GB)
- `CUDA0 KV buffer size   = 2944.00 MiB` (≈ 3 GB)
- Total VRAM used ≈ 7–8 GB

The 5060 Ti has 16 GB. **We have 7–9 GB of VRAM doing nothing.** We
can spend that on pinning the *first N* expert layers to GPU, which
proportionally cuts the CPU-side work.

## Optimizations — ranked by expected payoff

### Tier 1 — change `config/models.yaml` and restart (no downloads, no reboots)

1. **Hybrid MoE offload: pin early expert layers to GPU.** Replace the
   blanket `-ot ".ffn_.*_exps.=CPU"` with a two-rule regex that keeps
   the first ~12 layers' experts on GPU and pushes the rest to CPU.
   GLM-4.5-Air has 46 layers; each layer's expert block is roughly
   (55 GB / 46) ≈ 1.2 GB. With ~8 GB VRAM free and KV quantisation
   below, we can fit 6–10 layers of experts on GPU. Rough starting
   point to tune:

   ```yaml
   - "-ot"
   - "blk\\.([0-9]|1[01])\\.ffn_.*_exps\\.=CUDA0,ffn_.*_exps\\.=CPU"
   ```

   Expected: **+30–60 % tok/s** (the 6–10 fastest layers no longer
   pay PCIe bounce). Tune `([0-9]|1[01])` upward until llama-server
   logs an OOM, then back off by 1.

2. **`--flash-attn` (alias `-fa`).** FlashAttention on Blackwell gives
   a free 10–25 % speed-up on attention and frees ~0.5–1 GB of KV
   buffer. Blackwell (5060 Ti) supports it natively in the CUDA 13
   builds we ship. Just add `-fa`.

3. **Quantise the KV cache.** `--cache-type-k q4_0 --cache-type-v q4_0`
   cuts the 3 GB KV buffer in ~half with negligible quality impact
   on chat workloads. That freed VRAM feeds rule #1 (more expert
   layers on GPU). Use `q8_0` if you notice quality regressions.

4. **Lock the model in RAM.** `--mlock`. With 128 GB of RAM this is
   free — prevents the OS from paging out hot expert tensors under
   memory pressure from other apps, which manifests as sporadic
   stalls mid-generation.

5. **Pin threads to physical cores.** Add `-t <N>` where N = physical
   core count (*not* threads). llama.cpp's autodetection on Windows
   sometimes counts SMT siblings, which slows MoE compute by 10–20 %.
   On a 16-core CPU: `-t 16`. Pair with `-tb <N>` for batch threads.

6. **Larger compute ubatch for prompt processing.** `-b 512 -ub 512`
   (or `-ub 1024` if VRAM allows). Matters more when inputs get
   longer; openclaw tool turns can be several kB.

Putting 1–6 together, the launch args become:

```yaml
args:
  - "--jinja"
  - "-ngl"
  - "99"
  - "-ot"
  - "blk\\.([0-9]|1[01])\\.ffn_.*_exps\\.=CUDA0,ffn_.*_exps\\.=CPU"
  - "-fa"
  - "--cache-type-k"
  - "q4_0"
  - "--cache-type-v"
  - "q4_0"
  - "--mlock"
  - "-t"
  - "16"    # physical cores — adjust to your CPU
  - "-tb"
  - "16"
  - "-b"
  - "512"
  - "-ub"
  - "512"
  - "-c"
  - "16384"
  - "--alias"
  - "glm-4.5-air"
```

Realistic combined expectation: **~18–25 tok/s** on GLM (roughly 2–2.5×
the observed rate). Qwen dense already runs at ~65 tok/s so the
payoff there is smaller, but `-fa`, `--mlock`, and KV quant are still
worth adding for Qwen too.

### Tier 2 — system-level (check, but likely cheap wins)

7. **XMP / EXPO enabled in BIOS.** For MoE CPU offload, system RAM
   bandwidth *is* the bottleneck. DDR5 at default JEDEC 4800 MT/s
   ≈ 76 GB/s dual-channel; at XMP 6000 MT/s ≈ 96 GB/s. A 25 %
   bandwidth increase ≈ ~20 % tok/s increase for CPU-resident
   experts. Free speed-up if XMP is off — check BIOS.

8. **Both DIMM channels populated correctly.** 128 GB is usually 2×64
   or 4×32. On consumer boards, 4-DIMM configs often fall back to
   lower clocks. 2×64 populated in the A2+B2 slots is usually the
   fastest config.

9. **Windows power plan = "High performance" (or Ultimate).** Balanced
   parks cores and throttles ring bus frequency between tokens — you
   can see it as the per-core clock dropping between bursts.

10. **NVIDIA driver fresh.** Blackwell support on launch drivers has
    shipped multiple perf fixes; make sure the driver is ≥ the one
    released at the same time as the CUDA 13.1 llama.cpp build we
    fetch. `nvidia-smi` in the Install tab.

11. **Nothing else stealing VRAM/CPU.** Chrome with hardware
    acceleration, VS Code, Discord, and any game launcher reserve
    VRAM. `nvidia-smi` will show it. Close what you can before long
    runs.

12. **Hardware-accelerated GPU scheduling ON** (Settings → Display →
    Graphics → Change default graphics settings). Helps reduce CPU
    overhead of CUDA submit.

### Tier 3 — bigger changes (only if tier 1 + 2 aren't enough)

13. **Drop to Q3_K_M or IQ3_XXS quant of GLM-4.5-Air.** Smaller weights
    → fewer bytes moved per token → faster. Expected quality hit:
    small but non-zero. Only worth it if tier 1 didn't get you where
    you need.

14. **Switch GLM to Qwen3.5-35B-A3B (or Qwen3-Coder-Next 80B/3B
    active).** 3 B active params vs GLM's 12 B active = ~4× faster
    token gen on CPU offload, at the cost of slightly less raw
    capability. `Qwen3-Coder-Next` is explicitly agent-tool-tuned and
    was flagged in the original research. If openclaw throughput
    matters more than peak reasoning, this is the bigger lever than
    any flag tweak.

15. **Speculative decoding with a small draft model.** llama.cpp
    supports `--model-draft` with a small Qwen3 for GLM. Can be
    1.5–2× on structured/predictable output; variable on open-ended
    generation. Fiddly to tune; try only after tier 1.

## Recommendations for openclaw specifically

Openclaw is agentic — each turn is a small generation that triggers a
tool call that feeds back into another small generation. Two
properties of that workload:

- **Time-to-first-token matters more than raw tok/s.** Short replies
  dominate. Even at 20 tok/s, a 50-token tool decision is 2.5 s —
  tolerable. A 400-token reasoning block is 20 s — painful.
- **Many short turns multiply.** A 10-tool-call task × 20 s/turn =
  3.5 minutes of wall time just waiting on the model.

Concrete guidance:

1. **Route openclaw to `qwen3.5-9b` by default.** At ~65 tok/s it is
   the model that makes agents feel responsive on this hardware. Use
   GLM only when you explicitly want the bigger model for a hard
   task.
2. **Ship streaming (SSE) before heavy openclaw use.** It's already in
   the backlog in the README. llama-server emits SSE natively; the
   hub just has to forward it. With streaming, time-to-first-token
   is all the user perceives, and GLM at 10 tok/s feels noticeably
   better even without any flag changes.
3. **Warm the backend before the run.** The first request after cold
   start pays model-load cost (~10–20 s on GLM). A trivial `"hi"`
   ping at session start pre-populates the page cache and any KV
   state.
4. **Cap `max_tokens` aggressively** in openclaw tool-decision prompts
   so the model can't accidentally spend 60 s generating a preamble.

## What to do next, in order

1. Edit [`config/models.yaml`](../../config/models.yaml) with the tier-1
   flags. Restart GLM from the Models tab. Read the llama-server log
   for the new VRAM / CUDA buffer sizes and the `eval time … tokens
   per second` line after a prompt.
2. If VRAM fits, bump the `blk\\.([0-9]|1[01])` range up one at a
   time until the next start OOMs, then back off by one.
3. Check BIOS XMP, Windows power plan, NVIDIA driver.
4. Re-run the morning test (400 in / 1000 out). Record tok/s.
5. If still <20 tok/s on GLM, consider Qwen3-Coder-Next as a GLM
   replacement (pure registry change — add the model, download
   GGUF, enable it on the host) rather than deeper tier-3 tuning.
6. Prioritise streaming on the hub — that one change changes the
   *perceived* speed more than any backend tweak.

## Not recommended

- Lower `-c` below 16384 to save VRAM. Openclaw needs the context.
- Turning off `--jinja`. We'd lose native tool calling, which is the
  main reason to use the OpenAI-shape path.
- Moving everything to CPU (`-ngl 0`). Attention benefits from GPU
  even for MoE models; `-ngl 0` is slower, not faster.
