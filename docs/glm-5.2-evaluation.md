# GLM-5.2 local-backend evaluation — 2026-06-25

Spike for [#141](https://github.com/ferraroroberto/local-llm-hub/issues/141).
Coding-first assessment of Z.ai's **GLM-5.2** (released 2026-06-13) as a local
hub backend on the reference box: does it supersede anything in the active
rotation, and does it run on the hardware already here?

**Decision: NO-GO (does not fit the hardware).** GLM-5.2 ships as a single
**744B-parameter MoE** with no smaller sibling. The smallest usable quant needs
~239–245 GB of combined RAM+VRAM; the reference box has **16 GB VRAM + 128 GB
RAM ≈ 144 GB total**, so no quant loads at a quality worth using. The model is an
excellent *coder* — that's not the problem; the size is. Revisit if/when a
GLM-5.2-**Air**/**Flash** (~80–120 B MoE) ships (see
[When to revisit](#when-to-revisit)).

---

## What GLM-5.2 is

Z.ai's flagship open-weights coding model, announced **2026-06-13**, weights
**MIT-licensed** on Hugging Face under `zai-org/GLM-5.2` (plus a `GLM-5.2-FP8`
build). Architecture: **744B total parameters / ~40B active** Mixture-of-Experts
on the GLM-5 backbone, **1M-token** usable context (~131K output cap), with a
dual thinking-effort system (High / Max).

On coding it is genuinely frontier-class for an open model: ~81.0 vs 62.0 (GLM-5.1)
on Terminal-Bench 2.1, 62.1 vs 58.4 on SWE-bench Pro, edging out GPT-5.5 on
several long-horizon coding benchmarks at a fraction of the cost. So the verdict
below is **"right model, wrong size for this box,"** not "weak model."

## There is no smaller variant (yet)

The hub's only realistic path to a >100B MoE on a 16 GB card is the
expert-CPU-offload pattern already proven by the demoted `glm-4.5-air`
(`-ot ".ffn_.*_exps.=CPU"`, `launchers/run_model.bat glm`) — but that worked because
GLM-4.5-Air is ~106B-A12B and its **2-bit-ish quant fits in 128 GB RAM**.

GLM-5.2 has **no Air or Flash variant**. The community is actively requesting one
(open threads on `zai-org/GLM-5.2` discussions asking for an 80–120 B Air / a
35–120 B Flash), but as of this writing zai-org has published only the full 744B
model and its FP8 build. So the offload trick has nothing small enough to apply
to.

## Fit math vs. the reference box

Hardware (`config/machine_specs.yaml`): RTX 5060 Ti **16 GB VRAM**, Ryzen 7
7800X3D, **128 GB DDR5** → ~**144 GB** total addressable for weights+KV under
llama.cpp MoE CPU-offload.

| GLM-5.2 quant | Memory needed (weights) | Fits ~144 GB? |
| --- | --- | --- |
| Q4_K_M | ~476 GB | ❌ |
| UD-IQ2_M (2-bit dynamic) | ~239–245 GB | ❌ |
| ~1-bit | ~223 GB | ❌ |
| 8-bit | ~810 GB | ❌ |

The 2-bit dynamic quant (Unsloth UD-IQ2_M, the smallest build that retains
useful quality — ~82% of full-model accuracy) is **~239 GB on disk and needs
~245 GB RAM+VRAM combined** to run. Even a 1-bit quant needs ~223 GB. The
practical local floor reported across sources is a **256 GB** unified-memory Mac
or a 24 GB-GPU + 256 GB-RAM workstation. 144 GB is short by ~100 GB against even
the most aggressive quant — this is not a "tune the offload ratio" gap, it simply
does not load. **Acceptance criterion "measured coding throughput on this
machine" is therefore N/A — the model cannot be instantiated here.**

## Decision: NO-GO

- **Supersede the active rotation?** No — it cannot run, so it cannot replace
  `gemma4-26b-a4b-it` (`agentic_heavy`) or anything else. `gemma4-26b-a4b-it`
  (25.2B/3.8B-active, IQ4_XS ~13 GB, full GPU) remains the local deep-lane pick.
- **Bring up as a demoted candidate?** No — unlike `glm-4.5-air`, there is no
  quant that fits, so there's nothing to keep a launcher for.
- **Coding use specifically?** For coding the hub already routes to the cloud
  paths (`claude-*`, `gemini-*`), which remain the right tool; a local GLM-5.2
  would be both unusable on this hardware and, even if it ran, slower than those
  for interactive coding. The value GLM-5.2 *would* add — a private, offline,
  MIT-licensed frontier coder — only materializes at a size this box can hold.

## When to revisit

Reopen if **either** flips:

1. **A GLM-5.2-Air / Flash (~80–120 B MoE) ships.** That class fit before
   (`glm-4.5-air`, ~106B-A12B, ran via `-ot ".ffn_.*_exps.=CPU"` with the experts
   in the 128 GB RAM). A GLM-5.2-Air at a 2-bit-ish quant under ~120 GB would be a
   genuine `agentic_heavy` contender worth benchmarking against
   `gemma4-26b-a4b-it` on coding + tok/s. **This is the likely trigger** — watch
   the `zai-org/GLM-5.2` HF collection.
2. **The box gains memory** to clear ~245 GB+ combined (e.g. 256 GB RAM), at
   which point the full 744B at UD-IQ2_M becomes loadable — though expert-offload
   throughput on DDR5 would still need a real tok/s measurement before it could
   be called usable for interactive coding.

Until then GLM-5.2 stays out of the registry (see the "Intentional exclusions"
entry in [model-comparison.md](model-comparison.md)).
