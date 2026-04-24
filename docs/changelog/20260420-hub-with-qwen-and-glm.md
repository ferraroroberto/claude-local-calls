# What we did — 2026-04-20

Post-mortem of the work that turned `claude-local-calls` from a single-
backend Claude wrapper into a three-backend local models hub. Pairs with
[project-structure.md](project-structure.md) — that file shows the end-
state; this one records why, what, and what we learned.

---

## Starting point (2026-04-19)

- Repo had one job: a FastAPI server on :8000 that forwarded
  `POST /v1/messages` to `claude -p`. Great for drop-in use of the
  Anthropic SDK against the user's Claude Code subscription.
- Hardware on hand: RTX 5060 Ti 16 GB + 128 GB RAM (Windows PC),
  Mac mini M4 base 16 GB unified (later). CUDA 13.0 toolkit present.
- A separate research pass (previously in `local-llm-research.md`)
  had surveyed 2026-era open-weight models and inference engines. Its
  conclusions are now frozen here rather than kept as a separate doc.

## Research conclusions that drove the build

The research ranked candidate setups by what actually fits each box
and earns its keep for an openclaw-class agent (tool-calling, long-ish
context, 24/7).

1. **Mac mini's 16 GB unified memory is the ceiling** — caps it at
   dense ≤9B. Right role: always-on small helper.
2. **The 5060 Ti is the serious box.** 16 GB VRAM fits dense ≤14B at
   Q4; 128 GB of RAM unlocks MoE models via CPU offload with tolerable
   speed because active-param counts are low.
3. **Engine choice: `llama.cpp`'s `llama-server`** on both machines.
   OpenAI-compatible `/v1/chat/completions` out of the box, `--jinja`
   enables Hermes-style tool calling baked into Qwen3 and GLM chat
   templates, prebuilt binaries for CUDA-Windows and Metal-macOS, and
   critically: the `-ot` regex that sends MoE expert tensors to CPU
   while keeping attention on GPU — the exact trick that makes
   GLM-4.5-Air run on 16 GB VRAM. Accepted trade-off: MLX is 30–50 %
   faster for dense 9B on Apple Silicon; we use llama.cpp anyway for
   one mental model across hosts. Adding MLX later would be a new
   backend entry in the registry.
4. **Model picks:**
   - **GLM-4.5-Air** (Tier 1) — 106 B total / 12 B active MoE, agent
     and coding tuned. Runs on the PC via MoE CPU offload.
   - **Qwen3.5-9B** (Tier 2) — 9 B dense, strong small-model tool
     caller. Fits entirely in 16 GB VRAM at Q4; runs on both machines.
5. **Single entry point** on :8000 routing by `model` name — openclaw
   and other clients keep one `base_url` and swap models via a string.

## What we built

A single FastAPI hub on :8000 that routes by `model` name to three
backends, plus supporting infrastructure for a per-host, cross-platform
install.

### Routing

- `POST /v1/messages` — Anthropic shape. Claude models hit the existing
  `claude -p` path; qwen/glm get their messages flattened to OpenAI
  shape, forwarded to the matching llama-server, and the reply is
  translated back into an Anthropic envelope.
- `POST /v1/chat/completions` — OpenAI shape. Near-passthrough for
  qwen/glm; for Claude the reply is wrapped into OpenAI shape. Gives
  OpenAI-shape callers native tool calling via `llama-server --jinja`
  from day one.
- `GET /v1/models` — union of enabled entries in the registry.
- Unknown `model` → 400 with the list of known names.

### Registry-driven, per-host build

One config, [`config/models.yaml`](../../config/models.yaml), lists every
host (with a `platform` and an `enabled` whitelist) and every model
(display name, aliases, backend, engine, port, GGUF path, llama-server
args). Host resolution: `CLAUDE_LOCAL_CALLS_HOST` env var → hostname
match → `default: true` row.

Everything downstream respects the active host's `enabled` list: the
installer, the model downloader, the llama-server launcher, the UI's
Models tab, the Playground's picker, and the smoke test.

Adding a new machine later is one YAML block plus running
`python -m src.install --fix` — no code changes, no conditional
branches sprinkled through the Python.

### Installer as a first-class module

[`src/install.py`](../../src/install.py) exposes a table of checks
(python+venv, deps, host profile resolve, claude CLI, GPU, llama.cpp
binary, each enabled GGUF, ports free, and an opt-in end-to-end ping)
and a separate set of `fix_*` functions. Same checks feed two
surfaces: `python -m src.install [--fix] [--json]` (CLI) and the
Streamlit Install tab (`app/views/install.py`). On the Install tab,
failing rows expose a Fix button; long fixes (llama.cpp download,
60 GB GLM pull) stream progress.

### Process management

Two singletons, same ring-buffer-log pattern:

- [`src/server_process.py`](../../src/server_process.py) — hub Popen,
  reachability check, and a **kill-stray-on-port helper** added after
  we hit WinError 10048 from a zombie hub bound to :8000. The Server
  tab now surfaces the stray PID and a "💀 Kill stray process"
  button when the port is held by something we don't manage.
- [`src/llama_process.py`](../../src/llama_process.py) — per-model
  llama-server Popen keyed by model id, parallel API to the hub's.
  The Models tab renders one card per enabled model (start, stop,
  refresh, tailed log, launch args).

### Launchers

Thin `.bat` (Windows) and `.sh` (macOS) scripts that each activate the
venv and call `python -m src.run_backend {hub|qwen|glm}`. Real dispatch
lives in one Python module so we don't maintain two parallel launchers.
`run_all.*` starts everything enabled on the current host.

### Tests

15 tests, no GPU or Claude access required:

- `tests/test_server.py` — existing; monkeypatches `call_claude`.
- `tests/test_router.py` — new; mocks `call_openai_chat`, asserts
  Anthropic→OpenAI shape flattening, unknown-model 400, and that
  `/v1/chat/completions` passthrough hits the right upstream URL.
- `tests/test_model_registry.py` — new; hostname match precedence,
  `CLAUDE_LOCAL_CALLS_HOST` override, alias resolution, url synthesis.
- `tests/test_install.py` — new; shape of `run_all_checks()`,
  `Report.worst_status` ordering, `fix_fn_for` dispatch.

## Verification (PC, 2026-04-20)

End-to-end on the Windows PC:

| Step | Result |
|---|---|
| `python -m src.install --fix` | All checks green. llama.cpp CUDA binary extracted, Qwen GGUF (~6.6 GB) and GLM GGUF (~55 GB, multi-part) downloaded. |
| Qwen on :8081 | `ggml_cuda_init: found 1 CUDA devices`, all layers on GPU. ~65 tok/s. ~7 GB VRAM used. |
| GLM on :8082 | CUDA init + `-ot ".ffn_.*_exps.=CPU"` honored. Log shows `CPU_Mapped model buffer size = 47127.51 MiB + 19607.79 MiB` (~65 GB on CPU), `CUDA0 model buffer size = 3990.17 MiB` (~4 GB on GPU), `CUDA0 KV buffer size = 2944.00 MiB` (~3 GB KV cache). Total VRAM ≈ 7–8 GB — safely under 16 GB budget. |
| 3-model smoke test | `passed : 3 — claude-haiku-4-5, qwen3.5-9b, glm-4.5-air`. |
| From openclaw or any other LAN client | `base_url=http://<pc-lan>:8000`, `model="qwen3.5-9b"` or `"glm-4.5-air"` routes to the matching llama-server; `model="claude-*"` hits the Claude subscription. |

## Notable pitfalls we hit

- **Qwen3 returned empty `content` with 64 tokens reported.** With
  `--jinja` on, the reasoning template puts the answer in
  `message.reasoning_content` instead of `message.content`.
  Fix: `openai_response_text` falls back to `reasoning_content` when
  `content` is empty.
- **Claude smoke test 502.** The registry's `display_name: claude`
  isn't a real Claude alias; `claude -p` rejected it. Fix: set
  `claude.display_name: claude-haiku-4-5` in the YAML.
- **Tests passed `importlib.reload` plus monkeypatched module attrs,
  then saw the patches evaporate.** `reload` re-executes the module
  and redefines `CONFIG_PATH`. Fix: removed `reload` calls; rely on
  `_load_config()` reading the attribute at call time.
- **WinError 10048 on hub restart.** Stale hub process was still bound
  to :8000 after a previous terminate. Motivated the
  kill-stray-on-port helper and the "💀 Kill stray process" button in
  the Server tab (only shown when we aren't managing the port
  ourselves; uvicorn's worker PID differs from our parent PID, so the
  warning suppresses itself whenever `is_running()`).

## What's explicitly deferred

- **Tool-call translation across Anthropic ↔ OpenAI shapes** for
  Anthropic-shape callers to qwen/glm. OpenAI-shape callers already
  get native tool calls.
- **Streaming SSE.**
- **MLX on the Mac mini.** The research flagged MLX as 30–50 % faster
  than llama.cpp-Metal for dense 9B. We accepted the hit to keep one
  engine across hosts. If Qwen on Mac turns out painful, adding an MLX
  backend is a localized change (new `backend: "mlx"` in the registry
  + adapter alongside `openai_upstream.py`).
- **Mac mini bring-up.** Everything is cross-platform; the actual
  clone/install on the Mac is a later task.

## Detailed research appendix (what informed the build)

The 2026-era survey that drove the choices above. Kept here as a
record of what was true at decision time, not as an ongoing living
doc.

### Hardware

| Machine | Spec summary | Role |
|---|---|---|
| Mac mini (base M4) | 10-core CPU · 10-core GPU · 16 GB unified · 256 GB SSD · ~120 GB/s · idle ≈ 5 W, burst ≈ 30–65 W | Always-on small-model server. ~US$15–20/yr electricity. |
| Windows PC | ZOTAC RTX 5060 Ti AMP 16 GB GDDR7 (Blackwell, 4608 CUDA, 448 GB/s, 180 W TDP) + 128 GB system RAM | The serious box. 16 GB VRAM fits dense ≤14B at Q4; 128 GB RAM unlocks MoE via CPU offload. |

### Qwen lineup in April 2026

- **Qwen3.5 dense**: 0.8 / 2 / 4 / 9 / 27 B. 9 B ≈ 6.6 GB at Q4 and is
  the "good on a laptop" pick. 27 B is the upper end for 16 GB single-
  GPU or 32 GB Mac — too big for the 16 GB Mac mini.
- **Qwen3-Coder-Next** — 80 B / 3 B active · 256 k context · trained
  for tool use + long-horizon agents · out-of-box with Claude Code /
  Qwen Code / Cline / OpenCode. Unsloth: ≥45 GB memory for 4-bit,
  ≥30 GB for 2-bit-XL.
- **Qwen3-Next-80B-A3B** — 80 B / 3 B active · hybrid attention.
- **Qwen3.5-35B-A3B / 122B-A10B / 397B-A17B** — only 35B-A3B
  (~22 GB Q4) is realistic on 16 GB GPU with CPU offload.
- **Qwen3.6-35B-A3B** (April 2026) — newest; same memory profile.
- **Qwen3-VL** — dense 2/4/8/32 B; MoE 30B-A3B, 235B-A22B. 8B-VL is
  the sweet spot for 16 GB GPU + vision.
- **Qwen3-Omni** — audio + vision + text; less battle-tested.

### Non-Qwen worth knowing

- **GLM-4.5-Air** (Zhipu AI) — 106 B total / 12 B active MoE, agent/
  coding tuned. The pick for the PC.
- **gpt-oss-20B (MXFP4)** — ~488 TPS on 5060 Ti for short-context API
  workloads (hardware-corner benchmarks).
- **Gemma 3 / Phi-4** — small models (≤9 B) that compete with
  Qwen3.5-9B on the Mac.
- **Llama 3.1 8B** — on the 5060 Ti ≈ 71 tok/s; useful baseline.

### Inference engine comparison

| Engine | Best on | Why |
|---|---|---|
| MLX / MLX-LM | Mac mini | Native Metal. ~230 tok/s on optimized 7 B. 30–50 % faster than llama.cpp on Apple Silicon. |
| Ollama (w/ MLX backend on Apple) | Mac mini, PC (CUDA) | Easiest UX, OpenAI-compatible endpoint. Uses MLX under the hood on Apple Silicon. |
| llama.cpp (`llama-server`) | PC, Mac mini | OpenAI-compatible, `--jinja` enables native function calling, `-ot` regex for MoE offload. **What we chose.** |
| LM Studio | Either, GUI users | Good for model browsing; wraps llama.cpp. |
| vLLM | PC only (CUDA) | Production-grade, best throughput for Qwen3 / Qwen3-VL. Heavier setup. |
| Qwen-Agent | Any | Agent framework; function-calling, MCP, code interpreter, RAG. |

### Expected tok/s (per research)

- Qwen3.5-9B dense Q4 on 5060 Ti: 60–80 tok/s. **Measured: ~65.**
- GLM-4.5-Air Q4 with MoE offload on 5060 Ti + 128 GB: 8–15 tok/s.
  **Measured: consistent with the lower end.**
- Qwen3.5-9B on Mac mini M4 via MLX: ~30–45 tok/s expected (not yet
  measured — Mac bring-up deferred).

### Decisions we deliberately did not make

- No second engine (vLLM, Ollama, MLX) in phase 1. One engine across
  hosts keeps the mental model small. Adding one later is a new
  registry entry, not a refactor.
- No Qwen3-Coder-Next / Qwen3-Next MoE in phase 1. Bigger downloads
  for a second agent-tuned MoE we don't yet need; GLM-4.5-Air
  dominates that slot today.
- No auto-start of llama-servers with the hub. A 60 GB RAM model
  shouldn't load when someone only wants Claude. Keep them independent
  Popens controlled from the Models tab or `.bat` / `.sh`.
