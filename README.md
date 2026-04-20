# claude-local-calls

A tiny local HTTP hub that routes `POST /v1/messages` (Anthropic shape) and
`POST /v1/chat/completions` (OpenAI shape) to three backends by `model` name:

- **claude-*** — forwarded to the **`claude -p`** CLI on your machine, using
  your local Claude Code auth (your subscription) instead of an API key.
- **qwen3.5-9b** — forwarded to a local `llama-server` running
  [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)
  on `127.0.0.1:8081`.
- **glm-4.5-air** — forwarded to a local `llama-server` running
  [unsloth/GLM-4.5-Air-GGUF](https://huggingface.co/unsloth/GLM-4.5-Air-GGUF)
  on `127.0.0.1:8082` (MoE CPU offload — attention on GPU, expert tensors
  on RAM).

Point any client — the official `anthropic` or `openai` SDKs, openclaw,
a curl one-liner — at `http://127.0.0.1:8000` and swap backends by
changing the `model` string. Claude requests bill your subscription;
qwen/glm requests never leave the machine.

Inspired by the `_call_claude` pattern in
`E:\automation\inspiration-system\src\enrichment.py`.

## Scope & usage policy

This is a **personal playground** for running your own experiments
against your own Claude Code subscription and your own local GPU on
devices you personally own. It is **not** a hosted service, a
multi-tenant proxy, or a way to share subscription access.

To stay clearly within Anthropic's terms, please use it only as
intended:

- ✅ **Do** use it locally to call Claude from your own scripts,
  agents, and tools on devices you personally own.
- ✅ **Do** use it on a trusted LAN to reach your own second machine
  or VM (e.g. a local agent runtime).
- ✅ **Do** route non-Claude traffic to the local qwen/glm backends as
  much as you like — those are your own weights on your own silicon.
- ❌ **Don't** share the endpoint with other people — for Claude, that
  would be sharing subscription access, which Anthropic's
  [Consumer Terms](https://www.anthropic.com/legal/consumer-terms)
  don't allow.
- ❌ **Don't** port-forward it to the public internet or host it
  behind a domain.
- ❌ **Don't** build a product, commercial service, or large automated
  pipeline on top of the Claude path — for anything beyond personal
  experimentation use the paid API, which the
  [Usage Policy](https://www.anthropic.com/legal/aup) and Commercial
  Terms are designed for.
- ❌ **Don't** hammer `claude -p` in tight loops; keep volume at
  human-in-the-loop speeds so you don't abuse the service or get
  rate-limited. The local backends are rate-limited only by your GPU.

If your use case goes beyond "me, tinkering on my own machine,"
switch to the Anthropic API for Claude. When in doubt, check
[anthropic.com/legal](https://www.anthropic.com/legal/) or email
`support@anthropic.com`. This repo is provided as-is, with no
guarantee that it complies with Anthropic's terms for any particular
use.

## Architecture at a glance

```
openclaw / anthropic SDK / openai SDK / curl
                    │
                    ▼  http://<lan>:8000
   ┌────────────── FastAPI hub (src/server.py) ───────────────┐
   │  route by `model`:                                       │
   │    claude-*       → call_claude()   (claude -p subprocess) │
   │    qwen3.5-9b     → llama-server 127.0.0.1:8081            │
   │    glm-4.5-air    → llama-server 127.0.0.1:8082            │
   └──────────────────────────────────────────────────────────┘
```

See [docs/project-structure.md](docs/project-structure.md) for the full
mermaid diagrams (components, modules, request lifecycle) and
[docs/20260420-hub-with-qwen-and-glm.md](docs/20260420-hub-with-qwen-and-glm.md)
for the post-mortem of how the hub got built.

## Layout

```
claude-local-calls/
├── .venv/                    # local virtualenv
├── requirements.txt
├── run_hub.bat   / .sh       # start the FastAPI hub on :8000
├── run_qwen.bat  / .sh       # start llama-server for Qwen on :8081
├── run_glm.bat   / .sh       # start llama-server for GLM on :8082
├── run_all.bat   / .sh       # start everything enabled on this host
├── launch_app.bat / .sh      # Streamlit UI
├── config/
│   └── models.yaml           # host + model registry
├── src/
│   ├── server.py             # FastAPI hub (both shapes) + router
│   ├── claude_cli.py         # subprocess wrapper around `claude -p`
│   ├── openai_upstream.py    # httpx client for llama-server + shape translators
│   ├── model_registry.py     # YAML loader
│   ├── host_profile.py       # pick active host row
│   ├── install.py            # first-run checks + --fix
│   ├── run_backend.py        # hub|qwen|glm dispatcher
│   ├── server_process.py     # hub Popen + kill-stray-on-port
│   └── llama_process.py      # per-model llama-server Popen
├── app/                      # Streamlit UI (welcome/install/server/models/…)
├── scripts/
│   ├── smoke_test.py
│   ├── download_models.py    # huggingface_hub → models/
│   └── install_llama_cpp.py  # CUDA-Windows / Metal-macOS release
├── tests/                    # test_server / test_router / test_model_registry / test_install
├── vendor/llama.cpp/         # prebuilt llama-server binary (gitignored)
├── models/                   # downloaded GGUFs (gitignored)
└── docs/
    ├── project-structure.md
    └── 20260420-hub-with-qwen-and-glm.md
```

## Setup

One command does everything — deps, llama.cpp binary, GGUF downloads
for the models enabled for this host:

```bat
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m src.install --fix
```

The installer reads [config/models.yaml](config/models.yaml), figures
out which host row you are (by `CLAUDE_LOCAL_CALLS_HOST` env var, else
hostname match, else `default: true`), and only downloads what that
host's `enabled` list asks for. On the reference Windows PC that's
Qwen (~6.6 GB) + GLM (~55 GB); on the Mac mini it's Qwen only.

Plain check (no changes):

```bat
.venv\Scripts\python -m src.install
```

Or use the **Install** tab in the Streamlit UI — same checks, same
fixes, one button per row.

Requires the `claude` CLI on `PATH` (Claude Code) if any `claude-*`
model is enabled for your host.

## Run

```bat
run_hub.bat        :: FastAPI hub on :8000
run_qwen.bat       :: llama-server for Qwen on :8081
run_glm.bat        :: llama-server for GLM on :8082
run_all.bat        :: start every backend enabled for this host
```

Equivalent Python entrypoints:

```bat
.venv\Scripts\python -m src.run_backend hub
.venv\Scripts\python -m src.run_backend qwen
.venv\Scripts\python -m src.run_backend glm
```

The hub binds on `0.0.0.0:8000`, so other machines on your LAN can
also reach it. The llama-server backends bind on loopback — they're
only reachable through the hub.

## LAN access

The hub binds on `0.0.0.0:8000`, so any machine on the same network
(another laptop, a VM, an agent like openclaw running next to you) can
use it.

1. **Start the hub** (either `run_hub.bat` / `.sh` or the Streamlit
   *Server* tab). Start any local backends you need from
   `run_qwen.*` / `run_glm.*` or the *Models* tab.
2. **Find your LAN IP.** The Streamlit *Server* page shows it as a
   clickable **LAN** link. From a terminal:

   ```bat
   ipconfig | findstr IPv4
   ```

3. **First run on Windows:** the firewall will prompt to allow Python
   through. Accept on **Private** networks only — never Public.
4. **Point the remote client at the LAN URL:**

   ```python
   from anthropic import Anthropic
   client = Anthropic(
       api_key="local-dummy",
       base_url="http://192.168.1.42:8000",   # your LAN IP here
   )
   ```

**Security caveats.** There is no authentication — anyone who can reach
the port can spend your Claude quota and burn your GPU. Only run this
on trusted networks (home LAN, office LAN you own). Do **not**
port-forward it to the public internet, and do not accept the firewall
prompt on Public networks (cafés, airports, hotel Wi-Fi).

## Use it from Python

Anthropic SDK, any backend:

```python
from anthropic import Anthropic

client = Anthropic(api_key="local-dummy", base_url="http://127.0.0.1:8000")

# Claude via subscription
msg = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=128,
    messages=[{"role": "user", "content": "Hello"}],
)

# Local Qwen — no network, no cost
msg = client.messages.create(
    model="qwen3.5-9b",
    max_tokens=128,
    messages=[{"role": "user", "content": "Hello"}],
)

# Local GLM — MoE via CPU offload
msg = client.messages.create(
    model="glm-4.5-air",
    max_tokens=128,
    messages=[{"role": "user", "content": "Hello"}],
)
print(msg.content[0].text)
```

OpenAI SDK (get native tool calls for qwen/glm via `llama-server --jinja`):

```python
from openai import OpenAI
client = OpenAI(api_key="local-dummy", base_url="http://127.0.0.1:8000/v1")
msg = client.chat.completions.create(
    model="qwen3.5-9b",
    messages=[{"role": "user", "content": "Hello"}],
)
print(msg.choices[0].message.content)
```

Raw HTTP:

```bash
curl -s http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-4.5-air","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}'
```

List enabled models:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

## Test

Unit tests (fast, no real `claude` / GPU calls):

```bat
.venv\Scripts\python -m pytest -q
```

End-to-end smoke test (requires hub + whichever backends you care
about running):

```bat
.venv\Scripts\python scripts\smoke_test.py
```

It iterates every enabled model from the registry, skips backends
whose port isn't reachable, and reports per-model pass/fail.

## Limitations (intentional — lightweight)

- No streaming. `stream: true` is accepted but returns a single response.
- Multi-turn chats are flattened into a single prompt for `claude -p`.
  (The local backends handle multi-turn natively through llama-server.)
- Tool-use translation across Anthropic ↔ OpenAI shapes is not
  implemented for qwen/glm. OpenAI-shape callers get native tool calls
  from llama-server's `--jinja` templates; Anthropic-shape callers to
  qwen/glm are text-only for now. Claude tool use passes through
  unchanged.
- Images / documents / extended thinking blocks are dropped at the
  shape boundary.
- Token counts reflect what each backend reports in its response.

## Backlog for improvement

Ordered roughly by payoff for API parity / developer experience.

**High value — closes real compatibility gaps**

- **Streaming (SSE).** Map `claude -p --output-format stream-json` and
  llama-server's native SSE onto the Anthropic event shape
  (`message_start`, `content_block_delta`, `message_delta`,
  `message_stop`) so `client.messages.stream(...)` works unchanged.
- **Tool-use round-trips for qwen/glm on the Anthropic shape.** Accept
  `tools` + emit `tool_use`/`tool_result` content blocks, translating
  against llama-server's OpenAI-function-calling output.
- **Anthropic-shaped error responses.** Match
  `{"type":"error","error":{"type":"invalid_request_error","message":"..."}}`
  with the right status codes, so the SDK's retry / typed-exception
  logic behaves as it would against the real API.
- **Auth + version headers.** Accept and echo `x-api-key`,
  `anthropic-version`, `anthropic-beta` so clients that inspect them
  aren't surprised.
- **`POST /v1/messages/count_tokens`.** Useful for cost estimation;
  could shell out to a dry-run or use a tokenizer locally.
- **Request IDs.** Add `request-id` / `x-request-id` and thread them
  into logs for traceability.
- **CORS.** Enable it so browser-based clients and local webapps can
  call the hub directly.
- **Image & document content blocks.** Decode base64 attachments to a
  per-request temp dir, pass via `--add-dir` to Claude, and to the
  appropriate multimodal llama-server builds for qwen-VL when we add
  them.

**Medium value — fidelity and ergonomics**

- **Faithful multi-turn via `claude -p --input-format stream-json`.**
  Preserves prior assistant turns as real assistant messages rather
  than flattening. Better cache reuse.
- **`stop_sequences`, `temperature`, `top_p`, `top_k` passthrough** to
  every backend. Some the CLI supports, others must be documented as
  no-ops.
- **Stop-reason mapping.** Normalize CLI + llama-server stop reasons
  onto the Anthropic enum.
- **Persistent sessions via `--resume`.** Optional `session_id` in
  request metadata → reuse a CLI session for stateful chat with
  proper prompt-cache hits.
- **Metadata passthrough.** Log `metadata.user_id` and tie to request
  IDs for per-user observability.
- **Concurrency / process pooling for Claude.** Each request spawns a
  subprocess (~1–2 s overhead). A small warm pool or `--resume` reuse
  cuts p50 latency significantly.
- **Extended thinking blocks** (`thinking: {type:"enabled", budget_tokens}`)
  on the Claude path.
- **MLX backend for the Mac mini.** llama.cpp-Metal works; MLX is
  30–50 % faster for dense 9 B. Add as a new `backend: "mlx"` entry in
  the registry with a sibling to `openai_upstream.py`.

**Low value — nice-to-have**

- **`/v1/messages/batches`** (batch API).
- **Prompt-cache-control honoring** on system/message blocks.
- **Rate-limit response headers** (`anthropic-ratelimit-*`).
- **Structured logging + `/metrics`** for Prometheus-style
  observability.
- **Auto-start enabled backends with the hub.** Deliberately not done
  today so a 60 GB RAM model doesn't load when someone only wants
  Claude.

## License

[MIT](LICENSE). Use it, fork it, break it — just keep the copyright
notice. Note that the license covers *this code* only; your use of the
underlying `claude` CLI is still governed by Anthropic's terms (see
[Scope & usage policy](#scope--usage-policy) above) and the model
weights follow their own licenses
([Qwen](https://huggingface.co/Qwen),
[GLM / Zhipu AI](https://huggingface.co/zai-org)).
