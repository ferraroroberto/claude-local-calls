# Project structure

An LLM-oriented map of `local-llm-hub`. Three views: a **component
diagram** showing runtime data flow between clients, the hub, and the
backends (Claude subscription via the `claude -p` CLI + Gemini
subscription via the `agy` Antigravity CLI + local llama-server
processes for Qwen3.5-4B (primary), Gemma 4 26B-A4B, Gemma 4 E4B
(fallback), Qwen3.5-9B and GLM-4.5-Air (demoted, ad-hoc bring-up only) +
whisper.cpp ASR for both transcribe (turbo, GPU) and translate
(medium, CPU) + text-to-speech backends (Piper + Orpheus + Kokoro + Chatterbox) at
`/v1/audio/speech`); a pointer to the repo's **filesystem layout**
(README.md, not restated here); and a **request lifecycle** sequence.
Use this file as context when asking
an LLM to modify the project — it shows which file owns what, and what
talks to what. For per-model specs, quantisation, and docs links see
[model-comparison.md](model-comparison.md).

## Component diagram (runtime)

```mermaid
flowchart LR
    subgraph Clients["External clients"]
        SDK["anthropic SDK<br/>(base_url=127.0.0.1:8000)"]
        OAI["openai SDK<br/>(base_url=127.0.0.1:8000/v1)"]
        CURL["raw HTTP / curl"]
        LAN["LAN clients<br/>(other machines, openclaw)"]
    end

    subgraph UI["Admin SPA (app_web/) — sub-app mounted at /admin"]
        WSRV["app_web/server.py<br/>create_app() sub-app<br/>versioned static + bearer auth"]
        WROUT["app_web/routers/<br/>hub · models · playground · services<br/>telemetry · code_usage · glossary<br/>startup_profile · fleet_placement · fleet_maintenance<br/>roles · hosts · machines · diagnostics<br/>auth · webauthn · version · misc"]
        WSTATIC["app_web/static/<br/>index.html SPA + per-tab JS<br/>tabs: Hub · Models · Play · OTel · Code · Machines"]
    end

    subgraph Hub["FastAPI hub (src/)"]
        SRV["src/server.py<br/>POST /v1/messages<br/>POST /v1/chat/completions<br/>GET /v1/models /health /info<br/>GET / → 307 /admin/<br/>mounts /admin sub-app"]
        CHATT["src/chat_translation.py<br/>schemas · media extraction ·<br/>prompt flatten · per-backend dispatch"]
        IMG["src/server_images.py<br/>POST /v1/images/generations<br/>POST /v1/images/edits"]
        REG["src/model_registry.py<br/>YAML → Model rows"]
        HP["src/host_profile.py<br/>resolve active host"]
        CLI_WRAP["src/claude_cli.py<br/>call_claude()"]
        GEM_WRAP["src/gemini_cli.py<br/>call_gemini() / call_gemini_image()<br/>serialized model switch + ConPTY"]
        OAI_UP["src/openai_upstream.py<br/>call_openai_chat()<br/>+ shape translators"]
        RPROXY["src/remote_proxy.py<br/>resolve owning host's base URL<br/>for non-local model rows"]
    end

    subgraph Procs["Process managers"]
        PSUP["src/process_supervisor.py<br/>shared subprocess start/stop workflow"]
        SP["src/server_process.py<br/>hub Popen + log ring<br/>+ kill-port helper"]
        LP["src/backend_process.py<br/>per-model llama-server + whisper-server<br/>Popen + log ring"]
    end

    REMOTE["Remote owning-host hub<br/>e.g. mac-mini-m4 :8000<br/>(qwen3.5-9b, parakeet ASR)"]

    CLAUDE["claude -p CLI<br/>(Claude Code subscription)"]
    GEMINI["agy Antigravity CLI<br/>(Google AI Pro/Ultra subscription)<br/>ConPTY-hosted · Pro/Flash/Flash-Lite"]
    QWEN4B["llama-server :8088<br/>Qwen3.5-4B GGUF (agentic_light)<br/>all layers on GPU"]
    GEMMA426["llama-server :8087<br/>Gemma 4 26B-A4B IT GGUF (MoE, agentic_heavy)<br/>all layers on GPU (IQ4_XS)"]
    QWEN["llama-server :8081<br/>Qwen3.5-9B GGUF (ad-hoc)<br/>all layers on GPU"]
    GLM["llama-server :8082<br/>GLM-4.5-Air GGUF (ad-hoc)<br/>MoE experts on CPU"]
    GEMMA4E["llama-server :8086<br/>Gemma 4 E4B IT GGUF (fallback)<br/>all layers on GPU"]

    subgraph Dev["Dev / tests / scripts"]
        SMOKE["scripts/smoke_test.py<br/>iterate enabled_models()"]
        DLMODELS["scripts/download_models.py<br/>huggingface_hub"]
        DLLLAMA["scripts/install_llama_cpp.py<br/>CUDA-win / Metal-mac"]
        INSTALL_CLI["python -m src.install [--fix]"]
        TESTS["tests/test_server.py<br/>test_router.py<br/>test_model_registry.py<br/>test_install.py"]
    end

    CFG[("config/models.yaml<br/>hosts + models")]
    YAML_CACHE["models/<br/>GGUF files (gitignored)"]
    LLAMA_BIN["vendor/llama.cpp/<br/>llama-server binary"]

    SDK -->|POST /v1/messages| SRV
    OAI -->|POST /v1/chat/completions| SRV
    CURL -->|both shapes| SRV
    LAN -->|both shapes<br/>(0.0.0.0:8000)| SRV
    SMOKE -->|HTTP + SDK| SRV

    SRV -.->|mounts /admin| WSRV
    SRV --> REG
    REG --> HP
    REG -.reads.-> CFG
    HP -.reads.-> CFG

    SRV --> CHATT
    CHATT -->|backend=claude| CLI_WRAP
    CHATT -->|backend=gemini| GEM_WRAP
    CHATT -->|backend=openai| OAI_UP
    SRV -.->|chat_completions direct dispatch| CLI_WRAP
    SRV -.->|chat_completions direct dispatch| GEM_WRAP
    SRV -.->|chat_completions direct dispatch| OAI_UP
    SRV -.->|mounts images router| IMG
    IMG -->|call_gemini_image()| GEM_WRAP
    CLI_WRAP -->|subprocess.run<br/>--output-format json| CLAUDE
    GEM_WRAP -->|ConPTY (pywinpty)<br/>agy -p print mode| GEMINI
    OAI_UP -->|POST /v1/chat/completions| QWEN4B
    OAI_UP -->|POST /v1/chat/completions| GEMMA426
    OAI_UP -->|POST /v1/chat/completions| QWEN
    OAI_UP -->|POST /v1/chat/completions| GLM
    OAI_UP -->|POST /v1/chat/completions| GEMMA4E

    SRV -.->|non-local model row| RPROXY
    RPROXY -->|forwards request verbatim| REMOTE

    WSRV --> WROUT
    WSRV --> WSTATIC
    WROUT -->|hub tab: start/stop/logs<br/>kill stray PID| SP
    WROUT -->|models tab: start/stop/logs per model| LP
    WROUT -.->|play tab: httpx to /v1/messages| SRV

    SP --> PSUP
    LP --> PSUP
    SP -->|Popen python -m src.server| SRV
    LP -->|Popen llama-server --model ...| QWEN4B
    LP -->|Popen llama-server --model ...| GEMMA426
    LP -->|Popen llama-server --model ...| QWEN
    LP -->|Popen llama-server --model ...| GLM
    LP -->|Popen llama-server --model ...| GEMMA4E
    LP -.reads.-> LLAMA_BIN
    LP -.reads.-> YAML_CACHE

    DLMODELS -.writes.-> YAML_CACHE
    DLLLAMA  -.writes.-> LLAMA_BIN
    INSTALL_CLI -.dispatches.-> DLMODELS
    INSTALL_CLI -.dispatches.-> DLLLAMA

    TESTS -->|TestClient<br/>(monkeypatched)| SRV

    classDef ext fill:#2a2f3a,stroke:#555,color:#eee
    classDef hub fill:#1d2a1d,stroke:#4a7,color:#eee
    classDef ui fill:#2a1d2a,stroke:#a47,color:#eee
    classDef backend fill:#2a281d,stroke:#a94,color:#eee
    class Clients ext
    class CLAUDE,GEMINI,QWEN4B,GEMMA426,QWEN,GLM,GEMMA4E,REMOTE backend
    class Hub hub
    class UI ui
```

## Module diagram (filesystem)

Superseded as a standalone listing — README.md's ["Layout"](../README.md#layout)
section is the repo's one hand-maintained filesystem tree; keeping a second
one here let the two drift out of sync with each other and with the code
(`#452`). See it there rather than restating it in this file.

## Request lifecycle

Three paths depending on backend; same entry point.

### Claude backend (model=claude-*)

```mermaid
sequenceDiagram
    participant C as Client (SDK / curl)
    participant F as FastAPI hub (src/server.py)
    participant R as model_registry.resolve
    participant T as chat_translation._run_claude_backend
    participant W as claude_cli.call_claude
    participant K as claude -p CLI

    C->>F: POST /v1/messages<br/>{model:"claude-haiku-4-5", messages, ...}
    F->>R: resolve("claude-haiku-4-5")
    R-->>F: Model(backend="claude")
    F->>T: _run_claude_backend(model, req)
    T->>T: _flatten_messages()<br/>_system_to_text()
    T->>W: call_claude(prompt, model, system)
    W->>K: subprocess.run<br/>claude -p --output-format json
    K-->>W: JSON envelope {result, usage, stop_reason}
    W-->>T: dict
    T-->>F: envelope
    F->>F: _envelope_to_anthropic()
    F-->>C: 200 JSON {id, content, usage, stop_reason}
```

### Gemini backend (model=gemini_pro / gemini_flash / gemini_lite)

```mermaid
sequenceDiagram
    participant C as Client (SDK / curl)
    participant F as FastAPI hub (src/server.py)
    participant R as model_registry.resolve
    participant T as chat_translation._run_gemini_backend
    participant G as gemini_cli.call_gemini
    participant A as agy CLI (ConPTY)

    C->>F: POST /v1/messages<br/>{model:"gemini_pro", messages, ...}
    F->>R: resolve("gemini_pro")
    R-->>F: Model(backend="gemini")
    F->>T: _run_gemini_backend(model, req)
    T->>T: _extract_media_blocks()<br/>_flatten_messages() · _system_to_text()
    T->>G: call_gemini(prompt, model, system, attachments)
    Note over G,A: all calls serialized behind a lock
    G->>A: interactive /model switch<br/>(only when requested model differs)
    G->>A: agy -p print mode<br/>(ConPTY via pywinpty)
    A-->>G: ANSI-rendered reply<br/>(escape sequences stripped)
    G-->>T: envelope {result, usage=0, stop_reason}
    T-->>F: envelope
    F->>F: _envelope_to_anthropic()
    F-->>C: 200 JSON {id, content, usage, stop_reason}
```

`agy` surfaces no token counts, so the Gemini path reports usage as
zero. The model is global persisted CLI state (no per-call flag), which
is why a short interactive `/model` switch precedes print mode whenever
the requested Gemini row differs from the last-selected one.

### Local backend (model=qwen3.5-4b, gemma4-26b-a4b-it, plus qwen3.5-9b / glm-4.5-air / gemma4-e4b-it ad-hoc)

```mermaid
sequenceDiagram
    participant C as Client (Anthropic SDK)
    participant F as FastAPI hub (src/server.py)
    participant R as model_registry.resolve
    participant T as chat_translation._run_openai_backend
    participant U as openai_upstream.call_openai_chat
    participant L as llama-server :8088/:8087 (active) · :8081/:8082/:8086 (ad-hoc)

    C->>F: POST /v1/messages<br/>{model:"qwen3.5-4b", messages, ...}
    F->>R: resolve("qwen3.5-4b")
    R-->>F: Model(backend="openai", url="http://127.0.0.1:8088/v1")
    F->>T: _run_openai_backend(model, req)
    T->>T: anthropic_to_openai_messages()<br/>(flatten content blocks to strings)
    T->>U: call_openai_chat(url, model, messages, ...)
    U->>L: POST /v1/chat/completions
    L-->>U: {choices[0].message.content or reasoning_content, usage, finish_reason}
    U-->>T: dict
    T->>T: openai_to_anthropic_envelope()
    T-->>F: envelope {result, usage, stop_reason}
    F->>F: _envelope_to_anthropic()
    F-->>C: 200 JSON {id, content, usage, stop_reason}
```

OpenAI-shape callers (`POST /v1/chat/completions`) skip the
Anthropic translation hops on both paths — for Claude the hub wraps
the envelope into OpenAI shape; for the local llama-server backends
(qwen35_4b/qwen/glm/gemma4-e4b/gemma4-26b-a4b) it's near-passthrough.

## Key facts for LLM context

- **Purpose.** Single local HTTP endpoint that speaks both Anthropic
  and OpenAI shapes and routes by model name to several backends:
  Claude subscription (via the `claude -p` CLI), local Qwen 3.5 4B
  (agentic_light), local Gemma 4 26B-A4B IT MoE (agentic_heavy),
  whisper.cpp ASR (turbo transcribe + medium translate), plus Gemma 4 E4B IT
  (fallback) and Qwen3.5-9B / GLM-4.5-Air (ad-hoc candidates). Lets
  clients (openclaw, anthropic/openai SDKs) keep one `base_url` and
  swap models via a string. See
  [model-comparison.md](model-comparison.md) for per-model specs.
- **One config, per-host filtering.**
  [`config/models.yaml`](../config/models.yaml) lists every model and
  every host. Each host has an `enabled` whitelist — the installer,
  the registry, the UI, and the smoke test all respect it, so nothing
  is downloaded, launched, or listed that this host hasn't opted into.
  Host resolution: `LOCAL_LLM_HUB_HOST` env var, else hostname
  match, else `default: true` row.
- **Entry points.**
  - `python -m src.run_backend hub` (or `run_hub.bat` / `.sh` at the
    repo root, or `tray.bat` on Windows) — starts FastAPI on
    `0.0.0.0:8000`.
  - `python -m src.run_backend qwen35_4b` / `gemma4_26b` / `whisper`
    / `whisper_translate` / `chatterbox` (active rotation), plus
    `orpheus` (on-demand TTS) and `qwen` / `glm` / `gemma4_e4b`
    (ad-hoc / fallback) (or `launchers/run_model.bat` / `.sh <id>`,
    the parameterized launcher — #448) — starts the matching
    `llama-server` / `whisper-server` / TTS-shim child with args from
    `models.yaml`. The `whisper_translate` slot uses the
    `whisper-server` engine (eager-load, medium on CPU, ~1.5 GB RAM).
    The `piper` / `chatterbox` / `orpheus` / `kokoro` slots use the `tts-server` engine —
    the in-repo FastAPI shim `src/tts_server.py` (Orpheus runs a
    loopback `llama-server` child for its GGUF + SNAC decode).
    A lazy-load alternative exists — set
    `engine: whisper-server-lazy` + `internal_port` + `idle_seconds`
    to route through `src/whisper_translate_proxy.py`, which
    spawns/unloads the child around an idle window — but the active
    rotation runs eager.
  - `python -m src.install [--fix]` — runs every health check, fixes
    the fixable (CLI-only).
  - **Admin UI** — the `app_web/` SPA is a FastAPI sub-app mounted at
    `/admin` inside the hub process, so it comes up with the hub on
    `:8000`. Browse `http://127.0.0.1:8000/admin/` (`GET /` redirects
    there); no separate launcher.
- **Image generation.** `POST /v1/images/generations` and `/edits`
  (OpenAI Images shape) are handled by
  [`src/server_images.py`](../src/server_images.py), which calls
  `call_gemini_image()` in `src/gemini_cli.py`. The only reachable image
  backend is Google **Imagen**, hosted as an agentic tool inside an `agy`
  Gemini text session (there is no Nano Banana / picker image model) — see
  [docs/image-generation.md](image-generation.md) for the full rationale.
- **Multi-host: the Mac Mini.** A model row can declare an owning `host:`
  in `config/models.yaml`; when the active machine isn't that owner,
  [`src/remote_proxy.py`](../src/remote_proxy.py) resolves the owning
  host's own hub `base_url` and the request is forwarded there verbatim —
  a client never needs to know or care which machine actually runs a
  model. Today `qwen3.5-9b` and Parakeet ASR are owned by `mac-mini-m4`
  and proxied through this hub's `base_url`; see the README's "Multi-host:
  the Mac Mini" section for the full walkthrough.
- **The three backend-invocation entry points** are
  [`src/claude_cli.py`](../src/claude_cli.py) (owns
  `subprocess.run(["claude", "-p", ...])`),
  [`src/gemini_cli.py`](../src/gemini_cli.py) (spawns the `agy`
  Antigravity CLI under a Windows ConPTY via `pywinpty` for the
  `gemini-*` rows), and
  [`src/backend_process.py`](../src/backend_process.py) (owns
  `subprocess.Popen(["llama-server", ...])` /
  `subprocess.Popen(["whisper-server", ...])` for each local model). Many
  other modules also shell out for their own narrower purpose (installers,
  config writes, remote SSH, diagnostics sampling, the tray, per-script
  tooling) — this list is only the three that speak for an inbound LLM
  request.
- **Admin SPA runs inside the hub.** The `app_web/` sub-app is mounted
  at `/admin` in the same process as the public `/v1` surface, so its
  routers call the process managers in-process: `app_web/routers/hub.py`
  drives `src/server_process.py` and `app_web/routers/models.py` drives
  `src/backend_process.py` to start/stop/tail each backend, while the
  Play tab proxies through the hub's own `/v1/messages`. Both process
  modules expose module-level singletons so the long-lived hub keeps
  one handle per child across requests.
- **Tests don't touch Claude or the GPU.**
  [`tests/test_server.py`](../tests/test_server.py) and
  [`tests/test_router.py`](../tests/test_router.py) monkeypatch both
  `call_claude` and `call_openai_chat`. The real end-to-end check
  lives in [`scripts/smoke_test.py`](../scripts/smoke_test.py) and
  needs the hub plus the relevant backends running.
- **Intentional gaps.** Partial streaming — OpenAI-shape
  `/v1/chat/completions` proxies upstream SSE through; Anthropic-shape
  `/v1/messages` still single JSON. No tool-use translation between
  Anthropic ↔ OpenAI shapes (OpenAI-shape callers get tool calls
  natively from `llama-server --jinja`; Anthropic-shape callers to
  qwen/glm are text-only for now). Image and document content blocks
  (PDF plus text/data files) land on the `claude-*` / `gemini-*` paths;
  extended-thinking blocks are still dropped at the shape boundary. See
  [issue #453](https://github.com/ferraroroberto/local-llm-hub/issues/453)
  for the ordered backlog.
