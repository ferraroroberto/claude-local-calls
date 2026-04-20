# Project structure

An LLM-oriented map of `claude-local-calls`. Three views: a **component
diagram** showing runtime data flow between clients, the hub, and the
three backends (Claude subscription + two local llama-server
processes); a **module diagram** showing the Python package layout and
imports; and a **request lifecycle** sequence. Use this file as context
when asking an LLM to modify the project — it shows which file owns
what, and what talks to what.

## Component diagram (runtime)

```mermaid
flowchart LR
    subgraph Clients["External clients"]
        SDK["anthropic SDK<br/>(base_url=127.0.0.1:8000)"]
        OAI["openai SDK<br/>(base_url=127.0.0.1:8000/v1)"]
        CURL["raw HTTP / curl"]
        LAN["LAN clients<br/>(other machines, openclaw)"]
    end

    subgraph UI["Streamlit UI (app/)"]
        APP["app/app.py<br/>nav + CSS"]
        V_WELCOME["views/welcome.py"]
        V_INSTALL["views/install.py<br/>check + fix rows"]
        V_SERVER["views/server.py<br/>hub control + log"]
        V_MODELS["views/models.py<br/>per-backend start/stop/log"]
        V_TEST["views/testing.py<br/>pytest + smoke"]
        V_PLAY["views/playground.py<br/>prompt → hub"]
    end

    subgraph Hub["FastAPI hub (src/)"]
        SRV["src/server.py<br/>POST /v1/messages<br/>POST /v1/chat/completions<br/>GET /v1/models /health /info /"]
        REG["src/model_registry.py<br/>YAML → Model rows"]
        HP["src/host_profile.py<br/>resolve active host"]
        LAND["src/landing.py<br/>HTML for GET /"]
        CLI_WRAP["src/claude_cli.py<br/>call_claude()"]
        OAI_UP["src/openai_upstream.py<br/>call_openai_chat()<br/>+ shape translators"]
    end

    subgraph Procs["Process managers"]
        SP["src/server_process.py<br/>hub Popen + log ring<br/>+ kill-port helper"]
        LP["src/llama_process.py<br/>per-model llama-server<br/>Popen + log ring"]
    end

    CLAUDE["claude -p CLI<br/>(Claude Code subscription)"]
    QWEN["llama-server :8081<br/>Qwen3.5-9B GGUF<br/>all layers on GPU"]
    GLM["llama-server :8082<br/>GLM-4.5-Air GGUF<br/>MoE experts on CPU"]

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
    V_PLAY -->|httpx POST| SRV
    SMOKE -->|HTTP + SDK| SRV

    SRV --> LAND
    SRV --> REG
    REG --> HP
    REG -.reads.-> CFG
    HP -.reads.-> CFG

    SRV -->|backend=claude| CLI_WRAP
    SRV -->|backend=openai| OAI_UP
    CLI_WRAP -->|subprocess.run<br/>--output-format json| CLAUDE
    OAI_UP -->|POST /v1/chat/completions| QWEN
    OAI_UP -->|POST /v1/chat/completions| GLM

    APP --> V_WELCOME
    APP --> V_INSTALL
    APP --> V_SERVER
    APP --> V_MODELS
    APP --> V_TEST
    APP --> V_PLAY
    V_SERVER -->|start/stop/logs<br/>kill stray PID| SP
    V_MODELS -->|start/stop/logs per model| LP
    V_TEST -->|pytest / smoke| SP
    V_PLAY -->|reachable? BASE_URL| SP
    V_INSTALL -->|run_all_checks / fix_fn_for| INSTALL_CLI

    SP -->|Popen python -m src.server| SRV
    LP -->|Popen llama-server --model ...| QWEN
    LP -->|Popen llama-server --model ...| GLM
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
    class CLAUDE,QWEN,GLM backend
    class Hub hub
    class UI ui
```

## Module diagram (filesystem)

```mermaid
flowchart TB
    ROOT["claude-local-calls/"]
    ROOT --> README["README.md"]
    ROOT --> REQ["requirements.txt"]
    ROOT --> LIC["LICENSE"]
    ROOT --> LAUNCHERS["run_hub / run_qwen / run_glm / run_all<br/>.bat (Windows) + .sh (macOS)<br/>launch_app.bat / .sh"]

    ROOT --> CFGDIR["config/"]
    CFGDIR --> C1["models.yaml<br/>hosts + models registry"]

    ROOT --> SRC["src/"]
    SRC --> S1["server.py<br/>FastAPI hub + router"]
    SRC --> S2["claude_cli.py<br/>claude -p wrapper"]
    SRC --> S3["openai_upstream.py<br/>llama-server client +<br/>Anthropic ↔ OpenAI shapes"]
    SRC --> S4["model_registry.py<br/>YAML loader + Model class"]
    SRC --> S5["host_profile.py<br/>pick active host row"]
    SRC --> S6["install.py<br/>checks + fix dispatch"]
    SRC --> S7["run_backend.py<br/>hub|qwen|glm dispatcher"]
    SRC --> S8["server_process.py<br/>hub Popen + kill-port"]
    SRC --> S9["llama_process.py<br/>per-model llama-server Popen"]
    SRC --> S10["landing.py<br/>HTML for GET /"]

    ROOT --> APPDIR["app/"]
    APPDIR --> A1["app.py<br/>Streamlit nav"]
    APPDIR --> A2["styles/light.css"]
    APPDIR --> A3["views/"]
    A3 --> V1["welcome.py"]
    A3 --> V2["install.py"]
    A3 --> V3["server.py"]
    A3 --> V4["models.py"]
    A3 --> V5["testing.py"]
    A3 --> V6["playground.py"]

    ROOT --> SCDIR["scripts/"]
    SCDIR --> SC1["smoke_test.py"]
    SCDIR --> SC2["download_models.py<br/>huggingface_hub"]
    SCDIR --> SC3["install_llama_cpp.py<br/>CUDA-win / Metal-mac"]

    ROOT --> TDIR["tests/"]
    TDIR --> T1["test_server.py"]
    TDIR --> T2["test_router.py"]
    TDIR --> T3["test_model_registry.py"]
    TDIR --> T4["test_install.py"]

    ROOT --> VENDOR["vendor/<br/>(gitignored)"]
    VENDOR --> VLL["llama.cpp/<br/>llama-server binary"]

    ROOT --> MDLS["models/<br/>(gitignored)"]
    MDLS --> M1["Qwen3.5-9B-Q4_K_M.gguf"]
    MDLS --> M2["GLM-4.5-Air-Q4_K_M/<br/>multi-part GGUF"]

    ROOT --> DOCS["docs/"]
    DOCS --> D1["project-structure.md<br/>(this file)"]
    DOCS --> D2["20260420-hub-with-qwen-and-glm.md<br/>post-mortem of the hub build"]
```

## Request lifecycle

Two paths depending on backend; same entry point.

### Claude backend (model=claude-*)

```mermaid
sequenceDiagram
    participant C as Client (SDK / curl)
    participant F as FastAPI hub (src/server.py)
    participant R as model_registry.resolve
    participant W as claude_cli.call_claude
    participant K as claude -p CLI

    C->>F: POST /v1/messages<br/>{model:"claude-haiku-4-5", messages, ...}
    F->>R: resolve("claude-haiku-4-5")
    R-->>F: Model(backend="claude")
    F->>F: _flatten_messages()<br/>_system_to_text()
    F->>W: call_claude(prompt, model, system)
    W->>K: subprocess.run<br/>claude -p --output-format json
    K-->>W: JSON envelope {result, usage, stop_reason}
    W-->>F: dict
    F->>F: _envelope_to_anthropic()
    F-->>C: 200 JSON {id, content, usage, stop_reason}
```

### Local backend (model=qwen3.5-9b or glm-4.5-air)

```mermaid
sequenceDiagram
    participant C as Client (Anthropic SDK)
    participant F as FastAPI hub (src/server.py)
    participant R as model_registry.resolve
    participant U as openai_upstream.call_openai_chat
    participant L as llama-server :8081 or :8082

    C->>F: POST /v1/messages<br/>{model:"qwen3.5-9b", messages, ...}
    F->>R: resolve("qwen3.5-9b")
    R-->>F: Model(backend="openai", url="http://127.0.0.1:8081/v1")
    F->>F: anthropic_to_openai_messages()<br/>(flatten content blocks to strings)
    F->>U: call_openai_chat(url, model, messages, ...)
    U->>L: POST /v1/chat/completions
    L-->>U: {choices[0].message.content or reasoning_content, usage, finish_reason}
    U-->>F: dict
    F->>F: openai_to_anthropic_envelope()<br/>+ _envelope_to_anthropic()
    F-->>C: 200 JSON {id, content, usage, stop_reason}
```

OpenAI-shape callers (`POST /v1/chat/completions`) skip the
Anthropic translation hops on both paths — for Claude the hub wraps
the envelope into OpenAI shape; for qwen/glm it's near-passthrough.

## Key facts for LLM context

- **Purpose.** Single local HTTP endpoint that speaks both Anthropic
  and OpenAI shapes and routes by model name to three backends:
  Claude subscription (via the `claude -p` CLI), local Qwen3.5-9B,
  local GLM-4.5-Air. Lets clients (openclaw, anthropic/openai SDKs)
  keep one `base_url` and swap models via a string.
- **One config, per-host filtering.**
  [`config/models.yaml`](../config/models.yaml) lists every model and
  every host. Each host has an `enabled` whitelist — the installer,
  the registry, the UI, and the smoke test all respect it, so nothing
  is downloaded, launched, or listed that this host hasn't opted into.
  Host resolution: `CLAUDE_LOCAL_CALLS_HOST` env var, else hostname
  match, else `default: true` row.
- **Entry points.**
  - `python -m src.run_backend hub` (or `run_hub.bat` / `.sh`) — starts
    FastAPI on `0.0.0.0:8000`.
  - `python -m src.run_backend qwen` / `glm` (or `run_qwen.*` /
    `run_glm.*`) — starts the matching `llama-server` child with args
    from `models.yaml`.
  - `python -m src.install [--fix]` — runs every health check, fixes
    the fixable; shared with the Streamlit Install tab.
  - `streamlit run app/app.py` (or `launch_app.bat` / `.sh`) — UI.
- **Only two places shell out.**
  [`src/claude_cli.py`](../src/claude_cli.py) owns
  `subprocess.run(["claude", "-p", ...])`.
  [`src/llama_process.py`](../src/llama_process.py) owns the
  `subprocess.Popen(["llama-server", ...])` for each local model.
  Everything else is pure Python / FastAPI / httpx / Streamlit.
- **UI ↔ processes isolation.** The Streamlit app never imports the
  FastAPI app or the `llama-server` binary directly; it launches each
  as a child process and polls `/health` or `/v1/models`. That's why
  both process modules use module-level singletons (Streamlit reruns
  the script every interaction).
- **Tests don't touch Claude or the GPU.**
  [`tests/test_server.py`](../tests/test_server.py) and
  [`tests/test_router.py`](../tests/test_router.py) monkeypatch both
  `call_claude` and `call_openai_chat`. The real end-to-end check
  lives in [`scripts/smoke_test.py`](../scripts/smoke_test.py) and
  needs the hub plus the relevant backends running.
- **Intentional gaps.** No streaming. No tool-use translation between
  Anthropic ↔ OpenAI shapes (OpenAI-shape callers get tool calls
  natively from `llama-server --jinja`; Anthropic-shape callers to
  qwen/glm are text-only for now). Extended thinking, images, and
  multi-part content blocks are dropped at the shape boundary. See
  the README backlog for the ordered list.
