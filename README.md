# claude-local-calls

A tiny local HTTP server that mimics the Anthropic Messages API (`POST /v1/messages`)
but runs each request through the **`claude -p`** CLI on your machine, using
your local Claude Code auth (your subscription) instead of an API key.

Point any client — including the official `anthropic` SDK — at
`http://127.0.0.1:8000` and your existing code keeps working, charged to
your Claude account rather than API credits.

Inspired by the `_call_claude` pattern in `E:\automation\inspiration-system\src\enrichment.py`.

## Scope & usage policy

This is a **personal playground** for running your own experiments
against your own Claude Code subscription on your own machine. It is
**not** a hosted service, a multi-tenant proxy, or a way to share
subscription access.

To stay clearly within Anthropic's terms, please use it only as
intended:

- ✅ **Do** use it locally to call Claude from your own scripts,
  agents, and tools on devices you personally own.
- ✅ **Do** use it on a trusted LAN to reach your own second machine
  or VM (e.g. a local agent runtime).
- ❌ **Don't** share the endpoint with other people — that would be
  sharing subscription access, which Anthropic's
  [Consumer Terms](https://www.anthropic.com/legal/consumer-terms)
  don't allow.
- ❌ **Don't** port-forward it to the public internet or host it
  behind a domain.
- ❌ **Don't** build a product, commercial service, or large automated
  pipeline on top of it — for anything beyond personal experimentation
  use the paid API, which the
  [Usage Policy](https://www.anthropic.com/legal/aup) and Commercial
  Terms are designed for.
- ❌ **Don't** hammer `claude -p` in tight loops; keep volume at
  human-in-the-loop speeds so you don't abuse the service or get
  rate-limited.

If your use case goes beyond "me, tinkering on my own machine,"
switch to the Anthropic API. When in doubt, check
[anthropic.com/legal](https://www.anthropic.com/legal/) or email
`support@anthropic.com`. This repo is provided as-is, with no
guarantee that it complies with Anthropic's terms for any particular
use.

## Layout

```
claude-local-calls/
├── .venv/                 # local virtualenv
├── requirements.txt
├── run_server.bat         # double-click to start the server
├── src/
│   ├── claude_cli.py      # subprocess wrapper around `claude -p`
│   └── server.py          # FastAPI /v1/messages endpoint
├── tests/
│   └── test_server.py     # unit tests (monkeypatched — no real claude)
└── scripts/
    └── smoke_test.py      # end-to-end test via raw HTTP + anthropic SDK
```

## Setup

Already done — venv exists at `.venv`, deps installed from `requirements.txt`.
Requires the `claude` CLI on `PATH` (Claude Code).

## Run

```bat
run_server.bat
```

or:

```bat
.venv\Scripts\python -m src.server
```

Server listens on `http://127.0.0.1:8000` locally and binds on `0.0.0.0`,
so other machines on your LAN can also reach it.

## LAN access

The server binds on `0.0.0.0`, so any machine on the same network
(another laptop, a VM, an agent like openclaw running next to you) can
call your subscription-backed API directly.

1. **Start the server** (either `run_server.bat` or the Streamlit
   *Server* tab).
2. **Find your LAN IP.** The Streamlit *Server* page shows it as a
   clickable **LAN** link. From a terminal:

   ```bat
   ipconfig | findstr IPv4
   ```

3. **First run on Windows:** the firewall will prompt to allow Python
   through. Accept on **Private** networks only — never Public.
4. **Point the remote client at the LAN URL** instead of loopback:

   ```python
   from anthropic import Anthropic

   client = Anthropic(
       api_key="local-dummy",
       base_url="http://192.168.1.42:8000",   # your LAN IP here
   )
   ```

**Security caveats.** There is no authentication — anyone who can reach
the port can spend your Claude quota. Only run this on trusted networks
(home LAN, office LAN you own). Do **not** port-forward it to the public
internet, and do not accept the firewall prompt on Public networks
(cafés, airports, hotel Wi-Fi).

## Use it from Python

```python
from anthropic import Anthropic

client = Anthropic(api_key="local-dummy", base_url="http://127.0.0.1:8000")
msg = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=128,
    messages=[{"role": "user", "content": "Hello"}],
)
print(msg.content[0].text)
```

Or raw HTTP:

```bash
curl -s http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-haiku-4-5","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}'
```

## Test

Unit tests (fast, no real `claude` calls):

```bat
.venv\Scripts\python -m pytest -q
```

End-to-end smoke test (requires the server running):

```bat
.venv\Scripts\python scripts\smoke_test.py
```

## Limitations (intentional — lightweight)

- No streaming. `stream: true` is accepted but returns a single response.
- Multi-turn chats are flattened into a single prompt for `claude -p`.
- Tool use, images, and extended thinking are not implemented.
- Token counts reflect what `claude -p` reports in its JSON envelope.

## Backlog for improvement

Ordered roughly by payoff for API parity / developer experience.

**High value — closes real compatibility gaps**

- **Streaming (SSE).**
  *What it is:* instead of waiting for the whole reply and returning it
  in one JSON blob, the server sends the answer token-by-token over a
  long-lived HTTP connection using Server-Sent Events. The client sees
  text appear as it's generated — the "ChatGPT typing" effect. The
  Anthropic API does this with a sequence of events
  (`message_start`, `content_block_delta`, `message_delta`,
  `message_stop`); `client.messages.stream(...)` in the SDK expects it.
  *What we'd do:* map `claude -p --output-format stream-json` onto that
  event stream so streaming clients work unchanged.
- **Anthropic-shaped error responses.**
  *What it is:* the Anthropic API returns errors in a specific JSON
  shape — `{"type":"error","error":{"type":"invalid_request_error","message":"..."}}`
  — with specific HTTP status codes (`400`, `401`, `429`, `529`, ...).
  The official SDK inspects this shape to decide whether to retry,
  raise a typed exception, or surface a user-facing message. Today we
  return FastAPI's default `{"detail": "..."}`, which the SDK treats as
  an opaque failure. Matching the real shape makes the server a true
  drop-in.
- **Auth + version headers.**
  *What they are:* the Anthropic API expects three headers on every
  request — `x-api-key` (your credential), `anthropic-version` (which
  API revision you're targeting), and optionally `anthropic-beta` (to
  opt into preview features). The SDK sends them automatically. We
  currently ignore all three, which works but means client code that
  checks "did my version header round-trip?" will be surprised.
  Accepting and echoing them (and optionally validating) avoids those
  edge-case surprises.
- **`GET /v1/models`.** Return the list the CLI knows about so SDKs that
  call `client.models.list()` work.
- **`POST /v1/messages/count_tokens`.** Useful for cost estimation;
  could shell out to a dry-run or use a tokenizer locally.
- **Request IDs.** Add `request-id` / `x-request-id` headers and thread
  them into logs for traceability.
- **CORS.** Enable it so browser-based clients and local webapps can call
  the server directly.
- **Image & document content blocks.** Decode base64 attachments to a
  per-request temp dir, pass via `--add-dir`, reference by path. Caveats:
  indirect semantics (model chooses to read), fuzzy token accounting,
  filesystem side-effects — see notes in the main thread for full pros/cons.
- **Tool use round-trips.** Accept `tools` param and emit
  `tool_use`/`tool_result` content blocks. Non-trivial: the CLI's tool
  system is internal, not the API's function-calling protocol.

**Medium value — fidelity and ergonomics**

- **Faithful multi-turn via `--input-format stream-json`.** Preserves
  prior assistant turns as real assistant messages rather than flattening
  them into a prompt. Better cache reuse, better behavior on long chats.
- **`stop_sequences`, `temperature`, `top_p`, `top_k`.** Accept them —
  some the CLI supports, others must be documented as no-ops.
- **Stop-reason mapping.** Normalize the CLI's `stop_reason` onto the
  API's enum (`end_turn`, `max_tokens`, `stop_sequence`, `tool_use`,
  `pause_turn`).
- **Persistent sessions via `--resume`.** Optional `session_id` in
  request metadata → reuse a CLI session for stateful chat with proper
  prompt-cache hits.
- **Metadata passthrough.** Log `metadata.user_id` and tie it to request
  IDs for per-user observability.
- **Concurrency / process pooling.** Each request spawns a subprocess
  (~1–2s overhead). A small warm-pool or `--resume` reuse cuts p50
  latency significantly.
- **Extended thinking blocks.** `thinking: {type:"enabled", budget_tokens}`
  and mirrored `thinking` content blocks in the response.

**Low value — nice-to-have**

- **`/v1/messages/batches`** (batch API).
- **Prompt-cache-control honoring** on system/message blocks (currently
  parsed but unused).
- **Rate-limit response headers** (`anthropic-ratelimit-*`) — useful for
  clients that read them even if we don't actually rate-limit.
- **Structured logging + `/metrics` endpoint** for Prometheus-style
  observability.
- **Web UI chat playground** at `/playground` for smoke-testing in the
  browser without writing code.

## License

[MIT](LICENSE). Use it, fork it, break it — just keep the copyright
notice. Note that the license covers *this code* only; your use of the
underlying `claude` CLI is still governed by Anthropic's terms (see
[Scope & usage policy](#scope--usage-policy) above).
