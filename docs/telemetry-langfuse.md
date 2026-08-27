# Telemetry & observability — OpenTelemetry + Langfuse

The hub emits OpenTelemetry traces, metrics, and log-record correlation
via OTLP/gRPC into a local Langfuse stack. Trace data is portable
(Langfuse is the durable store; the OTel side is vendor-neutral so the
hub can repoint at Phoenix / Grafana / Honeycomb later with one env
var change). Everything below is local-only and personal-scale by
design — no auth on the OTel pipe, default-on prompt capture, single-host
Langfuse.

## Quick start

```bat
:: 1. Start the stack (Docker Desktop must be running; pulls ~3 GB the first time)
start_langfuse.bat

:: 2. Open http://localhost:3000, create a user + project, copy the
::    public + secret keys.

:: 3. Drop the keys into .env at repo root:
::      LANGFUSE_PUBLIC_KEY=pk-lf-...
::      LANGFUSE_SECRET_KEY=sk-lf-...
::    The OTLP exporter uses these to authenticate against Langfuse's
::    /api/public/otel/v1/traces receiver; without them every span is
::    rejected and the Telemetry tab's deep-links land on
::    "Trace not found". The hub logs `auth=MISSING` at startup if either
::    is absent — easy to spot.

:: 4. Start the hub (or restart if it was already running so OTel picks up env)
run_hub.bat

:: 5. Open the admin SPA -> 📊 Telem tab
::    http://127.0.0.1:8000/admin/
```

The Telemetry tab shows: stack health, per-model leaderboard from the
hub's in-memory ring (so it works even when Langfuse is offline), and a
live trace feed with deep-links into the Langfuse UI.

## Accessing Langfuse from mobile / LAN / Tailscale / Cloudflare

The Telemetry tab's per-row 🔗 Langfuse button (and the header 🔗) build
the URL the **browser** will hit using this rule, in order:

1. If `LANGFUSE_PUBLIC_URL` is set in `.env`, use it verbatim. Right for
   anything that isn't `<same-host>:3000` — Tailscale Serve, Cloudflare
   Tunnel, a custom domain in front of a reverse proxy.
2. Otherwise reuse the hostname the SPA itself was loaded from and swap
   the port for `langfuse_port` (default `3000`). Works automatically
   across **localhost / LAN** because the hub and Langfuse share the
   machine — the hostname your browser used to reach `:8000` also
   reaches `:3000`.

For **Tailscale on mobile** the bare `<host>:3000` path *also* works in
principle, but Windows Firewall blocks inbound TCP 3000 by default
(only :8000 has a pre-existing exception from the hub's first-launch
prompt). Two ways to get past it:

### A. Open Windows Firewall for TCP 3000 (one-time, needs admin)

```powershell
New-NetFirewallRule `
  -DisplayName "Langfuse local (docker) — TCP 3000" `
  -Direction Inbound -Protocol TCP -LocalPort 3000 `
  -Action Allow -Profile Private,Domain
```

Same posture as the hub's :8000 rule. After that, your phone hits
`http://tower:3000/...` over Tailscale and Langfuse responds. Leave
`LANGFUSE_PUBLIC_URL` empty — the SPA's same-host port-swap is enough.

### B. Tailscale Serve in front of Langfuse (recommended)

No admin needed, runs over Tailscale's identity-aware proxy, and you
get a real TLS cert for free. Run **once** on the host machine:

```powershell
tailscale serve --bg --https=3000 http://localhost:3000
```

This exposes Langfuse at `https://<your-tailnet-host>.ts.net:3000/` to
every device on your tailnet. Verify with:

```powershell
tailscale serve status
# https://<your-host>.ts.net:3000 (tailnet only)
# |-- / proxy http://localhost:3000
```

Then set the public URL in `.env` so the Telemetry tab's deep-links
land at the Tailscale URL instead of the local one:

```
LANGFUSE_PUBLIC_URL=https://<your-host>.ts.net:3000
```

…and restart the hub. Tear the serve back down with
`tailscale serve --https=3000 off`.

### Cloudflare Tunnel

Same `LANGFUSE_PUBLIC_URL` slot — point it at the tunneled hostname.
Cloudflare needs its own ingress rule for Langfuse (the hub's tunnel
already covers :8000). One-line snippet for `cloudflared.yml`:

```yaml
ingress:
  - hostname: langfuse.example.com
    service: http://localhost:3000
  - hostname: hub.example.com
    service: http://localhost:8000
  - service: http_status:404
```

Then `LANGFUSE_PUBLIC_URL=https://langfuse.example.com`.

### Login is per-browser

Langfuse's sign-in session is stored as a cookie scoped to whichever
hostname you logged in on. The first time you open the deep-link on a
new device (or via a new hostname — `tower.ts.net` vs `localhost`),
Langfuse will show its sign-in screen rather than the trace. Sign in
once; the cookie persists.

### Lost your Langfuse UI password

The UI login password is **not stored in plaintext anywhere** — not in
`.env`, not in the compose file. It's a bcrypt hash in the `users` table
of the `langfuse-postgres-1` container, so it can't be recovered, only
reset. The `postgres:16` image ships `pgcrypto`, whose `crypt()` /
`gen_salt('bf', 12)` produce the exact `$2a$12$…` bcrypt format Langfuse
verifies against:

```bat
docker exec langfuse-postgres-1 psql -U postgres -d postgres -c "CREATE EXTENSION IF NOT EXISTS pgcrypto; UPDATE users SET password = crypt('NEW_PASSWORD', gen_salt('bf', 12)) WHERE email='you@example.com';"
```

Success prints `UPDATE 1`; then log in at `http://localhost:3000` with
the new password. Avoid a single quote `'` in the password — it breaks
the SQL quoting. This touches **only** the UI login: the
`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` API keys in `.env` are
unaffected, so the hub keeps emitting traces throughout.

## What's captured

Every routed request creates an OTel span tree:

```
Root span: POST /v1/messages                                1620 ms
  gen_ai.system = llama_cpp
  gen_ai.request.model = qwen3.5-4b
  gen_ai.request.max_tokens = 512
  gen_ai.usage.input_tokens = 248
  gen_ai.usage.output_tokens = 312
  client.id = voice-transcriber               (from X-Client-Id)
  gen_ai.prompt = "..."                        (full text by default)
  gen_ai.completion = "..."
  +-- Child: HTTP POST 127.0.0.1:8088/v1/chat/completions  1580 ms
        gen_ai.response.time_to_first_token_ms = 142   (streaming only)
        gen_ai.response.tokens_per_second = 217.4      (streaming only)
        events: [first_token @ +142 ms, last_token @ +1580 ms]
  +-- Child: claude_cli.invoke                            1450 ms   (Claude path)
        claude_cli.exit_code = 0
        claude_cli.argv_hash = 3a5b8c
  +-- Child: gemini_cli.invoke                            2100 ms   (Gemini path)
        gemini_cli.model_switched = true
        events: [model_switch]
```

Metrics (exported every 15 s):

| Name | Type | Labels |
|---|---|---|
| `gen_ai.client.operation.duration` | histogram (ms) | `gen_ai.request.model`, `gen_ai.system`, `error.type` |
| `gen_ai.client.token.usage` | counter | same + `gen_ai.token.type` (input/output) |
| `hub.requests.total` | counter | `route`, `client` |
| `hub.upstream.errors.total` | counter | `gen_ai.system`, `error.type` |

## Trace ID contract

Two ways for a client to correlate the call later (for feedback,
debugging, or cross-system tracing):

- **W3C `traceparent`** — pass it on the request and the hub uses it
  natively. Standard OpenTelemetry propagation; works out of the box.
- **`X-Trace-Id`** — any UUID4, hyphenated or not, or a 32-char hex
  string. The hub maps your value deterministically (BLAKE2b → 16
  bytes) onto an OTel trace ID, so two requests with the same
  `X-Trace-Id` land in the same Langfuse trace.

Every response carries `X-Trace-Id: <32 hex>` set to the actual OTel
trace ID. Read it after the call to attach feedback later — see
[`clients-telemetry-contract.md`](clients-telemetry-contract.md) for a
copy-paste Python example.

## Prompt / completion capture (PII)

By default the hub stores **raw prompt + completion text** as span
attributes — this is a personal-localhost hub, debug value is high.

Flip to BLAKE2b hashes any time you bind beyond loopback or
share-screen something you might not want fully captured:

```bat
set OTEL_HASH_PROMPTS=true
```

(Or set it in `.env`.) Restart the hub. The Telemetry tab's "PII" chip
flips to `hashed` so you can tell at a glance.

## Disabling telemetry entirely

```bat
set OTEL_SDK_DISABLED=true
```

Hub keeps serving traffic; spans + metrics become no-ops; the
Telemetry tab shows "OTel disabled" but the in-memory leaderboard
still works (it doesn't depend on OTel).

## Feedback / scores

The Telemetry tab's 👍 / 👎 buttons POST to
`/admin/api/trace/{trace_id}/feedback`. The hub forwards to Langfuse's
`score()` API in a background task (returns 202 in <50 ms).

Same endpoint works for clients — see the client contract doc.

## Claude Code OTel metrics receiver (issue #68)

Everything above is the hub acting as an OTel **exporter** (hub → Langfuse).
This section is the opposite direction: the hub also runs a small OTLP-
metrics **receiver** so Claude Code's own telemetry export — which is the
only channel that sees sub-agent (Task tool) API calls, since Claude Code
never writes those to its JSONL session transcripts (#66) — can feed the
admin SPA's OTel tab.

### Enable on the host

Put the variables in `~/.claude/settings.json`'s `env` block — Claude Code
applies that to **every** session it starts, regardless of how the process
was launched:

```json
{
  "env": {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "OTEL_METRICS_EXPORTER": "otlp",
    "OTEL_EXPORTER_OTLP_METRICS_PROTOCOL": "http/protobuf",
    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT": "http://127.0.0.1:8000/v1/metrics"
  }
}
```

> **Why not `setx` user-level env vars?** That was the original #68 setup and
> it silently stopped capturing on 2026-07-11: sessions bridged through
> claude.ai/code (web/desktop remote-control) run under a process tree that
> does **not** inherit the user-level `OTEL_*` vars, so telemetry no-ops for
> exactly the sessions that also write no local JSONL usage — a double blind
> spot. The `settings.json` `env` block closes it because Claude Code itself
> injects those values at session start. (`setx` still works for plain
> terminal sessions; keeping both set is harmless.) Applies to *new*
> sessions — one already running won't pick the change up until restarted.

Deliberately scoped to the metrics signal only — not
`OTEL_LOGS_EXPORTER`/`OTEL_TRACES_EXPORTER`, which pull in prompt/tool-
content-adjacent event capture and need
`CLAUDE_CODE_ENHANCED_TELEMETRY_BETA=1`. These vars are independent of the
hub's own exporter above (`src/observability.py` hardcodes its Langfuse
endpoint from `LANGFUSE_HOST` rather than reading the standard
`OTEL_EXPORTER_OTLP_*` vars), so there's no interaction between the two
pipelines.

### What's captured — and what isn't

- `POST /v1/metrics` (`src/server_otel_receiver.py`) accepts an OTLP/HTTP
  protobuf `ExportMetricsServiceRequest`, unauthenticated — same posture as
  the hub's other `/v1/*` routes (personal-localhost hub; if you ever bind
  beyond loopback, this endpoint inherits the same no-auth posture as the
  rest of that surface).
- Only `claude_code.token.usage` and `claude_code.cost.usage` are kept
  (`src/claude_code_otel.py`). Both are `Sum` metrics exported with
  **delta** temporality (verified empirically against a real Claude Code
  export) — each received data point already is the incremental delta since
  the last export, so ingestion just adds values directly.
- **Data-minimization:** the raw export also carries identity attributes
  (`user.id`, `user.email`, `user.account_uuid`, `user.account_id`,
  `organization.id`, `session.id`, `terminal.type`). None of these are
  stored or logged — only `model`, `query_source` (`main`/`subagent`/
  `auxiliary`), (for the token metric) `type`, and (when set — see below)
  `project.name` are persisted, to
  `data/telemetry/claude_code_otel_usage.jsonl` (gitignored).
- **Best-effort delivery.** If the hub is down/restarting when Claude Code
  flushes its telemetry batch, that batch is lost — same as any OTLP push
  pipeline. Claude Code doesn't retry a failed export.
- **Per-model, not per-individual-subagent.** Which model ran is visible;
  which *named* sub-agent it was is not — that finer attribution is tracked
  upstream in [anthropics/claude-code#22625](https://github.com/anthropics/claude-code/issues/22625).
- The OTel tab's "Claude Code (host CLI)" panel is intentionally **not**
  summed into the Code tab's JSONL-sourced headline totals — the two sources
  are labelled and shown separately to avoid double-counting main-agent
  activity both would otherwise report.
- Rows are broken out **per day** as well as per model/source (#233), since
  each ingested point carries its own timestamp.

### Attributing a session to a project (issue #234, automated by fleet-config#310)

Unlike the Code tab — where "project" is just the JSONL session file's own
directory — Claude Code's OTel metrics carry no project/cwd attribute by
default. Verified empirically that setting `OTEL_RESOURCE_ATTRIBUTES` before
launching `claude` **does** flatten a custom attribute onto every data
point's own attributes (not just the resource level), so it round-trips
through this receiver correctly.

**This is automatic, not manual.** `fleet-config` (the repo that owns
`~/.claude`, hooks, skills, and the global `CLAUDE.md` on this machine) wires
a `claude` PowerShell function into `$PROFILE` that derives `project.name`
from the current git repo and sets `OTEL_RESOURCE_ATTRIBUTES` before every
invocation — no per-repo setup needed. See
[`fleet-config`'s `docs/otel-project-attribution.md`](https://github.com/ferraroroberto/fleet-config/blob/main/docs/otel-project-attribution.md)
for the actual mechanism, why it lives there rather than here (it's host/shell
infrastructure with exactly one consumer — this repo's receiver — not a
generic requirement), and how to verify/troubleshoot it.

The hook only fires in an interactive PowerShell session where `$PROFILE`
has loaded (a fresh shell after install). For a one-off invocation outside
that scope — a script, a scheduled task, a shell that doesn't load the
profile — set it by hand instead:

```bat
set OTEL_RESOURCE_ATTRIBUTES=project.name=my-repo-name
claude
```

Sessions that hit neither path show "—" in the panel's Project column.

## Architecture

```
   voice-transcriber       openClaw          curl / SDK
        |                     |                    |
        |  HTTP + traceparent + X-Trace-Id         |
        v                     v                    v
 +------------------------------------------------------+
 |  FastAPI hub (src/server.py)                         |
 |  - TraceIdHeaderMiddleware    (X-Trace-Id <-> OTel)  |
 |  - FastAPIInstrumentor        (root span per req)    |
 |  - GenAI attrs on handlers    (gen_ai.* semconv)     |
 |  +-> src/claude_cli.py        claude_cli.invoke span |
 |  +-> src/gemini_cli.py        gemini_cli.invoke span |
 |  +-> src/openai_upstream.py   httpx auto-span + TTFT |
 |  +-> whisper proxy            whisper.proxy span     |
 +-------------------+----------------------------------+
                     | OTLP/gRPC (localhost:4317)
                     v
 +------------------------------------------------------+
 |  Langfuse (docker compose, localhost:3000)           |
 |    Postgres + Clickhouse + Redis + MinIO             |
 +------------------------------------------------------+
                     ^
                     | langfuse.score() (background task)
                     |
 +------------------------------------------------------+
 |  POST /admin/api/trace/{id}/feedback                 |
 +------------------------------------------------------+
```

## Files

| Path | Role |
|---|---|
| `docker/langfuse/docker-compose.yml` | Local Langfuse v3 stack |
| `start_langfuse.bat` / `.sh` | Idempotent stack-start shortcut |
| `.env.example` | Schema for OTel + Langfuse env vars |
| `src/observability.py` | OTel bootstrap, GenAI helpers, metric instruments |
| `src/trace_id_middleware.py` | X-Trace-Id contract (in / out) |
| `app_web/routers/telemetry.py` | Health, trace feed, metrics, feedback endpoint |
| `app_web/static/telemetry.{js,css}` | SPA Telemetry tab |
| `docs/clients-telemetry-contract.md` | Client-side recipe |
| `src/server_otel_receiver.py` | `POST /v1/metrics` — Claude Code's OTLP metrics receiver (issue #68) |
| `src/claude_code_otel.py` | Parse/persist/rollup for the Claude Code OTel receiver |
| `data/telemetry/claude_code_otel_usage.jsonl` | Persisted usage log (gitignored) backing the OTel tab's Claude Code panel |

## Limitations / known gaps

- **No streaming on `/v1/messages`** — the hub still returns a single
  JSON for Anthropic-shape stream requests, so TTFT/TPS only land for
  OpenAI-shape streams against local llama-server.
- **Claude `usage` is best-effort** — `claude -p` returns it sometimes;
  when absent we leave `gen_ai.usage.*` unset rather than fake zeros.
- **Gemini `usage` is always zero** — the `agy` CLI does not surface
  token counts at all.
- **Single-host Langfuse** — distinguished by `service.instance.id`
  (hostname-PID). Multiple hubs pointing at one Langfuse works fine.
