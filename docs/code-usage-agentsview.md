# Code tab — AGY usage via AgentsView (issue #280)

The Code tab meters Claude, Codex, and Copilot with the hub's own native
parsers. For **AGY / Antigravity** — whose local session storage (SQLite
containing undocumented protobuf blobs) issues #72 and #279 declined to
reverse-engineer — the hub sources usage from **AgentsView**
([`kenn-io/agentsview`](https://github.com/kenn-io/agentsview)), a local-first
indexer that reads 40+ coding-agent session formats into its own SQLite index
and serves a REST API. Architecturally it's the same shape as the Langfuse
dependency behind the Telemetry tab: an **optional, isolated, external local
service** the hub polls over HTTP and degrades gracefully without.

**Scope is deliberately curated, not everything AgentsView sees.** A map in
`src/agentsview_usage.py` (`_AGENT_VENDOR_MAP`) decides what surfaces:
AgentsView's `gemini` slug (the hub's agy-routed one-shot calls) and
`antigravity-cli` slug (interactive sessions) merge into **one `agy` vendor**
— the AGY button in the Code tab. Claude/Codex/Copilot stay on the hub's
native parsers (#152 found AgentsView's Claude totals run ~15–35 % low), and
other slugs AgentsView indexes (cursor, cowork, pi, …) are ignored as noise —
add a map entry to surface one.

## Setup

The install is a **dedicated venv at the repo root** — never the hub's own
`.venv` (#280's isolation rule):

```bat
.venv\Scripts\python.exe -m venv .venv-agentsview
.venv-agentsview\Scripts\python.exe -m pip install agentsview
```

(`.venv-agentsview/` is gitignored. `pipx install agentsview` on PATH, or an
explicit `AGENTSVIEW_EXE` path in `.env`, work too — that's the exe
resolution order in `agentsview_service.agentsview_exe()`, env → local venv →
PATH.)

Nothing else to start by hand: **the hub launches and monitors AgentsView
itself.**

- **Startup** — the Models tab's Startup card has an **AgentsView** toggle
  (persisted to `config/startup_profile.json`); when on, the hub runs
  `agentsview serve` detached at boot, with its anonymous telemetry and
  update check disabled in the child env. Not installed → one soft-fail log
  line, the hub starts normally.
- **Services card** (Hub tab) — an AgentsView row shows up / down /
  not installed / disabled, the served version, and Start/Stop buttons
  (`POST /admin/api/services/agentsview/launch` and `.../stop`, issue #284)
  — Start when it's down but installed, Stop when it's reachable.

## Configuration

- `AGENTSVIEW_BASE_URL` (hub `.env`) — where the hub polls; default
  `http://127.0.0.1:8080`. **Empty string disables the integration
  entirely** (no probe, no background refresh, no launch).
- `AGENTSVIEW_EXE` (hub `.env`) — explicit path to the executable, overriding
  the `.venv-agentsview/` → PATH resolution.
- **Port-drift gotcha:** if something else already holds :8080, AgentsView
  auto-picks a different free port. Check `agentsview serve status` and set
  `AGENTSVIEW_BASE_URL` accordingly — the hub also guards against a foreign
  service squatting the port by requiring `/api/ping` to identify as
  `agentsview`.
- First-ever `agentsview serve` runs a full index sync before listening
  (~1–2 min on this host); the hub's launch helper waits up to 3 min.
  Steady-state restarts come up in seconds, then a file watcher + 15-min
  sweep keep the index fresh — the hub never shells out to sync it.

## How it behaves

- **Never on the request path.** `src/agentsview_usage.py` keeps an in-memory
  snapshot and refreshes it in a background thread (60 s TTL, 1 s connect
  timeout); the summary endpoint only ever reads the snapshot. Completed
  sessions are fetched once and cached — they're immutable.
- **Graceful degradation.** AgentsView down → the AGY button keeps showing
  last-fetched data and the tab shows a muted "AgentsView offline" hint (the
  Code-tab mirror of the Telemetry tab's `langfuse_reachable`). Not installed
  → the tab is exactly the native three-vendor view, no hint, no errors.
- **Costs are "as reported by AgentsView"** — its own estimates for
  subscription agents, passed through unchanged (never re-priced against the
  hub's rate tables), same pass-through rule as Copilot's exact credits.
  Records are per-call where AgentsView provides a usage breakdown
  (per-row model + timestamp), degrading to one session-granular record with
  whatever fields exist — the hub never fabricates numbers.

AgentsView's own web UI at the same address remains the richer session
browser (full transcripts, search); the hub consumes its API but doesn't
embed or replace it.
