# Project Instructions

Canonical instructions for AI coding agents in this repository — Claude Code reads it directly as project memory; other agents (Cursor, Codex, etc.) arrive via the one-line `AGENTS.md` pointer.

## This repository
Local HTTP hub routing Anthropic-shaped and OpenAI-shaped requests to multiple LLM/ASR backends, with a FastAPI + static-JS admin SPA mounted at `/admin`.
See `README.md` for setup, layout, and usage.

## Internal architecture

[`docs/architecture.mmd`](docs/architecture.mmd) — hand-authored Mermaid diagram of this repo's internal structure (entry points, `src/server.py`'s routing to the Claude/Gemini/llama-server backends, the process managers, the `app_web/` admin sub-app, observability, config) — the per-repo counterpart to the fleet map `/system-map` writes into `global-CLAUDE.md`. It is the repo's **only** component/runtime map: [`docs/project-structure.md`](docs/project-structure.md) points here and keeps only the per-backend request-lifecycle sequences plus the LLM key-facts briefing, after its own second copy drifted (#475). Update it in the **same PR** as any material structural change (backend added/removed, router moved, process manager relocated) — same anti-staleness contract as a `.fleet.toml` `description`. Not auto-generated; nothing tests it. Model **placement** (which host owns which row) is [`config/models.yaml`](config/models.yaml)'s alone — never restate it in a doc.

**Safe restart (never blanket-kill python):** the canonical restart is **`tray.bat --restart`** — orphan-proof reclaim-then-start: kills the tray subtree, reclaims the hub port **:8000** by PID scoped to this repo's `.venv` (CommandLine-matched), then starts fresh. It deliberately does **not** touch `:8090` (whisper-server, mutex-shared with `voice-transcriber`) or the llama-server model ports (8081/8082/8086/8087/8088). By hand only as a fallback: find the owner with `Get-NetTCPConnection -LocalPort 8000`, stop that PID, relaunch via `tray.bat`. **Build confirmation:** `GET http://127.0.0.1:8000/health` returns 200 once the hub is back up — *liveness only*, since its payload (`{"status": "ok"}`) is byte-identical before and after a restart and so can never prove the new build is live. For build identity poll `GET http://127.0.0.1:8000/admin/api/version` and check that `git_sha` changed — the auth-exempt endpoint `tray.bat` and `src/config_write.py` already use, and the same split `src/services.py`'s `peer_health()` makes on a peer.

## UX surface
*The design-conformance gate the `/issue-{start,finish,yolo}` skills read (convention: `project-scaffolding#83`). This is a live, parseable block — the admin PWA is the FastAPI + static app under `app_web/static/`, mounted at `/admin`.*

- design spec applies: yes        # `no` would make the gate a permanent no-op; this repo serves a real admin PWA
- paths:
  - app_web/static/**/*.css
  - app_web/static/**/*.{js,html}
- key views:                      # single tabbed SPA served at `/admin/`
  - /admin/    (Hub · Models · Playground · Telemetry · Code Usage · Machines tabs)
