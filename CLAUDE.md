# Project Instructions

Claude Code reads this file directly as project memory; other agents reach it via the `AGENTS.md` pointer.

## This repository
Local HTTP hub routing Anthropic-shaped and OpenAI-shaped requests to multiple LLM/ASR backends, with a FastAPI + static-JS admin SPA mounted at `/admin`.
See `README.md` for setup, layout, and usage.

## Internal architecture

[`docs/architecture.mmd`](docs/architecture.mmd) is the repo's **only** component/runtime map (entry points, `src/server.py`'s routing to the Claude/Gemini/llama-server backends, the process managers, the `app_web/` admin sub-app, observability, and config) — hand-authored, not auto-generated, not covered by any test. [`docs/project-structure.md`](docs/project-structure.md) points here and keeps only the per-backend request-lifecycle sequences plus the LLM key-facts briefing (#475 fixed a drifted second copy). Update `docs/architecture.mmd` in the same PR as any material structural change (a backend added/removed, a router moved, a process manager relocated). Model **placement** (which host owns which row) is [`config/models.yaml`](config/models.yaml)'s alone — never restate it in a doc.

**Safe restart (never blanket-kill python):** canonical restart is **`tray.bat --restart`** — kills the tray subtree, reclaims the hub port **:8000** by PID scoped to this repo's `.venv` (CommandLine-matched), then starts fresh. Deliberately does **not** touch `:8090` (whisper-server, mutex-shared with `voice-transcriber`) or the llama-server model ports (8081/8082/8086/8087/8088). Fallback only, by hand: find the owner with `Get-NetTCPConnection -LocalPort 8000`, stop that PID, then relaunch via `tray.bat`. **Build confirmation:** `GET http://127.0.0.1:8000/health` returning 200 is a *liveness* signal only — its payload (`{"status": "ok"}`) is byte-identical before and after a restart, so it never proves the new build is live. For build identity poll `GET http://127.0.0.1:8000/admin/api/version` and check that `git_sha` changed — the same auth-exempt endpoint `tray.bat`, `src/config_write.py`, and `src/services.py`'s `peer_health()` already use, with `peer_health()` making the identical liveness/identity split (`/health` vs `/admin/api/version`) when it probes a peer.

## UX surface
*The design-conformance gate the `/issue-{start,finish,yolo}` skills read (convention: `project-scaffolding#83`). This is a live, parseable block — the admin PWA is the FastAPI + static app under `app_web/static/`, mounted at `/admin`.*

- design spec applies: yes        # `no` would make the gate a permanent no-op; this repo serves a real admin PWA
- paths:
  - app_web/static/**/*.css
  - app_web/static/**/*.{js,html}
- key views:                      # single tabbed SPA served at `/admin/`
  - /admin/    (Hub · Models · Playground · Telemetry · Code Usage · Machines tabs)
