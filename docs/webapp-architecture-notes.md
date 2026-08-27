# Webapp architecture notes

Short reference covering the non-obvious bits of the `/admin` webapp.
Written for the next reader (you, an LLM, a future contributor) who's
debugging something and wants the gotchas in one place.

## Layout

The admin webapp is a FastAPI **sub-app** mounted at `/admin` on the
hub's own `:8000` — there is no second port, no second Python process,
no bundler. Five tabs (Hub / Models / Play / OTel / Code) cover every
operational concern.

```
src/server.py            ── parent FastAPI hub (:8000)
  app.mount("/admin", app_web.create_app())
  ↓
app_web/server.py        ── sub-app, owns its own middleware + routers
  ├── middleware.py      ── BearerTokenMiddleware (sub-app, /admin only)
  ├── routers/           ── misc / version / auth / webauthn / hub /
  │                        models / playground / services /
  │                        telemetry / code_usage / glossary / hosts
  └── static/            ── HTML + ES-module JS + CSS, no build step
```

The tray (`tray/tray.py`) drives the hub via the admin HTTP API — it
does **not** spawn model backends as its own children. The hub owns
every model subprocess.

## Non-obvious gotchas

### 1. `BaseHTTPMiddleware` runs before the parent's Mount path-strip

Starlette's `Mount("/admin", subapp)` strips `/admin` from `scope["path"]`
before the **routing layer** of the sub-app sees it — so route
decorators like `@router.get("/api/version")` (no `/admin/` prefix) match
correctly.

But `BaseHTTPMiddleware` on the sub-app runs **before** that strip
happens, so inside `dispatch()` you still see the full path
`/admin/api/version`. The sub-app's middleware strips `/admin` itself
before matching against its exempt-paths list. See
`app_web/middleware.py::BearerTokenMiddleware.dispatch`.

Why this bit us: `AUTH_EXEMPT_PREFIXES = ("/static/",)` never matched
because requests arrived as `/admin/static/...`. Result: every CSS / JS
asset returned 401 over the tunnel, leaving the SPA unstyled. The fix
is one line — strip the prefix in the middleware — but the symptom was
mysterious.

### 2. SSE named events don't fire `onmessage`

Per the EventSource spec, an SSE frame that includes `event: log\n`
becomes a **custom-named** event. Only `addEventListener('log', …)` on
the client fires; the default `onmessage` handler is **silently
skipped**.

Our `app_web/static/api.js::eventStream()` helper only subscribed to
`onmessage`. The server-side `_sse_pack(data, event="log")` calls were
emitting named events. Result: the Hub log pane and Live requests pane
appeared completely empty even though the API was streaming bytes.

Fix: drop the `event=...` keyword everywhere in the routers and let the
frames be default `message` events. If you ever want to multiplex
multiple kinds of events on one SSE stream, add
`es.addEventListener('log', …)` on the client to match.

### 3. Loopback bypass and reverse proxies

The bearer middleware bypasses auth when `request.client.host` is
`127.0.0.1` — the PC itself trusted by definition. But `tailscale
serve` and `cloudflared` both forward incoming traffic to
`127.0.0.1:8000`, so `request.client.host` looks like loopback even
when the real client is an iPhone over the tunnel.

The fix is to refuse the loopback bypass when proxy headers are
present (`X-Forwarded-For`, `X-Forwarded-Proto`, `cf-ray`,
`cf-connecting-ip`). See `app_web/middleware.py::_is_proxied`. The
list is the union of what tailscale-serve and cloudflared set; add to
it if you introduce a new reverse proxy.

### 4. Blocking calls in async endpoints pin the event loop

`subprocess.run()` and httpx-sync inside an `async def` route stall
**every** concurrent request handled by that uvicorn worker until they
return. We hit this with `install_status` shelling out to
`claude --version` / `nvidia-smi` / `llama-server --version` (multiple
seconds, cold); during that window the Playground tab's
`fetchPlaygroundModels()` call was queued behind it.

Workaround: wrap blocking calls in `await asyncio.to_thread(...)` —
see `app_web/routers/install.py::install_status` and
`app_web/routers/models.py::list_models_for_admin`. As a rule, anything
that shells out or hits disk slowly belongs off the event loop.

### 5. psutil beats netstat ~400× for "what's listening?"

`netstat -ano` (Windows) and `lsof -P` (POSIX) take ~1 s per
invocation. Walking every backend with one shell-out each multiplied
into ~10 s for the Models tab. `psutil.net_connections(kind='tcp')` is
~2 ms in-process.

See `src/server_process.py::snapshot_listening_pids`. We keep
`netstat` / `lsof` as a fallback for the rare case psutil hits
`AccessDenied` (locked-down Windows without admin), but the fast path
covers everyone normally.

### 6. Process inheritance after a hub restart

Each `bp.start()` spawns a subprocess and records the `Popen` handle
in a module-level `_STATES` dict — that state lives in **process
memory** and dies with the hub. A hub restart leaves the *previous
hub's* model children alive on their ports (they're not killed
because they were spawned with `CREATE_NEW_PROCESS_GROUP` to detach
from CTRL+BREAK), but the new hub has no `_STATES` entry → it sees
them as "external" / adopted.

`src/backend_process.py::inherit_running_backends()` runs at hub
startup, scans psutil for PIDs on the enabled-model ports, verifies
the exe looks like ours (`llama-server.exe` / `whisper-server.exe` /
`python`), and records the PID in `_BackendState.inherited_pid`.

Limitations for inherited backends:
- No log tail. Windows can't attach to another process's stdout
  post-hoc. The model card shows an empty log pane; the operator can
  Stop and Start to get a fresh hub-owned spawn with captured stdout.
- Stop uses `taskkill /F /PID` (or `SIGKILL` on POSIX) because we
  don't hold a Popen handle to politely terminate.

### 7. Tray drives the hub via HTTP, not by calling `backend_process`

Previously the tray spawned model backends as its own children. This
meant the hub couldn't see them in its `_STATES` — they were "external"
from the hub's point of view, the admin UI showed "adopted", logs were
empty.

Now the tray uses `httpx.post("http://127.0.0.1:8000/admin/api/models/{id}/start")`
and the hub spawns. Single source of truth, single subprocess parent,
log capture works.

Trade-off: the tray now depends on the hub being reachable to start
models. Acceptable — the tray autostarts the hub anyway.

### 8. Observability ring is volatile, by design

`src/hub_observability.py` keeps the last ~200 routed requests in an
in-memory deque. It is **not** the durable telemetry layer — that's
issue #4 (OpenTelemetry + Langfuse). The ring resets on hub restart;
that's the point, you want "what's happening right now" without
needing a database query.

If you find yourself wanting to query "how many tokens did I spend
last week on gemini-pro?", you want telemetry #4, not this ring.

## Pointer index

- Static asset cache busting: `src/static_versioning.py` — `?v=<hash>`
  on every `.js` / `.css` URL, computed once at boot, surfaced via
  `/admin/api/version` for visual diff.
- Hub log ring + SSE fan-out: `src/hub_log.py`. Filters
  `uvicorn.access` lines for `/admin/api/hub/*` polling so the log
  pane isn't drowned in self-noise.
- Audio proxy: `src/server.py::audio_transcriptions` /
  `audio_translations`. Forwards to the whisper backend so the call
  lands in the observability ring instead of going invisible.
- Inheritance: `src/backend_process.py::inherit_running_backends`
  (called from `src/server.py` startup).
- Bearer middleware: `app_web/middleware.py` —
  `BearerTokenMiddleware` (sub-app) and `ParentBearerTokenMiddleware`
  (parent hub). They share `_is_proxied`. Loopback callers bypass; a
  proxy header forces enforcement.
- Tailscale serve: not in this repo — the user runs
  `tailscale serve --bg --https=443 http://127.0.0.1:8000` once and
  it persists. The tray reads `webapp/cloudflared.yml` for the
  Cloudflare hostname (if configured) to surface "Copy Cloudflare URL".

## Where to add the next feature

- New tab? Add a `routers/<name>.py`, a `static/<name>.js`, an entry in
  `index.html` `<nav class="tabs">`, and route from `main.js`'s
  `onTabChange`.
- New observability metric? Push into `Observatory` in
  `src/hub_observability.py`; surface via a new `/admin/api/hub/*`
  endpoint; render in `hub.js`.
- New reverse-proxy front (e.g. Caddy, nginx)? Add its trust-IP
  header(s) to `PROXY_HEADERS` in `app_web/middleware.py`.
