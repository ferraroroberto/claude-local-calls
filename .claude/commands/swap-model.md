---
description: Interactively swap one (or more) of the active local roles to a different model. Reads the latest frontier run, asks the user what to change, and edits config/models.yaml + syncs docs/tests + (optionally) downloads the weights. Human-in-the-loop. Never touches the `claude` subscription path.
---

You are running a **manual, ad-hoc role swap** for the local LLM hub.
The plan is built interactively with the user — do not assume what
they want; ask. Plan-mode rules apply throughout.

## Hard constraints

- **Never** touch the `claude` model row in `config/models.yaml` or
  any `claude-*` alias. The Anthropic subscription path is not a
  local role.
- **Never** delete or overwrite `models/*` weights without explicit
  user confirmation in this session. (Adding new weights via
  `scripts/download_models.py` is fine after they confirm.)
- The four valid local roles are: `agentic_light`, `agentic_heavy`,
  `audio.transcribe`, `audio.translate`. Anything else is out of scope.
- Edit `config/models.yaml` as text (or with surgical replacements).
  Do **not** round-trip the whole file through `yaml.safe_dump` — it
  will strip the heavy in-line commenting that documents each entry.
- **Role aliases follow the role pointer.** The `agentic_light` and
  `agentic_heavy` slot rows carry an `aliases:` entry containing the
  role name itself, so external clients (voice-transcriber, openClaw,
  ad-hoc curl) can address `model="agentic_light"` and have the hub
  resolve to whoever currently holds that slot. On `upgrade` /
  `retire`, the role alias **must move** from the ex-incumbent row to
  the new role-holder's row in the same edit. Leaving it on the
  ex-incumbent silently keeps clients pointed at the old model after
  the swap. Audio roles do not currently use this pattern.
- After every `upgrade` or `retire` action, **the swap is not done
  until the rest of the repo agrees with the new role pointer**.
  README, `docs/model-comparison.md`, and the tests that name the
  active fast-lane / deep-lane model all carry copies of the
  role-bound ids and display names. (`launchers/` does not — it is
  parameterized off `config/models.yaml`, #448.) Step §5b enforces
  this sync — it is part of the workflow, not a "nice to have".

## Workflow

### 1. Load context (read-only)

- `config/models.yaml` — current `models:`, `roles:`, and the active
  host's `enabled:` list. Resolve the active host via
  `python -m src.host_profile` if needed (or just trust `default: true`
  + `LOCAL_LLM_HUB_HOST`).
- `docs/frontier/runs/LATEST` → that run's `report.md` and
  `frontier.json` for the recommendations (`shortlist`).
- Nothing to inspect in `launchers/` for this — the parameterized
  `launchers/run_model.bat` / `.sh <id>` (#448) already covers any
  registry id; there is no per-model launcher file to write or template.

### 2. Show the current state to the user

Print a compact summary:

```
Active host: tower
Roles right now:
  agentic_light    → gemma4_e4b
  agentic_heavy    → gemma4_26b
  audio_transcribe → whisper
  audio_translate  → whisper_translate
Latest frontier recommendations (docs/frontier/runs/<date>/):
  agentic_light    → upgrade   to qwen3_4b   (reason: <one line>)
  agentic_heavy    → keep
  audio_transcribe → runtime_upgrade (faster-whisper) — manual engine work
  audio_translate  → watch     (fallback only)
```

### 3. Ask the user what to do

One question at a time, with multi-choice where applicable:
- Which role(s) to act on this session
- For each role, which action: `keep` | `upgrade` | `runtime_upgrade` |
  `retire` | `watch`
- For `upgrade`: target model_id (default = shortlist's
  `recommended_id`). If the id is not yet in `models.yaml`, ask for
  `hf_repo`, `hf_pattern`, `model_path`, `port`, and any non-default
  `args` (e.g. `-c`, `--flash-attn on`, `--reasoning-format none`).
  Use the previous run's metadata + the report's per-model notes as
  defaults.
- For `runtime_upgrade`: which runtime, and confirm the user
  understands this is a documented change only (engine code is manual).
- For `retire`: confirm twice. Refuse if it would leave the role
  empty unless paired with an `upgrade` in the same session.
- Whether to download weights now (`scripts/download_models.py
  --only <id>`).
- Whether to also remove the previous role-holder from the host's
  `enabled:` list (default: leave it in for ad-hoc bring-up).

### 4. Show the planned diff and confirm

Before any write, print a clear diff. Always include the §5b sync
section so the user sees the full blast radius up front:

```
Will edit config/models.yaml:
  + models.qwen3_4b: { display_name: qwen3-4b-2507, backend: openai, ... }
  ~ roles.agentic_light.model_id: gemma4_e4b → qwen3_4b
  ~ models.gemma4_e4b.aliases: ["agentic_light"] → []   (alias follows role)
  ~ models.qwen3_4b.aliases:  []  → ["agentic_light"]   (new role-holder)
  ~ hosts.tower.enabled: + qwen3_4b
Will run: python scripts/download_models.py --only qwen3_4b   (~3 GB)
Will append to docs/frontier/runs/<latest>/report.md §10

Will sync code + docs (§5b — required for upgrade/retire):
  ~ README.md                (active-rotation bullet, role table, ASCII
                              diagram, autostart default, launcher list,
                              run_backend examples, Python SDK example)
  ~ docs/model-comparison.md (registry table row + role row)
  ~ docs/project-structure.md (local-backend lifecycle header, entry-points
                              bullet, key-facts "Purpose" paragraph)
  ~ docs/architecture.mmd    (LLAMA node: active ports/models)
    launchers/               (no edit — roster derived from models.yaml, #448)
  ~ tests/test_router.py + tests/test_streaming.py
                             (model name + port assertion match the new role)
```

Get explicit user approval ("yes" / "go") before executing.

### 5. Execute

- Edit `config/models.yaml` in place (text edits, preserve comments).
  For `agentic_light` / `agentic_heavy` upgrades, in the same edit:
  remove the role name from the ex-incumbent's `aliases:` list
  (delete the field entirely if it becomes empty) and add it to the
  new role-holder's `aliases:` list. The alias is the client-facing
  contract — the `roles:` pointer is internal bookkeeping; both must
  flip together or the hub will keep routing `model="agentic_light"`
  to the old row.
- Nothing to write in `launchers/` — `run_model.bat` / `.sh <id>` (#448)
  already covers any registry id with no per-model file needed.
- If the user approved download: shell out to
  `& .\.venv\Scripts\python.exe scripts/download_models.py --only <id>`
  on Windows (or `./.venv/bin/python ...` on POSIX) and stream the
  output. If it fails, stop and report — do not proceed with the
  remaining steps.
- Append a new row to the **§10 Current decisions** table in the
  latest run's `report.md`, including today's date in the "Decided"
  column. Update the existing role row if one exists.

### 5b. Sync code + docs to the new role pointer

**Skip this step for `keep` / `watch` / `runtime_upgrade` actions** —
those don't change the role-bound id and only touch §10. **Run it
for `upgrade` and `retire`** before declaring the swap done.

The role pointer is duplicated across several places in the repo so
each file stands on its own; flipping the yaml without updating them
leaves the codebase in a misleading state where docs claim one model,
the registry returns another, and tests assert against the
ex-incumbent.

1. **Audit references.** Grep the repo for both the old and new
   identifiers and group the hits:
   ```
   grep -rn "<old_id>\|<old_display_name>" --exclude-dir=.venv \
     --exclude-dir=models --exclude-dir=docs/frontier/runs
   ```
   Files under `docs/frontier/runs/<old>/` are **historical snapshots**
   — leave them. The §10 table in the *current* run's `report.md` was
   already updated in §5; don't touch it again. Everything else is fair
   game.

2. **README.md.** Update every place that named the old role-holder:
   - the "Active rotation" bullet at the top
   - the role table under "Roles & bi-weekly refresh"
   - the ASCII architecture diagram
   - the launcher list under "Layout"
   - the install description (model file sizes / what gets downloaded)
   - the launcher commands under "Run" (Active rotation block; add a
     "Fallback / ad-hoc" block if the previous holder is kept enabled)
   - the tray autostart default
   - the `python -m src.run_backend …` example list
   - the Python SDK code sample for the agentic_light / agentic_heavy
     role (whichever one swapped)

   When the previous holder is kept in `enabled:` for fallback (the
   default), add a one-line note in the "Demoted / fallback" section
   explaining when it fires and which launcher brings it up. Don't
   silently delete it — readers need to know the row in `enabled:`
   isn't a bug.

3. **`docs/model-comparison.md`.** This is the canonical per-model
   reference; the admin SPA links to it on GitHub from its footer
   (there is no in-app Comparison view) — edit the markdown directly.
   - The big registry table: insert a row for the new model
     (Family · Params · Quant · GGUF size · Context · VRAM fit · Hub
     port · tok/s · References). Pull numbers from the latest
     frontier `report.md` and the model card.
   - The "Roles at a glance" table: rewrite the row for the swapped
     role with a one-line rationale tied to the frontier reading.
   - If the previous holder is kept as fallback, mark its row in the
     registry table with a "(fallback)" tag rather than removing it.

4. **`docs/project-structure.md`.** Carries role-bound names in the
   request-lifecycle sequences and the "Key facts for LLM context" prose
   (it no longer holds a component diagram or a `models/` listing — those
   live in `docs/architecture.mmd` and README's "Layout", #475):
   - The **Local backend** request-lifecycle section header — list the
     new model in the active set, demote the ex-incumbent into the
     ad-hoc parenthetical.
   - The "near-passthrough" prose under the local-backend diagram —
     update the bracketed model list.
   - The "Entry points" bullet under "Key facts for LLM context" —
     update the active vs. ad-hoc split for `python -m src.run_backend`.
   - The "Purpose" paragraph at the top of "Key facts for LLM
     context" — rewrite the backend list so the active rotation reads
     as the active rotation and the fallback / ad-hoc rows are clearly
     labelled.

4b. **`docs/architecture.mmd`.** Its `LLAMA` node names the active
   llama-server ports/models — update it when a role-holder changes.
   Placement (which host owns the row) stays `config/models.yaml`'s
   alone; don't restate it here.

5. **`launchers/`.** Nothing to edit. `run_all.{bat,sh}` derives its
   roster live from `config/models.yaml` (`run_backend
   --list-launchable`, #448) and its summary line is generic, so the
   new id is picked up the moment the yaml lands — and the
   ex-incumbent keeps being launched for as long as it stays in
   `enabled:`. Do **not** add a `start … run_backend <new_id>` line or
   name the id in the `echo`: that reintroduces the hardcoded roster
   #448 removed.

6. **Tests.** The unit tests carry the active fast-lane and
   deep-lane display names as in-test fixtures so the registry and
   router code paths exercise *the actual rotation*, not a stale name.
   - `tests/test_router.py` — for `agentic_light` swaps, replace
     `"<old_display_name>"` with `"<new_display_name>"` in fixture
     responses, request bodies, and assertions; update the port-in-
     `base_url` assertion to the new port; ensure the
     `test_list_models_includes_enabled` check names the new model.
   - `tests/test_streaming.py` — same model-name swap in the SSE and
     non-stream fixtures.
   - `tests/test_model_registry.py` — leave alone unless a row was
     **retired** outright. The registry test uses synthetic configs;
     it doesn't depend on the role pointer.

7. **Run the unit tests** to prove nothing regressed:
   ```
   $env:LOCAL_LLM_HUB_HOST = "tower"
   & .\.venv\Scripts\python.exe -m pytest -q
   ```
   On POSIX, `LOCAL_LLM_HUB_HOST=tower ./.venv/bin/python -m pytest -q`.
   If anything fails, fix it before moving on; do not declare the swap
   done with red tests.

8. **Final grep sweep.** Re-run the audit grep from step 1. Anything
   left should be one of:
   - `docs/frontier/runs/<previous>/…` — historical, leave it
   - explicit "fallback" / "previous role-holder" notes you just wrote
   - `config/models.yaml` rows for the previous holder (if kept)

   If any active doc, code path, or test still names the old
   role-holder as the *current* role, fix it now. The rule of thumb:
   after this step a fresh reader of README + welcome + the
   comparison page should never come away thinking the old model is
   still the active role.

### 6. Restart the affected backends and validate live

**Skip this for `keep` / `watch` / `runtime_upgrade` doc-only swaps.**
For `upgrade` and `retire` actions, the swap is not done until the
new backend is running on its port and a real request goes through
the hub end-to-end. Walk through these in order; ask the user to
confirm before stopping anything they might still be using.

1. **Inventory what's bound right now.** The hub adopts external
   processes; the hub autostarts the registry-derived desired set (#430);
   per-model launchers may have been started by hand. Don't assume —
   look:
   ```
   Get-NetTCPConnection -State Listen `
     -LocalPort 8000,8081,8082,8086,8087,8088,8090,8091 `
     -ErrorAction SilentlyContinue |
     Select-Object LocalPort, OwningProcess,
       @{n='ProcessName';e={(Get-Process -Id $_.OwningProcess `
         -ErrorAction SilentlyContinue).ProcessName}} |
     Format-Table -AutoSize
   ```
   POSIX equivalent: `lsof -iTCP -sTCP:LISTEN -nP | grep -E '808[0-9]|8000'`.
   Map each LocalPort back to a `models.yaml` entry to know which
   model owns which PID.

2. **Stop the previous role-holder if (and only if) it's running.**
   Two cases:
   - **Kept in `enabled:` for fallback (default).** The new backend
     binds a different port, so the old one *can* keep running. But
     leaving it loaded steals VRAM from the new model and creates
     the false impression that the old model is still the active
     role. Stop it. Use the admin SPA's Models tab's *Stop* button when
     possible (it stops cleanly via `src.backend_process.stop`); if
     the process is adopted (no log tail), use *Stop external (PID
     xxx)* on the same tab, or `Stop-Process -Id <pid>` as a last
     resort.
   - **Retired outright.** Same — stop it, then optionally remove
     the `config/models.yaml` row in a follow-up.

   The hub on `:8000` does **not** need to stop. `src.model_registry`
   re-reads `config/models.yaml` per request, so the new role pointer
   is live the moment the yaml is saved. Do not bounce the hub unless
   the user asks (and never bounce the tray — its log buffer is in-
   process and a restart loses history the user may want).

3. **Start the new backend.** Run the parameterized launcher.
   Llama-server takes 3-30 s to load the GGUF and warm up; do not
   declare it ready on the basis of the launcher having printed a
   banner.
   ```
   # Foreground (own terminal): launchers\run_model.bat <new_id>
   # Background from this session:
   & .\.venv\Scripts\python.exe -m src.run_backend <new_id>
   ```
   When you start it as a background task from the swap session,
   redirect stdout to a known log file so the readiness probe in
   step 4 can grep it.

4. **Wait until the model's port answers `/health`.** Poll, don't
   sleep on a hunch. llama-server exposes `/health` on its own port;
   200 means the model is loaded and accepting completions.
   ```
   until curl -s -o /dev/null -w "%{http_code}" `
       http://127.0.0.1:<new_port>/health 2>$null | Select-String 200; `
       do Start-Sleep 2; done
   ```
   POSIX: `until curl -sf http://127.0.0.1:<new_port>/health
   >/dev/null; do sleep 2; done`. Cap your wait — if it doesn't come
   up within ~5 minutes, tail the launcher log and stop; do not
   march on with a half-loaded model.

5. **Run the smoke suite.** This is the canonical end-to-end check
   for the active rotation; it iterates every enabled model, skips
   ones whose port isn't reachable, and asserts a 200 from the hub
   for each chat backend:
   ```
   & .\.venv\Scripts\python.exe scripts\smoke_test.py
   ```
   Expected outcome after an `agentic_light` swap with the previous
   holder kept enabled but stopped: the new model passes, the
   previous role-holder appears in the **skipped** list (port not
   reachable), and the audio backends appear in **skipped** as
   non-chat. **Failures must not be ignored.** A 4xx/5xx for the new
   model means the registry is wrong, the port mapping is wrong, or
   the model didn't actually load — fix before moving on.

   The smoke test caps `max_tokens` low for cheapness; reasoning
   models like Qwen3-style hybrids may emit `<think>` content in the
   first 64 tokens because the response was cut off mid-reasoning.
   That is **expected** — the test asserts HTTP 200, not response
   shape. Don't chase it.

6. **Hit the new model directly through the hub** at least once with
   a small prompt to confirm the routing path (client → hub → new
   port → response):
   ```
   curl -s http://127.0.0.1:8000/v1/messages `
     -H "Content-Type: application/json" `
     -d '{"model":"<new_display_name>","max_tokens":32,
          "messages":[{"role":"user","content":"reply with: ok"}]}'
   ```
   The smoke suite already does this, but a manual call is cheap
   insurance and gives the user a copy-paste they can reuse later.

7. **Re-list `/v1/models` from the hub** to confirm the new id is
   exposed and the previous one (if kept enabled) is also listed:
   `curl -s http://127.0.0.1:8000/v1/models | jq '.data[].id'`.

### 7. Report and stop

- Re-read `config/models.yaml` and the role's row to confirm the
  pointer is correct after the live restart.
- Confirm `launchers\run_model.bat <new_id>` resolves the new row
  (no per-model file needed — #448).
- Summarize for the user:
  - what was running, what was stopped, what was started
  - smoke test result (passed / skipped / failed counts)
  - the `curl` one-liner from §6.6 they can rerun any time
  - any non-default startup follow-up — e.g. "next hub boot will
    autostart `<new_id>` because it is `startup: eager` on this host
    (#430); no action needed unless you also want to drop the previous
    holder from `enabled:`."
- Optionally summarize the swap in the GitHub issue (or open one)
  rather than writing a dated changelog file.

Do **not** edit the `claude` row. Do **not** restart or edit the
tray (its autostart list was already updated in §5; the change takes
effect on next tray boot, which the user controls). Do **not**
restart the hub unless step 6 surfaced a routing bug that genuinely
requires it.

## What success looks like

- `roles.<role>.model_id` flipped to the new id
- For `agentic_light` / `agentic_heavy` swaps: the role name appears
  in the new model row's `aliases:` list and is gone from the
  ex-incumbent's. `curl /v1/messages` with `model="<role_name>"`
  resolves to the new backend.
- `models.<new_id>` row present with all the fields needed by
  `src.run_backend`
- Active host's `enabled:` list contains the new id
- the new id is `startup: eager` on the intended host (when the swapped
  role should autostart, #430)
- `launchers/` is untouched — `run_model.bat` / `.sh <new_id>` needs no
  new file and `run_all.{bat,sh}` needs no new line: both are
  parameterized off `config/models.yaml` (#448)
- (If requested) weights downloaded to `models/`
- `docs/frontier/runs/<latest>/report.md` §10 reflects the new
  decision with today's date
- README.md, `docs/model-comparison.md`, and
  `docs/project-structure.md` describe the new role-holder as the
  *current* role-holder, and the previous one (if kept) is clearly
  labelled as fallback / ad-hoc
- `tests/test_router.py` and `tests/test_streaming.py` reference the
  new model display_name and port; full pytest suite passes
- `docs/model-comparison.md` (linked from the admin SPA footer on
  GitHub) shows the new model in the registry table and the new
  role-row rationale
- Smoke test passes for the new backend (claude row + the new
  role-holder both green; previous holder either green if kept
  running, or skipped if stopped per §6.2)
- The new backend's port answers `/health` with 200 and the hub's
  `/v1/messages` returns 200 for `model="<new_display_name>"`
- The previous role-holder, if kept enabled, was actually stopped
  during this session (so the user isn't accidentally still running
  the ex-incumbent against the new role pointer)
- A fresh `grep` for the old role-holder's id / display_name finds
  hits only in historical changelogs, prior frontier runs, fallback
  notes you wrote, and (if kept enabled) its own `config/models.yaml` row
