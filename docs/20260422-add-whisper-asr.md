# What we did — 2026-04-22

Post-mortem of adding a local whisper.cpp ASR server to the hub as a
seventh "model" entry, and renaming the per-backend process manager to
reflect that it now manages both `llama-server` and `whisper-server`.
Pairs with [20260420-hub-with-qwen-and-glm.md](20260420-hub-with-qwen-and-glm.md)
and [20260420-add-gemma-for-action-item-classification.md](20260420-add-gemma-for-action-item-classification.md).

---

## Starting point (2026-04-21)

- Hub shipping six chat backends (claude + qwen + glm + three gemma
  rows). All text, all routed by `model` name on :8000.
- Sister project `E:\automation\automation\audio\transcribe_voice`
  runs its own whisper.cpp server on :8090 and exposes OpenAI-compatible
  `/v1/audio/transcriptions` + `/v1/audio/translations`. Its
  `whisper_server/whisper_server.yaml` was already annotated
  *"intentionally identical between this repository and claude-local-calls
  so the server binds the same port"* — the design had always intended
  both projects to share :8090 as a mutual-exclusion lock.
- [scripts/install_whisper_cpp.py](../scripts/install_whisper_cpp.py)
  had been seeded (untracked) to download the CUDA Windows build of
  whisper.cpp and normalise the binary name to `whisper-server[.exe]`.
  Everything else was missing: registry entry, process wiring, UI
  surfacing, install checks.

## Goal

Add a single new model id `whisper` to the existing scaffolding. CUDA-
backed, Windows-first, fully reusing the llama-server pattern (model
registry + backend-process manager + Streamlit Models view + `src.install`
+ `src.run_backend`). No parallel app, no new UI surface.

## Decisions (and the options we rejected)

1. **No hub passthrough for audio endpoints.** Clients (the sister
   project, curl, any future caller) talk to whisper directly on
   `127.0.0.1:8090`. The hub stays text-only. This mirrors how
   transcribe_voice already works and avoids ~30 lines of async
   streaming code that nothing in the current workflow needs. The hub's
   only job for whisper is to surface it in `/v1/models` and return an
   actionable 400 if someone POSTs chat to `whisper-small`.
2. **Rename `src/llama_process.py` → `src/backend_process.py`.** Once
   the module handles both llama-server and whisper-server, the old
   name is a misnomer. Done while we were in there.
3. **YAML-only size selection, default small.** No CLI flags, no env
   vars. To switch to `ggml-tiny.bin` / `base` / `medium` / `large-v3`,
   edit `hf_pattern` + `model_path` in [config/models.yaml](../config/models.yaml)
   and re-run `python -m src.install --fix`. Matches how every other
   model in the registry picks a quant.
4. **Branch on `engine: whisper-server`**, not a new manager class.
   Two functions (`build_command`, `is_reachable`) gain a small
   conditional; everything else stays shared.

## What we built

### Registry row ([config/models.yaml](../config/models.yaml))

```yaml
whisper:
  display_name: whisper-small
  backend: whisper           # new backend type; not chat
  engine: whisper-server
  port: 8090
  hf_repo: "ggerganov/whisper.cpp"
  hf_pattern: "ggml-small.bin"
  model_path: "models/ggml-small.bin"
  args:
    - "--threads"
    - "4"
    - "--processors"
    - "1"
    - "--inference-path"
    - "/v1/audio/transcriptions"
```

Enabled on `pc-cuda`, not on `mac-mini-m4`.

### Process manager ([src/backend_process.py](../src/backend_process.py))

Renamed from `llama_process.py`. Two engine-aware helpers, everything
else shared:

- `build_command(model)` — if `model.engine == "whisper-server"`, use
  `vendor/whisper.cpp/whisper-server[.exe]` and emit `--model <path>`
  (whisper) instead of `-m <path>` (llama). `--host` / `--port` are the
  same for both.
- `is_reachable(model)` — llama-server health checks use `/health`;
  whisper.cpp v1.8.4 has no such endpoint, so whisper probes `GET /`
  for a 200.
- Added `VENDOR_WHISPER = PROJECT_ROOT / "vendor" / "whisper.cpp"` and
  wired `start()` to extend Windows `PATH` with that directory when
  launching a whisper-engine model (CUDA DLLs ship next to the binary).
- `running_backends()` widened to `backend in ("openai", "whisper")`.

### Dispatcher + hub router

- [src/run_backend.py](../src/run_backend.py) — accepts
  `backend in ("openai", "whisper")` and picks the right vendor dir for
  the Windows PATH augmentation.
- [src/server.py](../src/server.py) — both `/v1/messages` and
  `/v1/chat/completions` handlers add an explicit 400 when the resolved
  model is whisper, with a body pointing callers at
  `http://127.0.0.1:8090/v1/audio/transcriptions`.
- `/v1/models` needed zero code changes — it iterates `enabled_models()`
  and whisper rides along for free.

### Install + download

- [src/install.py](../src/install.py) — added `_check_whisper_cpp()`
  (gated on whether any enabled model has `engine == "whisper-server"`),
  `_fix_whisper_cpp()` that shells out to
  `scripts.install_whisper_cpp.main()`, and widened `_check_models()`
  and `_check_ports()` filters from `backend == "openai"` to
  `backend in ("openai", "whisper")`.
- [scripts/download_models.py](../scripts/download_models.py) —
  `ggerganov/whisper.cpp` hosts `ggml-small.bin` at the repo root, so
  `huggingface_hub.hf_hub_download` fits unchanged once the backend
  filter accepts `whisper`.

### UI + launchers

- [app/views/models.py](../app/views/models.py) — renamed
  `_render_llama_card` → `_render_local_card`; whisper cards show 🎙
  + "whisper-server", llama cards show 🦙 + "llama-server". Routing
  widened to `m.backend in ("openai", "whisper")`. No other functional
  changes — the card reuses the same start/stop/logs/health controls.
- New `run_whisper.bat` / `run_whisper.sh`. `run_all.*` gained a
  `whisper` line so "start everything" brings it up.

### Docs + tests

- [docs/model-comparison.md](model-comparison.md) — added a
  `whisper-small` row (engine whisper.cpp, port 8090, `ggml-small.bin`
  ~466 MB, role "speech-to-text, not chat") plus a roles-at-a-glance
  row with size-switching instructions.
- [tests/test_model_registry.py](../tests/test_model_registry.py) —
  `test_whisper_entry` asserts backend=`whisper`, engine=`whisper-server`,
  port=8090, `url == "http://127.0.0.1:8090/v1"`, with per-host
  filtering checked on both `pc-cuda` and `mac-mini-m4`.
- [README.md](../README.md) + [docs/project-structure.md](project-structure.md) —
  updated for the new row and the `llama_process → backend_process`
  rename.

## Surprises we hit on the way

### `--gpu 1` is not a whisper.cpp flag

Copy-pasted from an older plan draft. whisper.cpp v1.8.4 defaults GPU
on; the opt-out is `-ng` / `--no-gpu`. Removed from `args`.

### The default inference path is `/inference`, not `/v1/audio/…`

whisper-server's default route is `/inference`. The OpenAI-compatible
path only exists if you pass `--inference-path /v1/audio/transcriptions`
at startup. Added to `args` so the sister project's `TranscriptionClient`
(which posts to `/v1/audio/transcriptions`) works unchanged.

### whisper.cpp v1.8.4 exposes exactly ONE inference path per server

The sister project's client hits *two* different URL paths depending on
its `translate` flag (`/v1/audio/transcriptions` vs
`/v1/audio/translations`). whisper.cpp only accepts one
`--inference-path` at launch. The translate-path request 404s.

Not fixed in this repo — the workarounds are on the sister side: either
set `translate: false` in
`E:\automation\automation\audio\transcribe_voice`'s config (simplest),
or patch its `transcription_client.py` to POST to a single path and
pass `translate` as a form field.

## Verification (what we actually ran)

1. `python -m src.install --fix` — pulled the CUDA whisper.cpp release
   into `vendor/whisper.cpp/` and `ggml-small.bin` into `models/`.
   Overall: ok.
2. `python -m src.run_backend whisper` — log showed whisper-server
   listening on `0.0.0.0:8090`.
3. `curl http://127.0.0.1:8090/` → 200.
4. `curl -F file=@test_silence.wav -F response_format=json
    http://127.0.0.1:8090/v1/audio/transcriptions`
   → `{"text":" [BLANK_AUDIO]\n"}`.
5. **Mutual-exclusion cross-check** — from the sister project:
   `& e:/automation/automation/.venv/Scripts/python.exe
     e:/automation/automation/audio/transcribe_voice/launcher.py server status`
   reported
   `✅ running (external — started elsewhere) @ http://127.0.0.1:8090 [external]`.
   Exactly what we wanted: one project owns the port, the other
   detects it cleanly and doesn't try to stomp it.
6. Hub text routes rejected whisper with a 400 pointing callers at
   :8090.
7. `python -m pytest -q` — registry test, install test, server test all
   pass.

## Shape of the change

Minimal, as intended — most of the existing scaffolding absorbed
whisper without modification:

| area                              | change |
| --------------------------------- | ------ |
| `config/models.yaml`              | +1 model row, +1 entry in `pc-cuda.enabled` |
| `src/backend_process.py` (renamed) | 2 small engine-branches + `VENDOR_WHISPER` |
| `src/run_backend.py`              | widen backend filter, pick vendor dir |
| `src/server.py`                   | explicit 400 in two handlers |
| `src/install.py`                  | 1 new check + 1 new fix + 2 widened filters |
| `scripts/download_models.py`      | widen backend filter |
| `app/views/models.py`             | import rename + card-icon branch |
| `tests/test_model_registry.py`    | +1 test |
| `scripts/smoke_test.py`           | skip whisper rows |
| `run_whisper.{bat,sh}`            | new (trivial) |
| `run_all.{bat,sh}`                | +1 line |
| `docs/` + `README.md`             | updates |

No new modules. No hub passthrough. No parallel app.

## What's next

- Decide how to resolve the one-inference-path limitation for the
  sister project's default translate-true config. Three options ranked
  by cost: flip sister config to `translate: false` (zero code);
  teach sister `transcription_client.py` to POST to the single
  configured path with a `translate` form field (one file, small); run
  two whisper-server processes on different ports (overkill, burns
  VRAM twice).
- If a second whisper size ever gets used regularly, consider adding a
  second registry row (e.g. `whisper_large_v3` on a different port)
  rather than toggling `model_path`. Current YAML-only size selection
  is fine for the single-model case.
