# Repo tidy-up: `launchers/` folder + docs changelog — 2026-04-24

Follow-up to [20260422-add-whisper-asr.md](20260422-add-whisper-asr.md).
Housekeeping-only pass — no behaviour change, no new features. The
repo root had grown to 18 `run_*.bat` / `.sh` / `launch_app.*`
entrypoints sitting next to `README.md`, and the `docs/` folder was
mixing its two reference docs
([project-structure.md](../project-structure.md),
[model-comparison.md](../model-comparison.md)) with four dated plan
/ post-mortem files. Moved both groups into dedicated folders and
fixed every cross-reference.

## What moved

### Launchers → `launchers/`

Everything that a human (or a Streamlit button) double-clicks to
bring up a process.

```
launchers/
├── run_hub.bat         run_hub.sh          # :8000 FastAPI hub
├── run_qwen.bat        run_qwen.sh         # :8081 llama-server Qwen 3.5 9B
├── run_glm.bat         run_glm.sh          # :8082 llama-server GLM-4.5-Air
├── run_gemma3_12b.bat  run_gemma3_12b.sh   # :8083 llama-server Gemma 3 12B IT
├── run_gemma3_27b.bat  run_gemma3_27b.sh   # :8084 llama-server Gemma 3 27B IT QAT
├── run_gemma3n_e4b.bat run_gemma3n_e4b.sh  # :8085 llama-server Gemma 3n E4B IT
├── run_whisper.bat     run_whisper.sh      # :8090 whisper.cpp ASR
├── run_all.bat         run_all.sh          # launch every enabled backend
└── launch_app.bat      launch_app.sh       # Streamlit control panel
```

All 18 files were moved with `git mv`, preserving history. Each
script's first-op was rewritten so the working directory still
resolves to the project root even though the script now lives a
level deeper:

- **Windows (`.bat`)**: `cd /d "%~dp0"` → `cd /d "%~dp0.."`
- **POSIX (`.sh`)**: `cd "$(dirname "$0")"` → `cd "$(dirname "$0")/.."`

The rest of the scripts is byte-identical — they still invoke
`.venv\Scripts\python.exe -m src.run_backend <name>` / `streamlit run
app/app.py` from the root, so `src/`, `.venv/`, `config/models.yaml`,
and `app/` are found exactly as before.

### Dated docs → `docs/changelog/`

Everything whose filename starts with a `YYYYMMDD-` prefix. These are
time-anchored plan docs and post-mortems, not standing references;
isolating them makes the two evergreen files
([project-structure.md](../project-structure.md) and
[model-comparison.md](../model-comparison.md)) the first thing a
newcomer sees in `docs/`.

```
docs/changelog/
├── 20260420-hub-with-qwen-and-glm.md                       # original hub build
├── 20260420-add-gemma-for-action-item-classification.md    # Gemma 3 12B + 27B-QAT plan
├── 20260420-glm-performance-assessment.md                  # GLM tok/s + CPU/GPU balance follow-up
├── 20260422-add-whisper-asr.md                             # whisper.cpp ASR on :8090
└── 20260424-launchers-and-docs-reorg.md                    # this file
```

Every relative link inside those four files was re-levelled: links
that pointed at `../src/...` / `../config/...` / `../tests/...` /
`../scripts/...` / `../app/...` / `../README.md` from `docs/` now
point at `../../src/...` etc. from `docs/changelog/`. Intra-changelog
links (doc-to-doc inside the folder) stayed as bare filenames, which
still resolve because the files moved together. The 20260420-gemma
plan's bullet that pointed at the Gemma launchers was updated to
`../../launchers/run_gemma3_*.{bat,sh}` and annotated with a
note that those files live in `launchers/` as of today.

## `model-comparison.md` — extra references column

The user asked for per-model website references. The
`Official docs` column was renamed to
`References (official · card · benchmarks)` and expanded so each row
carries up to three links:

| Row | Links now shown |
|---|---|
| `claude-*` | Anthropic docs · llm-stats Anthropic index |
| `qwen3.5-9b` | Qwen HF org · the unsloth GGUF we actually ship · mlx-community Mac 4-bit build |
| `glm-4.5-air` | official `zai-org/GLM-4.5-Air` card · the unsloth GGUF we ship · [llm-stats page](https://llm-stats.com/models/glm-4.5-air) |
| `gemma3-12b-it` | Gemma docs · official `google/gemma-3-12b-it` card · the unsloth GGUF we ship |
| `gemma3-27b-it` | Google's QAT announcement · official QAT-Q4_0 card · the unsloth mirror we ship |
| `gemma3n-e4b-it` | Gemma 3n docs · official E4B card · the unsloth GGUF we ship |
| `whisper-small` | whisper.cpp repo · ggml model bucket · OpenAI Whisper paper |

"GGUF we ship" is the exact HuggingFace repo `src.install --fix`
pulls for that row today — see
[config/models.yaml](../../config/models.yaml) for the `hf_repo` /
`hf_pattern` pair. Including it makes the table a one-stop answer
to "where does the file on disk come from?", a question that
previously required grepping the YAML.

## Files touched (reference list)

| File | Change |
|---|---|
| `launchers/*.{bat,sh}` (×18) | moved from repo root; `cd` target re-pointed to `..` |
| `README.md` | Layout tree, Run block, Setup caption, and architecture links now reference `launchers/` + `docs/changelog/`; the `docs/` sub-tree gained `model-comparison.md` and `changelog/` |
| `docs/project-structure.md` | `LAUNCHERS` mermaid node now shows `launchers/` prefix; `DOCS` mermaid block sprouted a `changelog/` subtree with all four dated docs + this one; Key-facts bullets for entry points were updated to prefix `launchers/` |
| `docs/model-comparison.md` | Refs column rewritten (see above); "How to add a new row" step 2 now says `launchers/run_<id>.*` and `launchers/run_all.*`; Classifier row's link bumped to `changelog/20260420-add-gemma…` |
| `docs/changelog/20260420-*.md`, `docs/changelog/20260422-*.md` | All `](../...)` project-root links re-levelled to `](../../...)`; intra-changelog links unchanged |
| `app/views/server.py` | Out-of-date caption (`run_server.bat`, never existed) replaced with `launchers/run_hub.bat` |

Not touched: `src/`, `tests/`, `scripts/`, `config/models.yaml`,
`app/views/comparison.py` (which reads `docs/model-comparison.md`
through `Path(__file__).parent.parent.parent`, so the file staying
in `docs/` means zero code changes). The installer's checks,
`run_backend.py`'s dispatch, and the Streamlit `Models` /
`Server` tabs all continue to work because none of them referenced
the old root-level launcher paths — they all shell out to
`python -m src.run_backend <name>` directly.

## Why

- **Root visibility.** Before: 18 `run_*` / `launch_*` files at the
  top of a `ls`. After: one `launchers/` entry. The top of the repo
  now shows only project-level things (`README`, `LICENSE`,
  `requirements.txt`, `.gitignore`, `config/`, `src/`, `app/`, …).
- **Docs hygiene.** Standing references (`project-structure.md`,
  `model-comparison.md`) shouldn't be interleaved alphabetically
  with dated build logs. The evergreen-vs-changelog split is the
  convention most Python projects settle on eventually; doing it
  now while the changelog is only 4 files is cheap.
- **Single source of truth for model links.** Adding the
  `llm-stats`, `mlx-community`, and "GGUF we ship" columns kills
  three separate questions ("is there a benchmark page?", "does
  MLX work on the Mac?", "which repo is actually being downloaded?")
  that previously required skimming the YAML or two separate plan
  docs.

## Verification

- `git status --porcelain` shows 22 renames (`R  old -> new`) for
  the launcher + changelog moves and 6 `M` lines for edits to
  `README.md`, `docs/project-structure.md`,
  `docs/model-comparison.md`, and `app/views/server.py` (plus the
  new `docs/changelog/20260424-...md`).
- Spot-checked each launcher with `grep '^cd ' launchers/*` — every
  `.bat` now has `cd /d "%~dp0.."`, every `.sh` has
  `cd "$(dirname "$0")/.."`.
- Did **not** execute the launchers or restart the hub — this was a
  pure path refactor, and the Streamlit `Server` / `Models` tabs
  plus `src.run_backend` dispatch don't go through the launcher
  scripts, so the running installation on the user's machine keeps
  working without a restart.

## Next

No follow-up work. If the changelog grows past ~10 files, consider
an `INDEX.md` inside `docs/changelog/` with one-line blurbs per
entry; at five files it's still easier to scan the directory.
