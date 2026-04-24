#!/usr/bin/env bash
# claude-local-calls - llama-server for Gemma 3n E4B IT on :8085
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./.venv/bin/python -m src.run_backend gemma3n_e4b
