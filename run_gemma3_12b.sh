#!/usr/bin/env bash
# claude-local-calls - llama-server for Gemma 3 12B IT on :8083
set -euo pipefail
cd "$(dirname "$0")"
exec ./.venv/bin/python -m src.run_backend gemma3_12b
