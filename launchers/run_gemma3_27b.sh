#!/usr/bin/env bash
# claude-local-calls - llama-server for Gemma 3 27B IT QAT on :8084
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./.venv/bin/python -m src.run_backend gemma3_27b
