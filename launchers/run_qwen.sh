#!/usr/bin/env bash
# claude-local-calls - llama-server for Qwen3.5-9B on :8081
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./.venv/bin/python -m src.run_backend qwen
