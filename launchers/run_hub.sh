#!/usr/bin/env bash
# claude-local-calls - models hub on :8000
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./.venv/bin/python -m src.run_backend hub
