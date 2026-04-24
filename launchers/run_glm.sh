#!/usr/bin/env bash
# claude-local-calls - llama-server for GLM-4.5-Air on :8082
# (The Mac mini's default host profile doesn't enable glm; on that
# machine this script will exit immediately with an explanatory error.)
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./.venv/bin/python -m src.run_backend glm
