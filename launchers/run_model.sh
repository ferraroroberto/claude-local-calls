#!/usr/bin/env bash
# local-llm-hub - parameterized per-model launcher (#448 dedup).
# Replaces the 11 hand-rolled run_<id>.sh files, each of which was an
# identical shebang / set -euo pipefail / cd / exec around one hardcoded
# model id.
#
# Usage: launchers/run_model.sh <model-id>   (id from config/models.yaml)
set -euo pipefail
if [ -z "${1:-}" ]; then
  echo "usage: run_model.sh <model-id>" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
exec ./.venv/bin/python -m src.run_backend "$1"
