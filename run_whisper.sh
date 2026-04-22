#!/usr/bin/env bash
# claude-local-calls - whisper.cpp ASR server on :8090
set -euo pipefail
cd "$(dirname "$0")"
exec ./.venv/bin/python -m src.run_backend whisper
