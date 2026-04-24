#!/usr/bin/env bash
# claude-local-calls - start every enabled backend; each in its own
# process group so individual Ctrl+C works. Pids are printed so you
# can `kill` them; or just close the terminal.
set -uo pipefail
cd "$(dirname "$0")/.."

start() {
  local name="$1"; shift
  ./.venv/bin/python -m src.run_backend "$name" &
  echo "  $name pid=$!"
}

echo "launching hub + qwen + glm + gemma3-12b + gemma3-27b + gemma3n-e4b + whisper (disabled backends will exit immediately)..."
start hub
start qwen
start glm
start gemma3_12b
start gemma3_27b
start gemma3n_e4b
start whisper
wait
