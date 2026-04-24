#!/usr/bin/env bash
# claude-local-calls - Streamlit control panel
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./.venv/bin/python -m streamlit run app/app.py
