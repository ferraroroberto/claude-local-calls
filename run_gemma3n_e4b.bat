@echo off
REM ==========================================================
REM  claude-local-calls - llama-server for Gemma 3n E4B IT (port 8085)
REM  Edge/mobile-class ~4B effective params; full GPU offload.
REM ==========================================================
title claude-local-calls - gemma3n-e4b-it
cd /d "%~dp0"

echo ============================================================
echo   llama-server: gemma3n-e4b-it on http://127.0.0.1:8085/v1
echo   Ctrl+C to stop
echo ============================================================
echo.

.venv\Scripts\python.exe -m src.run_backend gemma3n_e4b
pause
