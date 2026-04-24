@echo off
REM ==========================================================
REM  claude-local-calls - models hub (FastAPI)
REM  Exposes /v1/messages + /v1/chat/completions on :8000
REM  Routes by model name: claude -> `claude -p`;
REM                        qwen*/glm* -> local llama-server
REM ==========================================================
title claude-local-calls - hub
cd /d "%~dp0.."

echo ============================================================
echo   claude-local-calls - models hub
echo   Local: http://127.0.0.1:8000   (docs at /docs)
echo   LAN:   binds on 0.0.0.0 - reachable from other machines
echo   Ctrl+C to stop
echo ============================================================
echo.

.venv\Scripts\python.exe -m src.run_backend hub
pause
