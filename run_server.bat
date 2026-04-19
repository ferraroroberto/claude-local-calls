@echo off
REM ==========================================================
REM  claude-local-calls - FastAPI server (standalone)
REM  Exposes POST /v1/messages on http://127.0.0.1:8000
REM  Shells out to `claude -p` using your local Claude auth
REM ==========================================================
title claude-local-calls - server
cd /d "%~dp0"

echo ============================================================
echo   claude-local-calls - FastAPI server
echo   Local: http://127.0.0.1:8000   (docs at /docs, landing at /)
echo   LAN:   binds on 0.0.0.0 - reachable from other machines
echo          (Windows may prompt to allow through firewall)
echo   Ctrl+C to stop
echo ============================================================
echo.

.venv\Scripts\python.exe -m src.server
pause
