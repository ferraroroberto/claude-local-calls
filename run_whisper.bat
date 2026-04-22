@echo off
REM ==========================================================
REM  claude-local-calls - whisper.cpp ASR server (port 8090)
REM  OpenAI-compatible /v1/audio/transcriptions + /v1/audio/translations.
REM  Port 8090 is a shared mutual-exclusion lock with
REM  E:\automation\automation\audio\transcribe_voice.
REM ==========================================================
title claude-local-calls - whisper-small
cd /d "%~dp0"

echo ============================================================
echo   whisper-server: whisper-small on http://127.0.0.1:8090
echo   POST WAV to /v1/audio/transcriptions
echo   Ctrl+C to stop
echo ============================================================
echo.

.venv\Scripts\python.exe -m src.run_backend whisper
pause
