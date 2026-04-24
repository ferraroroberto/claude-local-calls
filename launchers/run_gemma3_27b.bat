@echo off
REM ==========================================================
REM  claude-local-calls - llama-server for Gemma 3 27B IT QAT (port 8084)
REM  Partial GPU offload (-ngl 50); Q4_0 QAT GGUF, ~15.6 GB.
REM  Quality benchmark tier for action-item extraction.
REM ==========================================================
title claude-local-calls - gemma3-27b-it
cd /d "%~dp0.."

echo ============================================================
echo   llama-server: gemma3-27b-it on http://127.0.0.1:8084/v1
echo   Partial offload; first load takes a minute.
echo   Ctrl+C to stop
echo ============================================================
echo.

.venv\Scripts\python.exe -m src.run_backend gemma3_27b
pause
