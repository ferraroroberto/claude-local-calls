@echo off
REM ==========================================================
REM  claude-local-calls - llama-server for Gemma 3 12B IT (port 8083)
REM  Full GPU offload; Q4_K_M GGUF fits entirely in 16 GB VRAM.
REM  Fast classifier tier for action-item extraction.
REM ==========================================================
title claude-local-calls - gemma3-12b-it
cd /d "%~dp0.."

echo ============================================================
echo   llama-server: gemma3-12b-it on http://127.0.0.1:8083/v1
echo   Ctrl+C to stop
echo ============================================================
echo.

.venv\Scripts\python.exe -m src.run_backend gemma3_12b
pause
