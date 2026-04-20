@echo off
REM ==========================================================
REM  claude-local-calls - llama-server for Qwen3.5-9B (port 8081)
REM  Full GPU offload; Q4_K_M GGUF fits entirely in 16 GB VRAM.
REM ==========================================================
title claude-local-calls - qwen3.5-9b
cd /d "%~dp0"

echo ============================================================
echo   llama-server: qwen3.5-9b on http://127.0.0.1:8081/v1
echo   Ctrl+C to stop
echo ============================================================
echo.

.venv\Scripts\python.exe -m src.run_backend qwen
pause
