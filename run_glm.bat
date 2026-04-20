@echo off
REM ==========================================================
REM  claude-local-calls - llama-server for GLM-4.5-Air (port 8082)
REM  MoE CPU offload: attention on GPU, experts spill to 128 GB RAM.
REM ==========================================================
title claude-local-calls - glm-4.5-air
cd /d "%~dp0"

echo ============================================================
echo   llama-server: glm-4.5-air on http://127.0.0.1:8082/v1
echo   ~55 GB RAM committed; first load takes a minute or two.
echo   Ctrl+C to stop
echo ============================================================
echo.

.venv\Scripts\python.exe -m src.run_backend glm
pause
