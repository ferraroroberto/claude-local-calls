@echo off
REM ==========================================================
REM  claude-local-calls - start every enabled backend in its
REM  own console window. Close each window individually or run
REM  stop_all.bat (TODO) to shut them down.
REM ==========================================================
cd /d "%~dp0"

start "claude-local-calls - hub"      cmd /k .venv\Scripts\python.exe -m src.run_backend hub
start "claude-local-calls - qwen"     cmd /k .venv\Scripts\python.exe -m src.run_backend qwen
start "claude-local-calls - glm"      cmd /k .venv\Scripts\python.exe -m src.run_backend glm

echo Launched hub + qwen + glm in separate windows.
echo (If a model is not enabled on this host its window will exit immediately.)
