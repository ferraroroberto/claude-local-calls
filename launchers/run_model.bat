@echo off
REM ==========================================================
REM  local-llm-hub - parameterized per-model launcher (#448 dedup).
REM  Replaces the 22 hand-rolled run_<id>.bat files, each of which
REM  re-derived the same cd / title / echo-banner / pause around a
REM  hand-retyped port already declared in config/models.yaml (and
REM  logged again by run_backend itself) - nothing caught a stale
REM  or mistyped port when a model's config changed.
REM
REM  Usage: launchers\run_model.bat <model-id>   (id from config/models.yaml)
REM  Title/port/banner are pulled live from the registry via
REM  `run_backend --banner`, so there is exactly one place they
REM  can come from.
REM ==========================================================
if "%~1"=="" (
    echo usage: run_model.bat ^<model-id^>
    exit /b 2
)
cd /d "%~dp0.."

set "_LLH_FIRST=1"
for /f "usebackq delims=" %%L in (`.venv\Scripts\python.exe -m src.run_backend --banner %1`) do (
    if defined _LLH_FIRST (
        title %%L
        set "_LLH_FIRST="
        echo ============================================================
    ) else (
        echo   %%L
    )
)
echo ============================================================
echo.

.venv\Scripts\python.exe -m src.run_backend %1
pause
