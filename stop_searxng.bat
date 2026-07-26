@echo off
REM Stop the local SearXNG service (home-automation#321). Idempotent; safe to re-run.
REM Container stops but config\settings.yml (and its secret key) is kept.

setlocal
cd /d "%~dp0"
docker compose -f docker\searxng\docker-compose.yml down
if errorlevel 1 (
  echo.
  echo SearXNG failed to stop. Is Docker Desktop running?
  exit /b 1
)
echo.
echo SearXNG stopped.
endlocal
