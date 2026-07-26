@echo off
REM Start the local SearXNG service (home-automation#321). Idempotent; safe to re-run.
REM First run pulls the image and generates config\settings.yml with a fresh
REM random secret key; subsequent runs are seconds.

setlocal
cd /d "%~dp0"
docker compose -f docker\searxng\docker-compose.yml up -d
if errorlevel 1 (
  echo.
  echo SearXNG failed to start. Is Docker Desktop running?
  exit /b 1
)
echo.
echo SearXNG is starting at http://localhost:8085
endlocal
