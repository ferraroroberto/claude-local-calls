@echo off
REM Start the local SearXNG service (home-automation#321). Idempotent; safe to re-run.
REM First run pulls the image and generates config\settings.yml with a fresh
REM random secret key; subsequent runs are seconds. ensure_json_format.py then
REM patches search.formats to include json (required by format=json requests)
REM without touching that secret key - a restart is only needed the run it
REM actually changes the file.

setlocal
cd /d "%~dp0"
docker compose -f docker\searxng\docker-compose.yml up -d
if errorlevel 1 (
  echo.
  echo SearXNG failed to start. Is Docker Desktop running?
  exit /b 1
)

".venv\Scripts\python.exe" docker\searxng\ensure_json_format.py docker\searxng\config\settings.yml
if errorlevel 2 (
  echo.
  echo Failed to verify/patch SearXNG's settings.yml for json output.
  exit /b 1
)
if errorlevel 1 (
  echo Enabling json output - restarting SearXNG to apply...
  docker compose -f docker\searxng\docker-compose.yml restart searxng
  if errorlevel 1 (
    echo SearXNG restart failed.
    exit /b 1
  )
)

echo.
echo SearXNG is starting at http://localhost:8085
endlocal
