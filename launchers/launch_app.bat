@echo off
REM ==========================================================
REM  claude-local-calls - Streamlit control panel
REM  Welcome / Server / Testing / Playground
REM ==========================================================
title claude-local-calls - app
cd /d "%~dp0.."

echo ============================================================
echo   claude-local-calls - Streamlit control panel
echo   URL will print below (default http://localhost:8501)
echo ============================================================
echo.

.venv\Scripts\python.exe -m streamlit run app\app.py
pause
