@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  echo Python virtual environment not found at venv\Scripts\python.exe
  echo Create it first: python -m venv venv
  pause
  exit /b 1
)

echo Starting the AI Customer ^& Business Intelligence web app...
echo Open http://127.0.0.1:5000 in your browser.
echo Press Ctrl+C to stop.
echo.

set FLASK_ENV=development
venv\Scripts\python.exe run_web.py
pause
