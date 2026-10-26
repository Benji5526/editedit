@echo off
rem Double-click to start the desktop app (uses the project virtual environment, no console window).
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Virtual environment not found. See README.md - Install.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" app.py %*
