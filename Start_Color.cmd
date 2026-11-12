@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Local environment missing. Follow Installation in README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run.py color %*
if errorlevel 1 echo Program stopped with an error. Read the message above.
pause
