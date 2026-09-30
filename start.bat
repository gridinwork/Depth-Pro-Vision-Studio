@echo off
chcp 65001 >nul
setlocal DisableDelayedExpansion
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment was not found.
  echo Run install.bat first.
  pause
  exit /b 1
)

call ".venv\Scripts\activate.bat"
".venv\Scripts\python.exe" main.py
if errorlevel 1 goto failed
exit /b 0

:failed
echo.
echo Depth Pro Vision Studio exited with an error.
echo The traceback is above. See also logs\app.log
pause
exit /b 1
