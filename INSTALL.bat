@echo off
REM Double-click me. Installs everything Lecture Recorder needs (safe to re-run; it skips what is done).
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1"
set RC=%ERRORLEVEL%
echo.
pause
exit /b %RC%
