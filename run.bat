@echo off
REM Starts Lecture Recorder from this folder (data/, models/, .env live here).
REM Uses "python -m" rather than the .exe launcher: it keeps working if this folder is moved or copied.
cd /d "%~dp0"
set "LECTURE_RECORDER_HOME=%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Lecture Recorder is not set up in this folder yet. Run setup.bat first.
    pause
    exit /b 1
)
.venv\Scripts\python.exe -c "" 2>nul || (
    echo The Python this install was set up with is missing ^(Python 3.13 uninstalled or moved^).
    echo Install Python 3.13 from https://www.python.org/downloads/ and run setup.bat again.
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m lecture_recorder.cli serve %*
