@echo off
REM One-command setup for Windows.
REM   setup.bat        normal install (for everyone)
REM   setup.bat dev    developer install (editable + test tools)
REM Every dependency comes from a hash-pinned lockfile: wheels only, hashes required.
setlocal
cd /d "%~dp0"

set PY=python
where py >nul 2>nul && py -3.13 -c "" >nul 2>nul && set PY=py -3.13
%PY% -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,13) else 1)" 2>nul || (
    echo Python 3.13 is required. Install it from https://www.python.org/downloads/ and re-run.
    exit /b 1
)

REM Windows' 260-character path limit: the deepest file in .venv adds ~150 characters.
%PY% -c "import os, sys; sys.exit(0 if len(os.path.abspath('.')) <= 100 else 1)" || (
    echo This folder's path is too long for Windows ^(over 100 characters^): "%CD%"
    echo Move the folder somewhere shorter, e.g. C:\LectureRecorder or Documents\LectureRecorder, and re-run.
    exit /b 1
)

if not exist .venv (
    %PY% -m venv .venv || exit /b 1
)
REM A .venv whose base Python was uninstalled or moved: re-initialize it in place (packages stay).
.venv\Scripts\python.exe -c "" >nul 2>nul || (
    echo Re-linking .venv to this computer's Python 3.13...
    %PY% -m venv .venv || exit /b 1
)
set VPY=.venv\Scripts\python.exe
set PIP_DISABLE_PIP_VERSION_CHECK=1

if /i "%1"=="dev" (
    %VPY% -m pip install --require-hashes --no-deps --only-binary=:all: -r requirements\dev-win-py313.lock || exit /b 1
) else (
    %VPY% -m pip install --require-hashes --no-deps --only-binary=:all: -r requirements\runtime-win-py313.lock || exit /b 1
)
REM This package itself, always reinstalled: a moved or copied folder keeps .exe launchers that point at
REM the old folder (and would act on ITS data); this rewrites them for this folder. Nothing is downloaded.
if /i "%1"=="dev" (
    %VPY% -m pip install --no-build-isolation --no-deps --force-reinstall -e . || exit /b 1
) else (
    %VPY% -m pip install --no-build-isolation --no-deps --force-reinstall . || exit /b 1
)
%VPY% -m pip check || exit /b 1

if not exist .env copy .env.example .env >nul
if not exist config.toml copy config.example.toml config.toml >nul
REM Pick processing settings for this computer (config.auto.toml; never downloads, never edits config.toml).
REM If detection fails, setup still finishes: the app then decides at each job with the same rules.
%VPY% -m lecture_recorder.cli configure || (
    echo WARNING: could not detect this computer's settings. Setup continues with safe defaults
    echo ^(chosen at each job^). Run  .venv\Scripts\python.exe -m lecture_recorder.cli configure  later to retry.
)

echo.
echo Libraries installed. If INSTALL.bat started this, it continues with the models. Otherwise see README "Manual setup".
endlocal
