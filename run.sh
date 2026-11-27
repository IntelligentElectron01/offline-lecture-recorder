#!/bin/bash
# Starts Lecture Recorder from this folder (data/, models/, .env live here). macOS.
# Uses "python -m" rather than the launcher script: it keeps working if this folder is moved or copied.
cd "$(dirname "$0")" || exit 1
export LECTURE_RECORDER_HOME="$PWD"
if [ ! -x .venv/bin/python ]; then
    echo "Lecture Recorder is not set up in this folder yet. Run ./setup.sh first."
    exit 1
fi
if ! .venv/bin/python -c "" 2>/dev/null; then
    echo "The Python this install was set up with is missing (Python 3.13 uninstalled or moved)."
    echo "Install Python 3.13 from https://www.python.org/downloads/macos/ and run ./setup.sh again."
    exit 1
fi
exec .venv/bin/python -m lecture_recorder.cli serve "$@"
