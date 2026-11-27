#!/bin/bash
# Double-click in Finder to start Lecture Recorder. Keep the Terminal window open while you use it.
# The first time, macOS may say it can't verify the developer: right-click the file, choose Open, then Open.
cd "$(dirname "$0")" || exit 1
./run.sh
status=$?
echo
echo "Lecture Recorder has stopped. You can close this window."
exit $status
