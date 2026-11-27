#!/bin/bash
# Double-click in Finder to install Lecture Recorder's libraries. Safe to run again.
# The first time, macOS may say it can't verify the developer: right-click the file, choose Open, then Open.
cd "$(dirname "$0")" || exit 1
./setup.sh
status=$?
echo
if [ $status -ne 0 ]; then
    echo "Setup stopped. Read the message above, fix it, and double-click Setup.command again."
fi
read -r -p "Press Enter to close this window." _
exit $status
