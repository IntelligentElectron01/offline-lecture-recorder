#!/bin/bash
# One-command setup for macOS (Apple Silicon, macOS 14 or newer).
#   ./setup.sh        normal install (for everyone)
#   ./setup.sh dev    developer install (editable + test tools)
# Every dependency comes from a hash-pinned lockfile: wheels only, hashes required.
# Nothing here uses sudo or changes a system setting; everything goes into the private .venv folder.
set -u
cd "$(dirname "$0")" || exit 1

if [ "$(uname -s)" != "Darwin" ]; then
    echo "This script is for macOS. On Windows use setup.bat."
    exit 1
fi
if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null)" = "1" ]; then
    echo "This Terminal is running under Rosetta (translated). Open Terminal natively (Get Info on Terminal, untick \"Open using Rosetta\"), then run ./setup.sh again."
    exit 1
fi
if [ "$(uname -m)" != "arm64" ]; then
    echo "This needs a Mac with Apple Silicon (M1 or newer). Intel Macs are not supported."
    exit 1
fi

macos_major="$(sw_vers -productVersion 2>/dev/null | cut -d. -f1)"
if [ -n "$macos_major" ] && [ "$macos_major" -lt 14 ] 2>/dev/null; then
    echo "This needs macOS 14 (Sonoma) or newer (this Mac has $(sw_vers -productVersion)). Update macOS first."
    exit 1
fi

# Python 3.13 that runs natively (arm64), not under Rosetta
PY=""
for cand in python3.13 /opt/homebrew/bin/python3.13 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import sys, platform; sys.exit(0 if sys.version_info[:2]==(3,13) and platform.machine()=='arm64' else 1)" 2>/dev/null; then
        PY="$cand"
        break
    fi
done
if [ -z "$PY" ]; then
    echo "Python 3.13 for Apple Silicon is required (found none)."
    echo "Install it from https://www.python.org/downloads/macos/ (the macOS 64-bit universal2 installer), then run ./setup.sh again."
    exit 1
fi

if [ "${#PWD}" -gt 150 ]; then
    echo "This folder's path is very long. Move the folder somewhere shorter (for example your Documents folder) and run ./setup.sh again."
    exit 1
fi
case "$PWD" in
    *"/Library/Mobile Documents/"*|*"/iCloud Drive/"*)
        echo "Warning: this folder is synced by iCloud Drive. Recordings and models are large and iCloud will try to upload them."
        echo "A folder outside iCloud (for example ~/LectureRecorder) is better. You can move it and run ./setup.sh again."
        ;;
esac

MODE="runtime"
[ "${1:-}" = "dev" ] && MODE="dev"
LOCK="requirements/${MODE}-mac-py313.lock"
if [ ! -f "$LOCK" ]; then
    echo "$LOCK is missing: this copy does not include the Mac package list yet."
    exit 1
fi

if [ ! -d .venv ]; then
    "$PY" -m venv .venv || exit 1
fi
# A .venv whose base Python was removed or moved: re-initialize it in place (packages stay).
if ! .venv/bin/python -c "" >/dev/null 2>&1; then
    echo "Re-linking .venv to this computer's Python 3.13..."
    "$PY" -m venv .venv || exit 1
fi
VPY=.venv/bin/python
export PIP_DISABLE_PIP_VERSION_CHECK=1

"$VPY" -m pip install --require-hashes --no-deps --only-binary=:all: -r "$LOCK" || exit 1
# This package itself, always reinstalled so a moved or copied folder gets launchers for this folder. Nothing is downloaded.
if [ "$MODE" = "dev" ]; then
    "$VPY" -m pip install --no-build-isolation --no-deps --force-reinstall -e . || exit 1
else
    "$VPY" -m pip install --no-build-isolation --no-deps --force-reinstall . || exit 1
fi
"$VPY" -m pip check || exit 1

[ -f .env ] || cp .env.example .env
[ -f config.toml ] || cp config.example.toml config.toml
# Pick processing settings for this Mac (config.auto.toml; never downloads, never edits config.toml).
"$VPY" -m lecture_recorder.cli configure || {
    echo "WARNING: could not detect this Mac's settings. Setup continues with safe defaults (chosen at each job)."
    echo "Run  .venv/bin/python -m lecture_recorder.cli configure  later to retry."
}

# ---- Ollama (the local model that writes summaries): the one version this project is checked against, from its
# official GitHub release, verified by SHA-256 before anything is unpacked, installed for this user only (no sudo).
OLLAMA_VERSION="0.35.0"
OLLAMA_URL="https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/Ollama-darwin.zip"
OLLAMA_SHA256="3458972dfca5b3f8a4297a8d7ff503da3e180ec9452ba8de82c69c5ba3aacdd9"

install_ollama() {
    if [ -d "$HOME/Applications/Ollama.app" ] || [ -d /Applications/Ollama.app ] || command -v ollama >/dev/null 2>&1; then
        echo "Ollama is already installed."
        return 0
    fi
    if [ ! -t 0 ]; then
        echo "Ollama is not installed. Run ./setup.sh in a Terminal window to be offered the install."
        return 0
    fi
    echo
    echo "Ollama v${OLLAMA_VERSION} is not installed. It runs the local model that writes summaries (about 200 MB)."
    printf "Download it from github.com/ollama/ollama and install it into ~/Applications now? (no sudo) [Y/n] "
    read -r answer
    case "$answer" in n|N|no|NO) echo "Skipped. Install Ollama v${OLLAMA_VERSION} yourself from https://ollama.com/download."; return 0 ;; esac
    tmp="$(mktemp -d)" || return 1
    if ! curl -fL --proto '=https' --tlsv1.2 -o "$tmp/Ollama-darwin.zip" "$OLLAMA_URL"; then
        echo "The download failed. Check your internet connection and run ./setup.sh again."
        rm -rf "$tmp"
        return 1
    fi
    actual="$(shasum -a 256 "$tmp/Ollama-darwin.zip" | cut -d' ' -f1)"
    if [ "$actual" != "$OLLAMA_SHA256" ]; then
        echo "The downloaded file's checksum did not match, so it was NOT unpacked."
        echo "  expected $OLLAMA_SHA256"
        echo "  got      $actual"
        rm -rf "$tmp"
        return 1
    fi
    mkdir -p "$HOME/Applications" || { rm -rf "$tmp"; return 1; }
    ditto -x -k "$tmp/Ollama-darwin.zip" "$HOME/Applications" || { rm -rf "$tmp"; return 1; }
    rm -rf "$tmp"
    echo "Ollama installed in ~/Applications."
    echo
    echo "REQUIRED: open Ollama once (double-click it in ~/Applications), click its menu-bar icon, choose Settings, and turn OFF"
    echo "'Auto-download updates' (and keep 'Expose Ollama to the network' off). The app is checked against this one version;"
    echo "'doctor' warns about any other."
}
install_ollama || echo "Ollama was not installed (see above); the rest of the setup is done."

echo
echo "Libraries installed. Next steps (see the README, section \"macOS\"):"
echo "  1. Models:  .venv/bin/python scripts/download_models.py whisper"
echo "              .venv/bin/python scripts/download_models.py pyannote   (needs your free Hugging Face token in .env)"
echo "  2. Ollama:  with Ollama open, run  ollama pull qwen3:8b  and  ollama pull qwen3-embedding:0.6b"
echo "  3. Check:   .venv/bin/python -m lecture_recorder.cli doctor"
echo "  4. Start:   ./run.sh   (or double-click \"Lecture Recorder.command\")"
