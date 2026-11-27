# Installation

See [README](../README.md) for the requirements. Windows uses `INSTALL.bat`, a Mac uses `setup.sh`. The installer installs only what is missing and can be run again at any time.

## Windows
1. **Put the folder somewhere short**, for example inside your `Documents` folder (if you have the zip: right-click, *Extract All*, and don't run it from inside the zip). Avoid very long paths (the limit is 100 characters) and, if you can, folders synced by OneDrive: recordings and models are large. `%LOCALAPPDATA%\LectureRecorder` is a good alternative.
2. **Double-click `INSTALL.bat`** and press Enter. Leave the window open; it takes 20-40 minutes. If Windows says "Windows protected your PC", choose *More info*, then *Run anyway*. The installer installs only what is missing: Python 3.13 (through Windows' own `winget`), the app's libraries in a private `.venv` folder (nothing system-wide, every package hash-checked), the speech and speaker models (every file checksum-verified), and Ollama (a pinned version, checksum and signature checked) with two models. `INSTALL.bat` starts PowerShell with `-ExecutionPolicy Bypass` for that one script (`scripts\install.ps1`) only; it changes no system setting.
3. **It asks for two things from you:**
   - **Your own Hugging Face token** (see the next section).
   - **To turn off Ollama's automatic updates** in its settings (system tray icon, Settings, *Auto-download updates* off, *Expose Ollama to the network* off). The app is checked against one Ollama version; `doctor` warns about any other.
4. When it says **All set**, use the **Lecture Recorder** shortcut on your Desktop, or double-click `run.bat`. The app opens in your browser. Keep the black window open while you use it.

Something went wrong? Double-click `INSTALL.bat` again: it skips finished steps and resumes. To look without changing anything: `powershell -File scripts\install.ps1 -CheckOnly`. `.venv\Scripts\lecture-recorder.exe doctor` checks every part and says what to do about each problem.

### Hugging Face token
The speaker-separation model is "gated": you must accept its terms once with your own free account.
1. Sign up or log in at https://huggingface.co.
2. Open https://huggingface.co/pyannote/speaker-diarization-community-1 and click **Agree and access repository**.
3. Open https://huggingface.co/settings/tokens, choose **Create new token**, type **Read**, any name, and copy it.
4. Paste it when the installer asks (what you type is hidden). It is stored in `.env` in this folder, which is never shared or uploaded.
5. The app never needs the token again (it runs offline), so the installer offers to **remove it from `.env`** after the download. Say yes; you can create a new token any time.

### Ollama
Ollama runs the local model that writes summaries and powers concept search. The installer downloads the pinned version (0.35.0) from the official release, checks its SHA-256 and publisher signature, and pulls two models: `qwen3:8b` (summaries) and `qwen3-embedding:0.6b` (concept search). Keep Ollama's default of listening only on your own computer.

## macOS
Apple Silicon Macs (M1 or newer, macOS 14 Sonoma or newer) are supported. **This version was written and tested without a Mac**, so treat it as a first release: [mac-smoke-test.md](mac-smoke-test.md) is a 15-minute checklist for checking it on a real Mac, and problems are very welcome as issues. What differs from Windows:
- **Install:** install Python 3.13 for Apple Silicon from python.org (the universal2 installer), then run `./setup.sh` (or double-click `Setup.command`) in the project folder. It uses the hash-pinned `requirements/runtime-mac-py313.lock`, a private `.venv`, no `sudo`, and changes no system setting. If macOS refuses to run the scripts: `chmod +x setup.sh run.sh *.command` once, and the first time right-click a `.command` file and choose Open.
- **Models:** `.venv/bin/python scripts/download_models.py whisper`, then your own Hugging Face Read token in `.env` and `.venv/bin/python scripts/download_models.py pyannote` (remove the token afterwards), exactly as in the Hugging Face section above. `./setup.sh` offers to install Ollama v0.35.0 for you (downloaded from its official GitHub release, SHA-256 checked before unpacking, installed in `~/Applications`, no `sudo`); then turn off its automatic updates in its menu-bar settings and run `ollama pull qwen3:8b` and `ollama pull qwen3-embedding:0.6b`.
- **Check and run:** `.venv/bin/python -m lecture_recorder.cli doctor`, then `./run.sh` (or double-click `Lecture Recorder.command`). The app opens in your browser; keep the Terminal window open.
- **Speed:** transcription and speaker labels run on the CPU (there is no GPU path on a Mac in this version), so allow roughly an hour or more per lecture; process while plugged in. The app picks a thread count from your performance cores.
- **Microphone:** the first recording asks macOS for permission to use the microphone for Terminal (or Python). If you chose Don't Allow, switch it on in System Settings, Privacy & Security, Microphone. The app never changes that setting itself.
- **No "record from an app":** macOS has no equivalent of Windows' per-app capture, so that choice is hidden. Use Zoom's own local recording (Zoom settings, Recording, "Record to this Computer") and import the folder with **Import a recording**.
- Everything else (import, processing, summaries, search, units and exam prep, duplicates, class suggestions, compaction, Recently deleted, privacy and the sign-in cookie) works the same.

## Manual setup on Windows (only if `INSTALL.bat` can't be used)
1. Install Python 3.13 (64-bit) and put the folder in a short path.
2. Double-click `setup.bat` (or run it in a terminal). It creates a private `.venv` and installs exact, hash-checked versions of every dependency.
3. Models: `.venv\Scripts\python.exe scripts\download_models.py whisper`; put your Hugging Face Read token after `HF_TOKEN=` in `.env`, then `.venv\Scripts\python.exe scripts\download_models.py pyannote` and remove the token from `.env`; with Ollama running, `ollama pull qwen3:8b` and `ollama pull qwen3-embedding:0.6b`. Every model file is checked against a known SHA-256; on a mismatch the files are quarantined and not used.
4. `.venv\Scripts\lecture-recorder.exe doctor`, then `run.bat`.
