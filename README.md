# Offline Lecture Recorder

[![windows](https://github.com/IntelligentElectron01/offline-lecture-recorder/actions/workflows/windows.yml/badge.svg)](https://github.com/IntelligentElectron01/offline-lecture-recorder/actions/workflows/windows.yml)
[![macos](https://github.com/IntelligentElectron01/offline-lecture-recorder/actions/workflows/mac.yml/badge.svg)](https://github.com/IntelligentElectron01/offline-lecture-recorder/actions/workflows/mac.yml)

Offline Lecture Recorder records lectures (or imports recordings) and produces a transcript with speaker labels, a summary, search and exam-prep notes. It runs entirely on your own computer, on Windows or an Apple Silicon Mac: no cloud service is used and no audio or text leaves your machine.

> **You are responsible for getting permission before recording anyone.** Follow your school's rules and local recording laws.

## Features

- **Transcription** with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and **speaker labels** with pyannote, on an NVIDIA GPU or, more slowly, on the CPU.
- **Summaries and concept search** from a local model run by [Ollama](https://ollama.com).
- **Classes, units and exam prep:** split a class into units and prepare for an exam from the units you choose.
- **Exact and concept search** across every lecture, with a jump to that moment in the recording.
- Recording from a microphone or from one app's sound (for example Zoom), importing audio and Zoom recordings, a verified compact copy of every recording, duplicate detection, and "Recently deleted".

## Requirements

| | Requirement | Notes |
|---|---|---|
| OS | **Windows 10 (version 2004 or later) or Windows 11**, 64-bit Intel/AMD; or a **Mac with Apple Silicon** (M1 or newer, macOS 14 or newer) | Intel Macs, Windows on ARM and Linux are not supported. |
| GPU | An **NVIDIA GPU with 4 GB or more** and a current driver is best | Without one (including AMD and Intel graphics) everything still works on the CPU, several times slower: roughly 1-3 hours for a 90-minute lecture. The app picks the right settings for your computer. |
| RAM | 16 GB recommended | 8 GB works for transcription; the summary model needs about 6 GB more, so summaries are slow on 8 GB. |
| Disk | About 20 GB free | Program and models about 12 GB, plus roughly 20 MB per recorded hour. |
| Internet | For installing only | The app itself works offline. |
| Accounts | A free [Hugging Face](https://huggingface.co) account | Needed once, to download the speaker-separation model. |

## Quick start

**Windows**

1. Put the folder somewhere short, for example inside `Documents` (the path limit is 100 characters), and avoid folders synced by OneDrive.
2. Double-click **`INSTALL.bat`** and press Enter. It installs only what is missing and takes 20-40 minutes. It asks for your own Hugging Face token (see [Installation](docs/installation.md#hugging-face-token)) and reminds you to turn off Ollama's automatic updates. `INSTALL.bat` starts PowerShell with `-ExecutionPolicy Bypass` for that one script (`scripts\install.ps1`) only; it changes no system setting.
3. When it says **All set**, double-click **`run.bat`** (or the Desktop shortcut). The app opens at http://127.0.0.1:8765 in your browser; keep the black window open while you use it.

**Mac:** install Python 3.13 for Apple Silicon, then run `./setup.sh` (or double-click `Setup.command`) and start the app with `./run.sh`. See [Installation](docs/installation.md#macos); the Mac version has not yet been checked on real hardware, so [docs/mac-smoke-test.md](docs/mac-smoke-test.md) is a short checklist for the first Mac user.

Something wrong? Run `.venv\Scripts\lecture-recorder.exe doctor` (Mac: `.venv/bin/lecture-recorder doctor`); every line says what to do. More in [Troubleshooting](docs/troubleshooting.md).

## Documentation

| | |
|---|---|
| [Installation](docs/installation.md) | Windows and Mac install, Hugging Face token, Ollama, manual setup |
| [Using the app](docs/usage.md) | Recording, importing, processing, Zoom classes, compaction, deleting |
| [Configuration](docs/configuration.md) | Settings layers, files and folders |
| [Privacy and security](docs/privacy-and-security.md) | What stays local and who can reach the app |
| [Troubleshooting](docs/troubleshooting.md) | Common problems |
| [Architecture](docs/architecture.md) | Module map and data flow |
| [Development](docs/development.md) | Setup, tests, scripts, dependencies |
| [Dependencies](docs/dependencies.md) | Every package and download, with its purpose and checksums |
| [Mac smoke test](docs/mac-smoke-test.md) | A 15-minute checklist for a real Mac |

Contributions are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md). Security reports: [SECURITY.md](SECURITY.md). Release notes: [CHANGELOG.md](CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE).
