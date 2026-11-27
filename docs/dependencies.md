# Dependencies

Everything the app installs or downloads, and why. Nothing is fetched at run time: after installation the app works offline.

## Python packages

Installation uses hash-pinned lock files with `pip install --require-hashes --only-binary=:all:`, so every package, including transitive ones, is checked against a SHA-256 hash and no source builds happen.

| Lock file | For |
|---|---|
| `requirements/runtime-win-py313.lock` | Windows (x64, Python 3.13), what `setup.bat` installs |
| `requirements/dev-win-py313.lock` | Windows, runtime plus test tools (`setup.bat dev`) |
| `requirements/runtime-mac-py313.lock` | Apple Silicon Mac (macOS 14 or newer, Python 3.13), what `setup.sh` installs |
| `requirements/dev-mac-py313.lock` | Mac, runtime plus test tools |

The `.in` files next to them list the top-level packages the locks are resolved from. The versions below are the ones in those files.

| Package | Version | Purpose |
|---|---|---|
| torch, torchaudio | 2.14.0, 2.11.0 | Runtime for speaker diarization. The Windows locks use the CUDA 12.6 builds from download.pytorch.org (they also supply the CUDA libraries faster-whisper needs); the Mac locks use the PyPI builds |
| torchcodec | 0.16.0 | Required by pyannote (audio is passed in memory, ffmpeg is not used) |
| pyannote.audio | 4.0.7 | Speaker diarization |
| faster-whisper | 1.2.1 | Speech to text (bundles the Silero voice-activity model) |
| ctranslate2 | 4.8.2 | Inference engine for the Whisper models |
| numpy | 2.5.3 | Audio and vector math |
| fastapi | 0.141.1 | Local web server (kept below 0.142, which adds automatic OpenTelemetry export) |
| uvicorn | 0.53.0 | Runs the web server |
| jinja2 | 3.1.6 | HTML templates |
| python-dotenv | 1.2.3 | Reads `.env` |
| python-multipart | 0.0.32 | Audio file uploads when importing |
| sounddevice | 0.5.6 | Microphone capture |
| soundfile | 0.14.0 | WAV and Opus reading and writing |
| pywin32 | 312 | Windows only: sleep and lid events, keep-awake |
| ollama | 0.6.2 | Client for the local language model |
| requests | 2.33 or newer | Security floor for a transitive dependency |

Test and measurement tools (dev locks only): pytest 9.1.1, jiwer 4.0.0 (word error rate), pip-audit 2.10.1 (vulnerability scan), psutil 7.2.2 (resource measurement) and httpx 0.28.1 (test client).

Network lockdown: the app sets `PYANNOTE_METRICS_ENABLED=0`, `HF_HUB_OFFLINE=1` (after the model download), `HF_HUB_DISABLE_TELEMETRY=1` and `HF_HUB_DISABLE_UPDATE_CHECK=1`, clears any `OTEL_*` exporter settings and turns off ONNX Runtime telemetry.

## Ollama (summaries and concept search)

Ollama v0.35.0 is the version the app is tested with. `doctor` warns when another version is running. Turn Ollama's automatic updates off.

| File | Used by | SHA-256 |
|---|---|---|
| `OllamaSetup.exe` v0.35.0 (github.com/ollama/ollama/releases) | `INSTALL.bat` / `scripts/install.ps1` on Windows, which checks the hash before running it | `92e45c3518daf7d4ea60480896ee7bdb1bed71381593cfc9f2d7c1ceb941efde` |
| `Ollama-darwin.zip` v0.35.0 | `setup.sh` on Mac, which checks the hash before unpacking and installs for the current user only | `3458972dfca5b3f8a4297a8d7ff503da3e180ec9452ba8de82c69c5ba3aacdd9` |

Models pulled with `ollama pull` (Ollama checks each layer's SHA-256):

| Model | Purpose | Size | Model layer SHA-256 |
|---|---|---|---|
| `qwen3:8b` | Summaries and exam notes | 5.2 GB | `a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f` |
| `qwen3-embedding:0.6b` | Concept search | 0.64 GB | `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439` |

## Speech and speaker models (Hugging Face)

`scripts/download_models.py` downloads these into `models/`, pinned to a commit, and checks each file's SHA-256 (the values are in the script).

| Model | Revision | Purpose | License |
|---|---|---|---|
| `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf` | Primary speech-to-text model | MIT |
| `distil-whisper/distil-large-v3.5-ct2` | `9793ccc07920e0f830e1dba0343efcdf0ef8c903` | Optional faster fallback | MIT |
| `pyannote/speaker-diarization-community-1` | `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee` | Speaker separation. Gated: needs a free Hugging Face account and a Read token (see the README) | CC-BY-4.0 |

## Licenses

All direct dependencies use permissive licenses (MIT, BSD, Apache-2.0, PSF). The torch Windows wheels bundle NVIDIA CUDA libraries under NVIDIA's own redistribution terms. The full per-package license list is in each package's metadata (`pip show <name>`).
