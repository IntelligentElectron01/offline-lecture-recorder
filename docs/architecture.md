# Architecture

Lecture Recorder is a local Python application: a small web server (FastAPI on `127.0.0.1`) with a plain HTML/CSS/JS page, a crash-safe recorder process, and a serial queue that runs one heavy job at a time. Plain files in the data folder are the source of truth; the search index is rebuilt from them.

## Data flow

```
microphone / app audio / imported file
        |
   recorder (separate process)     raw PCM + fsync -> audio.wav, meta.json
        |
   queue (data/queue, one worker)  one GPU/CPU job at a time; waits while recording or on battery
        |
   pipeline: transcribe (faster-whisper) -> free VRAM -> diarize (pyannote) -> merge
        |                                                       transcript.json / .txt / .md
   summary: local LLM via Ollama (map-reduce over chunks)       summary.md
        |
   search index: exact (SQLite) + concept (embeddings)          data/app.db, embeddings in the lecture folder
        |
   compaction: verified Opus copy                               audio.opus
```

Each lecture folder holds `meta.json`, the audio, the transcript files, `summary.md` and its metadata. A class folder holds `class.json`, its lectures, and optional units and exam scopes (`units.json`, `exams/`).

## Modules (`src/lecture_recorder`)

| Module | Role |
|---|---|
| `cli.py` | The `lecture-recorder` command: serve, record, process, summarize, search, doctor, configure, compact, delete/restore, import-zoom and more |
| `web/` | The local server (`app.py`), sign-in cookie (`auth.py`) and static UI |
| `recorder/` | Crash-safe recording: sources (microphone, one app's audio on Windows), the safeguards state machine, PCM files rebuilt into WAV, process control and crash recovery |
| `pipeline/` | Transcription, diarization, merge, outputs, the serial job queue, GPU lock, progress files and the audio health check |
| `summary/` | Summaries and exam material from a local LLM (Ollama on loopback only) |
| `search/` | Exact and concept search, embeddings, and unit-boundary suggestions |
| `host/` | The one place that knows the operating system (Windows and macOS) |
| `library.py`, `units.py`, `trash.py`, `move.py`, `naming.py`, `dates.py`, `duplicates.py`, `autosort.py`, `retention.py`, `importer.py`, `zoom.py` | Classes and lectures on disk, units, Recently deleted, moving, titles and dates, duplicate and class suggestions, Opus compaction, importing |
| `config.py`, `settings.py`, `autoconfig.py` | Configuration layers and per-computer automatic settings |
| `runtime_env.py`, `safe_load.py`, `machine.py` | Network lockdown (imported first), safe model loading, and power/recording state that gates heavy jobs |

## Design rules

- **Local only.** The only network use at run time is Ollama on `127.0.0.1`; telemetry is switched off in the libraries and Hugging Face is forced offline.
- **Files first.** Nothing important lives only in the database. Deleting or overwriting user data always needs an explicit action.
- **One heavy job at a time.** Transcription, diarization, summaries and embeddings share one lock and never run while recording or on battery.
- **Idle means idle.** No busy loops; the UI stops polling while its tab is hidden.
