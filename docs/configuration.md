# Configuration

Settings come from three layers, lowest priority first:

1. Built-in defaults.
2. `config.auto.toml`, written by `lecture-recorder configure`. It picks the speech model, device, compute type and thread counts that suit this computer. Setup and the first start run it; run it again after changing hardware (`configure --dry-run` shows the result without writing).
3. `config.toml`, your own values. Copy [`config.example.toml`](../config.example.toml) to `config.toml` and change only what you need. It always wins over the automatic picks.

The **Processing and storage** and **Settings** areas in the app cover the common choices: processing mode (manual or automatic when plugged in), keep-awake while processing, the naming template for new lectures, deleting originals after a verified Opus copy, emptying Recently deleted after some days, quitting when no page is open, and re-detecting this computer. Changes made there are saved to `config.toml`.

## Files and folders

| Path | Contents |
|---|---|
| `data/` | Everything you record or import: plain audio, text and Markdown, one folder per class and lecture. Back this folder up. |
| `data/.trash/` | Recently deleted lectures and classes (restorable) |
| `data/app.db` | Search index. It is rebuilt from the lecture folders (`lecture-recorder reindex`) |
| `models/` | Downloaded speech and speaker models (set with `[paths] models_dir`) |
| `config.toml`, `config.auto.toml` | Your settings and the automatic picks |
| `.env` | Optional `HF_TOKEN` while downloading the speaker model |

`data/`, `models/`, `.env` and `config.toml` are ignored by git.

## Environment variables

| Variable | Effect |
|---|---|
| `LECTURE_RECORDER_HOME` | Use another folder as the app home (the folder that holds `data/`, `models/` and the config files) |
| `HF_TOKEN` | Hugging Face Read token, only needed by `scripts/download_models.py pyannote` |

## The server

The app listens on `127.0.0.1` only. `[server] host` refuses any other address, so a typo cannot expose your lectures. `[server] port` changes the port (default 8765). `[server] idle_quit_minutes` quits the app after that many minutes with no page open (0 = never).
