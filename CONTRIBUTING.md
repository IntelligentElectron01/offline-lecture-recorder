# Contributing

Thanks for helping. Bug reports, fixes and documentation improvements are welcome.

## Before you open an issue

- Run `.venv\Scripts\lecture-recorder.exe doctor` (Mac: `.venv/bin/lecture-recorder doctor`) and include its output.
- **Never attach recordings, transcripts, summaries, tokens or `.env`.** Remove your user name and folder names from logs and messages before posting. Use a short synthetic or public-domain clip if you need audio to reproduce a problem.

## Development setup

1. Install Python 3.13 (64-bit; Apple Silicon on a Mac).
2. Windows: `setup.bat dev`. Mac: `./setup.sh dev`. This creates `.venv` from the hash-pinned dev lock and installs the package in editable mode.
3. Run the tests that need no special hardware:

   ```
   .venv\Scripts\python.exe -m pytest -q -m "not gpu and not models and not ollama and not private and not network and not mic and not audio"
   ```

See [docs/development.md](docs/development.md) for markers, scripts and the layout of the code.

## Pull requests

- Keep a change focused, and add or update a test for behaviour you change.
- The tests above must pass. Tests for hardware you don't have are marked and skipped.
- Dependencies are hash-pinned. A new package needs a reason, a maintained upstream and an entry in [docs/dependencies.md](docs/dependencies.md); locks are rebuilt by running `scripts/dev/lock_from_report.py`.
- Keep the privacy rules: nothing leaves the machine, no telemetry, and the app listens on `127.0.0.1` only.
- Never commit recordings, transcripts, models, `.env` or tokens.

Security problems: see [SECURITY.md](SECURITY.md) instead of opening a public issue.
