# Development

## Setup

Windows: `setup.bat dev` installs the hash-pinned dev lock and the package in editable mode into `.venv`. Mac: `./setup.sh dev`.

```
.venv\Scripts\python.exe -m pip check
.venv\Scripts\lecture-recorder.exe doctor
```

## Tests

Run the tests that need no special hardware:

```
.venv\Scripts\python.exe -m pytest -q -m "not gpu and not models and not ollama and not private and not network and not mic and not audio"
```

Markers (see `pyproject.toml`): `gpu` (CUDA GPU), `models` (downloaded models), `ollama` (running Ollama), `private` (optional local recordings, git-ignored), `network` (vulnerability scan), `mic`, `audio` (plays and records test tones), `real_power`, `real_auth` and `mac`. Test files are named after the feature they cover (`test_pipeline.py`, `test_web_app.py`, `test_auth.py`, ...). `tests/conftest.py` makes every test run "plugged in" and signed in unless it opts out.

The synthetic fixture is generated, not committed: `.venv\Scripts\python.exe scripts\dev\make_synthetic_fixture.py`.

## Scripts

| Folder | Contents |
|---|---|
| `scripts/` | `install.ps1` and `download_models.py`, used by the installers |
| `scripts/dev/` | Maintenance: `lock_from_report.py` (hash-pinned locks), `make_package.py` (user zip), `make_synthetic_fixture.py`, `fresh_install_test.ps1` |
| `scripts/demo/` | Demo data on a throwaway folder (synthetic, no GPU) for trying the UI |
| `scripts/bench/` | Measurements: `measure_resources.py` (idle and job budgets), `benchmark_stt.py`, and other `measure_*` scripts |

## Dependencies

Packages are pinned with hashes in `requirements/*.lock` and installed with `--require-hashes --only-binary=:all:`. To re-lock, resolve a report with pip and convert it:

```
pip install --dry-run --ignore-installed --only-binary=:all: --uploaded-prior-to <date> --report .cache\r.json -r requirements\dev.in
python scripts\dev\lock_from_report.py .cache\r.json requirements\dev-win-py313.lock
```

Add a package to the matching `.in` file and to [dependencies](dependencies.md) with its purpose. Only well-known, maintained packages are accepted.

## Pull requests

See [CONTRIBUTING.md](../CONTRIBUTING.md).
