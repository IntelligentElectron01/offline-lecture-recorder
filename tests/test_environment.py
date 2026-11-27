"""Environment, lockfile integrity, git ignores, offline lockdown, GPU stack."""
import importlib.metadata as md
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements" / "dev-win-py313.lock"


LOCKS = [ROOT / "requirements" / "runtime-win-py313.lock", LOCK]


def test_installed_versions_match_lock():
    lock = ROOT / "requirements" / ("dev-mac-py313.lock" if sys.platform == "darwin" else "dev-win-py313.lock")
    pins = re.findall(r"^([A-Za-z0-9_.\-]+)(?:==(\S+)| @ (\S+))", lock.read_text(encoding="utf-8"), re.M)
    assert len(pins) >= 100
    mismatches = []
    for name, version, url in pins:
        if url:  # direct-URL pins (torch/torchaudio): version comes from the wheel filename
            version = unquote(url.rsplit("/", 1)[-1]).split("-")[1]
        installed = md.version(name)
        if installed != version:
            mismatches.append((name, version, installed))
    assert mismatches == []
    cuda = "+cu126" if sys.platform == "win32" else ""  # the Mac locks use the PyPI builds
    assert md.version("torch") == "2.14.0" + cuda and md.version("torchaudio") == "2.11.0" + cuda


@pytest.mark.parametrize("lock", LOCKS, ids=lambda p: p.name)
def test_every_lock_entry_has_sha256(lock):
    text = lock.read_text(encoding="utf-8")
    entries = re.findall(r"^\S.* \\\n    --hash=sha256:[0-9a-f]{64}$", text, re.M)
    specs = [l for l in text.splitlines() if l and not l.startswith(("#", " "))]
    assert len(specs) > 100 and len(entries) == len(specs)


@pytest.mark.network
def test_pip_audit_clean():
    """Queries PyPI/OSV with package names+versions only. torch/torchaudio aren't on PyPI (vetted via OSV)."""
    r = subprocess.run(
        [sys.executable, "-m", "pip_audit", "--local", "--progress-spinner", "off"],
        capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "No known vulnerabilities found" in r.stdout + r.stderr


@pytest.mark.parametrize(
    "path",
    [
        "data/Class/sessions/2026-09-30_10-00-00/audio.wav",
        "data/app.db",
        "tests/fixtures/private/clip.m4a",
        "tests/fixtures/private/fixture_truth.json",
        "tests/fixtures/synthetic/lecture.wav",
        ".env",
        "config.toml",
        "models/faster-whisper-large-v3-turbo/model.bin",
        ".venv/Scripts/python.exe",
        ".cache/pip/x",
        "anywhere/recording.opus",
    ],
)
def test_private_and_large_paths_are_gitignored(path):
    r = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
    assert r.returncode == 0, f"{path} is NOT ignored"


@pytest.mark.parametrize("path", [".env.example", "config.example.toml", "tests/fixtures/README.md"])
def test_templates_are_tracked(path):
    r = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
    assert r.returncode == 1, f"{path} should be committed"


def test_pyproject_declares_python_launcher_and_locked_backend():
    import tomllib

    py = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["requires-python"] == ">=3.13,<3.14"
    assert py["project"]["scripts"]["lecture-recorder"] == "lecture_recorder.cli:main"
    assert py["build-system"]["build-backend"] == "setuptools.build_meta"
    for lock in ("runtime-win-py313.lock", "dev-win-py313.lock"):
        assert re.search(r"^setuptools==\S+ \\\n    --hash=sha256:", (ROOT / "requirements" / lock).read_text(), re.M)


def test_no_machine_specific_paths_in_shipped_files():
    files = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True
    ).stdout.split()
    user_dir = str(Path.home())
    offenders = []
    for f in files:
        p = ROOT / f
        if p.suffix in {".py", ".bat", ".ps1", ".toml", ".in", ".lock", ".example", ".md", ".txt"} or p.name.startswith(".env"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            if user_dir in text or user_dir.replace("\\", "/") in text:
                offenders.append(f)
    assert offenders == []


def test_launcher_runs():
    from lecture_recorder.cli import main

    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0


def test_lockdown_overrides_hostile_env_in_fresh_process():
    """Importing lecture_recorder first must leave the libraries offline/telemetry-off even if the
    shell says otherwise. Run in a clean subprocess so earlier imports can't mask a failure."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HF_", "PYANNOTE_", "OTEL_"))}
    env.update(PYANNOTE_METRICS_ENABLED="1", HF_HUB_OFFLINE="0", OTEL_EXPORTER_OTLP_ENDPOINT="http://example.invalid:4318")
    code = (
        "import lecture_recorder, os\n"
        "import huggingface_hub.constants as c\n"
        "from pyannote.audio.telemetry.metrics import is_metrics_enabled\n"
        "assert c.HF_HUB_OFFLINE, 'hf not offline'\n"
        "assert c.HF_HUB_DISABLE_TELEMETRY, 'hf telemetry on'\n"
        "assert not is_metrics_enabled(), 'pyannote metrics on'\n"
        "assert 'OTEL_EXPORTER_OTLP_ENDPOINT' not in os.environ, 'otel endpoint kept'\n"
        "print('LOCKDOWN_OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert "LOCKDOWN_OK" in r.stdout, r.stdout + r.stderr


@pytest.mark.skipif(not (ROOT / ".venv" / "pyvenv.cfg").exists() or Path(sys.prefix).resolve() != (ROOT / ".venv").resolve(),
                    reason="needs to run from this folder's .venv (setup.bat makes it; CI does too)")
def test_app_home_is_folder_containing_venv(monkeypatch):
    from lecture_recorder.runtime_env import app_home

    monkeypatch.delenv("LECTURE_RECORDER_HOME", raising=False)
    monkeypatch.chdir(Path.home())  # someone opening a new terminal
    assert app_home() == Path(sys.prefix).parent.resolve() == ROOT.resolve()


SAFE_LOAD_PROBE = """
import lecture_recorder, os, pickle
from pathlib import Path
assert os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] == "1"
from pyannote.audio import Pipeline
try:
    Pipeline.from_pretrained("models/pyannote-community-1")   # no allowlist: must be refused
    print("UNSAFE_LOAD_ALLOWED")
except pickle.UnpicklingError as e:
    print("REFUSED_WITHOUT_ALLOWLIST" if "Unsupported global" in str(e) else f"OTHER {e}")
from lecture_recorder.safe_load import load_diarization_pipeline
p = load_diarization_pipeline(Path("models/pyannote-community-1"))
print("SAFE_LOAD_OK", type(p).__name__)
"""


@pytest.mark.models
def test_pyannote_loads_only_with_safe_unpickler():
    """Proves weights_only is enforced (an un-allowlisted load is refused) and that the
    3-class allowlist in safe_load.py is sufficient."""
    r = subprocess.run([sys.executable, "-c", SAFE_LOAD_PROBE], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert "REFUSED_WITHOUT_ALLOWLIST" in r.stdout, r.stdout + r.stderr[-2000:]
    assert "SAFE_LOAD_OK SpeakerDiarization" in r.stdout, r.stdout + r.stderr[-2000:]


@pytest.mark.gpu
def test_gpu_stack():
    import torch
    import torchaudio
    import ctranslate2

    assert torch.cuda.is_available()
    y = torchaudio.functional.resample(torch.randn(1, 16000, device="cuda"), 16000, 8000)
    assert y.shape[-1] == 8000
    assert ctranslate2.get_cuda_device_count() >= 1
    assert "int8_float16" in ctranslate2.get_supported_compute_types("cuda")
