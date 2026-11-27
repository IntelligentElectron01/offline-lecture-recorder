"""Portability: each item has a fix and a test (hardware mocked)."""
import dataclasses
import json
import shutil
import socket
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

import lecture_recorder  # noqa: F401  (lockdown)
from lecture_recorder import autoconfig as ac
from lecture_recorder.config import load_config

ROOT = Path(__file__).resolve().parents[1]
ODD = "dir with spaces é ü"  # spaces and non-ASCII in every path below


def profile(**kw) -> dict:
    p = {"gpu": None, "cuda_usable": False, "logical_cpus": 8, "ram_gb": 32, "disk_free_gb": 200, "battery": False,
         "ollama": {"installed": True, "reachable": True, "models": ["qwen3-embedding:0.6b", "qwen3:8b"],
                    "version": ac.APPROVED_OLLAMA_VERSION},
         "models": {"large-v3-turbo": True, "distil-large-v3.5": True, "pyannote": True}}
    return {**p, **kw}


def doctor_out(monkeypatch, capsys, home: Path, **prof) -> tuple[int, str]:
    from lecture_recorder import cli
    from lecture_recorder.web import app as web

    monkeypatch.setenv("LECTURE_RECORDER_HOME", str(home))
    monkeypatch.setattr(ac, "detect", lambda *a: profile(**prof))
    monkeypatch.setattr(web, "port_state", lambda *a: prof.get("_port", "free"))
    code = cli.doctor()
    return code, capsys.readouterr().out


# ------------------------------------------------------------------ setup.bat / run.bat
def test_setup_does_not_abort_when_configure_fails_and_repairs_moved_launchers():
    bat = (ROOT / "setup.bat").read_text(encoding="utf-8")
    conf = next(l for l in bat.splitlines() if "lecture_recorder.cli configure" in l and not l.lstrip().upper().startswith(("REM", "ECHO")))
    assert "exit /b" not in conf and "WARNING" in bat  # warns and continues with safe defaults
    assert "--force-reinstall" in bat  # launchers always rewritten for this folder (moved/copied)
    assert 'python.exe -c "" >nul 2>nul ||' in bat  # a venv whose base Python is gone is re-linked
    assert "Python 3.13 is required" in bat


def test_run_bat_survives_a_moved_folder_and_explains_missing_setup_or_python():
    bat = (ROOT / "run.bat").read_text(encoding="utf-8")
    assert "python.exe -m lecture_recorder.cli serve" in bat and "lecture-recorder.exe" not in bat
    assert "Run setup.bat first" in bat and "Python 3.13" in bat


def test_nothing_needs_admin_rights_or_writes_outside_the_folder():
    code = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in
                     list((ROOT / "src").rglob("*.py")) + [ROOT / "setup.bat", ROOT / "run.bat"])
    for bad in ("HKEY_LOCAL_MACHINE", "HKLM:", "runas", "Program Files", "setx ", "powercfg", "netsh"):
        assert bad not in code, bad


# ------------------------------------------------------------------ doctor reports each problem clearly
def test_doctor_all_good(monkeypatch, capsys, tmp_path):
    code, out = doctor_out(monkeypatch, capsys, tmp_path)
    assert code == 0 and "All checks passed." in out and "Port 8765: free" in out


def test_doctor_no_gpu_ollama_missing_models_missing_no_token(monkeypatch, capsys, tmp_path):
    (tmp_path / ".env").write_text("HF_TOKEN=\n", encoding="utf-8")
    code, out = doctor_out(monkeypatch, capsys, tmp_path, ollama={"installed": False, "reachable": False, "models": []},
                           models={"large-v3-turbo": False, "distil-large-v3.5": False, "pyannote": False})
    assert code == 1
    assert ac.DOWNLOAD_HINTS["large-v3-turbo"] in out and ac.DOWNLOAD_HINTS["pyannote"] in out
    assert "[!!] HF_TOKEN in .env" in out and ac.DOWNLOAD_HINTS["ollama"] in out
    assert "[--] Model distil-large-v3.5" in out  # optional fallback: not a failure by itself
    assert "No NVIDIA GPU: CPU path" in out and "AMD and Intel" in out  # from the (mocked) detection, not from importing torch


def test_doctor_ollama_not_running_and_port_busy(monkeypatch, capsys, tmp_path):
    code, out = doctor_out(monkeypatch, capsys, tmp_path, ollama={"installed": True, "reachable": False, "models": []},
                           _port="busy")
    assert code == 1 and "installed but not running" in out and "used by another program" in out


@pytest.mark.parametrize("version", ["0.40.11", None])
def test_doctor_warns_without_failing_on_an_unapproved_ollama_version(monkeypatch, capsys, short_home, version):
    models = ["qwen3-embedding:0.6b", "qwen3:8b"]
    code, out = doctor_out(monkeypatch, capsys, short_home,
                           ollama={"installed": True, "reachable": True, "models": models, "version": version})
    assert code == 0 and "All checks passed." in out
    assert f"[--] Ollama {version or 'version unknown'} is not the approved version {ac.APPROVED_OLLAMA_VERSION}" in out
    _, ok = doctor_out(monkeypatch, capsys, short_home)
    assert f"[ok] Ollama {ac.APPROVED_OLLAMA_VERSION} (the approved version)" in ok


def test_ollama_probe_reads_the_version_from_loopback_only(monkeypatch):
    seen = []

    class Resp:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            import io

            return io.BytesIO(json.dumps(self.body).encode())

        def __exit__(self, *a):
            return False

    class Opener:
        def open(self, url, timeout):
            seen.append(url)
            return Resp({"models": [{"name": "qwen3:8b"}]} if url.endswith("/api/tags") else {"version": "0.40.11"})

    monkeypatch.setattr(ac.urllib.request, "build_opener", lambda *a: Opener())
    out = ac._ollama("http://127.0.0.1:11434")
    assert out["reachable"] and out["version"] == "0.40.11" and out["models"] == ["qwen3:8b"]
    assert seen == ["http://127.0.0.1:11434/api/tags", "http://127.0.0.1:11434/api/version"]

    class NoVersion(Opener):  # an Ollama without /api/version: still reachable, models kept
        def open(self, url, timeout):
            if url.endswith("/api/version"):
                raise OSError("404")
            return super().open(url, timeout)

    monkeypatch.setattr(ac.urllib.request, "build_opener", lambda *a: NoVersion())
    out = ac._ollama("http://127.0.0.1:11434")
    assert out["reachable"] and out["models"] == ["qwen3:8b"] and out["version"] is None


def test_doctor_explains_a_broken_config_toml(monkeypatch, capsys, tmp_path):
    (tmp_path / "config.toml").write_text("[server\nport = ", encoding="utf-8")
    code, out = doctor_out(monkeypatch, capsys, tmp_path)
    assert code == 1 and "config.toml could not be read" in out


# ------------------------------------------------------------------ port already in use
def test_port_state_and_serve_refuses_a_busy_port(tmp_path):
    from lecture_recorder.web import app as web

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        assert web.port_state("127.0.0.1", port) == "busy"  # someone else's listener, not this app
        cfg = load_config(tmp_path)
        cfg = dataclasses.replace(cfg, data_dir=tmp_path / "data", server=dataclasses.replace(cfg.server, port=port))
        assert web.serve(cfg, open_browser=False) == 2  # clear message, no traceback
    assert web.port_state("127.0.0.1", port) == "free"


def test_serve_opens_the_running_copy_instead_of_failing(tmp_path, monkeypatch):
    from lecture_recorder.web import app as web

    opened = []
    monkeypatch.setattr(web, "port_state", lambda *a: "this app")
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    cfg = dataclasses.replace(load_config(tmp_path), data_dir=tmp_path / "data")
    from lecture_recorder.web.auth import write_login_key

    key = write_login_key(tmp_path, cfg.server.port)  # what the running copy stored: opened with it
    assert web.serve(cfg, open_browser=True) == 0 and opened == [f"http://127.0.0.1:8765/?k={key}"]


# ------------------------------------------------------------------ models missing at job time
def test_missing_models_fail_with_the_approved_command(tmp_path):
    from lecture_recorder.pipeline.diarize import diarize
    from lecture_recorder.pipeline.transcribe import choose

    with pytest.raises(FileNotFoundError, match="download_models.py whisper"):
        choose(tmp_path, "large-v3-turbo", "cpu", "int8")
    with pytest.raises(FileNotFoundError, match="download_models.py pyannote"):
        diarize(np.zeros(16000, np.float32), tmp_path, device="cpu")


# ------------------------------------------------------------------ paths: spaces, non-ASCII, relative/absolute, moved
def write_wav(path: Path, seconds: float = 1.0) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((0.2 * np.sin(np.arange(int(16000 * seconds)) / 3) * 32767).astype(np.int16).tobytes())


def test_home_with_spaces_and_accents_end_to_end_without_gpu(tmp_path, monkeypatch):
    from lecture_recorder import library as lib
    from lecture_recorder.pipeline.audio import load_audio, recorded_at
    from lecture_recorder.retention import compress_session

    home = tmp_path / ODD
    home.mkdir()
    monkeypatch.setattr(ac, "detect", lambda *a: profile())
    ac.configure(home)
    cfg = load_config(home)
    assert cfg.data_dir == home / "data" and cfg.transcription.device == "cpu"
    lib.create_class(cfg.data_dir, "Física 101")
    d = next((cfg.data_dir / "classes").iterdir()) / "sessions" / "2026-10-02_09-00-00"
    d.mkdir(parents=True)
    write_wav(d / "audio.wav", 3)
    assert load_audio(d / "audio.wav").shape[0] == 48000 and recorded_at(d / "audio.wav") is None
    assert compress_session(d)["verified"]  # PyAV/libsndfile with non-ASCII paths


def test_relative_and_absolute_data_dirs(tmp_path):
    (tmp_path / "config.toml").write_text('[paths]\ndata_dir = "my data"\n', encoding="utf-8")
    assert load_config(tmp_path).data_dir == tmp_path / "my data"
    absolute = (tmp_path / "elsewhere é").resolve()
    (tmp_path / "config.toml").write_text(f"[paths]\ndata_dir = {json.dumps(str(absolute))}\n", encoding="utf-8")
    assert load_config(tmp_path).data_dir == absolute


def test_cli_data_dir_relative_to_where_you_run_it(tmp_path, monkeypatch):
    from lecture_recorder import cli

    seen = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LECTURE_RECORDER_HOME", str(tmp_path))
    (tmp_path / ac.AUTO_FILE).write_text("", encoding="utf-8")  # no first-start detection in this test
    monkeypatch.setattr("lecture_recorder.web.app.serve", lambda cfg, open_browser: seen.setdefault("d", cfg.data_dir) and 0)
    cli.main(["serve", "--no-browser", "--data-dir", "demo data"])
    assert seen["d"] == (tmp_path / "demo data").resolve()


def test_a_copied_folder_uses_its_own_data_and_config(tmp_path):
    a, b = tmp_path / "original", tmp_path / ODD / "copy"
    a.mkdir()
    (a / "config.toml").write_text('[processing]\nmode = "auto"\n', encoding="utf-8")
    (a / "data").mkdir()
    shutil.copytree(a, b)
    cfg = load_config(b)
    assert cfg.home == b.resolve() and cfg.data_dir == b.resolve() / "data" and cfg.processing.mode == "auto"


@pytest.mark.skipif(sys.platform != "win32" or not (ROOT / ".venv" / "pyvenv.cfg").exists(),
                    reason="Windows launcher behavior; needs the project .venv (setup.bat makes it)")
def test_moved_venv_python_still_runs_the_app_module():
    """run.bat relies on .venv\\Scripts\\python.exe -m (no absolute path baked in, unlike the .exe launchers)."""
    cfg = (ROOT / ".venv" / "pyvenv.cfg").read_text(encoding="utf-8")
    home_line = next(l for l in cfg.splitlines() if l.startswith("home"))
    assert str(ROOT) not in home_line  # points at the base Python, not this folder
    r = subprocess.run([str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "lecture_recorder.cli", "--version"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "lecture-recorder" in r.stdout


# ------------------------------------------------------------------ no NVIDIA GPU (CUDA hidden)
def test_configure_with_cuda_hidden_picks_the_cpu(tmp_path):
    import os

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "LECTURE_RECORDER_HOME": str(tmp_path)}
    r = subprocess.run([sys.executable, "-m", "lecture_recorder.cli", "configure", "--dry-run", "--json"],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    picks = json.loads(r.stdout.strip().splitlines()[-1])["settings"]["transcription"]
    assert picks["device"] == "cpu" and picks["compute_type"] == "int8" and picks["model"] == "large-v3-turbo"


def test_port_state_refuses_non_loopback_and_handles_ipv6():
    from lecture_recorder.web import app as web

    with pytest.raises(ValueError):
        web.port_state("192.168.1.5", 8765)
    assert web.port_state("::1", 0) == "free"  # IPv6 loopback binds with the right family


def test_doctor_reports_a_non_loopback_host_without_probing_it(monkeypatch, capsys, tmp_path):
    (tmp_path / "config.toml").write_text('[server]\nhost = "0.0.0.0"\n', encoding="utf-8")
    from lecture_recorder import cli

    monkeypatch.setenv("LECTURE_RECORDER_HOME", str(tmp_path))
    monkeypatch.setattr(ac, "detect", lambda *a: profile())
    assert cli.doctor() == 1 and "[server] host" in capsys.readouterr().out
