"""The app must work on other people's computers. Hardware, Windows versions and folders are faked; nothing
touches the GPU, the network or the real data folder."""
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

import lecture_recorder  # noqa: F401  (lockdown)
from lecture_recorder import autoconfig as ac
from lecture_recorder.config import load_config
from lecture_recorder.host import host
from test_portability import doctor_out, profile

ROOT = Path(__file__).resolve().parents[1]


def stt(result):
    return result["settings"]["transcription"]


# ------------------------------------------------------------------ NVIDIA GPUs of every size
@pytest.mark.parametrize("vram_mb,device,summary_note", [
    (2048, "cpu", False),    # too small for the speech model: CPU
    (3072, "cpu", False),
    (4096, "cuda", True),    # 4 GB card: speech on the GPU, the summary model is split with the CPU by Ollama
    (6144, "cuda", True),
    (8192, "cuda", False),
    (12288, "cuda", False),
    (24576, "cuda", False),
])
def test_gpu_memory_tiers(vram_mb, device, summary_note):
    r = ac.pick(profile(gpu={"name": "Example GPU", "vram_mb": vram_mb}, cuda_usable=True))
    assert stt(r)["device"] == device
    assert stt(r)["compute_type"] == ("int8_float16" if device == "cuda" else "int8")
    assert stt(r)["model"] == "large-v3-turbo" and stt(r)["batch_size"] == 0
    assert any("Ollama runs part of it on the CPU" in n for n in r["notes"]) is summary_note


def test_a_gpu_whose_driver_does_not_work_means_cpu_with_advice():
    r = ac.pick(profile(gpu={"name": "Example GPU", "vram_mb": 8192}, cuda_usable=False))
    assert stt(r)["device"] == "cpu" and any("update the NVIDIA driver" in n for n in r["notes"])


@pytest.mark.parametrize("ram_gb,cpus", [(8, 4), (8, 2), (4, 2), (64, 64), (None, None)])
def test_small_and_unknown_machines_get_sane_settings(ram_gb, cpus):
    r = ac.pick(profile(ram_gb=ram_gb, logical_cpus=cpus or 2))
    assert 1 <= stt(r)["cpu_threads"] <= 4 and stt(r)["device"] == "cpu"
    assert any("RAM" in n for n in r["notes"]) is (ram_gb is not None and ram_gb < ac.MIN_RAM_GB)


# ------------------------------------------------------------------ what detection does when probes fail
def run_detect(monkeypatch, tmp_path, smi=None, which="nvidia-smi.exe", cuda=False):
    monkeypatch.setattr(ac.shutil, "which", lambda name: which if name == "nvidia-smi" and which else None)
    monkeypatch.setattr(ac, "_cuda_usable", lambda: cuda)
    monkeypatch.setattr(ac, "_ollama", lambda host: {"installed": False, "reachable": False, "models": [], "version": None})
    if smi is not None:
        monkeypatch.setattr(ac.subprocess, "run", smi)
    return ac.detect(tmp_path, tmp_path / "models", "http://127.0.0.1:11434")


def smi_result(stdout="", returncode=0):
    return lambda *a, **k: subprocess.CompletedProcess(a, returncode, stdout=stdout, stderr="")


def test_no_nvidia_tools_at_all_amd_intel_or_no_driver(monkeypatch, tmp_path):
    found = run_detect(monkeypatch, tmp_path, which=None)  # no nvidia-smi on the PATH
    assert found["gpu"] is None and found["cuda_usable"] is False
    assert stt(ac.pick(found))["device"] == "cpu"


@pytest.mark.parametrize("smi", [
    smi_result("", 9),                                           # driver not loaded
    smi_result("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver", 0),
    smi_result("garbage, not a number", 0),
    smi_result("Example GPU, [N/A]", 0),
    lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("nvidia-smi", 10)),
    lambda *a, **k: (_ for _ in ()).throw(OSError("access denied")),
])
def test_broken_nvidia_smi_never_stops_detection(monkeypatch, tmp_path, smi):
    found = run_detect(monkeypatch, tmp_path, smi=smi)
    assert found["gpu"] is None and stt(ac.pick(found))["device"] == "cpu"


def test_nvidia_smi_output_in_other_formats_is_read(monkeypatch, tmp_path):
    found = run_detect(monkeypatch, tmp_path, smi=smi_result("NVIDIA GeForce GTX 1650, 4096\r\n"), cuda=True)
    assert found["gpu"] == {"name": "NVIDIA GeForce GTX 1650", "vram_mb": 4096} and stt(ac.pick(found))["device"] == "cuda"
    found = run_detect(monkeypatch, tmp_path, smi=smi_result("Name, With Comma, 8188.0\n"), cuda=True)
    assert found["gpu"] == {"name": "Name, With Comma", "vram_mb": 8188}


def test_unknown_ram_battery_and_disk_do_not_stop_detection(monkeypatch, tmp_path):
    monkeypatch.setattr(ac, "_ram_gb", lambda: None)
    monkeypatch.setattr(ac, "_has_battery", lambda: None)
    monkeypatch.setattr(ac.shutil, "disk_usage", lambda p: (_ for _ in ()).throw(OSError("no such drive")))
    found = run_detect(monkeypatch, tmp_path, which=None)
    assert found["ram_gb"] is None and found["battery"] is None and found["disk_free_gb"] is None
    r = ac.pick(found)
    assert stt(r)["device"] == "cpu" and not any("RAM" in n or "disk" in n for n in r["notes"])


# ------------------------------------------------------------------ doctor says what to do next
def test_doctor_on_a_small_cpu_only_machine_with_a_tight_disk(monkeypatch, capsys, short_home):
    code, out = doctor_out(monkeypatch, capsys, short_home, ram_gb=6, disk_free_gb=9)
    assert code == 0  # warnings, not failures: the app still works
    assert "[--] 6 GB of RAM" in out and "[--] Only 9 GB free" in out
    assert "AMD and Intel graphics are not used" in out
    assert f"Next: {host().start_text}" in out


def test_doctor_failure_ends_with_the_next_step(monkeypatch, capsys, tmp_path):
    code, out = doctor_out(monkeypatch, capsys, tmp_path,
                           models={"large-v3-turbo": False, "distil-large-v3.5": False, "pyannote": True})
    assert code == 1 and ac.DOWNLOAD_HINTS["large-v3-turbo"] in out
    assert "Each [!!] line above says what to do" in out and host().installer_name in out


@pytest.mark.parametrize("build,name,ok", [(19041, "Windows 10", True), (19045, "Windows 10", True),
                                           (22631, "Windows 11", True), (18363, "Windows 10", False)])
@pytest.mark.skipif(sys.platform != "win32", reason="reads the Windows build")
def test_doctor_reports_the_windows_version(monkeypatch, capsys, short_home, build, name, ok):
    real = sys.getwindowsversion()
    fake = type("V", (), {"build": build, "major": real.major, "minor": real.minor})()
    monkeypatch.setattr(sys, "getwindowsversion", lambda: fake)
    code, out = doctor_out(monkeypatch, capsys, short_home)
    assert f"Windows build {build} ({name})" in out
    assert (code == 0) is ok and ("too old" in out) is (not ok)


@pytest.mark.skipif(sys.platform != "win32", reason="the OneDrive check is part of the Windows checks")
@pytest.mark.parametrize("folder", ["OneDrive", "onedrive-school", "plain"])
def test_doctor_warns_about_onedrive_without_failing(monkeypatch, capsys, tmp_path_factory, folder):
    home = tmp_path_factory.mktemp(folder)  # short, so the 100-character path limit isn't what fails here
    code, out = doctor_out(monkeypatch, capsys, home)
    assert code == 0
    assert ("[--] This folder is not inside OneDrive" in out) is ("onedrive" in folder.lower())


# ------------------------------------------------------------------ folders and locales
ODD_NAMES = ["with spaces", "caf\u00e9 \u00fc", "\u65e5\u672c\u8a9e", "\u0440\u0443\u0441\u0441\u043a\u0438\u0439", "100% [test] (1)"]


@pytest.mark.parametrize("name", ODD_NAMES)
def test_an_install_folder_with_odd_characters_works_end_to_end(name, tmp_path, monkeypatch, capsys):
    from lecture_recorder import cli
    from lecture_recorder.web import auth

    home = tmp_path / name
    home.mkdir()
    monkeypatch.setenv("LECTURE_RECORDER_HOME", str(home))
    monkeypatch.setattr(ac, "detect", lambda *a: profile())
    assert cli.configure(dry_run=False) == 0
    assert "[transcription]" in (home / "config.auto.toml").read_text(encoding="utf-8")
    cfg = load_config(home)
    assert cfg.data_dir == (home / "data").resolve() and cfg.models_dir == (home / "models").resolve()
    key = auth.write_login_key(home, 8765)
    assert auth.read_login_key(home, 8765) == key
    capsys.readouterr()
    monkeypatch.setattr(sys.modules["lecture_recorder.web.app"], "port_state", lambda *a: "free")
    assert cli.doctor() in (0, 1)  # prints without an encoding error, whatever the console's code page
    assert name in capsys.readouterr().out or str(Path.home()) in str(home)


@pytest.mark.parametrize("encoding", ["cp1252", "cp932", "cp437", "utf-8"])
def test_output_survives_every_console_code_page(encoding, tmp_path):
    """doctor and configure run with a legacy code page (non-English Windows): nothing may raise
    UnicodeEncodeError. The child's stdout is forced to that encoding with 'replace' like a console would."""
    code = ("import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=sys.argv[1], errors='strict'); "
            "from lecture_recorder import cli; sys.exit(cli.main(['configure', '--dry-run']))")
    r = subprocess.run([sys.executable, "-c", code, encoding], cwd=ROOT, capture_output=True, text=True, timeout=120,
                       env={**__import__("os").environ, "LECTURE_RECORDER_HOME": str(tmp_path), "PYTHONIOENCODING": ""})
    assert r.returncode == 0 and "UnicodeEncodeError" not in r.stderr, r.stderr[-400:]


def test_configure_json_for_the_settings_page_is_pure_ascii(monkeypatch, capsys, tmp_path):
    """The app reads this from a child process in the console's code page, so it must not depend on it."""
    from lecture_recorder import cli

    home = tmp_path / "caf\u00e9"
    home.mkdir()
    monkeypatch.setenv("LECTURE_RECORDER_HOME", str(home))
    monkeypatch.setattr(ac, "detect", lambda *a: profile())
    assert cli.configure(dry_run=True, as_json=True) == 0
    out = capsys.readouterr().out
    out.encode("ascii")
    assert json.loads(out)["settings"]["transcription"]["device"] == "cpu"


@pytest.mark.skipif(sys.platform != "win32", reason="per-app capture is Windows-only")
def test_windows_10_and_11_decide_app_capture_by_build(monkeypatch):
    from lecture_recorder.recorder import winaudio

    monkeypatch.setattr(sys, "platform", "win32")
    for build, ok in ((19045, False), (20348, True), (22000, True), (26100, True)):
        monkeypatch.setattr(sys, "getwindowsversion", lambda b=build: type("V", (), {"build": b})(), raising=False)
        assert (winaudio.supported() is None) is ok, build
        if not ok:
            assert "Windows 11" in winaudio.supported() and str(build) in winaudio.supported()


# ------------------------------------------------------------------ installer text
def test_installer_does_not_send_people_to_a_root_folder_and_asks_for_the_required_steps():
    ps = (ROOT / "scripts" / "install.ps1").read_text(encoding="utf-8")
    assert "C:\\LectureRecorder" not in ps and "LOCALAPPDATA" in ps
    assert "REQUIRED STEP: turn OFF Ollama's automatic updates" in ps and "Ollama-Autoupdate-Step" in ps
    assert "Offer-Blank-Token" in ps and ps.count("Offer-Blank-Token") >= 3  # defined, after download, and when already downloaded
    assert "SecureString" in ps  # the token is typed hidden
    ps.encode("ascii")  # Windows PowerShell 5.1 reads a BOM-less file in the ANSI code page: ASCII only
    assert "HF_TOKEN=" in ps and "Remove it from .env" in ps


def test_readme_has_no_personal_or_workflow_text():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for bad in ("C:\\LectureRecorder", "milestone", "advis" + "or", "the user "):
        assert bad.lower() not in text.lower(), bad
    for needed in ("INSTALL.bat", "Hugging Face token", "LICENSE", "run.bat"):
        assert needed in text, needed
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")
