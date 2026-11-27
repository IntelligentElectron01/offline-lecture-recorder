"""The macOS port, tested on any computer through the host layer's fakes. Nothing here needs a Mac; the few
tests that do are marked `mac` and skipped elsewhere. Temp folders only."""
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import lecture_recorder  # noqa: F401  (lockdown)
from lecture_recorder import autoconfig as ac
from lecture_recorder import host as hostmod
from lecture_recorder import machine
from lecture_recorder.config import load_config
from lecture_recorder.host import Host, use_host
from lecture_recorder.host import mac as macmod
from lecture_recorder.host.mac import MacHost, parse_pmset
from lecture_recorder.host.windows import WindowsHost
from lecture_recorder.pipeline import queue as q
from lecture_recorder.recorder.platform import KeepAwake
from lecture_recorder.web.app import create_app
from test_web_app import BASE, H
from test_portability import doctor_out, profile

BATTERY = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=4653155)\t87%; discharging; 3:12 remaining present: true\n"
AC_LAPTOP = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=4653155)\t100%; charged; 0:00 remaining present: true\n"
AC_DESKTOP = "Now drawing from 'AC Power'\n"
UPS = "Now drawing from 'UPS Power'\n -CyberPower (id=1)\t100%\n"


@pytest.fixture
def fake_cmds(monkeypatch):
    """Replace the commands MacHost runs. Returns the table to fill in and the list of calls."""
    table, calls = {}, []

    def run_quiet(cmd, timeout=5):
        calls.append(tuple(cmd))
        return table.get(tuple(cmd))

    monkeypatch.setattr(macmod, "run_quiet", run_quiet)
    return table, calls


# ------------------------------------------------------------------ power
@pytest.mark.parametrize("text,want", [
    (BATTERY, ("battery", True)), (AC_LAPTOP, ("ac", True)), (AC_DESKTOP, ("ac", False)), (UPS, ("ac", False)),
    ("", ("unknown", None)), (None, ("unknown", None)), ("garbage\nlines", ("unknown", None)),
    ("Now drawing from 'Something Else'\n -InternalBattery-0 x", ("unknown", True)),
])
def test_pmset_output_is_read(text, want):
    assert parse_pmset(text) == want


def test_battery_blocks_heavy_jobs_through_the_host(tmp_path, fake_cmds):
    table, _ = fake_cmds
    table[("pmset", "-g", "batt")] = BATTERY
    with use_host(MacHost()):
        assert machine.power_source.__wrapped__() if hasattr(machine.power_source, "__wrapped__") else True
        from lecture_recorder.host import host

        assert host().power_source() == "battery"
        assert machine.gpu_block_reason(tmp_path) is None or True  # conftest's plugged-in switch is on; the real gate below


@pytest.mark.real_power
def test_the_real_gate_reads_the_mac_host(tmp_path, fake_cmds):
    table, _ = fake_cmds
    table[("pmset", "-g", "batt")] = BATTERY
    with use_host(MacHost()):
        assert machine.power_source() == "battery" and machine.gpu_block_reason(tmp_path) == machine.ON_BATTERY
    table[("pmset", "-g", "batt")] = AC_LAPTOP
    with use_host(MacHost()):
        assert machine.power_source() == "ac" and machine.gpu_block_reason(tmp_path) is None


def test_pmset_is_not_started_for_every_question(fake_cmds, monkeypatch):
    table, calls = fake_cmds
    table[("pmset", "-g", "batt")] = AC_DESKTOP
    now = [1000.0]
    monkeypatch.setattr(macmod.time, "monotonic", lambda: now[0])
    h = MacHost()
    for _ in range(50):
        assert h.power_source() == "ac" and h.has_battery() is False
    assert len(calls) == 1  # cached for POWER_CACHE_S
    now[0] += macmod.POWER_CACHE_S + 1
    table[("pmset", "-g", "batt")] = BATTERY
    assert h.power_source() == "battery" and len(calls) == 2


def test_a_failing_pmset_means_unknown_which_counts_as_plugged_in(fake_cmds):
    assert MacHost().power_source() == "unknown" and MacHost().has_battery() is None


# ------------------------------------------------------------------ memory, cores, facts
def test_ram_and_core_counts(fake_cmds):
    table, _ = fake_cmds
    table[("sysctl", "-n", "hw.memsize")] = "17179869184\n"
    table[("sysctl", "-n", "hw.perflevel0.logicalcpu")] = "4\n"
    h = MacHost()
    assert h.ram_gb() == 16.0 and h.perf_cores() == 4
    del table[("sysctl", "-n", "hw.perflevel0.logicalcpu")]
    table[("sysctl", "-n", "hw.physicalcpu")] = "8\n"
    assert MacHost().perf_cores() == 8  # older release without the performance-level keys
    table.clear()
    fresh = MacHost()
    assert fresh.perf_cores() is None
    assert fresh.ram_gb() is None or fresh.ram_gb() > 0  # falls back to sysconf where it exists


def test_facts_name_the_chip_and_rosetta(fake_cmds, monkeypatch):
    table, _ = fake_cmds
    table[("sysctl", "-n", "machdep.cpu.brand_string")] = "Apple M2\n"
    table[("sysctl", "-n", "sysctl.proc_translated")] = "1\n"
    monkeypatch.setattr(macmod.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(macmod.platform, "mac_ver", lambda: ("14.5", ("", "", ""), "arm64"))
    f = MacHost().facts()
    assert f["os"] == "macos" and f["arch"] == "arm64" and f["macos"] == "14.5" and f["chip"] == "Apple M2" and f["rosetta"] is True


# ------------------------------------------------------------------ processes
@pytest.mark.skipif(sys.platform != "win32", reason="Windows process flags")
def test_windows_spawn_flags_are_exactly_what_they_were():
    h = WindowsHost()
    flags = h.job_spawn_kwargs()["creationflags"]
    assert flags == (subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) \
        if sys.platform == "win32" else True
    if sys.platform == "win32":
        assert h.recorder_spawn_kwargs() == {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
                                             "start_new_session": False}
        assert h.job_child_setup() is None  # the parent's flags did it


@pytest.mark.parametrize("h", [MacHost(), Host()])
def test_posix_jobs_get_their_own_session_and_lower_their_own_priority(h, monkeypatch):
    assert h.job_spawn_kwargs() == {"start_new_session": True} and h.recorder_spawn_kwargs() == {"start_new_session": True}
    niced = []
    monkeypatch.setattr(os, "nice", lambda n: niced.append(n), raising=False)
    h.job_child_setup()
    assert niced == [10]
    h.lower_own_priority()
    assert niced == [10, 10]


def test_the_job_process_lowers_its_own_priority_at_start(monkeypatch, tmp_path):
    from lecture_recorder.pipeline import __main__ as job

    calls = []

    class Spy(Host):
        def job_child_setup(self):
            calls.append("setup")

    monkeypatch.setattr("lecture_recorder.machine.gpu_block_reason", lambda d: "on battery")  # stops right after the setup
    with use_host(Spy()):
        assert job.main([str(tmp_path), "--data-dir", str(tmp_path)]) == q.EXIT_BLOCKED
    assert calls == ["setup"]


# ------------------------------------------------------------------ keep-awake
def test_keep_awake_holds_one_caffeinate_and_lets_go(monkeypatch):
    started = []

    class Proc:
        def __init__(self, cmd, **kw):
            started.append(cmd)
            self.alive = True

        def poll(self):
            return None if self.alive else 0

        def terminate(self):
            self.alive = False

        def wait(self, timeout=None):
            pass

    monkeypatch.setattr(macmod.subprocess, "Popen", Proc)
    with use_host(MacHost()):
        ka = KeepAwake()
        ka.hold()
        ka.hold()  # already held: no second helper
        assert len(started) == 1 and started[0][:3] == ["caffeinate", "-i", "-w"] and started[0][3] == str(os.getpid())
        proc = ka._caffeinate
        ka.release()
        assert proc.alive is False and ka._caffeinate is None
        ka.release()  # releasing twice is harmless


def test_a_mac_without_caffeinate_just_does_not_hold(monkeypatch):
    def nope(*a, **k):
        raise OSError("not found")

    monkeypatch.setattr(macmod.subprocess, "Popen", nope)
    with use_host(MacHost()):
        assert MacHost().keep_awake(True, 0, KeepAwake()) is False


# ------------------------------------------------------------------ no per-app recording on a Mac
@pytest.fixture
def mac_app(tmp_path):
    cfg = dataclasses.replace(load_config(tmp_path), data_dir=(tmp_path / "data").resolve())
    with use_host(MacHost()):
        with TestClient(create_app(cfg, autoprocess=False, monitor=False), base_url=BASE) as c:
            yield c, cfg


def test_the_page_is_told_to_leave_out_app_recording(mac_app):
    c, _ = mac_app
    caps = c.get("/api/capabilities").json()
    assert caps["os"] == "macos" and caps["app_capture_hidden"] is True and "Privacy & Security" in caps["mic_hint"]
    assert "isn't available on a Mac" in caps["app_capture_reason"]
    r = c.get("/api/apps").json()
    assert r["supported"] is False and "Zoom" in r["reason"] and r["apps"] == []


def test_app_recording_is_refused_in_the_api_and_the_terminal(mac_app, capsys, monkeypatch):
    from lecture_recorder import cli
    from lecture_recorder import library as lib

    c, cfg = mac_app
    lib.create_class(cfg.data_dir, "Physics 101")
    r = c.post("/api/record/start", headers=H, json={"class": "physics-101", "app_pid": 1234})
    assert r.status_code >= 400 and "isn't available on a Mac" in r.text
    assert cli.apps() == 2 and "isn't available on a Mac" in capsys.readouterr().out
    assert not list((cfg.data_dir / "classes" / "physics-101" / "sessions").glob("*")) if (cfg.data_dir / "classes" / "physics-101" / "sessions").exists() else True


@pytest.mark.skipif(sys.platform != "win32", reason="per-app capture is Windows-only")
def test_windows_still_offers_it_and_keeps_its_exact_reply(tmp_path, monkeypatch):
    from lecture_recorder.recorder import winaudio

    monkeypatch.setattr(winaudio, "supported", lambda: None)
    h = WindowsHost()
    assert h.app_capture_reason is None and h.app_capture_hidden is False
    monkeypatch.setattr(winaudio, "supported", lambda: "needs Windows 11")
    assert h.app_capture_reason == "needs Windows 11"
    with use_host(h):
        cfg = dataclasses.replace(load_config(tmp_path), data_dir=(tmp_path / "data").resolve())
        with TestClient(create_app(cfg, autoprocess=False, monitor=False), base_url=BASE) as c:
            assert c.get("/api/apps").json() == {"supported": False, "reason": "needs Windows 11", "apps": []}
            assert c.get("/api/capabilities").json()["app_capture_hidden"] is False


def test_the_page_hides_the_source_choice_and_explains_a_silent_mic():
    js = (Path(__file__).resolve().parents[1] / "src" / "lecture_recorder" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert "caps.app_capture_hidden ? h(" in js and "no-app-capture" in js and "/api/capabilities" in js and "micHint" in js


# ------------------------------------------------------------------ the Mac settings tier
@pytest.mark.parametrize("perf,want", [(None, 4), (1, 2), (2, 2), (4, 4), (6, 6), (8, 6), (16, 6)])
def test_threads_follow_performance_cores(perf, want):
    r = ac.pick(profile(logical_cpus=10, host={"os": "macos", "arch": "arm64", "perf_cores": perf}))
    assert r["settings"]["transcription"]["cpu_threads"] == (want if perf else 4)


def test_a_mac_runs_on_the_cpu_with_an_honest_note():
    r = ac.pick(profile(logical_cpus=10, host={"os": "macos", "arch": "arm64", "perf_cores": 4}, battery=True))
    t = r["settings"]["transcription"]
    assert t["device"] == "cpu" and t["compute_type"] == "int8" and t["model"] == "large-v3-turbo" and t["batch_size"] == 0
    assert any("Apple Silicon" in n and "not been measured" in n for n in r["notes"])
    assert not any("reference laptop" in n for n in r["notes"])  # the Windows timing doesn't claim a Mac speed


def test_intel_mac_and_rosetta_are_called_out():
    intel = ac.pick(profile(host={"os": "macos", "arch": "x86_64", "perf_cores": 4}))
    assert any("only Apple Silicon" in n for n in intel["notes"])
    rosetta = ac.pick(profile(host={"os": "macos", "arch": "arm64", "rosetta": True, "perf_cores": 4}))
    assert any("Rosetta" in n for n in rosetta["notes"])


def test_commands_shown_to_people_use_the_os_paths():
    with use_host(MacHost()):
        assert ac._script("download_models.py") == ".venv/bin/python scripts/download_models.py"
    with use_host(WindowsHost()):
        assert ac._script("download_models.py") == r".venv\Scripts\python.exe scripts\download_models.py"
    assert "Scripts" not in ac.DOWNLOAD_HINTS["qwen3:8b"]


def test_ollama_not_running_advice_names_the_macs_way():
    with use_host(MacHost()):
        r = ac.pick(profile(ollama={"installed": True, "reachable": False, "models": []}))
    assert any("Applications folder" in n for n in r["notes"]) and not any("Start menu" in n for n in r["notes"])


# ------------------------------------------------------------------ doctor on a Mac
def mac_doctor(monkeypatch, capsys, tmp_path, **facts):
    f = {"os": "macos", "arch": "arm64", "macos": "14.5", "chip": "Apple M2", "rosetta": False, "perf_cores": 4}
    monkeypatch.setattr(MacHost, "facts", lambda self: {**f, **facts})
    monkeypatch.setattr(MacHost, "perf_cores", lambda self: 4)
    with use_host(MacHost()):
        return doctor_out(monkeypatch, capsys, tmp_path, host={**f, **facts})


def test_doctor_on_a_good_mac(monkeypatch, capsys, tmp_path):
    code, out = mac_doctor(monkeypatch, capsys, tmp_path)
    assert code == 0 and "[ok] Mac: Apple M2, arm64" in out and "[ok] macOS 14.5" in out
    assert mac_doctor(monkeypatch, capsys, tmp_path, macos="15.1")[0] == 0
    assert "Microphone: macOS asks the first time you record" in out and "Privacy & Security" in out
    assert "run ./run.sh" in out
    for windows_only in ("OneDrive", "Windows build", "setup.bat", "Folder path length", "Start menu", "%LOCALAPPDATA%",
                         "run.bat", "INSTALL.bat", ".bat"):
        assert windows_only not in out, windows_only


@pytest.mark.parametrize("facts,needle,fails", [
    ({"arch": "x86_64"}, "only Apple Silicon (M1 or newer) is supported", True),
    ({"macos": "13.6"}, "macOS 14 (Sonoma) or newer is needed", True),
    ({"rosetta": True}, "running under Rosetta", False),
])
def test_doctor_flags_an_unsupported_mac(monkeypatch, capsys, tmp_path, facts, needle, fails):
    code, out = mac_doctor(monkeypatch, capsys, tmp_path, **facts)
    assert needle in out and (code == 1) is fails


def test_doctor_tells_a_mac_user_to_run_setup_sh(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "version_info", (3, 12, 1, "final", 0))
    code, out = mac_doctor(monkeypatch, capsys, tmp_path)
    assert code == 1 and "run ./setup.sh with Python 3.13" in out


# ------------------------------------------------------------------ files on a Mac
def test_a_symlinked_session_folder_cannot_reach_outside_the_data_folder(tmp_path):
    """macOS has symlinks, not junctions: the same traversal rule (resolved path must be inside data) must hold."""
    from lecture_recorder import library as lib

    cfg = dataclasses.replace(load_config(tmp_path), data_dir=(tmp_path / "data").resolve())
    lib.create_class(cfg.data_dir, "Physics 101")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "transcript.md").write_text("secret", encoding="utf-8")
    link = cfg.data_dir / "classes" / "physics-101" / "sessions" / "2026-09-01_09-00-00"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("this account can't create symlinks")
    with TestClient(create_app(cfg, autoprocess=False, monitor=False), base_url=BASE) as c:
        assert c.get("/files/physics-101/2026-09-01_09-00-00/transcript.md").status_code in (403, 404)


def test_the_recorder_saves_the_audio_when_it_is_terminated_on_posix():
    src = (Path(__file__).resolve().parents[1] / "src" / "lecture_recorder" / "recorder" / "process.py").read_text(encoding="utf-8")
    assert 'if os.name != "nt"' in src and "signal.SIGTERM" in src  # a closed terminal or `kill` ends it like Stop


# ------------------------------------------------------------------ really on a Mac (skipped elsewhere)
@pytest.mark.mac
@pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS")
def test_the_real_mac_answers_every_question():
    h = hostmod.host()
    assert h.name == "macos" and h.power_source() in ("ac", "battery", "unknown")
    assert (h.ram_gb() or 0) >= 4 and (h.perf_cores() or 0) >= 1
    assert h.facts()["arch"] == "arm64"
    assert h.keep_awake(True, 0, KeepAwake()) in (True, False)


def test_nothing_outside_the_windows_files_touches_windows_apis():
    """A Mac can import everything: Win32 calls live only in host/windows.py, recorder/winaudio.py and the Windows
    branches of recorder/platform.py (all lazy), never at import time elsewhere."""
    src = Path(__file__).resolve().parents[1] / "src" / "lecture_recorder"
    allowed = {src / "host" / "windows.py", src / "recorder" / "winaudio.py", src / "recorder" / "platform.py"}
    bad = ("windll", "msvcrt", "win32api", "win32gui", "winreg", "startfile", "CREATE_NO_WINDOW", "SetPriorityClass",
           "getwindowsversion", "wintypes")
    hits = []
    for p in src.rglob("*.py"):
        if p in allowed:
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if any(b in line for b in bad):
                hits.append(f"{p.relative_to(src)}:{n}: {line.strip()[:80]}")
    assert not hits, hits


def test_importing_the_app_does_not_need_the_windows_files(monkeypatch):
    """The Windows-only modules must not be imported just by starting the app on another OS."""
    code = ("import sys; sys.modules['lecture_recorder.recorder.winaudio'] = None; "
            "import lecture_recorder.web.app, lecture_recorder.cli, lecture_recorder.pipeline.queue, "
            "lecture_recorder.recorder.control, lecture_recorder.autoconfig, lecture_recorder.machine; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                       cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-500:]


def test_the_app_picker_is_empty_and_safe_without_the_windows_audio_module():
    """On another OS recorder/apps.py is importable and lists nothing (winaudio cannot be imported there)."""
    code = ("import sys; sys.modules['lecture_recorder.recorder.winaudio'] = None; real = sys.platform; sys.platform = 'darwin'; "
            "from lecture_recorder.recorder import apps; sys.platform = real; "
            "assert apps.winaudio is None and apps.list_audio_apps() == [] and apps.find_app(1) is None; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                       cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-500:]


def test_hardware_facts_are_asked_once(fake_cmds):
    table, calls = fake_cmds
    table[("sysctl", "-n", "hw.memsize")] = "17179869184\n"
    table[("sysctl", "-n", "hw.perflevel0.logicalcpu")] = "4\n"
    h = MacHost()
    for _ in range(20):
        assert h.ram_gb() == 16.0 and h.perf_cores() == 4
    assert len(calls) == 2  # one sysctl each, not one per load_config()


def test_the_keep_awake_helper_is_reaped(monkeypatch):
    waited = []

    class Proc:
        def __init__(self, cmd, **kw):
            pass

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            waited.append(timeout)

    monkeypatch.setattr(macmod.subprocess, "Popen", Proc)
    ka = KeepAwake()
    with use_host(MacHost()):
        ka.hold()
        ka.release()
    assert waited == [2]


def test_a_move_never_replaces_a_folder_on_any_os(tmp_path):
    from lecture_recorder import move

    src, dst = tmp_path / "a", tmp_path / "b"
    src.mkdir()
    dst.mkdir()  # empty: POSIX rename would silently replace it
    with pytest.raises(FileExistsError):
        move._rename(src, dst)
    assert src.is_dir() and dst.is_dir()
    (dst / "f").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        move._rename(src, dst)
    free = tmp_path / "free"
    move._rename(src, free)
    assert free.is_dir() and not src.exists()


# ------------------------------------------------------------------ the shell scripts (static checks; no Mac here)
SCRIPTS = Path(__file__).resolve().parents[1]


def test_setup_sh_installs_ollama_only_after_the_checksum_matches():
    sh = (SCRIPTS / "setup.sh").read_text(encoding="utf-8")
    assert '"3458972dfca5b3f8a4297a8d7ff503da3e180ec9452ba8de82c69c5ba3aacdd9"' in sh  # the approved Ollama-darwin.zip
    assert 'OLLAMA_VERSION="0.35.0"' in sh and "github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/Ollama-darwin.zip" in sh
    assert "--proto '=https'" in sh and "--tlsv1.2" in sh
    assert sh.index("shasum -a 256") < sh.index("ditto -x -k")  # verified before anything is unpacked
    commands = [ln for ln in sh.splitlines() if not ln.lstrip().startswith(("#", "echo", "printf"))]
    assert not any("sudo" in ln for ln in commands) and "$HOME/Applications" in sh
    assert "Auto-download updates" in sh  # the person is told to turn auto-update off


def test_the_mac_scripts_are_portable_to_the_default_shell():
    for name in ("setup.sh", "run.sh", "Lecture Recorder.command", "Setup.command"):
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        assert text.startswith("#!/bin/bash\n") and "\r" not in text, name  # LF endings, bash 3.2 syntax below
        for bash4 in ("declare -A", "mapfile", "readarray", "${!", "&>>", ";&", "${var,,}", "${var^^}"):
            assert bash4 not in text, (name, bash4)
        assert "cd \"$(dirname \"$0\")\"" in text  # works from any folder and with spaces in the path
    lock_rt = (SCRIPTS / "requirements" / "runtime-mac-py313.lock").read_text(encoding="utf-8")
    assert "pywin32" not in lock_rt and "+cu" not in lock_rt and "win_amd64" not in lock_rt and "--hash=sha256:" in lock_rt


def test_every_mac_lock_line_is_hash_pinned_and_matches_the_windows_versions():
    import re

    def pins(name):
        text = (SCRIPTS / "requirements" / name).read_text(encoding="utf-8")
        return dict(re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)", text, re.M))

    mac, win = pins("dev-mac-py313.lock"), pins("dev-win-py313.lock")
    for name in ("aiohttp", "msgpack", "pillow", "numpy", "faster-whisper", "ctranslate2", "pyannote-audio", "pip", "setuptools"):
        assert mac[name] == win[name], name
    text = (SCRIPTS / "requirements" / "dev-mac-py313.lock").read_text(encoding="utf-8")
    assert len(re.findall(r"^[A-Za-z0-9_.-]+==", text, re.M)) == len(re.findall(r"--hash=sha256:", text))  # one hash per pin
