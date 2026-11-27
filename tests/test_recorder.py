"""Crash-safe recorder, recovery, and safeguards.

Unit tests drive the safeguard state machine with simulated events. Process tests run the real
recorder process on a synthetic, real-time signal in a temporary app home.
"""
import hashlib
import os
import subprocess
import sys
import time
import wave
from datetime import datetime

import numpy as np
import pytest

from lecture_recorder.config import load_config
from lecture_recorder.recorder.control import RecorderError, active_recording, recover_sessions, start_recording
from lecture_recorder.recorder.pcm import PcmWriter, finalize_session
from lecture_recorder.recorder.platform import ProcessLock, is_locked
from lecture_recorder.recorder.safeguards import Safeguards, State
from lecture_recorder.recorder.session import SESSION_NAME_RE, new_session_dir, read_json, write_json_atomic

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows is the tested target")


def wav_seconds(path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def sg(**kw) -> Safeguards:
    return Safeguards(**{"max_length_s": 7200, "silence_pause_s": 600, "silence_dbfs": -50, **kw})


# ---------------------------------------------------------------- safeguards (simulated events)
@pytest.mark.parametrize("event,reason", [("suspend", "sleep"), ("lid_closed", "lid_closed")])
def test_sleep_or_lid_pauses_and_stays_paused_after_wake(event, reason):
    s = sg()
    assert s.on_power(event).kind == "pause" and s.state is State.PAUSED and s.reason == reason
    assert s.on_power("resume_from_sleep") is None and s.on_power("lid_open") is None
    assert s.state is State.PAUSED
    assert s.on_audio(0.1, -20) is None and s.recorded_s == 0  # nothing recorded while paused
    assert s.on_user("resume").kind == "resume" and s.state is State.RECORDING


def test_clock_gap_counts_as_sleep():
    s = sg()
    assert s.on_tick(1000.0) is None and s.on_tick(1000.1) is None
    a = s.on_tick(1000.1 + 3600)
    assert a.kind == "pause" and a.reason == "sleep_detected_by_clock_gap"


def test_pause_on_sleep_can_be_disabled():
    s = sg(pause_on_sleep=False)
    assert s.on_power("suspend") is None and s.state is State.RECORDING


def test_ten_minutes_of_silence_pauses_and_speech_resets_the_timer():
    s = sg()
    for _ in range(5999):  # 599.9 s of silence
        assert s.on_audio(0.1, -70) is None
    assert s.on_audio(0.1, -20) is None and s.silent_s == 0  # one spoken block resets
    for _ in range(5999):
        assert s.on_audio(0.1, -70) is None
    a = s.on_audio(0.1, -70)
    assert a.kind == "pause" and a.reason == "silence" and s.state is State.PAUSED
    s.on_user("resume")
    assert s.silent_s == 0 and s.state is State.RECORDING


def test_two_hours_stops_and_cannot_resume_past_limit():
    s = sg()
    assert s.max_frames_allowed(16000) == 7200 * 16000
    for _ in range(71999):
        assert s.on_audio(0.1, -20) is None
    a = s.on_audio(0.1, -20)
    assert a.kind == "stop" and a.reason == "max_length" and s.state is State.STOPPED
    assert s.on_user("resume") is None and s.on_power("suspend") is None

    s2 = sg(max_length_s=10)
    s2.on_audio(10, -20)
    assert s2.state is State.STOPPED


def test_manual_controls_and_device_loss():
    s = sg()
    assert s.on_user("resume") is None  # already recording
    assert s.on_user("pause").reason == "user" and s.on_user("pause") is None
    assert s.on_user("resume").kind == "resume"
    assert s.on_device_lost().reason == "device_lost" and s.state is State.PAUSED
    assert s.on_user("stop").kind == "stop" and s.on_user("resume") is None


# ---------------------------------------------------------------- files: names, PCM -> WAV
def test_session_default_name_is_start_datetime(tmp_path):
    when = datetime(2026, 9, 30, 10, 5, 7)
    a = new_session_dir(tmp_path, "bio-101", when)
    b = new_session_dir(tmp_path, "bio-101", when)
    assert a.name == "2026-09-30_10-05-07" and b.name == "2026-09-30_10-05-07-2"
    assert a.parent == tmp_path / "classes" / "bio-101" / "sessions"
    assert SESSION_NAME_RE.match(a.name) and SESSION_NAME_RE.match(b.name)
    with pytest.raises(ValueError):
        new_session_dir(tmp_path, "../escape")


def test_finalize_is_lossless_and_idempotent(tmp_path):
    write_json_atomic(tmp_path / "meta.json", {"sample_rate": 16000, "channels": 1})
    audio = (np.random.default_rng(1).normal(0, 3000, 16000 * 3)).astype(np.int16).tobytes()
    w = PcmWriter(tmp_path / "audio.pcm")
    w.write(audio)
    w.close()
    with open(tmp_path / "audio.pcm", "ab") as f:
        f.write(b"\x01")  # a torn half-sample, as a kill mid-write could leave
    wav = finalize_session(tmp_path, recovered=True)
    assert not (tmp_path / "audio.pcm").exists()
    with wave.open(str(wav), "rb") as r:
        assert r.getframerate() == 16000 and r.getnchannels() == 1 and r.getsampwidth() == 2
        assert hashlib.sha256(r.readframes(r.getnframes())).digest() == hashlib.sha256(audio).digest()
    meta = read_json(tmp_path / "meta.json")
    assert meta["recorded_s"] == 3.0 and meta["recovered"] and meta["stop_reason"] == "recovered_after_crash"
    assert finalize_session(tmp_path) == wav  # second run is a no-op


def test_finalize_keeps_pcm_if_verification_fails(tmp_path, monkeypatch):
    import lecture_recorder.recorder.pcm as pcm

    write_json_atomic(tmp_path / "meta.json", {"sample_rate": 16000, "channels": 1})
    (tmp_path / "audio.pcm").write_bytes(b"\x00\x01" * 1600)
    monkeypatch.setattr(pcm, "_sha256_file", lambda *a: "different")
    with pytest.raises(RuntimeError):
        finalize_session(tmp_path)
    assert (tmp_path / "audio.pcm").stat().st_size == 3200 and not (tmp_path / "audio.wav").exists()


def test_finalize_refuses_without_sample_rate(tmp_path):
    (tmp_path / "audio.pcm").write_bytes(b"\x00\x01" * 1600)
    with pytest.raises(RuntimeError):
        finalize_session(tmp_path)  # no meta.json: guessing the rate could produce wrong-speed audio
    assert (tmp_path / "audio.pcm").exists() and not (tmp_path / "audio.wav").exists()


def test_process_lock_detects_owner(tmp_path):
    lock = ProcessLock(tmp_path / "x.lock")
    assert lock.acquire() and is_locked(tmp_path / "x.lock")
    lock.release()
    assert not is_locked(tmp_path / "x.lock")


# ---------------------------------------------------------------- real recorder process
@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("LECTURE_RECORDER_HOME", str(tmp_path))
    c = load_config(tmp_path)
    from lecture_recorder.library import create_class

    for name in ("Class A", "Class B"):  # recording needs an existing class
        create_class(c.data_dir, name)
    yield c
    # A test that fails mid-recording must not leave its recorder running (it would keep the laptop
    # awake and hold the folder open; seen in a fresh-install run).
    h = active_recording(c.data_dir)
    if h is not None:
        try:
            h.stop(timeout=20)
        except Exception:
            if getattr(h, "proc", None) is not None:
                h.proc.kill()


def system_requires_awake() -> bool:
    """Is any process currently holding ES_SYSTEM_REQUIRED? (CallNtPowerInformation, no admin needed.)"""
    import ctypes

    state = ctypes.c_ulong()
    ctypes.windll.powrprof.CallNtPowerInformation(16, None, 0, ctypes.byref(state), ctypes.sizeof(state))
    return bool(state.value & 0x1)


def test_r1_kill_mid_session_keeps_audio_with_datetime_name(cfg):
    """Record 20 s, `taskkill /F` the recorder, recover, check the audio on disk."""
    h = start_recording("class-a", config=cfg, synthetic={"pattern": "tone"})
    st = h.wait_for(lambda s: s.get("state") == "recording", 10)
    t0 = time.monotonic()
    time.sleep(20)
    t_kill = time.monotonic()
    kill = subprocess.run(["taskkill", "/F", "/PID", str(st["pid"])], capture_output=True, text=True)
    elapsed = (t_kill + time.monotonic()) / 2 - t0  # the kill lands while taskkill runs
    assert kill.returncode == 0, kill.stdout + kill.stderr
    h.proc.wait(10)
    while is_locked(h.session_dir / "recorder.lock"):
        time.sleep(0.05)

    pcm = h.session_dir / "audio.pcm"
    assert pcm.exists() and not (h.session_dir / "audio.wav").exists()
    on_disk_s = pcm.stat().st_size / 2 / 16000
    print(f"\nkilled after {elapsed:.2f} s of recording; audio already on disk before recovery: {on_disk_s:.2f} s")
    assert abs(on_disk_s - elapsed) <= 1.0  # at most ~1 s lost

    assert recover_sessions(cfg.data_dir) == [h.session_dir]
    wav = h.session_dir / "audio.wav"
    seconds = wav_seconds(wav)
    print(f"recovered {wav.relative_to(cfg.home)}: {seconds:.2f} s")
    assert SESSION_NAME_RE.match(h.session_dir.name)
    assert h.session_dir.parent.parent.name == "class-a"
    assert abs(seconds - 20) <= 2
    assert not pcm.exists()
    meta = read_json(h.session_dir / "meta.json")
    assert meta["recovered"] and meta["stop_reason"] == "recovered_after_crash"


def test_pause_resume_bookmark_stop_and_keep_awake(cfg):
    others_hold_awake = system_requires_awake()  # e.g. a media player; then OS-level release can't be observed
    h = start_recording("class-a", config=cfg, synthetic={"pattern": "tone"})
    h.wait_for(lambda s: s.get("recorded_s", 0) >= 2, 10)
    assert h.status()["keep_awake"] and system_requires_awake()
    assert h.status()["power_events"] == "ok"  # hidden window + lid notification registered
    with pytest.raises(RecorderError):
        start_recording("class-b", config=cfg, synthetic={"pattern": "tone"})
    assert active_recording(cfg.data_dir).session_dir == h.session_dir

    h.bookmark("key idea")
    h.pause()
    paused = h.wait_for(lambda s: s.get("state") == "paused", 5)
    assert not paused["keep_awake"]
    if not others_hold_awake:
        # Released at the OS level while paused. The flag is system-wide, so another program can take it
        # mid-test (seen once in a fresh-install run): retry briefly, then warn instead of failing;
        # the recorder's own release is asserted above (keep_awake False).
        deadline = time.monotonic() + 3
        while system_requires_awake() and time.monotonic() < deadline:
            time.sleep(0.2)
        if system_requires_awake():
            import warnings

            warnings.warn("another process holds ES_SYSTEM_REQUIRED; OS-level release not observable")
    else:
        print("\nnote: another process holds ES_SYSTEM_REQUIRED; OS-level release not observable")
    time.sleep(2)
    assert abs(h.status()["recorded_s"] - paused["recorded_s"]) < 0.01  # nothing recorded while paused
    h.resume()
    h.wait_for(lambda s: s.get("state") == "recording" and s.get("recorded_s", 0) >= paused["recorded_s"] + 1, 10)
    meta = h.stop()

    assert meta["state"] == "stopped" and meta["stop_reason"] == "user" and not meta["recovered"]
    assert [e["type"] for e in meta["events"]] == ["start", "pause", "resume", "stop"]
    bm = read_json(h.session_dir / "bookmarks.json")
    assert len(bm) == 1 and bm[0]["label"] == "key idea" and 1.5 <= bm[0]["audio_s"] <= paused["recorded_s"]
    assert abs(wav_seconds(h.session_dir / "audio.wav") - meta["recorded_s"]) < 0.01
    assert not (h.session_dir / "audio.pcm").exists() and active_recording(cfg.data_dir) is None
    if not others_hold_awake:
        assert not system_requires_awake()


def test_recording_continues_while_status_file_is_held_open(cfg):
    """Another program (antivirus, indexer, the UI) holding status.json must not stop capture."""
    h = start_recording("class-a", config=cfg, synthetic={"pattern": "tone"})
    h.wait_for(lambda s: s.get("recorded_s", 0) >= 1, 10)
    with open(h.session_dir / "status.json", "rb"):  # blocks os.replace on Windows
        time.sleep(3)
    st = h.wait_for(lambda s: s.get("recorded_s", 0) >= 4, 10)
    assert st["state"] == "recording" and h.is_running()
    meta = h.stop()
    assert meta["stop_reason"] == "user" and wav_seconds(h.session_dir / "audio.wav") >= 4


@pytest.mark.parametrize("event,reason", [("suspend", "sleep"), ("lid_closed", "lid_closed")])
def test_process_pauses_on_simulated_power_event_and_stays_paused(cfg, event, reason):
    h = start_recording("class-a", config=cfg, synthetic={"pattern": "tone"}, allow_simulated_events=True)
    h.wait_for(lambda s: s.get("recorded_s", 0) >= 1, 10)
    h.send("simulate_power", event=event)
    st = h.wait_for(lambda s: s.get("state") == "paused", 5)
    assert st["reason"] == reason and not st["keep_awake"]
    h.send("simulate_power", event="resume_from_sleep" if event == "suspend" else "lid_open")
    time.sleep(1.5)
    assert h.status()["state"] == "paused"
    h.stop()


def test_process_pauses_on_clock_gap(cfg):
    h = start_recording("class-a", config=cfg, synthetic={"pattern": "tone"}, allow_simulated_events=True)
    h.wait_for(lambda s: s.get("recorded_s", 0) >= 1, 10)
    h.send("simulate_clock_gap", seconds=120)
    assert h.wait_for(lambda s: s.get("state") == "paused", 5)["reason"] == "sleep_detected_by_clock_gap"
    h.stop()


def test_simulated_events_ignored_without_test_flag(cfg):
    h = start_recording("class-a", config=cfg, synthetic={"pattern": "tone"})
    h.wait_for(lambda s: s.get("recorded_s", 0) >= 1, 10)
    h.send("simulate_power", event="suspend")
    time.sleep(1)
    assert h.status()["state"] == "recording"
    h.stop()


def test_process_auto_pauses_on_silence(cfg):
    h = start_recording("class-a", config=cfg, synthetic={"pattern": "silence"}, silence_pause_minutes=3 / 60)
    st = h.wait_for(lambda s: s.get("state") == "paused", 10)
    assert st["reason"] == "silence" and 2.9 <= st["recorded_s"] <= 3.2
    meta = h.stop()
    assert 2.9 <= wav_seconds(h.session_dir / "audio.wav") <= 3.2 and meta["stop_reason"] == "user"


def test_process_auto_stops_at_max_length(cfg):
    h = start_recording("class-a", config=cfg, synthetic={"pattern": "tone"}, max_length_minutes=3 / 60)
    h.proc.wait(20)
    meta = h.meta()
    assert meta["stop_reason"] == "max_length" and meta["state"] == "stopped"
    assert wav_seconds(h.session_dir / "audio.wav") == 3.0  # cut to the exact frame


def test_process_pauses_when_device_stops_delivering(cfg):
    h = start_recording("class-a", config=cfg, synthetic={"pattern": "tone", "stall_after_s": 1})
    st = h.wait_for(lambda s: s.get("state") == "paused", 15)
    assert st["reason"] == "device_lost"
    h.stop()


def test_cli_recover_and_devices(cfg):
    from lecture_recorder.cli import main

    assert main(["recover"]) == 0
    assert main(["devices"]) == 0


@pytest.mark.mic
def test_default_microphone_records(cfg):
    h = start_recording("class-a", config=cfg)
    h.wait_for(lambda s: s.get("recorded_s", 0) >= 2, 15)
    meta = h.stop()
    print(f"\nmic: {meta['source']['device_name']} @ {meta['sample_rate']} Hz, {meta['recorded_s']:.1f} s")
    assert wav_seconds(h.session_dir / "audio.wav") >= 2


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing locks")
def test_recorder_starts_while_a_scanner_briefly_holds_meta_json(cfg, monkeypatch):
    """Root cause of a rare 'Recorder failed to start (exit 1)': the child read meta.json while another
    process (antivirus/indexer) held the fresh file open without sharing, gave up after 0.4 s and
    crashed. Reproduce by holding it exclusively for 1.5 s right after it is written."""
    import ctypes
    import threading
    from ctypes import wintypes

    from lecture_recorder.recorder import control, session

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    held = threading.Event()

    def hold(path):
        h = k32.CreateFileW(str(path), 0x80000000, 0, None, 3, 0, None)  # GENERIC_READ, share mode 0
        assert h != wintypes.HANDLE(-1).value
        held.set()
        time.sleep(1.5)
        k32.CloseHandle(h)

    def write_then_hold(path, obj, retries=50):
        session.write_json_atomic(path, obj, retries)
        if path.name == "meta.json" and not held.is_set():
            threading.Thread(target=hold, args=(path,), daemon=True).start()
            assert held.wait(5)

    monkeypatch.setattr(control, "write_json_atomic", write_then_hold)
    h = control.start_recording("class-a", config=cfg, synthetic={"pattern": "tone"})
    assert h.status()["state"] == "recording"
    h.stop()
