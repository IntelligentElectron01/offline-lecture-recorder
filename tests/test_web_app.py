"""Local app, security guards, retention, offline UI.

Everything runs in temp folders with synthetic audio; the real data/ folder is never touched.
"""
import dataclasses
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import lecture_recorder  # noqa: F401  (lockdown)
from lecture_recorder import library as lib
from lecture_recorder.config import ProcessingConfig, load_config
from lecture_recorder.pipeline.outputs import RAW, write_outputs
from lecture_recorder.pipeline.queue import jobs
from lecture_recorder.retention import compress_session
from lecture_recorder.web.app import check_loopback, create_app

ROOT = Path(__file__).resolve().parents[1]
AUTO = ProcessingConfig("auto")
STATIC = ROOT / "src" / "lecture_recorder" / "web" / "static"
BASE = "http://127.0.0.1:8765"
H = {"X-Lecture-Recorder": "1"}
SESSION = "2026-09-30_10-00-00"


@pytest.fixture
def env(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (tmp_path / "secret.txt").write_text("TOP-SECRET", encoding="utf-8")  # outside data/
    cfg = dataclasses.replace(load_config(), data_dir=data, processing=AUTO)  # Stop/import queue (auto mode)
    app = create_app(cfg, autoprocess=False, monitor=False, synthetic_source={"pattern": "tone"})
    with TestClient(app, base_url=BASE) as client:
        yield client, cfg, data


def write_tone_wav(path: Path, seconds: float = 3.0, rate: int = 16000) -> None:
    t = np.arange(int(seconds * rate)) / rate
    x = (0.2 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(x.tobytes())


def fake_processed_session(data: Path, slug: str = "physics-101", name: str = SESSION) -> Path:
    if not (data / "classes" / slug / "class.json").exists():
        lib.create_class(data, "Physics 101")
    d = data / "classes" / slug / "sessions" / name
    d.mkdir(parents=True)
    write_tone_wav(d / "audio.wav")
    doc = {"speakers": [{"id": "Speaker 1", "role": "Main", "speech_s": 2.0},
                        {"id": "Speaker 2", "role": "Audience", "speech_s": 1.0}],
           "segments": [{"id": 0, "start": 0.0, "end": 2.0, "speaker": "Speaker 1", "role": "Main", "text": "Hello class.", "words": []},
                        {"id": 1, "start": 2.0, "end": 3.0, "speaker": "Speaker 1", "role": "Main", "text": "A question?", "words": []}],
           "processing": {"dropped_speech": {"total_s": 1.5, "stretches": [{"start": 1.0, "end": 2.5}]}}}
    write_outputs(d, doc)
    lib.write_summary_placeholder(d)
    return d


# ------------------------------------------------------------------ network guards
def test_refuses_non_loopback_bind():
    assert check_loopback("127.0.0.1") == "127.0.0.1"
    for bad in ("0.0.0.0", "192.168.1.5", "::", "example.com"):
        with pytest.raises(ValueError):
            check_loopback(bad)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_server_listens_on_127_0_0_1_only(tmp_path):
    import psutil
    import urllib.request

    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-m", "lecture_recorder.cli", "serve", "--no-browser", "--port", str(port),
                             "--data-dir", str(tmp_path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ROOT,
                            env={**os.environ, "LECTURE_RECORDER_HOME": str(tmp_path)})  # its login key goes here
    try:
        for _ in range(150):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1) as r:
                    assert json.load(r) == {"ok": True, "app": "lecture-recorder"}
                break
            except OSError:
                time.sleep(0.2)
        else:
            pytest.fail("server did not start")
        # every socket listening on this port, from any process (the venv launcher runs the server as a child)
        # (per process: the system-wide listing needs root on macOS, but a process may list its own and its children's)
        me = psutil.Process(proc.pid)
        sockets = [c for p in [me, *me.children(recursive=True)] for c in p.net_connections(kind="inet")]
        listening = [c for c in sockets if c.status == psutil.CONN_LISTEN and c.laddr.port == port]
        assert listening and all(c.laddr.ip == "127.0.0.1" for c in listening), listening
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def test_serve_refuses_configured_non_loopback_host(tmp_path):
    (tmp_path / "config.toml").write_text('[server]\nhost = "0.0.0.0"\n', encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "lecture_recorder.cli", "serve", "--no-browser"], cwd=ROOT,
                       env={**os.environ, "LECTURE_RECORDER_HOME": str(tmp_path)}, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "local-only" in r.stdout


def test_dns_rebinding_guard_rejects_foreign_host_header(env):
    client, _, _ = env
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/health", headers={"Host": "localhost:8765"}).status_code == 200
    for host in ("evil.example", "attacker.com:8765", "127.0.0.1.evil.example"):
        assert client.get("/api/health", headers={"Host": host}).status_code == 400, host


def test_state_changes_need_app_header_and_same_origin(env):
    client, _, data = env
    assert client.post("/api/classes", json={"name": "A"}).status_code == 403  # no header (e.g. a plain HTML form)
    assert client.post("/api/classes", json={"name": "A"}, headers={**H, "Origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/classes", json={"name": "A"}, headers={**H, "Origin": "null"}).status_code == 403
    assert not (data / "classes").exists()
    assert client.post("/api/classes", json={"name": "A"}, headers={**H, "Origin": BASE}).status_code == 201


TRAVERSALS = [
    "/files/physics-101/{s}/..%2F..%2F..%2F..%2Fsecret.txt",
    "/files/physics-101/{s}/%2e%2e%2f%2e%2e%2fsecret.txt",
    "/files/physics-101/..%2F..%2F..%2F/secret.txt",
    "/files/..%2F..%2F../{s}/transcript.txt",
    "/files/physics-101/{s}/..\\..\\..\\..\\secret.txt",
    "/files/physics-101/{s}/class.json",
    "/files/physics-101/{s}/recorder.log",
    "/files/physics-101/{s}/../../../../secret.txt",
    "/files/physics-101/{s}/C:%5CWindows%5Cwin.ini",
    "/static/..%2F..%2F..%2F..%2Fsecret.txt",
    "/static/../app.py",
    "/static/%2e%2e%2fapp.py",
]


@pytest.mark.parametrize("url", TRAVERSALS)
def test_file_routes_never_leave_data_dir(env, url):
    client, _, data = env
    fake_processed_session(data)
    r = client.get(url.format(s=SESSION))
    assert r.status_code in (400, 404), (url, r.status_code)
    assert b"TOP-SECRET" not in r.content and b"def create_app" not in r.content


def test_file_route_refuses_session_folder_that_resolves_outside_data(env, tmp_path):
    """A junction/symlink planted inside data/ must not expose files outside it."""
    client, _, data = env
    outside = tmp_path / "outside" / "sessions" / SESSION
    outside.mkdir(parents=True)
    (outside / "transcript.txt").write_text("TOP-SECRET", encoding="utf-8")
    link = data / "classes" / "evil"
    link.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(tmp_path / "outside")], capture_output=True)
        if r.returncode != 0:
            pytest.skip("could not create a junction")
    else:
        link.symlink_to(tmp_path / "outside")
    r = client.get(f"/files/evil/{SESSION}/transcript.txt")
    assert r.status_code == 404 and b"TOP-SECRET" not in r.content


# ------------------------------------------------------------------ UI is offline and shows the REC indicator
def test_ui_uses_no_external_assets_or_brand_names():
    for f in STATIC.iterdir():
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"https?://|//cdn|fonts\.googleapis|@import", text), f.name
        assert not re.search("|".join(("anthr" + "opic", "cla" + "ude")), text, re.I), f.name  # no brand names in the UI
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert re.findall(r'(?:src|href)="([^"#]+)"', html) == ["/static/app.css", "/static/app.js"]


def test_index_has_rec_indicator_and_permission_reminder(env):
    client, _, _ = env
    html = client.get("/").text
    assert 'id="recbar"' in html and 'id="rec-label">REC<' in html
    assert "You are responsible for getting permission before recording anyone." in html
    assert "default-src 'self'" in client.get("/").headers["content-security-policy"]
    assert client.get("/static/app.js").status_code == 200


# ------------------------------------------------------------------ classes and recording
def test_r5_session_recorded_under_a_class_appears_under_that_class(env):
    client, cfg, data = env
    for name in ("Physics 101", "Chemistry 210"):
        assert client.post("/api/classes", json={"name": name}, headers=H).status_code == 201
    assert client.post("/api/record/start", json={"class": ""}, headers=H).status_code == 400  # class is required
    assert client.post("/api/record/start", json={"class": "no-such-class"}, headers=H).status_code == 400

    r = client.post("/api/record/start", json={"class": "physics-101"}, headers=H)
    assert r.status_code == 201, r.text
    session = r.json()["session"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", session)  # date-time default name
    status = client.get("/api/record/status").json()
    assert status["recording"] and status["class_name"] == "Physics 101"
    assert "permission" in status["reminder"]
    assert client.post("/api/record/start", json={"class": "chemistry-210"}, headers=H).status_code == 409  # one at a time
    time.sleep(1.5)
    assert client.post("/api/record/bookmark", headers=H).status_code == 200
    time.sleep(0.5)
    assert client.post("/api/record/stop", headers=H).status_code == 200
    assert client.get("/api/record/status").json()["recording"] is False

    phys = client.get("/api/classes/physics-101").json()["sessions"]
    chem = client.get("/api/classes/chemistry-210").json()["sessions"]
    assert [s["name"] for s in phys] == [session] and chem == []
    assert (data / "classes" / "physics-101" / "sessions" / session / "audio.wav").is_file()
    detail = client.get(f"/api/sessions/physics-101/{session}").json()
    assert detail["state"] == "queued" and len(detail["bookmarks"]) == 1  # auto-queued for processing on Stop
    assert [Path(j["session"]).name for j in jobs(data)] == [session]
    classes = {c["slug"]: c["sessions"] for c in client.get("/api/classes").json()}
    assert classes == {"physics-101": 1, "chemistry-210": 0}


def test_recording_that_stops_itself_is_queued_too(env):
    """Auto-stop (per-class limit) goes through the same queue as Stop."""
    client, cfg, data = env
    lib.create_class(data, "Short")
    cj = data / "classes" / "short" / "class.json"
    cj.write_text(json.dumps({**json.loads(cj.read_text()), "max_length_minutes": 2 / 60}), encoding="utf-8")
    rec = client.app.state.recorder
    h = rec.start("short", None)
    for _ in range(200):
        if not h.is_running():
            break
        time.sleep(0.1)
    assert rec.reap() is not None
    assert [Path(j["session"]).name for j in jobs(data)] == [h.name]
    assert h.meta()["stop_reason"] == "max_length"


def test_class_validation(env):
    client, _, _ = env
    assert client.post("/api/classes", json={"name": "Bio"}, headers=H).status_code == 201
    assert client.post("/api/classes", json={"name": "bio"}, headers=H).status_code == 400  # same slug
    assert client.post("/api/classes", json={"name": "!!!"}, headers=H).status_code == 400
    assert client.post("/api/classes", json={"name": "X", "max_length_minutes": 0}, headers=H).status_code == 400
    assert client.get("/api/classes/..").status_code == 404


# ------------------------------------------------------------------ plain files, open from the app
def test_r6_recording_transcript_and_summary_open_from_app_and_exist_as_files(env):
    client, _, data = env
    d = fake_processed_session(data)
    detail = client.get(f"/api/sessions/physics-101/{SESSION}").json()
    assert detail["state"] == "processed" and detail["folder"] == str(d)
    for f in ("audio.wav", "transcript.md", "transcript.txt", "summary.md"):
        assert f in detail["files"] and (d / f).is_file(), f
        r = client.get(f"/files/physics-101/{SESSION}/{f}")
        assert r.status_code == 200 and r.content == (d / f).read_bytes(), f
    r = client.get(f"/files/physics-101/{SESSION}/audio.wav", headers={"Range": "bytes=100-199"})
    assert r.status_code == 206 and len(r.content) == 100  # seeking works
    assert client.get(f"/api/sessions/physics-101/{SESSION}/transcript").json()["segments"][0]["text"] == "Hello class."
    assert detail["dropped_speech"]["total_s"] == 1.5  # surfaced to the transcript view


def test_transcript_edits_go_to_editable_copy_only(env):
    client, _, data = env
    d = fake_processed_session(data)
    raw_before = (d / RAW).read_bytes()
    r = client.put(f"/api/sessions/physics-101/{SESSION}/transcript", headers=H,
                   json={"segments": [{"id": 1, "text": "A question from the back?", "speaker": "Speaker 2"}]})
    assert r.status_code == 200, r.text
    doc = json.loads((d / "transcript.json").read_text(encoding="utf-8"))
    assert doc["segments"][1] | {} == {**doc["segments"][1], "text": "A question from the back?", "speaker": "Speaker 2", "role": "Audience"}
    assert "Speaker 2 (Audience): A question from the back?" in (d / "transcript.txt").read_text(encoding="utf-8")
    assert (d / RAW).read_bytes() == raw_before
    bad = client.put(f"/api/sessions/physics-101/{SESSION}/transcript", headers=H,
                     json={"segments": [{"id": 0, "speaker": "Speaker 9"}]})
    assert bad.status_code == 400


def test_summary_placeholder_never_overwrites(tmp_path):
    (tmp_path / "summary.md").write_text("my notes", encoding="utf-8")
    assert lib.write_summary_placeholder(tmp_path) is False
    assert (tmp_path / "summary.md").read_text(encoding="utf-8") == "my notes"


def test_import_creates_session_in_class_and_queues_it(env, tmp_path):
    client, _, data = env
    lib.create_class(data, "Physics 101")
    write_tone_wav(tmp_path / "phone.wav")
    ms = int(time.mktime((2026, 9, 1, 9, 30, 0, 0, 0, -1)) * 1000)
    with open(tmp_path / "phone.wav", "rb") as f:
        r = client.post("/api/import", headers=H, files={"file": ("phone.wav", f, "audio/wav")},
                        data={"class_slug": "physics-101", "last_modified_ms": str(ms)})
    assert r.status_code == 201, r.text
    assert r.json()["session"] == "2026-09-01_09-30-00"
    d = data / "classes" / "physics-101" / "sessions" / "2026-09-01_09-30-00"
    assert (d / "audio.wav").read_bytes() == (tmp_path / "phone.wav").read_bytes()
    assert json.loads((d / "meta.json").read_text())["source"] == {"kind": "import", "file": "phone.wav", "bytes": (d / "audio.wav").stat().st_size,
                                                                   "sha256": hashlib.sha256((d / "audio.wav").read_bytes()).hexdigest()}
    assert [j["state"] for j in jobs(data)] == ["queued"]
    bad = client.post("/api/import", headers=H, files={"file": ("x.exe", b"MZ", "application/octet-stream")},
                      data={"class_slug": "physics-101"})
    assert bad.status_code == 400


# ------------------------------------------------------------------ retention (Opus)
def test_opus_copy_is_verified_and_wav_kept_by_default(tmp_path):
    write_tone_wav(tmp_path / "audio.wav", seconds=5)
    wav = (tmp_path / "audio.wav").read_bytes()
    rec = compress_session(tmp_path)
    assert rec["verified"] and rec["original_kept"] and (tmp_path / "audio.opus").stat().st_size < len(wav) / 4
    assert (tmp_path / "audio.wav").read_bytes() == wav
    assert json.loads((tmp_path / "retention.json").read_text())["verified"] is True


def test_wav_deleted_only_when_enabled_and_verified(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    write_tone_wav(a / "audio.wav")
    assert compress_session(a, delete_original=True)["original_kept"] is False and not (a / "audio.wav").exists()
    write_tone_wav(b / "audio.wav")
    shutil.copy(a / "audio.opus", b / "audio.opus")
    (b / "audio.opus").write_bytes((b / "audio.opus").read_bytes()[: 2000])  # truncated: must fail verification
    import lecture_recorder.retention as ret_mod

    real = ret_mod.verify_opus
    ret_mod.verify_opus = lambda *a, **k: {**real(*a, **k), "ok": False}  # the fresh copy fails too
    try:
        rec = compress_session(b, delete_original=True)
    finally:
        ret_mod.verify_opus = real
    assert rec["verified"] is False and (b / "audio.wav").exists()
    rec = compress_session(b, delete_original=True)  # a copy that fails is made again, then verified
    assert rec["verified"] and rec["original_kept"] is False and not (b / "audio.wav").exists()


def test_config_keeps_wav_by_default(tmp_path):
    from lecture_recorder.config import RetentionConfig

    assert RetentionConfig().delete_original_after_opus is False
    assert load_config(tmp_path).retention.delete_original_after_opus is False  # no config.toml -> default


# ------------------------------------------------------------------ regressions: nothing gets stranded
def test_monitor_queues_a_recording_that_stops_while_nobody_polls(tmp_path):
    data = tmp_path / "data"
    cfg = dataclasses.replace(load_config(), data_dir=data, processing=AUTO)
    lib.create_class(data, "Short")
    cj = data / "classes" / "short" / "class.json"
    cj.write_text(json.dumps({**json.loads(cj.read_text()), "max_length_minutes": 2 / 60}), encoding="utf-8")
    app = create_app(cfg, autoprocess=False, monitor=True, synthetic_source={"pattern": "tone"})
    h = app.state.recorder.start("short", None)
    for _ in range(300):  # background monitor only; the test never calls reap() or status
        if jobs(data):
            break
        time.sleep(0.1)
    assert [Path(j["session"]).name for j in jobs(data)] == [h.name]
    assert app.state.recorder.handle is None


def test_startup_queues_app_recordings_missed_while_closed_but_not_old_sessions(tmp_path):
    data = tmp_path / "data"
    cfg = dataclasses.replace(load_config(), data_dir=data)
    lib.create_class(data, "Physics 101")
    root = data / "classes" / "physics-101" / "sessions"
    for name, flag in (("2026-09-01_09-00-00", True), ("2026-08-01_09-00-00", False)):
        (root / name).mkdir(parents=True)
        write_tone_wav(root / name / "audio.wav")
        (root / name / "meta.json").write_text(json.dumps({"name": name, "process_on_stop": flag}), encoding="utf-8")
    rec = create_app(cfg, autoprocess=False, monitor=False).state.recorder
    assert [d.name for d in rec.queue_missed()] == ["2026-09-01_09-00-00"]
    assert rec.queue_missed() == []  # already queued
    assert [Path(j["session"]).name for j in jobs(data)] == ["2026-09-01_09-00-00"]


def test_failed_finalize_keeps_the_handle_for_a_retry(env):
    client, _, data = env
    lib.create_class(data, "Physics 101")
    rec = client.app.state.recorder
    h = rec.start("physics-101", None)
    real_stop = h.stop
    h.stop(timeout=60)  # recorder process ends; the WAV is written
    calls = {"n": 0}

    def flaky_stop(timeout=60):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("audio.pcm is locked by another program")
        return real_stop(timeout)

    h.stop = flaky_stop
    with pytest.raises(PermissionError):
        rec.reap()
    assert rec.handle is h and jobs(data) == []  # not forgotten
    rec.reap()
    assert rec.handle is None and len(jobs(data)) == 1


def test_kick_during_worker_exit_runs_the_worker_again(monkeypatch, tmp_path):
    import lecture_recorder.web.app as web

    cfg = dataclasses.replace(load_config(), data_dir=tmp_path)
    rec = web.Recorder(cfg, autoprocess=True)
    runs = []

    def fake_run_worker(data_dir, blocked=None, keep_awake=None, stop=None):
        runs.append(1)
        if len(runs) == 1:
            rec.kick()  # a Stop/import lands while the worker is finishing

    monkeypatch.setattr(web, "run_worker", fake_run_worker)
    rec.kick()
    for _ in range(100):
        if rec._worker is None:
            break
        time.sleep(0.02)
    assert len(runs) == 2 and rec._worker is None


def test_hand_edited_transcript_files_are_not_overwritten_by_app_edits(env):
    client, _, data = env
    d = fake_processed_session(data)
    (d / "transcript.md").write_text("# my own notes\n", encoding="utf-8")
    r = client.put(f"/api/sessions/physics-101/{SESSION}/transcript", headers=H,
                   json={"segments": [{"id": 0, "text": "Hello everyone."}]})
    assert r.json()["kept_hand_edited"] == ["transcript.md"]
    assert (d / "transcript.md").read_text(encoding="utf-8") == "# my own notes\n"
    assert "Hello everyone." in (d / "transcript.txt").read_text(encoding="utf-8")


def test_failed_import_leaves_no_broken_session(env, monkeypatch, tmp_path):
    import lecture_recorder.web.app as web

    client, _, data = env
    lib.create_class(data, "Physics 101")

    def disk_full(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(web.os, "replace", disk_full)
    with pytest.raises(OSError):
        client.post("/api/import", headers=H, files={"file": ("a.wav", b"RIFF0000WAVE", "audio/wav")},
                    data={"class_slug": "physics-101"})
    sessions = data / "classes" / "physics-101" / "sessions"
    assert not sessions.exists() or list(sessions.iterdir()) == []
    assert jobs(data) == []


# ------------------------------------------------------------------ end to end on GPU
@pytest.mark.gpu
@pytest.mark.models
def test_r6_end_to_end_import_process_open(env, tmp_path):
    """Import the synthetic lecture through the app, run the queue, open every file from the app."""
    from lecture_recorder.pipeline.queue import run_worker

    client, _, data = env
    out = tmp_path / "fx"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "dev" / "make_synthetic_fixture.py"), "--out", str(out)], check=True,
                   capture_output=True, timeout=300)
    lib.create_class(data, "Physics 101")
    with open(out / "lecture_clean.wav", "rb") as f:
        r = client.post("/api/import", headers=H, files={"file": ("lecture.wav", f, "audio/wav")},
                        data={"class_slug": "physics-101"})
    name = r.json()["session"]
    done = run_worker(data)
    assert [(j.get("kind"), j["state"]) for j in done] == [("transcribe", "done"), ("summarize", "done"), ("embed", "done")], done
    detail = client.get(f"/api/sessions/physics-101/{name}").json()
    assert detail["state"] == "processed"
    assert detail["summary"]["kind"] == "generated" and detail["summary"]["state"] == "generated"  # auto-summary
    # The job process reads this install's config.toml (same home as the user's app: one GPU lock, the same
    # models), so the WAV is kept or deleted exactly as the user's own retention setting says.
    deletes = load_config().retention.delete_original_after_opus
    assert detail["retention"]["verified"] and detail["retention"]["original_kept"] is (not deletes)
    for f in ("audio.opus", "transcript.md", "transcript.txt", "transcript.json", "summary.md") + (() if deletes else ("audio.wav",)):
        r = client.get(f"/files/physics-101/{name}/{f}")
        assert r.status_code == 200 and r.content == (Path(detail["folder"]) / f).read_bytes(), f
    text = client.get(f"/files/physics-101/{name}/transcript.txt").text
    assert "purple elephant theorem" in text and "Speaker 1 (Main)" in text and "(Audience)" in text
