"""Record one app's audio (Windows process loopback) and import Zoom recordings.

Mocked COM for the picker, the app source, auto-stop on app exit and the meta source; a real-hardware
test (marker "audio") with this suite's own tone players. Zoom is never started, joined or touched.
Everything runs in temp folders; the real data/ folder is never used. The system volume is never changed.
"""
import dataclasses
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient

if sys.platform != "win32":
    pytest.skip("Windows process loopback and Zoom-on-Windows tests", allow_module_level=True)

import lecture_recorder  # noqa: F401  (lockdown)
from lecture_recorder import importer as imp
from lecture_recorder import library as lib
from lecture_recorder import zoom
from lecture_recorder.config import load_config
from lecture_recorder.pipeline.audio import extract_audio, load_audio
from lecture_recorder.recorder import apps, control, process, sources, winaudio
from lecture_recorder.recorder.safeguards import Safeguards, State
from lecture_recorder.recorder.session import read_json
from lecture_recorder.web.app import create_app

ROOT = Path(__file__).resolve().parents[1]
H = {"X-Lecture-Recorder": "1"}
BASE = "http://127.0.0.1:8765"
RATE = 16000


# ------------------------------------------------------------------ picker (mocked COM)
TABLE = {  # pid: (parent, exe)
    10: (1, "explorer.exe"),
    100: (10, "Zoom.exe"), 101: (100, "Zoom.exe"),  # audio plays from a helper Zoom.exe
    200: (10, "brave.exe"), 201: (200, "brave.exe"),
    300: (10, "Spotify.exe"),
    400: (10, "steam.exe"),
}


@pytest.fixture
def fake_sessions(monkeypatch):
    sessions = [(201, True), (300, False), (101, True), (400, True), (999, True)]  # 999: gone already
    monkeypatch.setattr(winaudio, "process_table", lambda: TABLE)
    monkeypatch.setattr(winaudio, "audio_sessions", lambda: sessions)
    monkeypatch.setattr(winaudio, "exe_path", lambda pid: f"C:\\Apps\\{TABLE[pid][1]}")
    names = {"Zoom.exe": "Zoom Meetings", "brave.exe": "Brave Browser"}
    monkeypatch.setattr(winaudio, "friendly_name", lambda p: names.get(Path(p.replace("\\", "/")).name))
    return sessions


def test_picker_lists_zoom_first_by_root_process(fake_sessions):
    rows = apps.list_audio_apps()
    assert [r["name"] for r in rows] == ["Zoom Meetings", "Brave Browser", "steam", "Spotify"]
    zoom_row = rows[0]
    assert zoom_row == {"pid": 100, "exe": "Zoom.exe", "name": "Zoom Meetings", "active": True}  # root, not the helper
    assert rows[-1]["active"] is False  # not playing now: listed last
    assert apps.find_app(100)["name"] == "Zoom Meetings"
    assert apps.find_app(101) is None and apps.find_app(999) is None  # only listed roots are accepted


def test_picker_without_zoom_sorts_playing_apps_first(fake_sessions):
    fake_sessions[:] = [(300, True), (201, False)]
    assert [(r["name"], r["active"]) for r in apps.list_audio_apps()] == [("Spotify", True), ("Brave Browser", False)]


def test_root_stops_at_cycles_and_other_exes():
    table = {1: (2, "a.exe"), 2: (1, "a.exe"), 5: (6, "a.exe"), 6: (0, "b.exe")}
    assert apps._root(1, table) in (1, 2)
    assert apps._root(5, table) == 5


# ------------------------------------------------------------------ app source (fake loopback + watcher)
class FakeLoopback:
    def __init__(self, blocks=None, level=0.1):
        self.started = self.stopped = 0
        self.level = level
        self.blocks = blocks

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def read(self, timeout):
        if self.blocks is not None:
            if not self.blocks:
                time.sleep(timeout)
                return None
            return self.blocks.pop(0)
        time.sleep(0.1)  # real-time pace, like the engine
        t = np.arange(RATE // 10) / RATE
        return (self.level * 32767 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)


class FakeWatcher:
    instances: list = []

    def __init__(self, pid, on_exit):
        if pid == 404:
            raise ProcessLookupError("gone")
        self.pid, self.on_exit, self.closed = pid, on_exit, False
        FakeWatcher.instances.append(self)

    def close(self):
        self.closed = True


class FakeMic:
    def __init__(self, blocks):
        self.blocks = list(blocks)
        self.started = 0

    def start(self):
        self.started += 1

    def stop(self):
        pass

    def read(self, timeout):
        return self.blocks.pop(0) if self.blocks else None


def test_app_source_exit_ends_and_paused_read_waits_for_it():
    lb = FakeLoopback()
    src = sources.AppSource(100, RATE, loopback=lb, watcher=FakeWatcher)
    src.start()
    assert src.read(0.1).size == RATE // 10 and src.ended is None
    src.stop()  # paused
    t0 = time.monotonic()
    assert src.read(0.3) is None and time.monotonic() - t0 >= 0.25  # blocks like a stopped mic (no spin)
    w = FakeWatcher.instances[-1]
    threading.Timer(0.1, w.on_exit).start()
    t0 = time.monotonic()
    assert src.read(5) is None and time.monotonic() - t0 < 2  # the app closing wakes a paused read at once
    assert src.ended == "app_closed"
    src.close()
    assert w.closed and lb.stopped >= 2


def test_app_source_start_fails_if_app_is_gone():
    with pytest.raises(ProcessLookupError):
        sources.AppSource(404, RATE, loopback=FakeLoopback(), watcher=FakeWatcher).start()


def test_app_source_mixes_microphone_and_bounds_drift():
    app = [np.full(1600, 1000, np.int16) for _ in range(3)]
    mic = FakeMic([np.full(1600, 200, np.int16), np.full(800, 300, np.int16)])  # the mic delivers less
    src = sources.AppSource(100, RATE, mic=mic, loopback=FakeLoopback(blocks=app), watcher=FakeWatcher)
    src.start()
    a = src.read(0.1)
    assert a.size == 1600 and (a == 1200).all()
    b = src.read(0.1)
    assert (b[:800] == 1300).all() and (b[800:] == 1000).all()  # mic short: the app's timeline is kept
    assert mic.started == 1
    mic.blocks = [np.full(1600, 10, np.int16)] * 20  # the mic runs ahead: at most 0.5 s buffered
    src.read(0.1)
    assert src._mic_buf.size <= 0.5 * RATE
    big = sources.AppSource(100, RATE, mic=FakeMic([np.full(1600, 30000, np.int16)]),
                            loopback=FakeLoopback(blocks=[np.full(1600, 30000, np.int16)]), watcher=FakeWatcher)
    big.start()
    assert big.read(0.1).max() == 32767  # clipped, never wrapped


def test_safeguards_stop_when_app_closes_even_paused():
    sg = Safeguards(max_length_s=7200, silence_pause_s=600, silence_dbfs=-50)
    sg.on_user("pause")
    act = sg.on_source_ended("app_closed")
    assert act.kind == "stop" and sg.state is State.STOPPED and sg.reason == "app_closed"
    assert sg.on_source_ended("app_closed") is None


# ------------------------------------------------------------------ recorder: meta source + auto-stop on exit
@pytest.fixture
def cfg(tmp_path):
    data = tmp_path / "data"
    lib.create_class(data, "Zoom Class")
    return dataclasses.replace(load_config(), data_dir=data)


class FakeProc:
    returncode = 1

    def __init__(self, *a, **k):
        pass

    def poll(self):
        return 1


ZOOM = {"pid": 100, "exe": "Zoom.exe", "name": "Zoom Meetings", "active": True}


def start_meta_only(cfg, monkeypatch, **kw) -> Path:
    """start_recording up to the recorder launch (the child process is faked); returns the session folder."""
    monkeypatch.setattr(control.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(winaudio, "supported", lambda: None)
    with pytest.raises(control.RecorderError):
        control.start_recording("zoom-class", config=cfg, app=ZOOM, **kw)
    (d,) = (cfg.data_dir / "classes" / "zoom-class" / "sessions").iterdir()
    return d


def test_meta_marks_app_source(cfg, monkeypatch):
    d = start_meta_only(cfg, monkeypatch)
    src = read_json(d / "meta.json")["source"]
    assert src == {"kind": "app", "label": "app:Zoom Meetings", "app_name": "Zoom Meetings", "exe": "Zoom.exe",
                   "pid": 100, "include_mic": False}
    assert read_json(d / "meta.json")["sample_rate"] == cfg.recording.sample_rate


def test_app_recording_refused_where_unsupported(cfg, monkeypatch):
    monkeypatch.setattr(winaudio, "supported", lambda: "needs Windows 11")
    with pytest.raises(control.RecorderError, match="Windows 11"):
        control.start_recording("zoom-class", config=cfg, app=ZOOM)


def test_recorder_stops_when_app_closes_and_saves_wav(cfg, monkeypatch):
    d = start_meta_only(cfg, monkeypatch)
    made = []

    class ClosingApp(sources.AppSource):
        def __init__(self, pid, rate, mic=None):
            super().__init__(pid, rate, mic, loopback=FakeLoopback(), watcher=FakeWatcher)
            made.append(self)
            threading.Timer(1.5, lambda: FakeWatcher.instances[-1].on_exit()).start()  # "Zoom" quits after 1.5 s

    monkeypatch.setattr(process, "AppSource", ClosingApp)
    t0 = time.monotonic()
    assert process.main(["--session-dir", str(d)]) == 0
    assert time.monotonic() - t0 < 6
    meta = read_json(d / "meta.json")
    assert meta["stop_reason"] == "app_closed" and meta["source"]["label"] == "app:Zoom Meetings"
    assert (d / "audio.wav").is_file() and not (d / "audio.pcm").exists()
    assert 1.0 <= meta["recorded_s"] <= 3.0
    assert made[0]._watcher is None  # closed: the process handle is not leaked


def test_recorder_reports_app_already_closed(cfg, monkeypatch):
    d = start_meta_only(cfg, monkeypatch)
    monkeypatch.setattr(process, "AppSource", lambda pid, rate, mic=None: sources.AppSource(
        404, rate, loopback=FakeLoopback(), watcher=FakeWatcher))
    assert process.main(["--session-dir", str(d)]) == 2
    assert read_json(d / "meta.json")["stop_reason"] == "app_closed"


# ------------------------------------------------------------------ API
@pytest.fixture
def client(cfg):
    app = create_app(cfg, autoprocess=False, monitor=False)
    with TestClient(app, base_url=BASE) as c:
        yield c


def test_api_lists_apps_and_validates_the_pick(client, fake_sessions, monkeypatch):
    monkeypatch.setattr(winaudio, "supported", lambda: None)
    r = client.get("/api/apps").json()
    assert r["supported"] and r["apps"][0]["name"] == "Zoom Meetings"
    for bad in ("100", True, -1, 1.5):
        assert client.post("/api/record/start", headers=H, json={"class": "zoom-class", "app_pid": bad}).status_code == 400
    r = client.post("/api/record/start", headers=H, json={"class": "zoom-class", "app_pid": 101})  # not a listed app
    assert r.status_code == 409 and "Refresh the list" in r.json()["detail"]
    monkeypatch.setattr(winaudio, "supported", lambda: "Recording from an app needs Windows 11.")
    assert client.get("/api/apps").json() == {"supported": False, "reason": "Recording from an app needs Windows 11.", "apps": []}


def test_api_starts_app_recording_and_status_names_the_app(client, cfg, fake_sessions, monkeypatch):
    seen = {}

    def fake_start(slug, **kw):
        seen.update(kw)
        raise control.RecorderError("stop here")

    import lecture_recorder.web.app as webapp

    monkeypatch.setattr(webapp, "start_recording", fake_start)
    r = client.post("/api/record/start", headers=H, json={"class": "zoom-class", "app_pid": 100, "include_mic": True, "device": 3})
    assert r.status_code == 409
    assert seen["app"]["name"] == "Zoom Meetings" and seen["include_mic"] is True and seen["synthetic"] is None


def test_ui_has_app_picker_volume_note_and_zoom_import():
    js = (ROOT / "src/lecture_recorder/web/static/app.js").read_text(encoding="utf-8")
    for text in ('id: "src-app"', "/api/apps", "Also include my microphone", "Volume Mixer", "The system volume doesn't matter",
                 "Recording stops when the app closes", "/api/import/zoom-plan", "webkitdirectory", "No audio from ${app}"):
        assert text in js, text
    assert 'id: "record-with-mic" }' in js  # unchecked by default


# ------------------------------------------------------------------ Zoom import: planning
def test_plan_local_recording_prefers_audio_only_m4a():
    f = "2026-10-20 14.00.05 Physics Lecture 95812345678"
    p = zoom.plan([(f"{f}/video1234567890.mp4", 900_000_000), (f"{f}/audio1234567890.m4a", 40_000_000),
                   (f"{f}/chat.txt", 1000), (f"{f}/Audio Record/audioJaneDoe11234.m4a", 5_000_000),
                   (f"{f}/Audio Record/audioJohnRoe21234.m4a", 4_000_000)])
    assert p["error"] is None and p["audio"] == f"{f}/audio1234567890.m4a" and p["chat"] == f"{f}/chat.txt"
    assert p["participants"] == 2 and p["others"] == 0 and p["title"] == "Physics Lecture"
    assert p["when"].replace(tzinfo=None) == datetime(2026, 10, 20, 14, 0, 5)


def test_plan_cloud_download_with_transcript_and_mp4_fallback():
    f = "Physics 101 cloud"
    files = [(f"{f}/GMT20261020-180005_Recording_1920x1080.mp4", 500), (f"{f}/GMT20261020-180005_Recording_gallery_1280x720.mp4", 300),
             (f"{f}/GMT20261020-180005_Recording.cc.vtt", 10), (f"{f}/GMT20261020-180005_Recording.transcript.vtt", 20),
             (f"{f}/GMT20261020-180005_RecordingnewChat.txt", 5)]
    p = zoom.plan(files)
    assert p["audio"].endswith("_1920x1080.mp4") and p["others"] == 1  # no m4a: the largest mp4; the gallery one is reported
    assert p["transcript"].endswith(".transcript.vtt") and p["chat"].endswith("newChat.txt")
    assert p["when"].timestamp() == datetime(2026, 10, 20, 18, 0, 5, tzinfo=zoom.timezone.utc).timestamp()


def test_plan_explains_unconverted_and_empty_folders():
    p = zoom.plan([("2026-10-20 14.00.05 Physics 95812345678/double_click_to_convert_01.zoom", 10_000)])
    assert p["error"] == zoom.UNCONVERTED and p["audio"] is None
    assert "No Zoom audio" in zoom.plan([("x/notes.txt", 1)])["error"]


# ------------------------------------------------------------------ Zoom import: files
def make_media(path: Path, seconds: float = 2.0, video: bool = True, freq: float = 440.0) -> None:
    """A small mp4 (mpeg4 video + AAC audio) or m4a (AAC only), like Zoom's files."""
    with av.open(str(path), "w", format="mp4" if video else "ipod") as c:
        a = c.add_stream("aac", rate=48000, layout="mono")
        v = None
        if video:
            v = c.add_stream("mpeg4", rate=10)
            v.width, v.height, v.pix_fmt = 64, 48, "yuv420p"
            for i in range(int(seconds * 10)):
                frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 10 % 255, np.uint8), format="rgb24")
                for pkt in v.encode(frame):
                    c.mux(pkt)
        n = int(seconds * 48000)
        x = (0.3 * np.sin(2 * np.pi * freq * np.arange(n) / 48000)).astype(np.float32)
        for i in range(0, n, 1024):
            f = av.AudioFrame.from_ndarray(x[None, i:i + 1024], format="flt", layout="mono")
            f.sample_rate, f.pts = 48000, i
            for pkt in a.encode(f):
                c.mux(pkt)
        for s in (a, v):
            if s is not None:
                for pkt in s.encode(None):
                    c.mux(pkt)


VTT = """WEBVTT

1
00:00:00.000 --> 00:00:01.000
Prof. Ada Lovelace: Welcome to the lecture.

2
00:00:01.000 --> 00:00:01.600
Grace Hopper: Is this on the exam?

3
00:00:01.600 --> 00:00:02.000
Prof. Ada Lovelace: Yes.
"""


def test_extract_audio_keeps_only_audio_without_reencoding(tmp_path):
    make_media(tmp_path / "v.mp4")
    out = extract_audio(tmp_path / "v.mp4", tmp_path)
    assert out.name == "audio.m4a"
    with av.open(str(out)) as c:
        assert [s.type for s in c.streams] == ["audio"] and c.streams[0].codec_context.name == "aac"
    x = load_audio(out)
    assert abs(x.size / RATE - 2.0) < 0.1 and np.abs(x).max() > 0.2
    spec = np.abs(np.fft.rfft(x))
    assert abs(np.fft.rfftfreq(x.size, 1 / RATE)[spec.argmax()] - 440) < 5


def test_extract_audio_refuses_video_without_audio(tmp_path):
    with av.open(str(tmp_path / "silent.mp4"), "w") as c:
        v = c.add_stream("mpeg4", rate=10)
        v.width, v.height, v.pix_fmt = 64, 48, "yuv420p"
        for i in range(5):
            for pkt in v.encode(av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), np.uint8), format="rgb24")):
                c.mux(pkt)
        for pkt in v.encode(None):
            c.mux(pkt)
    with pytest.raises(ValueError, match="no audio track"):
        extract_audio(tmp_path / "silent.mp4", tmp_path)
    assert not list(tmp_path.glob("audio*"))


def test_vtt_names_and_speaker_hints():
    cues = zoom.parse_vtt(VTT)
    assert [(c["name"], c["text"]) for c in cues] == [("Prof. Ada Lovelace", "Welcome to the lecture."),
                                                      ("Grace Hopper", "Is this on the exam?"), ("Prof. Ada Lovelace", "Yes.")]
    segs = [{"start": 0.0, "end": 1.0, "speaker": "Speaker 1"}, {"start": 1.05, "end": 1.6, "speaker": "Speaker 2"},
            {"start": 1.6, "end": 2.0, "speaker": "Speaker 1"}]
    assert zoom.speaker_names(cues, segs) == [{"speaker": "Speaker 1", "name": "Prof. Ada Lovelace", "share": 1.0},
                                             {"speaker": "Speaker 2", "name": "Grace Hopper", "share": 1.0}]
    assert zoom.parse_vtt("WEBVTT\n\n00:01.000 --> 00:02.500\nno name here\n")[0] == {"start": 1.0, "end": 2.5, "name": None, "text": "no name here"}


def zoom_folder(tmp_path: Path, m4a: bool = True) -> Path:
    f = tmp_path / "zoom" / "2026-10-20 14.00.05 Physics Lecture 95812345678"
    (f / "Audio Record").mkdir(parents=True)
    make_media(f / "video1234567890.mp4")
    if m4a:
        make_media(f / "audio1234567890.m4a", video=False)
    make_media(f / "Audio Record" / "audioAda11234.m4a", video=False)
    (f / "GMT_Recording.transcript.vtt").write_text(VTT, encoding="utf-8")
    (f / "meeting_saved_chat.txt").write_text("14:01:00 From Grace to Everyone: hi\n", encoding="utf-8")
    return f


def snapshot(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def test_api_imports_zoom_folder_and_mp4(client, cfg, tmp_path):
    f = zoom_folder(tmp_path)
    files = [{"name": f"{f.name}/{p.relative_to(f).as_posix()}", "size": p.stat().st_size} for p in f.rglob("*") if p.is_file()]
    plan = client.post("/api/import/zoom-plan", headers=H, json={"files": files}).json()
    assert plan["audio"].endswith("audio1234567890.m4a") and plan["participants"] == 1
    up = lambda rel: (Path(rel).name, (f.parent / rel).read_bytes())  # noqa: E731
    r = client.post("/api/import/zoom", headers=H, data={"class_slug": "zoom-class", "folder": plan["folder"]},
                    files={"audio": up(plan["audio"]), "transcript": up(plan["transcript"]), "chat": up(plan["chat"])})
    assert r.status_code == 201, r.text
    d = cfg.data_dir / "classes" / "zoom-class" / "sessions" / r.json()["session"]
    assert d.name == "2026-10-20_14-00-05"
    assert sorted(p.name for p in d.iterdir()) == ["audio.m4a", "meta.json", zoom.CHAT_FILE, zoom.TRANSCRIPT_FILE]
    meta = read_json(d / "meta.json")
    assert meta["title"] == "Physics Lecture" and meta["date_from"] == "Zoom folder name" and meta["source"]["zoom"] is True
    assert client.get(f"/files/zoom-class/{d.name}/{zoom.TRANSCRIPT_FILE}").text.startswith("WEBVTT")
    assert not list(d.parent.glob(".import-*"))  # no temp files left

    # a plain mp4 through the normal import: only the audio track is kept
    r = client.post("/api/import", headers=H, data={"class_slug": "zoom-class"},
                    files={"file": ("lecture.mp4", (f / "video1234567890.mp4").read_bytes())})
    assert r.status_code == 201, r.text
    d2 = cfg.data_dir / "classes" / "zoom-class" / "sessions" / r.json()["session"]
    assert sorted(p.name for p in d2.iterdir()) == ["audio.m4a", "meta.json"]
    bad = client.post("/api/import", headers=H, data={"class_slug": "zoom-class"}, files={"file": ("x.mkv", b"abc")})
    assert bad.status_code == 400 and "mp4" in bad.json()["detail"]
    broken = client.post("/api/import", headers=H, data={"class_slug": "zoom-class"}, files={"file": ("broken.mp4", b"not a video")})
    assert broken.status_code == 400
    assert len(list((cfg.data_dir / "classes" / "zoom-class" / "sessions").iterdir())) == 2  # nothing left behind


def test_session_page_shows_zoom_speaker_hints(client, cfg, tmp_path):
    from lecture_recorder.pipeline.outputs import write_outputs

    d = cfg.data_dir / "classes" / "zoom-class" / "sessions" / "2026-10-20_14-00-05"
    d.mkdir(parents=True)
    make_media(d / "audio.m4a", video=False)
    (d / zoom.TRANSCRIPT_FILE).write_text(VTT, encoding="utf-8")
    write_outputs(d, {"speakers": [], "segments": [
        {"id": 0, "start": 0.0, "end": 1.0, "speaker": "Speaker 1", "role": "Main", "text": "Welcome.", "words": []},
        {"id": 1, "start": 1.0, "end": 1.6, "speaker": "Speaker 2", "role": "Audience", "text": "Exam?", "words": []}]})
    s = client.get(f"/api/sessions/zoom-class/{d.name}").json()
    assert zoom.TRANSCRIPT_FILE in s["files"] and "zoom_speakers" not in s  # not recomputed on status polls
    t = client.get(f"/api/sessions/zoom-class/{d.name}/transcript").json()
    assert t["zoom_speakers"][0] == {"speaker": "Speaker 1", "name": "Prof. Ada Lovelace", "share": 1.0}


def test_cli_import_zoom_falls_back_to_mp4_and_never_changes_the_folder(cfg, tmp_path, monkeypatch, capsys):
    from lecture_recorder import cli

    f = zoom_folder(tmp_path, m4a=False)
    before = snapshot(f)
    monkeypatch.setattr("lecture_recorder.config.load_config", lambda *a: cfg)
    assert cli.main(["import-zoom", str(f), "--class", "zoom-class"]) == 0
    assert snapshot(f) == before  # read only
    (d,) = (cfg.data_dir / "classes" / "zoom-class" / "sessions").iterdir()
    assert (d / "audio.m4a").is_file() and (d / zoom.TRANSCRIPT_FILE).is_file() and (d / zoom.CHAT_FILE).is_file()
    assert "1 per-participant audio file(s) were not imported" in capsys.readouterr().out

    unconverted = tmp_path / "zoom" / "2026-10-21 09.00.00 Physics 95812345678"
    unconverted.mkdir()
    (unconverted / "double_click_to_convert_01.zoom").write_bytes(b"\0" * 100)
    assert cli.main(["import-zoom", str(unconverted), "--class", "zoom-class"]) == 2
    assert "hasn't converted" in capsys.readouterr().out


# ------------------------------------------------------------------ real hardware (this suite's own tone players)
PLAYER = Path(__file__).with_name("audio_tone_player.py")


@pytest.fixture(autouse=True)
def _audio_only_when_asked(request):
    """Tests marked "audio" play tones and record them: they run only with `-m audio` (never in a
    default or filtered run)."""
    if request.node.get_closest_marker("audio") and "audio" not in (request.config.option.markexpr or "").replace("not audio", ""):
        pytest.skip("plays and records audio: run with -m audio")


def player(freq, amp, secs, *extra):
    p = subprocess.Popen([sys.executable, str(PLAYER), str(freq), str(amp), str(secs), *extra],
                         stdout=subprocess.PIPE, text=True, cwd=ROOT)
    assert p.stdout.readline().strip() == "PLAYING"
    return p


def capture(pid, seconds):
    lb = winaudio.ProcessLoopback(pid, RATE)
    lb.start()
    out, end = [], time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            if (b := lb.read(0.5)) is not None:
                out.append(b)
    finally:
        lb.stop()
    x = np.concatenate(out).astype(np.float64)[RATE // 4:] / 32768  # skip the start-up
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    f = np.fft.rfftfreq(x.size, 1 / RATE)
    band = lambda c: spec[(f > c - 15) & (f < c + 15)].sum() + 1e-30  # noqa: E731
    return {"dbfs": 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12), "other_db": 10 * np.log10(band(1000) / band(440)),
            "seconds": (x.size + RATE // 4) / RATE}


@pytest.mark.audio
@pytest.mark.skipif(winaudio.supported() is not None, reason="needs Windows process loopback")
def test_real_capture_isolates_the_app_and_ignores_system_volume(capsys):
    """(a) only the target tree, (b) the level equals the source at the current system volume (the user-approved
    spike measured 100%, 10%, muted and 48%: identical), (c) the app's own Volume Mixer slider lowers it."""
    endpoint = None
    try:
        import ctypes
        import ctypes.wintypes as wt

        dev = winaudio._default_render_device()
        ev = ctypes.c_void_p()
        winaudio._method(dev, 3, ctypes.POINTER(winaudio.GUID), wt.DWORD, ctypes.c_void_p, winaudio.PP)(
            ctypes.byref(winaudio.GUID.from_str("5CDF2C82-841E-4546-9722-0CF74078229A")), winaudio.CLSCTX_ALL, None, ctypes.byref(ev))
        v, m = ctypes.c_float(), wt.BOOL()
        winaudio._method(ev, 9, ctypes.POINTER(ctypes.c_float))(ctypes.byref(v))  # read only
        winaudio._method(ev, 15, ctypes.POINTER(wt.BOOL))(ctypes.byref(m))
        endpoint = (round(v.value, 3), bool(m.value))
        winaudio._release(ev)
        winaudio._release(dev)
    except OSError:
        pass
    amp = 0.02  # -37 dBFS RMS: quiet on the speakers
    want = 20 * np.log10(amp / np.sqrt(2))
    a, b = player(440, amp, 12), player(1000, amp, 12)
    try:
        ra = capture(a.pid, 3)
        rb = capture(b.pid, 3)
    finally:
        a.kill(), b.kill()
    c = player(440, amp, 5, "--own-volume", "0.1")  # exits by itself, restoring its own slider
    rc = capture(c.pid, 3)
    assert c.wait(15) == 0
    print(json.dumps({"system_volume_and_mute": endpoint, "source_dbfs": round(want, 2), "A": ra, "B": rb, "A_own_slider_0.1": rc}, default=float))
    assert abs(ra["dbfs"] - want) < 0.5 and ra["other_db"] < -60  # (a)+(b): exact level, the other app excluded
    assert abs(rb["dbfs"] - want) < 0.5 and rb["other_db"] > 60
    assert abs(rc["dbfs"] - (want - 20)) < 1.0  # (c): the app's own slider at 10% = -20 dB in the recording
    assert abs(ra["seconds"] - 3) < 0.15  # continuous timeline


@pytest.mark.audio
@pytest.mark.skipif(winaudio.supported() is not None, reason="needs Windows process loopback")
def test_real_recording_stops_when_the_app_exits(cfg, monkeypatch):
    """The real recorder process records a tone player and stops by itself when the player exits."""
    p = player(440, 0.02, 4)
    real = {"pid": p.pid, "exe": "python.exe", "name": "Tone player", "active": True}
    h = control.start_recording("zoom-class", config=cfg, app=real)
    end = time.monotonic() + 20
    while h.is_running() and time.monotonic() < end:
        time.sleep(0.2)
    meta = h.stop()
    assert meta["stop_reason"] == "app_closed" and meta["source"]["label"] == "app:Tone player"
    x = load_audio(h.session_dir / "audio.wav")
    assert 2.0 < x.size / RATE < 6.0
    assert 20 * np.log10(np.sqrt(np.mean(x[RATE // 2:RATE * 2] ** 2))) == pytest.approx(20 * np.log10(0.02 / np.sqrt(2)), abs=1.0)
