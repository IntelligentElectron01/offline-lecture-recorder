"""Zoom recording import: planning which files to use, audio extraction, speaker hints from the VTT, the
web and terminal import paths. Synthetic media in temp folders only; nothing here touches Zoom."""
import dataclasses
import json
from datetime import datetime
from pathlib import Path

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient

import lecture_recorder  # noqa: F401  (lockdown)
from lecture_recorder import library as lib
from lecture_recorder import zoom
from lecture_recorder.config import load_config
from lecture_recorder.pipeline.audio import extract_audio, load_audio
from lecture_recorder.recorder.session import read_json
from lecture_recorder.web.app import create_app

H = {"X-Lecture-Recorder": "1"}
BASE = "http://127.0.0.1:8765"
RATE = 16000


@pytest.fixture
def cfg(tmp_path):
    data = tmp_path / "data"
    lib.create_class(data, "Zoom Class")
    return dataclasses.replace(load_config(), data_dir=data)


@pytest.fixture
def client(cfg):
    app = create_app(cfg, autoprocess=False, monitor=False)
    with TestClient(app, base_url=BASE) as c:
        yield c


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
