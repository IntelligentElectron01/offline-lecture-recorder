"""Rename lectures in the app, the default naming template, and the original file name.
Temp folders only; never the real data/ folder.
"""
import dataclasses
import json
import threading
import time
import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import lecture_recorder  # noqa: F401  (lockdown)
from lecture_recorder import library as lib
from lecture_recorder import naming, trash
from lecture_recorder.config import NamingConfig, load_config
from lecture_recorder.recorder import control
from lecture_recorder.recorder.platform import ProcessLock
from lecture_recorder.recorder.session import meta_lock, read_json, update_meta, write_json_atomic
from lecture_recorder.search.index import SearchIndex
from lecture_recorder.web.app import create_app
from test_web_app import write_tone_wav
from test_units import BIO, S1, S2, SLUG, U1_TERMS, make_lecture, sentences

BASE = "http://127.0.0.1:8765"
H = {"X-Lecture-Recorder": "1"}
SOURCE = {"kind": "import", "file": "Voice Memo 007.m4a", "bytes": 1234}


@pytest.fixture
def cfg(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    return dataclasses.replace(load_config(home), data_dir=(home / "data").resolve())


@pytest.fixture
def cls(cfg):
    """The units test's class with two transcribed lectures; the second was imported from a phone file."""
    c = cfg.data_dir / "classes" / SLUG
    c.mkdir(parents=True)
    write_json_atomic(c / "class.json", {"name": "Bio and Econ", "slug": SLUG})
    make_lecture(cfg.data_dir, S1, sentences(BIO, U1_TERMS, 6))
    make_lecture(cfg.data_dir, S2, sentences(BIO, U1_TERMS, 6, offset=1))
    meta = read_json(c / "sessions" / S2 / "meta.json")
    write_json_atomic(c / "sessions" / S2 / "meta.json", {**meta, "source": SOURCE, "date_from": "file metadata"})
    return c


def client_for(cfg):
    return TestClient(create_app(cfg, autoprocess=False, monitor=False), base_url=BASE)


@pytest.fixture
def client(cfg):
    with client_for(cfg) as c:
        yield c


def rename(client, name, title, slug=SLUG):
    # sent as the browser's JSON would be (escapes allowed, e.g. a lone surrogate "\ud800")
    return client.post(f"/api/sessions/{slug}/{name}/rename", headers={**H, "Content-Type": "application/json"},
                       content=json.dumps({"title": title}))


# ------------------------------------------------------------------ 1. rename
def test_rename_round_trip_changes_only_the_title(client, cfg, cls):
    d = cls / "sessions" / S2
    before = read_json(d / "meta.json")
    files = sorted(p.name for p in d.iterdir())
    v0 = client.get("/api/record/status").json()["trash_version"]

    r = rename(client, S2, "  Cell organelles — Lecture 2 ✓ (Zellbiologie, 細胞)  ")
    assert r.status_code == 200, r.text
    new = "Cell organelles — Lecture 2 ✓ (Zellbiologie, 細胞)"
    assert r.json()["title"] == new and r.json()["trash_version"] != v0  # other open pages re-render
    after = read_json(d / "meta.json")
    assert after == {**before, "title": new}  # nothing else changed: source.file, created, class...
    assert after["source"] == SOURCE and d.name == S2 and sorted(p.name for p in d.iterdir()) == files

    # every view shows the new title: lecture page, class list, units, exact search
    assert client.get(f"/api/sessions/{SLUG}/{S2}").json()["title"] == new
    assert {s["name"]: s["title"] for s in client.get(f"/api/classes/{SLUG}").json()["sessions"]}[S2] == new
    assert {s["name"]: s["title"] for s in client.get(f"/api/classes/{SLUG}/units").json()["sessions"]}[S2] == new
    hits = client.get("/api/search", params={"q": U1_TERMS[1]}).json()["results"]
    assert {h["session"]: h["title"] for h in hits}[S2] == new
    assert {h["session"]: h["title"] for h in hits}[S1] == "Lecture 01"

    # and back again
    assert rename(client, S2, "Lecture 03").json()["title"] == "Lecture 03"
    assert read_json(d / "meta.json") == before


@pytest.mark.parametrize("title, why", [
    ("", "can't be empty"), ("   ", "can't be empty"), ("Line\nbreak", "control characters"),
    ("Tab\there", "control characters"), ("Bell\x07", "control characters"), ("DEL\x7f", "control characters"),
    ("evil\u202egnp.exe", "control characters"), ("lone\ud800surrogate", "control characters"), ("x" * 201, "too long"), (None, "must be text"), (42, "must be text"),
])
def test_rename_validation(client, cls, title, why):
    before = read_json(cls / "sessions" / S1 / "meta.json")
    r = rename(client, S1, title)
    assert r.status_code == 400 and why in r.json()["detail"]
    assert read_json(cls / "sessions" / S1 / "meta.json") == before


def test_clean_title_keeps_unicode_and_caps_length():
    assert naming.clean_title("  Économie 101 — 第三講 🎓 ") == "Économie 101 — 第三講 🎓"
    assert naming.clean_title("e\u0301") == "é"  # NFC: the same title however it was typed
    assert naming.clean_title("x" * 200) == "x" * 200


def test_rename_guards(client, cfg, cls):
    assert rename(client, "2026-01-01_00-00-00", "x").status_code == 404
    assert rename(client, "..", "x").status_code == 404
    assert client.post(f"/api/sessions/{SLUG}/{S1}/rename", json={"title": "x"}).status_code == 403  # no app header
    assert client.post(f"/api/sessions/{SLUG}/{S1}/rename", headers={**H, "Origin": "http://evil.example"},
                       json={"title": "x"}).status_code == 403
    d = cls / "sessions" / S1
    lock = ProcessLock(d / "recorder.lock")  # a live recorder keeps its own meta.json: refused
    assert lock.acquire()
    try:
        r = rename(client, S1, "x")
        assert r.status_code == 409 and "recording" in r.json()["detail"]
    finally:
        lock.release()
    (d / "audio.pcm").write_bytes(b"\0" * 64)  # recording not saved yet (crashed, not recovered)
    assert rename(client, S1, "x").status_code == 409
    (d / "audio.pcm").unlink()
    import lecture_recorder.recorder.session as rs

    real = rs.meta_lock
    with meta_lock(d):  # another writer holds meta.json: "try again", not a server error
        rs.meta_lock = lambda sd, timeout_s=10: real(sd, 0.2)
        try:
            r = rename(client, S1, "x")
        finally:
            rs.meta_lock = real
    assert r.status_code == 409 and "try again" in r.json()["detail"]
    assert rename(client, S1, "x").status_code == 200


def test_source_file_survives_rename_delete_and_restore(cfg, cls):
    idx = SearchIndex(cfg.data_dir, cfg)
    lib.rename_session(cfg.data_dir, SLUG, S2, "Renamed")
    r = trash.delete_session(cfg.data_dir, SLUG, S2, index=idx)
    assert r["title"] == "Renamed"
    trash.restore(cfg.data_dir, r["id"], index=idx)
    meta = read_json(cls / "sessions" / S2 / "meta.json")
    assert meta["title"] == "Renamed" and meta["source"] == SOURCE
    r = trash.delete_class(cfg.data_dir, SLUG, 2, index=idx)
    trash.restore(cfg.data_dir, r["id"], index=idx)
    assert read_json(cls / "sessions" / S2 / "meta.json")["source"] == SOURCE


def test_meta_writes_wait_for_each_other(tmp_path):
    """A rename and a processing job's facts both land (cross-process lock, here two handles)."""
    d = tmp_path / "sessions" / "2026-10-01_09-00-00"
    d.mkdir(parents=True)
    write_json_atomic(d / "meta.json", {"title": "old", "source": SOURCE})
    done = threading.Event()

    def job():
        with meta_lock(d):
            meta = read_json(d / "meta.json")
            time.sleep(0.3)  # a rename arrives meanwhile and must wait
            write_json_atomic(d / "meta.json", {**meta, "dropped_speech": {"total_s": 0}})
        done.set()

    t = threading.Thread(target=job)
    t.start()
    time.sleep(0.05)
    update_meta(d, title="new")
    t.join()
    assert done.is_set()
    assert read_json(d / "meta.json") == {"title": "new", "source": SOURCE, "dropped_speech": {"total_s": 0}}
    assert not (tmp_path / "sessions" / "2026-10-01_09-00-00" / ".meta.lock").exists()  # not in the lecture folder
    with pytest.raises(FileNotFoundError):  # never recreates a deleted lecture's folders
        update_meta(tmp_path / "gone" / "sessions" / "x", title="y")
    assert not (tmp_path / "gone").exists()


# ------------------------------------------------------------------ 2. naming template
VALUES = {"class": "Bio 101", "date": "2026-10-05", "time": "09:30", "n": "3", "original": "Voice Memo 007"}


@pytest.mark.parametrize("template, title", [
    ("{class} {date} Lecture {n}", "Bio 101 2026-10-05 Lecture 3"),
    ("{date} {time} · {original}", "2026-10-05 09:30 · Voice Memo 007"),
    ("Vorlesung {n} — {class} 🎓", "Vorlesung 3 — Bio 101 🎓"),
    ("{{{n}}} literal braces", "{3} literal braces"),
    ("{n}{n}", "33"),
    ("No placeholders", "No placeholders"),
])
def test_template_rendering(template, title):
    assert naming.check_template(template) == template
    assert naming.render(template, VALUES) == title


def test_template_edge_cases():
    empty = {**VALUES, "original": ""}
    assert naming.render("{class} - {original}", empty) == "Bio 101"  # no dangling separator
    assert naming.render("{original}", empty) is None  # empty: the built-in name is used
    assert naming.render("{original}  —  {class}", empty) == "Bio 101"
    assert naming.render("{original}", {**VALUES, "original": "a\tb\nc\u202e\udc80"}) == "abc"  # file names can't inject
    assert len(naming.render("{original}" * 3, {**VALUES, "original": "x" * 150})) == naming.MAX_TITLE
    assert naming.check_template("  {class}  ") == "{class}" and naming.check_template("") == ""


@pytest.mark.parametrize("template, why", [
    ("{klass} {n}", "Unknown placeholder {klass}"), ("{}", "Empty placeholder"), ("{0}", "Unknown placeholder {0}"),
    ("Lecture {n", "has no partner"), ("Lecture n}", "has no partner"), ("{n:03}", "without anything after"),
    ("{n!r}", "without anything after"), ("a\nb", "control characters"), ("x" * 201, "too long"), (7, "must be text"),
])
def test_invalid_templates(template, why):
    with pytest.raises(naming.NamingError, match=why):
        naming.check_template(template)


def test_n_counts_lectures_in_the_class_by_recording_date(cfg, cls):
    sessions = cls / "sessions"
    for name in ("2026-08-30_09-00-00", "2026-09-02_14-00-00", "2026-09-09_09-00-00"):
        (sessions / name).mkdir()
    # existing: 08-30, S1 09-01, 09-02, S2 09-03, 09-09 -> an import recorded on 09-02 at 15:00 is #4
    d = sessions / "2026-09-02_15-00-00"
    d.mkdir()
    v = naming.values_for(d, "Bio and Econ", "My Recording.final.m4a")
    assert v == {"class": "Bio and Econ", "date": "2026-09-02", "time": "15:00", "n": "4", "original": "My Recording.final"}
    same_second = sessions / "2026-09-02_15-00-00-2"
    same_second.mkdir()
    assert naming.values_for(same_second, "x")["n"] == "5"
    assert naming.title_for("{class} Lecture {n}", d) == "Bio and Econ Lecture 4"  # class name from class.json
    assert naming.title_for("", d) is None


def test_template_in_config_and_settings(cfg, cls, client):
    (cfg.home / "config.toml").write_text('[naming]\ntemplate = "{klass}"\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[naming\] template: Unknown placeholder"):
        load_config(cfg.home)
    (cfg.home / "config.toml").write_text("[processing]\nmode = \"manual\"\n", encoding="utf-8")
    assert load_config(cfg.home).naming.template == ""

    bad = client.put("/api/settings", headers=H, json={"naming_template": "Lecture {n"})
    assert bad.status_code == 400 and "no partner" in bad.json()["detail"]
    assert "naming" not in tomllib.loads((cfg.home / "config.toml").read_text(encoding="utf-8"))  # not saved
    ok = client.put("/api/settings", headers=H, json={"naming_template": '  {class} "{date}" \\ Lecture {n} '})
    assert ok.status_code == 200 and ok.json()["naming_template"] == '{class} "{date}" \\ Lecture {n}'
    saved = tomllib.loads((cfg.home / "config.toml").read_text(encoding="utf-8"))
    assert saved["naming"]["template"] == '{class} "{date}" \\ Lecture {n}' and saved["processing"]["mode"] == "manual"
    assert load_config(cfg.home).naming.template == '{class} "{date}" \\ Lecture {n}'
    assert client.put("/api/settings", headers=H, json={"naming_template": ""}).json()["naming_template"] == ""


def test_preview_api(client, cls):
    r = client.get("/api/naming/preview", params={"template": "{class} {date} Lecture {n} {original}"}).json()
    today = time.strftime("%Y-%m-%d")
    assert r["valid"] and r["error"] is None
    assert [x["title"] for x in r["examples"]] == [f"Bio and Econ {today} Lecture 3", f"Bio and Econ {today} Lecture 3 Voice Memo 12"]
    bad = client.get("/api/naming/preview", params={"template": "{klass}"}).json()
    assert bad == {"valid": False, "error": bad["error"], "examples": []} and "{klass}" in bad["error"]
    assert client.get("/api/naming/preview", params={"template": ""}).json() == {"valid": True, "error": None, "examples": []}
    only_original = client.get("/api/naming/preview", params={"template": "{original}"}).json()["examples"]
    assert only_original[0]["title"].startswith(today[:4]) and only_original[1]["title"] == "Voice Memo 12"


# ------------------------------------------------------------------ applying the template (import, Zoom, record)
def import_wav(client, tmp_path, filename, ms):
    write_tone_wav(tmp_path / "x.wav", seconds=1)
    r = client.post("/api/import", headers=H, data={"class_slug": SLUG, "last_modified_ms": str(ms), "anyway": "true"},  # same tone each time
                    files={"file": (filename, (tmp_path / "x.wav").read_bytes(), "audio/wav")})
    assert r.status_code == 201, r.text
    return read_json(Path(client.app.state.config.data_dir) / "classes" / SLUG / "sessions" / r.json()["session"] / "meta.json")


def test_import_applies_the_template_and_empty_keeps_file_names(cfg, cls, tmp_path):
    ms = int(time.mktime((2026, 9, 2, 10, 15, 0, 0, 0, -1)) * 1000)  # between S1 (09-01) and S2 (09-03)
    with client_for(cfg) as c:  # no template: today's behavior
        assert import_wav(c, tmp_path, "Phone memo.wav", ms)["title"] == "Phone memo"
    templated = dataclasses.replace(cfg, naming=NamingConfig("{class} {date} {time} Lecture {n} ({original})"))
    with client_for(templated) as c:
        meta = import_wav(c, tmp_path, "Recording 5.wav", ms + 60_000)
    assert meta["title"] == "Bio and Econ 2026-09-02 10:16 Lecture 3 (Recording 5)"  # S1, the 10:15 import, this
    assert meta["source"]["file"] == "Recording 5.wav"
    assert read_json(cls / "sessions" / S2 / "meta.json")["title"] == "Lecture 03"  # existing titles never change


def test_zoom_import_applies_the_template(cfg, cls, tmp_path, monkeypatch):
    from lecture_recorder import cli
    from test_app_capture import zoom_folder

    f = zoom_folder(tmp_path, m4a=True)
    templated = dataclasses.replace(cfg, naming=NamingConfig("{class} {date} Lecture {n} [{original}]"))
    monkeypatch.setattr("lecture_recorder.config.load_config", lambda *a: templated)
    assert cli.main(["import-zoom", str(f), "--class", SLUG]) == 0
    meta = read_json(cls / "sessions" / "2026-10-20_14-00-05" / "meta.json")
    assert meta["title"] == "Bio and Econ 2026-10-20 Lecture 3 [audio1234567890]"
    assert meta["source"]["file"] == "audio1234567890.m4a" and meta["source"]["zoom"] is True  # original file kept


class FakeProc:
    returncode = 1

    def __init__(self, *a, **k):
        pass

    def poll(self):
        return 1


def test_record_applies_the_template(cfg, cls, monkeypatch):
    monkeypatch.setattr(control.subprocess, "Popen", FakeProc)  # writes meta.json, then the child "fails"
    templated = dataclasses.replace(cfg, naming=NamingConfig("{class} {date} Lecture {n} {original}"))
    with pytest.raises(control.RecorderError):
        control.start_recording(SLUG, config=templated, synthetic={"pattern": "tone"})
    with pytest.raises(control.RecorderError):  # no template: named by date and time, as before
        control.start_recording(SLUG, config=cfg, synthetic={"pattern": "tone"})
    new = sorted(p for p in (cls / "sessions").iterdir() if p.name not in (S1, S2))
    first, second = (read_json(p / "meta.json") for p in new)
    assert first["title"] == f"Bio and Econ {new[0].name[:10]} Lecture 3"  # no original file: nothing dangling
    assert second["title"] == new[1].name and "file" not in second["source"]


# ------------------------------------------------------------------ 3. original file on the lecture page
def test_original_file_shown_for_imports_only(client, cls):
    assert client.get(f"/api/sessions/{SLUG}/{S2}").json()["original"] == "Voice Memo 007.m4a"
    assert client.get(f"/api/sessions/{SLUG}/{S1}").json()["original"] is None
    listed = {s["name"]: s["original"] for s in client.get(f"/api/classes/{SLUG}").json()["sessions"]}
    assert listed == {S1: None, S2: "Voice Memo 007.m4a"}


def test_ui_has_rename_template_and_original_file():
    js = (Path(lecture_recorder.__file__).parent / "web" / "static" / "app.js").read_text(encoding="utf-8")
    for needle in ("/rename`", "renameButton(slug, s, titleEl)", "Original file: ", "#lecture-title", "session-filter",
                   "/api/naming/preview?template=", "naming_template", "data-transient"):
        assert needle in js, needle
