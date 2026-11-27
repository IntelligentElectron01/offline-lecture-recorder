"""Delete lectures and classes (Recently deleted, restore, delete permanently), the terminal
recorder's class check, and folder-only classes. Temp folders only; never the real data/ folder.

The class is the units test's: three lectures, unit 2 starting at 90 s of the second one.
"""
import dataclasses
import hashlib
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import lecture_recorder  # noqa: F401  (lockdown)
from lecture_recorder import library as lib
from lecture_recorder import trash
from lecture_recorder import units as un
from lecture_recorder.config import RetentionConfig, load_config
from lecture_recorder.pipeline import queue as q
from lecture_recorder.recorder.control import RecorderError, start_recording
from lecture_recorder.recorder.platform import ProcessLock
from lecture_recorder.recorder.process import GLOBAL_LOCK
from lecture_recorder.recorder.session import read_json, write_json_atomic
from lecture_recorder.search.index import SearchIndex
from lecture_recorder.summary.exam import EXAM_MD, EXAM_META, _scope_key, material_is_stale
from lecture_recorder.web.app import create_app
from test_search import embed_all, hashed_vector
from test_units import BIO, BOUNDARY, ECON, S1, S2, S3, SLUG, U1_TERMS, U2_TERMS, make_lecture, sentences

BASE = "http://127.0.0.1:8765"
H = {"X-Lecture-Recorder": "1"}


@pytest.fixture
def cfg(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    return dataclasses.replace(load_config(home), data_dir=(home / "data").resolve())


@pytest.fixture
def cls(cfg):
    c = cfg.data_dir / "classes" / SLUG
    c.mkdir(parents=True)
    write_json_atomic(c / "class.json", {"name": "Bio and Econ", "slug": SLUG})
    make_lecture(cfg.data_dir, S1, sentences(BIO, U1_TERMS, 18))
    make_lecture(cfg.data_dir, S2, sentences(BIO, U1_TERMS, 9, offset=1) + sentences(ECON, U2_TERMS, 9))
    make_lecture(cfg.data_dir, S3, sentences(ECON, U2_TERMS, 18, offset=2))
    (c / "sessions" / S2 / "audio.wav").write_bytes(os.urandom(4096))  # binary content for the round trip
    return c


@pytest.fixture
def idx(cfg):
    return SearchIndex(cfg.data_dir, cfg)


@pytest.fixture
def client(cfg):
    with TestClient(create_app(cfg, autoprocess=False, monitor=False), base_url=BASE) as c:
        yield c


def tree_hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def junction(link: Path, target: Path) -> None:
    """A Windows junction, or a symlink on macOS/Linux: either way a folder that points somewhere else."""
    if os.name != "nt":
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as e:
            pytest.skip(f"could not create a symlink: {e}")
        return
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
    if r.returncode != 0 or not link.is_junction():
        pytest.skip(f"could not create a junction: {r.stdout}{r.stderr}")


# ------------------------------------------------------------------ move and restore
def test_delete_and_restore_a_lecture_round_trip_is_byte_identical(cfg, cls, idx):
    d = cls / "sessions" / S2
    before = tree_hashes(d)
    r = trash.delete_session(cfg.data_dir, SLUG, S2, index=idx)
    assert not d.exists() and r["kind"] == "session" and r["title"] == "Lecture 03"
    item = cfg.data_dir / trash.TRASH_DIR / r["id"]
    assert tree_hashes(item / trash.ITEM) == before  # moved as is
    assert read_json(item / trash.RECORD)["path"] == f"classes/{SLUG}/sessions/{S2}"
    assert [x["id"] for x in trash.list_trash(cfg.data_dir)] == [r["id"]]
    assert [s["name"] for s in lib.list_sessions(cfg.data_dir, SLUG)] == [S3, S1]  # the class view updates

    back = trash.restore(cfg.data_dir, r["id"], index=idx)
    assert back == {"kind": "session", "class": SLUG, "session": S2, "requeued": 0}
    assert tree_hashes(d) == before
    assert not item.exists() and trash.list_trash(cfg.data_dir) == []


def test_delete_and_restore_a_class_with_lectures(cfg, cls, idx):
    before = tree_hashes(cls)
    with pytest.raises(trash.Conflict):  # the user confirmed a different count: nothing happens
        trash.delete_class(cfg.data_dir, SLUG, expected_sessions=2, index=idx)
    assert tree_hashes(cls) == before
    r = trash.delete_class(cfg.data_dir, SLUG, expected_sessions=3, index=idx)
    assert r["sessions"] == 3 and not cls.exists() and lib.list_classes(cfg.data_dir) == []
    assert idx.exact("mitochondrion")["total"] == 0
    trash.restore(cfg.data_dir, r["id"], index=idx)
    assert tree_hashes(cls) == before and idx.exact("mitochondrion")["total"] > 0


def test_empty_class_delete(cfg):
    lib.create_class(cfg.data_dir, "Chemistry")
    r = trash.delete_class(cfg.data_dir, "chemistry", expected_sessions=0)
    assert r["sessions"] == 0 and lib.list_classes(cfg.data_dir) == []


def test_restore_refuses_an_occupied_place_or_a_missing_class(cfg, cls):
    r = trash.delete_session(cfg.data_dir, SLUG, S1)
    (cls / "sessions" / S1).mkdir()  # something new at the original place
    (cls / "sessions" / S1 / "note.txt").write_text("new", encoding="utf-8")
    assert trash.list_trash(cfg.data_dir)[0]["occupied"] is True
    with pytest.raises(trash.Conflict):
        trash.restore(cfg.data_dir, r["id"])
    assert (cls / "sessions" / S1 / "note.txt").read_text(encoding="utf-8") == "new"  # never replaced
    os.remove(cls / "sessions" / S1 / "note.txt")
    (cls / "sessions" / S1).rmdir()

    c = trash.delete_class(cfg.data_dir, SLUG)
    with pytest.raises(trash.Conflict, match="Restore the class first"):
        trash.restore(cfg.data_dir, r["id"])
    lib.create_class(cfg.data_dir, "Bio Econ")  # a new class took the slug
    with pytest.raises(trash.Conflict):
        trash.restore(cfg.data_dir, c["id"])
    assert {x["id"] for x in trash.list_trash(cfg.data_dir)} == {r["id"], c["id"]}  # both kept


# ------------------------------------------------------------------ busy
def test_busy_lectures_are_refused(cfg, cls):
    d = cls / "sessions" / S1
    before = tree_hashes(cls)
    lock = ProcessLock(d / "recorder.lock")  # a live recorder holds this
    assert lock.acquire()
    try:
        with pytest.raises(trash.Busy, match="recording"):
            trash.delete_session(cfg.data_dir, SLUG, S1)
        with pytest.raises(trash.Busy, match="recording"):
            trash.delete_class(cfg.data_dir, SLUG)
    finally:
        lock.release()

    g = ProcessLock(cfg.data_dir / GLOBAL_LOCK)  # a recording is starting: the folder exists, its lock not yet
    assert g.acquire()
    try:
        meta = read_json(d / "meta.json")
        write_json_atomic(d / "meta.json", {**meta, "state": "starting"})
        with pytest.raises(trash.Busy, match="starting"):
            trash.delete_session(cfg.data_dir, SLUG, S1)
        write_json_atomic(d / "meta.json", meta)
        trash.restore(cfg.data_dir, trash.delete_session(cfg.data_dir, SLUG, S3)["id"])  # another lecture is fine
    finally:
        g.release()

    job = q.enqueue(cfg.data_dir, d)
    job.update(state="running")
    q._save(job)
    with pytest.raises(trash.Busy, match="processing job is running"):
        trash.delete_session(cfg.data_dir, SLUG, S1)
    with pytest.raises(trash.Busy, match="processing job is running"):
        trash.delete_class(cfg.data_dir, SLUG)
    after = {k: v for k, v in tree_hashes(cls).items() if not k.endswith("recorder.lock")}  # the test's own lock file
    assert after == before and trash.list_trash(cfg.data_dir) == []


def test_files_in_use_leave_everything_as_it_was(cfg, cls, monkeypatch):
    def denied(src, dst):
        raise PermissionError("in use")

    monkeypatch.setattr(trash.os, "rename", denied)
    with pytest.raises(trash.Busy, match="in use"):
        trash.delete_session(cfg.data_dir, SLUG, S1)
    assert (cls / "sessions" / S1).is_dir() and trash.list_trash(cfg.data_dir) == []
    assert [p.name for p in (cfg.data_dir / trash.TRASH_DIR).iterdir()] == [trash.VERSION]  # no item left behind


# ------------------------------------------------------------------ queue, index, units, exams
def test_queued_jobs_are_cancelled_never_started_and_requeued_on_restore(cfg, cls, monkeypatch):
    d1, d3 = cls / "sessions" / S1, cls / "sessions" / S3
    q.enqueue(cfg.data_dir, d1, q.SUMMARIZE)
    q.enqueue(cfg.data_dir, d1, q.EMBED)
    q.enqueue(cfg.data_dir, d3, q.SUMMARIZE)
    r = trash.delete_session(cfg.data_dir, SLUG, S1)
    states = {(Path(j["session"]).name, j["kind"]): j["state"] for j in q.jobs(cfg.data_dir)}
    assert states == {(S1, "summarize"): "cancelled", (S1, "embed"): "cancelled", (S3, "summarize"): "queued"}

    ran = []

    def fake_run(cmd, **kw):
        ran.append(Path(cmd[3]).name)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(q, "_run_job", fake_run)
    monkeypatch.setattr(q, "gpu_block_reason", lambda data_dir: None)
    q.run_worker(cfg.data_dir)
    assert ran == [S3, S3]  # S3's summary and the embed it queued; nothing for the deleted lecture

    assert trash.restore(cfg.data_dir, r["id"])["requeued"] == 2
    active = sorted(j["kind"] for j in q.jobs(cfg.data_dir) if j["state"] == "queued" and Path(j["session"]).name == S1)
    assert active == ["embed", "summarize"]


def test_worker_never_starts_a_job_cancelled_after_it_was_listed(cfg, cls, monkeypatch):
    d = cls / "sessions" / S1
    q.enqueue(cfg.data_dir, d, q.SUMMARIZE)
    real_pending = q._pending

    def pending_then_delete(data_dir):
        jobs = real_pending(data_dir)
        if jobs and (cls / "sessions" / S1).exists():
            trash.delete_session(cfg.data_dir, SLUG, S1)  # the delete lands between listing and starting
        return jobs

    monkeypatch.setattr(q, "_pending", pending_then_delete)
    monkeypatch.setattr(q, "gpu_block_reason", lambda data_dir: None)
    monkeypatch.setattr(q, "_run_job", lambda *a, **k: pytest.fail("a cancelled job was started"))
    q.run_worker(cfg.data_dir)
    assert [j["state"] for j in q.jobs(cfg.data_dir)] == ["cancelled"]


def test_search_index_forgets_and_relearns_a_deleted_lecture(cfg, cls, idx):
    embed_all(cfg, [cls / "sessions" / n for n in (S1, S2, S3)])
    qvec = np.array(hashed_vector("tariff inflation monopoly market prices"), dtype=np.float32)
    assert {x["session"] for x in idx.exact("tariff")["results"]} == {S2, S3}
    assert S3 in {x["session"] for x in idx.concept(qvec)}
    r = trash.delete_session(cfg.data_dir, SLUG, S3, index=idx)
    with idx._db() as con:  # removed from app.db right away, not only at the next search
        assert con.execute("SELECT COUNT(*) FROM sessions WHERE name=?", (S3,)).fetchone()[0] == 0
    assert {x["session"] for x in idx.exact("tariff")["results"]} == {S2}
    assert S3 not in {x["session"] for x in idx.concept(qvec)}
    assert idx.concept_status()["transcribed"] == 2
    trash.restore(cfg.data_dir, r["id"], index=idx)
    assert {x["session"] for x in idx.exact("tariff")["results"]} == {S2, S3}
    assert S3 in {x["session"] for x in idx.concept(qvec)} and idx.concept_status()["indexed"] == 3


def test_units_boundaries_and_exam_material_follow_delete_and_restore(cfg, cls, idx):
    u2 = un.add_boundary(cls, S2, BOUNDARY, "Markets")
    u3 = un.add_boundary(cls, S3, 0)
    units_before = un.load_units(cls)
    exam = un.save_exam(cls, "Final", [u2["id"]])
    ed = un.exam_dir(cls, exam["id"])
    ed.mkdir(parents=True)
    (ed / EXAM_MD).write_text("# My own notes\n", encoding="utf-8")  # the user's exam.md
    write_json_atomic(ed / EXAM_META, {"state": "done", "sha256": "not-this-file",
                                       "spans": _scope_key(un.scope_spans(cls, exam))})
    assert not material_is_stale(ed, cls, exam)

    r = trash.delete_session(cfg.data_dir, SLUG, S2, index=idx)
    ids = [u["id"] for u in un.load_units(cls)]
    assert u2["id"] not in ids and u3["id"] in ids  # the boundary in the deleted lecture is gone
    assert [u["id"] for u in r["units_removed"]] == [u2["id"]]
    assert all(sp["session"] != S2 for u in idx.units(SLUG) for sp in u["spans"])
    assert material_is_stale(ed, cls, exam)  # shown as out of date
    assert (ed / EXAM_MD).read_text(encoding="utf-8") == "# My own notes\n"  # never rewritten

    trash.restore(cfg.data_dir, r["id"], index=idx)
    assert un.load_units(cls) == units_before  # names and boundaries exactly as before
    assert not material_is_stale(ed, cls, exam)
    assert any(sp["session"] == S2 and sp["start"] == BOUNDARY for u in idx.units(SLUG) for sp in u["spans"])


# ------------------------------------------------------------------ path safety
def test_names_with_path_tricks_are_refused(cfg, cls, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("x", encoding="utf-8")
    for slug, name in (("..", S1), (SLUG, ".."), ("../..", "outside"), (SLUG, "..\\..\\x"), (SLUG, S1 + "/../..")):
        with pytest.raises(trash.NotFound):
            trash.delete_session(cfg.data_dir, slug, name)
    for slug in ("..", ".trash", "a/../..", ""):
        with pytest.raises(trash.NotFound):
            trash.delete_class(cfg.data_dir, slug)
    for tid in ("..", "../../outside", "20260101-000000-abcdef12/../.."):
        with pytest.raises(trash.NotFound):
            trash.restore(cfg.data_dir, tid)
        with pytest.raises(trash.NotFound):
            trash.delete_permanently(cfg.data_dir, tid, "x")
    assert (outside / "keep.txt").exists() and len(tree_hashes(cls)) > 0


def test_junctions_are_never_moved_or_followed(cfg, cls, tmp_path):
    outside = tmp_path / "outside"
    (outside / "sessions" / S1).mkdir(parents=True)
    (outside / "sessions" / S1 / "keep.txt").write_text("x", encoding="utf-8")
    junction(cfg.data_dir / "classes" / "evil", outside)  # a class folder that points outside data/
    with pytest.raises(trash.TrashError, match="link"):
        trash.delete_class(cfg.data_dir, "evil")
    with pytest.raises(trash.TrashError, match="link"):
        trash.delete_session(cfg.data_dir, "evil", S1)
    junction(cls / "sessions" / "2026-09-09_09-00-00", outside / "sessions" / S1)  # a lecture folder link
    with pytest.raises(trash.TrashError, match="link"):
        trash.delete_session(cfg.data_dir, SLUG, "2026-09-09_09-00-00")
    assert (outside / "sessions" / S1 / "keep.txt").exists()


def test_permanent_delete_removes_a_link_inside_but_not_its_target(cfg, cls, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("x", encoding="utf-8")
    junction(cls / "sessions" / S1 / "linked", outside)  # inside a lecture folder
    r = trash.delete_session(cfg.data_dir, SLUG, S1)
    trash.delete_permanently(cfg.data_dir, r["id"], "Lecture 01")
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "x"


def test_a_tampered_record_is_refused(cfg, cls, tmp_path):
    r = trash.delete_session(cfg.data_dir, SLUG, S1)
    rec_path = cfg.data_dir / trash.TRASH_DIR / r["id"] / trash.RECORD
    rec = read_json(rec_path)
    for bad in ({"path": "../../outside"}, {"class": "..", "path": "classes/../sessions/x"}, {"session": "..\\.."}):
        write_json_atomic(rec_path, {**rec, **bad})
        with pytest.raises(trash.TrashError):
            trash.restore(cfg.data_dir, r["id"])
    assert not (tmp_path / "outside").exists()


# ------------------------------------------------------------------ permanent deletion
def test_permanent_delete_needs_the_exact_name(cfg, cls):
    r = trash.delete_session(cfg.data_dir, SLUG, S1)
    for wrong in ("", "lecture 01", "Lecture 0", None, "DELETE"):
        with pytest.raises(trash.TrashError, match="Type the name"):
            trash.delete_permanently(cfg.data_dir, r["id"], wrong)
    assert (cfg.data_dir / trash.TRASH_DIR / r["id"] / trash.ITEM).is_dir()
    trash.delete_permanently(cfg.data_dir, r["id"], " Lecture 01 ")
    assert trash.list_trash(cfg.data_dir) == []
    with pytest.raises(trash.NotFound):  # idempotent: a second request changes nothing
        trash.delete_permanently(cfg.data_dir, r["id"], "Lecture 01")


def test_nothing_is_purged_unless_turned_on(cfg, cls):
    r = trash.delete_session(cfg.data_dir, SLUG, S1)
    rec_path = cfg.data_dir / trash.TRASH_DIR / r["id"] / trash.RECORD
    old = (datetime.now().astimezone() - timedelta(days=40)).isoformat(timespec="seconds")
    write_json_atomic(rec_path, {**read_json(rec_path), "deleted": old})
    assert RetentionConfig().trash_days == 0 and trash.purge_expired(cfg.data_dir, 0) == 0
    assert trash.purge_expired(cfg.data_dir, 60) == 0  # not old enough
    trash.delete_session(cfg.data_dir, SLUG, S3)  # deleted now: kept
    assert trash.purge_expired(cfg.data_dir, 30) == 1
    assert [x["session"] for x in trash.list_trash(cfg.data_dir)] == [S3]
    with pytest.raises(ValueError):
        RetentionConfig(trash_days=-1)


# ------------------------------------------------------------------ API
def test_api_delete_restore_and_permanent_delete(cfg, cls, client):
    assert client.get(f"/api/sessions/{SLUG}/{S1}/delete").status_code == 405  # POST only
    assert client.post(f"/api/sessions/{SLUG}/{S1}/delete").status_code == 403  # app header required
    assert client.post(f"/api/sessions/{SLUG}/{S1}/delete", headers={**H, "Origin": "http://evil.example"}).status_code == 403
    assert client.post(f"/api/sessions/{SLUG}/{S1}/delete", headers={**H, "Host": "evil.example"}).status_code == 400
    assert (cls / "sessions" / S1).is_dir()

    r = client.post(f"/api/sessions/{SLUG}/{S1}/delete", headers=H)
    assert r.status_code == 200 and r.json()["kind"] == "session"
    assert client.post(f"/api/sessions/{SLUG}/{S1}/delete", headers=H).status_code == 404  # already deleted
    assert S1 not in [s["name"] for s in client.get(f"/api/classes/{SLUG}").json()["sessions"]]
    assert client.get("/api/search", params={"q": "mitochondrion"}).json()["sessions"] == 1  # S2 only
    items = client.get("/api/trash").json()["items"]
    assert [(x["kind"], x["title"], x["occupied"]) for x in items] == [("session", "Lecture 01", False)]
    tid = items[0]["id"]
    assert client.post(f"/api/trash/{tid}/restore", headers=H).json()["session"] == S1
    assert client.post(f"/api/trash/{tid}/restore", headers=H).status_code == 404  # idempotent
    assert client.get("/api/search", params={"q": "mitochondrion"}).json()["sessions"] == 2

    tid = client.post(f"/api/sessions/{SLUG}/{S1}/delete", headers=H).json()["id"]
    assert client.post(f"/api/trash/{tid}/delete", headers=H, json={"confirm": "nope"}).status_code == 400
    assert client.post(f"/api/trash/{tid}/delete", headers=H, json={"confirm": "Lecture 01"}).status_code == 200
    assert client.get("/api/trash").json()["items"] == []


def test_api_class_delete_needs_the_confirmed_count_and_refuses_busy(cfg, cls, client):
    assert client.post(f"/api/classes/{SLUG}/delete", headers=H, json={}).status_code == 400
    assert client.post(f"/api/classes/{SLUG}/delete", headers=H, json={"sessions": 1}).status_code == 409
    job = q.enqueue(cfg.data_dir, cls / "sessions" / S2)
    job.update(state="running")
    q._save(job)
    r = client.post(f"/api/classes/{SLUG}/delete", headers=H, json={"sessions": 3})
    assert r.status_code == 409 and "running" in r.json()["detail"]
    r = client.post(f"/api/sessions/{SLUG}/{S2}/delete", headers=H)
    assert r.status_code == 409 and "running" in r.json()["detail"]
    job.update(state="done")
    q._save(job)
    assert client.post(f"/api/classes/{SLUG}/delete", headers=H, json={"sessions": 3}).json()["sessions"] == 3
    assert client.get("/api/classes").json() == []


def test_settings_emptying_needs_a_permanent_deletion_confirmation(cfg, client):
    assert client.put("/api/settings", headers=H, json={"trash_days": 30}).status_code == 400
    assert client.put("/api/settings", headers=H, json={"trash_days": 30, "confirm_permanent_deletion": True}).json()["trash_days"] == 30
    assert load_config(cfg.home).retention.trash_days == 30
    assert client.put("/api/settings", headers=H, json={"trash_days": "30"}).status_code == 400
    assert client.put("/api/settings", headers=H, json={"trash_days": 0}).json()["trash_days"] == 0


# ------------------------------------------------------------------ C: terminal recording and folder-only classes
def test_recording_into_an_unknown_class_is_refused_and_creates_nothing(cfg, monkeypatch):
    with pytest.raises(RecorderError, match="Create it in the app first"):
        start_recording("physics-1o1", config=cfg, synthetic={"pattern": "tone"})
    assert not (cfg.data_dir / "classes").exists()
    from lecture_recorder import cli

    monkeypatch.setattr("lecture_recorder.config.load_config", lambda *a, **k: cfg)
    assert cli.main(["record", "--class", "physics-1o1"]) == 2
    assert not (cfg.data_dir / "classes").exists()
    with pytest.raises(SystemExit):
        cli.main(["record"])  # --class is required: no silent "unsorted" class


def test_a_class_folder_without_class_json_is_listed_and_can_be_registered(cfg, client):
    d = cfg.data_dir / "classes" / "old-terminal" / "sessions" / S1
    d.mkdir(parents=True)
    (d / "audio.wav").write_bytes(b"RIFF")
    c = client.get("/api/classes").json()
    assert [(x["slug"], x["has_class_file"], x["sessions"]) for x in c] == [("old-terminal", False, 1)]
    page = client.get("/api/classes/old-terminal").json()
    assert page["has_class_file"] is False and [s["name"] for s in page["sessions"]] == [S1]
    assert client.post("/api/classes/old-terminal/register", headers=H).json()["has_class_file"] is True
    assert client.post("/api/classes/old-terminal/register", headers=H).status_code == 400
    assert (d / "audio.wav").read_bytes() == b"RIFF"


def test_cli_delete_restore_and_trash(cfg, cls, monkeypatch, capsys):
    from lecture_recorder import cli

    dd = ["--data-dir", str(cfg.data_dir)]
    assert cli.main(["delete", SLUG, S1, *dd]) == 0
    tid = trash.list_trash(cfg.data_dir)[0]["id"]
    assert f"lecture-recorder restore {tid}" in capsys.readouterr().out
    assert cli.main(["delete", SLUG, S1, *dd]) == 2  # already deleted
    assert cli.main(["trash", "list", *dd]) == 0 and tid in capsys.readouterr().out
    assert cli.main(["restore", tid, *dd]) == 0 and (cls / "sessions" / S1).is_dir()
    assert cli.main(["delete", SLUG, *dd]) == 0 and not cls.exists()
    monkeypatch.setattr("builtins.input", lambda prompt="": "delete")
    assert cli.main(["trash", "empty", *dd]) == 1 and len(trash.list_trash(cfg.data_dir)) == 1
    monkeypatch.setattr("builtins.input", lambda prompt="": "DELETE")
    assert cli.main(["trash", "empty", *dd]) == 0 and trash.list_trash(cfg.data_dir) == []


def test_measure_resources_never_writes_into_data_by_default():
    text = (Path(__file__).resolve().parents[1] / "scripts" / "bench" / "measure_resources.py").read_text(encoding="utf-8")
    assert 'WORK = ROOT / ".cache" / "measure"' in text and "default=WORK" in text and '"data" / "benchmarks"' not in text


# ------------------------------------------------------------------ regressions
def test_races_around_delete_are_closed(cfg, cls):
    d = cls / "sessions" / S1
    meta = read_json(d / "meta.json")
    write_json_atomic(d / "meta.json", {**meta, "state": "starting", "created": datetime.now().astimezone().isoformat()})
    with pytest.raises(trash.Busy, match="starting"):  # its recorder has not taken any lock yet
        trash.delete_session(cfg.data_dir, SLUG, S1)
    write_json_atomic(d / "meta.json", {**meta, "state": "starting", "created": "2026-01-01T09:00:00+00:00"})
    r = trash.delete_session(cfg.data_dir, SLUG, S1)  # an old crashed start is not busy

    with pytest.raises(FileNotFoundError):  # a finished job's follow-up can't queue work for it
        q.enqueue(cfg.data_dir, d, q.SUMMARIZE)
    q._follow_up(cfg.data_dir, {"session": str(d), "kind": q.TRANSCRIBE}, ok=True)
    assert q.jobs(cfg.data_dir) == []

    other = trash.delete_session(cfg.data_dir, SLUG, S3)
    assert trash.empty_trash(cfg.data_dir, [r["id"]]) == 1  # only what the user was shown
    assert [x["id"] for x in trash.list_trash(cfg.data_dir)] == [other["id"]]


def test_a_lecture_is_not_restored_into_a_new_class_with_the_same_name(cfg, cls):
    r = trash.delete_session(cfg.data_dir, SLUG, S1)
    c = trash.delete_class(cfg.data_dir, SLUG)
    cls.mkdir()
    write_json_atomic(cls / "class.json", {"name": "Bio and Econ", "slug": SLUG, "created": "2026-10-13T10:00:00+00:00"})
    with pytest.raises(trash.Conflict, match="created again"):
        trash.restore(cfg.data_dir, r["id"])
    assert not (cls / "sessions" / S1).exists()
    os.remove(cls / "class.json")
    cls.rmdir()
    trash.restore(cfg.data_dir, c["id"])
    assert trash.restore(cfg.data_dir, r["id"])["session"] == S1


def test_cancelled_jobs_are_kept_in_the_record_even_if_units_fail(cfg, cls, monkeypatch):
    q.enqueue(cfg.data_dir, cls / "sessions" / S1, q.SUMMARIZE)

    def locked(*a):
        raise PermissionError("units.json in use")

    monkeypatch.setattr(un, "drop_session_boundaries", locked)
    with pytest.raises(PermissionError):
        trash.delete_session(cfg.data_dir, SLUG, S1)
    tid = trash.list_trash(cfg.data_dir)[0]["id"]
    assert read_json(cfg.data_dir / trash.TRASH_DIR / tid / trash.RECORD)["cancelled_jobs"] == [
        {"kind": "summarize", "path": f"classes/{SLUG}/sessions/{S1}"}]
    assert trash.restore(cfg.data_dir, tid)["requeued"] == 1
