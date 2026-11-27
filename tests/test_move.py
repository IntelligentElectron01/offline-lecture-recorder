"""Moving a lecture between classes, the risky part. Refused while busy, never overwrites, everything that
points at the lecture stays right, Undo, and a real crash at every step. Synthetic lectures in temp folders."""
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import lecture_recorder  # noqa: F401  (lockdown)
from lecture_recorder import cli
from lecture_recorder import duplicates as dup
from lecture_recorder import library as lib
from lecture_recorder import move as mv
from lecture_recorder import trash
from lecture_recorder import units as un
from lecture_recorder.config import load_config
from lecture_recorder.pipeline import queue as q
from lecture_recorder.recorder.platform import ProcessLock
from lecture_recorder.recorder.session import LOCK_FILE, read_json, update_meta, write_json_atomic
from lecture_recorder.search.embed import EMB_META, EMB_NPY
from lecture_recorder.search.index import SearchIndex
from lecture_recorder.summary.exam import EXAM_META, _scope_key, material_is_stale
from lecture_recorder.web.app import create_app
from test_web_app import BASE, H

A, B = "physics-101", "chem-200"
ONE, TWO, THREE = "2026-09-01_09-00-00", "2026-09-08_09-00-00", "2026-09-15_09-00-00"


@pytest.fixture
def cfg(tmp_path):
    data = tmp_path / "data"
    lib.create_class(data, "Physics 101")
    lib.create_class(data, "Chem 200")
    return dataclasses.replace(load_config(), data_dir=data.resolve(), home=tmp_path.resolve())


@pytest.fixture
def client(cfg):
    with TestClient(create_app(cfg, autoprocess=False, monitor=False), base_url=BASE) as c:
        yield c


def lecture(cfg, name=ONE, slug=A, text="alpha beta gamma delta", **meta) -> Path:
    d = cfg.data_dir / "classes" / slug / "sessions" / name
    d.mkdir(parents=True)
    write_json_atomic(d / "meta.json", {"name": name, "title": meta.pop("title", f"Lecture {name}"), "class": slug,
                                        "created": f"{name[:10]}T09:00:00+00:00", "duration_s": 3000.0,
                                        "source": {"kind": "mic"}, **meta}, fsync=False)
    (d / "audio.wav").write_bytes(b"RIFF" + name.encode())
    doc = {"segments": [{"id": 0, "start": 0.0, "end": 3.0, "speaker": "Speaker 1", "role": "Main", "words": [], "text": text}],
           "speakers": [{"id": "Speaker 1", "role": "Main"}], "duration_s": 3000.0}
    write_json_atomic(d / "transcript.json", doc, fsync=False)
    write_json_atomic(d / "transcript.raw.json", doc, fsync=False)
    return d


def files(d: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir()) if p.name not in ("meta.json", LOCK_FILE)}


def at(cfg, slug, name) -> Path:
    return cfg.data_dir / "classes" / slug / "sessions" / name


def journals(cfg) -> list:
    return list((cfg.data_dir / "queue" / mv.MOVES_DIR).glob("*.json"))


# ------------------------------------------------------------------ the move itself
def test_a_move_renames_the_folder_changes_only_class_and_name_and_keeps_every_file(cfg):
    d = lecture(cfg)
    before, meta = files(d), read_json(d / "meta.json")
    r = mv.move_session(cfg.data_dir, A, ONE, B)
    new = at(cfg, B, ONE)
    assert not d.exists() and new.is_dir() and files(new) == before
    assert (r["class"], r["session"], r["from_class"], r["from_session"]) == (B, ONE, A, ONE)
    assert read_json(new / "meta.json") == {**meta, "class": B, "name": ONE, "class_settled": True}
    assert not journals(cfg)


def test_a_name_collision_takes_the_next_free_name_and_never_merges(cfg):
    src, taken = lecture(cfg, title="mine"), lecture(cfg, slug=B, title="theirs")
    lecture(cfg, f"{ONE}-2", slug=B, title="also taken")
    theirs = {**files(taken), "title": read_json(taken / "meta.json")["title"]}
    r = mv.move_session(cfg.data_dir, A, ONE, B)
    assert r["session"] == f"{ONE}-3" and not src.exists()
    assert read_json(at(cfg, B, f"{ONE}-3") / "meta.json")["title"] == "mine"
    assert {**files(taken), "title": read_json(taken / "meta.json")["title"]} == theirs


def test_undo_goes_back_to_the_original_class_and_name_and_clears_the_decision(cfg):
    d = lecture(cfg)
    lecture(cfg, slug=B)  # so the move gets another name
    meta, before = read_json(d / "meta.json"), files(d)
    r = mv.move_session(cfg.data_dir, A, ONE, B)
    assert r["session"] == f"{ONE}-2" and read_json(at(cfg, B, r["session"]) / "meta.json")["class_settled"] is True
    back = mv.move_session(cfg.data_dir, B, r["session"], A, prefer=ONE, undo=True)
    assert (back["class"], back["session"]) == (A, ONE) and files(d) == before
    assert read_json(d / "meta.json") == {**meta, "class_settled": False}


def test_undo_never_overwrites_a_lecture_that_took_the_original_place(cfg):
    lecture(cfg)
    mv.move_session(cfg.data_dir, A, ONE, B)
    lecture(cfg, title="new one")
    back = mv.move_session(cfg.data_dir, B, ONE, A, prefer=ONE, undo=True)
    assert back["session"] == f"{ONE}-2" and read_json(at(cfg, A, ONE) / "meta.json")["title"] == "new one"


@pytest.mark.parametrize("to,err", [(A, trash.Conflict), ("no-such-class", trash.NotFound), ("../..", trash.NotFound)])
def test_a_bad_target_is_refused_and_nothing_changes(cfg, to, err):
    d = lecture(cfg)
    before = files(d)
    with pytest.raises(err):
        mv.move_session(cfg.data_dir, A, ONE, to)
    assert files(d) == before and not journals(cfg)


def test_a_folder_without_class_settings_is_not_a_target_and_bad_names_are_refused(cfg):
    lecture(cfg)
    (cfg.data_dir / "classes" / "bare").mkdir()
    with pytest.raises(trash.NotFound):
        mv.move_session(cfg.data_dir, A, ONE, "bare")
    with pytest.raises(trash.NotFound):
        mv.move_session(cfg.data_dir, A, ONE, B, prefer="..\\evil")
    with pytest.raises(trash.NotFound):
        mv.move_session(cfg.data_dir, A, "..", B)
    assert at(cfg, A, ONE).is_dir()


def test_moving_to_unsorted_creates_it_on_first_use(cfg):
    lecture(cfg)
    r = mv.move_session(cfg.data_dir, A, ONE, lib.UNSORTED)
    assert r["class_name"] == "Unsorted" and lib.get_class(cfg.data_dir, lib.UNSORTED)["has_class_file"]


# ------------------------------------------------------------------ refused while busy
def assert_refused(cfg, d, match):
    before = files(d)
    with pytest.raises(trash.Busy, match=match):
        mv.move_session(cfg.data_dir, A, ONE, B)
    assert d.is_dir() and files(d) == before and not journals(cfg) and not at(cfg, B, ONE).exists()


def test_refused_while_recording(cfg):
    d = lecture(cfg)
    lock = ProcessLock(d / LOCK_FILE)
    assert lock.acquire()
    try:
        assert_refused(cfg, d, "recording")
    finally:
        lock.release()
    (d / LOCK_FILE).unlink(missing_ok=True)


def test_refused_while_its_recording_is_unsaved(cfg):
    d = lecture(cfg)
    (d / "audio.pcm").write_bytes(b"\0" * 8)
    assert_refused(cfg, d, "isn't saved")


def test_refused_while_queued_and_while_processing(cfg):
    d = lecture(cfg)
    job = q.enqueue(cfg.data_dir, d)
    assert_refused(cfg, d, "queued")
    job.update(state="running")
    q._save(job)
    assert_refused(cfg, d, "running")


@pytest.mark.parametrize("state", ["queued", "running"])
def test_refused_while_its_class_exam_material_is_waiting_or_being_written(cfg, state):
    d = lecture(cfg)
    exam_dir = cfg.data_dir / "classes" / A / "exams" / "midterm"
    exam_dir.mkdir(parents=True)
    job = q.enqueue(cfg.data_dir, exam_dir, q.EXAM)
    job.update(state=state)
    q._save(job)
    assert_refused(cfg, d, "exam material")


def test_a_job_end_and_its_follow_up_are_one_step_so_no_move_lands_between(cfg):
    d = lecture(cfg)
    job = q.enqueue(cfg.data_dir, d)
    job.update(state="running")
    q._save(job)
    job.update(state="done")
    real = q._save
    seen = []

    def save(j):
        with pytest.raises(TimeoutError):  # nobody else can look at the queue now
            with q.queue_lock(cfg.data_dir, wait_s=0.05):
                pass
        seen.append(j["state"])
        real(j)

    q._save = save
    try:
        q._follow_up(cfg.data_dir, job, True, save=True)
    finally:
        q._save = real
    assert seen == ["done", "queued"]
    assert sorted(j["kind"] for j in q.jobs(cfg.data_dir)) == sorted([q.TRANSCRIBE, q.SUMMARIZE])


def test_an_index_failure_after_the_rename_is_not_a_failed_move(cfg):
    lecture(cfg)

    class Broken:
        def sync(self):
            raise OSError("database is locked")

    assert mv.move_session(cfg.data_dir, A, ONE, B, index=Broken())["class"] == B


@pytest.mark.parametrize("bad", [f"{TWO}\n", 5, ["x"], "../x"])
def test_a_bad_name_is_refused_and_nothing_moves(cfg, bad):
    d = lecture(cfg)
    with pytest.raises(mv.NotFound):
        mv.move_session(cfg.data_dir, A, ONE, B, prefer=bad)
    assert d.is_dir() and "moving" not in read_json(d / "meta.json")


def test_an_unreadable_meta_stops_the_move_with_nothing_changed(cfg):
    d = lecture(cfg)
    (d / "meta.json").write_text("{", encoding="utf-8")
    with pytest.raises(mv.Busy):
        mv.move_session(cfg.data_dir, A, ONE, B)
    assert d.is_dir() and not at(cfg, B, ONE).exists() and not journals(cfg)


def test_a_file_held_open_refuses_with_a_clear_message_and_leaves_it_in_place(cfg, monkeypatch):
    d = lecture(cfg)
    monkeypatch.setattr(mv, "RENAME_TRIES", 2)
    def held(src, dst):
        raise PermissionError(13, "in use")
    monkeypatch.setattr(mv.os, "rename", held)
    assert_refused(cfg, d, "in use")


def test_the_target_appearing_at_the_last_instant_is_not_replaced(cfg, monkeypatch):
    d = lecture(cfg)
    real, calls = os.rename, []

    def racy(src, dst):
        calls.append(dst)
        if len(calls) == 1:
            Path(dst).mkdir(parents=True)  # e.g. a recording started in the same second
            raise FileExistsError(dst)
        real(src, dst)
    monkeypatch.setattr(mv.os, "rename", racy)
    r = mv.move_session(cfg.data_dir, A, ONE, B)
    assert r["session"] == f"{ONE}-2" and at(cfg, B, ONE).is_dir() and not list(at(cfg, B, ONE).iterdir()) and not d.exists()
    assert not journals(cfg)


# ------------------------------------------------------------------ search
def test_search_is_right_straight_after_a_move_and_after_undo(cfg):
    lecture(cfg, text="the purple elephant theorem")
    idx = SearchIndex(cfg.data_dir, cfg)

    def hits(c=None):
        return [(r["class"], r["session"]) for r in idx.exact("purple elephant", c)["results"]]

    assert hits() == [(A, ONE)]
    mv.move_session(cfg.data_dir, A, ONE, B, index=idx)
    assert hits() == [(B, ONE)] and hits(A) == [] and hits(B) == [(B, ONE)]
    mv.move_session(cfg.data_dir, B, ONE, A, prefer=ONE, undo=True, index=idx)
    assert hits() == [(A, ONE)]


def test_stored_search_vectors_travel_with_the_lecture(cfg):
    from test_search import embed_all

    d = lecture(cfg, text="the purple elephant theorem " * 20)
    embed_all(cfg, [d])
    idx = SearchIndex(cfg.data_dir, cfg)
    before = idx.concept_status()
    assert before["windows"] and before["pending"] == 0
    vectors = (d / EMB_NPY).read_bytes()
    mv.move_session(cfg.data_dir, A, ONE, B, index=idx)
    new = at(cfg, B, ONE)
    assert (new / EMB_NPY).read_bytes() == vectors and (new / EMB_META).is_file()
    assert idx.concept_status() == before
    q_vec = np.ones(np.load(new / EMB_NPY).shape[1], np.float32)
    assert {r["class"] for r in idx.concept(q_vec)} == {B}


# ------------------------------------------------------------------ jobs
def test_finished_jobs_follow_the_lecture_so_a_failure_still_shows(cfg):
    d = lecture(cfg)
    (d / "transcript.raw.json").unlink()
    job = q.enqueue(cfg.data_dir, d)
    job.update(state="failed", finished="2026-09-01T10:00:00+00:00", error="boom")
    q._save(job)
    assert lib.list_sessions(cfg.data_dir, A)[0]["state"] == "failed"
    mv.move_session(cfg.data_dir, A, ONE, B)
    assert lib.list_sessions(cfg.data_dir, B)[0]["state"] == "failed"
    assert q.jobs(cfg.data_dir)[0]["session"] == str(at(cfg, B, ONE).resolve())


def test_a_moved_lecture_can_be_processed_at_its_new_place(cfg):
    lecture(cfg)
    mv.move_session(cfg.data_dir, A, ONE, B)
    job = q.enqueue(cfg.data_dir, at(cfg, B, ONE), q.SUMMARIZE)
    assert job["session"] == str(at(cfg, B, ONE).resolve())


# ------------------------------------------------------------------ Keep both pairs
def test_keep_both_survives_a_move_on_both_sides_and_undo(cfg):
    src = {"kind": "import", "file": "a.m4a", "bytes": 10, "sha256": "ab" * 32}
    lecture(cfg, source=src)
    lecture(cfg, TWO, source=src)
    dup.dismiss(cfg.data_dir, A, ONE, A, TWO)  # stored on ONE: it names TWO
    assert not dup.for_class(cfg.data_dir, A)
    mv.move_session(cfg.data_dir, A, TWO, B)
    assert read_json(at(cfg, A, ONE) / "meta.json")["dismissed_duplicates"] == [f"{B}/{TWO}"]
    assert not dup.for_class(cfg.data_dir, A) and not dup.for_class(cfg.data_dir, B)
    mv.move_session(cfg.data_dir, B, TWO, A, prefer=TWO, undo=True)
    assert read_json(at(cfg, A, ONE) / "meta.json")["dismissed_duplicates"] == [f"{A}/{TWO}"]
    mv.move_session(cfg.data_dir, A, ONE, B)  # the side that holds the pair: its own list stays right
    assert not dup.for_class(cfg.data_dir, B)


# ------------------------------------------------------------------ Recently deleted
def test_a_moved_lecture_deletes_and_restores_at_its_new_place(cfg):
    lecture(cfg)
    r = mv.move_session(cfg.data_dir, A, ONE, B)
    t = trash.delete_session(cfg.data_dir, B, r["session"])
    assert not at(cfg, B, ONE).exists()
    trash.restore(cfg.data_dir, t["id"])
    assert at(cfg, B, ONE).is_dir() and read_json(at(cfg, B, ONE) / "meta.json")["class"] == B


def test_a_lecture_in_the_trash_still_restores_to_its_class_after_another_moved_in(cfg):
    lecture(cfg)
    t = trash.delete_session(cfg.data_dir, A, ONE)
    lecture(cfg, slug=B)
    mv.move_session(cfg.data_dir, B, ONE, A)  # takes the deleted lecture's old place
    with pytest.raises(trash.Conflict):
        trash.restore(cfg.data_dir, t["id"])  # refused, not replaced
    mv.move_session(cfg.data_dir, A, ONE, B)
    trash.restore(cfg.data_dir, t["id"])
    assert at(cfg, A, ONE).is_dir() and at(cfg, B, ONE).is_dir()


def test_undo_after_the_original_class_was_deleted_is_refused_clearly(cfg):
    lecture(cfg)
    mv.move_session(cfg.data_dir, A, ONE, B)
    trash.delete_class(cfg.data_dir, A)
    with pytest.raises(trash.NotFound, match="No such class"):
        mv.move_session(cfg.data_dir, B, ONE, A, prefer=ONE, undo=True)
    assert at(cfg, B, ONE).is_dir()


# ------------------------------------------------------------------ units and exams
def exam_with_material(cfg, slug, unit_ids) -> tuple[Path, dict]:
    c = lib.class_dir(cfg.data_dir, slug)
    exam = un.save_exam(c, "Midterm", unit_ids)
    d = un.exam_dir(c, exam["id"])
    d.mkdir(parents=True)
    (d / "exam.md").write_text("hand-checked material", encoding="utf-8")
    write_json_atomic(d / EXAM_META, {"state": "done", "spans": _scope_key(un.scope_spans(c, exam))})
    return d, exam


def test_a_unit_that_started_in_the_moved_lecture_stays_shows_as_moved_and_undo_restores_it(cfg):
    for n in (ONE, TWO, THREE):
        lecture(cfg, n)
    c = lib.class_dir(cfg.data_dir, A)
    u2 = un.add_boundary(c, TWO, 0, "Waves")
    u3 = un.add_boundary(c, THREE, 100, "Optics")
    units_file, keep = c / un.UNITS_FILE, (c / un.UNITS_FILE).read_bytes()
    d, exam = exam_with_material(cfg, A, [u2["id"]])
    material = (d / "exam.md").read_bytes()
    spans = un.unit_spans(c)
    assert not material_is_stale(d, c, exam)

    mv.move_session(cfg.data_dir, A, TWO, B)
    assert units_file.read_bytes() == keep  # nothing is rewritten
    after = {u["name"]: u for u in un.unit_spans(c)}
    assert after["Waves"]["moved"] and [s["session"] for s in after["Waves"]["spans"]] == [THREE]  # visible, not silent
    assert [s["session"] for s in after["Optics"]["spans"]] == [THREE] and not after["Optics"]["moved"]
    assert [s["session"] for s in after["Unit 1"]["spans"]] == [ONE]
    assert un.get_exam(c, exam["id"])["units"] == [u2["id"]]  # the exam keeps its unit
    assert material_is_stale(d, c, exam) and (d / "exam.md").read_bytes() == material  # flagged, never rewritten

    mv.move_session(cfg.data_dir, B, TWO, A, prefer=TWO, undo=True)
    assert un.unit_spans(c) == spans and not material_is_stale(d, c, exam)
    assert u3["id"] in {u["id"] for u in un.load_units(c)}


def test_moving_the_last_lecture_of_a_unit_leaves_it_visible_with_no_lectures(cfg):
    for n in (ONE, TWO):
        lecture(cfg, n)
    c = lib.class_dir(cfg.data_dir, A)
    un.add_boundary(c, TWO, 0, "Waves")
    mv.move_session(cfg.data_dir, A, TWO, B)
    placed = {u["name"]: u for u in un.ordered_units(c)}
    assert placed["Waves"].get("orphan") and [u["name"] for u in un.unit_spans(c)] == ["Unit 1"]


def test_the_moved_lecture_lands_in_date_order_in_the_target_timeline_whose_exam_goes_out_of_date(cfg):
    lecture(cfg, ONE, slug=B)
    lecture(cfg, THREE, slug=B)
    c = lib.class_dir(cfg.data_dir, B)
    u = un.add_boundary(c, THREE, 0, "Later")
    d, exam = exam_with_material(cfg, B, ["u1"])
    lecture(cfg, TWO)
    mv.move_session(cfg.data_dir, A, TWO, B)
    assert [s["name"] for s in un.class_sessions(c)] == [ONE, TWO, THREE]
    spans = {x["name"]: [s["session"] for s in x["spans"]] for x in un.unit_spans(c)}
    assert spans == {"Unit 1": [ONE, TWO], "Later": [THREE]} and u["id"]
    assert material_is_stale(d, c, exam)


def test_search_scoped_to_a_unit_drops_the_moved_lecture(cfg):
    for n, t in ((ONE, "first words"), (TWO, "second words")):
        lecture(cfg, n, text=t)
    idx = SearchIndex(cfg.data_dir, cfg)
    assert idx.exact("words", A, scope={"units": ["u1"]})["total"] == 2
    mv.move_session(cfg.data_dir, A, TWO, B, index=idx)
    assert idx.exact("words", A, scope={"units": ["u1"]})["total"] == 1


# ------------------------------------------------------------------ a real crash at each step
CRASH = """
import os, sys
from pathlib import Path
import lecture_recorder
from lecture_recorder import move as mv, duplicates as dup
from lecture_recorder.pipeline import queue
from lecture_recorder import recorder
point, data = sys.argv[1], Path(sys.argv[2])
die = lambda *a, **k: os._exit(9)
if point == "before_rename":
    mv.os.rename = die
elif point == "after_rename":
    mv._finish = die
elif point == "after_meta":
    mv.repoint_jobs = die
elif point == "after_jobs":
    mv.dup.repoint = die
else:
    real = mv._finish
    def then_die(*a):
        real(*a)
        os._exit(9)
    mv._finish = then_die
mv.move_session(data, "physics-101", "2026-09-01_09-00-00", "chem-200")
"""


def consistent_at(cfg, slug, name):
    d = at(cfg, slug, name)
    meta = read_json(d / "meta.json")
    assert (meta["class"], meta["name"]) == (slug, name)
    assert q.jobs(cfg.data_dir)[0]["session"] == str(d.resolve())
    assert read_json(at(cfg, A, TWO) / "meta.json")["dismissed_duplicates"] == [f"{slug}/{name}"]
    assert (d / "audio.wav").is_file() and (d / "transcript.json").is_file()


@pytest.mark.parametrize("point,moved", [("before_rename", False), ("after_rename", True), ("after_meta", True),
                                         ("after_jobs", True), ("before_journal_gone", True)])
def test_a_crash_at_any_step_leaves_one_complete_lecture_and_recovery_finishes_it(cfg, point, moved):
    d = lecture(cfg)
    lecture(cfg, TWO, dismissed_duplicates=[f"{A}/{ONE}"])
    job = q.enqueue(cfg.data_dir, d)
    job.update(state="done", finished="2026-09-01T10:00:00+00:00")
    q._save(job)
    r = subprocess.run([sys.executable, "-c", CRASH, point, str(cfg.data_dir)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 9, r.stderr
    assert at(cfg, A, ONE).is_dir() != at(cfg, B, ONE).is_dir()  # exactly one, in every case
    assert at(cfg, B, ONE).is_dir() is moved
    assert len(journals(cfg)) == 1

    mv.finish_moves(cfg.data_dir)
    assert not journals(cfg)
    consistent_at(cfg, *((B, ONE) if moved else (A, ONE)))
    mv.finish_moves(cfg.data_dir)  # nothing left to do, and running it again changes nothing
    consistent_at(cfg, *((B, ONE) if moved else (A, ONE)))


def test_recover_and_app_start_finish_an_interrupted_move(cfg, monkeypatch, capsys):
    from lecture_recorder.recorder.control import recover_sessions

    lecture(cfg)
    monkeypatch.setattr(mv, "_finish", lambda *a: (_ for _ in ()).throw(RuntimeError("disk hiccup")))
    r = mv.move_session(cfg.data_dir, A, ONE, B)  # the rename stands; the rest waits for recovery
    assert r["class"] == B and len(journals(cfg)) == 1 and read_json(at(cfg, B, ONE) / "meta.json")["class"] == A
    monkeypatch.undo()
    recover_sessions(cfg.data_dir)
    assert not journals(cfg) and read_json(at(cfg, B, ONE) / "meta.json")["class"] == B


def stuck_move(cfg, monkeypatch) -> None:
    """A move whose fix-ups failed: the lecture has moved, its journal and tag remain."""
    with monkeypatch.context() as m:
        m.setattr(mv, "_finish", lambda *a: 1 / 0)
        mv.move_session(cfg.data_dir, A, ONE, B)
    assert len(journals(cfg)) == 1


def test_recovery_goes_by_the_lecture_not_by_which_names_exist(cfg, monkeypatch):
    lecture(cfg)
    stuck_move(cfg, monkeypatch)
    lecture(cfg)  # a new lecture took the old name: the journal must still finish the moved one
    assert mv.finish_moves(cfg.data_dir) == 1
    assert read_json(at(cfg, B, ONE) / "meta.json")["class"] == B and not journals(cfg)
    assert "moving" not in read_json(at(cfg, B, ONE) / "meta.json")
    assert read_json(at(cfg, A, ONE) / "meta.json")["class"] == A


def test_a_stuck_move_finishes_before_the_next_one(cfg, monkeypatch):
    lecture(cfg)
    stuck_move(cfg, monkeypatch)
    r = mv.move_session(cfg.data_dir, B, ONE, A)  # no restart in between
    assert r["class"] == A and not journals(cfg)
    assert read_json(at(cfg, A, ONE) / "meta.json")["class"] == A


def test_a_lecture_with_a_move_that_still_cannot_finish_is_not_moved_again(cfg, monkeypatch):
    lecture(cfg)
    stuck_move(cfg, monkeypatch)
    monkeypatch.setattr(mv, "_finish", lambda *a: 1 / 0)
    with pytest.raises(mv.Busy, match="last move"):
        mv.move_session(cfg.data_dir, B, ONE, A)
    assert at(cfg, B, ONE).is_dir() and len(journals(cfg)) == 1


def test_a_journal_whose_target_is_another_lecture_leaves_it_alone(cfg, monkeypatch):
    lecture(cfg)
    stuck_move(cfg, monkeypatch)
    moved = at(cfg, B, ONE)
    meta = read_json(moved / "meta.json")
    meta.pop("moving")
    write_json_atomic(moved / "meta.json", meta)  # same name, no tag: not the moved lecture
    assert mv.finish_moves(cfg.data_dir) == 0
    assert read_json(moved / "meta.json")["class"] == A and not journals(cfg)  # untouched


def test_a_move_that_never_happened_leaves_no_trace(cfg):
    d = lecture(cfg)
    before = read_json(d / "meta.json")
    tag = "a" * 32
    root = cfg.data_dir / "queue" / mv.MOVES_DIR
    root.mkdir(parents=True)
    write_json_atomic(root / f"{tag}.json", {"id": tag, "from": [A, ONE], "to": [B, ONE], "settled": True})
    update_meta(d, moving=tag)
    assert mv.finish_moves(cfg.data_dir) == 0
    assert not journals(cfg) and read_json(d / "meta.json") == before


def test_a_damaged_journal_is_kept_and_blocks_nothing(cfg):
    lecture(cfg)
    root = cfg.data_dir / "queue" / mv.MOVES_DIR
    root.mkdir(parents=True)
    (root / "bad.json").write_text("{", encoding="utf-8")
    (root / "evil.json").write_text(json.dumps({"from": ["..", ".."], "to": [B, ONE], "settled": True}), encoding="utf-8")
    assert mv.finish_moves(cfg.data_dir) == 0
    assert len(list(root.glob("*.json"))) == 2 and at(cfg, A, ONE).is_dir()
    mv.move_session(cfg.data_dir, A, ONE, B)
    assert at(cfg, B, ONE).is_dir()


def test_job_files_are_not_confused_by_the_journal_folder(cfg):
    lecture(cfg)
    mv.move_session(cfg.data_dir, A, ONE, B)
    assert q.jobs(cfg.data_dir) == []


# ------------------------------------------------------------------ the app and the terminal
def test_the_api_moves_undoes_and_changes_the_page_signal(cfg, client):
    lecture(cfg, text="the purple elephant theorem")
    v0 = client.get("/api/record/status").json()["trash_version"]
    r = client.post(f"/api/sessions/{A}/{ONE}/move", json={"to": B}, headers=H)
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["class"], body["session"], body["from_class"], body["from_session"], body["class_name"]) == (B, ONE, A, ONE, "Chem 200")
    assert body["trash_version"] != v0
    assert client.get(f"/api/sessions/{B}/{ONE}").status_code == 200 and client.get(f"/api/sessions/{A}/{ONE}").status_code == 404
    assert [x["class"] for x in client.get("/api/search?q=purple+elephant").json()["results"]] == [B]
    back = client.post(f"/api/sessions/{B}/{ONE}/move", json={"to": A, "name": ONE, "undo": True}, headers=H)
    assert back.status_code == 200 and back.json()["class"] == A
    assert [x["class"] for x in client.get("/api/search?q=purple+elephant").json()["results"]] == [A]


def test_the_api_refuses_a_busy_lecture_with_409_and_needs_the_app_header(cfg, client):
    d = lecture(cfg)
    assert client.post(f"/api/sessions/{A}/{ONE}/move", json={"to": B}).status_code == 403
    assert client.post(f"/api/sessions/{A}/{ONE}/move", json={"to": "nope"}, headers=H).status_code == 404
    assert client.post(f"/api/sessions/{A}/{ONE}/move", json={}, headers=H).status_code == 404
    q.enqueue(cfg.data_dir, d)
    r = client.post(f"/api/sessions/{A}/{ONE}/move", json={"to": B}, headers=H)
    assert r.status_code == 409 and "queued" in r.json()["detail"] and d.is_dir()


def test_an_unsorted_lecture_with_no_suggestion_moves_by_hand_and_undoes(cfg, client):
    lecture(cfg, text="plain words with no signal")
    mv.move_session(cfg.data_dir, A, ONE, lib.UNSORTED)
    assert read_json(at(cfg, lib.UNSORTED, ONE) / "meta.json")["class_settled"] is False  # not sure yet: suggestions stay on
    assert client.get(f"/api/sessions/{lib.UNSORTED}/{ONE}/suggestion").json()["suggestion"] is None
    r = client.post(f"/api/sessions/{lib.UNSORTED}/{ONE}/move", json={"to": B}, headers=H)
    assert r.status_code == 200 and (r.json()["class"], r.json()["from_class"]) == (B, lib.UNSORTED)
    back = client.post(f"/api/sessions/{B}/{ONE}/move", json={"to": lib.UNSORTED, "name": ONE, "undo": True}, headers=H)
    assert back.status_code == 200 and at(cfg, lib.UNSORTED, ONE).is_dir() and not at(cfg, B, ONE).exists()


def test_a_lecture_in_a_real_class_with_no_suggestion_can_go_to_unsorted_on_first_use(cfg, client):
    lecture(cfg)
    assert client.get(f"/api/sessions/{A}/{ONE}/suggestion").json()["suggestion"] is None
    assert lib.UNSORTED not in [c["slug"] for c in client.get("/api/classes").json()]
    r = client.post(f"/api/sessions/{A}/{ONE}/move", json={"to": lib.UNSORTED}, headers=H)
    assert r.status_code == 200 and at(cfg, lib.UNSORTED, ONE).is_dir()


def test_a_hand_move_of_an_unsorted_lecture_gets_the_same_refusals(cfg, client):
    lecture(cfg)
    mv.move_session(cfg.data_dir, A, ONE, lib.UNSORTED)
    d, url = at(cfg, lib.UNSORTED, ONE), f"/api/sessions/{lib.UNSORTED}/{ONE}/move"
    assert client.post(url, json={"to": B}).status_code == 403
    assert client.post(url, json={"to": lib.UNSORTED}, headers=H).status_code == 409  # already there
    assert client.post(url, json={"to": ""}, headers=H).status_code == 404  # nothing chosen
    q.enqueue(cfg.data_dir, d)
    r = client.post(url, json={"to": B}, headers=H)
    assert r.status_code == 409 and "queued" in r.json()["detail"] and d.is_dir() and not at(cfg, B, ONE).exists()


def test_keep_here_survives_a_move_and_its_undo(cfg):
    d = lecture(cfg, class_settled=True)
    meta = read_json(d / "meta.json")
    mv.move_session(cfg.data_dir, A, ONE, B)
    mv.move_session(cfg.data_dir, B, ONE, A, undo=True)
    assert read_json(d / "meta.json") == meta


def test_undo_returns_to_a_folder_without_class_settings_and_a_plain_move_there_stays_refused(cfg):
    d = lecture(cfg, slug="old-folder")
    mv.move_session(cfg.data_dir, "old-folder", ONE, B)
    with pytest.raises(trash.NotFound):
        mv.move_session(cfg.data_dir, B, ONE, "old-folder")
    mv.move_session(cfg.data_dir, B, ONE, "old-folder", undo=True)
    assert d.is_dir() and not at(cfg, B, ONE).exists()


def test_the_move_control_reuses_the_move_path_and_needs_a_choice():
    js = (Path(lecture_recorder.__file__).parent / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert js.count("/move`, { method:") == 1  # one call site: moveLecture
    assert js.count("const m = await moveLecture(") == 1  # the banner and Move to… share relocate(), so the same Undo toast
    assert "await relocate(slug, name, picker.value)" in js and "await relocate(slug, name, s.class)" in js
    control = js[js.index("function moveControl("):js.index('// Lecture page: "Looks like')]
    assert 'id: "move-go", text: "Move", disabled: true' in control and "go.disabled = !picker.value" in control
    assert 'id: "move-panel" });' in control and "panel.replaceChildren();" in control  # no select in the page while closed
    assert "c.slug !== slug" in control  # never offers the class it is already in
    assert 'disabled: state === "recording" || state === "running"' in control


def test_the_cli_moves_and_reports_refusals(cfg, capsys):
    lecture(cfg)
    assert cli.main(["move", A, ONE, B, "--data-dir", str(cfg.data_dir)]) == 0
    assert "Chem 200" in capsys.readouterr().out and at(cfg, B, ONE).is_dir()
    assert cli.main(["move", A, ONE, B, "--data-dir", str(cfg.data_dir)]) == 2
    assert "Not found" in capsys.readouterr().out


def test_the_cli_undo_hint_restores_the_original_name(cfg, capsys):
    lecture(cfg)
    lecture(cfg, slug=B)
    assert cli.main(["move", A, ONE, B, "--data-dir", str(cfg.data_dir)]) == 0
    hint = capsys.readouterr().out.split("lecture-recorder move ")[1].split()
    assert hint == [B, f"{ONE}-2", A, "--name", ONE]
    assert cli.main(["move", *hint, "--data-dir", str(cfg.data_dir)]) == 0
    assert at(cfg, A, ONE).is_dir()
