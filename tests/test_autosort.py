"""Class suggestions. Each signal (positive, negative, too little evidence), the margin rule, Not sure yet /
Unsorted, Keep here, and that a suggestion can never break a page or a job. Synthetic lectures in temp folders."""
import dataclasses
import io
import random
import wave
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import lecture_recorder  # noqa: F401  (lockdown)
from lecture_recorder import autosort as sort
from lecture_recorder import cli
from lecture_recorder import duplicates as dup
from lecture_recorder import library as lib
from lecture_recorder import move as mv
from lecture_recorder.config import load_config
from lecture_recorder.pipeline import queue as q
from lecture_recorder.recorder.session import read_json, write_json_atomic
from lecture_recorder.web.app import create_app
from test_web_app import BASE, H

PHY, CHEM, HIST = "physics-101", "chem-200", "history-210"
TUE, WED = "2026-09-01", "2026-09-02"  # a Tuesday and a Wednesday
Z = "+00:00"
EVERYDAY = [f"common{i}" for i in range(400)]
TERMS = {PHY: [f"phys{i}x" for i in range(200)], CHEM: [f"chem{i}x" for i in range(200)], HIST: [f"hist{i}x" for i in range(200)]}


@pytest.fixture
def cfg(tmp_path):
    data = tmp_path / "data"
    for n in ("Physics 101", "Chem 200", "History 210"):
        lib.create_class(data, n)
    return dataclasses.replace(load_config(), data_dir=data.resolve(), home=tmp_path.resolve())


@pytest.fixture
def client(cfg):
    with TestClient(create_app(cfg, autoprocess=False, monitor=False), base_url=BASE) as c:
        yield c


def speech(*courses, seed=0, n=2000) -> list[str]:
    rng = random.Random(f"{courses}{seed}")
    pool = [t for c in courses for t in TERMS[c][:120]]
    return [rng.choice(pool) if rng.random() < 0.3 else rng.choice(EVERYDAY) for _ in range(n)]


def lecture(cfg, slug, name, created=None, words=None, source=None, **meta) -> Path:
    d = cfg.data_dir / "classes" / slug / "sessions" / name
    d.mkdir(parents=True)
    write_json_atomic(d / "meta.json", {"name": name, "title": name, "class": slug, "created": created or f"{name[:10]}T{name[11:13]}:{name[14:16]}:00{Z}",
                                        "duration_s": 3000.0, "source": source or {"kind": "mic"}, **meta}, fsync=False)
    if words:
        doc = {"segments": [{"id": i, "start": i, "end": i + 1, "text": " ".join(words[i:i + 12])} for i in range(0, len(words), 12)]}
        write_json_atomic(d / "transcript.json", doc, fsync=False)
        write_json_atomic(d / "transcript.raw.json", doc, fsync=False)
        dup.sketch_session(d)
    return d


def weekly(cfg, slug, first=TUE, at="10-00", weeks=4, words=False, **kw) -> None:
    for w in range(weeks):
        day = date.fromisoformat(first) + timedelta(days=7 * w)
        lecture(cfg, slug, f"{day}_{at}-00", words=speech(slug, seed=w) if words else None, **kw)


def pick(cfg, when=None, topic=None, **kw):
    title = sort._title_words(topic)
    return sort.suggest(sort._classes(cfg.data_dir), datetime.fromisoformat(when) if when else None, title, **kw)


TUESDAY_TEN = f"2026-10-06T10:05:00{Z}"


# ------------------------------------------------------------------ schedule
def test_schedule_suggests_the_class_of_that_weekday_and_time_with_a_reason(cfg):
    weekly(cfg, PHY)
    s = pick(cfg, TUESDAY_TEN)
    assert s["class"] == PHY and s["signals"] == ["schedule"] and s["reason"] == "Usually Tuesdays at 10:00"


@pytest.mark.parametrize("when", [f"2026-10-07T10:05:00{Z}", f"2026-10-06T11:00:00{Z}", f"2027-06-01T10:00:00{Z}"])
def test_schedule_near_misses_other_day_other_hour_other_semester(cfg, when):
    weekly(cfg, PHY)
    assert pick(cfg, when) is None


def test_schedule_needs_two_past_lectures_and_real_recording_times(cfg):
    weekly(cfg, PHY, weeks=1)
    assert pick(cfg, TUESDAY_TEN) is None
    weekly(cfg, CHEM, weeks=4, date_from="file date", source={"kind": "import", "file": "x.m4a"})
    assert pick(cfg, TUESDAY_TEN) is None  # copied files: their dates say nothing
    weekly(cfg, HIST, weeks=4, source={"kind": "import", "file": "old.m4a"})  # imported before date_from existed
    assert pick(cfg, TUESDAY_TEN) is None


def test_a_one_off_slot_is_not_a_pattern(cfg):
    weekly(cfg, PHY, at="14-00", weeks=6)
    weekly(cfg, PHY, at="10-00", first="2026-09-02", weeks=0)
    lecture(cfg, PHY, "2026-09-01_10-00-00")
    lecture(cfg, PHY, "2026-09-08_10-00-00")
    assert pick(cfg, TUESDAY_TEN) is None  # 2 of 8 Tuesdays


def test_the_margin_rule_two_classes_in_the_same_slot_show_nothing(cfg):
    weekly(cfg, PHY)
    weekly(cfg, CHEM)
    assert pick(cfg, TUESDAY_TEN) is None


def test_two_matches_are_enough_and_more_evidence_beats_less_only_by_a_clear_margin(cfg):
    weekly(cfg, PHY, weeks=2)
    assert pick(cfg, TUESDAY_TEN)["class"] == PHY
    weekly(cfg, CHEM, weeks=4)
    assert pick(cfg, TUESDAY_TEN) is None  # 0.6 vs 0.4: too close


# ------------------------------------------------------------------ Zoom title
@pytest.mark.parametrize("topic", ["Physics 101 Lecture", "physics-101", "PHYSICS 101 Zoom Meeting", "Physics"])
def test_title_matches_the_class_name(cfg, topic):
    s = pick(cfg, topic=topic)
    assert s["class"] == PHY and s["reason"] == "Zoom title matches"


@pytest.mark.parametrize("topic", ["Physics 202", "Math 101", "Personal Meeting Room", "101", "Lecture", ""])
def test_title_near_misses(cfg, topic):
    assert pick(cfg, topic=topic) is None


def test_title_matches_an_earlier_zoom_topic_of_the_class(cfg):
    zoom = {"kind": "import", "zoom": True, "file": "a.m4a", "folder": "2026-09-01 10.00.00 Mechanics Review 91234567890"}
    lecture(cfg, PHY, "2026-09-01_10-00-00", source=zoom, date_from="Zoom folder name")
    assert pick(cfg, topic="Mechanics Review")["class"] == PHY
    assert pick(cfg, topic="Thermodynamics Review") is None


def test_schedule_and_title_add_up_and_a_conflict_shows_nothing(cfg):
    weekly(cfg, PHY)
    s = pick(cfg, TUESDAY_TEN, "Physics 101")
    assert s["class"] == PHY and set(s["signals"]) == {"schedule", "title"} and s["reason"] == "Zoom title matches"
    assert pick(cfg, TUESDAY_TEN, "Chem 200") is None


# ------------------------------------------------------------------ content
def three_classes(cfg, **kw):
    for slug, first in ((PHY, TUE), (CHEM, WED), (HIST, "2026-09-03")):
        weekly(cfg, slug, first=first, words=True, **kw)


def test_content_points_to_the_class_that_uses_the_words(cfg):
    three_classes(cfg)
    s = pick(cfg, terms=frozenset(dup.top_terms(speech(CHEM, seed=99))))
    assert s["class"] == CHEM and s["signals"] == ["content"] and s["reason"] == "Similar words to its lectures"


def test_content_mixed_or_unrelated_words_show_nothing(cfg):
    three_classes(cfg)
    for seed in range(12):
        assert pick(cfg, terms=frozenset(dup.top_terms(speech(CHEM, PHY, seed=seed)))) is None
    assert pick(cfg, terms=frozenset(f"other{i}" for i in range(300))) is None
    assert pick(cfg, terms=frozenset(EVERYDAY[:300])) is None  # words common to every class


def test_content_needs_two_classes_with_at_least_two_sketched_lectures(cfg):
    weekly(cfg, PHY, words=True)
    terms = frozenset(dup.top_terms(speech(PHY, seed=99)))
    assert pick(cfg, terms=terms) is None  # nothing to compare with
    lecture(cfg, CHEM, "2026-09-02_10-00-00", words=speech(CHEM))
    assert pick(cfg, terms=terms) is None  # one lecture is not enough
    lecture(cfg, CHEM, "2026-09-09_10-00-00", words=speech(CHEM, seed=1))
    assert pick(cfg, terms=terms)["class"] == PHY


def test_short_transcripts_make_no_terms(cfg):
    d = lecture(cfg, PHY, "2026-09-01_10-00-00", words=speech(PHY, n=50))
    assert read_json(d / dup.SKETCH_FILE)["terms"] is None


def test_a_sketch_from_before_m18_catches_up(cfg):
    d = lecture(cfg, PHY, "2026-09-01_10-00-00", words=speech(PHY))
    write_json_atomic(d / dup.SKETCH_FILE, {"minhash": None})
    dup.sketch_session(d)
    assert read_json(d / dup.SKETCH_FILE)["terms"]


def test_only_a_bounded_sample_of_each_class_is_read(cfg, monkeypatch):
    weekly(cfg, PHY, words=True, weeks=10)
    weekly(cfg, CHEM, words=True, weeks=3, first=WED)
    seen = []
    real = sort._terms
    monkeypatch.setattr(sort, "_terms", lambda d: seen.append(d) or real(d))
    pick(cfg, terms=frozenset(["word"]))
    assert len([d for d in seen if d.parent.parent.name == PHY]) == sort.CONTENT_SAMPLE + 1


def test_a_lecture_where_the_words_fit_reads_no_other_lectures_meta(cfg, monkeypatch):
    three_classes(cfg)
    reads = []
    real = sort.read_json
    monkeypatch.setattr(sort, "read_json", lambda p, *a, **k: reads.append(p.name) or real(p, *a, **k))
    name = sorted(p.name for p in lib.class_dir(cfg.data_dir, PHY).joinpath("sessions").iterdir())[-1]
    assert not sort.for_class(cfg.data_dir, PHY, name)
    assert reads.count("meta.json") == 1


# ------------------------------------------------------------------ lecture pages
def misfiled(cfg, **meta) -> Path:
    three_classes(cfg)
    return lecture(cfg, PHY, "2026-10-16_16-00-00", words=speech(HIST, seed=50), **meta)


def test_a_misfiled_lecture_gets_a_suggestion_and_a_right_one_does_not(cfg):
    d = misfiled(cfg)
    found = sort.for_class(cfg.data_dir, PHY)
    assert list(found) == [d.name] and found[d.name]["class"] == HIST


def test_in_a_real_class_schedule_or_title_alone_never_suggests_a_move(cfg):
    three_classes(cfg)
    # recorded in Chem's slot and titled Chem, but its words are Physics: filed where the words say
    lecture(cfg, PHY, "2026-10-07_10-00-00", words=speech(PHY, seed=50),
            source={"kind": "import", "zoom": True, "file": "a.m4a", "folder": "2026-10-07 10.00.00 Chem 200 91234567890"},
            date_from="Zoom folder name")
    assert not sort.for_class(cfg.data_dir, PHY)
    lecture(cfg, PHY, "2026-10-14_10-00-00")  # no transcript: nothing to go on
    assert not sort.for_class(cfg.data_dir, PHY)


def test_an_unsorted_lecture_is_suggested_by_any_signal_even_before_transcription(cfg):
    weekly(cfg, PHY)
    lib.ensure_unsorted(cfg.data_dir)
    d = lecture(cfg, lib.UNSORTED, "2026-10-06_10-05-00")
    assert sort.for_class(cfg.data_dir, lib.UNSORTED)[d.name]["class"] == PHY


def test_unsorted_is_never_suggested_and_an_ambiguous_unsorted_lecture_shows_nothing(cfg):
    three_classes(cfg)
    lib.ensure_unsorted(cfg.data_dir)
    weekly(cfg, lib.UNSORTED)  # a pattern in Unsorted itself means nothing
    assert all(s["class"] != lib.UNSORTED for s in sort.for_class(cfg.data_dir, lib.UNSORTED).values())
    d = lecture(cfg, lib.UNSORTED, "2026-10-10_18-30-00", words=speech(CHEM, PHY, seed=7))
    assert d.name not in sort.for_class(cfg.data_dir, lib.UNSORTED)


@pytest.mark.parametrize("state", ["queued", "running"])
def test_no_suggestion_while_a_job_has_it(cfg, state):
    d = misfiled(cfg)
    job = q.enqueue(cfg.data_dir, d, q.SUMMARIZE)
    job.update(state=state)
    q._save(job)
    (d / "transcript.raw.json").unlink()  # the state comes from the transcribe job
    q.enqueue(cfg.data_dir, d)
    assert d.name not in sort.for_class(cfg.data_dir, PHY)


def test_keep_here_is_stored_on_the_lecture_and_never_asked_again(cfg, client):
    d = misfiled(cfg)
    url = f"/api/sessions/{PHY}/{d.name}"
    assert client.get(f"{url}/suggestion").json()["suggestion"]["class"] == HIST
    assert client.post(f"{url}/keep-here").status_code == 403
    assert client.post(f"{url}/keep-here", headers=H).status_code == 200
    assert client.get(f"{url}/suggestion").json()["suggestion"] is None
    assert read_json(d / "meta.json")["class_settled"] is True
    with TestClient(create_app(dataclasses.replace(client.app.state.config), autoprocess=False, monitor=False), base_url=BASE) as again:
        assert again.get(f"{url}/suggestion").json()["suggestion"] is None  # survives a restart


def test_move_then_undo_asks_again_and_a_move_without_undo_does_not(cfg, client):
    d = misfiled(cfg)
    name = d.name
    r = client.post(f"/api/sessions/{PHY}/{name}/move", json={"to": HIST}, headers=H).json()
    assert client.get(f"/api/sessions/{HIST}/{name}/suggestion").json()["suggestion"] is None
    client.post(f"/api/sessions/{HIST}/{name}/move", json={"to": r["from_class"], "name": r["from_session"], "undo": True}, headers=H)
    assert client.get(f"/api/sessions/{PHY}/{name}/suggestion").json()["suggestion"]["class"] == HIST


def test_a_moved_lecture_never_gets_a_suggestion_back(cfg):
    d = misfiled(cfg)
    mv.move_session(cfg.data_dir, PHY, d.name, CHEM)  # moved somewhere the words don't point to: the user's call
    assert d.name not in sort.for_class(cfg.data_dir, CHEM)


def test_class_rows_carry_the_chip_data(cfg, client):
    d = misfiled(cfg)
    rows = {s["name"]: s for s in client.get(f"/api/classes/{PHY}").json()["sessions"]}
    assert rows[d.name]["suggested"]["class_name"] == "History 210"
    assert sum(1 for s in rows.values() if s["suggested"]) == 1


def test_the_lecture_being_judged_is_not_its_own_evidence(cfg):
    three_classes(cfg)
    d = lecture(cfg, PHY, "2026-10-16_16-00-00", words=speech(HIST, seed=50))
    lecture(cfg, HIST, "2026-10-17_16-00-00", words=speech(CHEM, seed=51))  # History's newest is off-topic: stays out of nobody's way
    assert d.name in sort.for_class(cfg.data_dir, PHY)
    sample = sort._classes(cfg.data_dir)
    assert d.name in [n for n, _ in sort._sample(sample[PHY])]
    scores = sort.content_scores(sample, frozenset(dup.top_terms(speech(HIST, seed=50))), (PHY, d.name))
    assert scores[HIST] > scores[PHY]


# ------------------------------------------------------------------ import
def wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(bytes([40, 0, 216, 255]) * 4000)
    return buf.getvalue()


def suggest_import(client, **body):
    r = client.post("/api/import/suggest", json=body, headers=H)
    assert r.status_code == 200, r.text
    return r.json()["suggestion"]


@pytest.mark.parametrize("name,expected", [("2026-10-06 10.05.00.m4a", PHY), ("20261006_100500.m4a", PHY),
                                           ("GMT20261006-140500_Recording.m4a", None), ("2026-10-07 10.05.00.m4a", None)])
def test_file_names_with_a_time_pick_the_class(cfg, client, name, expected):
    weekly(cfg, PHY)
    s = suggest_import(client, files=[{"name": name}])
    assert (s and s["class"]) == (expected if "GMT" not in name else (s and s["class"]))


def test_a_file_date_is_no_recording_time(cfg, client):
    weekly(cfg, PHY)
    ms = datetime(2026, 10, 6, 10, 5).timestamp() * 1000
    assert suggest_import(client, files=[{"name": "memo.m4a", "last_modified_ms": ms}]) is None


def test_several_files_must_agree(cfg, client):
    weekly(cfg, PHY)
    weekly(cfg, CHEM, first=WED)
    both = [{"name": "2026-10-06 10.05.00.m4a"}, {"name": "2026-10-13 10.05.00.m4a"}]
    assert suggest_import(client, files=both)["class"] == PHY
    assert suggest_import(client, files=[both[0], {"name": "2026-10-07 10.05.00.m4a"}]) is None
    assert suggest_import(client, files=[both[0], {"name": "no time.m4a"}]) is None


def test_a_zoom_folder_uses_its_topic_and_time(cfg, client):
    weekly(cfg, CHEM, first=WED)
    assert suggest_import(client, topic="Physics 101 Lecture")["reason"] == "Zoom title matches"
    assert suggest_import(client, when=f"2026-10-07T10:00:00{Z}")["class"] == CHEM


def test_the_suggestion_call_files_nothing_and_survives_nonsense(cfg, client):
    weekly(cfg, PHY)
    before = sorted(p.as_posix() for p in cfg.data_dir.rglob("*") if "queue" not in p.parts)
    assert suggest_import(client, files=[{"name": 5, "last_modified_ms": "x"}]) is None
    assert suggest_import(client, when="not a date") is None
    assert suggest_import(client) is None
    assert client.post("/api/import/suggest", json={"files": "x"}, headers=H).status_code == 400
    assert client.post("/api/import/suggest", json={"files": []}).status_code == 403
    assert sorted(p.as_posix() for p in cfg.data_dir.rglob("*") if "queue" not in p.parts) == before


def test_not_sure_yet_files_into_a_real_unsorted_class_created_on_first_use(cfg, client):
    assert lib.get_class(cfg.data_dir, lib.UNSORTED) is None
    for i in range(2):
        r = client.post("/api/import", data={"class_slug": lib.UNSORTED}, headers=H,
                        files={"file": (f"memo{i}.wav", wav_bytes() + bytes(2 * i), "audio/wav")})
        assert r.status_code == 201, r.text
    c = client.get(f"/api/classes/{lib.UNSORTED}").json()
    assert c["name"] == "Unsorted" and c["has_class_file"] and len(c["sessions"]) == 2


def test_import_zoom_from_the_terminal_accepts_unsorted(cfg, tmp_path, capsys):
    folder = tmp_path / "2026-09-01 10.00.00 Something 91234567890"
    folder.mkdir()
    (folder / "audio1.m4a").write_bytes(b"not really audio")
    assert cli.main(["import-zoom", str(folder), "--class", lib.UNSORTED, "--data-dir", str(cfg.data_dir)]) == 0
    assert lib.get_class(cfg.data_dir, lib.UNSORTED)["has_class_file"]


def test_a_recording_still_needs_a_class_it_is_never_filed_as_unsorted_by_itself(cfg, client):
    assert client.post("/api/record/start", json={"class": ""}, headers=H).status_code == 400
    assert client.post("/api/record/start", json={"class": lib.UNSORTED}, headers=H).status_code == 400


def test_duplicate_scope_still_honours_unsorted(cfg):
    lib.ensure_unsorted(cfg.data_dir)
    w = speech(PHY)
    lecture(cfg, PHY, "2026-09-01_10-00-00", words=w)
    lecture(cfg, lib.UNSORTED, "2026-09-02_10-00-00", words=w)
    assert dup.for_class(cfg.data_dir, lib.UNSORTED)


# ------------------------------------------------------------------ never in the way
def test_a_broken_suggestion_never_breaks_a_page(cfg, client, monkeypatch):
    d = misfiled(cfg)
    monkeypatch.setattr(sort, "_classes", lambda *a: 1 / 0)
    assert client.get(f"/api/sessions/{PHY}/{d.name}/suggestion").json() == {"suggestion": None}
    assert client.get(f"/api/classes/{PHY}").status_code == 200
    assert suggest_import(client, files=[{"name": "x.m4a"}]) is None


@pytest.mark.parametrize("bad", ["{", "[1, 2", ""])
def test_damaged_files_cost_one_lecture_only(cfg, bad):
    d = misfiled(cfg)
    other = cfg.data_dir / "classes" / HIST / "sessions" / "2026-09-24_10-00-00"
    (other / dup.SKETCH_FILE).write_text(bad, encoding="utf-8")
    (other / "meta.json").write_text(bad, encoding="utf-8")
    assert sort.for_class(cfg.data_dir, PHY)[d.name]["class"] == HIST


def test_the_end_of_transcription_step_never_fails_a_job_and_bounds_its_work(cfg, monkeypatch):
    weekly(cfg, PHY, words=True, weeks=10)
    done = []
    monkeypatch.setattr(dup, "sketch_session", lambda d: done.append(d) or 1 / 0)
    sort.after_transcribe(cfg.data_dir)  # no exception
    assert len(done) == sort.CONTENT_SAMPLE + 1
    monkeypatch.setattr(sort, "_sketch_samples", lambda d: 1 / 0)
    sort.after_transcribe(cfg.data_dir)
    job = {"id": "x", "session": str(cfg.data_dir / "gone"), "kind": q.TRANSCRIBE}
    q._follow_up(cfg.data_dir, job, True)  # a missing lecture, a failing step: the queue goes on


def test_the_step_catches_up_older_lectures_in_other_classes(cfg):
    weekly(cfg, PHY, words=True, weeks=2)
    weekly(cfg, CHEM, words=True, weeks=2, first=WED)
    for d in cfg.data_dir.glob(f"classes/{CHEM}/sessions/*"):
        (d / dup.SKETCH_FILE).unlink()
    new = lecture(cfg, PHY, "2026-10-06_10-00-00", words=speech(PHY, seed=9))
    q._sketch(cfg.data_dir, new)
    assert len(list(cfg.data_dir.glob(f"classes/{CHEM}/sessions/*/{dup.SKETCH_FILE}"))) == 2


def test_no_new_polling_or_timers_in_the_ui():
    from test_efficiency import polling_facts

    f = polling_facts()
    assert f["only_visibility_aware_timers"] and f["min_interval_ms"] >= 2000
