"""Possible duplicate lectures. Each signal with a near miss, Keep both, the trash + Undo path, the
import warning, and that the check can never break a job. Synthetic lectures in temp folders; nothing in data/.
"""
import dataclasses
import hashlib
import io
import json
import random
import subprocess
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import lecture_recorder  # noqa: F401  (lockdown)
from lecture_recorder import duplicates as dup
from lecture_recorder import library as lib
from lecture_recorder import trash
from lecture_recorder.config import load_config
from lecture_recorder.pipeline import queue as q
from lecture_recorder.recorder.session import read_json, write_json_atomic
from lecture_recorder.web.app import create_app
from test_web_app import BASE, H

SLUG = "physics-101"
A, B = "2026-09-01_09-00-00", "2026-09-01_09-05-00"


@pytest.fixture
def cfg(tmp_path):
    data = tmp_path / "data"
    lib.create_class(data, "Physics 101")
    return dataclasses.replace(load_config(), data_dir=data.resolve(), home=tmp_path.resolve())


@pytest.fixture
def client(cfg):
    with TestClient(create_app(cfg, autoprocess=False, monitor=False), base_url=BASE) as c:
        yield c


def lecture(cfg, name=A, slug=SLUG, created="2026-09-01T09:00:00+00:00", duration=3000.0, words=None, **meta) -> Path:
    d = cfg.data_dir / "classes" / slug / "sessions" / name
    d.mkdir(parents=True)
    write_json_atomic(d / "meta.json", {"name": name, "title": meta.pop("title", name), "class": slug, "created": created,
                                        "duration_s": duration, "source": {"kind": "mic"}, **meta}, fsync=False)
    if words:
        doc = {"segments": [{"id": i, "text": " ".join(words[i:i + 12])} for i in range(0, len(words), 12)]}
        write_json_atomic(d / "transcript.json", doc, fsync=False)
        write_json_atomic(d / "transcript.raw.json", doc, fsync=False)
        dup.sketch_session(d)
    return d


def words(seed: int, n: int = 2000) -> list[str]:
    rng = random.Random(seed)
    return [f"w{rng.randrange(800)}" for _ in range(n)]


def noisy(w: list[str], seed: int, rate: float = 0.05) -> list[str]:
    rng = random.Random(seed)
    return [f"x{rng.randrange(800)}" if rng.random() < rate else t for t in w]


def signals(cfg, name=A, slug=SLUG) -> list[str]:
    found = dup.for_class(cfg.data_dir, slug).get(name, [])
    return found[0]["signals"] if found else []


# ------------------------------------------------------------------ signal 1: same source file
def test_same_hash_is_flagged_even_across_classes_and_a_different_hash_is_not(cfg):
    lib.create_class(cfg.data_dir, "Chem 200")
    src = {"kind": "import", "file": "a.m4a", "bytes": 10, "sha256": "ab" * 32}
    lecture(cfg, A, source=src, created="2026-09-01T09:00:00+00:00")
    lecture(cfg, B, slug="chem-200", source={**src, "file": "renamed.m4a"}, created="2026-10-01T09:00:00+00:00")
    assert signals(cfg) == ["file"] and signals(cfg, B, "chem-200") == ["file"]
    third = "2026-09-02_09-00-00"
    lecture(cfg, third, source={**src, "sha256": "cd" * 32}, created="2026-09-02T09:00:00+00:00")  # same name and size, other bytes
    assert not dup.for_class(cfg.data_dir, SLUG).get(third)


def test_an_import_from_before_m17_is_matched_by_name_and_size(cfg):
    old = {"kind": "import", "file": "a.m4a", "bytes": 10}
    lecture(cfg, A, source=old)
    lecture(cfg, B, source={**old, "sha256": "ab" * 32}, created="2026-09-02T09:00:00+00:00")
    assert signals(cfg) == ["file"]
    assert dup.find_source(cfg.data_dir, "ef" * 32, "a.m4a", 11) is None  # other size


# ------------------------------------------------------------------ signal 2: overlapping time
def test_half_of_the_shorter_one_overlapping_flags_and_less_does_not(cfg):
    lecture(cfg, A, duration=3000)
    lecture(cfg, B, created="2026-09-01T09:40:00+00:00", duration=1200)  # 600 s shared of 1200 s: 50%
    assert signals(cfg) == ["time"]
    meta = cfg.data_dir / "classes" / SLUG / "sessions" / B / "meta.json"
    write_json_atomic(meta, {**read_json(meta), "created": "2026-09-01T09:41:00+00:00"}, fsync=False)  # 540 s: 45%
    assert signals(cfg) == []


def test_time_zones_are_compared_as_moments(cfg):
    lecture(cfg, A, created="2026-09-01T09:00:00+02:00", duration=3000)
    lecture(cfg, B, created="2026-09-01T07:05:00+00:00", duration=3000)
    assert signals(cfg) == ["time"]


@pytest.mark.parametrize("date_from", ["file date", "import time"])
def test_a_guessed_date_never_counts_as_recorded_at_the_same_time(cfg, date_from):
    lecture(cfg, A)
    lecture(cfg, B, date_from=date_from)
    assert signals(cfg) == []


def test_time_is_compared_only_within_a_class_plus_unsorted(cfg):
    lib.create_class(cfg.data_dir, "Chem 200")
    lecture(cfg, A)
    lecture(cfg, B, slug="chem-200")
    assert signals(cfg) == []
    lib.create_class(cfg.data_dir, "Unsorted")
    other = "2026-09-01_09-10-00"
    lecture(cfg, other, slug=lib.UNSORTED)
    assert dup.for_class(cfg.data_dir, SLUG)[A][0]["other"]["class"] == lib.UNSORTED
    assert sorted(m["other"]["class"] for m in dup.for_class(cfg.data_dir, lib.UNSORTED)[other]) == ["chem-200", SLUG]


def test_an_import_with_no_date_source_only_has_its_import_time(cfg):
    src = {"kind": "import", "file": "a.m4a", "bytes": 1}
    lecture(cfg, A, source=src)
    lecture(cfg, B, source={**src, "file": "b.m4a"})
    assert signals(cfg) == []


def test_a_class_with_no_imports_never_reads_other_classes(cfg, monkeypatch):
    lib.create_class(cfg.data_dir, "Chem 200")
    other = lecture(cfg, B, slug="chem-200", created="2026-10-01T09:00:00+00:00")
    lecture(cfg, A)
    real, seen = dup.read_json, []
    monkeypatch.setattr(dup, "read_json", lambda p, *a, **k: seen.append(Path(p)) or real(p, *a, **k))
    dup.for_class(cfg.data_dir, SLUG)
    assert seen and not any(p.is_relative_to(other) for p in seen)


def test_the_decoded_length_replaces_the_one_from_the_file_header(cfg):
    from lecture_recorder.pipeline.run import _remember_in_meta

    d = lecture(cfg, A, duration=100.0)
    _remember_in_meta(d, 123.4, {})
    assert read_json(d / "meta.json")["duration_s"] == 123.4


def test_a_lecture_still_recording_is_left_out(cfg):
    lecture(cfg, A)
    d = lecture(cfg, B)
    (d / "audio.pcm").write_bytes(b"\0" * 10)  # "unfinished"
    assert signals(cfg) == []


# ------------------------------------------------------------------ signal 3: transcript similarity
def test_same_lecture_transcribed_twice_is_flagged_and_a_different_lecture_is_not(cfg):
    w = words(1)
    lecture(cfg, A, words=noisy(w, 2), created="2026-09-01T09:00:00+00:00")
    lecture(cfg, B, words=noisy(w, 3), created="2026-09-08T09:00:00+00:00")
    other = "2026-09-03_09-00-00"
    lecture(cfg, other, words=words(9), created="2026-09-03T09:00:00+00:00")
    assert signals(cfg) == ["content"]
    assert not dup.for_class(cfg.data_dir, SLUG).get(other)


def test_a_recap_of_part_of_a_lecture_is_a_near_miss(cfg):
    w = words(1)
    lecture(cfg, A, words=noisy(w, 2, 0.1), created="2026-09-01T09:00:00+00:00")
    lecture(cfg, B, words=noisy(w[:200] + words(5, 1800), 3, 0.1), created="2026-09-08T09:00:00+00:00")
    assert signals(cfg) == []


def test_short_transcripts_are_not_compared():
    assert dup.minhash(["word"] * (dup.MIN_WORDS - 1)) is None


def test_a_sketch_is_rewritten_only_after_the_transcript_changes(cfg):
    d = lecture(cfg, A, words=words(1))
    first = (d / dup.SKETCH_FILE).stat().st_mtime_ns
    dup.sketch_session(d)
    assert (d / dup.SKETCH_FILE).stat().st_mtime_ns == first
    write_json_atomic(d / "transcript.json", {"segments": [{"text": " ".join(words(2))}]}, fsync=False)
    before = read_json(d / dup.SKETCH_FILE)["minhash"]
    dup.sketch_session(d)
    assert read_json(d / dup.SKETCH_FILE)["minhash"] != before


def test_detection_runs_at_the_end_of_a_transcription_and_needs_no_gpu_job(cfg, monkeypatch):
    w = words(1)
    older = lecture(cfg, A, words=noisy(w, 2), created="2026-09-01T09:00:00+00:00")
    (older / dup.SKETCH_FILE).unlink()  # an older lecture catches up too
    d = lecture(cfg, B, created="2026-09-08T09:00:00+00:00")
    q.enqueue(cfg.data_dir, d)
    kinds = []

    def job(cmd, **kw):
        kinds.append(cmd[cmd.index("--task") + 1])
        if kinds[-1] == "transcribe":
            doc = {"segments": [{"text": " ".join(noisy(w, 3))}]}
            write_json_atomic(d / "transcript.json", doc, fsync=False)
            write_json_atomic(d / "transcript.raw.json", doc, fsync=False)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(q, "_run_job", job)
    q.run_worker(cfg.data_dir)
    assert kinds == ["transcribe", "summarize", "embed"]  # no job of its own
    assert signals(cfg) == ["content"]


@pytest.mark.parametrize("boom", ["sketch_scope", "minhash"])
def test_a_failing_check_never_breaks_the_job(cfg, monkeypatch, boom, caplog):
    d = lecture(cfg, A, words=words(1))
    (d / dup.SKETCH_FILE).unlink()
    q.enqueue(cfg.data_dir, d)
    monkeypatch.setattr(q, "_run_job", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""))

    def fail(*a):
        raise RuntimeError("boom")

    monkeypatch.setattr(dup, boom, fail)
    done = q.run_worker(cfg.data_dir)
    assert [j["state"] for j in done] == ["done", "done", "done"]
    assert "boom" in caplog.text


def test_a_failing_check_hides_the_hint_not_the_page(cfg, client, monkeypatch):
    lecture(cfg, A)
    monkeypatch.setattr(dup, "_rows", lambda d: 1 / 0)
    assert client.get(f"/api/classes/{SLUG}").json()["sessions"][0]["duplicate"] is False
    assert client.get(f"/api/sessions/{SLUG}/{A}/duplicates").json() == {"matches": []}


def test_a_transcript_held_by_a_scanner_is_retried_not_marked_as_empty(cfg, monkeypatch):
    d = lecture(cfg, A, words=words(1))
    (d / dup.SKETCH_FILE).unlink()
    real = dup.read_json
    monkeypatch.setattr(dup, "read_json", lambda p, *a, **k: None if Path(p).name == "transcript.json" else real(p, *a, **k))
    dup.after_transcribe(cfg.data_dir, d)
    assert not (d / dup.SKETCH_FILE).exists()
    monkeypatch.setattr(dup, "read_json", real)
    dup.after_transcribe(cfg.data_dir, d)
    assert read_json(d / dup.SKETCH_FILE)["minhash"]


def test_a_broken_duplicates_module_or_a_busy_queue_cannot_stop_the_step_or_the_worker(cfg, monkeypatch):
    d = lecture(cfg, A, words=words(1))
    job = {"session": str(d), "kind": q.TRANSCRIBE}
    monkeypatch.setitem(sys.modules, "lecture_recorder.duplicates", None)  # the import itself fails
    monkeypatch.setattr(q, "_queue_follow_up", lambda *a: None)
    q._follow_up(cfg.data_dir, job, True)  # no exception
    monkeypatch.delitem(sys.modules, "lecture_recorder.duplicates")
    (d / dup.SKETCH_FILE).unlink()

    def busy(*a):
        raise TimeoutError("queue is busy; try again")

    monkeypatch.setattr(q, "_queue_follow_up", busy)
    with pytest.raises(TimeoutError):
        q._follow_up(cfg.data_dir, job, True)
    assert (d / dup.SKETCH_FILE).exists()  # the sketch still ran


# ------------------------------------------------------------------ what the pages get
def test_the_api_describes_both_sides_and_preselects_the_unprocessed_one(cfg, client):
    lecture(cfg, A, words=words(1), title="Mic copy")
    lecture(cfg, B, title="Zoom copy", duration=3000, source={"kind": "import", "zoom": True, "file": "a.m4a"}, date_from="Zoom folder name")
    rows = {s["name"]: s for s in client.get(f"/api/classes/{SLUG}").json()["sessions"]}
    assert rows[A]["duplicate"] and rows[B]["duplicate"]
    m = client.get(f"/api/sessions/{SLUG}/{A}/duplicates").json()["matches"][0]
    assert m["labels"] == ["recorded at the same time"] and m["other"]["title"] == "Zoom copy"
    assert m["this"]["source"] == "Microphone" and m["other"]["source"] == "Zoom" and m["this"]["processed"] and not m["other"]["processed"]
    assert m["move"] == "other"
    assert client.get(f"/api/sessions/{SLUG}/{B}/duplicates").json()["matches"][0]["move"] == "this"  # the same lecture


def test_with_both_processed_the_shorter_one_is_preselected_and_no_text_is_returned(cfg, client):
    lecture(cfg, A, words=words(1), duration=3000)
    lecture(cfg, B, words=words(2), duration=2800)
    r = client.get(f"/api/sessions/{SLUG}/{A}/duplicates")
    assert r.json()["matches"][0]["move"] == "other" and "w1" not in r.text.replace(A, "").replace(B, "")


def test_several_matches_are_all_listed(cfg, client):
    lecture(cfg, A)
    lecture(cfg, B)
    lecture(cfg, "2026-09-01_09-10-00")
    assert len(client.get(f"/api/sessions/{SLUG}/{A}/duplicates").json()["matches"]) == 2


# ------------------------------------------------------------------ Keep both
def test_keep_both_is_remembered_for_that_pair_only_and_survives_a_restart(cfg, client):
    lecture(cfg, A)
    lecture(cfg, B)
    third = "2026-09-01_09-10-00"
    lecture(cfg, third)
    r = client.post(f"/api/sessions/{SLUG}/{A}/duplicates/dismiss", headers=H, json={"class": SLUG, "session": B})
    assert r.status_code == 200
    assert [m["other"]["session"] for m in client.get(f"/api/sessions/{SLUG}/{A}/duplicates").json()["matches"]] == [third]
    with TestClient(create_app(cfg, autoprocess=False, monitor=False), base_url=BASE) as again:
        assert [m["other"]["session"] for m in again.get(f"/api/sessions/{SLUG}/{B}/duplicates").json()["matches"]] == [third]


def test_two_quick_keep_both_clicks_both_stay(cfg):
    d = lecture(cfg, A)
    others = [lecture(cfg, f"2026-09-01_09-{m:02d}-00").name for m in range(10, 18)]
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda n: dup.dismiss(cfg.data_dir, SLUG, A, SLUG, n), others))
    assert sorted(read_json(d / "meta.json")["dismissed_duplicates"]) == [f"{SLUG}/{n}" for n in others]


def test_keep_both_never_overwrites_an_unreadable_meta(cfg):
    d = lecture(cfg, A)
    lecture(cfg, B)
    (d / "meta.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(dup.MetaUnreadable):
        dup.dismiss(cfg.data_dir, SLUG, A, SLUG, B)
    assert (d / "meta.json").read_text(encoding="utf-8") == "{broken"


def test_keep_both_refuses_a_lecture_that_does_not_exist_and_needs_the_app_header(cfg, client):
    lecture(cfg, A)
    url = f"/api/sessions/{SLUG}/{A}/duplicates/dismiss"
    assert client.post(url, headers=H, json={"class": SLUG, "session": B}).status_code == 404
    assert client.post(url, headers=H, json={"class": "../..", "session": "x"}).status_code == 404
    assert client.post(url, json={"class": SLUG, "session": A}).status_code == 403


# ------------------------------------------------------------------ Move to Recently deleted + Undo
def test_moving_one_clears_the_flag_and_undo_brings_lecture_and_flag_back(cfg, client):
    lecture(cfg, A)
    d = lecture(cfg, B)
    before = sorted(p.name for p in d.iterdir())
    r = client.post(f"/api/sessions/{SLUG}/{B}/delete", headers=H)
    assert r.status_code == 200 and not d.exists()
    assert client.get(f"/api/sessions/{SLUG}/{A}/duplicates").json()["matches"] == []
    assert [i["id"] for i in trash.list_trash(cfg.data_dir)] == [r.json()["id"]]  # restorable: nothing permanent
    assert client.post(f"/api/trash/{r.json()['id']}/restore", headers=H).status_code == 200
    assert sorted(p.name for p in d.iterdir()) == before and signals(cfg) == ["time"]


def test_the_module_never_deletes_anything():
    src = Path(dup.__file__).read_text(encoding="utf-8")
    assert not any(w in src for w in ("unlink", "rmtree", "os.remove", "os.rename", "delete_"))


# ------------------------------------------------------------------ import warning
def wav(seconds: float = 1.0, tone: int = 1) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(bytes([tone, 0]) * int(16000 * seconds))
    return out.getvalue()


def upload(client, body: bytes, name="clip.wav", **form):
    return client.post("/api/import", headers=H, files={"file": (name, body)},
                       data={"class_slug": SLUG, "last_modified_ms": "1788000000000", **form})


def sessions(cfg) -> list[str]:
    return [s["name"] for s in lib.list_sessions(cfg.data_dir, SLUG)]


def test_import_stores_the_hash_and_length_and_a_repeat_is_refused_until_import_anyway(cfg, client):
    body = wav(2.0)
    first = upload(client, body).json()
    meta = read_json(cfg.data_dir / "classes" / SLUG / "sessions" / first["session"] / "meta.json")
    assert meta["source"]["sha256"] == hashlib.sha256(body).hexdigest() and meta["duration_s"] == 2.0
    again = upload(client, body, name="another name.wav")
    assert again.status_code == 409 and again.json()["already_imported"]["session"] == first["session"]
    assert "Already imported as clip" in again.json()["detail"]
    assert len(sessions(cfg)) == 1
    assert not list((cfg.data_dir / "classes" / SLUG / "sessions").glob(".import-*"))  # the upload left nothing
    assert upload(client, body, anyway="true").status_code == 201 and len(sessions(cfg)) == 2
    assert signals(cfg, first["session"]) == ["file"]  # imported anyway: still flagged on the lecture page


def test_a_different_file_imports_normally(cfg, client):
    assert upload(client, wav(tone=1)).status_code == 201
    assert upload(client, wav(tone=2)).status_code == 201


def test_zoom_import_warns_too_and_keeps_nothing(cfg, client):
    body = b"not really m4a"
    def post(**form):
        return client.post("/api/import/zoom", headers=H, data={"class_slug": SLUG, "folder": "2026-09-01 09.00.00 Physics", **form},
                           files={"audio": ("audio1.m4a", body), "chat": ("chat.txt", b"hi")})
    assert post().status_code == 201
    r = post()
    assert r.status_code == 409 and r.json()["already_imported"]["title"] == "Physics"
    assert not list((cfg.data_dir / "classes" / SLUG / "sessions").glob(".import-*"))
    assert post(anyway="true").status_code == 201


def test_cli_import_zoom_refuses_a_repeat_without_anyway(cfg, tmp_path, monkeypatch, capsys):
    from lecture_recorder import cli

    src = tmp_path / "2026-09-01 09.00.00 Physics"
    src.mkdir()
    (src / "audio1.m4a").write_bytes(b"not really m4a")
    monkeypatch.setattr("lecture_recorder.config.load_config", lambda *a: cfg)
    base = ["prog", "import-zoom", str(src), "--class", SLUG, "--data-dir", str(cfg.data_dir)]
    run = lambda *extra: (monkeypatch.setattr("sys.argv", base + list(extra)), cli.main())[1]  # noqa: E731
    assert run() == 0 and len(sessions(cfg)) == 1
    capsys.readouterr()
    assert run() == 2 and "Already imported as Physics" in capsys.readouterr().out and len(sessions(cfg)) == 1
    assert run("--anyway") == 0 and len(sessions(cfg)) == 2


def test_a_failing_check_does_not_stop_an_import(cfg, client, monkeypatch):
    monkeypatch.setattr(dup, "_rows", lambda d: 1 / 0)
    assert upload(client, wav()).status_code == 201


def test_no_new_timers_or_polling_in_the_page():
    js = (Path(__file__).parents[1] / "src" / "lecture_recorder" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert js.count("startPoll(") == 3 and js.count("setInterval(") == 2
    assert json.dumps("dup-banner")[1:-1] in js
