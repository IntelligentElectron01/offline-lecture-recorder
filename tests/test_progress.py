"""Per-item processing progress. The progress file (throttled, atomic, never fails a job), the honesty
rules of `describe`, the job-side counters, the worker's progress file, and the status API's "processing".
No GPU, no models; temp folders only."""
import json
import subprocess
from pathlib import Path

import pytest

import lecture_recorder  # noqa: F401  (lockdown)
from lecture_recorder.pipeline import progress as pr
from lecture_recorder.pipeline import queue as q
from lecture_recorder.recorder import session as rs
from lecture_recorder.recorder.platform import ProcessLock
from lecture_recorder.recorder.process import GLOBAL_LOCK
from lecture_recorder.recorder.session import now_iso, read_json, write_json_atomic
from test_delete import client, cfg  # noqa: F401  (fixtures)
from test_summary import FakeOllama, fake_llm
from test_search import FakeEmbed, doc_from_sentences, fake_embedder

SLUG = "bio"


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class Recorder:
    """Stands in for Progress: records stage() and update() calls."""

    def __init__(self):
        self.calls = []

    def stage(self, name, total=None, unit=None, sub=None):
        self.calls.append(("stage", name, total, unit))

    def update(self, done):
        self.calls.append(("update", done))

    def diarization_hook(self):
        return None


# ------------------------------------------------------------------ the progress file
def test_updates_are_throttled_and_a_stage_change_always_writes(tmp_path, monkeypatch):
    writes = []
    real = rs.write_json_atomic
    monkeypatch.setattr(rs, "write_json_atomic", lambda path, obj, **kw: writes.append(obj) or real(path, obj, **kw))
    clock = Clock()
    p = pr.Progress(tmp_path / "p.json", clock=clock, wall=clock)
    p.stage("transcribe", total=100, unit="s")
    assert len(writes) == 1
    for i in range(20):  # 20 updates within 1 s of the stage's write: none written
        clock.t += 0.05
        p.update(i)
    assert len(writes) == 1
    clock.t += 0.6  # 1.6 s after the last write
    p.update(30)
    p.update(31)
    assert len(writes) == 2 and writes[-1]["done"] == 30
    p.stage("speakers", sub="segmentation")  # a stage change writes at once
    assert len(writes) == 3 and writes[-1]["stage"] == "speakers"
    raw = json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))  # always whole, valid JSON
    assert raw["stage"] == "speakers" and raw["total"] is None and raw["updated"] == clock.t


def test_a_failing_write_never_raises(tmp_path, monkeypatch):
    p = pr.Progress(tmp_path / "missing-folder" / "p.json")
    p.stage("map", total=3, unit="part")  # the folder doesn't exist: OSError, swallowed
    p.update(1)
    monkeypatch.setattr(rs, "write_json_atomic", lambda *a, **k: (_ for _ in ()).throw(PermissionError("busy")))
    p2 = pr.Progress(tmp_path / "p.json")
    p2.stage("map", total=3, unit="part")
    assert not list(tmp_path.glob("*.tmp"))


def test_the_file_is_not_fsynced_and_retries_little(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(rs, "write_json_atomic", lambda path, obj, **kw: seen.update(kw))
    pr.Progress(tmp_path / "p.json").stage("load")
    assert seen == {"retries": 3, "fsync": False}


def test_from_env(tmp_path, monkeypatch):
    monkeypatch.delenv(pr.ENV, raising=False)
    assert pr.from_env() is pr.NULL
    pr.NULL.stage("x", total=3)  # no-op
    pr.NULL.update(1)
    monkeypatch.setenv(pr.ENV, str(tmp_path / "p.json"))
    p = pr.from_env()
    assert isinstance(p, pr.Progress) and p.path == tmp_path / "p.json"


def test_diarization_hook_maps_pyannote_steps(tmp_path):
    clock = Clock()
    p = pr.Progress(tmp_path / "p.json", clock=clock, wall=clock, min_interval=0)
    hook = p.diarization_hook()
    p.stage("speakers")
    hook("segmentation", None, file={}, total=10, completed=0)
    hook("segmentation", None, file={}, total=10, completed=16)  # pyannote's last batch can overshoot
    raw = read_json(p.path)
    assert (raw["sub"], raw["done"], raw["total"]) == ("segmentation", 10, 10)
    hook("segmentation", object(), file={})  # the finished step's artifact: no change
    hook("speaker_counting", object(), file={})
    hook("embeddings", None, file={}, total=4, completed=0)
    hook("embeddings", None, file={}, total=4, completed=2)
    assert (read_json(p.path)["sub"], read_json(p.path)["done"]) == ("embeddings", 2)
    hook("embeddings", object(), file={})  # embeddings done: clustering has no counter
    raw = read_json(p.path)
    assert raw["sub"] == "clustering" and raw["total"] is None
    hook("bogus", None, total="x", completed=object())  # never raises


# ------------------------------------------------------------------ honesty rules
def snap(done, total, started, updated, **kw):
    return {"stage": "transcribe", "sub": None, "done": done, "total": total, "unit": "s",
            "stage_started": started, "updated": updated, **kw}


def test_describe_fraction_and_eta():
    now = 10_000.0
    d = pr.describe(snap(150, 100, now - 100, now), now)
    assert d["fraction"] == 1.0  # clamped
    assert pr.describe(snap(None, None, now - 100, now), now)["fraction"] is None  # no counter: indeterminate
    assert pr.describe({**snap(5, None, now - 100, now)}, now)["fraction"] is None
    assert pr.describe(snap(4, 100, now - 100, now), now)["eta_s"] is None  # below 5 %
    assert pr.describe(snap(50, 100, now - 19, now), now)["eta_s"] is None  # under 20 s into the step
    stale = pr.describe(snap(50, 100, now - 100, now - 31), now)
    assert stale["stale"] and stale["eta_s"] is None and stale["fraction"] == 0.5  # keeps its label and bar
    d = pr.describe(snap(25, 100, now - 62, now - 2), now)  # 60 s for 25 %: 180 s left at the update, 178 now
    assert d["fraction"] == 0.25 and d["eta_s"] == 178 and not d["stale"]
    assert pr.describe(None, now) is None and pr.describe({"stage": 5}, now) is None
    assert pr.describe({"stage": "x" * 100}, now) is None
    junk = pr.describe({"stage": "map", "sub": ["no"], "unit": 3, "done": True, "total": "7"}, now)
    assert junk["sub"] is None and junk["unit"] is None and junk["fraction"] is None and junk["stale"]


# ------------------------------------------------------------------ job-side counters
def test_map_parts_reports_parts_done_before_each_request():
    from lecture_recorder.summary.summarize import doc_evidence, map_parts

    doc = doc_from_sentences([f"Sentence {i} " + "about heat engines and their efficiency " * 12 for i in range(60)])
    llm = fake_llm(FakeOllama(), num_ctx=8192, max_prompt_tokens=3000, num_predict=500)
    rec = Recorder()
    parts = map_parts(doc, llm, doc_evidence(doc), rec)
    n = len(parts)
    assert n >= 3
    assert rec.calls == [("stage", "map", n, "part")] + [("update", i) for i in range(n)]


def test_reduce_parts_reports_condense_and_combine():
    from lecture_recorder.summary.summarize import doc_evidence, map_parts, reduce_parts

    doc = doc_from_sentences([f"Sentence {i} " + "about heat engines and their efficiency " * 12 for i in range(60)])
    llm = fake_llm(FakeOllama(notes_words=800), num_ctx=8192, max_prompt_tokens=3000, num_predict=500)
    ev = doc_evidence(doc)
    parts = map_parts(doc, llm, ev)
    rec = Recorder()
    reduce_parts(parts, llm, "T", ev, rec)
    stages = [c for c in rec.calls if c[0] == "stage"]
    assert stages[0][:2] == ("stage", "condense") and stages[0][3] == "part"
    assert stages[-1] == ("stage", "combine", None, None)
    groups = stages[0][2]
    assert rec.calls[1:1 + groups] == [("update", j) for j in range(groups)]


def test_embed_session_reports_passages(cfg):
    from lecture_recorder.search import embed as emb

    d = cfg.data_dir / "classes" / SLUG / "sessions" / "2026-10-01_09-00-00"
    d.mkdir(parents=True)
    write_json_atomic(d / "transcript.json", doc_from_sentences([f"Sentence number {i} about cells." for i in range(80)]))
    rec = Recorder()
    info = emb.embed_session(d, cfg, llm=fake_embedder(FakeEmbed(), cfg), progress=rec)
    n, b = info["count"], max(1, cfg.llm.embed_batch)
    assert rec.calls[0] == ("stage", "gpu_wait", None, None)
    assert rec.calls[1] == ("stage", "embed", n, "passage")
    assert rec.calls[2:] == [("update", i) for i in range(0, n, b)]


# ------------------------------------------------------------------ the worker
def test_worker_names_a_progress_file_and_removes_it(tmp_path, monkeypatch):
    seen = []

    def fake_run(cmd, **kw):
        path = Path(kw["env"][pr.ENV])
        pr.Progress(path).stage("transcribe", total=10, unit="s")  # the job writes it
        seen.append((path, path.is_file()))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(q, "_run_job", fake_run)
    monkeypatch.setattr(q, "gpu_block_reason", lambda d: None)
    d = tmp_path / "s"
    d.mkdir()
    job = q.enqueue(tmp_path, d)
    q.run_worker(tmp_path)
    assert seen[0] == (q.progress_path(tmp_path, job["id"]), True)
    assert len(seen) == 2 and seen[1][0] != seen[0][0]  # its summary follow-up gets its own file
    assert seen[0][0].parent.parent == tmp_path / "queue"  # a subfolder: never read as a job
    assert not any(path.exists() for path, _ in seen)
    assert [j["state"] for j in q.jobs(tmp_path)] == ["done", "done"]


def test_worker_removes_the_file_when_the_job_fails(tmp_path, monkeypatch):
    def fake_run(cmd, **kw):
        pr.Progress(Path(kw["env"][pr.ENV])).stage("load")
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(q, "_run_job", fake_run)
    monkeypatch.setattr(q, "gpu_block_reason", lambda d: None)
    d = tmp_path / "s"
    d.mkdir()
    job = q.enqueue(tmp_path, d)
    q.run_worker(tmp_path)
    assert not q.progress_path(tmp_path, job["id"]).exists()
    assert q.jobs(tmp_path)[0]["state"] == "failed"


def test_recovery_removes_leftover_progress_files(tmp_path):
    left = q.progress_path(tmp_path, "crashed")
    left.parent.mkdir(parents=True)
    left.write_text("{}", encoding="utf-8")
    q.recover_stale(tmp_path)
    assert not left.exists()


# ------------------------------------------------------------------ the status API
def make_session(cfg, name, title):
    d = cfg.data_dir / "classes" / SLUG / "sessions" / name
    d.mkdir(parents=True)
    write_json_atomic(d / "meta.json", {"name": name, "title": title, "class": SLUG, "created": name[:10] + "T09:00:00+00:00"})
    return d


def set_state(job, **kw):
    job.update(kw)
    write_json_atomic(job["_path"], {k: v for k, v in job.items() if k != "_path"})


def processing(client):
    return client.get("/api/record/status").json()["processing"]


def test_status_reports_no_jobs(cfg, client):
    p = processing(client)
    assert p["active"] == [] and p["waiting"] is None and p["recent_failed"] == []


def test_status_reports_running_and_queued_jobs(cfg, client):
    a = make_session(cfg, "2026-10-01_09-00-00", "Cells")
    b = make_session(cfg, "2026-10-02_09-00-00", "Osmosis")
    c = make_session(cfg, "2026-10-03_09-00-00", "Enzymes")
    run = q.enqueue(cfg.data_dir, a)
    set_state(run, state="running", started=now_iso(), created="2026-11-01T09:00:00+00:00")
    set_state(q.enqueue(cfg.data_dir, b), created="2026-11-01T09:00:01+00:00")  # FIFO by creation time
    set_state(q.enqueue(cfg.data_dir, c, q.SUMMARIZE), created="2026-11-01T09:00:02+00:00")
    prog = pr.Progress(q.progress_path(cfg.data_dir, run["id"]))
    prog.path.parent.mkdir(parents=True)
    prog.stage("transcribe", total=3600, unit="s")
    p = processing(client)
    assert [(j["session"], j["state"], j.get("position")) for j in p["active"]] == [
        (a.name, "running", None), (b.name, "queued", 1), (c.name, "queued", 2)]
    first = p["active"][0]
    assert first["title"] == "Cells" and first["class"] == SLUG and first["kind"] == "transcribe"
    assert first["progress"]["stage"] == "transcribe" and first["progress"]["fraction"] == 0
    assert p["active"][2]["kind"] == "summarize" and "title" not in p["active"][1]
    assert all(j["waiting"] is None for j in p["active"][1:])
    text = json.dumps(p)
    assert "segments" not in text and "words" not in text  # labels and numbers only


def test_status_without_a_progress_file_gives_null(cfg, client):
    run = q.enqueue(cfg.data_dir, make_session(cfg, "2026-10-01_09-00-00", "Cells"))
    set_state(run, state="running", started=now_iso())
    assert processing(client)["active"][0]["progress"] is None


def test_status_says_why_queued_jobs_wait(cfg, client):
    q.enqueue(cfg.data_dir, make_session(cfg, "2026-10-01_09-00-00", "Cells"))
    lock = ProcessLock(cfg.data_dir / GLOBAL_LOCK)
    assert lock.acquire()
    try:
        p = processing(client)
    finally:
        lock.release()
    assert p["waiting"] == "recording in progress" and p["active"][0]["waiting"] == "recording in progress"


def test_status_lists_recently_failed_lectures(cfg, client):
    d = make_session(cfg, "2026-10-01_09-00-00", "Cells")
    job = q.enqueue(cfg.data_dir, d)
    set_state(job, state="failed", finished=now_iso(), error="boom")
    old = q.enqueue(cfg.data_dir, make_session(cfg, "2026-10-02_09-00-00", "Old"))
    set_state(old, state="failed", finished="2026-01-01T00:00:00+00:00")
    assert processing(client)["recent_failed"] == [f"{SLUG}/{d.name}"]


def test_the_lecture_page_no_longer_polls_on_its_own():
    js = (Path(__file__).parents[1] / "src" / "lecture_recorder" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert js.count("startPoll(") == 3  # its definition, the 2 s status poll, the exam page's poll
    assert "setInterval(" in js and js.count("setInterval(") == 2  # both inside startPoll / visibility


def test_old_failed_jobs_are_archived_only_once_a_newer_job_replaces_them(tmp_path):
    """Regression: failed jobs were never archived, so the 2 s status poll's queue scan only grew."""
    d, other = tmp_path / "s", tmp_path / "t"
    d.mkdir()
    other.mkdir()
    old = "2026-01-01T00:00:00+00:00"
    first = q.enqueue(tmp_path, d)
    set_state(first, state="failed", finished=old, created="2026-01-01T00:00:00+00:00")
    retry = q.enqueue(tmp_path, d)
    set_state(retry, state="done", finished=old, created="2026-01-02T00:00:00+00:00")
    alone = q.enqueue(tmp_path, other)
    set_state(alone, state="failed", finished=old)
    q._archive_done(tmp_path)
    assert [j["id"] for j in q.jobs(tmp_path)] == [alone["id"]]  # the only failure that still shows stays
    assert (tmp_path / "queue" / "archive" / f"{first['id']}.json").exists()


def test_a_failed_summary_followed_by_its_embed_job_still_counts_as_failed(cfg, client):
    """Regression: the embed job queued after a failed summary hid the failure ("Done")."""
    d = make_session(cfg, "2026-10-01_09-00-00", "Cells")
    s = q.enqueue(cfg.data_dir, d, q.SUMMARIZE)
    set_state(s, state="failed", finished=now_iso(), created="2026-11-01T09:00:00+00:00")
    e = q.enqueue(cfg.data_dir, d, q.EMBED)
    set_state(e, state="done", finished=now_iso(), created="2026-11-01T09:00:01+00:00")
    assert processing(client)["recent_failed"] == [f"{SLUG}/{d.name}"]
    again = q.enqueue(cfg.data_dir, d, q.SUMMARIZE)  # a newer summary job replaces the failure
    set_state(again, state="done", finished=now_iso(), created="2026-11-01T09:00:02+00:00")
    assert processing(client)["recent_failed"] == []


def test_the_waiting_reason_is_live_not_the_workers_last_one(cfg, client):
    """Regression: self.waiting is refreshed only when the worker exits."""
    q.enqueue(cfg.data_dir, make_session(cfg, "2026-10-01_09-00-00", "Cells"))
    client.app.state.recorder.waiting = "recording in progress"  # stale: nothing blocks now
    p = processing(client)
    assert p["waiting"] is None and p["active"][0]["waiting"] is None


def test_the_job_runs_even_if_the_progress_folder_cannot_be_made(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(q, "_run_job", lambda cmd, **kw: seen.update(env=kw["env"]) or subprocess.CompletedProcess(cmd, 0, "", ""))
    monkeypatch.setattr(q, "gpu_block_reason", lambda d: None)
    (tmp_path / "queue").mkdir()
    (tmp_path / "queue" / q.PROGRESS_DIR).write_text("a file where the folder should be", encoding="utf-8")
    d = tmp_path / "s"
    d.mkdir()
    q.enqueue(tmp_path, d, q.EMBED)
    q.run_worker(tmp_path)
    assert pr.ENV not in seen["env"] and q.jobs(tmp_path)[0]["state"] == "done"
