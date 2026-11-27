"""User decisions: manual processing by default (A), the GPU rule (never while recording or on
battery), and WAV retention with a confirmed cleanup (B). Temp folders and a simulated power source
only; no GPU work and nothing in data/ is touched.
"""
import dataclasses
import os
import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import lecture_recorder  # noqa: F401  (lockdown)
from lecture_recorder import library as lib
from lecture_recorder import machine
from lecture_recorder.config import ProcessingConfig, RetentionConfig, load_config
from lecture_recorder.pipeline import queue as q
from lecture_recorder.recorder.control import start_recording
from lecture_recorder.recorder.session import read_json
from lecture_recorder.retention import compact, compact_plan, compress_session
from lecture_recorder.web.app import Recorder, create_app
from test_web_app import BASE, H, write_tone_wav

AUTO = ProcessingConfig("auto")


@pytest.fixture
def power(monkeypatch):
    """Simulated power source: power["now"] = "ac" | "battery"."""
    state = {"now": "ac"}
    monkeypatch.setattr(machine, "power_source", lambda: state["now"])
    return state


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LECTURE_RECORDER_HOME", str(home))
    c = load_config(home)
    return dataclasses.replace(c, data_dir=home / "data")


def session(data: Path, name: str, slug: str = "physics-101") -> Path:
    if not (data / "classes" / slug / "class.json").exists():
        lib.create_class(data, "Physics 101")
    d = data / "classes" / slug / "sessions" / name
    d.mkdir(parents=True)
    write_tone_wav(d / "audio.wav")
    (d / "meta.json").write_text('{"name": "%s", "process_on_stop": true}' % name, encoding="utf-8")
    return d


class FakeRun:
    """Stands in for the job subprocess; records which job kinds were started."""

    def __init__(self, during=None):
        self.kinds, self.during = [], during

    def __call__(self, cmd, **kw):
        self.kinds.append(cmd[cmd.index("--task") + 1])
        if self.during:
            self.during()

        class R:
            returncode, stdout, stderr = 0, "", ""
        return R()


# ------------------------------------------------------------------ A: manual by default
def test_processing_is_manual_by_default(cfg):
    assert cfg.processing.mode == "manual"
    with pytest.raises(ValueError):
        ProcessingConfig("sometimes")


def test_manual_stop_and_import_only_save_and_never_auto_start(cfg, power, monkeypatch):
    started = []
    monkeypatch.setattr("lecture_recorder.web.app.run_worker", lambda *a, **k: started.append(1))
    lib.create_class(cfg.data_dir, "Physics 101")
    leftover = session(cfg.data_dir, "2026-09-01_09-00-00")
    q.enqueue(cfg.data_dir, leftover)  # queued in an earlier run
    app = create_app(cfg, autoprocess=True, monitor=False, synthetic_source={"pattern": "tone"})
    with TestClient(app, base_url=BASE) as c:
        assert c.post("/api/record/start", headers=H, json={"class": "physics-101"}).status_code == 201
        name = c.get("/api/record/status").json()["session"]
        assert c.post("/api/record/stop", headers=H).status_code == 200
        r = c.post("/api/import", headers=H, files={"file": ("talk.wav", (leftover / "audio.wav").read_bytes())},
                   data={"class_slug": "physics-101"})
        assert r.status_code == 201
        assert c.get(f"/api/sessions/physics-101/{name}").json()["state"] == "not processed"
        assert c.get(f"/api/sessions/physics-101/{r.json()['session']}").json()["state"] == "not processed"
        app.state.recorder.check_waiting()  # monitor tick
    assert [Path(j["session"]).name for j in q.jobs(cfg.data_dir)] == [leftover.name]  # nothing new queued
    assert started == [] and app.state.recorder._worker is None  # and nothing started


def test_process_all_pending_queues_each_unprocessed_session_once(cfg, power, monkeypatch):
    monkeypatch.setattr("lecture_recorder.web.app.run_worker", lambda *a, **k: [])
    a, b = session(cfg.data_dir, "2026-09-01_09-00-00"), session(cfg.data_dir, "2026-09-02_09-00-00")
    done = session(cfg.data_dir, "2026-09-03_09-00-00")
    (done / "transcript.raw.json").write_text("{}", encoding="utf-8")
    (cfg.data_dir / "classes" / "physics-101" / "sessions" / "2026-09-04_09-00-00").mkdir()  # no audio
    app = create_app(cfg, autoprocess=False, monitor=False)
    with TestClient(app, base_url=BASE) as c:
        assert c.get("/api/processing").json()["unprocessed"] == 2
        assert c.post("/api/process-pending", headers=H).json() == {"queued": 2, "blocked": None}
        assert c.post("/api/process-pending", headers=H).json()["queued"] == 0
        assert c.get("/api/processing").json()["unprocessed"] == 0
    assert sorted(Path(j["session"]).name for j in q.jobs(cfg.data_dir)) == [a.name, b.name]
    assert {q.kind_of(j) for j in q.jobs(cfg.data_dir)} == {"transcribe"}


def test_battery_blocks_new_jobs_and_a_running_job_finishes(cfg, power, monkeypatch):
    d1, d2 = session(cfg.data_dir, "2026-09-01_09-00-00"), session(cfg.data_dir, "2026-09-02_09-00-00")
    q.enqueue(cfg.data_dir, d1)
    q.enqueue(cfg.data_dir, d2)
    power["now"] = "battery"
    fake = FakeRun()
    monkeypatch.setattr(q, "_run_job", fake)
    blocked = []
    assert q.run_worker(cfg.data_dir, blocked=blocked) == [] and blocked == ["on battery"] and fake.kinds == []
    # plugged in, then unplugged during the first job: it finishes, the rest wait
    power["now"] = "ac"
    fake = FakeRun(during=lambda: power.update(now="battery"))
    monkeypatch.setattr(q, "_run_job", fake)
    blocked = []
    done = q.run_worker(cfg.data_dir, blocked=blocked)
    assert [j["state"] for j in done] == ["done"] and fake.kinds == ["transcribe"] and blocked == ["on battery"]
    # same-second jobs run in random-id order, so check whichever session ran first
    states = {(Path(j["session"]).name, q.kind_of(j)): j["state"] for j in q.jobs(cfg.data_dir)}
    first = Path(done[0]["session"]).name
    other = ({d1.name, d2.name} - {first}).pop()
    assert states == {(first, "transcribe"): "done", (other, "transcribe"): "queued", (first, "summarize"): "queued"}


def test_recording_blocks_new_jobs(cfg, power, monkeypatch):
    d = session(cfg.data_dir, "2026-09-01_09-00-00")
    q.enqueue(cfg.data_dir, d)
    fake = FakeRun()
    monkeypatch.setattr(q, "_run_job", fake)
    h = start_recording("physics-101", config=cfg, synthetic={"pattern": "tone"})
    try:
        assert machine.gpu_block_reason(cfg.data_dir) == machine.RECORDING
        blocked = []
        q.run_worker(cfg.data_dir, blocked=blocked)
        assert blocked == [machine.RECORDING] and fake.kinds == []
    finally:
        h.stop()
    assert machine.gpu_block_reason(cfg.data_dir) is None
    q.run_worker(cfg.data_dir)
    assert fake.kinds == ["transcribe", "summarize"]  # the chain continues once recording ended


@pytest.mark.parametrize("mode,starts", [("manual", False), ("auto", True)])
def test_plugging_in_resumes_waiting_jobs_only_in_auto_mode(cfg, power, monkeypatch, mode, starts):
    calls = []
    monkeypatch.setattr("lecture_recorder.web.app.run_worker", lambda *a, **k: calls.append(1) or [])
    rec = Recorder(dataclasses.replace(cfg, processing=ProcessingConfig(mode)), autoprocess=True)
    rec.waiting = "on battery"
    power["now"] = "battery"
    rec.check_waiting()
    assert calls == []
    power["now"] = "ac"  # plugged in
    rec.check_waiting()
    if rec._worker:
        rec._worker.join(5)
    assert bool(calls) is starts


def test_auto_mode_queues_on_stop_but_waits_on_battery(cfg, power, monkeypatch):
    calls = []
    monkeypatch.setattr("lecture_recorder.web.app.run_worker",
                        lambda data_dir, blocked=None, **k: calls.append(1) or (blocked.append("on battery") if blocked is not None else None) or [])
    lib.create_class(cfg.data_dir, "Physics 101")
    app = create_app(dataclasses.replace(cfg, processing=AUTO), autoprocess=True, monitor=False,
                     synthetic_source={"pattern": "tone"})
    with TestClient(app, base_url=BASE) as c:
        c.post("/api/record/start", headers=H, json={"class": "physics-101"})
        c.post("/api/record/stop", headers=H)
        rec = app.state.recorder
        if rec._worker:
            rec._worker.join(5)
        assert len(q.jobs(cfg.data_dir)) == 1 and calls and rec.waiting == "on battery"
        st = c.get("/api/processing").json()
        assert st["mode"] == "auto" and st["queued"] == 1 and st["waiting"] == "on battery"


def test_cli_process_reports_why_jobs_wait(cfg, power, monkeypatch, capsys):
    from lecture_recorder.cli import main

    d = session(cfg.data_dir, "2026-09-01_09-00-00")
    power["now"] = "battery"
    monkeypatch.setattr(q, "_run_job", FakeRun())
    monkeypatch.setattr("lecture_recorder.config.load_config", lambda home=None: cfg)
    assert main(["process", str(d)]) == 3
    assert "Waiting: on battery" in capsys.readouterr().out


def test_search_by_meaning_needs_the_gpu_and_is_paused_on_battery(cfg, power, monkeypatch):
    import numpy as np

    app = create_app(cfg, autoprocess=False, monitor=False)
    idx = app.state.search
    monkeypatch.setattr(idx, "concept_status", lambda **k: {"transcribed": 1, "indexed": 1, "pending": 0, "windows": 3})
    monkeypatch.setattr(idx, "concept", lambda vec, cls=None, **k: [])
    monkeypatch.setattr("lecture_recorder.search.embed.embed_query", lambda q, c: np.ones(4, dtype=np.float32))
    power["now"] = "battery"
    with TestClient(app, base_url=BASE) as c:
        r = c.get("/api/search", params={"q": "energy", "mode": "concept"})
        assert r.status_code == 503 and "on battery" in r.json()["detail"]
        assert c.get("/api/search", params={"q": "energy"}).status_code == 200  # exact search still works
        power["now"] = "ac"
        assert c.get("/api/search", params={"q": "energy", "mode": "concept"}).status_code == 200


@pytest.mark.real_power
def test_real_power_status_is_readable():
    assert machine.power_source() in ("ac", "battery", "unknown")


@pytest.mark.real_power
@pytest.mark.skipif(os.name != "nt", reason="reads the Win32 power status")
def test_the_real_gate_blocks_gpu_jobs_on_battery_with_nothing_replaced_but_windows(tmp_path, monkeypatch):
    """The real power_source and gpu_block_reason, fed by a fake Win32 answer: conftest.py's "plugged in"
    is off here, so this proves the gate itself, whatever the laptop is doing now."""
    import ctypes

    def status(ac_line):
        def fake(ref):
            ref._obj.ACLineStatus = ac_line
            return 1
        return fake

    monkeypatch.setattr(ctypes.windll.kernel32, "GetSystemPowerStatus", status(0))
    assert machine.power_source() == "battery" and machine.gpu_block_reason(tmp_path) == machine.ON_BATTERY
    q.enqueue(tmp_path, tmp_path)
    monkeypatch.setattr(q, "_run_job", lambda *a, **k: pytest.fail("a job started on battery"))
    blocked = []
    assert q.run_worker(tmp_path, blocked=blocked) == [] and blocked == [machine.ON_BATTERY]
    assert q.jobs(tmp_path)[0]["state"] == "queued"
    monkeypatch.setattr(ctypes.windll.kernel32, "GetSystemPowerStatus", status(1))
    assert machine.power_source() == "ac" and machine.gpu_block_reason(tmp_path) is None


def test_the_app_has_no_switch_that_turns_the_power_gate_off():
    src = Path(machine.__file__).read_text(encoding="utf-8")
    assert "environ" not in src and "getenv" not in src  # tests replace power_source in their own process only


# ------------------------------------------------------------------ settings saved to config.toml
def test_settings_save_to_config_toml_and_keep_other_settings(cfg):
    (cfg.home / "config.toml").write_text('# mine\n[server]\nport = 9000\n\n[processing]\nmode = "manual"  # old\n',
                                          encoding="utf-8")
    app = create_app(cfg, autoprocess=False, monitor=False)
    with TestClient(app, base_url=BASE) as c:
        assert c.put("/api/settings", json={"processing_mode": "auto"}).status_code == 403  # app header needed
        assert c.put("/api/settings", headers=H, json={"processing_mode": "auto"}).json()["mode"] == "auto"
        assert c.put("/api/settings", headers=H, json={"processing_mode": "always"}).status_code == 400
        r = c.put("/api/settings", headers=H, json={"delete_wav_after_opus": True})
        assert r.status_code == 400 and "permanent" in r.json()["detail"]
        assert c.put("/api/settings", headers=H, json={"delete_wav_after_opus": "yes",
                                                       "confirm_permanent_deletion": True}).status_code == 400
        r = c.put("/api/settings", headers=H, json={"delete_wav_after_opus": True, "confirm_permanent_deletion": True})
        assert r.json()["delete_original_after_opus"] is True  # the older name is still accepted
    text = (cfg.home / "config.toml").read_text(encoding="utf-8")
    parsed = tomllib.loads(text)
    assert text.startswith("# mine") and parsed["server"]["port"] == 9000
    assert parsed["processing"]["mode"] == "auto" and parsed["retention"] == {"delete_original_after_opus": True}
    assert load_config(cfg.home).processing.mode == "auto"


# ------------------------------------------------------------------ B: WAV retention
def processed(data: Path, name: str, opus: str = "good") -> Path:
    d = session(data, name)
    (d / q.RAW).write_text("{}", encoding="utf-8")  # transcribed: compaction only touches transcribed lectures
    if opus != "none":
        compress_session(d)  # writes and verifies audio.opus, keeps the WAV
    if opus == "unverified":
        rec = read_json(d / "retention.json")
        (d / "retention.json").write_text('{"verified": false}', encoding="utf-8")
        assert rec["verified"]
    if opus == "corrupt-later":  # verified at the time, damaged since
        (d / "audio.opus").write_bytes(b"OggS" + b"\0" * 100)
    return d


def test_wav_deletion_is_off_by_default_and_cleanup_refused_while_off(cfg, capsys):
    from lecture_recorder.cli import main

    assert cfg.retention.delete_original_after_opus is False and RetentionConfig().delete_original_after_opus is False
    d = processed(cfg.data_dir, "2026-09-01_09-00-00")
    assert (d / "audio.wav").exists() and read_json(d / "retention.json")["original_kept"] is True
    app = create_app(cfg, autoprocess=False, monitor=False)
    with TestClient(app, base_url=BASE) as c:
        assert c.get("/api/compact").status_code == 409
        assert c.post("/api/compact", headers=H, json={"token": "x", "confirm": True}).status_code == 409
    assert main(["cleanup-wav", "--data-dir", str(cfg.data_dir)]) == 2
    assert main(["compact", "--data-dir", str(cfg.data_dir)]) == 2
    assert (d / "audio.wav").exists()


def test_dry_run_deletes_nothing_and_never_a_wav_without_a_verified_opus(cfg):
    data = cfg.data_dir
    good = processed(data, "2026-09-01_09-00-00")
    none = processed(data, "2026-09-02_09-00-00", opus="none")
    unverified = processed(data, "2026-09-03_09-00-00", opus="unverified")
    corrupt = processed(data, "2026-09-04_09-00-00", opus="corrupt-later")
    before = sorted(p.name for p in data.rglob("audio.wav"))
    plan = compact_plan(data, encode=False)
    assert sorted(p.name for p in data.rglob("audio.wav")) == before and len(before) == 4  # dry run
    assert [x["session"].split("/")[1] for x in plan["items"]] == [good.name, corrupt.name]
    reasons = {x["session"].split("/")[1]: x["reason"] for x in plan["skipped"]}
    assert reasons == {none.name: "no Opus copy yet (`lecture-recorder compact` makes one)",
                       unverified.name: "Opus copy failed verification (`lecture-recorder compact` makes it again)"}
    with pytest.raises(ValueError):
        compact(data, "not-the-token", encode=False)
    assert len(list(data.rglob("audio.wav"))) == 4
    r = compact(data, plan["token"], encode=False)
    assert [x["session"].split("/")[1] for x in r["done"]] == [good.name]
    assert [x["session"].split("/")[1] for x in r["kept"]] == [corrupt.name]  # re-verified right before deleting
    assert not (good / "audio.wav").exists() and (good / "audio.opus").exists()
    assert read_json(good / "retention.json")["original_kept"] is False
    for d in (none, unverified, corrupt):
        assert (d / "audio.wav").exists()


def test_cleanup_needs_confirmation_in_app_and_cli(cfg, monkeypatch, capsys):
    from lecture_recorder.cli import main

    on = dataclasses.replace(cfg, retention=RetentionConfig(delete_original_after_opus=True))
    d = processed(cfg.data_dir, "2026-09-01_09-00-00")
    app = create_app(on, autoprocess=False, monitor=False)
    with TestClient(app, base_url=BASE) as c:
        plan = c.get("/api/compact").json()
        assert len(plan["items"]) == 1 and (d / "audio.wav").exists()
        assert c.post("/api/compact", headers=H, json={"token": plan["token"]}).status_code == 400
    monkeypatch.setattr("lecture_recorder.cli._cfg", lambda data_dir=None: on)
    monkeypatch.setattr("builtins.input", lambda prompt="": "yes")
    assert main(["cleanup-wav"]) == 1 and (d / "audio.wav").exists()
    assert main(["cleanup-wav", "--dry-run"]) == 0 and (d / "audio.wav").exists()
    assert "Nothing has been changed" in capsys.readouterr().out
    monkeypatch.setattr("builtins.input", lambda prompt="": "DELETE")
    assert main(["cleanup-wav"]) == 0 and not (d / "audio.wav").exists() and (d / "audio.opus").exists()


# ------------------------------------------------------------------ regressions
def test_cli_search_by_meaning_obeys_the_gpu_rule(cfg, power, monkeypatch, capsys):
    from lecture_recorder.cli import main
    from lecture_recorder.search.index import SearchIndex

    called = []
    monkeypatch.setattr("lecture_recorder.cli._cfg", lambda data_dir=None: cfg)
    monkeypatch.setattr(SearchIndex, "concept_status", lambda self, **k: {"windows": 3})
    monkeypatch.setattr("lecture_recorder.search.embed.embed_query", lambda *a, **k: called.append(1))
    power["now"] = "battery"
    assert main(["search", "entropy", "--concept"]) == 3
    assert "paused (on battery)" in capsys.readouterr().out and called == []


def test_wav_deletion_setting_is_read_when_deleting_not_at_job_start(cfg):
    from lecture_recorder.pipeline.run import _after_processing

    d = session(cfg.data_dir, "2026-09-01_09-00-00")
    started_with_on = dataclasses.replace(cfg, retention=RetentionConfig(delete_original_after_opus=True))
    (cfg.home / "config.toml").write_text("[retention]\ndelete_wav_after_opus = false\n", encoding="utf-8")  # turned off since
    _after_processing(d, started_with_on)
    assert (d / "audio.wav").exists() and read_json(d / "retention.json")["verified"] is True


def test_job_process_blocked_at_its_own_start_goes_back_to_the_queue(cfg, power, monkeypatch):
    d = session(cfg.data_dir, "2026-09-01_09-00-00")
    q.enqueue(cfg.data_dir, d)

    def blocked_child(cmd, **kw):
        power["now"] = "battery"  # unplugged between the worker's check and the job's start

        class R:
            returncode, stdout, stderr = q.EXIT_BLOCKED, "on battery\n", ""
        return R()

    monkeypatch.setattr(q, "_run_job", blocked_child)
    blocked = []
    assert q.run_worker(cfg.data_dir, blocked=blocked) == [] and blocked == ["on battery"]
    assert [j["state"] for j in q.jobs(cfg.data_dir)] == ["queued"]


def test_pipeline_job_refuses_to_start_on_battery(cfg, power):
    from lecture_recorder.pipeline.__main__ import main as job_main

    d = session(cfg.data_dir, "2026-09-01_09-00-00")
    power["now"] = "battery"
    assert job_main([str(d), "--data-dir", str(cfg.data_dir)]) == q.EXIT_BLOCKED
    assert not (d / "transcript.raw.json").exists()


def test_stale_running_job_is_shown_as_queued_at_startup_in_manual_mode(cfg, power, monkeypatch):
    started = []
    monkeypatch.setattr("lecture_recorder.web.app.run_worker", lambda *a, **k: started.append(1))
    d = session(cfg.data_dir, "2026-09-01_09-00-00")
    job = q.enqueue(cfg.data_dir, d)
    job.update(state="running")
    q._save(job)  # the app crashed mid-job
    create_app(cfg, autoprocess=True, monitor=False)
    assert [j["state"] for j in q.jobs(cfg.data_dir)] == ["queued"] and started == []


def test_concurrent_settings_saves_never_corrupt_config(cfg):
    import threading

    from lecture_recorder.settings import set_config_values

    errors = []

    def save(i):
        try:
            set_config_values(cfg.home, {("processing", "mode"): "auto" if i % 2 else "manual",
                                         ("retention", "delete_original_after_opus"): bool(i % 3 == 0)})
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=save, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    parsed = tomllib.loads((cfg.home / "config.toml").read_text(encoding="utf-8"))
    assert errors == [] and parsed["processing"]["mode"] in ("auto", "manual")
    assert not list(cfg.home.glob("config.toml.*.tmp"))
