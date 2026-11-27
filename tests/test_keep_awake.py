""""Keep the laptop awake while processing". The Win32 call (SetThreadExecutionState) is mocked;
temp folders and a simulated power source only, no GPU work, nothing in data/ is touched.
"""
import dataclasses

import pytest
from fastapi.testclient import TestClient

import lecture_recorder  # noqa: F401  (lockdown)
from lecture_recorder.config import ProcessingConfig, load_config
from lecture_recorder.pipeline import queue as q
from lecture_recorder.recorder.platform import KeepAwake
from lecture_recorder.web.app import Recorder, create_app
from test_web_app import BASE, H
from test_processing_modes import FakeRun, cfg, power, session  # noqa: F401  (fixtures)

HOLD = KeepAwake.ES_CONTINUOUS | KeepAwake.ES_SYSTEM_REQUIRED
RELEASE = KeepAwake.ES_CONTINUOUS


@pytest.fixture
def es(monkeypatch):
    """Records every SetThreadExecutionState call instead of making it."""
    calls: list[int] = []
    monkeypatch.setattr(KeepAwake, "_set", lambda self, flags: calls.append(flags) or True)
    return calls


def held(es: list[int]) -> bool:
    return bool(es) and es[-1] == HOLD


ON, OFF = (lambda: True), (lambda: False)


def test_default_on_and_user_config_toml_wins(cfg):
    assert cfg.processing.keep_awake is True and ProcessingConfig().keep_awake is True
    (cfg.home / "config.auto.toml").write_text("[processing]\nkeep_awake = true\n", encoding="utf-8")
    (cfg.home / "config.toml").write_text("[processing]\nkeep_awake = false\n", encoding="utf-8")
    assert load_config(cfg.home).processing.keep_awake is False
    (cfg.home / "config.toml").write_text('[processing]\nkeep_awake = "yes"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="keep_awake"):
        load_config(cfg.home)


def test_held_only_while_jobs_run_and_released_when_the_queue_is_empty(cfg, power, es, monkeypatch):
    assert q.run_worker(cfg.data_dir, keep_awake=ON) == [] and es == []  # empty queue: never held
    q.enqueue(cfg.data_dir, session(cfg.data_dir, "2026-09-01_09-00-00"))
    during = []
    monkeypatch.setattr(q, "_run_job", FakeRun(during=lambda: during.append(held(es))))
    done = q.run_worker(cfg.data_dir, keep_awake=ON)
    assert [q.kind_of(j) for j in done] == ["transcribe", "summarize"]  # follow-up job ran in the same hold
    assert during == [True, True]
    assert es == [HOLD, RELEASE]  # one hold, released once the queue was empty


def test_released_on_job_failure(cfg, power, es, monkeypatch):
    q.enqueue(cfg.data_dir, session(cfg.data_dir, "2026-09-01_09-00-00"))

    class Failed:
        returncode, stdout, stderr = 1, "", "boom"
    monkeypatch.setattr(q, "_run_job", lambda *a, **k: Failed())
    done = q.run_worker(cfg.data_dir, keep_awake=ON)
    assert [j["state"] for j in done] == ["failed"] and es == [HOLD, RELEASE]


@pytest.mark.parametrize("error", [OSError("cannot start python"), KeyboardInterrupt()])
def test_released_when_the_worker_stops_on_an_error_or_ctrl_c(cfg, power, es, monkeypatch, error):
    q.enqueue(cfg.data_dir, session(cfg.data_dir, "2026-09-01_09-00-00"))

    def crash(*a, **k):
        raise error
    monkeypatch.setattr(q, "_run_job", crash)
    with pytest.raises(type(error)):
        q.run_worker(cfg.data_dir, keep_awake=ON)
    assert es == [HOLD, RELEASE]


def test_waiting_jobs_do_not_hold_it(cfg, power, es, monkeypatch):
    """On battery (or recording) jobs wait: nothing is held; unplugged mid-job, it is released after that job."""
    q.enqueue(cfg.data_dir, session(cfg.data_dir, "2026-09-01_09-00-00"))
    power["now"] = "battery"
    monkeypatch.setattr(q, "_run_job", FakeRun())
    blocked = []
    assert q.run_worker(cfg.data_dir, blocked=blocked, keep_awake=ON) == [] and blocked == ["on battery"]
    assert es == []
    power["now"] = "ac"
    monkeypatch.setattr(q, "_run_job", FakeRun(during=lambda: power.update(now="battery")))
    done = q.run_worker(cfg.data_dir, blocked=blocked, keep_awake=ON)
    assert len(done) == 1 and es == [HOLD, RELEASE]


def test_setting_off_never_calls_windows(cfg, power, es, monkeypatch):
    q.enqueue(cfg.data_dir, session(cfg.data_dir, "2026-09-01_09-00-00"))
    monkeypatch.setattr(q, "_run_job", FakeRun())
    assert len(q.run_worker(cfg.data_dir, keep_awake=OFF)) == 2
    assert len(q.run_worker(cfg.data_dir)) == 0  # callers that don't pass the setting never hold it
    assert es == []


def test_setting_turned_off_mid_queue_releases_before_the_next_job(cfg, power, es, monkeypatch):
    q.enqueue(cfg.data_dir, session(cfg.data_dir, "2026-09-01_09-00-00"))
    setting = {"on": True}
    during = []

    def job():
        during.append(held(es))
        setting["on"] = False  # the user unticks it while the first job runs
    monkeypatch.setattr(q, "_run_job", FakeRun(during=job))
    q.run_worker(cfg.data_dir, keep_awake=lambda: setting["on"])
    assert during == [True, False] and es == [HOLD, RELEASE]


@pytest.mark.parametrize("on", [True, False])
def test_cli_process_honors_the_setting(cfg, power, es, monkeypatch, on):
    from lecture_recorder.cli import main

    d = session(cfg.data_dir, "2026-09-01_09-00-00")
    mine = dataclasses.replace(cfg, processing=ProcessingConfig(keep_awake=on))
    monkeypatch.setattr("lecture_recorder.config.load_config", lambda home=None: mine)
    monkeypatch.setattr(q, "_run_job", FakeRun())
    assert main(["process", str(d)]) == 0
    assert es == ([HOLD, RELEASE] if on else [])


def test_app_setting_saves_to_config_toml_and_reaches_the_worker(cfg, power, monkeypatch):
    (cfg.home / "config.toml").write_text('[processing]\nmode = "manual"\n', encoding="utf-8")
    app = create_app(cfg, autoprocess=False, monitor=False)
    with TestClient(app, base_url=BASE) as c:
        assert c.get("/api/processing").json()["keep_awake"] is True
        assert c.put("/api/settings", headers=H, json={"keep_awake": "no"}).status_code == 400
        assert c.put("/api/settings", headers=H, json={"keep_awake": False}).json()["keep_awake"] is False
    assert load_config(cfg.home).processing.keep_awake is False

    seen = []
    monkeypatch.setattr("lecture_recorder.web.app.run_worker",
                        lambda *a, keep_awake, **k: seen.append(keep_awake()) or [])
    rec = Recorder(cfg, autoprocess=True)
    rec._work()  # the worker thread's body, run here
    rec.cfg = dataclasses.replace(cfg, processing=ProcessingConfig(keep_awake=False))
    rec._work()
    assert seen == [True, False]
