"""Quit in the app, closing the console window, and the optional idle quit.
Temp folders only (own app home, so Settings never touch the real config.toml); synthetic recordings.
"""
import dataclasses
import sys
import time

import pytest
from fastapi.testclient import TestClient

import lecture_recorder  # noqa: F401  (lockdown)
from lecture_recorder import library as lib
from lecture_recorder.config import ProcessingConfig, load_config
from lecture_recorder.pipeline import queue as q
from lecture_recorder.recorder import platform
from lecture_recorder.recorder.session import read_json
from lecture_recorder.web import app as webapp
from lecture_recorder.web.app import IdleQuit, create_app

BASE = "http://127.0.0.1:8765"
H = {"X-Lecture-Recorder": "1"}


@pytest.fixture
def cfg(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    cfg = dataclasses.replace(load_config(home), data_dir=(home / "data").resolve())
    lib.create_class(cfg.data_dir, "Physics 101")
    return cfg


def make_client(cfg, **kw):
    app = create_app(cfg, monitor=False, **kw)
    exits = []
    app.state.request_exit = lambda: exits.append(time.monotonic())
    return TestClient(app, base_url=BASE), exits


def quit_with_plan(client):
    plan = client.get("/api/quit").json()
    return plan, client.post("/api/quit", headers=H, json={"confirm": plan})


def test_quit_while_idle(cfg):
    client, exits = make_client(cfg, autoprocess=False)
    with client:
        plan, r = quit_with_plan(client)
        assert plan == {"recording": None, "running": [], "queued": 0}
        assert r.status_code == 200 and r.json() == {"closed": True, "recording": None}
        assert len(exits) == 1
        assert client.post("/api/quit", headers=H, json={"confirm": plan}).json() == {"closed": True}  # once only
        assert len(exits) == 1
        assert client.post("/api/quit", json={"confirm": plan}).status_code == 403  # app header required


def test_quit_while_recording_saves_it_and_needs_the_shown_plan(cfg):
    client, exits = make_client(cfg, autoprocess=False, synthetic_source={"pattern": "tone"})
    with client:
        r = client.post("/api/record/start", headers=H, json={"class": "physics-101"})
        assert r.status_code == 201, r.text
        time.sleep(1.5)
        stale = client.post("/api/quit", headers=H, json={"confirm": {"recording": None, "running": [], "queued": 0}})
        assert stale.status_code == 409 and not exits  # the confirmation didn't mention the recording
        plan, r = quit_with_plan(client)
        d = cfg.data_dir / "classes" / "physics-101" / "sessions" / plan["recording"]["session"]
        assert plan["recording"]["title"] == d.name and r.status_code == 200 and r.json()["recording"] == d.name
        assert (d / "audio.wav").is_file() and not (d / "audio.pcm").exists()  # saved through the Stop path
        assert read_json(d / "meta.json")["state"] == "stopped" and exits
        assert not client.get("/api/record/status").json()["recording"]


def test_quit_while_a_job_runs_ends_it_and_requeues_it(cfg, monkeypatch):
    real = q._run_job
    started = []

    def slow_job(cmd, **kw):  # a real child process that would run for a minute
        started.append(time.monotonic())
        return real([sys.executable, "-c", "import time; time.sleep(60)"], **kw)

    monkeypatch.setattr(q, "_run_job", slow_job)
    monkeypatch.setattr(q, "gpu_block_reason", lambda d: None)
    monkeypatch.setattr(webapp, "gpu_block_reason", lambda d: None)
    d = cfg.data_dir / "classes" / "physics-101" / "sessions" / "2026-10-01_09-00-00"
    d.mkdir(parents=True)
    (d / "meta.json").write_text('{"title": "Optics"}', encoding="utf-8")
    (d / "audio.wav").write_bytes(b"RIFF")
    other = d.parent / "2026-10-02_09-00-00"
    other.mkdir()
    client, exits = make_client(dataclasses.replace(cfg, processing=ProcessingConfig(mode="manual")), autoprocess=True)
    with client:
        assert client.post(f"/api/sessions/physics-101/{d.name}/process", headers=H).status_code == 202
        deadline = time.monotonic() + 20
        while not started and time.monotonic() < deadline:
            time.sleep(0.05)
        q.enqueue(cfg.data_dir, other)  # queued behind the running job
        plan = client.get("/api/quit").json()
        assert plan == {"recording": None, "running": [{"kind": "transcribe", "title": "Optics"}], "queued": 1}
        t0 = time.monotonic()
        r = client.post("/api/quit", headers=H, json={"confirm": plan})
        assert r.status_code == 200 and time.monotonic() - t0 < 10  # the 60 s job was ended, not waited for
        states = {j["session"]: (j["state"], "started_ts" in j) for j in q.jobs(cfg.data_dir)}
        assert states == {str(d.resolve()): ("queued", False), str(other.resolve()): ("queued", False)}
        assert len(started) == 1 and exits  # the queued one never started
        client.app.state.recorder.kick()  # nothing starts after Quit
        assert client.app.state.recorder._worker is None


def test_quit_still_ends_the_server_if_stopping_fails(cfg, monkeypatch):
    client, exits = make_client(cfg, autoprocess=False)

    def stuck(self):
        raise TimeoutError("recorder did not stop in time")

    monkeypatch.setattr(webapp.Recorder, "shutdown", stuck)
    with client:
        plan, r = quit_with_plan(client)
        assert r.status_code == 200 and "recovered when the app starts" in r.json()["error"] and len(exits) == 1


def test_jobs_have_no_console_so_only_quit_ends_them():
    if platform.IS_WINDOWS:
        import subprocess

        flags = q.LOW_PRIORITY["creationflags"]
        for f in (subprocess.BELOW_NORMAL_PRIORITY_CLASS, subprocess.CREATE_NEW_PROCESS_GROUP, subprocess.CREATE_NO_WINDOW):
            assert flags & f
    else:
        assert q.LOW_PRIORITY["start_new_session"] is True


def test_console_close_handler(monkeypatch):
    calls = []
    handler = platform.console_close_handler(lambda: calls.append("quit"))
    assert handler(0) is False and handler(1) is False and not calls  # Ctrl+C / Ctrl+Break stay with Python
    assert handler(2) is True and calls == ["quit"]  # window closed
    assert handler(6) is True and len(calls) == 2  # Windows shutting down

    def boom():
        raise RuntimeError("x")
    assert platform.console_close_handler(boom)(2) is True  # the process ends anyway

    if platform.IS_WINDOWS:
        import ctypes

        registered = []

        class Kernel32:
            def SetConsoleCtrlHandler(self, routine, add):  # noqa: N802
                registered.append((routine, add))
                return 1

        monkeypatch.setattr(ctypes, "windll", type("W", (), {"kernel32": Kernel32()})())
        routine = platform.on_console_close(lambda: calls.append("closed"))
        assert registered == [(routine, True)]
        assert routine(2) == 1 and calls[-1] == "closed"  # called through ctypes like Windows would


def test_serve_quits_cleanly_when_the_window_closes(cfg, monkeypatch):
    import uvicorn

    seen = {}

    class FakeServer:
        def __init__(self, config):
            self.app, self.should_exit = config.app, False

        def run(self):
            seen["close"]()  # the user closes the console window while the server runs
            seen["should_exit"] = self.should_exit

    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    monkeypatch.setattr(platform, "on_console_close", lambda fn: seen.setdefault("close", fn) and "routine")
    monkeypatch.setattr(webapp, "port_state", lambda h, p: "free")
    shut = []
    real = webapp.Recorder.shutdown
    monkeypatch.setattr(webapp.Recorder, "shutdown", lambda self: shut.append(1) or real(self))
    assert webapp.serve(cfg, open_browser=False) == 0
    assert seen["should_exit"] is True and shut == [1]  # once, though serve's own cleanup also ran


def test_idle_quit_with_a_fake_clock():
    now, minutes, busy = [1000.0], [10], [False]
    idle = IdleQuit(lambda: minutes[0], clock=lambda: now[0])
    now[0] += 599
    assert not idle.due(lambda: busy[0])
    now[0] += 1
    busy[0] = True
    assert not idle.due(lambda: busy[0])  # recording or a job: never, and the count starts again
    busy[0] = False
    now[0] += 599
    assert not idle.due(lambda: busy[0])
    now[0] += 1
    assert idle.due(lambda: busy[0])
    idle.touch()  # a page polled
    assert not idle.due(lambda: busy[0])
    minutes[0] = 0
    now[0] += 10**6
    assert not idle.due(lambda: pytest.fail("busy is not checked when off"))  # off (the default)


def test_idle_quit_counts_requests_and_never_quits_while_recording(cfg):
    client, exits = make_client(cfg, autoprocess=False, synthetic_source={"pattern": "tone"})
    with client:
        idle = client.app.state.idle
        rec = client.app.state.recorder
        assert rec.cfg.server.idle_quit_minutes == 0  # off by default
        idle.last = 0
        client.get("/api/record/status")
        assert idle.last > 0  # the status poll is the heartbeat
        assert client.post("/api/record/start", headers=H, json={"class": "physics-101"}).status_code == 201
        assert rec.busy()
        client.post("/api/record/stop", headers=H)
        assert not rec.busy()


def test_idle_quit_setting(cfg):
    client, _ = make_client(cfg, autoprocess=False)
    with client:
        for bad in (5000, -1, True, "10"):
            assert client.put("/api/settings", headers=H, json={"idle_quit_minutes": bad}).status_code == 400
        r = client.put("/api/settings", headers=H, json={"idle_quit_minutes": 10})
        assert r.status_code == 200 and r.json()["idle_quit_minutes"] == 10
        assert client.app.state.recorder.cfg.server.idle_quit_minutes == 10
        assert load_config(cfg.home).server.idle_quit_minutes == 10


def test_ui_has_quit_and_closed_page():
    from pathlib import Path

    static = Path(lecture_recorder.__file__).parent / "web" / "static"
    js = (static / "app.js").read_text(encoding="utf-8")
    assert 'id="quit"' in (static / "index.html").read_text(encoding="utf-8")
    for needle in ("Lecture Recorder has closed.", 'api("/api/quit")', "json: { confirm: plan }", "idle_quit_minutes",
                   "e.status === undefined && ++unanswered >= 3) showClosed()", "location.reload()"):
        assert needle in js, needle
