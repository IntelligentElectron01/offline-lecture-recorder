"""Local sign-in. Any program on this computer can reach 127.0.0.1:8765, so every request needs the session
cookie that a browser opened by the app (or by a second start) gets. Real gate on (real_auth); temp folders only."""
import dataclasses
import json
import stat
import sys
import threading

import pytest
from fastapi.testclient import TestClient

import lecture_recorder  # noqa: F401  (lockdown)
from lecture_recorder.config import load_config
from lecture_recorder.web import app as webapp
from lecture_recorder.web import auth
from lecture_recorder.web.app import create_app
from test_web_app import BASE, H, fake_processed_session

pytestmark = pytest.mark.real_auth
KEY = "the-login-key"
SESSION = "/files/physics-101/2026-09-30_10-00-00"


@pytest.fixture
def cfg(tmp_path):
    return dataclasses.replace(load_config(tmp_path), data_dir=(tmp_path / "data").resolve())


@pytest.fixture
def client(cfg):
    fake_processed_session(cfg.data_dir)
    with TestClient(create_app(cfg, autoprocess=False, monitor=False, login_key=KEY), base_url=BASE,
                    follow_redirects=False) as c:
        yield c


def test_no_cookie_no_access_anywhere(client):
    for path in ("/api/classes", "/api/record/status", "/api/processing", "/api/search?q=hello", "/static/app.js",
                 "/static/index.html", f"{SESSION}/audio.wav", f"{SESSION}/transcript.md", "/api/quit", "/nothing-here"):
        r = client.get(path)
        assert r.status_code == 401, path
        assert "Hello" not in r.text
    for method, path in (("POST", "/api/record/start"), ("PUT", "/api/settings"), ("POST", "/api/quit"),
                         ("DELETE", "/api/classes/physics-101")):
        assert client.request(method, path, headers=H, json={}).status_code == 401, (method, path)
    r = client.get("/")  # the page itself only says how to open the app
    assert r.status_code == 401 and "run.bat" in r.text and "Lecture Recorder" in r.text


def test_only_the_health_probe_is_open_and_it_returns_no_data(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json() == {"ok": True, "app": "lecture-recorder"}


def test_wrong_or_missing_key_is_refused_and_sets_no_cookie(client):
    for q in ("?k=wrong", f"?k={KEY}x", "?kk=" + KEY):
        r = client.get("/" + q)
        assert r.status_code == 401 and "set-cookie" not in r.headers, q
    assert client.get("/api/classes?k=" + KEY).status_code == 401  # the key works on the page only
    assert client.post("/?k=" + KEY, headers=H).status_code == 401


def test_the_right_key_signs_in_with_a_strict_http_only_cookie_and_leaves_the_address(client):
    r = client.get(f"/?k={KEY}")
    assert r.status_code == 303 and r.headers["location"] == "/"
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "path=/" in cookie
    assert cookie.startswith(auth.cookie_name(8765).lower() + "=")  # one cookie per port: two instances don't sign each other out
    assert KEY not in r.headers["set-cookie"]  # the cookie is a separate per-run secret, not the login key
    assert r.headers["referrer-policy"] == "no-referrer"
    # the browser keeps the cookie: everything works now, including files and static
    assert client.get("/").status_code == 200
    assert client.get("/api/classes").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get(f"{SESSION}/transcript.md").status_code == 200
    assert client.post("/api/classes", headers=H, json={"name": "Chem"}).status_code == 201
    assert client.post("/api/classes", json={"name": "Chem2"}).status_code == 403  # the old CSRF header rule still applies


def test_a_forged_cookie_is_refused(client):
    name = auth.cookie_name(8765)
    client.cookies.set(name, "guess")
    assert client.get("/api/classes").status_code == 401
    client.cookies.set(name, KEY)  # the login key is not the session
    assert client.get("/api/classes").status_code == 401
    assert client.get("/?k=caf%C3%A9").status_code == 401  # non-ASCII input is refused, not a server error
    assert auth.Auth("k").has_session("caf" + chr(233)) is False and auth.Auth("k").accepts_key("caf" + chr(233)) is False


def test_a_restart_invalidates_old_tabs_but_a_second_start_signs_in_again(cfg):
    fake_processed_session(cfg.data_dir)
    first = create_app(cfg, autoprocess=False, monitor=False, login_key="k1")
    with TestClient(first, base_url=BASE, follow_redirects=False) as c:
        c.get("/?k=k1")
        cookie = c.cookies.get(auth.cookie_name(8765))
        assert c.get("/api/classes").status_code == 200
    again = create_app(cfg, autoprocess=False, monitor=False, login_key="k2")
    with TestClient(again, base_url=BASE, follow_redirects=False) as c:
        c.cookies.set(auth.cookie_name(8765), cookie)
        assert c.get("/api/classes").status_code == 401  # a tab from the previous run
        assert c.get("/?k=k1").status_code == 401
        c.get("/?k=k2")
        assert c.get("/api/classes").status_code == 200


def test_strangers_do_not_keep_an_idle_app_alive(cfg):
    app = create_app(cfg, autoprocess=False, monitor=False, login_key=KEY)
    touched = []
    app.state.idle.touch = lambda: touched.append(1)
    with TestClient(app, base_url=BASE, follow_redirects=False) as c:
        c.get("/api/classes")
        c.get("/api/health")
        assert touched == []
        c.get(f"/?k={KEY}")
        c.get("/api/classes")
        assert touched == [1]


def test_host_header_rule_still_comes_first(client):
    assert client.get("/api/health", headers={"Host": "evil.example"}).status_code == 400


def test_login_key_file_is_per_port_private_and_never_in_output(tmp_path, capsys):
    key = auth.write_login_key(tmp_path, 8765)
    assert auth.read_login_key(tmp_path, 8765) == key and auth.read_login_key(tmp_path, 8766) is None
    assert len(key) >= 40 and auth.write_login_key(tmp_path, 8765) != key  # fresh at every serve
    assert not list(tmp_path.glob("*.tmp"))
    if sys.platform != "win32":
        assert stat.S_IMODE((tmp_path / ".app-token-8765").stat().st_mode) == 0o600
    assert key not in capsys.readouterr().out


def test_serve_opens_the_browser_with_the_key_and_a_second_start_reuses_the_file(cfg, monkeypatch, capsys):
    import uvicorn
    import webbrowser

    opened, served = [], []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))

    class Timer:  # run the delayed browser launch at once
        def __init__(self, delay, fn):
            self.fn = fn

        def start(self):
            self.fn()

    class Server:
        def __init__(self, config):
            served.append(config.app)
            self.should_exit = False

        def run(self):
            pass

    monkeypatch.setattr(threading, "Timer", Timer)
    monkeypatch.setattr(uvicorn, "Server", Server)
    monkeypatch.setattr(uvicorn, "Config", lambda app, **k: type("C", (), {"app": app})())
    monkeypatch.setattr(webapp, "port_state", lambda host, port: "free")
    assert webapp.serve(cfg, open_browser=True) == 0
    key = auth.read_login_key(cfg.home, cfg.server.port)
    assert opened == [f"http://127.0.0.1:{cfg.server.port}/?k={key}"]
    assert served[0].state.auth.login_key == key
    assert key not in capsys.readouterr().out

    # the second run.bat: the port is this app's; it must open the running app, not replace its key
    opened.clear()
    monkeypatch.setattr(webapp, "port_state", lambda host, port: "this app")
    assert webapp.serve(cfg, open_browser=True) == 0
    assert opened == [f"http://127.0.0.1:{cfg.server.port}/?k={key}"]
    assert auth.read_login_key(cfg.home, cfg.server.port) == key
    assert key not in capsys.readouterr().out
    (cfg.home / f".app-token-{cfg.server.port}").unlink()  # running, but its file is gone: say so, open nothing
    opened.clear()
    assert webapp.serve(cfg, open_browser=True) == 1 and opened == []
    assert "sign-in file is missing" in capsys.readouterr().out


def test_port_state_recognizes_this_app_only_by_the_open_probe(monkeypatch):
    import io
    import urllib.request

    class Opener:
        def __init__(self, body):
            self.body = body

        def open(self, req, timeout=0):
            return io.BytesIO(self.body)

    def in_use(self, addr):
        raise OSError("in use")

    monkeypatch.setattr("socket.socket.bind", in_use)
    for body, want in ((json.dumps({"ok": True, "app": "lecture-recorder"}).encode(), "this app"),
                       (b'{"ok": true}', "busy"), (b"<html>", "busy")):
        monkeypatch.setattr(urllib.request, "build_opener", lambda *h, b=body: Opener(b))
        assert webapp.port_state("127.0.0.1", 59999) == want


def test_the_app_has_no_switch_that_turns_sign_in_off():
    from pathlib import Path

    for mod in (auth, webapp):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "environ" not in src and "getenv" not in src, mod.__name__
    assert auth.enforced() is True  # only tests/conftest.py replaces it, in its own process


# ------------------------------------------------------------------ secrets and paths in output
def test_dotenv_secrets_stay_in_the_file(tmp_path, monkeypatch):
    import os

    from lecture_recorder import runtime_env

    (tmp_path / ".env").write_text("HF_TOKEN=hf_example\nMY_API_KEY=x\nDEMO_SETTING=1\n", encoding="utf-8")
    monkeypatch.setenv("LECTURE_RECORDER_HOME", str(tmp_path))
    for k in ("HF_TOKEN", "MY_API_KEY", "DEMO_SETTING"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DEMO_SETTING", "")  # registers the undo for the value apply_lockdown sets
    monkeypatch.delenv("DEMO_SETTING")
    runtime_env.apply_lockdown()
    assert "HF_TOKEN" not in os.environ and "MY_API_KEY" not in os.environ
    assert os.environ["DEMO_SETTING"] == "1" and os.environ["HF_HUB_OFFLINE"] == "1"
    monkeypatch.delenv("DEMO_SETTING")


def test_paths_shown_in_terminal_output_hide_the_account_name(tmp_path):
    from pathlib import Path

    from lecture_recorder.runtime_env import shown

    home, mark = Path.home(), "%USERPROFILE%" if sys.platform == "win32" else "~"
    assert shown(home) == mark
    assert shown(home / "Documents" / "LectureRecorder") == mark + str(Path("/Documents/LectureRecorder")).replace("/", "\\" if sys.platform == "win32" else "/")
    assert shown(home / "a").startswith(mark) and home.name not in shown(home / "a")
    other = Path(Path.home().anchor) / "elsewhere"  # not under the profile folder
    assert shown(other) == str(other)
    assert shown(str(home) + "extra") == str(home) + "extra"  # a sibling folder with the same prefix is not inside it


def test_doctor_never_prints_the_profile_folder(tmp_path, monkeypatch, capsys):
    from pathlib import Path

    from lecture_recorder import cli

    monkeypatch.setenv("LECTURE_RECORDER_HOME", str(Path.home() / "LectureRecorderDoctorDemo"))
    cli.doctor()
    out = capsys.readouterr().out
    assert str(Path.home()) not in out and "LectureRecorderDoctorDemo" in out


def test_a_valid_key_always_redirects_even_when_already_signed_in(client):
    client.get(f"/?k={KEY}")
    r = client.get(f"/?k={KEY}")  # a second run.bat in the same browser: the key doesn't stay in the address bar
    assert r.status_code == 303 and r.headers["location"] == "/"


def test_the_key_file_is_private_before_it_holds_the_key(tmp_path, monkeypatch, capsys):
    seen = []
    real = auth._private

    def spy(path):
        seen.append(path.read_text(encoding="ascii"))  # still empty when it is made private
        return real(path)

    monkeypatch.setattr(auth, "_private", spy)
    key = auth.write_login_key(tmp_path, 8765)
    assert seen == [""] and auth.read_login_key(tmp_path, 8765) == key
    monkeypatch.setattr(auth, "_private", lambda p: False)  # could not restrict it: the key is not kept on disk
    key2 = auth.write_login_key(tmp_path, 8765)
    assert auth.read_login_key(tmp_path, 8765) is None and key2 != key
    assert "not kept" in capsys.readouterr().out and not list(tmp_path.glob("*.tmp"))


@pytest.mark.skipif(sys.platform != "win32", reason="icacls")
def test_on_windows_only_this_account_can_read_the_key_file(tmp_path):
    import subprocess

    auth.write_login_key(tmp_path, 8765)
    out = subprocess.run(["icacls", str(tmp_path / ".app-token-8765")], capture_output=True, text=True).stdout
    entries = [line for line in out.splitlines() if ":(" in line]
    assert len(entries) == 1, out  # no Users, Everyone, Administrators or inherited entries
