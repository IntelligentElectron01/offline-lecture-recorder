"""Who may talk to the local app.

The app listens on 127.0.0.1 only, but any program running as any user on this computer can reach that port.
So every request needs a session cookie that only a browser opened by this app (or by `run.bat` again) has:

- `serve` makes a random login key and writes it to `<app home>/.app-token-<port>`, readable by the current
  user only (chmod 600; icacls on Windows, best effort). The key is never printed.
- The browser is opened at `/?k=<key>`. A right key sets a random per-run session cookie (HttpOnly,
  SameSite=Strict) and redirects to `/`, so the key doesn't stay in the address bar.
- Every other request without that cookie is refused (401), including /files/ and /static/. Only
  `/api/health` (answers {"ok": true, "app": ...}, no data) is open, so a second start can tell this app
  from another program on the port.
- A second `run.bat` reads the key file and opens the running app. Restarting the app makes a new key and
  session, so a tab from before needs one reload through run.bat.

There is deliberately no switch (no setting, no variable of any kind) that turns this off. Tests replace
`enforced` in their own process only (tests/conftest.py).
"""
import hmac
import os
import secrets
import subprocess
from pathlib import Path

COOKIE = "lr_session"  # the real name has the port appended: browsers share cookies between ports
LOGIN_PARAM = "k"
OPEN_PATHS = ("/api/health",)
LOGIN_PAGE = ("<!doctype html><meta charset=utf-8><title>Lecture Recorder</title>"
              "<body style=\"font:16px system-ui;margin:3rem auto;max-width:34rem\"><h1>Lecture Recorder</h1>"
              "<p>This page needs to be opened by the app. Start it again (run.bat, or the shortcut it made): "
              "it opens the right address for you.</p>")


def enforced() -> bool:
    return True


def token_path(home: Path, port: int) -> Path:
    return Path(home) / f".app-token-{port}"


def cookie_name(port: int) -> str:
    return f"{COOKIE}_{port}"


def _my_account() -> str:
    """The current account for icacls: its SID (works for Microsoft and work accounts too), else the user name."""
    import getpass

    try:
        out = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"], capture_output=True, text=True, timeout=10).stdout
        sid = out.strip().split(",")[-1].strip().strip('"')
        if sid.startswith("S-1-"):
            return "*" + sid
    except Exception:
        pass
    return getpass.getuser()


# SYSTEM, Administrators, OWNER RIGHTS, Users, Authenticated Users, Everyone
_OTHER_SIDS = ("S-1-5-18", "S-1-5-32-544", "S-1-3-4", "S-1-5-32-545", "S-1-5-11", "S-1-1-0")


def _private(path: Path) -> bool:
    """Only this user can read the file (checked, not assumed). False: say so and don't keep the key on disk."""
    try:
        if os.name == "nt":
            me = _my_account()
            r = subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", f"{me}:(F)"],
                               capture_output=True, timeout=10)
            if r.returncode != 0:
                return False
            # An elevated (admin) token also gives the new file SYSTEM, Administrators and OWNER RIGHTS entries of its own:
            # drop them, so only this account is left (unless this account is one of them).
            others = [f"*{sid}" for sid in _OTHER_SIDS if f"*{sid}" != me]
            r = subprocess.run(["icacls", str(path), "/remove:g", *others], capture_output=True, timeout=10)
            return r.returncode == 0
        os.chmod(path, 0o600)
        return True
    except Exception:
        return False


def write_login_key(home: Path, port: int) -> str:
    """A new login key, stored for run.bat's second start. Called only when this process will serve.

    The file is created empty, made private, and only then filled, so the key is never in a file others can
    read. If it can't be made private the key is not stored (the browser is still opened with it; a second
    start then says the sign-in file is missing instead of using a readable key)."""
    key = secrets.token_urlsafe(32)
    path = token_path(home, port)
    path.unlink(missing_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    try:
        if not _private(tmp):
            print("Note: could not restrict the sign-in file to your account, so it is not kept; "
                  "a second start of the app can't reopen this one until you restart it.")
            return key
        tmp.write_text(key, encoding="ascii")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return key


def read_login_key(home: Path, port: int) -> str | None:
    try:
        return token_path(home, port).read_text(encoding="ascii").strip() or None
    except OSError:
        return None


def _same(a: str | None, b: str) -> bool:
    """Constant-time compare that never raises (non-ASCII input is simply wrong)."""
    try:
        return bool(a) and hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
    except Exception:
        return False


class Auth:
    def __init__(self, login_key: str | None = None, port: int = 8765):
        self.login_key = login_key or secrets.token_urlsafe(32)
        self.session = secrets.token_urlsafe(32)  # in memory only: dies with the process
        self.cookie = cookie_name(port)

    def has_session(self, cookie: str | None) -> bool:
        return _same(cookie, self.session)

    def accepts_key(self, key: str | None) -> bool:
        return _same(key, self.login_key)
