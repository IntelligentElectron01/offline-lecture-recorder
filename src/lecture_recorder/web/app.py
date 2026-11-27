"""The local app server.

Security model (single user, local only):
- Bound to 127.0.0.1 (`serve` refuses any non-loopback host).
- Host header must be 127.0.0.1 or localhost (blocks DNS rebinding).
- Every request needs the session cookie that only a browser opened by the app has (web/auth.py);
  /api/health alone is open and returns no data.
- State-changing requests need the custom header `X-Lecture-Recorder: 1` (a cross-site page can't
  send it without a CORS preflight, which this server never approves) and, if the browser sends an
  Origin, it must be this app's own origin.
- Files are served only by name from an allowlist, from a session folder whose resolved path is
  inside data/. Nothing outside data/ (or the bundled static UI) is ever served.
"""
import dataclasses
import hashlib
import ipaddress
import json
import os
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from lecture_recorder import autosort
from lecture_recorder import duplicates as dup
from lecture_recorder import importer as imp
from lecture_recorder import library as lib
from lecture_recorder import units as un
from lecture_recorder import zoom
from lecture_recorder.config import Config, load_config
from lecture_recorder.web import auth
from lecture_recorder.machine import gpu_block_reason
from lecture_recorder.pipeline.outputs import EDITABLE, MD, RAW, TXT, render_md, render_txt, write_text_atomic
from lecture_recorder.pipeline.progress import describe as describe_progress
from lecture_recorder.pipeline.queue import EMBED, SUMMARIZE, Stop, enqueue, jobs, kind_of, progress_path, run_worker
from lecture_recorder.recorder.control import (
    PERMISSION_REMINDER, RecorderError, RecordingHandle, active_recording, start_recording,
)
from lecture_recorder.recorder.session import (
    META_FILE, PCM_FILE, WAV_FILE, MetaUnreadable, iter_session_dirs, now_iso, read_json, write_json_atomic,
)

STATIC = Path(__file__).parent / "static"
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]
CSRF_HEADER = "x-lecture-recorder"


def check_loopback(host: str) -> str:
    try:
        ok = ipaddress.ip_address(host).is_loopback
    except ValueError:
        ok = host == "localhost"
    if not ok:
        raise ValueError(f"refusing to listen on {host!r}: the app is local-only (use 127.0.0.1)")
    return host


def _job_title(job: dict) -> str:
    """A job's lecture title (meta.json), or its folder name (exam jobs name an exam folder)."""
    d = Path(job["session"])
    if kind_of(job) == "exam":
        return (un.get_exam(d.parent.parent, d.name) or {}).get("name") or d.name
    return (read_json(d / META_FILE, {}) or {}).get("title") or d.name


class Recorder:
    """The one recording this app controls, and the processing queue's worker thread.

    Processing mode (config.toml [processing] mode): "manual" (default) only saves a recording on
    Stop/import; the user starts processing. "auto" queues it and runs it when plugged in. In every
    mode the queue never starts a GPU job while recording or on battery (machine.py); jobs then wait,
    and in auto mode they resume by themselves once that changes.
    """

    def __init__(self, cfg: Config, autoprocess: bool, synthetic: dict | None = None):
        self.cfg = cfg  # replaced when the user changes a setting
        self.autoprocess = autoprocess  # False in tests: jobs are queued but no worker thread starts
        self.synthetic = synthetic  # tests only: record a generated signal instead of a microphone
        self.handle: RecordingHandle | None = active_recording(cfg.data_dir)  # re-attach after a UI restart
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._wake = False
        self.waiting: str | None = None  # why queued jobs are waiting (recording / on battery), if they are
        self.stopping = Stop()  # set by Quit: no job starts after that, a running one is ended

    @property
    def auto(self) -> bool:
        return self.cfg.processing.mode == "auto"

    def start(self, class_slug: str, device, app_pid: int | None = None, include_mic: bool = False) -> RecordingHandle:
        """Record from the microphone `device`, or from the app `app_pid` (plus the mic if `include_mic`)."""
        with self._lock:
            self.reap()
            c = lib.get_class(self.cfg.data_dir, class_slug)
            if not c or not c["has_class_file"]:
                raise HTTPException(400, "Choose an existing class before recording.")
            app = None
            if app_pid is not None:
                from lecture_recorder.host import host

                if why := host().app_capture_reason:  # before anything Windows-only is imported
                    raise HTTPException(409, why)
                from lecture_recorder.recorder.apps import find_app

                app = find_app(app_pid)  # only an app that has audio open right now
                if app is None:
                    raise HTTPException(409, "That app isn't playing audio any more (or was closed). Refresh the list and pick it again.")
            self.handle = start_recording(class_slug, config=self.cfg, device=device, app=app, include_mic=include_mic,
                                          synthetic=None if app else self.synthetic,
                                          max_length_minutes=c.get("max_length_minutes"), process_on_stop=True)
            return self.handle

    def reap(self) -> dict | None:
        """If the recording ended (Stop, 2 h limit, crash), finalize it and queue processing."""
        with self._lock:
            h = self.handle
            if h is None or h.is_running():
                return None
            meta = h.stop()  # already stopped: just finalizes a leftover PCM, if any
            if self.auto and (h.session_dir / "audio.wav").exists():
                enqueue(self.cfg.data_dir, h.session_dir)  # manual mode: stays "Not processed"
            self.handle = None  # only after finalize + enqueue succeeded; otherwise the next check retries
        if self.auto:
            self.kick()
        return meta

    def queue_missed(self) -> list[Path]:
        """At startup: queue app recordings that ended while the app was closed (never older sessions)."""
        queued_or_done = {j["session"] for j in jobs(self.cfg.data_dir)}
        missed = []
        for d in iter_session_dirs(self.cfg.data_dir):
            meta = read_json(d / META_FILE, {}) or {}
            if (meta.get("process_on_stop") and (d / WAV_FILE).exists() and not (d / PCM_FILE).exists()
                    and not (d / RAW).exists() and str(d.resolve()) not in queued_or_done
                    and (self.handle is None or self.handle.session_dir != d)):
                enqueue(self.cfg.data_dir, d)
                missed.append(d)
        return missed

    def stop(self) -> dict:
        with self._lock:
            if self.handle is None:
                raise HTTPException(409, "Nothing is recording.")
            self.handle.stop()
            return self.reap() or {}

    def kick(self) -> None:
        """Make sure the queue worker runs (one at a time; GPU lock inside each job).

        A kick that arrives while the worker is finishing sets `_wake`, so the worker loops once more
        instead of exiting with a job left in the queue.
        """
        if not self.autoprocess or self.stopping.is_set():
            return
        with self._lock:
            self._wake = True
            if self._worker is None:
                self._worker = threading.Thread(target=self._work, daemon=True, name="queue")
                self._worker.start()

    def _work(self) -> None:
        while True:
            with self._lock:
                self._wake = False
            blocked: list[str] = []
            try:
                run_worker(self.cfg.data_dir, blocked=blocked, keep_awake=lambda: self.cfg.processing.keep_awake,
                           stop=self.stopping)
            except Exception:  # job failures are recorded in the job file; keep the thread's bookkeeping right
                import logging

                logging.getLogger(__name__).exception("queue worker stopped")  # a worker bug, not a job failure
            with self._lock:
                self.waiting = blocked[0] if blocked else None
                if not self._wake or blocked:
                    self._worker = None
                    return

    def activity(self) -> dict:
        """What Quit would stop now: the recording, this app's running job, and how many jobs wait."""
        self.reap()
        rec = None
        if self.handle is not None:
            m = self.handle.meta()
            rec = {"class": m.get("class"), "session": self.handle.name, "title": m.get("title") or self.handle.name}
        mine = self._worker is not None  # a job run by `lecture-recorder process` elsewhere is not ours to stop
        active = jobs(self.cfg.data_dir)
        running = [{"kind": kind_of(j), "title": _job_title(j)} for j in active if j["state"] == "running" and mine]
        return {"recording": rec, "running": running, "queued": sum(j["state"] == "queued" for j in active)}

    def busy(self) -> bool:
        """Recording (here or from the terminal), or any job queued or running."""
        from lecture_recorder.recorder.platform import is_locked
        from lecture_recorder.recorder.process import GLOBAL_LOCK

        return (self.handle is not None or is_locked(self.cfg.data_dir / GLOBAL_LOCK)
                or any(j["state"] in ("queued", "running") for j in jobs(self.cfg.data_dir)))

    def shutdown(self) -> dict:
        """Quit: start no further job and end the running one (it goes back to queued), then stop
        and save the recording through the Stop button's path. Queued jobs stay queued. Safe to repeat."""
        self.stopping.set()
        worker = self._worker
        meta = self.stop() if self.handle is not None else None
        if worker is not None:
            worker.join(timeout=15)  # puts the ended job back to "queued" (if not, the next start does)
        return {"recording": meta.get("name") if meta else None}

    def check_waiting(self) -> None:
        """Monitor tick: in auto mode, restart waiting jobs once recording ended and power is back."""
        if self.auto and self.waiting and self._worker is None and gpu_block_reason(self.cfg.data_dir) is None:
            self.kick()

    def processing_status(self) -> dict:
        queued = [j for j in jobs(self.cfg.data_dir) if j["state"] in ("queued", "running")]
        return {"mode": self.cfg.processing.mode, "queued": len(queued),
                "running": any(j["state"] == "running" for j in queued) and self._worker is not None,
                "waiting": self.waiting if queued else None, "blocked_now": gpu_block_reason(self.cfg.data_dir)}

    def progress_status(self) -> dict:
        """Per-item progress for the status poll: active jobs in queue order, the queued ones with
        their position and why they wait, the running one with its title and progress snapshot. Numbers
        and labels only, never transcript text."""
        all_jobs = jobs(self.cfg.data_dir)
        # Why queued jobs wait: what blocks a start now (checked only if one waits). Not the worker's last
        # reason (self.waiting): that is only refreshed when the worker exits, so it can be stale.
        waiting = gpu_block_reason(self.cfg.data_dir) if any(j["state"] == "queued" for j in all_jobs) else None
        now, position, active = time.time(), 0, []
        for j in all_jobs:
            if j["state"] not in ("queued", "running"):
                continue
            d = Path(j["session"])
            item = {"id": j["id"], "kind": kind_of(j), "class": d.parent.parent.name, "session": d.name, "state": j["state"]}
            if j["state"] == "queued":
                position += 1
                item.update(position=position, waiting=waiting)
            else:
                item.update(title=_job_title(j),
                            progress=describe_progress(read_json(progress_path(self.cfg.data_dir, j["id"]), None), now))
            active.append(item)
        # Lectures with a job that failed in the last few minutes and was not replaced by a newer job of the same
        # kind (a failed summary is followed by an embed job): a bar that disappears shows "Failed", not "Done".
        latest = {(j["session"], kind_of(j)): j for j in all_jobs}  # oldest first: the newest per kind wins
        failed = []
        for j in latest.values():
            try:
                recent = j["state"] == "failed" and now - datetime.fromisoformat(j.get("finished", "")).timestamp() < 600
            except (TypeError, ValueError):
                recent = False
            key = f"{Path(j['session']).parent.parent.name}/{Path(j['session']).name}"
            if recent and key not in failed:
                failed.append(key)
        return {"mode": self.cfg.processing.mode, "waiting": waiting, "active": active, "recent_failed": failed}

    def status(self) -> dict:
        self.reap()
        h = self.handle
        if h is None:
            return {"recording": False, "reminder": PERMISSION_REMINDER}
        s = h.status()
        meta = h.meta()
        c = lib.get_class(self.cfg.data_dir, meta.get("class", "")) or {}
        return {"recording": True, "state": s.get("state"), "reason": s.get("reason"),
                "recorded_s": s.get("recorded_s", 0), "level_dbfs": s.get("level_dbfs"),
                "no_audio": bool(s.get("no_audio")), "no_audio_since_s": s.get("no_audio_since_s"),
                "class": meta.get("class"), "class_name": c.get("name"),
                "source": {k: (meta.get("source") or {}).get(k) for k in ("kind", "label", "app_name", "include_mic")},
                "session": h.name, "max_length_s": meta.get("max_length_s"), "reminder": PERMISSION_REMINDER}


class IdleQuit:
    """Optional quit when no page is open ([server] idle_quit_minutes, 0 = never, the default).

    Every request counts as contact: an open, visible page polls status every 2 s. Hidden tabs make no
    requests (efficiency rule), so a page in a hidden or minimized tab counts as closed. A recording or a
    queued or running job restarts the count, so the app never quits while one is active."""

    def __init__(self, minutes, clock=time.monotonic):
        self.minutes, self.clock = minutes, clock
        self.last = clock()

    def touch(self) -> None:
        self.last = self.clock()

    def due(self, busy) -> bool:
        m = self.minutes()
        if not m or self.clock() - self.last < m * 60:
            return False
        if busy():  # checked only once the time is up (it reads the queue folder)
            self.touch()
            return False
        return True


def create_app(config: Config | None = None, autoprocess: bool = True, monitor: bool = True,
               synthetic_source: dict | None = None, login_key: str | None = None) -> FastAPI:
    """`autoprocess=False` queues jobs without starting the worker, `monitor=False` skips the background
    end-of-recording check, `synthetic_source` replaces the microphone (all three for tests).
    `login_key`: the key `serve` stored for the browser launch (a random one nobody knows if not given)."""
    cfg = config or load_config()
    data_dir = cfg.data_dir.resolve()
    rec = Recorder(cfg, autoprocess, synthetic_source)
    app = FastAPI(title="Lecture Recorder", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.recorder = rec
    app.state.config = cfg
    app.state.auth = login = auth.Auth(login_key, cfg.server.port)
    origins = {f"http://{h}:{cfg.server.port}" for h in ALLOWED_HOSTS}

    idle = IdleQuit(lambda: rec.cfg.server.idle_quit_minutes)
    app.state.idle = idle

    @app.middleware("http")
    async def guard_mutations(request: Request, call_next):
        response = None
        key = request.query_params.get(auth.LOGIN_PARAM)
        front_door = request.url.path == "/" and request.method == "GET"
        if auth.enforced() and front_door and login.accepts_key(key):
            response = RedirectResponse("/", status_code=303)  # the key leaves the address bar, signed in or not
            response.set_cookie(login.cookie, login.session, httponly=True, samesite="strict", path="/")
        elif auth.enforced() and request.url.path not in auth.OPEN_PATHS and not login.has_session(request.cookies.get(login.cookie)):
            if front_door and not key:
                response = HTMLResponse(auth.LOGIN_PAGE, status_code=401)
            else:
                response = JSONResponse({"detail": "not signed in: open the app with run.bat"}, status_code=401)
        else:
            if request.url.path not in auth.OPEN_PATHS:
                idle.touch()  # contact from an open page (the visible page polls status every 2 s); strangers don't count
            if request.method not in ("GET", "HEAD", "OPTIONS"):
                if request.headers.get(CSRF_HEADER) != "1":
                    return JSONResponse({"detail": "missing app header"}, status_code=403)
                origin = request.headers.get("origin")
                if origin and origin not in origins:
                    return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
        if response is None:
            response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path == "/" or request.url.path.startswith(("/static/", "/api/")):
            response.headers["Cache-Control"] = "no-cache"  # app updates show up without a hard refresh
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; media-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        return response

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)  # added last = runs first

    from lecture_recorder.pipeline.queue import recover_stale

    recover_stale(data_dir)  # a crash may have left a job "running": show it as queued (starts nothing)
    from lecture_recorder import trash

    try:
        trash.purge_expired(data_dir, cfg.retention.trash_days)  # opt-in "Empty after N days" (default never)
    except (OSError, trash.TrashError):
        import logging

        logging.getLogger(__name__).exception("could not empty old items from Recently deleted")
    if autoprocess and rec.auto:  # auto mode: resume queued jobs and recordings that ended while closed
        rec.queue_missed()
        rec.kick()

    closed: list[str] = []

    def quit_app(reason: str) -> dict:
        """Stop the recording and processing (Recorder.shutdown), then end the server, also if stopping
        failed part-way (the page is told; start-up recovery finishes a recording). Runs once."""
        if closed:
            return {}
        closed.append(reason)
        try:
            r = rec.shutdown()
        except Exception as e:  # e.g. the recorder didn't confirm its stop in time
            r = {"error": f"The recording could not be confirmed as saved ({e}). Open the app again to check it: "
                          "an unfinished recording is recovered when the app starts."}
        finally:
            app.state.request_exit()
        print(f"Lecture Recorder has closed ({reason}).{' ' + r['error'] if 'error' in r else ''}", flush=True)
        return r

    app.state.quit = quit_app
    app.state.request_exit = lambda: None  # serve() replaces it with the server's own exit

    if monitor:
        def _monitor():  # catches recordings that end on their own (2 h limit, device loss)
            while not closed:
                try:
                    rec.reap()
                    rec.check_waiting()
                    if idle.due(rec.busy):
                        with rec._lock:  # no recording can start between this check and the quit
                            if not rec.busy():
                                quit_app(f"no open page for {rec.cfg.server.idle_quit_minutes} min")
                except Exception:
                    pass
                time.sleep(2)

        threading.Thread(target=_monitor, daemon=True, name="recording-monitor").start()

    from lecture_recorder.search.index import SearchIndex

    search_index = SearchIndex(data_dir, cfg)
    app.state.search = search_index
    query_vectors: OrderedDict = OrderedDict()  # recent concept queries -> vector (no GPU for a repeat)

    def _session(slug: str, name: str) -> Path:
        try:
            d = lib.session_dir(data_dir, slug, name).resolve()
        except lib.LibraryError as e:
            raise HTTPException(404, str(e))
        if not d.is_relative_to(data_dir) or not d.is_dir():
            raise HTTPException(404, "No such session.")
        return d

    # ---------------------------------------------------------------- UI
    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", media_type="text/html")

    @app.get("/static/{name}")
    def static(name: str):
        p = (STATIC / name).resolve()
        if p.parent != STATIC.resolve() or not p.is_file():
            raise HTTPException(404)
        return FileResponse(p)

    @app.get("/api/health")
    def health():
        return {"ok": True, "app": "lecture-recorder"}  # the one open path: no data, so a second start can recognize us

    @app.get("/api/quit")
    def quit_plan():
        """What Quit would stop (the page asks for confirmation with exactly this)."""
        return rec.activity()

    @app.post("/api/quit")
    def quit_now(body: dict = Body(...)):
        """Quit: `confirm` must be the plan the user saw; anything else active now is refused (409)."""
        plan = rec.activity()
        if body.get("confirm") != plan:
            raise HTTPException(409, "Something started or stopped meanwhile. Press Quit again to see what will stop.")
        return {"closed": True, **quit_app("Quit in the app")}

    # ---------------------------------------------------------------- classes and sessions
    @app.get("/api/classes")
    def classes():
        return lib.list_classes(data_dir)

    @app.post("/api/classes", status_code=201)
    def new_class(body: dict = Body(...)):
        try:
            return lib.create_class(data_dir, body.get("name", ""), body.get("max_length_minutes"))
        except (lib.LibraryError, TypeError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.get("/api/classes/{slug}")
    def class_info(slug: str):
        try:
            c = lib.get_class(data_dir, slug)
        except lib.LibraryError:
            c = None
        if not c:
            raise HTTPException(404, "No such class.")
        states = lib._job_states(data_dir)  # one sweep of the job files for both of these
        flagged, suggested = dup.for_class(data_dir, slug), autosort.for_class(data_dir, slug, states=states)
        return {**c, "sessions": [{**s, "duplicate": s["name"] in flagged, "suggested": suggested.get(s["name"])}
                                  for s in lib.list_sessions(data_dir, slug, states)]}

    @app.post("/api/classes/{slug}/register", status_code=201)
    def register_class(slug: str):
        """Give a class folder without class.json (an older terminal recording) its settings file."""
        try:
            return lib.register_class(data_dir, slug)
        except lib.LibraryError as e:
            raise HTTPException(400, str(e))

    # ---------------------------------------------------------------- Recently deleted
    def _trash(fn, *args, **kw) -> dict:
        try:
            with rec._lock:  # not while this app starts a recording; a just-ended one is finalized (and queued) first
                rec.reap()
                r = fn(*args, **kw)
        except trash.TrashError as e:
            raise HTTPException(e.status, str(e))
        return {k: r.get(k) for k in ("id", "kind", "class", "session", "title", "sessions", "requeued", "deleted",
                                      "class_name", "from_class", "from_session")
                if k in r} | {"trash_version": trash.version(data_dir)}

    @app.post("/api/sessions/{slug}/{name}/delete")
    def delete_session(slug: str, name: str):
        """Move a lecture to Recently deleted (refused while it records or its job runs)."""
        return _trash(trash.delete_session, data_dir, slug, name, index=search_index)

    @app.post("/api/classes/{slug}/delete")
    def delete_class(slug: str, body: dict = Body(...)):
        """Move a class and all its lectures to Recently deleted. `sessions` is the lecture count the user
        confirmed; if it no longer matches, nothing is deleted."""
        n = body.get("sessions")
        if isinstance(n, bool) or not isinstance(n, int):
            raise HTTPException(400, "Confirm how many lectures the class has.")
        return _trash(trash.delete_class, data_dir, slug, expected_sessions=n, index=search_index)

    @app.post("/api/sessions/{slug}/{name}/move")
    def move_lecture(slug: str, name: str, body: dict = Body(...)):
        """Move a lecture to another class. `name` and `undo`: Undo, back to the original name."""
        from lecture_recorder.move import move_session

        return _trash(move_session, data_dir, slug, name, str(body.get("to", "")), prefer=body.get("name"),
                      undo=body.get("undo") is True, index=search_index)

    @app.get("/api/trash")
    def trash_list():
        return {"items": trash.list_trash(data_dir), "trash_days": rec.cfg.retention.trash_days}

    @app.post("/api/trash/{tid}/restore")
    def trash_restore(tid: str):
        r = _trash(trash.restore, data_dir, tid, index=search_index)
        if r.get("requeued") and rec.auto:
            rec.kick()  # manual mode: its jobs wait in the queue, as before the delete
        return r

    @app.post("/api/trash/{tid}/delete")
    def trash_delete(tid: str, body: dict = Body(...)):
        """Permanent. `confirm` must be the item's name, typed by the user."""
        return _trash(trash.delete_permanently, data_dir, tid, body.get("confirm"))

    @app.get("/api/sessions/{slug}/{name}")
    def session(slug: str, name: str):
        _session(slug, name)
        return lib.session_detail(data_dir, slug, name)

    @app.get("/api/sessions/{slug}/{name}/duplicates")
    def session_duplicates(slug: str, name: str):
        d = _session(slug, name)
        return {"matches": dup.for_class(data_dir, d.parent.parent.name, d.name).get(d.name, [])}

    @app.post("/api/sessions/{slug}/{name}/duplicates/dismiss")
    def dismiss_duplicate(slug: str, name: str, body: dict = Body(...)):
        """Keep both: this pair is not flagged again."""
        _session(slug, name)
        try:
            dup.dismiss(data_dir, slug, name, str(body.get("class", "")), str(body.get("session", "")))
        except (lib.LibraryError, FileNotFoundError):
            raise HTTPException(404, "No such lecture.")
        except (OSError, MetaUnreadable) as e:
            raise HTTPException(409, f"Could not save that, try again ({e}).")
        return {"ok": True}

    @app.get("/api/sessions/{slug}/{name}/suggestion")
    def session_suggestion(slug: str, name: str):
        """The class this lecture seems to belong to, if the evidence is clear."""
        d = _session(slug, name)
        return {"suggestion": autosort.for_class(data_dir, d.parent.parent.name, d.name).get(d.name)}

    @app.post("/api/sessions/{slug}/{name}/keep-here")
    def keep_here(slug: str, name: str):
        """Keep here: this lecture's class is never questioned again."""
        _session(slug, name)
        try:
            autosort.keep_here(data_dir, slug, name)
        except (lib.LibraryError, FileNotFoundError):
            raise HTTPException(404, "No such lecture.")
        except (OSError, MetaUnreadable) as e:
            raise HTTPException(409, f"Could not save that, try again ({e}).")
        return {"ok": True}

    @app.post("/api/sessions/{slug}/{name}/rename")
    def rename_session(slug: str, name: str, body: dict = Body(...)):
        """Change the lecture's title (meta.json only; never the folder or the original file name)."""
        from lecture_recorder.naming import NamingError

        d = _session(slug, name)
        try:
            title = lib.rename_session(data_dir, slug, name, body.get("title"))
        except NamingError as e:
            raise HTTPException(400, str(e))
        except lib.LibraryError as e:
            raise HTTPException(409, str(e))
        except FileNotFoundError:
            raise HTTPException(404, "No such session.")
        except (OSError, MetaUnreadable) as e:  # the lock stayed busy, meta.json unreadable or held elsewhere
            raise HTTPException(409, f"Could not save the title, try again ({e}).")
        trash.signal_change(data_dir)  # other open pages re-render (a signal on the existing poll)
        return {"title": title, "class": slug, "session": d.name, "trash_version": trash.version(data_dir)}

    @app.get("/api/naming/preview")
    def naming_preview(template: str = ""):
        """What the naming template gives for a recording made now and for an imported file; nothing is saved."""
        from lecture_recorder.naming import NamingError, check_template, render, values
        from lecture_recorder.recorder.session import SESSION_NAME_FMT

        try:
            template = check_template(template)
        except NamingError as e:
            return {"valid": False, "error": str(e), "examples": []}
        if not template:
            return {"valid": True, "error": None, "examples": []}
        c = next((c for c in lib.list_classes(data_dir) if c["has_class_file"]), None)
        name, n = (c["name"], c["sessions"] + 1) if c else ("Biology 101", 1)
        now = datetime.now()
        memo = "Voice Memo 12.m4a"  # an imported file
        return {"valid": True, "error": None, "examples": [
            {"what": f"A lecture recorded now in {name}",
             "title": render(template, values(now, name, n)) or now.strftime(SESSION_NAME_FMT)},
            {"what": f"Importing “{memo}”", "title": render(template, values(now, name, n, memo)) or Path(memo).stem}]}

    @app.get("/api/sessions/{slug}/{name}/transcript")
    def transcript(slug: str, name: str):
        d = _session(slug, name)
        doc = read_json(d / EDITABLE, None)
        if doc is None:
            raise HTTPException(404, "Not transcribed yet.")
        return {**doc, "zoom_speakers": lib.zoom_speaker_hints(d, doc)}

    @app.put("/api/sessions/{slug}/{name}/transcript")
    def edit_transcript(slug: str, name: str, body: dict = Body(...)):
        """Save the user's edits (segment text and speaker) to transcript.json and re-render .md/.txt.
        transcript.raw.json is never touched."""
        d = _session(slug, name)
        doc = read_json(d / EDITABLE, None)
        if doc is None:
            raise HTTPException(404, "Not transcribed yet.")
        before = {MD: render_md(doc, d.name), TXT: render_txt(doc)}
        speakers = {s["id"]: s for s in doc.get("speakers", [])}
        by_id = {s["id"]: s for s in doc["segments"]}
        for edit in body.get("segments", []):
            seg = by_id.get(edit.get("id"))
            if seg is None:
                raise HTTPException(400, f"unknown segment {edit.get('id')!r}")
            if "text" in edit:
                text = str(edit["text"]).strip()
                if len(text) > 20000:
                    raise HTTPException(400, "segment text too long")
                seg["text"] = text
            if "speaker" in edit:
                if edit["speaker"] not in speakers:
                    raise HTTPException(400, f"unknown speaker {edit['speaker']!r}")
                seg["speaker"] = edit["speaker"]
                seg["role"] = speakers[edit["speaker"]]["role"]
        doc["edited"] = now_iso()
        write_json_atomic(d / EDITABLE, doc)
        kept = []
        for fname, text in ((MD, render_md(doc, d.name)), (TXT, render_txt(doc))):
            p = d / fname
            on_disk = p.read_text(encoding="utf-8") if p.exists() else None
            if on_disk is not None and on_disk != before[fname]:
                kept.append(fname)  # the user changed this file by hand: never overwrite it
            else:
                write_text_atomic(p, text)
        from lecture_recorder.search.embed import EMB_META, embed_state

        if (d / EMB_META).exists() and embed_state(d, cfg, doc) != "current":
            enqueue(data_dir, d, EMBED)  # edited text: refresh its concept-search vectors (exact search is live)
            if rec.auto:
                rec.kick()  # manual mode: waits for Process all pending / Index them now
        return {"ok": True, "edited": doc["edited"], "kept_hand_edited": kept}

    @app.post("/api/sessions/{slug}/{name}/process", status_code=202)
    def process(slug: str, name: str, body: dict | None = Body(None)):
        d = _session(slug, name)
        if (d / RAW).exists():
            raise HTTPException(409, "Already transcribed.")
        if (body or {}).get("anyway") is True:  # "Transcribe anyway" on a file flagged as no usable audio
            lib.transcribe_anyway(d)
        job = enqueue(data_dir, d)  # already queued: returns that job, and this just starts the queue
        rec.kick()
        return {"state": job["state"], "blocked": gpu_block_reason(data_dir)}

    @app.post("/api/process-pending", status_code=202)
    def process_pending():
        """Queue every session that was never processed (once), then start the queue (it still waits
        while recording or on battery)."""
        pending = lib.unprocessed_sessions(data_dir)
        for d in pending:
            enqueue(data_dir, d)
        rec.kick()
        return {"queued": len(pending), "blocked": gpu_block_reason(data_dir)}

    @app.get("/api/processing")
    def processing():
        from lecture_recorder.autoconfig import read_auto, user_overrides

        return {**rec.processing_status(), "unprocessed": len(lib.unprocessed_sessions(data_dir)),
                "delete_original_after_opus": rec.cfg.retention.delete_original_after_opus,
                "keep_awake": rec.cfg.processing.keep_awake, "trash_days": rec.cfg.retention.trash_days,
                "naming_template": rec.cfg.naming.template, "idle_quit_minutes": rec.cfg.server.idle_quit_minutes,
                "machine": {"picks": read_auto(cfg.home).get("transcription", {}), "overrides": user_overrides(cfg.home)}}

    @app.post("/api/configure")
    def configure_machine():
        """Detect this computer again and rewrite config.auto.toml (~1 s, no downloads, config.toml untouched).
        Job processes read the config when they start, so the next job uses the new picks."""
        from lecture_recorder.cli import run_configure_child

        try:
            r = run_configure_child(cfg.home, as_json=True)  # a child process: the server never loads CUDA DLLs
        except Exception as e:
            raise HTTPException(500, f"Could not detect this computer: {e}")
        return {"picks": r["settings"]["transcription"], "overrides": r["overrides"], "notes": r["notes"],
                "missing": r["missing"]}

    @app.put("/api/settings")
    def settings(body: dict = Body(...)):
        """Save Processing mode, keep-awake, the naming template, original-audio retention and "Empty Recently deleted after N days" to
        config.toml. Turning deletion of originals or emptying on needs `confirm_permanent_deletion: true` (the UI asks first)."""
        from lecture_recorder.settings import set_config_values

        values = {}
        if "processing_mode" in body:
            values[("processing", "mode")] = body["processing_mode"]
        if "keep_awake" in body:
            values[("processing", "keep_awake")] = body["keep_awake"]
        if "idle_quit_minutes" in body:
            values[("server", "idle_quit_minutes")] = body["idle_quit_minutes"]
        if "trash_days" in body:
            days = body["trash_days"]
            if days and body.get("confirm_permanent_deletion") is not True:
                raise HTTPException(400, "Emptying Recently deleted is permanent; confirm it to turn this on.")
            values[("retention", "trash_days")] = days
        if "naming_template" in body:
            from lecture_recorder.naming import NamingError, check_template

            try:
                values[("naming", "template")] = check_template(body["naming_template"])
            except NamingError as e:
                raise HTTPException(400, str(e))
        key = next((k for k in ("delete_original_after_opus", "delete_wav_after_opus") if k in body), None)
        if key:  # the older name is still accepted
            on = body[key]
            if on is True and body.get("confirm_permanent_deletion") is not True:
                raise HTTPException(400, "Deleting the original recordings is permanent; confirm it to turn this on.")
            values[("retention", "delete_original_after_opus")] = on
        try:
            set_config_values(cfg.home, values)
        except ValueError as e:
            raise HTTPException(400, str(e))
        fresh = load_config(cfg.home)
        rec.cfg = dataclasses.replace(rec.cfg, processing=fresh.processing, retention=fresh.retention, naming=fresh.naming,
                                      server=dataclasses.replace(rec.cfg.server, idle_quit_minutes=fresh.server.idle_quit_minutes))
        if rec.auto:
            rec.kick()  # switching to auto starts anything already queued (the GPU rule still applies)
        return processing()

    @app.get("/api/compact")
    def compact_dry_run():
        """Dry run of "Delete originals with an Opus copy": deletes nothing. Encoding older recordings that
        have no copy yet is `lecture-recorder compact` (long CPU work; not in a web request)."""
        from lecture_recorder.retention import compact_plan

        if not rec.cfg.retention.delete_original_after_opus:
            raise HTTPException(409, "Turn on \"Delete the original after a verified Opus copy\" first.")
        return compact_plan(data_dir, encode=False)

    @app.post("/api/compact")
    def compact_run(body: dict = Body(...)):
        """Delete the originals listed by the dry run the user confirmed (`token`), re-verifying each Opus copy."""
        from lecture_recorder.retention import compact

        if not rec.cfg.retention.delete_original_after_opus:
            raise HTTPException(409, "Turn on \"Delete the original after a verified Opus copy\" first.")
        if body.get("confirm") is not True or not isinstance(body.get("token"), str):
            raise HTTPException(400, "Confirm the dry run first.")
        try:
            return compact(data_dir, body["token"], encode=False)
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/sessions/{slug}/{name}/summarize", status_code=202)
    def summarize(slug: str, name: str):
        """Queue a summary. Only a missing or untouched placeholder summary.md is ever replaced."""
        from lecture_recorder.summary.session import replaceable

        d = _session(slug, name)
        if not (d / EDITABLE).exists():
            raise HTTPException(409, "Transcribe the lecture first.")
        if not replaceable(d):
            raise HTTPException(409, "summary.md already has content (yours or an earlier summary), so it won't be "
                                     "replaced. Rename or remove it to generate a new one.")
        job = enqueue(data_dir, d, SUMMARIZE)
        rec.kick()
        return {"state": job["state"]}

    # ---------------------------------------------------------------- search
    def _scope(class_slug: str | None, exam: str | None, units: str | None, date_from: str | None,
               date_to: str | None) -> dict | None:
        """Search scope from the query: a saved exam, unit ids, or a date range (all need the class)."""
        if not (exam or units is not None or date_from or date_to):
            return None
        if not class_slug:
            raise HTTPException(400, "A unit, exam or date scope needs its class.")
        if exam:
            e = un.get_exam(lib.class_dir(data_dir, class_slug), exam) if un.EXAM_ID_RE.match(exam) else None
            if e is None:
                raise HTTPException(404, "No such exam.")
            scope = {k: e[k] for k in ("units", "from", "to") if k in e}
        elif units is not None:
            scope = {"units": [u.strip() for u in units.split(",") if u.strip()]}
        else:
            scope = {"from": date_from, "to": date_to}
        try:
            return un.check_scope(scope)  # an empty or malformed scope is refused, never widened
        except un.UnitError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/search")
    def search(q: str = "", mode: str = "exact", class_slug: str | None = None, exam: str | None = None,
               units: str | None = None, date_from: str | None = None, date_to: str | None = None):
        if class_slug and not lib.CLASS_SLUG_RE.match(class_slug):
            raise HTTPException(400, "invalid class")
        scope = _scope(class_slug, exam, units, date_from, date_to)
        q = q[:500]
        if mode == "exact":
            return {**search_index.exact(q, class_slug, scope=scope), "scope": scope}
        if mode != "concept":
            raise HTTPException(400, "mode must be exact or concept")
        from lecture_recorder.pipeline.gpu import GpuBusy
        from lecture_recorder.search.embed import embed_query
        from lecture_recorder.search.text import tokens
        from lecture_recorder.summary.llm import LLMError

        status = search_index.concept_status()
        out = {"mode": "concept", "query": q, "status": status, "results": []}
        if not tokens(q):
            out["message"] = "Type a few words describing the idea you are looking for."
            return out
        if not status["windows"]:
            out["message"] = "No lectures are indexed for concept search yet."
            return out
        key = (cfg.llm.embedding_model, " ".join(q.split()).lower())
        vec = query_vectors.get(key)
        if vec is None:
            blocked = gpu_block_reason(data_dir)  # the query embedding is GPU use too (a cached one is not)
            if blocked:
                return JSONResponse({"detail": f"Search by meaning is paused ({blocked}): it needs the GPU. "
                                               "Exact search works now."}, status_code=503)
            try:
                vec = embed_query(q, cfg)
            except GpuBusy:
                return JSONResponse({"detail": "The GPU is busy with a processing job (transcription, summary or "
                                               "indexing). Concept search needs it for a moment: try again shortly. "
                                               "Exact search works now."}, status_code=503)
            except LLMError as e:
                return JSONResponse({"detail": f"Concept search is unavailable: {e}"}, status_code=503)
            query_vectors[key] = vec
            while len(query_vectors) > 64:
                query_vectors.popitem(last=False)
        out["results"] = search_index.concept(vec, class_slug, sync=False, scope=scope)  # status just synced
        out["scope"] = scope
        return out

    @app.get("/api/search/status")
    def search_status():
        return search_index.concept_status()

    @app.post("/api/search/index-missing", status_code=202)
    def index_missing():
        """Queue concept-search indexing (GPU, serial queue) for transcribed sessions that lack it."""
        pending = search_index.pending_embedding()
        for d in pending:
            enqueue(data_dir, d, EMBED)
        if pending:
            rec.kick()
        return {"queued": len(pending)}

    # ---------------------------------------------------------------- units and exams
    def _class(slug: str) -> Path:
        try:
            c = lib.class_dir(data_dir, slug)
        except lib.LibraryError:
            raise HTTPException(404, "No such class.")
        if not (c / "class.json").is_file():
            raise HTTPException(404, "No such class.")
        return c

    def _unit_call(fn, *args):
        try:
            return fn(*args)
        except un.UnitError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/classes/{slug}/units")
    def class_units(slug: str):
        c = _class(slug)
        sessions = un.class_sessions(c)
        placed = un.ordered_units(c, sessions)
        return {"units": un.unit_spans(c, sessions), "sessions": sessions,
                "orphans": [u["name"] for u in placed if u.get("orphan")],
                "moved": [u["name"] for u in placed if u.get("moved")]}

    @app.post("/api/classes/{slug}/units", status_code=201)
    def add_unit(slug: str, body: dict = Body(...)):
        """User action: a new unit starts at (session, t). Suggestions only become boundaries this way.
        Optional "id": Undo of a removal brings the unit back under its old id (exams keep it)."""
        try:
            t = float(body.get("t"))
        except (TypeError, ValueError):
            raise HTTPException(400, "t must be a number of seconds")
        uid = body.get("id")  # Undo after a removal: the same unit comes back
        return _unit_call(un.add_boundary, _class(slug), str(body.get("session", "")), t, body.get("name"),
                          None if uid is None else str(uid))

    @app.patch("/api/classes/{slug}/units/{unit_id}")
    def rename_unit(slug: str, unit_id: str, body: dict = Body(...)):
        _unit_call(un.rename_unit, _class(slug), unit_id, str(body.get("name", "")))
        return {"ok": True}

    @app.delete("/api/classes/{slug}/units/{unit_id}")
    def remove_unit(slug: str, unit_id: str):
        _unit_call(un.remove_boundary, _class(slug), unit_id)
        return {"ok": True}

    @app.get("/api/classes/{slug}/unit-suggestions")
    def unit_suggestions(slug: str):
        """Topic-shift suggestions from the stored search vectors (CPU only, nothing is changed)."""
        from lecture_recorder.search.units_suggest import suggest_boundaries

        return suggest_boundaries(_class(slug), cfg, index=search_index)

    def _exam_view(c: Path, e: dict, sessions: list[dict], all_jobs: list[dict]) -> dict:
        from lecture_recorder.summary.exam import EXAM_META, material_is_stale, material_state

        d = un.exam_dir(c, e["id"])
        job = next((j for j in reversed(all_jobs) if kind_of(j) == "exam" and j["session"] == str(d.resolve())), None)
        meta = read_json(d / EXAM_META, {}) or {}
        scope_units = un.scope_spans(c, e, sessions)
        material = material_state(d)
        return {**e, "scope_units": scope_units, "material": material,
                "stale": material != "none" and material_is_stale(d, c, e),
                "moved": [u["name"] for u in scope_units if u.get("moved")],
                "job": job["state"] if job else None, "material_state": meta.get("state"),
                "error": meta.get("error") if meta.get("state") == "failed" else None, "warning": meta.get("warning")}

    @app.get("/api/classes/{slug}/exams")
    def class_exams(slug: str):
        c = _class(slug)
        sessions, all_jobs = un.class_sessions(c), jobs(data_dir)  # read once for all exams
        return [_exam_view(c, e, sessions, all_jobs) for e in un.load_exams(c)]

    @app.post("/api/classes/{slug}/exams", status_code=201)
    def new_exam(slug: str, body: dict = Body(...)):
        ids = body.get("units") or None
        if ids is not None and not (isinstance(ids, list) and all(isinstance(x, str) for x in ids)):
            raise HTTPException(400, "units must be a list of unit ids")
        return _unit_call(un.save_exam, _class(slug), str(body.get("name", "")), ids, body.get("from"), body.get("to"))

    def _exam(slug: str, exam_id: str) -> tuple[Path, dict]:
        c = _class(slug)
        e = un.get_exam(c, exam_id) if un.EXAM_ID_RE.match(exam_id) else None
        if e is None:
            raise HTTPException(404, "No such exam.")
        return c, e

    @app.get("/api/exams/{slug}/{exam_id}")
    def exam_info(slug: str, exam_id: str):
        c, e = _exam(slug, exam_id)
        sessions = un.class_sessions(c)
        return {**_exam_view(c, e, sessions, jobs(data_dir)), "class": slug, "blocked": gpu_block_reason(data_dir),
                "titles": {s["name"]: s["title"] for s in sessions}}

    @app.post("/api/exams/{slug}/{exam_id}/generate", status_code=202)
    def exam_generate(slug: str, exam_id: str):
        """Queue the study material (local LLM, serial queue, GPU/power rules). Never replaces an edited exam.md."""
        from lecture_recorder.pipeline.queue import EXAM
        from lecture_recorder.summary.exam import material_state

        c, e = _exam(slug, exam_id)
        d = un.exam_dir(c, e["id"])
        if material_state(d) == "user":
            raise HTTPException(409, "exam.md was edited, so it won't be replaced. Rename or remove it first.")
        d.mkdir(parents=True, exist_ok=True)
        job = enqueue(data_dir, d, EXAM)
        rec.kick()  # an explicit user request, like Process now
        return {"state": job["state"], "blocked": gpu_block_reason(data_dir)}

    @app.get("/files-exam/{slug}/{exam_id}/exam.md")
    def exam_file(slug: str, exam_id: str):
        from lecture_recorder.summary.exam import EXAM_MD

        c, e = _exam(slug, exam_id)
        p = (un.exam_dir(c, e["id"]) / EXAM_MD).resolve()
        if not p.is_relative_to(data_dir) or not p.is_file():
            raise HTTPException(404)
        return FileResponse(p, media_type="text/markdown; charset=utf-8")

    @app.get("/api/exams/{slug}/{exam_id}/bundle")
    def exam_bundle(slug: str, exam_id: str, include_audience: bool = False):
        """Download-only Markdown for the user to give to another AI themselves. Built here, sent nowhere."""
        from fastapi.responses import Response

        from lecture_recorder.summary.bundle import build_bundle

        c, e = _exam(slug, exam_id)
        text = build_bundle(c, e, include_audience=include_audience)
        return Response(text, media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{slug}-{e["id"]}-bundle.md"'})

    @app.get("/api/queue")
    def queue():
        return [{k: j.get(k) for k in ("id", "state", "created", "started", "finished")}
                | {"kind": kind_of(j), "session": Path(j["session"]).name, "class": Path(j["session"]).parent.parent.name}
                for j in jobs(data_dir)]

    # ---------------------------------------------------------------- files (plain files on disk)
    @app.get("/files/{slug}/{name}/{filename}")
    def file(slug: str, name: str, filename: str):
        if filename not in lib.SERVABLE:
            raise HTTPException(404)
        d = _session(slug, name)
        p = (d / filename).resolve()
        if not p.is_relative_to(data_dir) or p.parent != d or not p.is_file():
            raise HTTPException(404)
        media = {".md": "text/markdown; charset=utf-8", ".txt": "text/plain; charset=utf-8",
                 ".json": "application/json", ".opus": "audio/ogg", ".ogg": "audio/ogg", ".m4a": "audio/mp4",
                 ".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
                 ".vtt": "text/plain; charset=utf-8"}.get(p.suffix)
        return FileResponse(p, media_type=media)  # supports Range requests (seeking)

    # ---------------------------------------------------------------- recording
    @app.get("/api/devices")
    def devices():
        from lecture_recorder.recorder.sources import list_input_devices

        try:
            return list_input_devices()
        except Exception as e:
            return JSONResponse({"detail": f"could not list microphones: {e}"}, status_code=503)

    @app.get("/api/capabilities")
    def capabilities():
        """What this computer's OS offers, so the page can leave out what doesn't exist."""
        from lecture_recorder.host import host

        return {"os": host().name, "app_capture_hidden": host().app_capture_hidden, "mic_hint": host().mic_hint(),
                "app_capture_reason": host().app_capture_reason if host().app_capture_hidden else None}

    @app.get("/api/apps")
    def audio_apps():
        """Apps with audio open on the default output, for "Record from an app" (Zoom first)."""
        from lecture_recorder.host import host

        if why := host().app_capture_reason:
            return {"supported": False, "reason": why, "apps": []}
        from lecture_recorder.recorder.apps import list_audio_apps

        try:
            return {"supported": True, "reason": None, "apps": list_audio_apps()}
        except OSError as e:
            return JSONResponse({"detail": f"could not list apps: {e}"}, status_code=503)

    @app.get("/api/record/status")
    def record_status():
        # pages re-render when trash_version changes; "processing" drives the progress bars
        return {**rec.status(), "trash_version": trash.version(data_dir), "processing": rec.progress_status()}

    @app.post("/api/record/start", status_code=201)
    def record_start(body: dict = Body(...)):
        try:
            pid = body.get("app_pid")
            if pid is not None and (not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0):
                raise HTTPException(400, "Invalid app.")
            h = rec.start(str(body.get("class", "")), body.get("device"), pid, body.get("include_mic") is True)
        except RecorderError as e:
            raise HTTPException(409, str(e))
        except lib.LibraryError as e:
            raise HTTPException(400, str(e))
        return {"session": h.name, "reminder": PERMISSION_REMINDER}

    def _cmd(action: str):
        h = rec.handle
        if h is None or not h.is_running():
            raise HTTPException(409, "Nothing is recording.")
        getattr(h, action)()
        return {"ok": True}

    @app.post("/api/record/pause")
    def record_pause():
        return _cmd("pause")

    @app.post("/api/record/resume")
    def record_resume():
        return _cmd("resume")

    @app.post("/api/record/bookmark")
    def record_bookmark():
        return _cmd("bookmark")

    @app.post("/api/record/stop")
    def record_stop():
        return {"ok": True, "meta": rec.stop()}

    # ---------------------------------------------------------------- import
    def _import_class(class_slug: str) -> str:
        if not lib.real_class(data_dir, class_slug):  # "Not sure yet" (Unsorted) appears on first use
            raise HTTPException(400, "Choose an existing class.")
        return class_slug

    def _upload(file: UploadFile, class_slug: str, exts: set[str], limit: int, what: str) -> tuple[Path, str, int, str]:
        """Save one upload next to the class's sessions (fsynced). Returns (temp file, original name, bytes, SHA-256)."""
        original = Path(file.filename or "audio").name
        ext = Path(original).suffix.lower()
        if ext not in exts:
            raise HTTPException(400, f"Unsupported {what} type {ext or '(none)'}; use {', '.join(sorted(e[1:] for e in exts))}.")
        try:
            tmp = imp.temp_path(data_dir, class_slug, ext)
        except imp.ImportFailed as e:
            raise HTTPException(400, str(e))
        size = 0
        sha = hashlib.sha256()
        try:
            with open(tmp, "wb") as out:
                while chunk := file.file.read(1 << 20):
                    size += len(chunk)
                    sha.update(chunk)
                    if size > limit:
                        raise HTTPException(413, f"File too large ({limit // 2**20} MB max)." if limit < 2**30
                                            else f"File too large ({limit // 2**30} GB max).")
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)  # only the partial file this request wrote
            raise
        return tmp, original, size, sha.hexdigest()

    def _already_imported(sha: str, original: str, size: int) -> JSONResponse | None:
        """The 409 that stops a repeat import until the user chooses Import anyway."""
        seen = dup.find_source(data_dir, sha, original, size)
        return seen and JSONResponse(status_code=409, content={
            "detail": f"Already imported as {seen['title']}.", "already_imported": seen})

    def _imported(d: Path) -> dict:
        if rec.auto:  # manual mode: stays "Not processed" until the user starts it
            enqueue(data_dir, d)
            rec.kick()
        return {"class": d.parent.parent.name, "session": d.name}

    @app.post("/api/import/suggest")
    def import_suggest(body: dict = Body(...)):
        """Which class to pre-select for the picked files, or for a Zoom folder's topic and time. Names only;
        nothing is read, uploaded or filed."""
        files = body.get("files") or []
        if not isinstance(files, list) or not all(isinstance(f, dict) for f in files):
            raise HTTPException(400, "Invalid file list.")
        topic, when = body.get("topic"), body.get("when")
        return {"suggestion": autosort.for_import(data_dir, files[:200], str(topic)[:300] if topic else None,
                                                  str(when) if when else None)}

    @app.post("/api/import", status_code=201)
    def import_audio(file: UploadFile = File(...), class_slug: str = Form(...), last_modified_ms: int | None = Form(None),
                     anyway: bool = Form(False)):
        # Upload next to the sessions first: the folder name comes from the recording date in the file.
        tmp, original, size, sha = _upload(file, _import_class(class_slug), imp.IMPORT_EXTS, imp.MAX_IMPORT_BYTES, "file")
        if not anyway and (seen := _already_imported(sha, original, size)):
            tmp.unlink(missing_ok=True)
            return seen
        try:
            d = imp.create_session(data_dir, class_slug, tmp, original, size=size, last_modified_ms=last_modified_ms,
                                   source={"sha256": sha}, template=rec.cfg.naming.template)
        except imp.ImportFailed as e:
            raise HTTPException(400, str(e))
        return _imported(d)

    @app.post("/api/import/zoom-plan")
    def import_zoom_plan(body: dict = Body(...)):
        """Which files of a picked Zoom folder to upload (names and sizes only; nothing is read)."""
        try:
            files = [(str(f["name"])[:1000], int(f.get("size") or 0)) for f in body.get("files", [])[:5000]]
        except (TypeError, ValueError, KeyError):
            raise HTTPException(400, "Invalid file list.")
        p = zoom.plan(files)
        return {**p, "when": p["when"].isoformat(timespec="seconds") if p["when"] else None}

    @app.post("/api/import/zoom", status_code=201)
    def import_zoom(class_slug: str = Form(...), folder: str = Form(""), audio: UploadFile = File(...),
                    transcript: UploadFile | None = File(None), chat: UploadFile | None = File(None),
                    last_modified_ms: int | None = Form(None), anyway: bool = Form(False)):
        _import_class(class_slug)
        title, when = zoom.folder_date_title(folder[:300], audio.filename or "")
        tmps: list[Path] = []
        try:
            tmp, original, size, sha = _upload(audio, class_slug, {".m4a", ".mp4"}, imp.MAX_IMPORT_BYTES, "Zoom audio")
            tmps.append(tmp)
            if not anyway and (seen := _already_imported(sha, original, size)):
                return seen
            extras = {}
            for up, name, ext in ((transcript, zoom.TRANSCRIPT_FILE, ".vtt"), (chat, zoom.CHAT_FILE, ".txt")):
                if up is not None and up.filename:
                    t, *_ = _upload(up, class_slug, {ext}, zoom.MAX_TEXT_BYTES, "file")
                    tmps.append(t)
                    extras[name] = t
            d = imp.create_session(data_dir, class_slug, tmp, original, size=size, when=when,
                                   date_from="Zoom folder name" if when else None, last_modified_ms=last_modified_ms,
                                   title=title or Path(original).stem, source={"zoom": True, "folder": folder[:255], "sha256": sha},
                                   extras=extras, template=rec.cfg.naming.template)
        except imp.ImportFailed as e:
            raise HTTPException(400, str(e))
        finally:
            for t in tmps:
                t.unlink(missing_ok=True)  # consumed on success; leftovers only after a failed upload
        return _imported(d)

    return app


def port_state(host: str, port: int) -> str:
    """'free', 'this app' (another copy of Lecture Recorder answers there) or 'busy'. Loopback only."""
    import socket

    host = check_loopback(host)  # never bind to or contact anything but this computer
    v6 = ":" in host
    with socket.socket(socket.AF_INET6 if v6 else socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return "free"
        except OSError:
            pass
    import urllib.request

    netloc = f"[{host}]:{port}" if v6 else f"{host}:{port}"
    try:
        req = urllib.request.Request(f"http://{netloc}/api/health", headers={"Host": netloc})
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=2) as r:
            return "this app" if json.loads(r.read() or b"{}").get("app") == "lecture-recorder" else "busy"
    except Exception:
        return "busy"


def serve(config: Config | None = None, open_browser: bool = True) -> int:
    import uvicorn

    cfg = config or load_config()
    host = check_loopback(cfg.server.host)
    from lecture_recorder.recorder.control import recover_sessions

    for d in recover_sessions(cfg.data_dir):
        print(f"recovered unfinished session {d.name}")
    url = f"http://{host}:{cfg.server.port}/"
    state = port_state(host, cfg.server.port)
    if state != "free":
        if state == "this app":
            key = auth.read_login_key(cfg.home, cfg.server.port)
            if key is None:
                print(f"Lecture Recorder is already running at {url}, but its sign-in file is missing. Use Quit in the "
                      "open app (or end its window), then start it again.")
                return 1
            print(f"Lecture Recorder is already running at {url}; opening it.")
            if open_browser:
                import webbrowser

                webbrowser.open(f"{url}?{auth.LOGIN_PARAM}={key}")
            return 0
        print(f"Port {cfg.server.port} on {host} is used by another program. Close it, or start on another "
              f"port: run.bat --port 8766 (or set [server] port in config.toml).")
        return 2
    print(f"Lecture Recorder running at {url}  (Quit in the app, Ctrl+C, or close this window)")
    print(PERMISSION_REMINDER)
    key = auth.write_login_key(cfg.home, cfg.server.port)
    if open_browser:
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(f"{url}?{auth.LOGIN_PARAM}={key}")).start()
    app = create_app(cfg, login_key=key)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=cfg.server.port, log_level="warning",
                                           proxy_headers=False, server_header=False))
    app.state.request_exit = lambda: setattr(server, "should_exit", True)
    from lecture_recorder.recorder.platform import on_console_close

    keep = on_console_close(lambda: app.state.quit("the window was closed"))  # noqa: F841  (must stay referenced)
    try:
        server.run()
    finally:  # Ctrl+C or Quit: the recording is saved and a job ends (it runs again next time)
        app.state.quit("stopped")
    return 0

