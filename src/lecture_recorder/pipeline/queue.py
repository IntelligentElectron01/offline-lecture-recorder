"""Serial processing queue: plain JSON job files in `<data>/queue/`, one worker, one job at a time.

States: queued -> running -> done | failed, or queued -> cancelled (its lecture was deleted). Only one worker runs (OS file lock), so a job left
"running" by a crashed worker is re-queued when the next worker starts. Each job runs in its own
process (`python -m lecture_recorder.pipeline`), which also takes the app's GPU lock (`<app home>/.gpu.lock`).

Kinds: "transcribe" (Whisper + pyannote), "summarize" (local LLM) and "embed" (concept-search
vectors). A finished transcription queues its summary, and a finished summary job (done or not)
queues the embedding, so the models always run one after another, never alongside each other.
"""
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from lecture_recorder.host import host
from lecture_recorder.machine import gpu_block_reason
from lecture_recorder.pipeline.progress import ENV as PROGRESS_ENV
from lecture_recorder.recorder.platform import KeepAwake, ProcessLock
from lecture_recorder.recorder.session import now_iso, read_json, write_json_atomic

# Background jobs run at below-normal priority (per process; no system setting is changed).
# Jobs also get their own process group and no console, like the recorder: Ctrl+C or closing the
# app's window doesn't end them behind the queue's back; the app's Quit (queue.Stop) does, and re-queues them.
# Per OS in host/ (Windows: creation flags; elsewhere the job lowers its own priority when it starts).
LOW_PRIORITY = host().job_spawn_kwargs()
EXIT_BLOCKED = 75  # the job process found a recording or battery at its own start; the job stays queued
QUEUE_DIR = "queue"
WORKER_LOCK = ".worker.lock"
ENQUEUE_LOCK = ".enqueue.lock"
PROGRESS_DIR = "progress"  # a subfolder: jobs() globs only queue/*.json
ACTIVE = ("queued", "running")
RAW = "transcript.raw.json"
TRANSCRIBE, SUMMARIZE, EMBED, EXAM = "transcribe", "summarize", "embed", "exam"
KINDS = (TRANSCRIBE, SUMMARIZE, EMBED, EXAM)  # exam jobs name an exam folder instead of a session


def _queue_dir(data_dir: Path) -> Path:
    d = Path(data_dir) / QUEUE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def jobs(data_dir: Path) -> list[dict]:
    out = []
    for p in _queue_dir(data_dir).glob("*.json"):
        job = read_json(p)
        if job:
            job["_path"] = p
            out.append(job)
    return sorted(out, key=lambda j: (j["created"], j["id"]))


def progress_path(data_dir: Path, job_id: str) -> Path:
    """The running job's progress snapshot (pipeline/progress.py); gone once the job ends."""
    return Path(data_dir) / QUEUE_DIR / PROGRESS_DIR / f"{job_id}.json"


def _save(job: dict) -> None:
    write_json_atomic(job["_path"], {k: v for k, v in job.items() if k != "_path"})


@contextmanager
def queue_lock(data_dir: Path, wait_s: float = 10):
    lock = ProcessLock(_queue_dir(data_dir) / ENQUEUE_LOCK)
    deadline = time.monotonic() + wait_s
    while not lock.acquire():
        if time.monotonic() >= deadline:
            raise TimeoutError("queue is busy; try again")
        time.sleep(0.05)
    try:
        yield
    finally:
        lock.release()


def kind_of(job: dict) -> str:
    return job.get("kind", TRANSCRIBE)  # older job files have no kind


def enqueue(data_dir: Path, session_dir: Path, kind: str = TRANSCRIBE) -> dict:
    """Add a session job to the queue. The same session and kind is not added twice while active."""
    if kind not in KINDS:
        raise ValueError(f"unknown job kind {kind!r}")
    with queue_lock(data_dir):  # check-then-write is atomic across processes
        return _enqueue(data_dir, session_dir, kind)


def _enqueue(data_dir: Path, session_dir: Path, kind: str) -> dict:
    """`enqueue`, for a caller that already holds queue_lock."""
    session = str(Path(session_dir).resolve())
    if not Path(session).is_dir():  # e.g. moved to Recently deleted a moment ago
        raise FileNotFoundError(f"{session} no longer exists")
    for job in jobs(data_dir):
        if job["session"] == session and kind_of(job) == kind and job["state"] in ACTIVE:
            return job
    created = now_iso()
    job_id = f"{created.replace(':', '').replace('-', '')[:15]}-{uuid.uuid4().hex[:8]}"
    job = {"id": job_id, "session": session, "kind": kind, "state": "queued", "created": created,
           "_path": _queue_dir(data_dir) / f"{job_id}.json"}
    _save(job)
    return job


def _pending(data_dir: Path) -> list[dict]:
    return [j for j in jobs(data_dir) if j["state"] == "queued"]


def _under(job: dict, folder: Path) -> bool:
    return Path(job["session"]).is_relative_to(folder)


def running_under(data_dir: Path, folder: Path) -> bool:
    """True if a job for `folder` (a session, or anything inside a class folder) is running."""
    folder = Path(folder).resolve()
    return any(j["state"] == "running" and _under(j, folder) for j in jobs(data_dir))


def cancel_queued(data_dir: Path, folder: Path) -> list[dict]:
    """Mark the queued jobs for `folder` (or inside it) "cancelled" and return them. The caller holds
    `queue_lock`, and the worker starts a job only after re-reading it under that lock, so a cancelled
    job can never start."""
    folder = Path(folder).resolve()
    out = []
    for j in jobs(data_dir):
        if j["state"] == "queued" and _under(j, folder):
            j.update(state="cancelled", finished=now_iso())
            _save(j)
            out.append(j)
    return out


def repoint_jobs(data_dir: Path, old: Path, new: Path) -> None:
    """A lecture moved: its finished jobs follow it, so a failed job still shows. The caller holds
    `queue_lock` and has refused active jobs."""
    for j in jobs(data_dir):
        if Path(j["session"]) == old and j["state"] not in ACTIVE:
            j["session"] = str(new)
            _save(j)


class Stop:
    """The app is quitting: no further job starts, and a running job's process is ended; that job
    goes back to "queued" and starts from the beginning next time (its outputs are written only at the end)."""

    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._procs: set[subprocess.Popen] = set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def set(self) -> None:
        with self._lock:
            self._event.set()
            for p in self._procs:
                p.kill()

    def track(self, p: subprocess.Popen) -> None:
        with self._lock:
            self._procs.add(p)
            if self._event.is_set():  # quit arrived while this job was starting
                p.kill()

    def untrack(self, p: subprocess.Popen) -> None:
        with self._lock:
            self._procs.discard(p)


STOPPING = "the app is closing"


def _run_job(cmd: list[str], *, timeout: float, stop: Stop | None = None, **kw) -> subprocess.CompletedProcess:
    """subprocess.run(capture_output=True, text=True) that `stop` can end."""
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw) as p:
        if stop is not None:
            stop.track(p)
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
            raise
        finally:
            if stop is not None:
                stop.untrack(p)
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)


def run_worker(data_dir: Path, timeout_s: float = 4 * 3600, blocked: list | None = None,
               keep_awake: Callable[[], bool] = lambda: False, stop: Stop | None = None) -> list[dict] | None:
    """Process queued jobs in order until none are left, or until a GPU job may not start (recording
    or on battery; the reason is appended to `blocked`). Returns the finished jobs, or None if another
    worker is already running (it will pick up anything queued).

    `keep_awake()` (config [processing] keep_awake, read before each job) asks Windows not to idle-sleep
    while a job runs: held on this thread only while there is a job to run, released when the queue is
    empty or waits, on any error, and by Windows itself if the process ends.
    `stop` (the app's Quit): returns with STOPPING in `blocked` once the running job has been ended."""
    data_dir = Path(data_dir).resolve()
    finished: list[dict] = []
    awake = KeepAwake()  # per thread: hold() and release() both happen on this call's thread
    while True:
        lock = ProcessLock(_queue_dir(data_dir) / WORKER_LOCK)
        if not lock.acquire():
            return finished or None
        try:
            reason = _drain(data_dir, timeout_s, finished, awake, keep_awake, stop)
        finally:
            awake.release()
            lock.release()
        if reason:
            if blocked is not None:
                blocked.append(reason)
            return finished
        # A job queued after our last check but before the release would otherwise wait for the
        # next `process` call (its own worker was refused the lock). Look once more.
        if not _pending(data_dir):
            return finished


ARCHIVE_AFTER_S = 24 * 3600


def _archive_done(data_dir: Path) -> None:
    """Move finished (or cancelled) jobs older than a day to queue/archive/, so the per-request queue scan stays small.
    Session state never needs a "done" job (the files show it); failed jobs stay so the UI can say so,
    until a newer job of the same kind for that lecture replaces them."""
    from datetime import datetime

    archive = _queue_dir(data_dir) / "archive"
    now = time.time()
    all_jobs = jobs(data_dir)
    newest = {(j["session"], kind_of(j)): j["id"] for j in all_jobs}  # oldest first: the newest wins
    for job in all_jobs:
        try:
            age = now - datetime.fromisoformat(job.get("finished", "")).timestamp()
        except ValueError:
            continue
        # A failed job stays until a newer job of the same kind replaces it (the UI shows the newest one).
        superseded = job["state"] == "failed" and newest[(job["session"], kind_of(job))] != job["id"]
        if (job["state"] in ("done", "cancelled") or superseded) and age > ARCHIVE_AFTER_S:
            archive.mkdir(exist_ok=True)
            try:
                os.replace(job["_path"], archive / job["_path"].name)
            except OSError:
                pass  # busy right now; next time


def recover_stale(data_dir: Path) -> None:
    """A job left "running" by a crashed worker goes back to "queued" (or "done" if it had finished).
    Nothing is started. Does nothing if a worker is running right now."""
    lock = ProcessLock(_queue_dir(Path(data_dir)) / WORKER_LOCK)
    if not lock.acquire():
        return
    try:
        _recover(Path(data_dir).resolve())
    finally:
        lock.release()


def _drain(data_dir: Path, timeout_s: float, finished: list[dict], awake: KeepAwake,
           keep_awake: Callable[[], bool], stop: Stop | None = None) -> str | None:
    _archive_done(data_dir)
    _recover(data_dir)
    return _run_pending(data_dir, timeout_s, finished, awake, keep_awake, stop)


def _recover(data_dir: Path) -> None:
    for job in jobs(data_dir):
        if job["state"] == "running":  # we hold the worker lock, so its worker is gone
            done = _finished_before_crash(job)
            job.update(state="done" if done else "queued")
            if done:  # it crashed before queuing its follow-up job
                _follow_up(data_dir, job, ok=True, save=True)
            else:
                _save(job)
    # We hold the worker lock, so no job runs now: any progress snapshot is a crashed job's leftover.
    for p in (Path(data_dir) / QUEUE_DIR / PROGRESS_DIR).glob("*"):
        try:
            p.unlink()
        except OSError:
            pass


def _run_pending(data_dir: Path, timeout_s: float, finished: list[dict], awake: KeepAwake,
                 keep_awake: Callable[[], bool], stop: Stop | None = None) -> str | None:
    while True:
        if stop is not None and stop.is_set():
            return STOPPING
        pending = _pending(data_dir)
        if not pending:
            return None
        reason = gpu_block_reason(data_dir)  # every job kind uses the GPU
        if reason:
            return reason  # the remaining jobs wait; a running job was never interrupted
        if keep_awake():  # read per job, so a changed setting applies from the next job
            awake.hold()
        else:
            awake.release()
        job = pending[0]
        with queue_lock(data_dir):  # a delete cancels jobs under this lock: never start a cancelled one
            if (read_json(job["_path"]) or {}).get("state") != "queued":
                continue
            job.update(state="running", started=now_iso(), started_ts=time.time())
            _save(job)
        progress = progress_path(data_dir, job["id"])
        env = os.environ.copy()
        try:
            progress.parent.mkdir(exist_ok=True)
            env[PROGRESS_ENV] = str(progress)
        except OSError:  # no progress file then: the job runs anyway
            pass
        try:
            r = _run_job(
                [sys.executable, "-m", "lecture_recorder.pipeline", job["session"], "--data-dir", str(data_dir),
                 "--task", kind_of(job)],
                text=True, timeout=timeout_s, env=env, stop=stop, **LOW_PRIORITY,
            )
            ok, tail = r.returncode == 0, (r.stdout + r.stderr)[-2000:]
            if not ok and stop is not None and stop.is_set():  # ended by Quit: runs again next time
                job.update(state="queued")
                job.pop("started_ts", None)
                _save(job)
                return STOPPING
            if r.returncode == EXIT_BLOCKED:  # blocked in the instant between our check and its start
                job.update(state="queued")
                job.pop("started_ts", None)
                _save(job)
                return gpu_block_reason(data_dir) or (r.stdout.strip().splitlines() or ["blocked"])[-1]
        except subprocess.TimeoutExpired:
            ok, tail = False, f"timed out after {timeout_s:.0f} s"
        finally:
            try:
                progress.unlink(missing_ok=True)
            except OSError:
                pass
        job.update(state="done" if ok else "failed", finished=now_iso())
        if not ok:
            job["error"] = tail
        finished.append(job)
        _follow_up(data_dir, job, ok, save=True)  # picked up by this same loop


def _follow_up(data_dir: Path, job: dict, ok: bool, save: bool = False) -> None:
    """transcribe (ok) -> summarize -> embed. A failed summary still gets its session searchable.
    Nothing is queued for a lecture deleted meanwhile (enqueue refuses a missing folder). `save` writes the
    job's end in the same lock hold: a move or delete between the two would lose the rest of the pipeline."""
    try:
        with queue_lock(data_dir):
            if save:
                _save(job)
            _queue_follow_up(data_dir, job, ok)
    except FileNotFoundError:
        pass
    except TimeoutError:
        if save:
            _save(job)
        raise
    finally:
        if ok and kind_of(job) == TRANSCRIBE:
            _sketch(data_dir, Path(job["session"]))


def _sketch(data_dir: Path, session_dir: Path) -> None:
    """A tiny CPU step, not a job of its own (duplicate check, class suggestions). Nothing here may ever
    stop the queue."""
    try:
        from lecture_recorder import autosort, duplicates

        duplicates.after_transcribe(data_dir, session_dir)
        autosort.after_transcribe(data_dir)
    except Exception:
        logging.getLogger(__name__).exception("sketch skipped for %s", session_dir)


def _queue_follow_up(data_dir: Path, job: dict, ok: bool) -> None:
    """Queue what follows `job`. The caller holds queue_lock."""
    d = Path(job["session"])
    if ok and kind_of(job) == TRANSCRIBE:
        _enqueue(data_dir, d, SUMMARIZE)
    elif kind_of(job) == SUMMARIZE and (d / "transcript.json").exists():
        _enqueue(data_dir, d, EMBED)
    elif ok and kind_of(job) == EMBED:
        from lecture_recorder.config import load_config
        from lecture_recorder.search.embed import embed_state

        # The text was edited while the job ran (its enqueue was merged into this running job): embed again.
        # Only for an edit after this job started, so a job that can't make the vectors current never loops.
        try:
            edited_during = (d / "transcript.json").stat().st_mtime >= job.get("started_ts", float("inf"))
        except OSError:
            edited_during = False
        if edited_during and embed_state(d, load_config()) == "stale":
            _enqueue(data_dir, d, EMBED)


def _finished_before_crash(job: dict) -> bool:
    d = Path(job["session"])
    if kind_of(job) == SUMMARIZE:
        return (read_json(d / "summary.meta.json", {}) or {}).get("state") in ("done", "skipped")
    if kind_of(job) == EXAM:
        return (read_json(d / "exam.meta.json", {}) or {}).get("state") in ("done", "skipped")
    if kind_of(job) == EMBED:
        from lecture_recorder.config import load_config
        from lecture_recorder.search.embed import embed_state

        return embed_state(d, load_config()) == "current"
    return (d / RAW).exists()
