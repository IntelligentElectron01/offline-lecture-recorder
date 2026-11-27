"""Recently deleted: deleting a lecture or a class moves its folder into the data folder's trash;
only "Delete permanently" (which needs the item's name typed) or the opt-in "Empty after N days"
setting removes anything.

data/.trash/<id>/trash.json   {kind, path (relative to data/), class, session, title, deleted, ...}
data/.trash/<id>/item/        the moved folder, unchanged (a session folder, or a whole class folder)

Safety:
- Only folders inside the data folder are touched: class and session names must match their patterns
  (no `..` or separators), and every path component below the data folder must be a real folder, not
  a symlink or junction (checked before every move and permanent deletion).
- Moves are a single `os.rename` inside the data folder (same volume): never a copy-then-delete.
- A lecture that is recording, whose job is running, or whose recorder lock is live is refused (Busy).
- Deleting holds the queue lock, so the queue worker can't start that lecture's job meanwhile; its
  queued jobs are cancelled and re-queued by a restore.
"""
import functools
import os
import re
import shutil
import stat
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from lecture_recorder import library as lib
from lecture_recorder import units as un
from lecture_recorder.pipeline.queue import cancel_queued, enqueue, queue_lock, running_under
from lecture_recorder.recorder.platform import is_locked
from lecture_recorder.recorder.process import GLOBAL_LOCK
from lecture_recorder.recorder.session import (
    CLASS_SLUG_RE, LOCK_FILE, META_FILE, SESSION_NAME_RE, now_iso, read_json, session_dirs, write_json_atomic,
)

TRASH_DIR, ITEM, RECORD = ".trash", "item", "trash.json"
VERSION = "version.json"  # data/.trash/version.json: changes after every delete, restore or permanent delete
TRASH_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{8}$")
LIVE_STATES = ("starting", "recording", "paused", "finalizing")
STARTING_S = 60  # start_recording waits up to 20 s for its child; a "starting" meta this new is a live start


class TrashError(ValueError):
    status = 400


class NotFound(TrashError):
    status = 404


class Busy(TrashError):
    status = 409


class Conflict(TrashError):
    status = 409


# ------------------------------------------------------------------ paths
def _real_dir(data_dir: Path, rel: Path) -> Path:
    """data_dir / rel, refused unless every component below the data folder is a real folder (no
    symlink, no junction, no `..`) and the result stays inside the data folder."""
    data = Path(data_dir).resolve()
    if rel.is_absolute() or any(part in ("", ".", "..") for part in rel.parts):
        raise TrashError("invalid path")
    p = data
    for part in rel.parts:
        p = p / part
        try:
            st = os.lstat(p)
        except FileNotFoundError:
            raise NotFound("Not found (already deleted or restored?).")
        if stat.S_ISLNK(st.st_mode) or p.is_junction() or not stat.S_ISDIR(st.st_mode):
            raise TrashError(f"{rel.as_posix()} is a link or not a folder; it is left alone")
    if p.resolve() != p or not p.is_relative_to(data):
        raise TrashError("path outside the data folder")
    return p


def _session_rel(slug: str, name: str) -> Path:
    if not CLASS_SLUG_RE.match(slug or "") or not SESSION_NAME_RE.match(name or ""):
        raise NotFound("No such lecture.")
    return Path("classes", slug, "sessions", name)


def _class_rel(slug: str) -> Path:
    if not CLASS_SLUG_RE.match(slug or ""):
        raise NotFound("No such class.")
    return Path("classes", slug)


def _trash_root(data_dir: Path) -> Path:
    root = Path(data_dir).resolve() / TRASH_DIR
    root.mkdir(exist_ok=True)
    return _real_dir(data_dir, Path(TRASH_DIR))


def _item_dir(data_dir: Path, tid: str) -> Path:
    if not TRASH_ID_RE.match(tid or ""):
        raise NotFound("No such item in Recently deleted.")
    _trash_root(data_dir)
    return _real_dir(data_dir, Path(TRASH_DIR, tid))


# ------------------------------------------------------------------ change signal
def version(data_dir: Path) -> str:
    """Changes after each delete, restore or permanent delete (this app, another tab or the terminal), once
    the operation has finished (index synced). Open pages compare it on their existing status poll and
    re-render from fresh data."""
    v = read_json(Path(data_dir) / TRASH_DIR / VERSION, {})
    return str(v.get("v", "")) if isinstance(v, dict) else ""


def signal_change(data_dir: Path) -> None:
    """Also after a lecture rename: open pages pick up the new title on their next status poll."""
    try:
        write_json_atomic(_trash_root(data_dir) / VERSION, {"v": uuid.uuid4().hex}, fsync=False)
    except (OSError, TrashError):
        pass  # informational: open pages then update on their next navigation


def _signals_change(fn):
    """Bump the version when `fn` (data_dir first) returns or fails part-way: a refusal costs open pages
    one re-render, a missed change would leave them stale."""
    @functools.wraps(fn)
    def wrapper(data_dir, *args, **kw):
        try:
            return fn(data_dir, *args, **kw)
        finally:
            signal_change(data_dir)
    return wrapper


# ------------------------------------------------------------------ busy
def _busy_reason(data_dir: Path, folder: Path, sessions: list[Path]) -> str | None:
    recording = is_locked(Path(data_dir) / GLOBAL_LOCK)
    for d in sessions:
        if is_locked(d / LOCK_FILE):
            return f"{d.name} is recording"
        meta = read_json(d / META_FILE, {}) or {}
        if recording and meta.get("state") in LIVE_STATES or meta.get("state") == "starting" and started_recently(meta.get("created")):
            return f"{d.name} is starting to record"
    if running_under(data_dir, folder):
        return "its processing job is running"
    return None


def started_recently(iso: str | None) -> bool:
    """A "starting" meta this new is a live start, even before the recorder takes its lock."""
    try:
        return (datetime.now().astimezone() - datetime.fromisoformat(iso)).total_seconds() < STARTING_S
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------ delete
def _move_in(data_dir: Path, src: Path, record: dict) -> tuple[str, Path]:
    """Write the record, then rename `src` into the trash. On failure nothing is left behind."""
    tid = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    tdir = _trash_root(data_dir) / tid
    tdir.mkdir()
    write_json_atomic(tdir / RECORD, {**record, "id": tid, "deleted": now_iso()})
    try:
        os.rename(src, tdir / ITEM)
    except OSError as e:
        (tdir / RECORD).unlink(missing_ok=True)
        tdir.rmdir()
        if isinstance(e, PermissionError):
            raise Busy("Some of its files are in use (is it playing, or open in another program?). "
                       "Close it and try again.")
        raise
    return tid, tdir


def _rel_jobs(data_dir: Path, cancelled: list[dict]) -> list[dict]:
    data = Path(data_dir).resolve()
    return [{"kind": j.get("kind", "transcribe"), "path": Path(j["session"]).relative_to(data).as_posix()}
            for j in cancelled]


@_signals_change
def delete_session(data_dir: Path, slug: str, name: str, index=None) -> dict:
    """Move a lecture to Recently deleted. Its queued jobs are cancelled, unit boundaries that start in
    it are removed (kept for a restore), and the search index forgets it. Exam material that covered it
    becomes out of date by itself (its spans changed; summary/exam.py material_is_stale)."""
    rel = _session_rel(slug, name)
    with queue_lock(data_dir):
        d = _real_dir(data_dir, rel)
        class_dir = d.parent.parent
        why = _busy_reason(data_dir, d, [d])
        if why:
            raise Busy(f"This lecture can't be deleted now: {why}. Try again when it has finished.")
        meta = read_json(d / META_FILE, {}) or {}
        cname = lib.get_class(data_dir, slug)["name"]
        record = {"kind": "session", "path": rel.as_posix(), "class": slug, "class_name": cname, "session": name,
                  "title": meta.get("title") or name, "created": meta.get("created"),
                  "class_created": lib.get_class(data_dir, slug)["created"]}
        tid, tdir = _move_in(data_dir, d, record)
        record = {**read_json(tdir / RECORD, {}), "cancelled_jobs": _rel_jobs(data_dir, cancel_queued(data_dir, d))}
        write_json_atomic(tdir / RECORD, record)  # before anything else can fail
    record["units_removed"] = un.drop_session_boundaries(class_dir, name)
    write_json_atomic(tdir / RECORD, record)
    _sync(index)
    return record


@_signals_change
def delete_class(data_dir: Path, slug: str, expected_sessions: int | None = None, index=None) -> dict:
    """Move a whole class (lectures, units, exams) to Recently deleted. `expected_sessions` is the count
    the user confirmed; if the class has a different number now, nothing is deleted."""
    rel = _class_rel(slug)
    with queue_lock(data_dir):
        c = _real_dir(data_dir, rel)
        sessions = session_dirs(c)
        if expected_sessions is not None and expected_sessions != len(sessions):
            raise Conflict(f"This class now has {len(sessions)} lecture(s); confirm again.")
        why = _busy_reason(data_dir, c, sessions)
        if why:
            raise Busy(f"This class can't be deleted now: {why}. Try again when it has finished.")
        info = lib.get_class(data_dir, slug)
        record = {"kind": "class", "path": rel.as_posix(), "class": slug, "class_name": info["name"],
                  "title": info["name"], "sessions": len(sessions), "has_class_file": info["has_class_file"]}
        tid, tdir = _move_in(data_dir, c, record)
        record = {**read_json(tdir / RECORD, {}), "cancelled_jobs": _rel_jobs(data_dir, cancel_queued(data_dir, c))}
        write_json_atomic(tdir / RECORD, record)
    _sync(index)
    return record


# ------------------------------------------------------------------ list, restore, delete permanently
def _record(data_dir: Path, tid: str) -> tuple[Path, dict]:
    tdir = _item_dir(data_dir, tid)
    rec = read_json(tdir / RECORD, None)
    if not isinstance(rec, dict) or rec.get("kind") not in ("session", "class"):
        raise TrashError("This item's record is unreadable; it was left unchanged.")
    return tdir, rec


def _original(rec: dict) -> Path:
    """The record's original path, re-validated (never trusted as written)."""
    if rec["kind"] == "session":
        rel = _session_rel(rec.get("class"), rec.get("session"))
    else:
        rel = _class_rel(rec.get("class"))
    if rel.as_posix() != rec.get("path"):
        raise TrashError("This item's record doesn't match its name; it was left unchanged.")
    return rel


def list_trash(data_dir: Path) -> list[dict]:
    root = Path(data_dir).resolve() / TRASH_DIR
    if not root.is_dir():
        return []
    data = Path(data_dir).resolve()
    out = []
    for tdir in root.iterdir():
        if not TRASH_ID_RE.match(tdir.name):
            continue
        rec = read_json(tdir / RECORD, None)
        if not isinstance(rec, dict) or not (tdir / ITEM).is_dir():
            continue  # a delete interrupted before its move: nothing was moved
        try:
            occupied = (data / _original(rec)).exists()
        except TrashError:
            occupied = None
        out.append({k: rec.get(k) for k in ("id", "kind", "class", "class_name", "session", "title", "created",
                                             "deleted", "sessions")} | {"occupied": occupied})
    return sorted(out, key=lambda r: r["deleted"] or "", reverse=True)


@_signals_change
def restore(data_dir: Path, tid: str, index=None) -> dict:
    """Put an item back exactly where it was, re-queue its cancelled jobs, restore its unit boundaries and
    re-index it. Refused if the original place is taken, or (for a lecture) if its class is gone."""
    data = Path(data_dir).resolve()
    with queue_lock(data_dir):
        tdir, rec = _record(data_dir, tid)
        rel = _original(rec)
        item = _real_dir(data_dir, Path(TRASH_DIR, tid, ITEM))
        if os.path.lexists(data / rel):
            raise Conflict(f"Something is already at {rel.as_posix()}; it was not replaced. Move it away first.")
        if rec["kind"] == "session":
            try:
                class_dir = _real_dir(data_dir, rel.parent.parent)
            except NotFound:
                raise Conflict(f"Its class ({rec.get('class_name') or rec['class']}) no longer exists. "
                               "Restore the class first.")
            if "class_created" in rec and lib.get_class(data_dir, rec["class"])["created"] != rec["class_created"]:
                raise Conflict(f"The class {rec['class']} was deleted and created again since; this lecture "
                               "belongs to the old one. Restore that class first (move the new one away).")
            (class_dir / "sessions").mkdir(exist_ok=True)
            _real_dir(data_dir, rel.parent)
        else:
            (data / "classes").mkdir(exist_ok=True)
            _real_dir(data_dir, Path("classes"))
        try:
            os.rename(item, data / rel)
        except PermissionError:
            raise Busy("Some of its files are in use; close them and try again.")
        (tdir / RECORD).unlink(missing_ok=True)  # the item is back: its trash entry goes (under the lock)
        tdir.rmdir()
    requeued = 0
    for j in rec.get("cancelled_jobs") or []:  # enqueue takes the queue lock itself
        target = (data / str(j.get("path", ""))).resolve()
        if target.is_relative_to(data / rel) and target.is_dir():
            enqueue(data_dir, target, j.get("kind", "transcribe"))
            requeued += 1
    if rec["kind"] == "session":
        un.restore_boundaries(data / rel.parent.parent, rec.get("units_removed") or [])
    _sync(index)
    return {"kind": rec["kind"], "class": rec["class"], "session": rec.get("session"), "requeued": requeued}


@_signals_change
def delete_permanently(data_dir: Path, tid: str, confirm: str) -> dict:
    """Remove one item from Recently deleted for good. `confirm` must be the item's name as shown."""
    with queue_lock(data_dir):  # never alongside a restore of the same item
        tdir, rec = _record(data_dir, tid)
        if not isinstance(confirm, str) or confirm.strip() != (rec.get("title") or "").strip():
            raise TrashError(f"Type the name exactly ({rec.get('title')}) to delete it permanently.")
        _remove(data_dir, tdir)
    return {"deleted": tid, "title": rec.get("title")}


def _remove(data_dir: Path, tdir: Path) -> None:
    _real_dir(data_dir, tdir.relative_to(Path(data_dir).resolve()))  # checked again right before
    # shutil.rmtree never follows links or junctions inside the tree (it removes the link itself).
    shutil.rmtree(tdir)


@_signals_change
def empty_trash(data_dir: Path, ids: list[str]) -> int:
    """Delete the listed items for good: exactly those the user was shown and confirmed."""
    n = 0
    with queue_lock(data_dir):
        for tid in ids:
            try:
                _remove(data_dir, _item_dir(data_dir, tid))
                n += 1
            except NotFound:
                continue  # restored or deleted meanwhile
    return n


def purge_expired(data_dir: Path, days: int) -> int:
    """The opt-in "Empty after N days" setting (0 = never, the default)."""
    if not days:
        return 0
    cutoff = datetime.now().astimezone() - timedelta(days=days)
    with queue_lock(data_dir):
        n = _purge(data_dir, cutoff)
    if n:
        signal_change(data_dir)
    return n


def _purge(data_dir: Path, cutoff: datetime) -> int:
    n = 0
    for r in list_trash(data_dir):
        try:
            old = datetime.fromisoformat(r["deleted"]) < cutoff
        except (TypeError, ValueError):
            continue
        if old:
            try:
                _remove(data_dir, _item_dir(data_dir, r["id"]))
                n += 1
            except (OSError, TrashError):
                continue  # in use or not a real folder: kept, tried again next time
    return n


def _sync(index) -> None:
    if index is not None:
        index.sync()
