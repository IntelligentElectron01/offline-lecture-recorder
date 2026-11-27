"""Move one lecture to another class. Only ever on an explicit user action; never merges or overwrites.

A lecture is one folder, and the move is one os.rename on the same disk, so at every moment it is complete in
exactly one place. Nothing in a session folder stores its own path. What depends on the path outside it is
fixed up right after the rename, under a journal (data/queue/moves/), so a crash in between is finished at the
next start, `recover` or move:
- meta.json class and name;
- finished job files (active ones can't exist: a queued or running lecture is refused);
- Keep both pairs stored on other lectures (duplicates.repoint).
The lecture carries the journal's id in meta.json (MOVING) until all of that is done, travelling with the folder,
so recovery knows from the lecture itself whether it was moved, whatever else now has its old or new name.
The search index catches up in its next sync, from the folders. Unit boundaries and exams are not touched at all:
units.json keeps pointing at the lecture, so that unit shows as "moved" in the old class, exams keep its id,
their material shows as out of date, and Undo brings everything back as it was.
"""
import errno
import logging
import os
import time
import uuid
from pathlib import Path

from lecture_recorder import duplicates as dup
from lecture_recorder import library as lib
from lecture_recorder.pipeline.queue import ACTIVE, EXAM, QUEUE_DIR, jobs, kind_of, queue_lock, repoint_jobs
from lecture_recorder.recorder.session import (
    META_FILE, PCM_FILE, MetaUnreadable, meta_lock, read_json, update_meta, write_json_atomic,
)
from lecture_recorder.trash import Busy, Conflict, NotFound, _busy_reason, _real_dir, _session_rel, _signals_change

log = logging.getLogger(__name__)
MOVES_DIR = "moves"  # a subfolder: jobs() globs only queue/*.json
MOVING = "moving"  # meta.json: the journal of a move that isn't finished
WAS_SETTLED = "class_was_settled"  # meta.json: Keep here was on before this move, so Undo puts it back
RENAME_TRIES = 10  # ~1 s: a scanner or the search indexer briefly holding a file open


def _refuse_if_busy(data: Path, d: Path) -> None:
    why = _busy_reason(data, d, [d])
    if (d / PCM_FILE).exists():
        why = why or "its recording isn't saved yet"
    all_jobs = jobs(data)
    if any(j["state"] in ACTIVE and Path(j["session"]).is_relative_to(d) for j in all_jobs):
        why = why or "it is queued for processing"
    if any(j["state"] in ACTIVE and kind_of(j) == EXAM and Path(j["session"]).is_relative_to(d.parent.parent)
           for j in all_jobs):  # a queued one reads the units when it starts, so it would quietly lose this lecture
        why = why or "exam material for its class is queued or being written"
    tag = (read_json(d / META_FILE, {}) or {}).get(MOVING)
    if tag and _journal_path(data, tag).exists():  # _finish_pending couldn't finish it just now
        why = why or "its last move hasn't finished"
    if why:
        raise Busy(f"This lecture can't be moved now: {why}. Try again when it has finished.")


def _target_root(data: Path, to: str, undo: bool) -> Path:
    """The target class's sessions folder. Only a real class (a class.json), inside the data folder; Undo may also
    return to the folder without class settings that the lecture came from."""
    if not lib.real_class(data, to) and not (undo and lib.CLASS_SLUG_RE.fullmatch(to or "") and lib.get_class(data, to)):
        raise NotFound("No such class.")
    root = _real_dir(data, Path("classes", to)) / "sessions"
    root.mkdir(exist_ok=True)
    return _real_dir(data, Path("classes", to, "sessions"))


def _names(name: str):
    """`name`, then the next free-name candidates: never a merge, never an overwrite."""
    yield name
    yield from (f"{name[:19]}-{n}" for n in range(2, 1000))  # name[:19]: the date-time part


def _journal_path(data: Path, tag: str) -> Path:
    return data / QUEUE_DIR / MOVES_DIR / f"{tag}.json"


def _rename(src: Path, dst: Path) -> None:
    for attempt in range(RENAME_TRIES):
        if os.path.lexists(dst):  # macOS/Linux rename would silently replace an empty folder
            raise FileExistsError(str(dst))
        try:
            os.rename(src, dst)  # Windows: FileExistsError if dst appeared, so nothing is ever replaced
            return
        except OSError as e:
            if e.errno in (errno.EEXIST, errno.ENOTEMPTY):  # POSIX: a non-empty folder appeared in the last instant
                raise FileExistsError(str(dst)) from e
            if not isinstance(e, PermissionError):
                raise
            if attempt == RENAME_TRIES - 1:
                raise
            time.sleep(0.1)


def _finish(data: Path, entry: dict) -> None:
    """Everything after the rename. Each step can run twice; the MOVING tag goes last, so a crash anywhere in
    here is still recognised as this move and finished."""
    (old_class, old_name), (new_class, new_name) = entry["from"], entry["to"]
    new = lib.session_dir(data, new_class, new_name)
    was = entry.get(WAS_SETTLED)
    update_meta(new, _drop=() if was else (WAS_SETTLED,), **{"class": new_class, "name": new_name, "class_settled": entry["settled"],
                                                             **({WAS_SETTLED: True} if was else {})})
    repoint_jobs(data, lib.session_dir(data, old_class, old_name), new)
    dup.repoint(data, f"{old_class}/{old_name}", f"{new_class}/{new_name}")
    update_meta(new, _drop=(MOVING,))


def _abandon(d: Path, journal: Path) -> None:
    """The move didn't happen. If `d` has gone it did: the journal stays for recovery."""
    try:
        if d.is_dir():
            update_meta(d, _drop=(MOVING,))
            journal.unlink(missing_ok=True)
    except Exception:  # the journal stays, and the next start or move clears the tag
        log.exception("could not clear the move of %s", d)


def _do_move(data: Path, slug: str, name: str, to: str, prefer: str | None, undo: bool) -> dict:
    d = _real_dir(data, _session_rel(slug, name))
    _refuse_if_busy(data, d)
    root = _target_root(data, to, undo)
    if prefer is not None and not (isinstance(prefer, str) and lib.SESSION_NAME_RE.fullmatch(prefer)):
        raise NotFound("No such lecture.")
    tag = uuid.uuid4().hex
    meta = read_json(d / META_FILE, {}) or {}
    # A move is a decision, except to Unsorted ("not sure yet": suggestions stay on). Undo puts back what was there.
    entry = {"id": tag, "from": [slug, name], "settled": bool(meta.get(WAS_SETTLED)) if undo else to != lib.UNSORTED,
             WAS_SETTLED: not undo and bool(meta.get("class_settled"))}
    journal = _journal_path(data, tag)
    journal.parent.mkdir(parents=True, exist_ok=True)
    update_meta(d, **{MOVING: tag})  # an unreadable meta.json stops the move here, with nothing changed
    try:
        for new_name in _names(prefer or name):
            if os.path.lexists(root / new_name):
                continue
            entry["to"] = [to, new_name]
            write_json_atomic(journal, entry)
            try:
                with meta_lock(d):  # a title rename never lands in the middle
                    _rename(d, root / new_name)
                break
            except FileExistsError:  # appeared in the last instant
                continue
        else:
            raise Conflict("No free name left in that class.")
    except BaseException:
        _abandon(d, journal)
        raise
    try:
        _finish(data, entry)
        journal.unlink(missing_ok=True)
    except Exception:  # the lecture has moved; the journal stays, and the next start or move finishes this
        log.exception("move of %s/%s: left for recovery", slug, name)
    meta = read_json(root / new_name / META_FILE, {}) or {}
    return {"class": to, "session": new_name, "from_class": slug, "from_session": name,
            "title": meta.get("title") or new_name, "class_name": lib.get_class(data, to)["name"]}


@_signals_change
def move_session(data_dir: Path, slug: str, name: str, to: str, *, prefer: str | None = None, undo: bool = False,
                 index=None) -> dict:
    """Move a lecture to class `to`, keeping its folder name if free. `prefer` asks for another name (Undo:
    the original one). `undo` also forgets that the user settled this lecture's class."""
    data = Path(data_dir).resolve()
    if to == slug:
        raise Conflict("The lecture is already in that class.")
    try:
        with queue_lock(data):  # no job of this lecture can start meanwhile
            _finish_pending(data)
            r = _do_move(data, slug, name, to, prefer, undo)
    except PermissionError:
        raise Busy("Some of its files are in use (is it playing, or open in another program?). Close it and try again.")
    except (TimeoutError, MetaUnreadable) as e:
        raise Busy(f"The app is busy; try again ({e}).")
    if index is not None:
        try:
            index.sync()
        except Exception:  # the lecture has moved, and the next search syncs anyway: not a failed move
            log.exception("index sync after moving %s/%s", slug, name)
    return r


def finish_moves(data_dir: Path) -> int:
    """After a crash: finish what a journal says. Starts nothing, moves nothing."""
    data = Path(data_dir).resolve()
    root = data / QUEUE_DIR / MOVES_DIR
    if not root.is_dir() or not any(root.glob("*.json")):
        return 0
    with queue_lock(data):
        return _finish_pending(data)


def _tag_at(d: Path) -> str | None:
    if not d.is_dir():
        return None
    meta = read_json(d / META_FILE, None)
    if not isinstance(meta, dict):
        raise MetaUnreadable(f"{d / META_FILE} could not be read")  # can't tell yet: the journal stays
    return meta.get(MOVING)


def _finish_pending(data: Path) -> int:
    """Under queue_lock. The tag says where the lecture really is: at the journal's target, the move happened
    and its fix-ups get finished; at the old place, it never did. A target with no tag is a finished move (or
    another lecture with that name), so only the journal goes."""
    done = 0
    for p in sorted((data / QUEUE_DIR / MOVES_DIR).glob("*.json")):
        try:
            entry = read_json(p, None)
            old, new = (data / _session_rel(*entry[k]) for k in ("from", "to"))
            if _tag_at(new) == entry["id"]:
                _finish(data, entry)
                done += 1
            elif _tag_at(old) == entry["id"]:
                update_meta(old, _drop=(MOVING,))
            elif not new.is_dir():
                log.warning("move %s: its lecture has since been moved or deleted", p.name)
            p.unlink()
        except Exception:  # unreadable, or still can't finish: kept and tried again later
            log.exception("could not finish the move in %s", p.name)
    return done
