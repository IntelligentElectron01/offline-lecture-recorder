"""Session folders, default names, and crash-safe JSON files."""
import json
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

SESSION_NAME_FMT = "%Y-%m-%d_%H-%M-%S"
SESSION_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(-\d+)?$")
CLASS_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

PCM_FILE = "audio.pcm"
WAV_FILE = "audio.wav"
META_FILE = "meta.json"
STATUS_FILE = "status.json"
BOOKMARKS_FILE = "bookmarks.json"
LOCK_FILE = "recorder.lock"
CONTROL_DIR = "control"
LOG_FILE = "recorder.log"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sessions_root(data_dir: Path, class_slug: str) -> Path:
    if not CLASS_SLUG_RE.match(class_slug):
        raise ValueError(f"invalid class slug {class_slug!r} (use a-z, 0-9 and '-')")
    return data_dir / "classes" / class_slug / "sessions"


def new_session_dir(data_dir: Path, class_slug: str, now: datetime | None = None) -> Path:
    """Create and return an empty session folder named after the start date-time (local time).

    Two sessions started in the same second get `-2`, `-3`… suffixes; an existing folder is never reused.
    """
    root = sessions_root(data_dir, class_slug)
    root.mkdir(parents=True, exist_ok=True)
    base = (now or datetime.now()).strftime(SESSION_NAME_FMT)
    for n in range(1, 1000):
        path = root / (base if n == 1 else f"{base}-{n}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"could not allocate a session folder for {base}")


def session_dirs(class_dir: Path) -> list[Path]:
    """One class's session folders, in date-time order."""
    root = Path(class_dir) / "sessions"
    return sorted(p for p in root.iterdir() if p.is_dir() and SESSION_NAME_RE.match(p.name)) if root.is_dir() else []


def iter_session_dirs(data_dir: Path):
    classes = data_dir / "classes"
    if not classes.is_dir():
        return
    for path in sorted(classes.glob("*/sessions/*")):
        if path.is_dir() and SESSION_NAME_RE.match(path.name):
            yield path


def write_json_atomic(path: Path, obj, retries: int = 50, fsync: bool = True) -> None:
    """Write to a temp file, fsync, then rename over the target: readers see the old or new file, never half.
    `fsync=False` for informational files rewritten every second (status.json)."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        if fsync:
            os.fsync(f.fileno())
    for attempt in range(retries):  # Windows refuses the rename while another process has the target open
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.02)
    os.replace(tmp, path)


class MetaUnreadable(RuntimeError):
    """meta.json exists but could not be read: never replace it with a partial copy."""


META_LOCK = ".meta.lock"  # in the class's sessions folder (not a file in the lecture's own folder)


@contextmanager
def meta_lock(session_dir: Path, timeout_s: float = 10):
    """Cross-process lock for a read-merge-write of meta.json (the app's rename, a processing job's
    facts), so neither loses the other's fields. The live recorder never shares meta.json: renames
    are refused while it runs."""
    from lecture_recorder.recorder.platform import ProcessLock

    if not Path(session_dir).is_dir():  # never let the lock file recreate a deleted lecture's class folder
        raise FileNotFoundError(f"no session folder {session_dir}")
    lock = ProcessLock(Path(session_dir).parent / META_LOCK)
    deadline = time.monotonic() + timeout_s
    while not lock.acquire():
        if time.monotonic() >= deadline:
            raise MetaUnreadable(f"meta.json in {session_dir} stayed busy for {timeout_s:.0f} s; it was left unchanged")
        time.sleep(0.02)
    try:
        yield
    finally:
        lock.release()


def update_meta(session_dir: Path, _drop: tuple[str, ...] = (), **fields) -> dict:
    """Merge `fields` into meta.json under `meta_lock`, removing the keys in `_drop`. A missing file is created;
    an existing one that stays locked or invalid for 10 s raises MetaUnreadable instead of being overwritten
    (title, class, events...)."""
    path = Path(session_dir) / META_FILE
    with meta_lock(session_dir):
        meta = read_json(path, None, wait_s=10)
        if meta is None and path.exists() or meta is not None and not isinstance(meta, dict):
            raise MetaUnreadable(f"{path} could not be read; it was left unchanged")
        meta = {k: v for k, v in {**(meta or {}), **fields}.items() if k not in _drop}
        write_json_atomic(path, meta)
    return meta


def read_json(path: Path, default=None, wait_s: float = 0.4):
    """Parsed JSON, or `default` if the file is missing or still unreadable after `wait_s` (antivirus
    scanners and the search indexer briefly hold freshly written files open without sharing)."""
    deadline = time.monotonic() + wait_s
    while True:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except (PermissionError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                return default
            time.sleep(0.02)
