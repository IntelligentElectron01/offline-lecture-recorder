"""Classes and sessions on disk. The folders are the source of truth; nothing here deletes data.

data/classes/<slug>/class.json                {name, slug, created, max_length_minutes}
data/classes/<slug>/sessions/<date-time>/     one folder per recording or import
"""
import re
from pathlib import Path

from lecture_recorder.dates import class_times, lecture_times
from lecture_recorder.pipeline.audio import AUDIO_NAMES
from lecture_recorder.pipeline.queue import SUMMARIZE, TRANSCRIBE, jobs, kind_of
from lecture_recorder.recorder.platform import is_locked
from lecture_recorder.recorder.session import (
    BOOKMARKS_FILE, CLASS_SLUG_RE, LOCK_FILE, META_FILE, PCM_FILE, SESSION_NAME_RE, now_iso, read_json,
    write_json_atomic,
)
from lecture_recorder.zoom import CHAT_FILE as ZOOM_CHAT, TRANSCRIPT_FILE as ZOOM_TRANSCRIPT
from lecture_recorder.summary.session import (  # noqa: F401  (PLACEHOLDER_MARK re-exported)
    PLACEHOLDER_MARK, SUMMARY_FILE, SUMMARY_META, placeholder_text, summary_kind,
)

CLASS_FILE = "class.json"
UNSORTED = "unsorted"  # the real class behind "Not sure yet"
# Files the app may serve from a session folder (nothing else, ever).
SERVABLE = (*AUDIO_NAMES, "transcript.md", "transcript.txt", "transcript.json", "transcript.raw.json",
            SUMMARY_FILE, META_FILE, BOOKMARKS_FILE, ZOOM_TRANSCRIPT, ZOOM_CHAT)


class LibraryError(ValueError):
    pass


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:64].strip("-")
    if not slug or not CLASS_SLUG_RE.match(slug):
        raise LibraryError("A class name needs at least one letter or digit.")
    return slug


def classes_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "classes"


def class_dir(data_dir: Path, slug: str) -> Path:
    if not CLASS_SLUG_RE.match(slug or ""):
        raise LibraryError(f"invalid class {slug!r}")
    return classes_dir(data_dir) / slug


def session_dir(data_dir: Path, slug: str, name: str) -> Path:
    if not SESSION_NAME_RE.match(name or ""):
        raise LibraryError(f"invalid session {name!r}")
    return class_dir(data_dir, slug) / "sessions" / name


def get_class(data_dir: Path, slug: str) -> dict | None:
    d = class_dir(data_dir, slug)
    if not d.is_dir():
        return None
    info = read_json(d / CLASS_FILE, {}) or {}
    return {"slug": slug, "name": info.get("name") or slug, "created": info.get("created"),
            "max_length_minutes": info.get("max_length_minutes"), "has_class_file": bool(info)}


def list_classes(data_dir: Path) -> list[dict]:
    root = classes_dir(data_dir)
    out = []
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if d.is_dir() and CLASS_SLUG_RE.match(d.name):
                c = get_class(data_dir, d.name)
                sessions = d / "sessions"
                c["sessions"] = sum(1 for s in sessions.iterdir() if s.is_dir() and SESSION_NAME_RE.match(s.name)) \
                    if sessions.is_dir() else 0
                out.append(c)
    return sorted(out, key=lambda c: c["name"].lower())


def create_class(data_dir: Path, name: str, max_length_minutes: float | None = None) -> dict:
    name = (name or "").strip()
    if not name or len(name) > 100:
        raise LibraryError("Class name must be 1-100 characters.")
    if max_length_minutes is not None and not (1 <= float(max_length_minutes) <= 600):
        raise LibraryError("Auto-stop length must be between 1 and 600 minutes.")
    slug = slugify(name)
    d = class_dir(data_dir, slug)
    if (d / CLASS_FILE).exists():
        raise LibraryError(f"A class named {name!r} already exists.")
    d.mkdir(parents=True, exist_ok=True)
    write_json_atomic(d / CLASS_FILE, {"name": name, "slug": slug, "created": now_iso(),
                                       "max_length_minutes": max_length_minutes})
    return get_class(data_dir, slug)


def ensure_unsorted(data_dir: Path) -> dict:
    """The Unsorted class, created on first use."""
    c = get_class(data_dir, UNSORTED)
    if c and c["has_class_file"]:
        return c
    try:
        return create_class(data_dir, "Unsorted")
    except LibraryError:  # another request made it just now
        return get_class(data_dir, UNSORTED)


def real_class(data_dir: Path, slug: str) -> dict | None:
    """The class to file something into, if `slug` is one with a class.json. Unsorted appears on first use."""
    if not CLASS_SLUG_RE.fullmatch(slug or ""):
        return None
    if slug == UNSORTED:
        ensure_unsorted(data_dir)
    c = get_class(data_dir, slug)
    return c if c and c["has_class_file"] else None


def register_class(data_dir: Path, slug: str) -> dict:
    """Give a class folder that has no class.json (e.g. made by an older terminal recording) its settings
    file, named after the folder, so it becomes a normal class. Its lectures are untouched."""
    d = class_dir(data_dir, slug)
    if not d.is_dir():
        raise LibraryError("No such class folder.")
    if (d / CLASS_FILE).exists():
        raise LibraryError("This class already has its settings.")
    write_json_atomic(d / CLASS_FILE, {"name": slug, "slug": slug, "created": now_iso(), "max_length_minutes": None})
    return get_class(data_dir, slug)


def rename_session(data_dir: Path, slug: str, name: str, title) -> str:
    """Change only meta.json `title`: the folder, `source.file` and every other field stay as they
    are. Refused while the lecture is recording or its recording isn't saved yet (the recorder keeps
    its own copy of meta.json until then)."""
    from lecture_recorder.naming import clean_title
    from lecture_recorder.recorder.session import update_meta
    from lecture_recorder.trash import started_recently

    title = clean_title(title)  # NamingError (a ValueError) explains what to fix
    d = session_dir(data_dir, slug, name)
    if not d.is_dir():
        raise FileNotFoundError("No such lecture.")
    meta = read_json(d / META_FILE, {}) or {}
    live = session_state(d, None, meta) in ("recording", "unfinished")
    if live or meta.get("state") == "starting" and started_recently(meta.get("created")):
        raise LibraryError("This lecture is still recording. Rename it after you stop the recording.")
    update_meta(d, title=title)
    return title


def has_audio(d: Path) -> bool:
    return any((d / n).is_file() for n in AUDIO_NAMES)


def transcribe_anyway(d: Path) -> None:
    """The user's choice for a file flagged "no usable audio": the next transcription skips the refusal."""
    from lecture_recorder.recorder.session import update_meta

    update_meta(d, audio_check_override=True)


def unprocessed_sessions(data_dir: Path) -> list[Path]:
    """Session folders with audio that were never processed and have no active or failed job."""
    from lecture_recorder.recorder.session import iter_session_dirs

    states = _job_states(data_dir)
    return [d for d in iter_session_dirs(Path(data_dir))
            if has_audio(d) and session_state(d, states.get(str(d.resolve()))) == "not processed"]


def _job_states(data_dir: Path, kind: str = TRANSCRIBE, all_jobs: list[dict] | None = None) -> dict[str, dict]:
    latest = {}
    for j in jobs(data_dir) if all_jobs is None else all_jobs:  # oldest first: the newest per session wins
        if kind_of(j) == kind:
            latest[j["session"]] = j
    return latest


def session_state(d: Path, job: dict | None, meta: dict | None = None) -> str:
    if is_locked(d / LOCK_FILE):
        return "recording"
    if (d / PCM_FILE).exists():
        return "unfinished"
    if job and job["state"] in ("queued", "running"):
        return job["state"]
    if (d / "transcript.raw.json").exists():
        return "processed"
    if meta is None:
        meta = read_json(d / META_FILE, {}) or {}
    if (meta.get("audio_check") or {}).get("usable") is False and not meta.get("audio_check_override"):
        return "no audio"  # until the user chooses "Transcribe anyway"; after that, a real failure shows as failed
    if job and job["state"] == "failed":
        return "failed"
    return "not processed"


def _summary(d: Path, slug: str, job: dict | None, meta: dict | None = None, shown: str | None = None) -> dict:
    """`shown`: the date to show (dates.lecture_times), if the caller worked it out; else meta.json's."""
    meta = read_json(d / META_FILE, {}) or {} if meta is None else meta
    duration = meta.get("recorded_s") or meta.get("duration_s")  # imports: copied there after processing
    if duration is None and (d / "transcript.raw.json").exists():  # sessions processed before that
        duration = (read_json(d / "transcript.raw.json", {}) or {}).get("duration_s")
    source = meta.get("source") or {}
    return {"class": slug, "name": d.name, "title": meta.get("title") or d.name, "created": shown or meta.get("created"),
            "duration_s": duration, "source": source.get("kind"), "original": source.get("file"),
            "state": session_state(d, job, meta)}


def by_date(s: dict):
    """Sort key: the date shown (an import's may come from its file name), then the folder's date-time."""
    return (s["created"] or s["name"])[:10], s["name"]


def list_sessions(data_dir: Path, slug: str, states: dict | None = None) -> list[dict]:
    root = class_dir(data_dir, slug) / "sessions"
    if not root.is_dir():
        return []
    states = _job_states(data_dir) if states is None else states
    metas = {d: read_json(d / META_FILE, {}) or {} for d in root.iterdir() if d.is_dir() and SESSION_NAME_RE.match(d.name)}
    times = lecture_times({d.name: m for d, m in metas.items()})
    out = [_summary(d, slug, states.get(str(d.resolve())), m, times[d.name][1]) for d, m in metas.items()]
    return sorted(out, key=by_date, reverse=True)  # newest first


def session_detail(data_dir: Path, slug: str, name: str) -> dict | None:
    d = session_dir(data_dir, slug, name)
    if not d.is_dir():
        return None
    all_jobs = jobs(data_dir)
    key = str(d.resolve())
    job = _job_states(data_dir, TRANSCRIBE, all_jobs).get(key)
    info = _summary(d, slug, job, shown=class_times(class_dir(data_dir, slug)).get(name, (None, None))[1])
    meta = read_json(d / META_FILE, {}) or {}
    dropped = meta.get("dropped_speech")
    if (dropped is None or info["duration_s"] is None) and (d / "transcript.raw.json").exists():
        raw = read_json(d / "transcript.raw.json", {}) or {}  # sessions processed before this was in meta.json
        dropped = dropped if dropped is not None else (raw.get("processing") or {}).get("dropped_speech")
        if info["duration_s"] is None:
            info["duration_s"] = raw.get("duration_s")
    audio = next((n for n in AUDIO_NAMES if (d / n).is_file()), None)
    info.update(
        folder=str(d),
        audio=audio,
        files=[n for n in SERVABLE if (d / n).is_file()],
        bookmarks=read_json(d / BOOKMARKS_FILE, []) or [],
        meta=meta,
        dropped_speech=dropped,
        job={k: job.get(k) for k in ("state", "created", "started", "finished", "error")} if job else None,
        retention=read_json(d / "retention.json", None),
        summary=summary_info(d, _job_states(data_dir, SUMMARIZE, all_jobs).get(key)),
    )
    return info


def zoom_speaker_hints(d: Path, doc: dict) -> list[dict]:
    """Zoom's transcript names matched to the speakers of `doc` (the lecture's transcript.json); a hint,
    labels are not changed. Called with the transcript the page loads anyway, not on status polls."""
    vtt = d / ZOOM_TRANSCRIPT
    if not vtt.is_file():
        return []
    from lecture_recorder.zoom import parse_vtt, speaker_names

    try:
        cues = parse_vtt(vtt.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return []
    return speaker_names(cues, doc.get("segments") or [])


def summary_info(d: Path, job: dict | None) -> dict:
    """What the session page shows about the summary: the file's kind, plus the job state if one is
    active, failed, or skipped. Sizes and error messages only, never transcript text."""
    kind = summary_kind(d)
    meta = read_json(d / SUMMARY_META, None) or {}
    state = kind
    if job and job["state"] in ("queued", "running"):
        state = job["state"]
    elif kind in ("none", "placeholder") and (meta.get("state") == "failed" or (job and job["state"] == "failed")):
        state = "failed"
    error = None
    if state == "failed" and meta.get("state") == "failed":
        error = meta.get("error")
    elif state == "failed":  # failed before writing its status file: last line of the job's output
        tail = ((job or {}).get("error") or "").strip().splitlines()
        error = tail[-1] if tail else "the summary job failed"
    return {"kind": kind, "state": state, "warning": meta.get("warning") if meta.get("state") == "skipped" else None,
            "error": error,
            "generated": meta.get("finished") if meta.get("state") == "done" else None, "model": meta.get("model")}


def write_summary_placeholder(d: Path) -> bool:
    """Create summary.md until the summary job replaces it. Never replaces an existing file."""
    p = d / SUMMARY_FILE
    if p.exists():
        return False
    p.write_text(placeholder_text(d.name), encoding="utf-8")
    return True
