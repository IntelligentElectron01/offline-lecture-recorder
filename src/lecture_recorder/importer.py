"""Turn an imported file (already copied into the class folder as a temp file) into a session.

Shared by the app's upload endpoints and `lecture-recorder import-zoom`. The user's original files are
never moved or changed: the app uploads a copy, and the CLI copies before importing.
"""
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from lecture_recorder.naming import MAX_TITLE, title_for
from lecture_recorder.pipeline.audio import audio_seconds, extract_audio, recorded_at
from lecture_recorder.recorder.session import META_FILE, new_session_dir, now_iso, sessions_root, write_json_atomic

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".opus", ".ogg", ".flac"}
VIDEO_EXTS = {".mp4"}  # only the audio track is kept
IMPORT_EXTS = AUDIO_EXTS | VIDEO_EXTS
MAX_IMPORT_BYTES = 4 * 2**30


class ImportFailed(ValueError):
    pass


def temp_path(data_dir: Path, class_slug: str, suffix: str) -> Path:
    """A fresh temp file name next to the class's sessions (same disk: the final move is a rename)."""
    root = sessions_root(data_dir, class_slug)
    try:
        root.mkdir(exist_ok=True)  # never recreates a class folder that was just deleted
    except FileNotFoundError:
        raise ImportFailed("Choose an existing class.")
    return root / f".import-{uuid.uuid4().hex}{suffix}.part"


def copy_to_temp(src: Path, data_dir: Path, class_slug: str) -> Path:
    tmp = temp_path(data_dir, class_slug, src.suffix.lower())
    try:
        shutil.copyfile(src, tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return tmp


def create_session(data_dir: Path, class_slug: str, tmp: Path, original: str, *, size: int,
                   when: datetime | None = None, date_from: str | None = None, last_modified_ms: int | None = None,
                   title: str | None = None, source: dict | None = None, extras: dict[str, Path] | None = None,
                   keep_source: bool = False, template: str = "") -> Path:
    """New session from `tmp` (audio or mp4) plus `extras` {file name in the session: temp file}. All
    temp files are consumed: moved in, or deleted on failure (with the session folder this created).
    `keep_source=True`: `tmp` is the user's own mp4, only read for its audio and never deleted.
    Title: the naming `template` if set (naming.py), else `title`, else the file name."""
    extras = extras or {}
    ext = Path(original).suffix.lower()
    created: list[Path] = []
    d = None
    try:
        if ext not in (VIDEO_EXTS if keep_source else IMPORT_EXTS):
            raise ImportFailed(f"Unsupported file type {ext or '(none)'}; use wav, mp3, m4a, opus, ogg, flac or mp4.")
        if when is None:
            when, date_from = recorded_at(tmp), "file metadata"
        if when is None:
            if last_modified_ms:
                when, date_from = datetime.fromtimestamp(last_modified_ms / 1000).astimezone(), "file date"
            else:
                when, date_from = datetime.now().astimezone(), "import time"
        d = new_session_dir(data_dir, class_slug, now=when.replace(tzinfo=None))
        if ext in VIDEO_EXTS:
            try:
                created.append(extract_audio(tmp, d))
            except ValueError as e:
                raise ImportFailed(str(e))
            except Exception as e:  # PyAV/FFmpeg errors on a damaged or unsupported file
                raise ImportFailed(f"Could not read the audio of {original}: {e}")
            if not keep_source:
                tmp.unlink()
        else:
            os.replace(tmp, d / f"audio{ext}")
            created.append(d / f"audio{ext}")
        for name, path in extras.items():
            os.replace(path, d / name)
            created.append(d / name)
    except BaseException:
        for p in ((() if keep_source else (tmp,)) + (*extras.values(), *created)):
            p.unlink(missing_ok=True)  # only files this import wrote
        if d is not None:
            try:
                d.rmdir()  # the folder this import created, if nothing else is in it
            except OSError:
                pass
        raise
    title = title_for(template, d, original) or (title or Path(original).stem)[:MAX_TITLE] or d.name
    seconds = audio_seconds(created[0])  # known before processing: the list shows it, the duplicate check needs it
    write_json_atomic(d / META_FILE, {"name": d.name, "title": title,
                                      "class": class_slug, "created": when.isoformat(timespec="seconds"),
                                      "date_from": date_from, "imported": now_iso(),
                                      "source": {"kind": "import", "file": original[:255], "bytes": size, **(source or {})},
                                      "state": "stopped", **({"duration_s": seconds} if seconds else {})})
    return d
