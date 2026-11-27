"""When a lecture really happened. Worked out when it is read, from meta.json, so lectures imported
before this fix get it too and nothing on disk is rewritten.

An imported file's container creation_time ("file metadata") is often when it was exported or downloaded, not
when the lecture was recorded: a batch of exports carries times seconds apart. So such a time counts only when
no other import of the class has one within BATCH_S, and the date in the file name (if any) agrees with it.
Otherwise the lecture has no known start (no "same time" duplicate, no schedule evidence), and the date shown
is the one from its file name if there is one. When unsure, no time rather than a wrong one.
"""
import re
from datetime import date, datetime
from pathlib import Path

from lecture_recorder.recorder.session import META_FILE, read_json, session_dirs

TRUSTED_DATES = (None, "file metadata", "Zoom folder name")  # None: a recording, stamped when it started
BATCH_S = 600  # imports' file times this close together: exported or downloaded together
_YMD = re.compile(r"(?<!\d)(20\d{2})[-_.](\d{1,2})[-_.](\d{1,2})(?!\d)")
_MDY = re.compile(r"(?<!\d)(\d{1,2})[-_.](\d{1,2})[-_.](\d{4}|\d{2})(?!\d)")
_MD = re.compile(r"(?<![\d._-])(\d{1,2})[-_](\d{1,2})(?![\d._-])")  # no year: "Notes 9-24"


def parse(iso: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(iso) if iso else None
    except (TypeError, ValueError):
        return None


def name_date(original: str | None, latest: date) -> date | None:
    """The date in a file name ("Course 101 9-24-26", "2026-09-24 lecture"), US month-day order, never after
    `latest`. With no year: the most recent such date not after `latest`. None if there is none, or if it is
    impossible or after `latest` (a wrong date is worse than none)."""
    stem = Path(original or "").stem
    if m := _YMD.search(stem):
        y, mo, d = map(int, m.groups())
    elif m := _MDY.search(stem):
        mo, d, y = map(int, m.groups())
        y += 2000 if y < 100 else 0
    elif len(found := _MD.findall(stem)) == 1:
        mo, d = map(int, found[0])
        for y in (latest.year, latest.year - 1):
            try:
                if (day := date(y, mo, d)) <= latest:
                    return day
            except ValueError:
                return None
        return None
    else:
        return None
    try:
        day = date(y, mo, d)
    except ValueError:
        return None
    return day if day <= latest else None


def trusted_time(date_from: str | None, source_kind: str | None, created: str | None) -> datetime | None:
    """The start stored in one meta.json, if its source can say when the recording happened (see lecture_times
    for imports, which also need the rest of their class)."""
    if date_from not in TRUSTED_DATES:
        return None
    if date_from is None and source_kind == "import":  # imported before date_from existed: the import time
        return None
    return parse(created)


def lecture_times(metas: dict[str, dict]) -> dict[str, tuple[datetime | None, str | None]]:
    """One class's lectures {folder name: meta.json} -> {name: (start, shown)}: `start` when it really started,
    None if unknown; `shown` the ISO date-time to show (a date only, when it comes from the file name)."""
    def imported_by_file_time(meta: dict) -> bool:
        return (meta.get("source") or {}).get("kind") == "import" and meta.get("date_from") == "file metadata"

    file_times = {n: t for n, m in metas.items() if imported_by_file_time(m) and (t := parse(m.get("created")))}
    out = {}
    for n, meta in metas.items():
        src = meta.get("source") or {}
        start = trusted_time(meta.get("date_from"), src.get("kind"), meta.get("created"))
        named = None
        if src.get("kind") == "import" and meta.get("date_from") != "Zoom folder name":
            stamp = parse(meta.get("created")) or parse(meta.get("imported")) or datetime.now().astimezone()
            named = name_date(src.get("file"), stamp.date())
            if start is not None and (named not in (None, start.date()) or any(
                    abs((start - t).total_seconds()) < BATCH_S for o, t in file_times.items() if o != n)):
                start = None
        out[n] = (start, meta.get("created") if start is not None or named is None else named.isoformat())
    return out


def class_times(class_dir: Path) -> dict[str, tuple[datetime | None, str | None]]:
    """lecture_times for every lecture folder of a class."""
    return lecture_times({d.name: read_json(d / META_FILE, {}) or {} for d in session_dirs(Path(class_dir))})
