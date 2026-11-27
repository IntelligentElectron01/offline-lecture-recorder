"""Possible duplicate lectures: CPU only, no model calls, and nothing is ever deleted here.

Signals, strongest first:
- "file": the same source file (meta.json source.sha256; older imports have no hash, so the
  same file name and size counts). Any class.
- "time": both recording times overlap by at least OVERLAP of the shorter one. Only when at least one of them
  was recorded live by this app (mic or app capture), only for times that say when the recording really
  happened (dates.lecture_times: an import's file time can be its export time), and only within a class plus
  Unsorted. Two imports are linked by file or content only.
- "content": MinHash of the transcript's word shingles (content.sketch.json in the session folder,
  written at the end of transcription), same scope as "time".

Pairs are worked out when a page asks (no stored results, so a delete, restore or rename can't leave a
stale pair); the sketches make that cheap. "Keep both" is the only state: meta.json dismissed_duplicates
of the lecture it was pressed on. Nothing here may ever fail or block a processing job.
"""
import hashlib
import logging
import zlib
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from lecture_recorder import library as lib
from lecture_recorder.dates import lecture_times
from lecture_recorder.pipeline.outputs import EDITABLE
from lecture_recorder.recorder.session import (
    META_FILE, MetaUnreadable, iter_session_dirs, meta_lock, read_json, write_json_atomic,
)
from lecture_recorder.search.text import tokens

log = logging.getLogger(__name__)

OVERLAP = 0.5
LIVE = frozenset({"mic", "app"})  # recorded by this app: its start time is when the lecture happened
SKETCH_FILE = "content.sketch.json"
SHINGLE_WORDS = 4
MIN_WORDS = 300  # shorter transcripts have too few shingles to compare
# Top terms (for class suggestions): lectures of a course share terms, not sentences, so the shingles above
# can't compare them. A course's terms come up again and again; everyday spoken words say nothing about it.
TOP_TERMS = 300
TERM_MIN_LETTERS = 4
TERM_SKIP = frozenset("""about actually after again also always another because been before being between both
    could does doing done down each even every first from going good have here into just kind know last like look
    make many maybe mean might more most much must need next okay only other over people probably question really
    right said same should show some something sort still such take talk than that their them then there these
    they thing things think this those three through time today very want well were what when where which while
    will with work would yeah year years your""".split())
PERMS = 256
SIMILAR = 0.12  # measured in scripts/bench/measure_duplicates.py
SIGNALS = {"file": "same source file", "time": "recorded at the same time", "content": "same content"}

_SEEDS = np.random.default_rng(1729).integers(1, 2**63, size=(PERMS, 2), dtype=np.uint64)
_A, _B = _SEEDS[:, 0] | np.uint64(1), _SEEDS[:, 1]  # multiply-shift hashes: `a` must be odd


# ------------------------------------------------------------------ the transcript's fingerprint
def minhash(words: list[str]) -> list[int] | None:
    """PERMS numbers whose share of equal positions between two texts estimates their shingle Jaccard."""
    if len(words) < MIN_WORDS:
        return None
    grams = {" ".join(words[i:i + SHINGLE_WORDS]) for i in range(len(words) - SHINGLE_WORDS + 1)}
    x = np.fromiter((zlib.crc32(g.encode("utf-8")) for g in grams), np.uint64, len(grams))
    return [int(((a * x + b) >> np.uint64(32)).min()) for a, b in zip(_A, _B)]


def top_terms(words: list[str]) -> list[str] | None:
    """The words the lecture repeats most, for comparing it with a class (autosort.py)."""
    if len(words) < MIN_WORDS:
        return None
    counts = Counter(w for w in words if len(w) >= TERM_MIN_LETTERS and not w.isdigit() and w not in TERM_SKIP)
    return [w for w, _ in counts.most_common(TOP_TERMS)]


def sketch_session(d: Path) -> None:
    """(Re)write the session's sketch if its transcript is newer. Sessions without one are skipped."""
    src, out = d / EDITABLE, d / SKETCH_FILE
    try:
        stamp = src.stat().st_mtime_ns
    except OSError:
        return
    try:
        # The file is small, and this runs once at the end of a job, so an older sketch can be spotted.
        if out.stat().st_mtime_ns >= stamp and "terms" in (read_json(out, {}) or {}):
            return
    except OSError:
        pass
    doc = read_json(src, None)
    if not isinstance(doc, dict):  # e.g. a scanner still holds the new file: no sketch, so the next sweep retries
        raise ValueError(f"could not read {src}")
    words = [w for seg in doc.get("segments") or [] for w in tokens(seg.get("text") or "")]
    write_json_atomic(out, {"minhash": minhash(words), "terms": top_terms(words)}, fsync=False)


def sketch_scope(data_dir: Path, slug: str) -> None:
    """Sketch every lecture `slug` is compared with; older ones, or ones edited since, catch up."""
    for d in iter_session_dirs(Path(data_dir)):
        if _in_scope(slug, d.parent.parent.name):
            try:
                sketch_session(d)
            except Exception:  # one bad file costs one lecture's content check, nothing else
                log.exception("could not sketch %s", d)


def after_transcribe(data_dir: Path, session_dir: Path) -> None:
    """The tiny CPU step at the end of a transcription job. Never raises."""
    try:
        sketch_scope(data_dir, Path(session_dir).parent.parent.name)
    except Exception:
        log.exception("duplicate check skipped for %s", session_dir)


# ------------------------------------------------------------------ the lectures to compare
def _source_label(src: dict) -> str:
    if src.get("kind") == "import":
        return "Zoom" if src.get("zoom") else "Import"
    if src.get("kind") == "app":
        return src.get("app_name") or "App"
    return "Microphone" if src.get("kind") == "mic" else (src.get("kind") or "Unknown")


def _add_times(rows: list[dict]) -> None:
    """Each row's shown date and (start, end) in seconds since the epoch if its start is known, worked out per
    class (an import's file time is judged against the class's other imports)."""
    for cls in {r["class"] for r in rows}:
        mine = [r for r in rows if r["class"] == cls]
        times = lecture_times({r["name"]: r.pop("meta") for r in mine})
        for r in mine:
            start, r["created"] = times[r["name"]]
            r["span"] = (start.timestamp(), start.timestamp() + float(r["duration_s"])) if start and r["duration_s"] else None


def _in_scope(slug: str, cls: str) -> bool:
    """Is a lecture in class `cls` compared by time and content with those in `slug`?"""
    return slug == lib.UNSORTED or cls in (slug, lib.UNSORTED)  # a lecture without a class yet: compared with all


def _row(d: Path, states: dict, meta: dict) -> dict | None:
    row = lib._summary(d, d.parent.parent.name, states.get(str(d.resolve())), meta)
    if row["state"] in ("recording", "unfinished"):
        return None
    src = meta.get("source") or {}
    row.update(dir=d, meta=meta, sha=src.get("sha256"), file=src.get("file"), bytes=src.get("bytes"),
               source_label=_source_label(src), dismissed=meta.get("dismissed_duplicates") or [],
               processed=(d / "transcript.raw.json").exists())
    return row


def _rows(data_dir: Path, slug: str | None = None) -> list[dict]:
    """The lectures `slug` is compared with (None: all). Other classes matter only for the same source file,
    so a class with no imported lecture never reads them, and of the rest only a matching file is kept."""
    states = lib._job_states(data_dir)
    rows, others = [], []
    for d in iter_session_dirs(Path(data_dir)):
        if slug is not None and not _in_scope(slug, d.parent.parent.name):
            others.append(d)
        elif row := _row(d, states, read_json(d / META_FILE, {}) or {}):
            rows.append(row)
    shas = {r["sha"] for r in rows if r["class"] == slug and r["sha"]}
    files = {(r["file"], r["bytes"]) for r in rows if r["class"] == slug and r["file"]}
    if shas or files:
        for d in others:
            meta = read_json(d / META_FILE, {}) or {}
            src = meta.get("source") or {}
            if (src.get("sha256") in shas or (src.get("file"), src.get("bytes")) in files) and (row := _row(d, states, meta)):
                rows.append(row)
    _add_times(rows)  # complete for the compared classes; another class's rows are compared by file only
    return rows


def _key(row: dict) -> str:
    return f"{row['class']}/{row['name']}"


def _overlaps(a: dict, b: dict) -> bool:
    if not (a["span"] and b["span"]) or not LIVE & {a["source"], b["source"]}:
        return False
    shared = min(a["span"][1], b["span"][1]) - max(a["span"][0], b["span"][0])
    return shared > 0 and shared >= OVERLAP * min(a["span"][1] - a["span"][0], b["span"][1] - b["span"][0])


def _same_file(a: dict, b: dict) -> bool:
    if a["sha"] and b["sha"]:
        return a["sha"] == b["sha"]
    return a["file"] is not None and a["bytes"] is not None and (a["file"], a["bytes"]) == (b["file"], b["bytes"])


def _sketches(rows: list[dict]) -> tuple[list[dict], np.ndarray | None]:
    have, vectors = [], []
    for r in rows:
        m = (read_json(r["dir"] / SKETCH_FILE, {}) or {}).get("minhash") if r["processed"] else None
        if isinstance(m, list) and len(m) == PERMS:
            have.append(r)
            vectors.append(m)
    return have, np.array(vectors, np.uint64) if vectors else None


def _pairs(rows: list[dict], slug: str, wanted) -> list[tuple[dict, dict, list[str]]]:
    """The pairs with a `wanted` lecture, each with its signals in SIGNALS order. Dismissed ones are left out."""
    mine = [r for r in rows if wanted(r)]
    if not mine:
        return []
    found: dict[tuple[str, str], tuple[dict, dict, list[str]]] = {}

    def add(a: dict, b: dict, signal: str) -> None:
        found.setdefault(tuple(sorted((_key(a), _key(b)))), (a, b, []))[2].append(signal)

    for a, b in combinations(rows, 2):
        if (wanted(a) or wanted(b)) and _same_file(a, b):
            add(a, b, "file")
    scope = [r for r in rows if _in_scope(slug, r["class"])]
    for a, b in combinations(scope, 2):
        if (wanted(a) or wanted(b)) and _overlaps(a, b):
            add(a, b, "time")
    if any(r["processed"] for r in mine):
        have, m = _sketches(scope)
        for i, j in combinations(range(len(have)), 2):
            if (wanted(have[i]) or wanted(have[j])) and float((m[i] == m[j]).mean()) >= SIMILAR:
                add(have[i], have[j], "content")
    return [(a, b, signals) for a, b, signals in found.values() if _key(b) not in a["dismissed"] and _key(a) not in b["dismissed"]]


def _move_first(a: dict, b: dict) -> dict:
    """Which of the two to suggest moving: the unprocessed one, else the shorter one, else the newer one."""
    if a["processed"] != b["processed"]:
        return b if a["processed"] else a
    da, db = a["duration_s"] or 0, b["duration_s"] or 0
    if da != db:
        return a if da < db else b
    return a if a["name"] > b["name"] else b


def _side(r: dict) -> dict:
    return {"class": r["class"], "session": r["name"], "title": r["title"], "created": r["created"],
            "duration_s": r["duration_s"], "source": r["source_label"], "processed": r["processed"]}


def _views(rows: list[dict], slug: str, wanted) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for a, b, signals in _pairs(rows, slug, wanted):
        move = _move_first(a, b)
        for this, other in ((a, b), (b, a)):
            if wanted(this):
                out.setdefault(this["name"], []).append({"this": _side(this), "other": _side(other), "signals": signals,
                                                         "labels": [SIGNALS[s] for s in signals], "move": "this" if move is this else "other"})
    return out


# ------------------------------------------------------------------ what the app calls
def for_class(data_dir: Path, slug: str, name: str | None = None) -> dict[str, list[dict]]:
    """{lecture folder name: its possible duplicates}, for one lecture if `name` is given. A problem here
    hides the hint, never the page."""
    wanted = lambda r: r["class"] == slug and name in (None, r["name"])  # noqa: E731
    try:
        return _views(_rows(data_dir, slug), slug, wanted)
    except Exception:
        log.exception("duplicate check skipped for class %s", slug)
        return {}


def find_source(data_dir: Path, sha256: str, original: str, size: int) -> dict | None:
    """The lecture already imported from this exact file, if any (before an import starts)."""
    new = {"sha": sha256, "file": original[:255], "bytes": size}
    try:
        return next((_side(r) for r in _rows(data_dir) if _same_file(new, r)), None)
    except Exception:
        log.exception("could not check for an earlier import")
        return None


def dismiss(data_dir: Path, slug: str, name: str, other_class: str, other_name: str) -> None:
    """Keep both, for good. Raises LibraryError for a bad name, FileNotFoundError for a missing lecture."""
    d, other = lib.session_dir(data_dir, slug, name), lib.session_dir(data_dir, other_class, other_name)
    if not d.is_dir() or not other.is_dir():
        raise FileNotFoundError("No such lecture.")
    key = f"{other_class}/{other_name}"
    _edit_kept(d, lambda kept: kept if key in kept else [*kept, key])


def _edit_kept(d: Path, change) -> None:
    with meta_lock(d):  # two quick Keep both clicks must both stay
        meta = read_json(d / META_FILE, None, wait_s=10)
        if not isinstance(meta, dict):
            raise MetaUnreadable(f"{d / META_FILE} could not be read; it was left unchanged")
        kept = meta.get("dismissed_duplicates") or []
        if (new := change(kept)) != kept:
            write_json_atomic(d / META_FILE, {**meta, "dismissed_duplicates": new})


def repoint(data_dir: Path, old_key: str, new_key: str) -> None:
    """A lecture moved: Keep both pairs stored on other lectures follow it. Pairs held by lectures in
    Recently deleted stay as they are (a restore is byte-identical): at worst a pair is asked about again."""
    for d in iter_session_dirs(Path(data_dir)):
        if old_key in ((read_json(d / META_FILE, {}) or {}).get("dismissed_duplicates") or []):
            _edit_kept(d, lambda kept: [new_key if k == old_key else k for k in kept])


def file_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
