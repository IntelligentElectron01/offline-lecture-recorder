"""Which class a lecture seems to belong to. Only a hint: nothing here moves or files anything, a failure
costs the hint and never a page or a job, and it stays quick. CPU only, no model calls.

Each class gets one score from three signals; a suggestion needs enough evidence (MIN_SCORE) and a clear lead
over the second class (MARGIN), otherwise nothing is shown.
- Schedule: same weekday and start within SLOT_MIN of at least MIN_LECTURES of the class's lectures in the
  same semester. Only real recording times count (dates.lecture_times: not an import's export time).
- Zoom title: the imported meeting's topic against the class name and its earlier Zoom topics.
- Content: the lecture's top terms (duplicates.top_terms) against each class's recent lectures, each term
  weighted by how few classes use it, so words common to every class count for nothing. It needs at least two
  classes to compare and one with a clear lead.

Where a lecture already is in a real class, only content can suggest moving it. "Keep here" and any move set
meta.json class_settled, so that lecture is never asked about again. Thresholds were measured on synthetic text only.
"""
import logging
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from lecture_recorder import duplicates as dup
from lecture_recorder import library as lib
from lecture_recorder.dates import class_times, lecture_times
from lecture_recorder.recorder.session import META_FILE, read_json, session_dirs, update_meta
from lecture_recorder.search.text import tokens
from lecture_recorder.zoom import _CLOUD_NAME, folder_date_title

log = logging.getLogger(__name__)

SLOT_MIN = 30
MIN_LECTURES = 2
SEMESTER_DAYS = 150  # last term's class in the same room and hour must not win
SCHEDULE_SCORE, SCHEDULE_FULL = 0.6, 3  # the full score from this many matching lectures on
TITLE_SCORE = 0.7
CONTENT_SCORE = 0.6
MIN_SCORE, MARGIN = 0.4, 0.2
CONTENT_SAMPLE = 6  # compare with each class's newest lectures
CONTENT_MIN, CONTENT_LEAD = 0.15, 1.5  # the top class must score this many times the second, as a lecture on two
# classes' subjects scores high on both. Measured in scripts/bench/measure_autosort.py
TITLE_NOISE = frozenset("""a an and class for in lecture lectures meeting my of on personal recorded recording room
    s section session the to week zoom""".split())
QUIET_STATES = ("processed", "not processed", "failed", "no audio")  # not while it records or a job has it
_NAMED_TIME = re.compile(r"(?<!\d)(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})[ _T-]+(\d{2})[-_.:]?(\d{2})(?:[-_.:]?\d{2})?(?!\d)")
WEEKDAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()


# ------------------------------------------------------------------ what is known about each class
def _title_words(text: str | None) -> set[str]:
    return {t for t in tokens(text or "") if t not in TITLE_NOISE}


def _facts(d: Path, meta: dict, start: datetime | None) -> dict:
    src = meta.get("source") or {}
    topic = folder_date_title(src.get("folder") or "", src.get("file") or "")[0] if src.get("zoom") else None
    return {"name": d.name, "dir": d, "start": start, "topic": _title_words(topic)}


def _classes(data_dir: Path) -> dict[str, dict]:
    """Every real class (not Unsorted), and its lecture folders. What's in them is read when it is needed."""
    return {c["slug"]: {"name": c["name"], "words": _title_words(c["name"]),
                        "dirs": session_dirs(lib.class_dir(data_dir, c["slug"]))}
            for c in lib.list_classes(data_dir) if c["has_class_file"] and c["slug"] != lib.UNSORTED}


def _lectures(c: dict) -> list[dict]:
    if "lectures" not in c:  # one read of each meta.json, by the first signal that needs them
        metas = {d: read_json(d / META_FILE, {}) or {} for d in c["dirs"]}
        times = lecture_times({d.name: m for d, m in metas.items()})
        c["lectures"] = [_facts(d, m, times[d.name][0]) for d, m in metas.items()]
    return c["lectures"]


# ------------------------------------------------------------------ the signals
def _schedule(c: dict, when: datetime, me: str | None) -> tuple[float, str] | None:
    minutes = when.hour * 60 + when.minute
    same_day = [x["start"] for x in _lectures(c) if x["start"] and x["name"] != me and x["start"].weekday() == when.weekday()
                and abs((x["start"].date() - when.date()).days) <= SEMESTER_DAYS]
    fits = sorted(s.hour * 60 + s.minute for s in same_day if abs(s.hour * 60 + s.minute - minutes) <= SLOT_MIN)
    if len(fits) < MIN_LECTURES or len(fits) * 2 < len(same_day):  # a one-off slot is no pattern
        return None
    usual = fits[len(fits) // 2]
    return (SCHEDULE_SCORE * min(len(fits), SCHEDULE_FULL) / SCHEDULE_FULL,
            f"Usually {WEEKDAYS[when.weekday()]}s at {usual // 60:02d}:{usual % 60:02d}")


def titles_match(a: set[str], b: set[str]) -> bool:
    """Most of the shorter title's words are in the other, and not just a number ("Math 101" / "Physics 101")."""
    shared = a & b
    if not any(not w.isdigit() for w in shared) or len(shared) * 4 < 3 * min(len(a), len(b)):
        return False
    numbers = {w for w in a if w.isdigit()}, {w for w in b if w.isdigit()}
    return not (numbers[0] and numbers[1]) or bool(numbers[0] & numbers[1])  # "Physics 101" is not "Physics 202"


def _title(c: dict, words: set[str], me: str | None) -> bool:
    return titles_match(words, c["words"]) or any(titles_match(words, x["topic"]) for x in _lectures(c) if x["name"] != me)


def _terms(d: Path) -> frozenset[str] | None:
    v = (read_json(d / dup.SKETCH_FILE, {}) or {}).get("terms")
    return frozenset(v) if isinstance(v, list) and v else None


def _sample(c: dict) -> list[tuple[str, frozenset[str]]]:
    """The class's newest sketched lectures (one extra, as the lecture being judged may be among them). Read once."""
    if "sample" not in c:
        c["sample"] = []
        for d in reversed(c["dirs"]):
            if t := _terms(d):
                c["sample"].append((d.name, t))
                if len(c["sample"]) > CONTENT_SAMPLE:
                    break
    return c["sample"]


def _rates(c: dict, sample: list[tuple[str, frozenset[str]]]) -> dict[str, float]:
    """How often the sampled lectures use each term. Kept per sample: a page asks about many lectures."""
    tables = c.setdefault("rates", {})
    key = tuple(name for name, _ in sample)
    if key not in tables:
        tables[key] = {t: n / len(sample) for t, n in Counter(w for _, x in sample for w in x).items()}
    return tables[key]


def content_scores(classes: dict[str, dict], terms: frozenset[str], me: tuple[str, str] | None = None) -> dict[str, float]:
    """Per class, 0-1: how much of what is distinctive in `terms` its recent lectures use. Empty unless at
    least two classes have enough lectures to compare."""
    used = {}
    for slug, c in classes.items():
        others = [x for x in _sample(c) if me != (slug, x[0])][:CONTENT_SAMPLE]
        if len(others) >= MIN_LECTURES:
            used[slug] = _rates(c, others)
    if len(used) < 2:
        return {}
    classes_with = Counter(w for words in used.values() for w in words)
    weight = {w: math.log(len(used) / classes_with[w]) for w in terms if classes_with[w]}  # 0 for a word in every class
    total = sum(weight.values())
    return {slug: sum(v * words.get(w, 0.0) for w, v in weight.items()) / total for slug, words in used.items()} if total else {}


def _content(classes: dict[str, dict], terms: frozenset[str], me: tuple[str, str] | None) -> str | None:
    """The class the words clearly point to, if any."""
    ranked = sorted(content_scores(classes, terms, me).items(), key=lambda kv: -kv[1])
    if ranked and ranked[0][1] >= CONTENT_MIN and ranked[0][1] >= CONTENT_LEAD * ranked[1][1]:
        return ranked[0][0]
    return None


# ------------------------------------------------------------------ one suggestion
def suggest(classes: dict[str, dict], when: datetime | None = None, title: set[str] | None = None,
            terms: frozenset[str] | None = None, me: tuple[str, str] | None = None) -> dict | None:
    """The one class with enough evidence and a clear lead, or None."""
    scores: dict[str, float] = {}
    reasons: dict[str, list] = {}

    def add(slug: str, signal: str, points: float, reason: str) -> None:
        scores[slug] = scores.get(slug, 0.0) + points
        reasons.setdefault(slug, []).append((points, signal, reason))

    name = me[1] if me else None
    for slug, c in classes.items():
        mine = name if me and me[0] == slug else None
        if when and (hit := _schedule(c, when, mine)):
            add(slug, "schedule", *hit)
        if title and _title(c, title, mine):
            add(slug, "title", TITLE_SCORE, "Zoom title matches")
    if terms and (slug := _content(classes, terms, me)):
        add(slug, "content", CONTENT_SCORE, "Similar words to its lectures")
    if not scores:
        return None
    ranked = sorted(scores, key=lambda s: -scores[s])
    top = ranked[0]
    lead = scores[top] - (scores[ranked[1]] if len(ranked) > 1 else 0.0)
    if round(scores[top], 6) < MIN_SCORE or round(lead, 6) <= MARGIN:  # rounded: 0.6 * 2 / 3 is 0.39999999999999997
        return None
    return {"class": top, "class_name": classes[top]["name"], "reason": max(reasons[top])[2],
            "signals": [r[1] for r in reasons[top]]}


# ------------------------------------------------------------------ what the app calls
def _quietly(fallback, fn, *args):
    try:
        return fn(*args)
    except Exception:  # a hint is never worth an error page or a failed job
        log.exception("class suggestion skipped")
        return fallback


def _for_class(data_dir: Path, slug: str, name: str | None, states: dict | None) -> dict[str, dict]:
    classes, times, out = None, None, {}
    unsorted = slug == lib.UNSORTED
    for d in session_dirs(lib.class_dir(data_dir, slug)):
        if name not in (None, d.name):
            continue
        meta = read_json(d / META_FILE, {}) or {}
        if meta.get("class_settled"):
            continue
        terms = _terms(d)
        if not terms and not unsorted:  # in a real class, only the words can say it is in the wrong one
            continue
        states = lib._job_states(data_dir) if states is None else states
        if lib.session_state(d, states.get(str(d.resolve())), meta) not in QUIET_STATES:
            continue
        classes = classes if classes is not None else _classes(data_dir)
        if not unsorted and _content(classes, terms, (slug, d.name)) in (None, slug):
            continue  # the usual case: nothing else could suggest a move, so no other lecture's meta.json is read
        times = times if times is not None else class_times(lib.class_dir(data_dir, slug))
        facts = _facts(d, meta, times[d.name][0])
        s = suggest(classes, facts["start"], facts["topic"], terms, (slug, d.name))
        if s and s["class"] != slug and (unsorted or "content" in s["signals"]):
            out[d.name] = s
    return out


def for_class(data_dir: Path, slug: str, name: str | None = None, states: dict | None = None) -> dict[str, dict]:
    """{lecture folder name: where it seems to belong}, for one lecture if `name` is given. `states`: the job
    states, if the caller has them."""
    return _quietly({}, _for_class, data_dir, slug, name, states)


def _time_of(name: str) -> datetime | None:
    """When a file picked for import was recorded: a date and time in its name. Its file date proves nothing
    (for a download it is the download), and the schedule signal takes real recording times only."""
    if _CLOUD_NAME.search(name):
        return folder_date_title("", name)[1]
    if m := _NAMED_TIME.search(name):
        try:
            return datetime(*map(int, m.groups()))
        except ValueError:
            pass
    return None


def _for_import(data_dir: Path, files: list[dict], topic: str | None, when: str | None) -> dict | None:
    classes = _classes(data_dir)
    if topic or when:  # a Zoom folder: one recording
        return suggest(classes, datetime.fromisoformat(when) if when else None, _title_words(topic))
    picks = [suggest(classes, _time_of(str(f.get("name") or ""))) for f in files]
    return picks[0] if picks and all(p and p["class"] == picks[0]["class"] for p in picks) else None  # one picker: all must agree


def for_import(data_dir: Path, files: list[dict], topic: str | None = None, when: str | None = None) -> dict | None:
    """The class to pre-select for the files picked for import. Nothing is filed by this."""
    return _quietly(None, _for_import, data_dir, files, topic, when)


def keep_here(data_dir: Path, slug: str, name: str) -> None:
    """Never ask about this lecture again. Raises LibraryError, or FileNotFoundError for a missing lecture."""
    d = lib.session_dir(data_dir, slug, name)
    if not d.is_dir():
        raise FileNotFoundError("No such lecture.")
    update_meta(d, class_settled=True)


def _sketch_samples(data_dir: Path) -> None:
    for c in _classes(data_dir).values():
        for d in c["dirs"][-CONTENT_SAMPLE - 1:]:
            try:
                dup.sketch_session(d)
            except Exception:  # one bad file costs one lecture's comparison
                log.exception("could not sketch %s", d)


def after_transcribe(data_dir: Path) -> None:
    """A new lecture is compared with every class's recent lectures: those that predate this feature catch up here,
    in the same tiny CPU step as the duplicate sketch. Bounded per class; never raises."""
    _quietly(None, _sketch_samples, data_dir)
