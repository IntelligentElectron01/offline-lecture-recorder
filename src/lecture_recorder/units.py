"""Units and exam scopes. Plain files in the class folder are the source of truth:

data/classes/<slug>/units.json   {"units": [{"id", "name", "start": null | {"session", "t"}}]}
data/classes/<slug>/exams.json   {"exams": [{"id", "name", "units": [ids]} | {"id", "name", "from", "to"}]}

A class's lectures form one timeline (sessions in date-time order). A unit starts at its `start`
point (the first unit at the very beginning, start = null) and runs until the next unit's start, so
a boundary can fall mid-lecture and one lecture can belong to several units. Every moment belongs
to exactly one unit. Boundary suggestions (search/units_suggest.py) are never applied without the
user's confirmation; adding a boundary here is always a user action.
"""
import math
import re
import threading
import uuid
from datetime import date
from pathlib import Path

from lecture_recorder.dates import lecture_times
from lecture_recorder.library import by_date
from lecture_recorder.recorder.session import now_iso, read_json, session_dirs, write_json_atomic

UNITS_FILE, EXAMS_FILE = "units.json", "exams.json"
EXAMS_DIR = "exams"
EXAM_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
UNIT_ID_RE = re.compile(r"^u-[0-9a-f]{8}$")  # ids add_boundary makes (the first unit, "u1", is never removed)
_write_lock = threading.RLock()  # read-modify-write of units.json / exams.json, one request at a time


class UnitError(ValueError):
    pass


# ------------------------------------------------------------------ timeline
def class_sessions(class_dir: Path) -> list[dict]:
    """[{'name', 'title', 'created', 'duration_s'}] in time order: the date shown (an import's may come from its file
    name, dates.py), then the folder's start date-time."""
    metas = {d: read_json(d / "meta.json", {}) or {} for d in session_dirs(Path(class_dir))}
    times = lecture_times({d.name: m for d, m in metas.items()})
    out = []
    for d, meta in metas.items():
        dur = meta.get("recorded_s") or meta.get("duration_s")
        if dur is None and (d / "transcript.raw.json").exists():
            dur = (read_json(d / "transcript.raw.json", {}) or {}).get("duration_s")
        out.append({"name": d.name, "title": meta.get("title") or d.name, "created": times[d.name][1],
                    "duration_s": float(dur) if dur else None})
    return sorted(out, key=by_date)


def _placed(units: list[dict], sessions: list[dict]) -> list[dict]:
    """Units in timeline order, each with its start position `key` = (lecture index, seconds).

    A unit whose start lecture was deleted is not merged into the previous unit: it now starts at the
    beginning of the next lecture after the deleted one (`moved`), or, if there is none, it has no
    time at all (`orphan`). Both are shown to the user."""
    names = [s["name"] for s in sessions]
    order = {n: i for i, n in enumerate(names)}
    out = []
    for u in units:
        st = u.get("start")
        if st is None:
            out.append({**u, "key": (-1, 0.0)})
        elif st.get("session") in order:
            out.append({**u, "key": (order[st["session"]], float(st["t"]))})
        else:
            nxt = next((i for i, n in enumerate(names) if n > str(st.get("session"))), None)
            out.append({**u, "key": (nxt, 0.0), "moved": True} if nxt is not None else {**u, "orphan": True})
    placed = sorted((u for u in out if not u.get("orphan")), key=lambda u: u["key"])
    return placed + [u for u in out if u.get("orphan")]


# ------------------------------------------------------------------ units.json
def load_units(class_dir: Path) -> list[dict]:
    data = read_json(Path(class_dir) / UNITS_FILE, None) or {}
    units = [u for u in data.get("units", []) if isinstance(u, dict) and u.get("id")]
    if not any(u.get("start") is None for u in units):
        units.insert(0, {"id": "u1", "name": "Unit 1", "start": None})  # there is always a first unit
    return units


def save_units(class_dir: Path, units: list[dict]) -> None:
    clean = [{k: v for k, v in u.items() if k not in ("key", "moved", "orphan")} for u in units]
    write_json_atomic(Path(class_dir) / UNITS_FILE, {"units": clean, "updated": now_iso()})


def ordered_units(class_dir: Path, sessions: list[dict] | None = None) -> list[dict]:
    return _placed(load_units(class_dir), sessions if sessions is not None else class_sessions(class_dir))


def unit_spans(class_dir: Path, sessions: list[dict] | None = None) -> list[dict]:
    """[{'id', 'name', 'index', 'moved', 'spans': [{'session', 'start', 'end'}]}]; end None = to the end."""
    sessions = sessions if sessions is not None else class_sessions(class_dir)
    units = [u for u in ordered_units(class_dir, sessions) if not u.get("orphan")]
    out = [{"id": u["id"], "name": u["name"], "index": i, "moved": bool(u.get("moved")), "spans": []}
           for i, u in enumerate(units)]
    keys = [u["key"] for u in units]
    for si, s in enumerate(sessions):
        # the unit active at this lecture's beginning, then each start that falls inside it
        cur = max(i for i, k in enumerate(keys) if k <= (si, 0.0))
        t = 0.0
        for i in range(cur + 1, len(units)):
            if keys[i][0] != si:
                continue
            bt = keys[i][1]
            if bt > t:
                out[cur]["spans"].append({"session": s["name"], "start": t, "end": bt})
            cur, t = i, bt
        out[cur]["spans"].append({"session": s["name"], "start": t, "end": None})
    return out


def add_boundary(class_dir: Path, session: str, t: float, name: str | None = None, unit_id: str | None = None) -> dict:
    """User action: a new unit starts at (session, t). Returns the new unit.

    `unit_id` is for Undo after a removal: the unit comes back under its old id, so exams that
    list it cover it again."""
    t = float(t)
    if not math.isfinite(t):
        raise UnitError("The boundary time must be a number of seconds.")
    if unit_id is not None and not UNIT_ID_RE.match(str(unit_id)):
        raise UnitError("Invalid unit id.")
    if 0 <= t < 1.0:
        t = 0.0  # the first second is the lecture's start (no "0:00-0:00" part, one gap button)
    with _write_lock:
        sessions = class_sessions(class_dir)
        by_name = {s["name"]: s for s in sessions}
        if session not in by_name:
            raise UnitError("No such lecture in this class.")
        dur = by_name[session]["duration_s"]
        if t < 0 or (dur is not None and t >= dur):
            raise UnitError("The boundary must be inside the lecture.")
        units = load_units(class_dir)
        if any((u.get("start") or {}).get("session") == session and abs(float(u["start"]["t"]) - t) < 1.0 for u in units):
            raise UnitError("A unit already starts there.")
        if session == sessions[0]["name"] and t < 1.0:
            raise UnitError("The first unit already starts at the beginning of the first lecture.")
        if unit_id is not None and any(u["id"] == unit_id for u in units):
            raise UnitError("That unit already exists.")
        unit = {"id": unit_id or f"u-{uuid.uuid4().hex[:8]}", "name": (name or "").strip()[:100] or "New unit",
                "start": {"session": session, "t": round(t, 2)}, "created": now_iso()}
        _save_numbered(class_dir, units + [unit], sessions)
        return {k: v for k, v in next(u for u in load_units(class_dir) if u["id"] == unit["id"]).items()}


def _save_numbered(class_dir: Path, units: list[dict], sessions: list[dict] | None = None) -> None:
    """Save; units the user never named are called 'Unit N' in timeline order."""
    placed = _placed(units, sessions if sessions is not None else class_sessions(class_dir))
    for i, u in enumerate(placed, 1):
        if not u.get("orphan") and (u["name"] == "New unit" or re.fullmatch(r"Unit \d+", u["name"])):
            u["name"] = f"Unit {i}"
    save_units(class_dir, placed)


def rename_unit(class_dir: Path, unit_id: str, name: str) -> None:
    name = (name or "").strip()
    if not 1 <= len(name) <= 100:
        raise UnitError("A unit name must be 1-100 characters.")
    with _write_lock:
        units = load_units(class_dir)
        for u in units:
            if u["id"] == unit_id:
                u["name"] = name
                save_units(class_dir, units)
                return
    raise UnitError("No such unit.")


def remove_boundary(class_dir: Path, unit_id: str) -> None:
    """Remove a unit's start boundary: its time joins the previous unit. The first unit can't be removed."""
    with _write_lock:
        units = load_units(class_dir)
        keep = [u for u in units if not (u["id"] == unit_id and u.get("start") is not None)]
        if len(keep) == len(units):
            raise UnitError("No such boundary (the first unit always starts at the beginning).")
        _save_numbered(class_dir, keep)


def drop_session_boundaries(class_dir: Path, session: str) -> list[dict]:
    """A lecture moved to Recently deleted: remove the units that start in it (their time joins the
    previous unit) and return them as they were, so a restore can put them back."""
    with _write_lock:
        units = load_units(class_dir)
        removed = [u for u in units if (u.get("start") or {}).get("session") == session]
        if removed:
            _save_numbered(class_dir, [u for u in units if u not in removed])
        return removed


def restore_boundaries(class_dir: Path, removed: list[dict]) -> None:
    """Undo drop_session_boundaries after the lecture is back (units already present are left alone)."""
    with _write_lock:
        units = load_units(class_dir)
        ids = {u["id"] for u in units}
        back = [u for u in removed if isinstance(u, dict) and u.get("id") and u["id"] not in ids and u.get("start")]
        if back:
            _save_numbered(class_dir, units + back)


# ------------------------------------------------------------------ exams
def load_exams(class_dir: Path) -> list[dict]:
    return [e for e in (read_json(Path(class_dir) / EXAMS_FILE, None) or {}).get("exams", []) if isinstance(e, dict)]


def get_exam(class_dir: Path, exam_id: str) -> dict | None:
    return next((e for e in load_exams(class_dir) if e.get("id") == exam_id), None)


def check_scope(scope: dict) -> dict:
    """{'units': [non-empty ids]} or {'from', 'to'} (YYYY-MM-DD, inclusive). Anything else is refused,
    so an empty or malformed scope can never widen to the whole class."""
    if "units" in scope:
        ids = scope["units"]
        if not (isinstance(ids, list) and ids and all(isinstance(x, str) and x for x in ids)):
            raise UnitError("Choose at least one unit.")
        return {"units": list(dict.fromkeys(ids))}
    lo, hi = scope.get("from"), scope.get("to")
    try:
        if date.fromisoformat(lo) > date.fromisoformat(hi):
            raise UnitError("The date range ends before it starts.")
    except (TypeError, ValueError) as e:
        if isinstance(e, UnitError):
            raise
        raise UnitError("Choose at least one unit, or a date range (YYYY-MM-DD).")
    return {"from": lo, "to": hi}


def save_exam(class_dir: Path, name: str, unit_ids: list[str] | None = None, date_from: str | None = None,
              date_to: str | None = None) -> dict:
    """A named exam scope: unit checkboxes, or a date range (inclusive, YYYY-MM-DD)."""
    name = (name or "").strip()
    if not 1 <= len(name) <= 100:
        raise UnitError("An exam name must be 1-100 characters.")
    scope = check_scope({"units": unit_ids} if unit_ids else {"from": date_from, "to": date_to})
    with _write_lock:
        if "units" in scope and not set(scope["units"]) <= {u["id"] for u in load_units(class_dir)}:
            raise UnitError("Unknown unit in the exam scope.")
        exams = load_exams(class_dir)
        base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48] or "exam"
        exam_id, n = base, 2
        while any(e.get("id") == exam_id for e in exams):
            exam_id, n = f"{base}-{n}", n + 1
        exam = {"id": exam_id, "name": name, "created": now_iso(), **scope}
        write_json_atomic(Path(class_dir) / EXAMS_FILE, {"exams": exams + [exam]})
        return exam


def filter_scope(units: list[dict], scope: dict) -> list[dict]:
    """Keep what `scope` covers from units-with-spans (used by the index and by the files alike). A date
    range keeps whole lectures on those days, still grouped by the units they fall in."""
    scope = check_scope(scope)
    if "units" in scope:
        wanted = set(scope["units"])
        return [u for u in units if u["id"] in wanted]
    out = []
    for u in units:
        spans = [sp for sp in u["spans"] if scope["from"] <= sp["session"][:10] <= scope["to"]]
        if spans:
            out.append({**u, "spans": spans})
    return out


def scope_spans(class_dir: Path, scope: dict, sessions: list[dict] | None = None) -> list[dict]:
    """The spans an exam (or ad-hoc scope) covers, from the files: [{'id', 'name', 'index', 'spans'}]."""
    return filter_scope(unit_spans(class_dir, sessions), scope)


def in_spans(spans_by_session: dict[str, list[tuple[float, float | None]]], session: str, start: float,
             end: float | None = None) -> bool:
    """True if the point `start` (or the interval [start, end]) lies inside one span of `session`;
    spans are half-open [lo, hi), hi None = to the end of the lecture."""
    end = start if end is None else end
    return any(start >= lo and (hi is None or (start < hi and end <= hi))
               for lo, hi in spans_by_session.get(session, []))


def flatten(units: list[dict]) -> dict[str, list[tuple[float, float | None]]]:
    out: dict[str, list] = {}
    for u in units:
        for sp in u["spans"]:
            out.setdefault(sp["session"], []).append((sp["start"], sp["end"]))
    return out


def exam_dir(class_dir: Path, exam_id: str) -> Path:
    if not EXAM_ID_RE.match(exam_id or ""):
        raise UnitError("invalid exam")
    return Path(class_dir) / EXAMS_DIR / exam_id
