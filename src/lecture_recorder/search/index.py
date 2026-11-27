"""Search index: `<data>/app.db`, a SQLite file rebuilt from the session folders.

Files stay the source of truth. Every search first syncs the index with the folders (a stat of
transcript.json, meta.json and the embedding files per session), so edits to transcript.json, made
in the app or by hand, are searchable at once and deleted sessions disappear. `rebuild()` empties the
index and re-reads everything; it gives the same results because nothing is stored only here.

Exact search: each session's normalized token stream (search/text.py) is one FTS5 row, used as a
prefilter with a quoted phrase query built only from letter/digit tokens and passed as a bound
parameter. Matches are then located in Python on the same tokens, so phrases that cross segment
boundaries are found, and each hit maps to a time (word timestamps where the segment text is
unchanged, else interpolated within the segment).

Concept search: window vectors from <session>/embeddings.npy, cosine similarity in numpy.
"""
import bisect
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from lecture_recorder.config import Config
from lecture_recorder.pipeline.outputs import EDITABLE
from lecture_recorder.recorder.session import META_FILE, iter_session_dirs, read_json
from lecture_recorder.search.embed import EMB_META, EMB_NPY, embed_state, load_vectors, text_hash
from lecture_recorder.search.text import fts_phrase, token_spans, token_times, tokens
from lecture_recorder.units import filter_scope, flatten, in_spans

log = logging.getLogger("lecture_recorder.search")
DB_NAME = "app.db"
SCHEMA_VERSION = 2  # 2: unit_spans
CONTEXT_TOKENS = 14  # words of context on each side of an exact match
MAX_QUERY_TOKENS = 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS info(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS sessions(
  id INTEGER PRIMARY KEY, sid TEXT UNIQUE NOT NULL, class TEXT NOT NULL, name TEXT NOT NULL, title TEXT,
  created TEXT, duration_s REAL, sig TEXT NOT NULL, text_hash TEXT, tokens TEXT NOT NULL, tok_times TEXT NOT NULL,
  emb_state TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS segments(
  session_id INTEGER NOT NULL, idx INTEGER NOT NULL, seg_id INTEGER, start_s REAL, end_s REAL, speaker TEXT,
  role TEXT, text TEXT, tok_start INTEGER, tok_end INTEGER, PRIMARY KEY(session_id, idx));
CREATE TABLE IF NOT EXISTS windows(
  session_id INTEGER NOT NULL, idx INTEGER NOT NULL, start_s REAL, end_s REAL, text TEXT, vec BLOB NOT NULL,
  PRIMARY KEY(session_id, idx));
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(tokens, tokenize='unicode61 remove_diacritics 2');
CREATE TABLE IF NOT EXISTS unit_spans(
  class TEXT NOT NULL, unit_id TEXT NOT NULL, unit_name TEXT NOT NULL, unit_index INTEGER NOT NULL,
  session TEXT NOT NULL, start_s REAL NOT NULL, end_s REAL);
"""


def _sig(d: Path, cfg: Config) -> str:
    """Files' (mtime, size) plus the settings the stored vectors depend on (a model change reindexes)."""
    parts = [f"{cfg.llm.embedding_model}@{cfg.search.window_s}/{cfg.search.step_s}"]
    for name in (EDITABLE, META_FILE, EMB_META, EMB_NPY):
        try:
            st = (d / name).stat()
            parts.append(f"{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            parts.append("-")
    return "|".join(parts)


class SearchIndex:
    def __init__(self, data_dir: Path, config: Config):
        self.data_dir = Path(data_dir).resolve()
        self.path = self.data_dir / DB_NAME
        self.cfg = config
        self._lock = threading.RLock()
        self._vec_cache: tuple | None = None  # (generation, rows, matrix)
        self._schema_ready = False

    # ------------------------------------------------------------------ storage
    @contextmanager
    def _db(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            con.row_factory = sqlite3.Row
            if not self._schema_ready or not self.path.exists():
                con.execute("PRAGMA journal_mode=WAL")  # persistent in the file
                ver = con.execute("PRAGMA user_version").fetchone()[0]
                if ver not in (0, SCHEMA_VERSION):  # an index from another version: it's only an index
                    for t in ("info", "sessions", "segments", "windows", "fts", "unit_spans"):
                        con.execute(f"DROP TABLE IF EXISTS {t}")
                con.executescript(SCHEMA)
                con.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                self._schema_ready = True
            yield con
        finally:
            con.close()

    def _generation(self, con) -> int:
        row = con.execute("SELECT value FROM info WHERE key='generation'").fetchone()
        return int(row[0]) if row else 0

    def _delete(self, con, session_id: int) -> None:
        for t in ("segments", "windows"):
            con.execute(f"DELETE FROM {t} WHERE session_id=?", (session_id,))
        con.execute("DELETE FROM fts WHERE rowid=?", (session_id,))
        con.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    def _on_disk(self) -> dict[str, Path]:
        return {f"{d.parent.parent.name}/{d.name}": d for d in iter_session_dirs(self.data_dir)
                if (d / EDITABLE).is_file()}

    def sync(self) -> dict:
        """Bring the index in line with the folders. Returns counts of added/updated/removed sessions."""
        stats = self._sync_sessions()
        stats["unit_classes"] = self._sync_units()
        return stats

    def _sync_units(self) -> int:
        """Rebuild a class's unit spans when its units.json or its list of lectures changed."""
        from lecture_recorder.units import UNITS_FILE, unit_spans

        root = self.data_dir / "classes"
        classes = {c.name: c for c in root.iterdir() if c.is_dir()} if root.is_dir() else {}
        sigs = {}
        for slug, c in classes.items():
            try:
                st = (c / UNITS_FILE).stat()
                usig = f"{st.st_mtime_ns}:{st.st_size}"
            except OSError:
                usig = "-"
            sessions = sorted(p.name for p in (c / "sessions").iterdir()) if (c / "sessions").is_dir() else []
            sigs[slug] = usig + "|" + ",".join(sessions)
        with self._lock, self._db() as con:
            known = {r["key"][6:]: r["value"] for r in con.execute("SELECT key, value FROM info WHERE key LIKE 'units:%'")}
            todo = [s for s in sigs if known.get(s) != sigs[s]] + [s for s in known if s not in sigs]
            if not todo:
                return 0
            con.execute("BEGIN IMMEDIATE")
            try:
                for slug in todo:
                    con.execute("DELETE FROM unit_spans WHERE class=?", (slug,))
                    con.execute("DELETE FROM info WHERE key=?", (f"units:{slug}",))
                    if slug not in classes:
                        continue
                    rows = [(slug, u["id"], u["name"], u["index"], sp["session"], sp["start"], sp["end"])
                            for u in unit_spans(classes[slug]) for sp in u["spans"]]
                    con.executemany("INSERT INTO unit_spans VALUES(?,?,?,?,?,?,?)", rows)
                    con.execute("INSERT INTO info(key, value) VALUES(?, ?)", (f"units:{slug}", sigs[slug]))
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
            return len(todo)

    def units(self, class_slug: str, scope: dict | None = None, sync: bool = True) -> list[dict]:
        """A class's units with their spans, from the index; with `scope` ({'units': [...]} or
        {'from', 'to'} dates), only what the scope covers (units.filter_scope, same as the files)."""
        if sync:
            self._sync_units()
        with self._db() as con:
            rows = con.execute("SELECT * FROM unit_spans WHERE class=? ORDER BY unit_index, session, start_s",
                               (class_slug,)).fetchall()
        out: dict[str, dict] = {}
        for r in rows:
            out.setdefault(r["unit_id"], {"id": r["unit_id"], "name": r["unit_name"], "index": r["unit_index"],
                                          "spans": []})["spans"].append(
                {"session": r["session"], "start": r["start_s"], "end": r["end_s"]})
        units = list(out.values())
        return filter_scope(units, scope) if scope else units

    def class_windows(self, class_slug: str) -> tuple[list[dict], np.ndarray | None, list[str]]:
        """A class's concept-search windows in timeline order (validated vectors from the index) and
        the transcribed lectures without current vectors. For unit boundary suggestions."""
        self.sync()
        with self._db() as con:
            rows = con.execute(
                "SELECT s.name, w.idx, w.start_s, w.vec FROM windows w JOIN sessions s ON s.id = w.session_id "
                "WHERE s.class=? ORDER BY s.name, w.idx", (class_slug,)).fetchall()
            missing = [r["name"] for r in con.execute(
                "SELECT name FROM sessions WHERE class=? AND emb_state != 'current' ORDER BY name", (class_slug,))]
        vecs = [np.frombuffer(r["vec"], dtype=np.float32) for r in rows]
        if vecs and len({v.shape[0] for v in vecs}) > 1:
            return [], None, missing
        return ([{"session": r["name"], "t": r["start_s"], "first": r["idx"] == 0} for r in rows],
                np.vstack(vecs) if vecs else None, missing)

    def _unit_of(self, con, class_slug: str | None = None) -> dict:
        """sid -> [(start, end, unit name, unit id)] for labelling results with their unit."""
        out: dict[str, list] = {}
        rows = (con.execute("SELECT * FROM unit_spans WHERE class=?", (class_slug,)) if class_slug
                else con.execute("SELECT * FROM unit_spans"))
        for r in rows:
            out.setdefault(f"{r['class']}/{r['session']}", []).append(
                (r["start_s"], r["end_s"], r["unit_name"], r["unit_id"]))
        return out

    def _sync_sessions(self) -> dict:
        with self._lock, self._db() as con:
            disk = {sid: (d, _sig(d, self.cfg)) for sid, d in self._on_disk().items()}  # sig taken BEFORE reading
            known = {r["sid"]: (r["id"], r["sig"]) for r in con.execute("SELECT id, sid, sig FROM sessions")}
            gone = [sid for sid in known if sid not in disk]
            changed = [sid for sid, (_, sig) in disk.items() if known.get(sid, (None, None))[1] != sig]
            stats = {"added": 0, "updated": 0, "removed": 0}
            if not gone and not changed:
                return stats
            con.execute("BEGIN IMMEDIATE")
            try:
                known = {r["sid"]: (r["id"], r["sig"]) for r in con.execute("SELECT id, sid, sig FROM sessions")}
                for sid in gone:
                    if sid in known:
                        self._delete(con, known[sid][0])
                        stats["removed"] += 1
                for sid in changed:
                    d, sig = disk[sid]
                    if sid in known:
                        if known[sid][1] == sig:  # another process indexed it meanwhile
                            continue
                        self._delete(con, known[sid][0])
                    con.execute("SAVEPOINT one")
                    try:
                        ok = self._index_session(con, sid, d, sig)
                        con.execute("RELEASE one")
                    except Exception as e:  # a malformed hand-edited file: skip it, keep searching the rest
                        con.execute("ROLLBACK TO one")
                        con.execute("RELEASE one")
                        log.warning("search index: skipped %s (%s)", sid, type(e).__name__)
                        ok = False
                    if ok:
                        stats["updated" if sid in known else "added"] += 1
                    else:
                        stats.setdefault("skipped", 0)
                        stats["skipped"] += 1
                con.execute("INSERT INTO info(key, value) VALUES('generation', '1') ON CONFLICT(key) "
                            "DO UPDATE SET value = CAST(value AS INTEGER) + 1")
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
            return stats

    def rebuild(self) -> dict:
        """Forget everything and re-read all session folders."""
        with self._lock:
            with self._db() as con:
                con.execute("BEGIN IMMEDIATE")
                for t in ("sessions", "segments", "windows", "fts", "unit_spans"):
                    con.execute(f"DELETE FROM {t}")
                con.execute("DELETE FROM info WHERE key LIKE 'units:%'")
                con.execute("COMMIT")
            self._vec_cache = None
            return self.sync()

    def _index_session(self, con, sid: str, d: Path, sig: str) -> bool:
        doc = read_json(d / EDITABLE, None)
        if not isinstance(doc, dict) or not isinstance(doc.get("segments"), list):
            return False
        meta = read_json(d / META_FILE, {}) or {}
        segs = sorted(doc["segments"], key=lambda s: float(s["start"]))
        all_tokens, all_times, seg_rows = [], [], []
        for i, s in enumerate(segs):
            st = tokens(s.get("text", ""))
            seg_rows.append((i, s.get("id"), float(s["start"]), float(s["end"]), s.get("speaker"), s.get("role"),
                             s.get("text", ""), len(all_tokens), len(all_tokens) + len(st)))
            all_tokens += st
            all_times += token_times(s, st)
        duration = meta.get("recorded_s") or doc.get("duration_s") or (float(segs[-1]["end"]) if segs else 0)
        state = embed_state(d, self.cfg, doc)
        cur = con.execute(
            "INSERT INTO sessions(sid, class, name, title, created, duration_s, sig, text_hash, tokens, tok_times, emb_state)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (sid, d.parent.parent.name, d.name, meta.get("title") or d.name, meta.get("created") or "",
             duration, sig, text_hash(doc), " ".join(all_tokens), json.dumps(all_times), state))
        sess_id = cur.lastrowid
        con.executemany("INSERT INTO segments VALUES(?,?,?,?,?,?,?,?,?,?)", [(sess_id, *r) for r in seg_rows])
        con.execute("INSERT INTO fts(rowid, tokens) VALUES(?, ?)", (sess_id, " ".join(all_tokens)))
        if state == "current":
            emeta = read_json(d / EMB_META, {}) or {}
            m = load_vectors(d, emeta)
            if m is None:  # the .npy doesn't match its meta file: treat as not embedded
                con.execute("UPDATE sessions SET emb_state='stale' WHERE id=?", (sess_id,))
            else:
                con.executemany("INSERT INTO windows VALUES(?,?,?,?,?,?)",
                                [(sess_id, i, w["start"], w["end"], w["text"], m[i].tobytes())
                                 for i, w in enumerate(emeta["windows"])])
        return True

    # ------------------------------------------------------------------ exact search
    def exact(self, query: str, class_slug: str | None = None, limit: int | None = None,
              scope: dict | None = None) -> dict:
        """`scope` (with `class_slug`): {'units': [ids]} or {'from', 'to'}; only matches inside it count,
        and their context never reaches past the scope's span."""
        limit = limit or self.cfg.search.exact_limit
        q = tokens(query or "")[:MAX_QUERY_TOKENS]
        out = {"mode": "exact", "query": query, "tokens": q, "total": 0, "sessions": 0, "results": []}
        self.sync()
        spans = self._scope_spans(class_slug, scope)
        if not q:
            out["message"] = "Type at least one word or number."
            return out
        with self._db() as con:
            sql = ("SELECT s.* FROM fts JOIN sessions s ON s.id = fts.rowid WHERE fts MATCH ?"
                   + (" AND s.class = ?" if class_slug else "") + " ORDER BY s.created DESC, s.sid DESC")
            rows = con.execute(sql, (fts_phrase(q), class_slug) if class_slug else (fts_phrase(q),)).fetchall()
            names = _class_names(self.data_dir)
            unit_of = self._unit_of(con, class_slug)
            k = len(q)
            for row in rows:
                toks = row["tokens"].split(" ")
                hits = [i for i in range(len(toks) - k + 1) if toks[i] == q[0] and toks[i:i + k] == q]
                times = json.loads(row["tok_times"])
                if spans is not None:
                    hits = [i for i in hits if in_spans(spans, row["name"], times[i], times[i + k - 1])]
                if not hits:
                    continue
                out["total"] += len(hits)
                out["sessions"] += 1
                room = limit - len(out["results"])
                if room <= 0:
                    continue
                segs = con.execute("SELECT * FROM segments WHERE session_id=? ORDER BY idx", (row["id"],)).fetchall()
                ctx = _Context(segs)
                for i in hits[:room]:
                    j = i + k - 1
                    si, sj = ctx.seg_of(i), ctx.seg_of(j)
                    lo, hi = _span_tokens(times, spans.get(row["name"], []) if spans is not None else None, i)
                    unit = _label(unit_of.get(row["sid"], []), times[i])
                    out["results"].append({
                        **_session_fields(row, names),
                        "start": times[i], "end": segs[sj]["end_s"],
                        "segment_id": segs[si]["seg_id"], "speaker": segs[si]["speaker"], "role": segs[si]["role"],
                        "unit": unit[0], "unit_id": unit[1],
                        "context": ctx.around(i, j, CONTEXT_TOKENS, lo, hi),
                    })
        if out["total"] > len(out["results"]):
            out["message"] = f"Showing the first {len(out['results'])} of {out['total']} matches."
        return out

    # ------------------------------------------------------------------ concept search
    def concept_status(self, sync: bool = True) -> dict:
        if sync:
            self.sync()
        with self._db() as con:
            counts = dict(con.execute("SELECT emb_state, COUNT(*) FROM sessions GROUP BY emb_state").fetchall())
            windows = con.execute("SELECT COUNT(*) FROM windows").fetchone()[0]
        ready = counts.get("current", 0)
        return {"transcribed": sum(counts.values()), "indexed": ready, "pending": sum(counts.values()) - ready,
                "windows": windows}

    def pending_embedding(self) -> list[Path]:
        """Session folders whose concept-search vectors are missing or stale."""
        self.sync()
        disk = self._on_disk()
        with self._db() as con:
            return [disk[r["sid"]] for r in con.execute("SELECT sid FROM sessions WHERE emb_state != 'current'")
                    if r["sid"] in disk]

    def _scope_spans(self, class_slug: str | None, scope: dict | None) -> dict | None:
        if not scope:
            return None
        if not class_slug:
            raise ValueError("a unit or date scope needs its class")
        return flatten(self.units(class_slug, scope, sync=False))  # callers synced just before

    def _matrix(self, con):
        gen = self._generation(con)
        if self._vec_cache is None or self._vec_cache[0] != gen:
            rows = con.execute(
                "SELECT w.session_id, w.idx, w.start_s, w.end_s, w.text, w.vec, s.sid, s.class, s.name, s.title, "
                "s.created, s.duration_s FROM windows w JOIN sessions s ON s.id = w.session_id "
                "ORDER BY w.session_id, w.idx").fetchall()
            vecs = [np.frombuffer(r["vec"], dtype=np.float32) for r in rows]
            if vecs and len({v.shape[0] for v in vecs}) > 1:  # mixed sizes can't be compared: keep the common one
                dims = max({v.shape[0] for v in vecs}, key=[v.shape[0] for v in vecs].count)
                keep = [i for i, v in enumerate(vecs) if v.shape[0] == dims]
                rows, vecs = [rows[i] for i in keep], [vecs[i] for i in keep]
            m = np.vstack(vecs) if vecs else None
            self._vec_cache = (gen, [dict(r) | {"vec": None} for r in rows], m)
        return self._vec_cache[1], self._vec_cache[2]

    def concept(self, qvec: np.ndarray, class_slug: str | None = None, k: int | None = None,
                sync: bool = True, scope: dict | None = None) -> list[dict]:
        """Top-k windows by cosine similarity; a window overlapping a better one from the same session is
        skipped. With a scope, only windows lying entirely inside it count (a window that straddles a
        unit boundary would carry the other unit's words)."""
        k = k or self.cfg.search.concept_k
        if sync:
            self.sync()
        spans = self._scope_spans(class_slug, scope)
        with self._lock, self._db() as con:
            rows, m = self._matrix(con)
            unit_of = self._unit_of(con, class_slug)
        if m is None or m.shape[1] != qvec.shape[0]:
            return []
        scores = m @ qvec.astype(np.float32)
        names = _class_names(self.data_dir)
        picked: list[dict] = []
        for i in np.argsort(-scores, kind="stable"):
            r = rows[int(i)]
            if class_slug and r["class"] != class_slug:
                continue
            if spans is not None and not in_spans(spans, r["name"], r["start_s"], r["end_s"]):
                continue
            if any(p["sid"] == r["sid"] and p["start"] < r["end_s"] and r["start_s"] < p["end"] for p in picked):
                continue
            unit = _label(unit_of.get(r["sid"], []), r["start_s"])
            picked.append({**_session_fields(r, names), "start": r["start_s"], "end": r["end_s"],
                           "unit": unit[0], "unit_id": unit[1],
                           "text": r["text"], "score": round(float(scores[i]), 4), "rank": len(picked) + 1})
            if len(picked) >= k:
                break
        return picked


class _Context:
    """Maps global token positions back to the original segment text, for match context."""

    def __init__(self, segs):
        self.segs = segs
        self.starts = [s["tok_start"] for s in segs]
        self._spans: dict[int, list] = {}

    def seg_of(self, t: int) -> int:
        return bisect.bisect_right(self.starts, t) - 1

    def span(self, t: int) -> tuple[int, int, int]:
        si = self.seg_of(t)
        if si not in self._spans:
            self._spans[si] = token_spans(self.segs[si]["text"])
        _, a, b = self._spans[si][t - self.segs[si]["tok_start"]]
        return si, a, b

    def around(self, i: int, j: int, w: int, lo: int = 0, hi: int | None = None) -> dict:
        n = self.segs[-1]["tok_end"]
        a, b = max(lo, i - w), min(n - 1 if hi is None else hi, j + w)
        sa, ca, _ = self.span(a)
        sb, _, cb = self.span(b)
        _, mi, _ = self.span(i)
        _, _, mj = self.span(j)
        si, sj = self.seg_of(i), self.seg_of(j)
        text, offs = "", {}
        for s in range(sa, sb + 1):
            seg_text = self.segs[s]["text"]
            lo = ca if s == sa else 0
            hi = cb if s == sb else len(seg_text)
            if text:
                text += " "
            offs[s] = len(text) - lo
            text += seg_text[lo:hi]
        ms, me = offs[si] + mi, offs[sj] + mj
        return {"before": ("…" if a > 0 else "") + text[:ms], "match": text[ms:me],
                "after": text[me:] + ("…" if b < n - 1 else "")}


def _span_tokens(times: list[float], spans: list | None, i: int) -> tuple[int, int | None]:
    """Token index range of the span that contains token i (the whole lecture without a scope)."""
    if not spans:
        return 0, None
    t = times[i]
    lo_t, hi_t = next(((lo, hi) for lo, hi in spans if lo <= t and (hi is None or t < hi)), (0.0, None))
    lo = bisect.bisect_left(times, lo_t)  # token times never decrease, so bisect finds the span edges
    hi = (bisect.bisect_left(times, hi_t) if hi_t is not None else len(times)) - 1
    return lo, hi


def _label(spans: list, t: float) -> tuple[str | None, str | None]:
    for lo, hi, name, uid in spans:
        if lo <= t and (hi is None or t < hi):
            return name, uid
    return None, None


def _class_names(data_dir: Path) -> dict[str, str]:
    out = {}
    root = Path(data_dir) / "classes"
    if root.is_dir():
        for c in root.iterdir():
            info = read_json(c / "class.json", {}) or {}
            out[c.name] = info.get("name") or c.name
    return out


def _session_fields(row, names: dict) -> dict:
    """What every result carries, so later scoping (units, dates) can filter on it."""
    return {"sid": row["sid"], "class": row["class"], "class_name": names.get(row["class"], row["class"]),
            "session": row["name"], "title": row["title"], "created": row["created"], "duration_s": row["duration_s"]}
