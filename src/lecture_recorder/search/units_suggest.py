"""Unit boundary suggestions from topic shifts. CPU only: reuses the concept-search vectors the
search index already holds (validated against each session's embeddings files); no model is loaded
and no GPU is used.

The class's windows are laid end to end in time order. At each point between two windows, the mean
vector of the K windows before is compared with the mean of the K windows after (a TextTiling-style
depth score: 1 - cosine). Peaks well above the class's typical score are suggested. Suggestions are
only shown; a boundary exists only after the user confirms it (units.add_boundary).
"""
from pathlib import Path

import numpy as np

from lecture_recorder.config import Config
from lecture_recorder.units import ordered_units

K = 3  # windows on each side (~90 s of speech at 30 s steps)
MIN_GAP_S = 120.0  # no suggestion within 2 min of another suggestion or an existing boundary
MAX_SUGGESTIONS = 8


def suggest_boundaries(class_dir: Path, cfg: Config, k: int = K, index=None) -> dict:
    """{'suggestions': [{'session', 't', 'score', 'at_lecture_start'}], 'not_indexed': [session names]}"""
    from lecture_recorder.search.index import SearchIndex

    class_dir = Path(class_dir)
    index = index or SearchIndex(class_dir.parent.parent, cfg)
    rows, m, missing = index.class_windows(class_dir.name)
    out = {"suggestions": [], "not_indexed": missing}
    if m is None or len(rows) < 2 * k + 1:
        return out
    c = np.cumsum(np.vstack([np.zeros((1, m.shape[1]), dtype=np.float64), m.astype(np.float64)]), axis=0)
    scores = np.full(len(rows), -1.0)
    for p in range(1, len(rows)):  # boundary just before window p
        lo, hi = max(0, p - k), min(len(rows), p + k)
        left, right = c[p] - c[lo], c[hi] - c[p]
        nl, nr = np.linalg.norm(left), np.linalg.norm(right)
        if nl and nr:
            scores[p] = 1.0 - float(left @ right / (nl * nr))
    valid = scores[scores >= 0]
    threshold = float(valid.mean() + valid.std())
    taken = [(u["start"]["session"], float(u["start"]["t"])) for u in ordered_units(class_dir)
             if u.get("start") and not u.get("orphan") and not u.get("moved")]
    first_session = rows[0]["session"]
    order = {s: i for i, s in enumerate(dict.fromkeys(r["session"] for r in rows))}

    def near(sess: str, t: float) -> bool:
        return any(s == sess and abs(t2 - t) < MIN_GAP_S for s, t2 in taken)

    for p in np.argsort(-scores, kind="stable"):
        p = int(p)
        if scores[p] < threshold or len(out["suggestions"]) >= MAX_SUGGESTIONS:
            break
        local_peak = scores[p] >= max(scores[max(1, p - 1):p + 2])
        r = rows[p]
        t = 0.0 if r["first"] else r["t"]  # a shift between lectures: the next lecture starts the unit
        if not local_peak or (r["session"] == first_session and t < 1.0) or near(r["session"], t):
            continue
        out["suggestions"].append({"session": r["session"], "t": round(t, 2), "score": round(float(scores[p]), 4),
                                   "at_lecture_start": bool(r["first"])})
        taken.append((r["session"], t))
    out["suggestions"].sort(key=lambda s: (order[s["session"]], s["t"]))
    return out
