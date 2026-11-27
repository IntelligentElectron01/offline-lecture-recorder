"""Text normalization shared by indexing and querying (exact search).

A token is a run of letters/digits, with inner apostrophes joined ("don't" -> "dont"). Case,
accents-insensitive casefold and all other punctuation are ignored, so "Purple Elephant-Theorem!"
and "purple elephant theorem" are the same three tokens. User input is only ever turned into these
tokens, so nothing typed can reach FTS5 or SQL as syntax.
"""
import re
import unicodedata

TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")


def norm_token(raw: str) -> str:
    t = unicodedata.normalize("NFKD", raw.casefold())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[\W_]", "", t)  # NFKD can add separators ("½" -> "1⁄2"); keep letters/digits only


def token_spans(text: str) -> list[tuple[str, int, int]]:
    """[(normalized token, char start, char end)] in the original text."""
    out = []
    for m in TOKEN_RE.finditer(text or ""):
        t = norm_token(m.group())
        if t:
            out.append((t, m.start(), m.end()))
    return out


def tokens(text: str) -> list[str]:
    return [t for t, _, _ in token_spans(text)]


def token_times(seg: dict, seg_tokens: list[str] | None = None) -> list[float]:
    """When each token of a segment was said, rounded to 0.01 s: the word timestamps while the words
    still spell the segment's text, else linearly interpolated over the segment (edited text).
    The one rule for "which side of a unit boundary is this word on" (search, material, bundle)."""
    seg_tokens = tokens(seg.get("text", "")) if seg_tokens is None else seg_tokens
    start, end = float(seg["start"]), float(seg["end"])
    words = seg.get("words") or []
    flat, times = [], []
    for w in words:
        for t in tokens(w.get("w", "")):
            flat.append(t)
            times.append(float(w.get("start", start)))
    if not (words and flat == seg_tokens):
        n = len(seg_tokens)
        times = [start + (end - start) * i / n for i in range(n)]
    return [round(t, 2) for t in times]


def segment_part(seg: dict, lo: float, hi: float | None) -> dict | None:
    """The part of a segment whose tokens were said in [lo, hi), with its own text and times, or None."""
    text = str(seg.get("text", ""))
    spans = token_spans(text)
    times = token_times(seg, [t for t, _, _ in spans])
    keep = [i for i, t in enumerate(times) if t >= lo and (hi is None or t < hi)]
    if not keep:
        return None
    if len(keep) == len(times):
        return seg
    a, b = keep[0], keep[-1]  # times never decrease within a segment, so the kept tokens are contiguous
    end = times[b + 1] if b + 1 < len(times) else float(seg["end"])
    words = [w for w in seg.get("words") or [] if float(w.get("start", 0)) >= lo and (hi is None or float(w["start"]) < hi)]
    return {**seg, "start": times[a], "end": end, "text": text[spans[a][1]:spans[b][2]], "words": words}


def fts_phrase(query_tokens: list[str]) -> str:
    """An FTS5 phrase query built only from normalized tokens (letters/digits), always quoted."""
    if not query_tokens or any(not re.fullmatch(r"[^\W_]+", t) for t in query_tokens):
        raise ValueError("query tokens must be non-empty letter/digit runs")
    return '"' + " ".join(query_tokens) + '"'


# ------------------------------------------------------------------ test helper
STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from how i in into is it its of on or that the their them "
    "then there these they this to was we what when where which who why will with you your".split())


def stem(word: str) -> str:
    """Very small suffix stripper, enough to tell 'producing'/'production'/'produce' apart from other words."""
    w = norm_token(word)
    for suf in ("ational", "ations", "ation", "ions", "ion", "ings", "ing", "ies", "ied", "edly", "ed", "es",
                "ers", "er", "ally", "al", "ly", "ive", "s", "e"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def content_words(text: str) -> set[str]:
    return {stem(t) for t in tokens(text) if t not in STOPWORDS}
