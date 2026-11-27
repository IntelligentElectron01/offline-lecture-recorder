"""Duplicate-check numbers (synthetic text; CPU; no GPU, no user data).

    .venv\\Scripts\\python.exe scripts\\bench\\measure_duplicates.py

Same-lecture pairs model a mic recording and a Zoom import transcribed separately (independent word errors).
Different-lecture pairs share a course's vocabulary and spoken filler, the worst case for shingles; the recap
pairs also repeat part of an earlier lecture word for word. Prints the score ranges for `SIMILAR`, and the time
for one lecture's check in a 40-lecture class.
"""
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import lecture_recorder  # noqa: F401  (offline lockdown)
from lecture_recorder import duplicates as dup
from lecture_recorder import library as lib
from lecture_recorder.pipeline import queue as q
from lecture_recorder.recorder.session import write_json_atomic

WORDS_90_MIN = 13_500
FILLER = ("so what we are going to do is now if you look at this you can see that and this is the part where "
          "we want to think about how the idea of it works okay right let me show you one more example here").split()


def course(rng: random.Random, n: int = 900) -> list[str]:
    return [f"{rng.choice('bcdfgklmnprst')}{rng.choice('aeiou')}{rng.choice('lmnrst')}{i}" for i in range(n)]


def lecture(rng: random.Random, vocab: list[str], words: int) -> list[str]:
    topic = rng.sample(vocab, 120)
    out: list[str] = []
    while len(out) < words:
        out += [rng.choice(FILLER) for _ in range(rng.randint(3, 7))]
        out += rng.choices(topic, k=rng.randint(2, 4))
    return out[:words]


def garble(rng: random.Random, words: list[str], rate: float, vocab: list[str]) -> list[str]:
    """Independent speech-to-text errors: substitutions, deletions and insertions."""
    out = []
    for w in words:
        r = rng.random()
        if r >= rate:
            out.append(w)
        elif r < rate * 0.6:
            out.append(rng.choice(vocab))
        elif r < rate * 0.8:
            out += [w, rng.choice(FILLER)]
    return out


def score(a: list[str], b: list[str]) -> float:
    return float((np.array(dup.minhash(a), np.uint64) == np.array(dup.minhash(b), np.uint64)).mean())


def scores() -> None:
    rng = random.Random(17)
    rows: dict[str, list[float]] = {}
    for _ in range(12):
        vocab = course(rng)
        for words in (4_500, 13_500):
            base = lecture(rng, vocab, words)
            for rate in (0.05, 0.10, 0.15):
                rows.setdefault(f"same lecture, {rate:.0%} errors in each copy", []).append(
                    score(garble(rng, base, rate, vocab), garble(rng, base, rate, vocab)))
            rows.setdefault("same class, different lecture", []).append(
                score(garble(rng, base, 0.1, vocab), garble(rng, lecture(rng, vocab, words), 0.1, vocab)))
            for share in (0.10, 0.20):
                cut = int(words * share)
                recap = base[:cut] + lecture(rng, vocab, words - cut)
                rows.setdefault(f"{share:.0%} recap of an earlier lecture", []).append(
                    score(garble(rng, base, 0.1, vocab), garble(rng, recap, 0.1, vocab)))
    print(f"shingle = {dup.SHINGLE_WORDS} words, {dup.PERMS} hashes, threshold {dup.SIMILAR}")
    for name, v in rows.items():
        print(f"  {name:42} min {min(v):.3f}  max {max(v):.3f}")


def speed() -> None:
    rng = random.Random(3)
    vocab = course(rng)
    with tempfile.TemporaryDirectory(dir=".cache") as tmp:
        data = Path(tmp).resolve()
        lib.create_class(data, "Timing")
        root = data / "classes" / "timing" / "sessions"
        for i in range(40):
            day = f"2026-{9 + i // 28:02d}-{1 + i % 28:02d}"  # one a day: this is the check's time, not a duplicate test
            d = root / f"{day}_09-00-00"
            d.mkdir(parents=True)
            write_json_atomic(d / "meta.json", {"title": f"L{i}", "created": f"{day}T09:00:00+00:00",
                                                "duration_s": 5400, "source": {"kind": "mic"}}, fsync=False)
            words = lecture(rng, vocab, WORDS_90_MIN)
            step = 12
            write_json_atomic(d / "transcript.json", {"segments": [{"text": " ".join(words[j:j + step])}
                                                                   for j in range(0, len(words), step)]}, fsync=False)
            (d / "transcript.raw.json").write_text("{}", encoding="utf-8")
        start = time.perf_counter()
        dup.sketch_scope(data, "timing")
        first = time.perf_counter() - start
        start = time.perf_counter()
        (d / dup.SKETCH_FILE).unlink()
        dup.after_transcribe(data, d)
        one = time.perf_counter() - start
        start = time.perf_counter()
        n = len(dup.for_class(data, "timing"))
        read = time.perf_counter() - start
        for c in range(9):  # the rest of a 400-lecture library, all imported, with a queue folder in use
            lib.create_class(data, f"Other {c}")
            for i in range(40):
                d = data / "classes" / f"other-{c}" / "sessions" / f"2026-01-{1 + i % 28:02d}_09-{i:02d}-00"
                d.mkdir(parents=True)
                write_json_atomic(d / "meta.json", {"title": "x", "created": "2026-01-01T09:00:00+00:00", "duration_s": 60,
                                                    "source": {"kind": "import", "file": f"{c}-{i}.m4a", "bytes": i, "sha256": f"{c}-{i}"}}, fsync=False)
                q.enqueue(data, d)
        for slug in ("timing", "other-0"):
            dup.for_class(data, slug)  # the first read of these new files is mostly the virus scanner
        start = time.perf_counter()
        dup.for_class(data, "timing")
        library = time.perf_counter() - start
        start = time.perf_counter()
        dup.for_class(data, "other-0")
        imports = time.perf_counter() - start
    print(f"same, in a 400-lecture library (files read before): mic-only class {library:.3f} s, class of imports {imports:.3f} s")
    print(f"first sketches, 40 lectures of 90 min: {first:.2f} s ({first / 40:.3f} s each)")
    print(f"end-of-transcription step (one new lecture, 39 up to date): {one:.3f} s")
    print(f"class page's check, 40 lectures: {read:.3f} s ({n} flagged)")


if __name__ == "__main__":
    scores()
    speed()
    sys.exit(0)
