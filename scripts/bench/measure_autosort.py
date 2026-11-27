"""Class-suggestion numbers (synthetic text; CPU; no GPU, no user data).

    .venv\\Scripts\\python.exe scripts\\bench\\measure_autosort.py

Lectures are drawn the way speech is: most tokens from one everyday vocabulary (a Zipf curve shared by every
class), the rest from the course's own terms. Prints the content scores for a lecture against its own class and
against other classes (for CONTENT_MIN and CONTENT_LEAD), how often a suggestion would be right, wrong or
absent, and the time per lecture in a 6-class, 40-lectures-per-class library at 90 minutes of speech.
"""
import random
import sys
import tempfile
import time
from pathlib import Path

import lecture_recorder  # noqa: F401  (offline lockdown)
from lecture_recorder import autosort as sort
from lecture_recorder import duplicates as dup
from lecture_recorder import library as lib
from lecture_recorder.pipeline import queue as q
from lecture_recorder.recorder.session import write_json_atomic

WORDS_90_MIN = 13_500
EVERYDAY = 3000


def pseudo_word(rng: random.Random) -> str:
    return "".join(rng.choice("bcdfgklmnprstv") + rng.choice("aeiou") for _ in range(rng.randint(2, 4)))


def world(seed: int, classes: int, related: bool = False):
    """(everyday words with Zipf weights, each class's own terms). `related`: the classes share half their terms."""
    rng = random.Random(seed)
    everyday = [pseudo_word(rng) for _ in range(EVERYDAY)]
    weights = [1 / (i + 1) for i in range(EVERYDAY)]
    shared = [pseudo_word(rng) for _ in range(300)]
    return everyday, weights, [(shared if related else []) + [pseudo_word(rng) for _ in range(300 if related else 600)]
                               for _ in range(classes)]


def speech(rng: random.Random, w, course: int, words: int, topic_share: float = 0.2, also: int | None = None) -> list[str]:
    everyday, weights, terms = w
    topic = rng.sample(terms[course], 150)  # one lecture uses part of the course's terms
    if also is not None:  # half and half: two courses in one lecture
        topic = topic[:75] + rng.sample(terms[also], 75)
    n = int(words * topic_share)
    return rng.choices(everyday, weights, k=words - n) + rng.choices(topic, k=n)


def terms(words: list[str]) -> frozenset[str]:
    return frozenset(dup.top_terms(words))


def scores(topic_share: float, related: bool, words: int, also: int | None = None) -> tuple[list[float], list[float]]:
    own, other = [], []
    for seed in range(12):
        rng = random.Random(seed)
        w = world(seed, 4, related)
        classes = {f"c{i}": {"name": f"c{i}", "words": set(), "lectures": [],
                             "sample": [(f"l{k}", terms(speech(rng, w, i, words, topic_share))) for k in range(sort.CONTENT_SAMPLE)]}
                   for i in range(4)}
        for _ in range(3):
            got = sort.content_scores(classes, terms(speech(rng, w, 0, words, topic_share, also)))
            own.append(got["c0"])
            other.append(max(v for k, v in got.items() if k != "c0"))
    return own, other


def timing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data"
        rng = random.Random(1)
        w = world(1, 6)
        for c in range(6):
            lib.create_class(data, f"Class {c}")
            for k in range(40):
                d = data / "classes" / f"class-{c}" / "sessions" / f"2026-{9 + k // 25:02d}-{k % 25 + 1:02d}_{9 + c:02d}-00-00"
                d.mkdir(parents=True)
                (data / "queue").mkdir(exist_ok=True)
                write_json_atomic(d / "meta.json", {"created": f"{d.name[:10]}T{9 + c:02d}:00:00+00:00", "class": f"class-{c}",
                                                    "source": {"kind": "mic"}, "duration_s": 5400}, fsync=False)
                doc = {"segments": [{"id": 0, "text": " ".join(speech(rng, w, c, WORDS_90_MIN))}]}
                write_json_atomic(d / "transcript.json", doc, fsync=False)
                write_json_atomic(d / "transcript.raw.json", doc, fsync=False)
                dup.sketch_session(d)
                for kind in (q.TRANSCRIBE, q.SUMMARIZE, q.EMBED):  # finished jobs stay on file
                    job_id = f"{c}-{k}-{kind}"
                    write_json_atomic(data / "queue" / f"{job_id}.json", {
                        "id": job_id, "session": str(d.resolve()), "kind": kind, "state": "done",
                        "created": "2026-10-01T10:00:00+00:00"}, fsync=False)
        last = sorted((data / "classes" / "class-0" / "sessions").iterdir())[-1]

        def class_page():  # what the page's request does (web/app.py class_info)
            states = lib._job_states(data)
            dup.for_class(data, "class-0")
            sort.for_class(data, "class-0", states=states)
            lib.list_sessions(data, "class-0", states)

        def end_of_transcription():
            dup.after_transcribe(data, last)
            sort.after_transcribe(data)

        for label, fn in (("one lecture's suggestion", lambda: sort.for_class(data, "class-0", last.name)),
                          ("suggestions for a whole class", lambda: sort.for_class(data, "class-0")),
                          ("the class page's request", class_page),
                          ("end-of-transcription step", end_of_transcription),
                          ("import picker", lambda: sort.for_import(data, [{"name": "2026-09-08 09.00.00.m4a"}]))):
            runs = []
            for _ in range(5):
                t0 = time.perf_counter()
                fn()
                runs.append(time.perf_counter() - t0)
            print(f"  {label}: first {runs[0]:.3f} s, then {min(runs[1:]):.3f}-{max(runs[1:]):.3f} s")


def report(label: str, own: list[float], other: list[float]) -> None:
    def hits(a, b):
        return sum(x >= sort.CONTENT_MIN and x >= sort.CONTENT_LEAD * y for x, y in zip(a, b))
    print(f"  {label}: own {min(own):.3f}-{max(own):.3f}, other {min(other):.3f}-{max(other):.3f}, multiple "
          f"{min(x / y for x, y in zip(own, other)):.2f}-{max(x / y for x, y in zip(own, other)):.2f}; "
          f"suggested {hits(own, other)}/{len(own)}, wrong {hits(other, own)}")


def main() -> int:
    print("content score, a lecture against its own class / the best other class / own as a multiple of other")
    for share, related, words, label in ((0.2, False, WORDS_90_MIN, "90 min, 20% course terms, unrelated classes"),
                                         (0.1, False, WORDS_90_MIN, "90 min, 10% course terms, unrelated classes"),
                                         (0.2, True, WORDS_90_MIN, "90 min, 20% course terms, classes share half their terms"),
                                         (0.1, True, WORDS_90_MIN, "90 min, 10% course terms, classes share half their terms"),
                                         (0.2, False, 4_500, "30 min, 20% course terms, unrelated classes")):
        own, other = scores(share, related, words)
        report(label, own, other)
    # half and half: neither class may be suggested, so here own and other are just the two classes
    a, b = scores(0.2, False, WORDS_90_MIN, also=1)
    report("90 min, half this class, half another (should suggest nothing)", a, b)
    print("time per call, 6 classes of 40 lectures:")
    timing()
    return 0


if __name__ == "__main__":
    sys.exit(main())
