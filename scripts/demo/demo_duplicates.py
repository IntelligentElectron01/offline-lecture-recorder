"""Synthetic duplicates demo: one class with a re-imported file, a mic + Zoom pair of the same lecture, a
content-only match, and a near miss that must NOT be flagged. No models, no GPU; the audio is a short tone.

    .venv\\Scripts\\python.exe scripts\\demo\\demo_duplicates.py --data-dir .cache\\dupes-demo\\data

Then start the app on that folder (launch.json "dupes-demo", port 8778). `.cache\\dupes-demo\\Week 3.wav`
is the first lecture's file, for trying the import warning. Re-running starts over; never point it at the real
data folder.
"""
import argparse
import hashlib
import random
import shutil
import sys
import wave
from pathlib import Path

import lecture_recorder  # noqa: F401  (offline lockdown)
from lecture_recorder import duplicates as dup
from lecture_recorder.recorder.session import write_json_atomic

SLUG = "demo-dupes"
MIC, ZOOM = {"kind": "mic", "device_name": "Laptop mic"}, {"kind": "import", "zoom": True, "file": "audio1.m4a", "bytes": 912345}
Z = "+00:00"


def script(seed: int, n: int = 1500) -> list[str]:
    rng = random.Random(seed)
    return [f"term{rng.randrange(600)}" for _ in range(n)]


def garble(words: list[str], seed: int, rate: float = 0.05) -> list[str]:
    rng = random.Random(seed)
    return [f"other{rng.randrange(600)}" if rng.random() < rate else w for w in words]


def write_wav(path: Path) -> bytes:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(bytes([40, 0, 216, 255]) * 8000)
    return path.read_bytes()


def session(c: Path, name: str, title: str, created: str, seconds: float, source: dict, words: list[str] | None = None,
            **meta) -> None:
    d = c / "sessions" / name
    d.mkdir(parents=True)
    write_json_atomic(d / "meta.json", {"name": name, "title": title, "class": SLUG, "state": "stopped", "created": created,
                                        "duration_s": seconds, "source": source, **meta}, fsync=False)
    if words:
        step = 12
        segs = [{"id": i, "start": i * 4, "end": i * 4 + 3, "speaker": "Speaker 1", "role": "Main", "words": [],
                 "text": " ".join(words[j:j + step])} for i, j in enumerate(range(0, len(words), step))]
        doc = {"speakers": [{"id": "Speaker 1", "role": "Main", "speech_s": seconds}], "segments": segs, "duration_s": seconds}
        write_json_atomic(d / "transcript.json", doc, fsync=False)
        write_json_atomic(d / "transcript.raw.json", doc, fsync=False)
        dup.sketch_session(d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, required=True)
    data = ap.parse_args().data_dir.resolve()
    if data.name != "data" or ".cache" not in data.parts:
        sys.exit("Use a folder named data under .cache (for example .cache\\dupes-demo\\data), never the real data folder.")
    shutil.rmtree(data, ignore_errors=True)
    c = data / "classes" / SLUG
    c.mkdir(parents=True)
    write_json_atomic(c / "class.json", {"name": "History 210 (demo)", "slug": SLUG})
    body = write_wav(data.parent / "Week 3.wav")
    file = {"kind": "import", "file": "Week 3.wav", "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}

    # 1. the same file imported twice ("Import anyway")
    for name, title, created in (("2026-09-02_10-00-00", "Week 3", f"2026-09-02T10:00:00{Z}"),
                                 ("2026-09-02_10-00-00-1", "Week 3 (again)", f"2026-09-02T10:00:00{Z}")):
        d = c / "sessions" / name
        session(c, name, title, created, 2.0, file, date_from="file date")
        shutil.copyfile(data.parent / "Week 3.wav", d / "audio.wav")
    # 2. the same lecture from the mic and from Zoom
    session(c, "2026-09-09_10-00-00", "Week 4 (mic)", f"2026-09-09T10:00:00{Z}", 3000, MIC, garble(script(4), 41))
    session(c, "2026-09-09_10-02-00", "Week 4 (Zoom)", f"2026-09-09T10:02:00{Z}", 2940, ZOOM, date_from="Zoom folder name")
    # 3. same content, nothing else in common
    session(c, "2026-09-16_10-00-00", "Week 5", f"2026-09-16T10:00:00{Z}", 3000, MIC, garble(script(5), 51))
    session(c, "2026-09-23_10-00-00", "Week 5 from another device", f"2026-09-23T10:00:00{Z}", 2800,
            {"kind": "import", "file": "recording.m4a", "bytes": 777}, garble(script(5), 52), date_from="file metadata")
    # 4. near miss: starts during Week 6 (20% of it) and shares a short recap. Must not be flagged.
    week6, week7 = script(6), script(7)
    session(c, "2026-09-30_10-00-00", "Week 6", f"2026-09-30T10:00:00{Z}", 3000, MIC, garble(week6, 61))
    session(c, "2026-09-30_10-40-00", "Week 7 (make-up class)", f"2026-09-30T10:40:00{Z}", 3000, MIC,
            garble(week6[:120] + week7[120:], 71))
    print(f"Demo class '{SLUG}' ready in {data}.")
    for name, found in sorted(dup.for_class(data, SLUG).items()):
        print(f"  {name}: {'; '.join(m['other']['title'] + ' (' + ', '.join(m['labels']) + ')' for m in found)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
