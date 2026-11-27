"""Synthetic class-suggestion demo: three classes with a schedule, one earlier Zoom import, one Unsorted
lecture, one misfiled lecture, and one ambiguous case that must show no suggestion. No models, no GPU.

    .venv\\Scripts\\python.exe scripts\\demo\\demo_autosort.py --data-dir .cache\\autosort-demo\\data

Then start the app on that folder (launch.json "autosort-demo", port 8779). `.cache\\autosort-demo\\pick\\`
holds a file and a Zoom folder to try in the import picker. Re-running starts over; never point it at the real
data folder.
"""
import argparse
import random
import shutil
import sys
import wave
from pathlib import Path

import lecture_recorder  # noqa: F401  (offline lockdown)
from lecture_recorder import autosort as sort
from lecture_recorder import duplicates as dup
from lecture_recorder import library as lib
from lecture_recorder.recorder.session import write_json_atomic

Z = "+00:00"
EVERYDAY = ("the we is a so of and to in that it you this for are on with as be can what if have here at one two from "
            "because when they will not all there which about would their more some into these then see look "
            "example notice again remember number point important result".split())
TERMS = {
    "physics-101": "momentum velocity acceleration inertia torque friction newton kinetic potential energy joule force vector "
                   "gravity pendulum oscillation frequency wavelength amplitude collision trajectory projectile".split(),
    "chem-200": "molecule covalent ionic electron orbital valence reagent catalyst enzyme oxidation reduction titration "
                "molarity solvent solute equilibrium enthalpy entropy isomer polymer benzene aldehyde".split(),
    "history-210": "treaty empire monarchy parliament revolution colonial dynasty feudal peasant senate republic constitution "
                   "reformation crusade merchant guild conquest diplomacy archive chronicle treason abolition".split(),
}
SLOTS = {"physics-101": ("2026-09-01", "10:00"), "chem-200": ("2026-09-02", "14:00"), "history-210": ("2026-09-03", "09:00")}


def text(rng: random.Random, *courses: str, words: int = 900) -> list[str]:
    pool = [t for c in courses for t in TERMS[c]]
    return [rng.choice(pool) if rng.random() < 0.25 else rng.choice(EVERYDAY) for _ in range(words)]


def write_wav(path: Path, seconds: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(bytes([40, 0, 216, 255]) * (8000 * seconds))


def lecture(data: Path, slug: str, name: str, title: str, created: str, words: list[str] | None, source: dict, **meta) -> Path:
    d = data / "classes" / slug / "sessions" / name
    d.mkdir(parents=True)
    write_json_atomic(d / "meta.json", {"name": name, "title": title, "class": slug, "state": "stopped", "created": created,
                                        "duration_s": 3000, "source": source, **meta}, fsync=False)
    write_wav(d / "audio.wav", 2)
    if words:
        segs = [{"id": i, "start": i * 4, "end": i * 4 + 3, "speaker": "Speaker 1", "role": "Main", "words": [],
                 "text": " ".join(words[j:j + 12])} for i, j in enumerate(range(0, len(words), 12))]
        doc = {"speakers": [{"id": "Speaker 1", "role": "Main", "speech_s": 3000}], "segments": segs, "duration_s": 3000}
        write_json_atomic(d / "transcript.json", doc, fsync=False)
        write_json_atomic(d / "transcript.raw.json", doc, fsync=False)
        dup.sketch_session(d)
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, required=True)
    data = ap.parse_args().data_dir.resolve()
    if data.name != "data" or ".cache" not in data.parts:
        sys.exit("Use a folder named data under .cache (for example .cache\\autosort-demo\\data), never the real data folder.")
    shutil.rmtree(data, ignore_errors=True)
    shutil.rmtree(data.parent / "pick", ignore_errors=True)
    rng = random.Random(18)
    mic = {"kind": "mic", "device_name": "Laptop mic"}
    from datetime import date, timedelta

    for slug, name in (("physics-101", "Physics 101"), ("chem-200", "Chem 200"), ("history-210", "History 210")):
        lib.create_class(data, name)
        day, at = SLOTS[slug]
        for week in range(5):
            d = date.fromisoformat(day) + timedelta(days=7 * week)
            src = mic
            extra = {}
            if slug == "physics-101" and week == 3:  # an earlier Zoom import: its topic is known
                src, extra = {"kind": "import", "zoom": True, "file": "audio1.m4a", "bytes": 1000,
                              "folder": f"{d} 10.00.00 Mechanics Review 91234567890"}, {"date_from": "Zoom folder name"}
            lecture(data, slug, f"{d}_{at.replace(':', '-')}-00", f"{name} week {week + 1}", f"{d}T{at}:00{Z}", text(rng, slug), src, **extra)
    lib.ensure_unsorted(data)
    lecture(data, "unsorted", "2026-10-09_18-30-00", "Voice memo (it's Chem)", f"2026-10-09T18:30:00{Z}", text(rng, "chem-200"),
            {"kind": "import", "file": "recording.m4a", "bytes": 777}, date_from="file date")
    lecture(data, "unsorted", "2026-10-10_18-30-00", "Voice memo (half Chem, half Physics)", f"2026-10-10T18:30:00{Z}",
            text(rng, "chem-200", "physics-101"), {"kind": "import", "file": "recording 2.m4a", "bytes": 778}, date_from="file date")
    wrong = lecture(data, "physics-101", "2026-10-16_16-00-00", "Make-up class (really History)", f"2026-10-16T16:00:00{Z}",
                    text(rng, "history-210"), mic)
    write_wav(wrong / "audio.wav", 600)  # a long file, as the player holds it open while it streams

    pick = data.parent / "pick"
    zoom = pick / "2026-11-03 10.00.00 Physics 101 Lecture 91234567890"
    zoom.mkdir(parents=True)
    write_wav(pick / "2026-11-03 10.00.00 tuesday.wav", 1)
    write_wav(zoom / "audio1234.m4a", 1)
    print(f"Demo ready in {data}.")
    for c in lib.list_classes(data):
        for name, s in sorted(sort.for_class(data, c["slug"]).items()):
            print(f"  {c['slug']}/{name}: looks like {s['class_name']} ({s['reason']})")
    print(f"  import {pick / '2026-11-03 10.00.00 tuesday.wav'}: {sort.for_import(data, [{'name': '2026-11-03 10.00.00 tuesday.wav'}])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
