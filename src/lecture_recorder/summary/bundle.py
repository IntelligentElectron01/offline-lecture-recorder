"""The optional "bundle for another AI": one Markdown file the user downloads and may
give to a paid model themselves. Built locally on request; the app never sends it anywhere.

Classmates did not agree to have their words shared with an outside service, so by default the bundle
holds only the lecturer's (Main speaker) text and the summaries, with any "Questions from students"
sections removed. Student (Audience) speech is included only if the user ticks the opt-in.
Only what is inside the exam's scope is included: a lecture's own summary only if the whole lecture is
in scope; transcript text only inside the scope's spans.
"""
import re
from pathlib import Path

from lecture_recorder.pipeline.outputs import EDITABLE, hms
from lecture_recorder.recorder.session import read_json
from lecture_recorder.summary.exam import EXAM_MD, material_is_stale, segments_in_span
from lecture_recorder.summary.summarize import GENERATED_MARK, ROLE_NAMES
from lecture_recorder.units import exam_dir, scope_spans

# Any heading that names student questions (the notes use "Questions from students"; models vary).
STUDENT_SECTION = re.compile(
    r"(?ims)^#{1,6}[^\n]*(question[^\n]*student|student[^\n]*question)[^\n]*$.*?(?=^#{1,6}\s|\Z)")
AUDIENCE_NOTE = ("Student (Audience) speech is not included, and the summaries' sections on student questions are "
                 "removed: classmates did not agree to have their words shared with an outside service. (Summaries "
                 "may still mention a point a student raised.)")


def _clean(md: str, audience: bool) -> str:
    md = md.replace(GENERATED_MARK, "").strip()
    md = re.sub(r"(?m)^(#{1,5})(\s)", r"#\1\2", md)  # nest under the bundle's own headings
    return md if audience else STUDENT_SECTION.sub("", md).strip()


def build_bundle(class_dir: Path, exam: dict, include_audience: bool = False) -> str:
    class_dir = Path(class_dir)
    class_name = (read_json(class_dir / "class.json", {}) or {}).get("name") or class_dir.name
    units = scope_spans(class_dir, exam)
    out = [f"# {exam['name']} ({class_name}): study bundle", "",
           "Made locally by Lecture Recorder, for you to give to an AI service yourself if you choose. "
           "It contains only lectures inside this exam's scope.", "",
           f"_{'Student (Audience) speech is included at your request.' if include_audience else AUDIENCE_NOTE}_", ""]
    d = exam_dir(class_dir, exam["id"])
    material = d / EXAM_MD
    if material.exists() and material_is_stale(d, class_dir, exam):
        out += ["_The exam's study material is left out: its units changed since it was generated._", ""]
    elif material.exists():
        out += ["## Exam study material", "", _clean(material.read_text(encoding="utf-8"), include_audience), ""]
    for u in units:
        out += [f"## {u['name']}", ""]
        for sp in u["spans"]:
            d = class_dir / "sessions" / sp["session"]
            meta = read_json(d / "meta.json", {}) or {}
            title = meta.get("title") or sp["session"]
            end = hms(sp["end"]) if sp["end"] is not None else "end"
            out += [f"### {title} ({sp['session'][:10]}, {hms(sp['start'])} to {end})", ""]
            whole = sp["start"] == 0 and sp["end"] is None
            summary = d / "summary.md"
            if whole and summary.exists() and summary.read_text(encoding="utf-8").startswith(GENERATED_MARK):
                out += ["#### Lecture summary", "", _clean(summary.read_text(encoding="utf-8"), include_audience), ""]
            doc = read_json(d / EDITABLE, None)
            if not doc:
                out += ["_Not transcribed yet._", ""]
                continue
            lines = [f"[{hms(s['start'])}] {ROLE_NAMES.get(s.get('role'), 'Speaker')}: {' '.join(s['text'].split())}"
                     for s in segments_in_span(doc, sp["start"], sp["end"])
                     if include_audience or s.get("role") != "Audience"]
            out += ["#### Transcript" + ("" if include_audience else " (lecturer only)"), "", *lines, ""]
    return "\n".join(out).rstrip() + "\n"
