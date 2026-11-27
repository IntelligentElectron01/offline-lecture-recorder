"""Exam material: combined unit summaries + a key-terms glossary for a named exam scope.

Built only from the transcript text inside the exam's spans (words outside a mid-lecture boundary are
cut off using word timestamps), so nothing from out-of-scope units reaches the LLM. Runs as an "exam"
queue job under the GPU lock and the power/recording rule, with every summary guard (LocalLLM.chat:
num_ctx 8192, prompt <= 5,000 tokens, truncate:false, done_reason must be "stop"), sizes-only logs, and
the model unloaded at the end.

exams/<id>/exam.md          the material (never replaced once the user has edited it)
exams/<id>/exam.meta.json   state, sha256 of the last generated exam.md, per-request sizes
"""
import hashlib
import re
from pathlib import Path

from lecture_recorder.config import Config, load_config
from lecture_recorder.pipeline.gpu import gpu_job, settle_vram, vram_used_mb
from lecture_recorder.pipeline.outputs import EDITABLE, hms, write_text_atomic
from lecture_recorder.pipeline.progress import NULL
from lecture_recorder.recorder.session import now_iso, read_json, write_json_atomic
from lecture_recorder.search.text import segment_part
from lecture_recorder.summary.llm import LocalLLM, estimate_tokens
from lecture_recorder.summary.summarize import (
    GENERATED_MARK, SYSTEM, Line, _as_subsection_body, _budget, chunk_lines, clean_output, detailed_parts, map_parts,
    neutralize, reduce_parts,
)
from lecture_recorder.units import get_exam, scope_spans

EXAM_MD, EXAM_META = "exam.md", "exam.meta.json"
GLOSSARY_TASK = (
    "These are the key terms from the study notes of the lectures in one exam's scope.\n\n<notes>\n{text}\n</notes>\n\n"
    "Write a glossary in Markdown: one bullet per term, exactly '- **Term**: one-sentence definition', sorted "
    "alphabetically. Merge duplicates. Use only terms and meanings found in the notes. At most 60 terms."
)


class ExamMaterialEdited(FileExistsError):
    pass


# ------------------------------------------------------------------ scope -> text
def segments_in_span(doc: dict, start: float, end: float | None) -> list[dict]:
    """The parts of segments said in [start, end), by the same word-timing rule as search
    (search.text.token_times), so a segment split by a unit boundary lands on both sides alike."""
    return [p for s in doc.get("segments", []) if (p := segment_part(s, start, end)) is not None]


def unit_evidence(class_dir: Path, unit: dict) -> dict:
    """What the lecturer said in this unit's spans and the questions asked there (summarize.evidence),
    so question echoes are dropped from the unit's notes and its at-a-glance list."""
    from lecture_recorder.summary.summarize import doc_evidence

    segs = []
    for sp in unit["spans"]:
        doc = read_json(Path(class_dir) / "sessions" / sp["session"] / EDITABLE, None)
        segs += segments_in_span(doc, sp["start"], sp["end"]) if doc else []
    return doc_evidence({"segments": segs})


def unit_parts(class_dir: Path, unit: dict, llm: LocalLLM, ev: dict) -> list[dict]:
    """Part notes for a unit, in time order. A lecture that lies wholly in the unit and still has its
    generated summary.md contributes those notes (no new LLM work); any other span (a lecture split by
    a boundary, or one without a usable summary) is mapped from only its in-span transcript text."""
    from lecture_recorder.summary.session import SUMMARY_FILE, summary_kind

    parts = []
    for sp in unit["spans"]:
        d = Path(class_dir) / "sessions" / sp["session"]
        label = f"Lecture of {sp['session'][:10]}"
        if sp["start"] == 0 and sp["end"] is None and summary_kind(d) == "generated":
            reused = detailed_parts((d / SUMMARY_FILE).read_text(encoding="utf-8"))
            if reused:
                parts += [{**p, "notes": f"({label})\n{p['notes']}"} for p in reused]
                continue
        doc = read_json(d / EDITABLE, None)
        segs = segments_in_span(doc, sp["start"], sp["end"]) if doc else []
        if segs:
            parts += [{**p, "notes": f"({label})\n{p['notes']}"} for p in map_parts({"segments": segs}, llm, ev)]
    return parts


# ------------------------------------------------------------------ generation
def _key_terms(notes: str) -> str:
    m = re.search(r"(?ims)^#+\s*Key terms\s*$(.*?)(?=^#+\s|\Z)", notes)
    body = (m.group(1) if m else "").strip()
    return "" if body.lower() in ("", "none", "- none") else body


def glossary(llm: LocalLLM, term_lists: list[str], progress=NULL) -> str:
    text = "\n".join(t for t in term_lists if t)
    if not text.strip():
        return "_No key terms were found in the notes._"
    sys_n = SYSTEM.format(tag="notes")
    budget = _budget(llm, sys_n, GLOSSARY_TASK)
    parts = [g.text for g in chunk_lines([Line(0, 0, neutralize(t)) for t in text.splitlines() if t.strip()],
                                         budget, overlap=False)]
    entries: dict[str, str] = {}
    progress.stage("glossary", total=len(parts), unit="part")
    for i, part in enumerate(parts, 1):
        progress.update(i - 1)
        out = clean_output(llm.chat(sys_n, GLOSSARY_TASK.format(text=part), stage="glossary", part=i))
        for line in out.splitlines():
            m = re.match(r"\s*[-*]\s*\*\*(.+?)\*\*\s*[:–-]\s*(.+)", line)
            if m:
                entries.setdefault(m.group(1).strip().lower(), f"- **{m.group(1).strip()}**: {m.group(2).strip()}")
    return "\n".join(entries[k] for k in sorted(entries)) or "_No key terms were found in the notes._"


def render_exam_md(exam: dict, class_name: str, units: list[dict], results: list[dict], glossary_md: str,
                   created: str, model: str) -> str:
    scope = (", ".join(u["name"] for u in units) if exam.get("units")
             else f"lectures from {exam['from']} to {exam['to']}")
    out = [GENERATED_MARK, f"# Exam: {exam['name']}", "",
           f"_{class_name}. Scope: {scope}. Generated {created[:16].replace('T', ' ')} with {model} (local), only "
           "from the transcripts inside this scope. You can edit this file; the app never replaces it once edited._",
           ""]
    for u, r in zip(units, results):
        spans = "; ".join(f"{sp['session'][:16].replace('_', ' ')} {hms(sp['start'])}"
                          + (f"-{hms(sp['end'])}" if sp["end"] is not None else "-end") for sp in u["spans"])
        out += [f"## {u['name']}", "", f"_Lectures: {spans}_", ""]
        if r is None:
            out += ["_No transcribed lecture text in this unit yet._", ""]
            continue
        out += ["### At a glance", "", r["glance"], "", "### Detailed notes", ""]
        for i, p in enumerate(r["parts"], 1):
            out += [f"#### Part {i}", "", re.sub(r"(?m)^#### ", "##### ", _as_subsection_body(p["notes"])), ""]
    out += ["## Glossary", "", glossary_md, ""]
    return "\n".join(out).rstrip() + "\n"


def _scope_key(units: list[dict]) -> list:
    return [[u["id"], [[sp["session"], sp["start"], sp["end"]] for sp in u["spans"]]] for u in units]


def material_is_stale(d: Path, class_dir: Path, exam: dict) -> bool:
    """True if the exam's units now cover different time than when exam.md was generated (a boundary
    moved, a lecture was added or removed), so the material may hold text from outside the scope."""
    meta = read_json(Path(d) / EXAM_META, {}) or {}
    return meta.get("spans") is not None and meta["spans"] != _scope_key(scope_spans(class_dir, exam))


def material_state(d: Path) -> str:
    """'none' | 'generated' (exactly what the app last wrote) | 'user' (edited or not ours)."""
    p = Path(d) / EXAM_MD
    if not p.exists():
        return "none"
    meta = read_json(Path(d) / EXAM_META, {}) or {}
    return "generated" if hashlib.sha256(p.read_bytes()).hexdigest() == meta.get("sha256") else "user"


def generate_exam_material(exam_dir: Path, config: Config | None = None, llm: LocalLLM | None = None,
                           gpu_wait_s: float = 3600, progress=NULL) -> dict:
    cfg = config or load_config()
    d = Path(exam_dir)
    class_dir = d.parent.parent
    exam = get_exam(class_dir, d.name)
    if exam is None:
        raise FileNotFoundError(f"no exam {d.name!r} in {class_dir.name}")
    d.mkdir(parents=True, exist_ok=True)
    meta_path = d / EXAM_META
    previous = read_json(meta_path, {}) or {}
    if material_state(d) == "user":
        info = {**previous, "state": "skipped", "checked": now_iso(),
                "warning": "exam.md was edited, so it was not replaced. Rename or remove it to generate it again."}
        write_json_atomic(meta_path, info)
        return info
    units = scope_spans(class_dir, exam)
    llm = llm or LocalLLM.from_config(cfg.llm)
    info = {"state": "running", "started": now_iso(), "model": llm.model, "num_ctx": llm.num_ctx,
            "max_prompt_tokens": llm.max_prompt_tokens, "units": [u["id"] for u in units], "vram_mb": {},
            "sha256": previous.get("sha256"), "spans": previous.get("spans")}
    write_json_atomic(meta_path, info)
    try:
        results = []
        progress.stage("gpu_wait")
        with gpu_job(cfg.home, wait_s=gpu_wait_s):
            llm.release_models((cfg.llm.summary_model, cfg.llm.embedding_model))  # left by a killed job
            info["vram_mb"]["before"] = vram_used_mb()
            try:
                progress.stage("exam", total=len(units), unit="unit")
                for k, u in enumerate(units):
                    progress.update(k)
                    ev = unit_evidence(class_dir, u)
                    parts = unit_parts(class_dir, u, llm, ev)
                    results.append(reduce_parts(parts, llm, title=u["name"], ev=ev) if parts else None)
                gloss = glossary(llm, [_key_terms(p["notes"]) for r in results if r for p in r["parts"]], progress)
            finally:
                info["unloaded"] = llm.unload()
                info["vram_mb"]["after_unload"] = settle_vram(info["vram_mb"]["before"])
        class_name = (read_json(class_dir / "class.json", {}) or {}).get("name") or class_dir.name
        text = render_exam_md(exam, class_name, units, results, gloss, now_iso(), llm.model)
        if material_state(d) == "user":  # edited while we worked: the user's version wins
            raise ExamMaterialEdited("exam.md changed while the material was being generated; it was not replaced")
        write_text_atomic(d / EXAM_MD, text)
        info.update(state="done", sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), spans=_scope_key(units),
                    chars=len(text), est_tokens=estimate_tokens(text))
    except Exception as e:
        info.update(state="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        info["finished"] = now_iso()
        info["requests"] = llm.requests  # sizes only
        write_json_atomic(meta_path, info)
        llm.close()
    return info
