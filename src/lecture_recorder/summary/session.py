"""Summarize one session folder: transcript.json -> summary.md (+ summary.meta.json with sizes only).

summary.md is written only if it is missing or is still the app's untouched placeholder. Anything
else (a summary the user wrote or edited, or an earlier generated one) is never replaced: the job is
skipped with a warning before any LLM work, and checked again right before writing.

The LLM runs under the same GPU lock as Whisper/pyannote, and the model is unloaded (keep_alive=0)
when the job ends, even if it failed.
"""
import logging
import time
from pathlib import Path

from lecture_recorder.config import Config, load_config
from lecture_recorder.pipeline.gpu import gpu_job, settle_vram, vram_used_mb
from lecture_recorder.pipeline.outputs import EDITABLE, write_text_atomic
from lecture_recorder.pipeline.progress import NULL
from lecture_recorder.recorder.session import BOOKMARKS_FILE, META_FILE, now_iso, read_json, write_json_atomic
from lecture_recorder.summary.llm import LocalLLM
from lecture_recorder.summary.summarize import render_summary_md, summarize_doc

SUMMARY_FILE = "summary.md"
SUMMARY_META = "summary.meta.json"
PLACEHOLDER_MARK = "<!-- lecture-recorder:placeholder -->"
log = logging.getLogger("lecture_recorder.summary")


class SummaryExists(FileExistsError):
    pass


def placeholder_text(name: str) -> str:
    return f"{PLACEHOLDER_MARK}\n# Summary: {name}\n\nThe summary for this lecture has not been generated yet.\n"


def summary_kind(d: Path) -> str:
    """'none' | 'placeholder' (the app's untouched placeholder) | 'generated' | 'user' (anything else)."""
    p = Path(d) / SUMMARY_FILE
    if not p.exists():
        return "none"
    text = p.read_text(encoding="utf-8", errors="replace")
    if text == placeholder_text(Path(d).name):
        return "placeholder"
    from lecture_recorder.summary.summarize import GENERATED_MARK

    return "generated" if text.startswith(GENERATED_MARK) else "user"


def replaceable(d: Path) -> bool:
    return summary_kind(d) in ("none", "placeholder")


def summarize_session(session_dir: Path, config: Config | None = None, llm: LocalLLM | None = None,
                      gpu_wait_s: float = 3600, progress=NULL) -> dict:
    cfg = config or load_config()
    d = Path(session_dir)
    doc = read_json(d / EDITABLE, None)
    if doc is None:
        raise FileNotFoundError(f"{d.name} has no transcript.json yet")
    meta_path = d / SUMMARY_META
    if not replaceable(d):
        why = "is an earlier generated summary" if summary_kind(d) == "generated" else "was written or edited by you"
        info = {"state": "skipped", "created": now_iso(),
                "warning": f"summary.md {why}, so it was not replaced. Rename or remove it to generate a new summary."}
        write_json_atomic(meta_path, info)
        log.warning("%s: %s", d.name, info["warning"])
        return info

    llm = llm or LocalLLM.from_config(cfg.llm)
    title = (read_json(d / META_FILE, {}) or {}).get("title") or d.name
    info = {"state": "running", "started": now_iso(), "model": llm.model, "num_ctx": llm.num_ctx,
            "max_prompt_tokens": llm.max_prompt_tokens, "num_predict": llm.num_predict, "vram_mb": {}}
    write_json_atomic(meta_path, info)  # a crash mid-job leaves "running", so recovery re-queues it
    try:
        progress.stage("gpu_wait")
        with gpu_job(cfg.home, wait_s=gpu_wait_s):  # never alongside Whisper/pyannote or another LLM job
            llm.release_models((cfg.llm.summary_model, cfg.llm.embedding_model))  # left by a killed job
            info["vram_mb"]["before"] = vram_used_mb()
            try:
                result = summarize_doc(doc, llm, title=title, progress=progress)
            finally:
                info["unloaded"] = llm.unload()  # keep_alive=0, even after an error
                info["vram_mb"]["after_unload"] = settle_vram(info["vram_mb"]["before"])
        text = render_summary_md(result, title, now_iso(), llm.model, read_json(d / BOOKMARKS_FILE, []) or [], doc)
        if not replaceable(d):  # the user saved a summary while we were working: theirs wins
            raise SummaryExists("summary.md changed while the summary was being generated; it was not replaced")
        write_text_atomic(d / SUMMARY_FILE, text)
        info.update(state="done", parts=len(result["parts"]), reduce_levels=result["reduce_levels"])
    except Exception as e:
        info.update(state="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        info["finished"] = now_iso()
        info["requests"] = llm.requests  # sizes only: no prompt or response text
        write_json_atomic(meta_path, info)
        llm.close()
    return info
