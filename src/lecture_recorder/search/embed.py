"""Concept-search embeddings: overlapping transcript windows -> vectors, stored in the session folder.

<session>/embeddings.npy   float32 [windows x dims], L2-normalized
<session>/embeddings.json  windows (start, end, text), the transcript text hash they came from, the
                           model and window settings, the .npy file's sha256, and per-request sizes
                           (no text in the log). Written last; "done" + matching hash = up to date.

The files are derived data (the search index loads them, so rebuilding the index needs no GPU); an
edit to the transcript text changes the hash, which marks them stale until the next embed job.
Embedding runs as a queue job under the GPU lock, and the model is unloaded (keep_alive=0) at the end.
A query at search time takes the GPU lock without waiting: if a job holds it, GpuBusy is raised.
"""
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

from lecture_recorder.config import Config, load_config
from lecture_recorder.pipeline.gpu import gpu_job, settle_vram, vram_used_mb
from lecture_recorder.pipeline.outputs import EDITABLE
from lecture_recorder.pipeline.progress import NULL
from lecture_recorder.recorder.session import now_iso, read_json, write_json_atomic
from lecture_recorder.summary.llm import CHARS_PER_TOKEN, LocalLLM, embedder_from_config

EMB_NPY, EMB_META = "embeddings.npy", "embeddings.json"
# Qwen3-Embedding is trained with an instruction on the query side only; documents are embedded as-is.
QUERY_PREFIX = "Instruct: Given a search query, retrieve passages of a lecture transcript that discuss it\nQuery: "
QUERY_MAX_CHARS = 500


def text_hash(doc: dict) -> str:
    """Hash of what the windows are built from (segment times and text; speaker edits don't matter)."""
    rows = [[round(float(s["start"]), 3), round(float(s["end"]), 3), s.get("text", "")] for s in doc.get("segments", [])]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()


def _split(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    out, cur = [], ""
    for word in text.split():
        while len(word) > max_chars:  # a pathological "word" (pasted text with no spaces)
            if cur:
                out.append(cur)
                cur = ""
            out.append(word[:max_chars])
            word = word[max_chars:]
        if cur and len(cur) + 1 + len(word) > max_chars:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        out.append(cur)
    return out


def build_windows(doc: dict, window_s: float, step_s: float, max_chars: int) -> list[dict]:
    """Windows of the segments starting within [t, t + window_s), for t every step_s seconds."""
    segs = sorted((s for s in doc.get("segments", []) if (s.get("text") or "").strip()), key=lambda s: s["start"])
    if not segs:
        return []
    out, last_ids, t = [], None, float(segs[0]["start"])
    end_all = max(float(s["end"]) for s in segs)
    while t <= end_all:
        inside = [s for s in segs if t <= float(s["start"]) < t + window_s]
        ids = tuple(id(s) for s in inside)
        if inside and ids != last_ids and not set(ids) <= set(last_ids or ()):
            text = " ".join(s["text"].strip() for s in inside)
            start, end = float(inside[0]["start"]), max(float(s["end"]) for s in inside)
            for piece in _split(text, max_chars):
                out.append({"start": round(start, 3), "end": round(end, 3), "text": piece})
            last_ids = ids
        t += step_s
    return out


def max_window_chars(cfg: Config) -> int:
    # Leave room for the estimate's +8 and rounding: estimate_tokens(text) stays under the per-input cap.
    return int((cfg.llm.embed_max_input_tokens - 16) * CHARS_PER_TOKEN)


def embed_state(d: Path, cfg: Config, doc: dict | None = None) -> str:
    """'none' (never embedded) | 'stale' (transcript text, model or settings changed) | 'current'."""
    meta = read_json(Path(d) / EMB_META, None)
    if not meta or meta.get("state") != "done":  # never run, running, or failed
        return "none"
    doc = doc if doc is not None else read_json(Path(d) / EDITABLE, None)
    if doc is None:
        return "stale"
    same = (meta.get("text_hash") == text_hash(doc) and meta.get("model") == cfg.llm.embedding_model
            and meta.get("window_s") == cfg.search.window_s and meta.get("step_s") == cfg.search.step_s)
    return "current" if same else "stale"


def load_vectors(d: Path, meta: dict) -> np.ndarray | None:
    """The .npy matrix if it matches the meta file exactly (hash and shape); otherwise None."""
    p = Path(d) / EMB_NPY
    try:
        raw = p.read_bytes()
    except OSError:
        return None
    if hashlib.sha256(raw).hexdigest() != meta.get("npy_sha256"):
        return None
    m = np.load(io.BytesIO(raw), allow_pickle=False)
    if m.dtype != np.float32 or m.shape != (len(meta.get("windows", [])), meta.get("dims")):
        return None
    return m


def _normalize(vecs) -> np.ndarray:
    m = np.asarray(vecs, dtype=np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1
    return m / n


def embed_session(session_dir: Path, config: Config | None = None, llm: LocalLLM | None = None,
                  gpu_wait_s: float = 3600, force: bool = False, progress=NULL) -> dict:
    cfg = config or load_config()
    d = Path(session_dir)
    doc = read_json(d / EDITABLE, None)
    if doc is None:
        raise FileNotFoundError(f"{d.name} has no transcript.json yet")
    if not force and embed_state(d, cfg, doc) == "current":
        return {"state": "current", "windows": len(read_json(d / EMB_META, {}).get("windows", []))}
    windows = build_windows(doc, cfg.search.window_s, cfg.search.step_s, max_window_chars(cfg))
    llm = llm or embedder_from_config(cfg.llm)
    meta_path = d / EMB_META
    info = {"state": "running", "started": now_iso(), "model": llm.model, "num_ctx": llm.num_ctx,
            "window_s": cfg.search.window_s, "step_s": cfg.search.step_s, "text_hash": text_hash(doc),
            "vram_mb": {}}
    write_json_atomic(meta_path, info)  # a crash leaves "running", so the queue re-runs the job
    try:
        vecs = []
        if windows:
            progress.stage("gpu_wait")
            with gpu_job(cfg.home, wait_s=gpu_wait_s):  # serial with Whisper/pyannote/summary jobs
                llm.release_models((cfg.llm.summary_model, cfg.llm.embedding_model))  # left by a killed job
                info["vram_mb"]["before"] = vram_used_mb()
                try:
                    b = max(1, cfg.llm.embed_batch)
                    progress.stage("embed", total=len(windows), unit="passage")
                    for i in range(0, len(windows), b):
                        progress.update(i)
                        vecs += llm.embed([w["text"] for w in windows[i:i + b]], stage="windows")
                finally:
                    info["unloaded"] = llm.unload()  # keep_alive=0, even after an error
                    info["vram_mb"]["after_unload"] = settle_vram(info["vram_mb"]["before"])
        m = _normalize(vecs) if vecs else np.zeros((0, 0), dtype=np.float32)
        buf = io.BytesIO()
        np.save(buf, m, allow_pickle=False)
        raw = buf.getvalue()
        tmp = d / f"{EMB_NPY}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, d / EMB_NPY)
        info.update(state="done", dims=int(m.shape[1]) if m.size else 0, count=len(windows),
                    npy_sha256=hashlib.sha256(raw).hexdigest(), windows=windows)
    except Exception as e:
        info.update(state="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        info["finished"] = now_iso()
        info["requests"] = llm.requests  # sizes only
        write_json_atomic(meta_path, info)
        llm.close()
    return info


def embed_query(query: str, config: Config, llm: LocalLLM | None = None) -> np.ndarray:
    """Short GPU use: take the GPU lock without waiting (GpuBusy if a job holds it), embed, unload."""
    llm = llm or embedder_from_config(config.llm)
    try:
        with gpu_job(config.home, wait_s=0):
            vec = llm.embed([QUERY_PREFIX + query[:QUERY_MAX_CHARS]], stage="query", keep_alive=0)
            llm.unload(wait_s=10)  # keep_alive=0 already asked for it; wait so the lock covers the unload
        return _normalize(vec)[0]
    finally:
        llm.close()
