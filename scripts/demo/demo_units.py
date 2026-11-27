"""Synthetic units-timeline demo: a class of six lectures with two units, the second starting inside
lecture 4, and search-by-meaning vectors so "Suggest boundaries" has topic shifts to show (one at the start
of lecture 3, one inside lecture 6). No models, no GPU, no audio: the vectors are word hashes made here.

    .venv\\Scripts\\python.exe scripts\\demo\\demo_units.py --data-dir .cache\\units-demo\\data

Then start the app on that folder (launch.json "units-demo", port 8777). Re-running resets the demo's
units. Never point it at the real data folder.
"""
import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import httpx
import numpy as np

import lecture_recorder  # noqa: F401  (offline lockdown)
from lecture_recorder import units as un
from lecture_recorder.config import load_config
from lecture_recorder.recorder.session import write_json_atomic
from lecture_recorder.search import embed as emb
from lecture_recorder.search.text import content_words
from lecture_recorder.summary.llm import embedder_from_config

SLUG = "demo-units"
SEG_S = 30  # one sentence per 30 s: a 60-sentence lecture is 30 min
TOPICS = {
    "cells": ["membrane", "organelle", "nucleus", "cytoplasm", "ribosome", "mitochondrion"],
    "energy": ["glucose", "respiration", "photosynthesis", "chlorophyll", "adenosine", "metabolism"],
    "genes": ["chromosome", "allele", "mutation", "genotype", "heredity", "replication"],
    "ecology": ["predator", "habitat", "biome", "ecosystem", "population", "foodweb"],
}
TEMPLATES = ["Today the {t} matters because the {u} depends on it.", "Notice how the {t} and the {u} interact here.",
             "A common exam question asks about the {t} versus the {u}.", "Write down that the {t} changes the {u} over time."]
LECTURES = [  # name, title, [(topic, sentences)]
    ("2026-09-01_09-00-00", "Lecture 1 · Cell structure", [("cells", 60)]),
    ("2026-09-03_09-00-00", "Lecture 2 · Membranes", [("cells", 60)]),
    ("2026-09-08_09-00-00", "Lecture 3 · Cellular energy", [("energy", 60)]),
    ("2026-09-10_09-00-00", "Lecture 4 · Respiration and DNA", [("energy", 36), ("genes", 34)]),
    ("2026-09-15_09-00-00", "Lecture 5 · Inheritance", [("genes", 60)]),
    ("2026-09-17_09-00-00", "Lecture 6 · Mutations and habitats", [("genes", 30), ("ecology", 36)]),
]
SPLIT = ("2026-09-10_09-00-00", 36 * SEG_S)  # Genetics starts at 18:00 of lecture 4


def sentences(topic: str, n: int) -> list[str]:
    w = TOPICS[topic]
    return [TEMPLATES[i % 4].format(t=w[i % 6], u=w[(i + 2) % 6]) for i in range(n)]


def transcript(sents: list[str]) -> dict:
    segs = [{"id": i, "start": i * SEG_S, "end": i * SEG_S + SEG_S - 1, "speaker": "Speaker 1", "role": "Main",
             "text": s, "words": []} for i, s in enumerate(sents)]
    return {"speakers": [{"id": "Speaker 1", "role": "Main", "speech_s": len(segs) * SEG_S}], "segments": segs,
            "duration_s": len(segs) * SEG_S}


def hashed(text: str, dims: int = 256) -> list[float]:
    v = np.zeros(dims, dtype=np.float32)
    for t in content_words(text.removeprefix(emb.QUERY_PREFIX)):
        v[int(hashlib.md5(t.encode()).hexdigest(), 16) % dims] += 1
    return (v / (np.linalg.norm(v) or 1)).tolist()


def fake_ollama(request: httpx.Request) -> httpx.Response:
    """Stands in for the local embedding model (nothing leaves this process)."""
    if request.url.path == "/api/ps":
        return httpx.Response(200, json={"models": []})
    body = json.loads(request.content)
    if request.url.path == "/api/generate":
        return httpx.Response(200, json={"done": True, "done_reason": "unload"})
    return httpx.Response(200, json={"model": body["model"], "embeddings": [hashed(t) for t in body["input"]],
                                     "prompt_eval_count": sum(len(t) // 4 + 1 for t in body["input"])})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, required=True)
    data = ap.parse_args().data_dir.resolve()
    if data.name != "data" or ".cache" not in data.parts:
        sys.exit("Use a folder named data under .cache (for example .cache\\units-demo\\data), never the real data folder.")
    home = data.parent
    cfg = dataclasses.replace(load_config(home), data_dir=data)  # GPU lock file in the demo's own home
    c = data / "classes" / SLUG
    c.mkdir(parents=True, exist_ok=True)
    write_json_atomic(c / "class.json", {"name": "Biology 101 (demo)", "slug": SLUG})
    for name, title, parts in LECTURES:
        d = c / "sessions" / name
        d.mkdir(parents=True, exist_ok=True)
        doc = transcript([s for topic, n in parts for s in sentences(topic, n)])
        write_json_atomic(d / "transcript.json", doc)
        write_json_atomic(d / "transcript.raw.json", doc)
        write_json_atomic(d / "meta.json", {"name": name, "title": title, "class": SLUG, "state": "processed",
                                            "created": name[:10] + "T09:00:00+00:00", "duration_s": doc["duration_s"]})
        llm = embedder_from_config(cfg.llm)  # a job closes its client, so one per lecture
        llm._http = httpx.Client(  # LocalLLM's client slot (summary/llm.py), as the tests' fake embedder does
            base_url=llm.host, transport=httpx.MockTransport(fake_ollama))
        emb.embed_session(d, cfg, llm=llm, gpu_wait_s=5)
    (c / un.UNITS_FILE).unlink(missing_ok=True)
    un.add_boundary(c, *SPLIT, "Genetics")
    un.rename_unit(c, "u1", "Cells")
    print(f"Demo class '{SLUG}' ready in {data}: 6 lectures, units Cells and Genetics (split inside lecture 4).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
