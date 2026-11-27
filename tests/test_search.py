"""Exact search and concept search.

Everything runs on synthetic transcripts written for these tests, in temp data folders; nothing in
data/ is read. Unit tests use a fake Ollama (httpx mock transport) whose "embeddings" are hashed
bags of words; the `ollama`/`gpu` tests use the installed qwen3-embedding:0.6b through the real
queue worker and GPU lock.
"""
import dataclasses
import hashlib
import json
import re
import shutil
import threading

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import lecture_recorder  # noqa: F401  (lockdown)
from lecture_recorder.config import load_config
from lecture_recorder.pipeline import queue as q
from lecture_recorder.pipeline.gpu import gpu_job
from lecture_recorder.pipeline.outputs import write_outputs
from lecture_recorder.recorder.session import read_json, write_json_atomic
from lecture_recorder.search import embed as emb
from lecture_recorder.search.index import DB_NAME, SearchIndex
from lecture_recorder.search.text import content_words, fts_phrase, stem, tokens
from lecture_recorder.summary.llm import ContextOverflow, LocalLLM, embedder_from_config, release_llm_vram
from lecture_recorder.web.app import create_app

BASE = "http://127.0.0.1:8765"
H = {"X-Lecture-Recorder": "1"}
SEG_S = 10.0

# ------------------------------------------------------------------ synthetic lectures (concept-search content)
# Three lectures, three topics each, one sentence per 10 s segment. Each concept query names its target
# topic's idea in words that the target passage never uses (checked by content_words below).
LECTURES = {
    ("biology-101", "Biology 101", "2026-09-01_09-00-00"): [
        ("membranes", [
            "Good morning. Today we start with the membrane that wraps every living thing we study.",
            "It is built from phospholipids, molecules with a water-loving head and two oily tails.",
            "The tails hide from water, so the molecules line up in two sheets, tail to tail.",
            "Proteins float in that double layer like icebergs, and some of them form channels.",
            "Small uncharged molecules slip straight through, but ions need a channel or a pump.",
            "So the membrane decides what gets in and what stays out."]),
        ("atp", [
            "Now let's follow what happens to the sugar and fat you ate for breakfast.",
            "Their electrons are handed along a chain of proteins sitting in a folded inner membrane.",
            "As electrons move down that chain, protons get pumped across, piling up on one side.",
            "The protons rush back through an enzyme called ATP synthase, which spins like a tiny turbine.",
            "Each turn snaps a phosphate onto ADP, making ATP, the currency your muscles and neurons spend.",
            "Oxygen waits at the end of the chain to catch the spent electrons, and that forms water."]),
        ("dna", [
            "Before the break, a word about copying the genome.",
            "The double helix unzips, and each strand serves as a template for a new partner strand.",
            "DNA polymerase reads the template and adds matching nucleotides, A with T and G with C.",
            "One strand is copied smoothly, the other in short pieces that are stitched together later.",
            "Proofreading keeps the error rate to about one mistake in a billion letters.",
            "That is why your copies stay faithful generation after generation."]),
    ],
    ("econ-200", "Economics 200", "2026-09-02_11-00-00"): [
        ("supply", [
            "Let's begin with the market for coffee beans.",
            "When the price goes up, farmers want to grow more, and that is the supply curve.",
            "Buyers, on the other hand, want fewer bags when the price is high: the demand curve.",
            "Where the two curves cross, the quantity sellers offer equals what buyers want.",
            "A frost in Brazil shifts supply left, and the crossing point moves to a higher price.",
            "Markets keep finding that crossing point again after every shock."]),
        ("inflation", [
            "Now picture your grandparents paying fifty cents for a loaf of bread.",
            "Today the same loaf is several dollars, even though it is the same bread.",
            "When the central bank creates money faster than the economy grows, the general level of prices climbs.",
            "Every year your paycheck buys a little less at the grocery store unless your salary keeps up.",
            "Savers get hurt, because a dollar kept under the mattress buys less next year than it does today.",
            "Economists track this with a basket of everyday goods and report the yearly percentage change."]),
        ("trade", [
            "Finally, why do countries trade at all?",
            "Portugal can make both wine and cloth, but it gives up less to make wine.",
            "England gives up less to make cloth, so each should specialize where it gives up least.",
            "This is comparative advantage, and both sides end up with more of both goods.",
            "It holds even when one country is better at making everything.",
            "Next week we look at tariffs and what happens when trade is blocked."]),
    ],
    ("cs-150", "Computer Science 150", "2026-09-03_14-00-00"): [
        ("recursion", [
            "Today we start with functions that call themselves.",
            "To compute a factorial, multiply n by the factorial of n minus one.",
            "Every such function needs a base case, or it never stops calling itself.",
            "Each call waits on a stack until the one it started returns an answer.",
            "Too many nested calls and the stack overflows, which crashes the program.",
            "Many problems on trees are easiest to write this way."]),
        ("binary-search", [
            "Suppose a phone book has a million names in alphabetical order and you want one of them.",
            "Open it exactly in the middle and compare the name there with the one you want.",
            "If yours comes earlier in the alphabet, throw away the back half; otherwise throw away the front half.",
            "Repeat on the half that remains, halving it again and again.",
            "After about twenty halvings only one page is left, so a million entries need only twenty comparisons.",
            "That is why the number of steps grows with the logarithm of the size."]),
        ("gc", [
            "Last topic: what happens to memory you no longer use.",
            "In C you must free every block yourself, and forgetting causes leaks.",
            "Languages like Java and Python track which objects are still reachable.",
            "Anything nothing points to anymore is garbage and gets reclaimed.",
            "Reference counting frees an object the moment its count drops to zero.",
            "Cycles need a separate collector that walks the graph from the roots."]),
    ],
}
R9_QUERIES = [  # (query, class, topic)
    ("how mitochondria produce energy for the cell", "biology-101", "atp"),
    ("rising cost of living and loss of purchasing power", "econ-200", "inflation"),
    ("efficient algorithm for finding an element in a sorted array", "cs-150", "binary-search"),
]


def topic_span(slug: str, topic: str) -> tuple[float, float]:
    for (s, _, _), topics in LECTURES.items():
        if s == slug:
            i = [t for t, _ in topics].index(topic)
            n = len(topics[0][1])
            return i * n * SEG_S, (i + 1) * n * SEG_S
    raise KeyError(slug)


def doc_from_sentences(sentences: list[str], with_words: bool = True) -> dict:
    segs = []
    for i, text in enumerate(sentences):
        start = i * SEG_S
        words = []
        if with_words:
            ws = text.split(" ")
            step = (SEG_S - 1) / len(ws)
            words = [{"w": " " + w, "start": round(start + k * step, 3), "end": round(start + (k + 1) * step, 3),
                      "p": 0.9} for k, w in enumerate(ws)]
        segs.append({"id": i, "start": start, "end": start + SEG_S - 1, "speaker": "Speaker 1", "role": "Main",
                     "text": text, "words": words})
    return {"speakers": [{"id": "Speaker 1", "role": "Main", "speech_s": len(segs) * SEG_S},
                         {"id": "Speaker 2", "role": "Audience", "speech_s": 0.0}],
            "segments": segs, "duration_s": len(segs) * SEG_S}


def make_session(data, slug, class_name, name, sentences, title=None, with_words=True):
    if not (data / "classes" / slug / "class.json").exists():
        (data / "classes" / slug).mkdir(parents=True, exist_ok=True)
        write_json_atomic(data / "classes" / slug / "class.json", {"name": class_name, "slug": slug})
    d = data / "classes" / slug / "sessions" / name
    d.mkdir(parents=True)
    write_outputs(d, doc_from_sentences(sentences, with_words))
    write_json_atomic(d / "meta.json", {"name": name, "title": title or f"{class_name} lecture", "class": slug,
                                        "created": name[:10] + "T" + name[11:].replace("-", ":") + "+00:00"})
    return d


def make_corpus(data) -> dict[str, object]:
    out = {}
    for (slug, cname, name), topics in LECTURES.items():
        out[slug] = make_session(data, slug, cname, name, [s for _, ss in topics for s in ss])
    return out


# ------------------------------------------------------------------ fake Ollama for embeddings
def hashed_vector(text: str, dims: int = 256) -> list[float]:
    text = text.removeprefix(emb.QUERY_PREFIX)
    v = np.zeros(dims, dtype=np.float32)
    for t in content_words(text):
        v[int(hashlib.md5(t.encode()).hexdigest(), 16) % dims] += 1
    return (v / (np.linalg.norm(v) or 1)).tolist()


class FakeEmbed:
    def __init__(self, status=200, error=None, loaded=()):
        self.bodies, self.unloads, self.status, self.error, self.loaded = [], [], status, error, list(loaded)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.host == "127.0.0.1" and request.url.port == 11434
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": m} for m in self.loaded]})
        body = json.loads(request.content)
        if request.url.path == "/api/generate":
            assert body["keep_alive"] == 0
            self.unloads.append(body["model"])
            self.loaded = [m for m in self.loaded if m != body["model"]]
            return httpx.Response(200, json={"done": True, "done_reason": "unload"})
        assert request.url.path == "/api/embed"
        self.bodies.append(body)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": self.error})
        return httpx.Response(200, json={"model": body["model"], "embeddings": [hashed_vector(t) for t in body["input"]],
                                         "prompt_eval_count": sum(len(t) // 4 + 1 for t in body["input"])})


def fake_embedder(fake: FakeEmbed, cfg) -> LocalLLM:
    llm = embedder_from_config(cfg.llm)
    llm._http = httpx.Client(base_url=llm.host, transport=httpx.MockTransport(fake))
    return llm


@pytest.fixture
def cfg(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    return dataclasses.replace(load_config(), data_dir=data)


@pytest.fixture
def corpus(cfg):
    return make_corpus(cfg.data_dir)


def embed_all(cfg, sessions) -> FakeEmbed:
    fake = FakeEmbed()
    for d in sessions:
        emb.embed_session(d, cfg, llm=fake_embedder(fake, cfg))
    return fake


@pytest.fixture
def client(cfg):
    app = create_app(cfg, autoprocess=False, monitor=False)
    with TestClient(app, base_url=BASE) as c:
        yield c


# ------------------------------------------------------------------ text normalization
def test_tokens_ignore_case_punctuation_and_accents():
    assert tokens("The Purple Elephant-Theorem!") == ["the", "purple", "elephant", "theorem"]
    assert tokens("Don't  STOP, café_au_lait 3.14") == ["dont", "stop", "cafe", "au", "lait", "3", "14"]
    assert tokens('"*NEAR( OR \'); DROP') == ["near", "or", "drop"]
    assert fts_phrase(["near", "or"]) == '"near or"'
    with pytest.raises(ValueError):
        fts_phrase(['a"b'])


# ------------------------------------------------------------------ exact search
def test_r8_exact_phrase_returns_session_class_time_and_context(cfg, corpus):
    idx = SearchIndex(cfg.data_dir, cfg)
    r = idx.exact("atp SYNTHASE, which spins")
    assert r["total"] == 1 and len(r["results"]) == 1
    hit = r["results"][0]
    assert (hit["class"], hit["class_name"], hit["session"]) == ("biology-101", "Biology 101", "2026-09-01_09-00-00")
    assert {x["class_name"] for x in SearchIndex(cfg.data_dir, cfg).exact("the")["results"]} == {
        "Biology 101", "Economics 200", "Computer Science 150"}
    assert hit["sid"] == "biology-101/2026-09-01_09-00-00"
    seg = 6 + 3  # topic 2, sentence 4
    words = doc_from_sentences(LECTURES[("biology-101", "Biology 101", "2026-09-01_09-00-00")][1][1])["segments"][3]["words"]
    expected = seg * SEG_S + words[[w["w"].strip() for w in words].index("ATP")]["start"] - 3 * SEG_S
    assert abs(hit["start"] - expected) < 0.01  # the word's own timestamp, not the segment start
    c = hit["context"]
    assert c["match"] == "ATP synthase, which spins"
    assert c["before"].endswith("an enzyme called ") and c["after"].startswith(" like a tiny turbine")
    assert hit["end"] >= hit["start"] and hit["created"].startswith("2026-09-01")


def test_r8_case_punctuation_insensitive_and_whole_words_only(cfg, corpus):
    idx = SearchIndex(cfg.data_dir, cfg)
    assert idx.exact("water loving head")["total"] == 1  # text says "water-loving head"
    assert idx.exact("LET S BEGIN")["total"] == 0  # "Let's" is one token: lets
    assert idx.exact("lets begin")["total"] == 1
    assert idx.exact("synthas")["total"] == 0  # no partial words
    assert idx.exact("the")["total"] > 10


def test_r8_phrase_across_segment_boundary_and_interpolated_time(cfg):
    d = make_session(cfg.data_dir, "physics-101", "Physics 101", "2026-09-04_08-00-00",
                     ["The answer is the purple", "elephant theorem, as promised."], with_words=False)
    r = SearchIndex(cfg.data_dir, cfg).exact("Purple Elephant Theorem")
    assert r["total"] == 1
    hit = r["results"][0]
    assert hit["context"]["match"] == "purple elephant theorem"
    assert abs(hit["start"] - 9 * 4 / 5) < 0.01  # no words: interpolated inside the 9 s segment
    assert hit["end"] == 19.0 and d.is_dir()


def test_r8_class_filter_and_newest_first(cfg, corpus):
    make_session(cfg.data_dir, "biology-101", "Biology 101", "2026-09-08_09-00-00", ["The membrane again, briefly."])
    idx = SearchIndex(cfg.data_dir, cfg)
    r = idx.exact("membrane")
    assert [x["session"] for x in r["results"]][0] == "2026-09-08_09-00-00"
    assert r["sessions"] == 2
    assert idx.exact("membrane", class_slug="econ-200")["total"] == 0


EVIL = ['"', '""', "*", "NEAR(", "NEAR(a b, 2)", "OR", "a OR b", "AND", "NOT the", "'); DROP TABLE sessions; --",
        "^the", "tokens:the", "{tokens}: the", "-the", "the*", "(the", "the)", "\\", "%", "_", "\x00", "'", "a'b",
        "SELECT * FROM fts", "x" * 600]


def brute_force(cfg, query: str) -> int:
    q = tokens(query)[:64]
    if not q:
        return 0
    n = 0
    for (slug, _, name), topics in LECTURES.items():
        toks = tokens(" ".join(s for _, ss in topics for s in ss))
        n += sum(toks[i:i + len(q)] == q for i in range(len(toks) - len(q) + 1))
    return n


@pytest.mark.parametrize("query", EVIL)
def test_r8_user_input_is_never_fts_or_sql_syntax(cfg, corpus, client, query):
    r = client.get("/api/search", params={"q": query, "mode": "exact"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == brute_force(cfg, query[:500])  # literal words only, same as a plain scan
    if not tokens(query):
        assert body["message"].startswith("Type at least one word")
    with __import__("sqlite3").connect(cfg.data_dir / DB_NAME) as con:  # the index is intact
        assert con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 3


def test_r8_or_and_near_are_searched_as_words(cfg, corpus):
    make_session(cfg.data_dir, "physics-101", "Physics 101", "2026-09-05_08-00-00",
                 ["Stand near the door or by the window."])
    idx = SearchIndex(cfg.data_dir, cfg)
    assert idx.exact("NEAR(")["total"] == 1
    assert idx.exact("door OR")["total"] == 1
    assert idx.exact("door AND window")["total"] == 0


def test_r8_edits_are_searchable_in_app_and_by_hand(cfg, corpus, client):
    s = "2026-09-03_14-00-00"
    assert client.get("/api/search", params={"q": "tiny turbine"}).json()["total"] == 1
    r = client.put(f"/api/sessions/cs-150/{s}/transcript", headers=H,
                   json={"segments": [{"id": 1, "text": "To compute a factorial, use the purple elephant theorem."}]})
    assert r.status_code == 200
    assert client.get("/api/search", params={"q": "purple elephant theorem"}).json()["results"][0]["session"] == s
    assert client.get("/api/search", params={"q": "multiply n by the factorial"}).json()["total"] == 0
    # by hand, outside the app
    p = corpus["econ-200"] / "transcript.json"
    doc = read_json(p)
    doc["segments"][0]["text"] = "Let's begin with the market for tea leaves."
    write_json_atomic(p, doc)
    assert client.get("/api/search", params={"q": "tea leaves"}).json()["total"] == 1
    assert client.get("/api/search", params={"q": "coffee beans"}).json()["total"] == 0


def test_removed_session_leaves_the_index(cfg, corpus):
    idx = SearchIndex(cfg.data_dir, cfg)
    assert idx.exact("comparative advantage")["total"] == 1
    shutil.rmtree(corpus["econ-200"])  # a temp test folder
    assert idx.exact("comparative advantage")["total"] == 0


def test_rebuild_from_files_gives_identical_results(cfg, corpus):
    embed_all(cfg, corpus.values())
    queries = ["the", "ATP synthase", "crossing point", "stack overflows", "purple"]
    qvecs = [np.asarray(hashed_vector(emb.QUERY_PREFIX + t), dtype=np.float32) for t, _, _ in R9_QUERIES]

    def snapshot(idx):
        return ([idx.exact(t) for t in queries], [idx.concept(v) for v in qvecs], idx.concept_status())

    first = snapshot(SearchIndex(cfg.data_dir, cfg))
    assert first[2] == {"transcribed": 3, "indexed": 3, "pending": 0, "windows": first[2]["windows"]}
    assert first[2]["windows"] > 0 and first[1][0]
    idx = SearchIndex(cfg.data_dir, cfg)
    assert idx.rebuild()["added"] == 3
    assert snapshot(idx) == first
    for f in cfg.data_dir.glob(DB_NAME + "*"):  # delete the index entirely: it is rebuilt from the folders
        f.unlink()
    assert snapshot(SearchIndex(cfg.data_dir, cfg)) == first


def test_reindex_command(cfg, corpus, capsys):
    from lecture_recorder.cli import main

    assert main(["reindex", "--data-dir", str(cfg.data_dir)]) == 0
    out = capsys.readouterr().out
    assert "3 session(s)" in out and "missing or stale for 3" in out
    assert main(["search", "tiny turbine", "--data-dir", str(cfg.data_dir)]) == 0
    assert "biology-101/2026-09-01_09-00-00  00:01:" in capsys.readouterr().out


# ------------------------------------------------------------------ concept search plumbing
def test_windows_overlap_and_respect_the_input_cap(cfg):
    doc = doc_from_sentences([f"sentence number {i}" for i in range(30)])  # 300 s
    ws = emb.build_windows(doc, 45, 30, 10_000)
    assert [w["start"] for w in ws[:3]] == [0, 30, 60] and ws[0]["end"] == 49 and ws[1]["start"] < ws[0]["end"]
    long = {"segments": [{"start": 0, "end": 9, "text": "word " * 3000}]}
    pieces = emb.build_windows(long, 45, 30, emb.max_window_chars(cfg))
    assert len(pieces) > 1 and all(len(p["text"]) <= emb.max_window_chars(cfg) for p in pieces)


def test_embed_job_sets_num_ctx_truncate_false_logs_sizes_and_unloads(cfg, corpus):
    d = corpus["biology-101"]
    fake = FakeEmbed(loaded=[cfg.llm.summary_model])  # left over by a killed summary job
    info = emb.embed_session(d, cfg, llm=fake_embedder(fake, cfg))
    assert info["state"] == "done" and info["count"] == len(info["windows"]) > 0
    assert fake.unloads == [cfg.llm.summary_model, cfg.llm.embedding_model]  # leftover first, ours at the end
    for b in fake.bodies:
        assert b["truncate"] is False and b["options"] == {"num_ctx": cfg.llm.embed_num_ctx}
        assert b["model"] == cfg.llm.embedding_model and len(b["input"]) <= cfg.llm.embed_batch
    meta = read_json(d / emb.EMB_META)
    for r in meta["requests"]:  # sizes only
        assert set(r) == {"stage", "model", "num_ctx", "inputs", "input_chars", "max_input_chars", "max_est_tokens",
                          "prompt_eval_count", "seconds", "error"}
        assert r["num_ctx"] == 2048 and r["error"] is None
    assert emb.embed_state(d, cfg) == "current"
    m = emb.load_vectors(d, meta)
    assert m.shape == (meta["count"], meta["dims"]) and np.allclose(np.linalg.norm(m, axis=1), 1, atol=1e-5)
    assert emb.embed_session(d, cfg, llm=fake_embedder(FakeEmbed(), cfg))["state"] == "current"  # no GPU work


def test_text_edit_marks_vectors_stale_and_queues_embedding(cfg, corpus, client):
    embed_all(cfg, corpus.values())
    s = "2026-09-03_14-00-00"
    client.put(f"/api/sessions/cs-150/{s}/transcript", headers=H, json={"segments": [{"id": 0, "speaker": "Speaker 2"}]})
    assert emb.embed_state(corpus["cs-150"], cfg) == "current"  # speaker-only edit: vectors still valid
    assert not [j for j in q.jobs(cfg.data_dir) if q.kind_of(j) == q.EMBED]
    client.put(f"/api/sessions/cs-150/{s}/transcript", headers=H, json={"segments": [{"id": 0, "text": "Hello."}]})
    assert emb.embed_state(corpus["cs-150"], cfg) == "stale"
    assert [q.kind_of(j) for j in q.jobs(cfg.data_dir)] == [q.EMBED]
    st = client.get("/api/search/status").json()
    assert st["indexed"] == 2 and st["pending"] == 1  # stale vectors are never searched


def test_embedding_fails_loudly_when_ollama_refuses_an_overlong_input(cfg, corpus):
    fake = FakeEmbed(status=400, error="the input length exceeds the context length")
    with pytest.raises(ContextOverflow):
        emb.embed_session(corpus["cs-150"], cfg, llm=fake_embedder(fake, cfg))
    meta = read_json(corpus["cs-150"] / emb.EMB_META)
    assert meta["state"] == "failed" and "ContextOverflow" in meta["error"]
    assert fake.unloads == [cfg.llm.embedding_model]  # unloaded even after the error
    assert emb.embed_state(corpus["cs-150"], cfg) == "none"
    llm = fake_embedder(FakeEmbed(), cfg)
    with pytest.raises(ContextOverflow):  # refused before sending
        llm.embed(["word " * 5000], stage="windows")


def test_concept_search_api_and_results_carry_scope_fields(cfg, corpus, client, monkeypatch):
    embed_all(cfg, corpus.values())
    fake = FakeEmbed()
    monkeypatch.setattr(emb, "embedder_from_config", lambda llm_cfg: fake_embedder(fake, cfg))
    r = client.get("/api/search", params={"q": "ATP synthase turbine", "mode": "concept"})
    assert r.status_code == 200, r.text
    top = r.json()["results"][0]
    assert {"sid", "class", "session", "created", "start", "end", "text", "score", "rank"} <= set(top)
    assert top["class"] == "biology-101" and top["rank"] == 1
    assert fake.bodies[0]["input"][0].startswith("Instruct:") and fake.bodies[0]["options"]["num_ctx"] == 2048
    assert fake.bodies[0]["keep_alive"] == 0 and fake.unloads == [cfg.llm.embedding_model]
    client.get("/api/search", params={"q": "atp  synthase TURBINE", "mode": "concept"})
    assert len(fake.bodies) == 1  # repeated query: cached, no GPU use
    only = client.get("/api/search", params={"q": "ATP synthase turbine", "mode": "concept", "class_slug": "cs-150"})
    assert {x["class"] for x in only.json()["results"]} == {"cs-150"}
    results = r.json()["results"]
    for a in results:  # overlapping windows of one session are collapsed into the best one
        assert not any(b is not a and b["sid"] == a["sid"] and b["start"] < a["end"] and a["start"] < b["end"]
                       for b in results)


def test_concept_query_says_busy_instead_of_waiting_when_a_gpu_job_runs(cfg, corpus, client, monkeypatch):
    embed_all(cfg, corpus.values())
    called = []
    monkeypatch.setattr(emb, "embedder_from_config", lambda llm_cfg: called.append(1) or fake_embedder(FakeEmbed(), cfg))
    held, release = threading.Event(), threading.Event()

    def job():
        with gpu_job(cfg.home, wait_s=5):
            held.set()
            release.wait(30)

    t = threading.Thread(target=job)
    t.start()
    try:
        assert held.wait(10)
        r = client.get("/api/search", params={"q": "cell energy", "mode": "concept"})
        assert r.status_code == 503 and "busy" in r.json()["detail"] and "try again" in r.json()["detail"]
        assert client.get("/api/search", params={"q": "tiny turbine"}).json()["total"] == 1  # exact still works
    finally:
        release.set()
        t.join()
    assert client.get("/api/search", params={"q": "cell energy", "mode": "concept"}).status_code == 200


def test_queue_chains_embedding_after_summary_and_recovers(cfg, corpus, monkeypatch):
    d = corpus["cs-150"]
    calls = []

    class R:
        returncode, stdout, stderr = 0, "", ""

    def fake_run(cmd, **kw):
        calls.append(cmd[cmd.index("--task") + 1])
        return R()

    monkeypatch.setattr(q, "_run_job", fake_run)
    q.enqueue(cfg.data_dir, d, q.SUMMARIZE)
    q.run_worker(cfg.data_dir)
    assert calls == ["summarize", "embed"]
    # a summary that failed still leads to search indexing
    calls.clear()
    R.returncode = 1
    q.enqueue(cfg.data_dir, d, q.SUMMARIZE)
    q.run_worker(cfg.data_dir)
    assert calls == ["summarize", "embed"]
    # an embed job left "running" by a crash: re-queued unless its vectors are current
    job = q.enqueue(cfg.data_dir, d, q.EMBED)
    job.update(state="running")
    q._save(job)
    R.returncode = 0
    calls.clear()
    q.run_worker(cfg.data_dir)
    assert calls == ["embed"]


def test_index_missing_queues_only_sessions_without_current_vectors(cfg, corpus, client):
    embed_all(cfg, [corpus["cs-150"]])
    r = client.post("/api/search/index-missing", headers=H)
    assert r.json() == {"queued": 2}
    assert sorted(j["session"] for j in q.jobs(cfg.data_dir)) == sorted(
        str(corpus[s].resolve()) for s in ("biology-101", "econ-200"))


def test_gpu_jobs_release_both_models(monkeypatch):
    from lecture_recorder.summary import llm as m

    c = load_config().llm
    fake = FakeEmbed(loaded=[c.summary_model, c.embedding_model, "someone-elses-model"])
    real = m.LocalLLM.from_config

    def from_config(llm_cfg):
        x = real(llm_cfg)
        x._http = httpx.Client(base_url=x.host, transport=httpx.MockTransport(fake))
        return x

    monkeypatch.setattr(m.LocalLLM, "from_config", classmethod(lambda cls, llm_cfg: from_config(llm_cfg)))
    assert release_llm_vram(c) is True
    assert fake.unloads == [c.summary_model, c.embedding_model] and fake.loaded == ["someone-elses-model"]


def test_ui_has_search_box_and_jump_to_time():
    from pathlib import Path

    static = Path(lecture_recorder.__file__).parent / "web" / "static"
    html = (static / "index.html").read_text(encoding="utf-8")
    js = (static / "app.js").read_text(encoding="utf-8")
    assert 'id="search-q"' in html and 'value="concept"' in html
    assert ".innerHTML" not in js and "?t=${r.start}" in js and "audio.currentTime = Math.max(0, t - 1)" in js


# ------------------------------------------------------------------ with the real embedding model
def passage_check(query: str, passage: str) -> set[str]:
    """Query content words (lowercased, simply stemmed) that appear in the passage; must be empty."""
    found = set()
    p_stems = {stem(t) for t in tokens(passage)}
    p_tokens = set(tokens(passage))
    for qw in content_words(query):
        if qw in p_stems or (len(qw) >= 4 and any(t.startswith(qw) for t in p_tokens)):
            found.add(qw)
    return found


def test_r9_synthetic_targets_share_no_words_with_their_queries():
    for query, slug, topic in R9_QUERIES:
        text = " ".join(next(ss for (s, _, _), ts in LECTURES.items() if s == slug for t, ss in ts if t == topic))
        assert passage_check(query, text) == set(), (query, passage_check(query, text))


@pytest.mark.gpu
@pytest.mark.ollama
def test_r9_concept_search_finds_passage_without_the_query_words(cfg, corpus, client, capsys):
    # Index through the real queue worker (own process, GPU lock, unload at the end).
    for d in corpus.values():
        q.enqueue(cfg.data_dir, d, q.EMBED)
    done = q.run_worker(cfg.data_dir)
    assert done and all(j["state"] == "done" for j in done), [j.get("error") for j in done or []]
    for d in corpus.values():
        meta = read_json(d / emb.EMB_META)
        assert meta["state"] == "done" and meta["unloaded"] is True
        assert all(r["error"] is None and r["num_ctx"] == 2048 and r["prompt_eval_count"] for r in meta["requests"])
    report = []
    for query, slug, topic in R9_QUERIES:
        r = client.get("/api/search", params={"q": query, "mode": "concept"})
        assert r.status_code == 200, r.text
        results = r.json()["results"]
        lo, hi = topic_span(slug, topic)

        def is_target(x):
            overlap = min(x["end"], hi) - max(x["start"], lo)
            return x["class"] == slug and overlap >= 0.5 * (x["end"] - x["start"])

        rank = next((x["rank"] for x in results if is_target(x)), None)
        top = results[0]
        report.append(f"Concept {query!r}: expected passage at rank {rank} (score {results[rank - 1]['score'] if rank else '-'}), "
                      f"top = {top['class']} {top['start']:.0f}-{top['end']:.0f}s score {top['score']}, "
                      f"runner-up {results[1]['class']} {results[1]['start']:.0f}s score {results[1]['score']}")
        assert rank == 1, report[-1]
        assert passage_check(query, top["text"]) == set(), (query, top["text"])
    with capsys.disabled():
        print("\n" + "\n".join(report))


@pytest.mark.gpu
@pytest.mark.ollama
def test_real_ollama_refuses_overlong_embedding_input(cfg):
    llm = embedder_from_config(cfg.llm)
    llm.max_prompt_tokens = 100_000  # bypass our own guard to prove Ollama's truncate:false refusal
    try:
        with gpu_job(cfg.home, wait_s=600):
            with pytest.raises(ContextOverflow, match="exceeds the context length"):
                llm.embed(["photosynthesis " * 3000], stage="probe")
            assert llm.unload()
    finally:
        llm.close()
    assert llm.requests[0]["num_ctx"] == 2048 and llm.requests[0]["error"] == "ContextOverflow"


# ------------------------------------------------------------------ regressions
def test_edit_during_a_running_embed_job_is_embedded_again(cfg, corpus, monkeypatch):
    d = corpus["cs-150"]

    class R:
        returncode, stdout, stderr = 0, "", ""

    calls = []

    def fake_run(cmd, **kw):  # the "job" embeds the old text, then the user edits it before it finishes
        calls.append(cmd[cmd.index("--task") + 1])
        if len(calls) == 1:
            embed_all(cfg, [d])
            doc = read_json(d / "transcript.json")
            doc["segments"][0]["text"] = "Edited while indexing."
            write_json_atomic(d / "transcript.json", doc)
        return R()

    monkeypatch.setattr(q, "_run_job", fake_run)
    q.enqueue(cfg.data_dir, d, q.EMBED)
    q.run_worker(cfg.data_dir)
    assert calls == ["embed", "embed"]  # re-queued because the vectors no longer match the text


@pytest.mark.parametrize("bad", [{"start": None}, {"end": "x"}, {"text": 5}, "drop-end"])
def test_one_malformed_transcript_is_skipped_not_fatal(cfg, corpus, bad):
    p = corpus["econ-200"] / "transcript.json"
    doc = read_json(p)
    if bad == "drop-end":
        del doc["segments"][2]["end"]
    else:
        doc["segments"][2].update(bad)
    write_json_atomic(p, doc)
    idx = SearchIndex(cfg.data_dir, cfg)
    assert idx.exact("tiny turbine")["total"] == 1  # the other lectures still work
    assert idx.rebuild()["added"] == 2 and idx.concept_status()["transcribed"] == 2


@pytest.mark.parametrize("query", ["½", "x²", "ﬁne", "Ⅻ", "é"])
def test_unicode_compatibility_characters_never_error(cfg, client, query):
    make_session(cfg.data_dir, "physics-101", "Physics 101", "2026-09-06_08-00-00", [f"Take {query} of it."])
    r = client.get("/api/search", params={"q": query})
    assert r.status_code == 200 and r.json()["total"] == 1


def test_changing_the_embedding_model_reindexes_and_marks_vectors_stale(cfg, corpus):
    embed_all(cfg, corpus.values())
    assert SearchIndex(cfg.data_dir, cfg).concept_status()["indexed"] == 3
    other = dataclasses.replace(cfg, llm=dataclasses.replace(cfg.llm, embedding_model="another-embedder"))
    idx = SearchIndex(cfg.data_dir, other)  # same app.db, new setting
    assert idx.concept_status() == {"transcribed": 3, "indexed": 0, "pending": 3, "windows": 0}
    assert len(idx.pending_embedding()) == 3
