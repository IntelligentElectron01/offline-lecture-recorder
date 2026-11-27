"""Units as time spans, boundary suggestions, exam scopes, scoped search, exam
material and the download-only bundle. Synthetic lectures in temp folders only; nothing in data/.

The class has three lectures: the first is all unit 1 (cell biology), the second switches to unit 2
(economics) at 90 s, the third is all unit 2. Each unit has its own unique terms; "key idea" is said
in both, including right before the mid-lecture boundary.
"""
import dataclasses
import json
import re

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import lecture_recorder  # noqa: F401  (lockdown)
from lecture_recorder import units as un
from lecture_recorder.config import load_config
from lecture_recorder.pipeline import queue as q
from lecture_recorder.recorder.session import read_json, write_json_atomic
from lecture_recorder.search.index import DB_NAME, SearchIndex
from lecture_recorder.search.units_suggest import suggest_boundaries
from lecture_recorder.summary.bundle import build_bundle
from lecture_recorder.summary.exam import EXAM_MD, EXAM_META, generate_exam_material, material_state
from lecture_recorder.summary.llm import LocalLLM, OutputTruncated
from lecture_recorder.web.app import create_app
from test_search import FakeEmbed, SEG_S, doc_from_sentences, embed_all, fake_embedder, hashed_vector

BASE = "http://127.0.0.1:8765"
H = {"X-Lecture-Recorder": "1"}
SLUG = "bio-econ"
U1_TERMS = ["mitochondrion", "ribosome", "cytoplasm", "chloroplast", "lysosome", "vacuole"]
U2_TERMS = ["elasticity", "tariff", "inflation", "monopoly", "subsidy", "equilibrium"]
BIO = ["The {t} sits inside the living cell and keeps the organelle membranes busy.",
       "Under the microscope the {t} looks small, but the cell depends on it every second.",
       "Remember the key idea: the {t} works together with the other organelles of the cell.",
       "Biologists stain the {t} so the cell structure shows up clearly on the slide."]
ECON = ["The {t} changes how buyers and sellers trade in the market every quarter.",
        "Governments watch the {t} closely because prices and wages respond to it.",
        "Remember the key idea: the {t} shifts the supply and demand curves of the market.",
        "Economists measure the {t} with data from firms, households and trade statistics."]
STUDENT_LINE = "Sorry, my roommate Jordan asked whether this is on the test?"
S1, S2, S3 = "2026-09-01_09-00-00", "2026-09-03_09-00-00", "2026-09-05_09-00-00"
BOUNDARY = 9 * SEG_S  # second lecture: unit 2 starts at 90 s


def sentences(templates, terms, n, offset=0):
    return [templates[(i + offset) % len(templates)].format(t=terms[(i + offset) % len(terms)]) for i in range(n)]


def make_lecture(data, name, sents, student_at=None):
    d = data / "classes" / SLUG / "sessions" / name
    d.mkdir(parents=True)
    doc = doc_from_sentences(sents)
    if student_at is not None:
        doc["segments"][student_at].update(speaker="Speaker 2", role="Audience", text=STUDENT_LINE,
                                           words=doc_from_sentences([STUDENT_LINE])["segments"][0]["words"])
        for w in doc["segments"][student_at]["words"]:
            w["start"] += student_at * SEG_S
            w["end"] += student_at * SEG_S
    write_json_atomic(d / "transcript.json", doc)
    write_json_atomic(d / "transcript.raw.json", doc)
    write_json_atomic(d / "meta.json", {"name": name, "title": f"Lecture {name[8:10]}", "class": SLUG,
                                        "created": name[:10] + "T09:00:00+00:00", "duration_s": len(sents) * SEG_S})
    return d


@pytest.fixture
def cfg(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    return dataclasses.replace(load_config(home), data_dir=home / "data")


@pytest.fixture
def cls(cfg):
    data = cfg.data_dir
    c = data / "classes" / SLUG
    c.mkdir(parents=True)
    write_json_atomic(c / "class.json", {"name": "Bio and Econ", "slug": SLUG})
    make_lecture(data, S1, sentences(BIO, U1_TERMS, 18), student_at=4)
    # lecture 2: unit 1 until 90 s (its last unit-1 line says "key idea"), then unit 2
    make_lecture(data, S2, sentences(BIO, U1_TERMS, 8, offset=1) + [BIO[2].format(t="ribosome")]
                 + sentences(ECON, U2_TERMS, 9))
    make_lecture(data, S3, sentences(ECON, U2_TERMS, 18, offset=2))
    return c


@pytest.fixture
def split(cls):
    """The confirmed mid-lecture boundary: unit 2 starts at 90 s of lecture 2."""
    u2 = un.add_boundary(cls, S2, BOUNDARY, "Markets")
    un.rename_unit(cls, "u1", "Cells")
    return {"u1": "u1", "u2": u2["id"]}


def text_has(text, terms):
    low = text.lower()
    return [t for t in terms if t in low]


# ------------------------------------------------------------------ units as time spans
def test_one_unit_by_default_then_a_mid_lecture_boundary_splits_a_lecture(cls):
    spans = un.unit_spans(cls)
    assert [u["name"] for u in spans] == ["Unit 1"] and len(spans[0]["spans"]) == 3
    u2 = un.add_boundary(cls, S2, BOUNDARY)
    spans = {u["id"]: u for u in un.unit_spans(cls)}
    assert spans["u1"]["spans"] == [{"session": S1, "start": 0.0, "end": None},
                                    {"session": S2, "start": 0.0, "end": BOUNDARY}]
    assert spans[u2["id"]]["spans"] == [{"session": S2, "start": BOUNDARY, "end": None},
                                        {"session": S3, "start": 0.0, "end": None}]
    assert spans[u2["id"]]["name"] == "Unit 2"  # unnamed units are numbered in time order
    assert read_json(cls / un.UNITS_FILE)["units"][-1]["start"] == {"session": S2, "t": BOUNDARY}  # plain file


def test_boundary_validation_rename_and_remove(cls):
    with pytest.raises(un.UnitError):
        un.add_boundary(cls, "2026-01-01_00-00-00", 10)
    with pytest.raises(un.UnitError):
        un.add_boundary(cls, S2, 10_000)  # past the end of the lecture
    with pytest.raises(un.UnitError):
        un.add_boundary(cls, S1, 0)  # the first unit already starts there
    u = un.add_boundary(cls, S3, 0)  # a unit can start at the beginning of a lecture
    with pytest.raises(un.UnitError):
        un.add_boundary(cls, S3, 0.5)
    un.rename_unit(cls, u["id"], "Trade")
    assert [x["name"] for x in un.unit_spans(cls)] == ["Unit 1", "Trade"]
    un.remove_boundary(cls, u["id"])
    assert len(un.unit_spans(cls)) == 1
    with pytest.raises(un.UnitError):
        un.remove_boundary(cls, "u1")  # the first unit always exists


# ------------------------------------------------------------------ boundary suggestions (CPU only)
def test_suggestions_find_the_topic_shift_on_cpu_and_change_nothing(cls, cfg, monkeypatch):
    embed_all(cfg, [cls / "sessions" / s for s in (S1, S2, S3)])
    before = sorted(p.name for p in cls.rglob("*"))
    monkeypatch.setattr("lecture_recorder.summary.llm.LocalLLM.embed", lambda *a, **k: pytest.fail("no model use"))
    out = suggest_boundaries(cls, cfg)
    assert out["not_indexed"] == []
    best = max(out["suggestions"], key=lambda s: s["score"])
    assert best["session"] == S2 and abs(best["t"] - BOUNDARY) <= 30, out
    assert sorted(p.name for p in cls.rglob("*")) == before and not (cls / un.UNITS_FILE).exists()
    un.add_boundary(cls, best["session"], best["t"])  # the user confirms it
    assert not any(s["session"] == S2 and abs(s["t"] - best["t"]) < 120
                   for s in suggest_boundaries(cls, cfg)["suggestions"])  # not suggested again


# ------------------------------------------------------------------ scoped search
@pytest.mark.parametrize("unit,own,other", [("u1", U1_TERMS, U2_TERMS), ("u2", U2_TERMS, U1_TERMS)])
def test_r7_exact_search_in_one_unit_returns_nothing_from_the_other(cls, cfg, split, unit, own, other):
    idx = SearchIndex(cfg.data_dir, cfg)
    scope = {"units": [split[unit]]}
    everywhere = idx.exact("key idea")
    assert {x["unit_id"] for x in everywhere["results"]} == {split["u1"], split["u2"]}
    r = idx.exact("key idea", SLUG, scope=scope)
    assert 0 < r["total"] < everywhere["total"]
    spans = un.flatten(idx.units(SLUG, scope))
    for x in r["results"]:
        assert x["unit_id"] == split[unit] and un.in_spans(spans, x["session"], x["start"])
        ctx = x["context"]["before"] + x["context"]["match"] + x["context"]["after"]
        assert text_has(ctx, other) == [], ctx  # context never crosses the boundary
    if unit == "u1":  # the line just before the mid-lecture boundary is found, without unit-2 context
        assert any(x["session"] == S2 and BOUNDARY - SEG_S <= x["start"] < BOUNDARY for x in r["results"])
    for term in other:
        assert idx.exact(term, SLUG, scope=scope)["total"] == 0
    assert all(idx.exact(t, SLUG, scope=scope)["total"] > 0 for t in own)


@pytest.mark.parametrize("unit,other", [("u1", U2_TERMS), ("u2", U1_TERMS)])
def test_r7_search_by_meaning_in_one_unit_returns_nothing_from_the_other(cls, cfg, split, unit, other):
    embed_all(cfg, [cls / "sessions" / s for s in (S1, S2, S3)])
    idx = SearchIndex(cfg.data_dir, cfg)
    scope = {"units": [split[unit]]}
    # a query about the OTHER unit's topic: unscoped it finds the other unit, scoped it must not
    qvec = np.asarray(hashed_vector(" ".join(other)), dtype=np.float32)
    assert idx.concept(qvec, k=1)[0]["unit_id"] != split[unit]
    results = idx.concept(qvec, SLUG, k=20, scope=scope)
    spans = un.flatten(idx.units(SLUG, scope))
    assert results and all(x["unit_id"] == split[unit] for x in results)
    for x in results:
        assert un.in_spans(spans, x["session"], x["start"], x["end"]) and text_has(x["text"], other) == []
    assert any(x["session"] == S2 for x in results)  # the split lecture contributes its own side only


def test_saved_exams_by_units_or_dates_and_scoped_search_api(cls, cfg, split):
    app = create_app(cfg, autoprocess=False, monitor=False)
    with TestClient(app, base_url=BASE) as c:
        r = c.post(f"/api/classes/{SLUG}/exams", headers=H, json={"name": "Midterm", "units": ["u1"]})
        assert r.status_code == 201 and r.json()["id"] == "midterm"
        r = c.post(f"/api/classes/{SLUG}/exams", headers=H, json={"name": "Last week", "from": "2026-09-04", "to": "2026-09-30"})
        assert r.status_code == 201
        assert c.post(f"/api/classes/{SLUG}/exams", headers=H, json={"name": "X"}).status_code == 400
        assert c.post(f"/api/classes/{SLUG}/exams", headers=H, json={"name": "X", "units": ["nope"]}).status_code == 400
        exams = {e["id"]: e for e in c.get(f"/api/classes/{SLUG}/exams").json()}
        assert [u["id"] for u in exams["midterm"]["scope_units"]] == ["u1"]
        assert exams["last-week"]["scope_units"] == [{**exams["last-week"]["scope_units"][0], "spans": [
            {"session": S3, "start": 0.0, "end": None}]}]  # whole lectures on those dates
        r = c.get("/api/search", params={"q": "key idea", "class_slug": SLUG, "exam": "midterm"}).json()
        assert r["total"] > 0 and {x["unit"] for x in r["results"]} == {"Cells"}
        assert c.get("/api/search", params={"q": "tariff", "class_slug": SLUG, "exam": "midterm"}).json()["total"] == 0
        r = c.get("/api/search", params={"q": "key idea", "class_slug": SLUG, "exam": "last-week"}).json()
        assert {x["session"] for x in r["results"]} == {S3}
        assert c.get("/api/search", params={"q": "x", "exam": "midterm"}).status_code == 400  # needs the class


def test_units_api_manual_placement_and_guards(cls, cfg):
    app = create_app(cfg, autoprocess=False, monitor=False)
    with TestClient(app, base_url=BASE) as c:
        assert c.post(f"/api/classes/{SLUG}/units", json={"session": S2, "t": 90}).status_code == 403  # app header
        r = c.post(f"/api/classes/{SLUG}/units", headers=H, json={"session": S2, "t": 90, "name": "Markets"})
        assert r.status_code == 201
        uid = r.json()["id"]
        assert c.post(f"/api/classes/{SLUG}/units", headers=H, json={"session": S2, "t": "x"}).status_code == 400
        assert c.patch(f"/api/classes/{SLUG}/units/{uid}", headers=H, json={"name": ""}).status_code == 400
        assert c.patch(f"/api/classes/{SLUG}/units/{uid}", headers=H, json={"name": "Trade"}).status_code == 200
        units = c.get(f"/api/classes/{SLUG}/units").json()
        assert [u["name"] for u in units["units"]] == ["Unit 1", "Trade"] and len(units["sessions"]) == 3
        assert c.delete(f"/api/classes/{SLUG}/units/{uid}").status_code == 403
        assert c.delete(f"/api/classes/{SLUG}/units/{uid}", headers=H).status_code == 200
        assert c.get(f"/api/classes/nope/units").status_code == 404


def test_reindex_rebuilds_unit_membership_from_files(cls, cfg, split):
    idx = SearchIndex(cfg.data_dir, cfg)
    first = idx.units(SLUG)
    assert [u["name"] for u in first] == ["Cells", "Markets"]
    idx.rebuild()
    assert idx.units(SLUG) == first
    for f in cfg.data_dir.glob(DB_NAME + "*"):
        f.unlink()
    assert SearchIndex(cfg.data_dir, cfg).units(SLUG) == first and un.unit_spans(cls)[1]["spans"] == first[1]["spans"]
    un.remove_boundary(cls, split["u2"])  # a change in units.json shows up on the next search
    idx.exact("key idea")
    assert [u["name"] for u in idx.units(SLUG)] == ["Cells"]


# ------------------------------------------------------------------ exam material (LLM through the queue)
class EchoLLM:
    """Fake Ollama: answers only with the unit terms it was given, so any leak would show up."""

    def __init__(self, done_reason="stop"):
        self.bodies, self.done_reason = [], done_reason

    def __call__(self, request):
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        body = json.loads(request.content)
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        self.bodies.append(body)
        user = body["messages"][1]["content"].lower()
        seen = [t for t in U1_TERMS + U2_TERMS if t in user]
        if "write study notes" in user:
            text = "#### Topics\nOrganelles\n#### Key terms\n" + "\n".join(f"- {t}: meaning" for t in seen) + \
                   "\n#### Questions from students\n- is it on the test -> yes"
        elif "glossary" in user:
            text = "\n".join(f"- **{t}**: a defined term" for t in seen)
        else:
            text = "Covered: " + ", ".join(seen)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": text}, "done_reason": self.done_reason,
                                         "prompt_eval_count": len(user) // 4 + 50, "eval_count": 40, "done": True})


def echo_llm(fake, cfg):
    llm = LocalLLM.from_config(cfg.llm)
    llm._http = httpx.Client(base_url=llm.host, transport=httpx.MockTransport(fake))
    return llm


@pytest.mark.parametrize("unit,own,other", [("u1", U1_TERMS, U2_TERMS), ("u2", U2_TERMS, U1_TERMS)])
def test_r7_exam_material_uses_only_in_scope_text_with_m4_guards(cls, cfg, split, unit, own, other):
    exam = un.save_exam(cls, f"Exam {unit}", [split[unit]])
    d = un.exam_dir(cls, exam["id"])
    fake = EchoLLM()
    info = generate_exam_material(d, cfg, llm=echo_llm(fake, cfg))
    assert info["state"] == "done"
    prompts = "\n".join(b["messages"][1]["content"] for b in fake.bodies).lower()
    assert text_has(prompts, other) == [] and set(text_has(prompts, own)) == set(own)
    for b in fake.bodies:
        assert b["options"]["num_ctx"] == 8192 and b["truncate"] is False and b["think"] is False
    for r in info["requests"]:
        assert r["est_prompt_tokens"] <= 5000 and r["done_reason"] == "stop" and r["prompt_eval_count"] <= 5000
    md = (d / EXAM_MD).read_text(encoding="utf-8")
    assert text_has(md, other) == [] and "## Glossary" in md and f"**{own[0]}**" in md
    assert material_state(d) == "generated"


def test_exam_material_never_overwrites_user_edits_and_fails_loudly_on_cut_output(cls, cfg, split):
    exam = un.save_exam(cls, "Final", ["u1", split["u2"]])
    d = un.exam_dir(cls, exam["id"])
    generate_exam_material(d, cfg, llm=echo_llm(EchoLLM(), cfg))
    generate_exam_material(d, cfg, llm=echo_llm(EchoLLM(), cfg))  # unchanged since generated: may be refreshed
    (d / EXAM_MD).write_text((d / EXAM_MD).read_text(encoding="utf-8") + "\nMy own note.\n", encoding="utf-8")
    fake = EchoLLM()
    info = generate_exam_material(d, cfg, llm=echo_llm(fake, cfg))
    assert info["state"] == "skipped" and fake.bodies == [] and "My own note." in (d / EXAM_MD).read_text(encoding="utf-8")
    other = un.exam_dir(cls, un.save_exam(cls, "Quiz", ["u1"])["id"])
    with pytest.raises(OutputTruncated):
        generate_exam_material(other, cfg, llm=echo_llm(EchoLLM(done_reason="length"), cfg))
    assert read_json(other / EXAM_META)["state"] == "failed" and not (other / EXAM_MD).exists()


def test_exam_job_goes_through_the_queue_and_the_gpu_rule(cls, cfg, split, monkeypatch):
    from lecture_recorder import machine

    calls = []

    class R:
        returncode, stdout, stderr = 0, "", ""

    monkeypatch.setattr(q, "_run_job", lambda cmd, **kw: calls.append(cmd[cmd.index("--task") + 1]) or R())
    app = create_app(cfg, autoprocess=False, monitor=False)
    un.save_exam(cls, "Midterm", ["u1"])
    with TestClient(app, base_url=BASE) as c:
        assert c.post(f"/api/exams/{SLUG}/midterm/generate").status_code == 403
        r = c.post(f"/api/exams/{SLUG}/midterm/generate", headers=H)
        assert r.status_code == 202 and r.json()["state"] == "queued"
    monkeypatch.setattr(machine, "power_source", lambda: "battery")
    blocked = []
    q.run_worker(cfg.data_dir, blocked=blocked)
    assert calls == [] and blocked == ["on battery"]
    monkeypatch.setattr(machine, "power_source", lambda: "ac")
    q.run_worker(cfg.data_dir)
    assert calls == ["exam"]


# ------------------------------------------------------------------ download-only bundle
def test_bundle_excludes_student_speech_by_default_and_stays_in_scope(cls, cfg, split):
    exam = un.save_exam(cls, "Midterm", ["u1"])
    generate_exam_material(un.exam_dir(cls, exam["id"]), cfg, llm=echo_llm(EchoLLM(), cfg))
    default = build_bundle(cls, exam)
    assert STUDENT_LINE not in default and "roommate" not in default
    assert "Questions from students" not in default and "classmates did not agree" in default
    assert "Lecturer:" in default and text_has(default, U2_TERMS) == []
    opted = build_bundle(cls, exam, include_audience=True)
    assert STUDENT_LINE in opted and "Student:" in opted
    app = create_app(cfg, autoprocess=False, monitor=False)
    with TestClient(app, base_url=BASE) as c:
        r = c.get(f"/api/exams/{SLUG}/midterm/bundle")
        assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
        assert STUDENT_LINE not in r.text
        assert STUDENT_LINE in c.get(f"/api/exams/{SLUG}/midterm/bundle", params={"include_audience": "true"}).text


def test_ui_has_units_exams_and_bundle_opt_in():
    from pathlib import Path

    js = (Path(lecture_recorder.__file__).parent / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert "unit-suggestions" in js and "/units" in js and "/exams" in js and "include_audience" in js
    assert "Start a new unit here" in js and ".innerHTML" not in js
    assert re.search(r"include_audience[^\n]*false|checked: false", js) or "audience.checked" in js


# ------------------------------------------------------------------ with the real local models (GPU, main session)
@pytest.mark.gpu
@pytest.mark.ollama
def test_real_embeddings_suggest_the_boundary_and_scoped_meaning_search_stays_in_unit(cls, cfg, capsys):
    for s in (S1, S2, S3):  # through the real queue: own process, GPU lock, power rule, unload
        q.enqueue(cfg.data_dir, cls / "sessions" / s, q.EMBED)
    done = q.run_worker(cfg.data_dir)
    assert done and all(j["state"] == "done" for j in done), [j.get("error") for j in done or []]
    out = suggest_boundaries(cls, cfg)
    best = max(out["suggestions"], key=lambda s: s["score"])
    assert best["session"] == S2 and abs(best["t"] - BOUNDARY) <= 30, out
    u2 = un.add_boundary(cls, best["session"], best["t"])
    app = create_app(cfg, autoprocess=False, monitor=False)
    with TestClient(app, base_url=BASE) as c:
        for unit, query, other in (("u1", "money, prices and international trade", U2_TERMS),
                                   (u2["id"], "organelles inside living cells", U1_TERMS)):
            r = c.get("/api/search", params={"q": query, "mode": "concept", "class_slug": SLUG, "units": unit})
            assert r.status_code == 200, r.text
            res = r.json()["results"]
            assert res and {x["unit_id"] for x in res} == {unit}
            assert all(text_has(x["text"], other) == [] for x in res)
    with capsys.disabled():
        print(f"\nM6 suggestions (real embeddings): {out['suggestions']}")


@pytest.mark.gpu
@pytest.mark.ollama
def test_real_llm_exam_material_for_one_unit_has_nothing_from_the_other(cls, cfg, split, capsys):
    exam = un.save_exam(cls, "Midterm", ["u1"])
    d = un.exam_dir(cls, exam["id"])
    q.enqueue(cfg.data_dir, d, q.EXAM)
    done = q.run_worker(cfg.data_dir)
    assert [j["state"] for j in done] == ["done"], done[0].get("error")
    meta = read_json(d / EXAM_META)
    assert meta["state"] == "done" and meta["unloaded"] is True
    for r in meta["requests"]:
        assert r["num_ctx"] == 8192 and r["done_reason"] == "stop" and r["prompt_eval_count"] <= 5000
    md = (d / EXAM_MD).read_text(encoding="utf-8")
    assert text_has(md, U2_TERMS) == [] and len(text_has(md, U1_TERMS)) >= 3 and "## Glossary" in md
    with capsys.disabled():
        print(f"\nM6 exam material (real LLM): {len(meta['requests'])} requests, max prompt "
              f"{max(r['prompt_eval_count'] for r in meta['requests'])} tokens, unit-1 terms found: "
              f"{len(text_has(md, U1_TERMS))}/6, unit-2 terms: 0, VRAM {meta['vram_mb']}")


# ------------------------------------------------------------------ regressions
STRADDLE = 94.0  # inside lecture 2's first economics segment (90-99 s): its words fall on both sides


@pytest.mark.parametrize("edited", [False, True], ids=["word-timestamps", "edited-no-words"])
def test_a_segment_split_by_a_boundary_lands_on_each_side_the_same_way_everywhere(cls, cfg, edited):
    from lecture_recorder.search.text import token_spans, token_times
    from lecture_recorder.summary.exam import segments_in_span

    d = cls / "sessions" / S2
    if edited:  # the user edited the text: the words no longer spell it, so times are interpolated
        doc = read_json(d / "transcript.json")
        doc["segments"][9]["text"] = "The elasticity changes how buyers and sellers trade in markets every quarter now."
        write_json_atomic(d / "transcript.json", doc)
    u2 = un.add_boundary(cls, S2, STRADDLE)
    seg = read_json(d / "transcript.json")["segments"][9]
    toks = token_spans(seg["text"])
    times = token_times(seg, [t for t, _, _ in toks])
    before = {t for (t, _, _), x in zip(toks, times) if x < STRADDLE}
    after = {t for (t, _, _), x in zip(toks, times) if x >= STRADDLE}
    assert before and after
    econ_words = (before | after) - {"the", "and", "how", "in", "every"}
    idx = SearchIndex(cfg.data_dir, cfg)
    doc = read_json(d / "transcript.json")
    unit1_text = " ".join(s["text"] for s in segments_in_span(doc, 0.0, STRADDLE)).lower()
    unit2_text = " ".join(s["text"] for s in segments_in_span(doc, STRADDLE, None)).lower()
    for w in econ_words:
        found = idx.exact(w, SLUG, scope={"units": ["u1"]})["total"] > 0
        assert found == (w in before), (w, found)  # search: unit 1 has exactly the words said before
        assert (w in tokens_of(unit1_text)) == (w in before)  # material and bundle text: the same rule
        if w in after:
            assert w in tokens_of(unit2_text)
    for x in idx.exact("buyers", SLUG, scope={"units": ["u1"]})["results"]:
        ctx = x["context"]["before"] + x["context"]["match"] + x["context"]["after"]
        assert not (after - before) & tokens_of(ctx)  # context stops at the boundary
    embed_all(cfg, [cls / "sessions" / s for s in (S1, S2, S3)])
    qvec = np.asarray(hashed_vector(" ".join(U2_TERMS + U1_TERMS)), dtype=np.float32)
    for unit in ("u1", u2["id"]):
        for x in idx.concept(qvec, SLUG, k=50, scope={"units": [unit]}):
            assert not (x["session"] == S2 and x["start"] < STRADDLE < x["end"])  # straddling windows skipped


def tokens_of(text):
    from lecture_recorder.search.text import tokens

    return set(tokens(text))


def test_material_goes_stale_when_units_change_and_leaves_the_bundle(cls, cfg, split):
    exam = un.save_exam(cls, "Midterm", ["u1"])
    d = un.exam_dir(cls, exam["id"])
    generate_exam_material(d, cfg, llm=echo_llm(EchoLLM(), cfg))
    from lecture_recorder.summary.exam import material_is_stale

    assert not material_is_stale(d, cls, exam) and "## Exam study material" in build_bundle(cls, exam)
    un.remove_boundary(cls, split["u2"])  # unit 1 now covers lecture 3 too
    assert material_is_stale(d, cls, exam)
    b = build_bundle(cls, exam)
    assert "## Exam study material" not in b and "units changed since it was generated" in b
    app = create_app(cfg, autoprocess=False, monitor=False)
    with TestClient(app, base_url=BASE) as c:
        assert c.get(f"/api/exams/{SLUG}/midterm").json()["stale"] is True


def test_deleting_a_units_start_lecture_does_not_merge_it_silently(cls, cfg, split):
    import shutil

    shutil.rmtree(cls / "sessions" / S2)  # a temp test folder
    spans = {u["id"]: u for u in un.unit_spans(cls)}
    assert spans["u1"]["spans"] == [{"session": S1, "start": 0.0, "end": None}]
    assert spans[split["u2"]]["moved"] and spans[split["u2"]]["spans"] == [{"session": S3, "start": 0.0, "end": None}]
    app = create_app(cfg, autoprocess=False, monitor=False)
    with TestClient(app, base_url=BASE) as c:
        assert c.get(f"/api/classes/{SLUG}/units").json()["moved"] == ["Markets"]
        assert SearchIndex(cfg.data_dir, cfg).exact("tariff", SLUG, scope={"units": ["u1"]})["total"] == 0


def test_bad_scope_and_boundary_inputs_are_refused(cls, cfg, split):
    app = create_app(cfg, autoprocess=False, monitor=False)
    with TestClient(app, base_url=BASE) as c:
        for units in ("", ",", " ,"):
            r = c.get("/api/search", params={"q": "key idea", "class_slug": SLUG, "units": units})
            assert r.status_code == 400, units  # never widened to the whole class
        assert c.get("/api/search", params={"q": "x", "class_slug": SLUG, "date_from": "2026-09-01"}).status_code == 400
        for t in ("nan", "inf", "-inf"):
            assert c.post(f"/api/classes/{SLUG}/units", headers=H, json={"session": S3, "t": t}).status_code == 400, t
    assert all(isinstance(u["start"]["t"], float) for u in un.load_units(cls) if u.get("start"))


def test_concurrent_saves_keep_every_boundary_and_exam(cls):
    import threading

    errs = []

    def run(fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=run, args=(lambda i=i: un.add_boundary(cls, S3, 10.0 + 20 * i),)) for i in range(6)]
    ts += [threading.Thread(target=run, args=(lambda: un.save_exam(cls, "Final", ["u1"]),)) for _ in range(5)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert errs == []
    assert len([u for u in un.load_units(cls) if u.get("start")]) == 6
    assert sorted(e["id"] for e in un.load_exams(cls)) == ["final", "final-2", "final-3", "final-4", "final-5"]


@pytest.mark.parametrize("heading", ["#### Questions from students", "### Student questions:",
                                     "## Questions from the students", "#### Questions asked by students (Q&A)"])
def test_bundle_removes_any_student_questions_section(cls, cfg, split, heading):
    from lecture_recorder.summary.summarize import GENERATED_MARK

    (cls / "sessions" / S1 / "summary.md").write_text(
        f"{GENERATED_MARK}\n# Summary\n\n## At a glance\n\nCells.\n\n{heading}\n- Jordan asked about grades\n\n"
        "#### Key terms\n- ribosome: makes proteins\n", encoding="utf-8")
    b = build_bundle(cls, un.save_exam(cls, "Midterm", ["u1"]))
    assert "Jordan asked" not in b and "ribosome: makes proteins" in b


def test_whole_lectures_reuse_their_summary_notes_instead_of_new_map_requests(cls, cfg, split):
    from lecture_recorder.summary.summarize import GENERATED_MARK

    (cls / "sessions" / S1 / "summary.md").write_text(
        f"{GENERATED_MARK}\n# Summary: L1\n\n## At a glance\n\nCells.\n\n## Detailed notes\n\n"
        "### Part 1: 00:00:00 to 00:01:30\n\n#### Key terms\n- mitochondrion: organelle\n\n"
        "### Part 2: 00:01:30 to 00:02:59\n\n#### Key terms\n- ribosome: organelle\n", encoding="utf-8")
    exam = un.save_exam(cls, "Midterm", ["u1"])
    fake = EchoLLM()
    info = generate_exam_material(un.exam_dir(cls, exam["id"]), cfg, llm=echo_llm(fake, cfg))
    maps = [r for r in info["requests"] if r["stage"] == "map"]
    assert len(maps) == 1  # only lecture 2's in-unit part was mapped; lecture 1 reused its 2 parts
    reduce_prompt = next(b for b in fake.bodies if "at-a-glance" in b["messages"][1]["content"])["messages"][1]["content"]
    assert "Lecture of 2026-09-01" in reduce_prompt and "Lecture of 2026-09-03" in reduce_prompt
