"""The units timeline. Backend additions only (the UI is checked in the browser): lecture dates in
/units, and Undo of a removed boundary, which brings the unit back under its old id so exams that list
it cover it again. Synthetic lectures in temp folders only; nothing in data/.
"""
import re
from pathlib import Path

from fastapi.testclient import TestClient

import lecture_recorder
from lecture_recorder import units as un
from lecture_recorder.summary.exam import generate_exam_material, material_is_stale
from lecture_recorder.web.app import create_app
from test_units import BASE, BOUNDARY, H, S1, S2, S3, SLUG, EchoLLM, cfg, cls, echo_llm, split  # noqa: F401


def test_units_api_lists_lecture_dates_and_durations_in_order(cls, cfg):
    app = create_app(cfg, autoprocess=False, monitor=False)
    with TestClient(app, base_url=BASE) as c:
        sessions = c.get(f"/api/classes/{SLUG}/units").json()["sessions"]
    assert [s["name"] for s in sessions] == [S1, S2, S3]
    assert sessions[0]["created"] == "2026-09-01T09:00:00+00:00" and sessions[0]["duration_s"] > 0


def test_a_gap_insert_at_a_lecture_start_is_named_unit_n_and_undo_removes_it(cls, cfg):
    app = create_app(cfg, autoprocess=False, monitor=False)
    with TestClient(app, base_url=BASE) as c:
        r = c.post(f"/api/classes/{SLUG}/units", headers=H, json={"session": S3, "t": 0})
        assert r.status_code == 201 and r.json()["name"] == "Unit 2"  # the name field shows this, selected
        uid = r.json()["id"]
        spans = c.get(f"/api/classes/{SLUG}/units").json()["units"]
        assert [sp["session"] for sp in spans[1]["spans"]] == [S3]
        assert c.delete(f"/api/classes/{SLUG}/units/{uid}", headers=H).status_code == 200  # Undo
        assert len(c.get(f"/api/classes/{SLUG}/units").json()["units"]) == 1


def test_undo_of_a_removal_restores_the_same_unit_name_and_exam_scope(cls, cfg, split):
    exam = un.save_exam(cls, "Final", [split["u2"]])
    d = un.exam_dir(cls, exam["id"])
    generate_exam_material(d, cfg, llm=echo_llm(EchoLLM(), cfg))
    before = un.unit_spans(cls)
    app = create_app(cfg, autoprocess=False, monitor=False)
    with TestClient(app, base_url=BASE) as c:
        assert c.delete(f"/api/classes/{SLUG}/units/{split['u2']}", headers=H).status_code == 200
        assert material_is_stale(d, cls, exam)
        # the UI re-POSTs the removed unit's session, t, name and id
        r = c.post(f"/api/classes/{SLUG}/units", headers=H,
                   json={"session": S2, "t": BOUNDARY, "name": "Markets", "id": split["u2"]})
        assert r.status_code == 201 and r.json()["id"] == split["u2"]
        assert c.get(f"/api/exams/{SLUG}/{exam['id']}").json()["scope_units"][0]["name"] == "Markets"
    assert un.unit_spans(cls) == before and not material_is_stale(d, cls, exam)


def test_undo_id_is_validated_and_never_duplicates_a_unit(cls, cfg, split):
    app = create_app(cfg, autoprocess=False, monitor=False)
    with TestClient(app, base_url=BASE) as c:
        post = lambda body: c.post(f"/api/classes/{SLUG}/units", headers=H, json=body)  # noqa: E731
        for bad in ("u1", "../x", "u-XYZ12345", "u-1234567", 5):
            assert post({"session": S3, "t": 0, "id": bad}).status_code == 400, bad
        assert post({"session": S3, "t": 0, "id": split["u2"]}).status_code == 400  # it exists already
        assert post({"session": S3, "t": 0, "id": "u-0123abcd"}).status_code == 201
    ids = [u["id"] for u in un.load_units(cls)]
    assert len(ids) == len(set(ids)) == 3


def test_ui_timeline_has_labelled_gap_buttons_undo_and_no_new_polling():
    js = (Path(lecture_recorder.__file__).parent / "web" / "static" / "app.js").read_text(encoding="utf-8")
    css = (Path(lecture_recorder.__file__).parent / "web" / "static" / "app.css").read_text(encoding="utf-8")
    m = re.search(r"function unitsCard[\s\S]*?(?=\nfunction examsCard)", js)
    assert m, "unitsCard not found"
    card = m.group(0)
    assert "Start a new unit before " in card and "Remove this boundary (merge into previous unit)" in card
    assert "Undo" in card and "Suggest boundaries" in card and "Clear suggestions" in card
    assert "startPoll" not in card and "setInterval" not in card and ".innerHTML" not in js
    assert "Remove boundary" not in card and "Add boundary" not in card  # the old per-row buttons are gone
    assert "(hover: none)" in css and "prefers-reduced-motion" in css and ".gap-btn" in css


def test_a_boundary_in_the_first_second_of_a_lecture_is_its_start(cls):
    """Regression: the lecture page posts the playback time; 0.4 s drew a '0:00-0:00' part and a dead gap."""
    u = un.add_boundary(cls, S3, 0.4)
    assert u["start"] == {"session": S3, "t": 0.0}
    spans = un.unit_spans(cls)
    assert [sp["session"] for sp in spans[0]["spans"]] == [S1, S2] and spans[1]["spans"][0]["start"] == 0.0
