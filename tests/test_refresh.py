"""After a delete, restore or permanent delete, every endpoint the UI re-fetches reflects it, and the
status poll's trash_version tells open pages (other tabs, the terminal) to re-render. Temp folders only.

The browser side (re-render, "This lecture was deleted" page, polls stopping) is checked on the demo
folder; these tests pin the data it relies on.
"""
from lecture_recorder import cli, trash
from lecture_recorder import units as un
from test_delete import H, cfg, client, cls  # noqa: F401  (fixtures)
from test_units import S1, S2, S3, SLUG


def version(client) -> str:
    return client.get("/api/record/status").json()["trash_version"]


def ui_view(client, exam_id: str) -> dict:
    """What the Library, class, units, exam, search and Recently deleted views show (their own requests)."""
    classes = {c["slug"]: c["sessions"] for c in client.get("/api/classes").json()}
    c = client.get(f"/api/classes/{SLUG}")
    exam = client.get(f"/api/exams/{SLUG}/{exam_id}")
    units = client.get(f"/api/classes/{SLUG}/units")
    return {
        "library": classes.get(SLUG),
        "class": [s["name"] for s in c.json()["sessions"]] if c.status_code == 200 else c.status_code,
        "units": sorted({sp["session"] for u in units.json()["units"] for sp in u["spans"]}) if units.status_code == 200 else units.status_code,
        "exam": sorted({sp["session"] for u in exam.json()["scope_units"] for sp in u["spans"]}) if exam.status_code == 200 else exam.status_code,
        "search": sorted({r["session"] for r in client.get("/api/search", params={"q": "the"}).json()["results"]}),
        "trash": [(x["kind"], x.get("session") or x["class"]) for x in client.get("/api/trash").json()["items"]],
    }


def test_a_deleted_lecture_leaves_every_view_and_comes_back_on_restore(cfg, cls, client):
    un.add_boundary(cls, S3, 0, "Unit 2")
    exam = client.post(f"/api/classes/{SLUG}/exams", headers=H, json={
        "name": "Final", "units": [u["id"] for u in un.load_units(cls)]}).json()
    before = ui_view(client, exam["id"])
    assert before == {"library": 3, "class": [S3, S2, S1], "units": [S1, S2, S3], "exam": [S1, S2, S3],
                      "search": [S1, S2, S3], "trash": []}
    v0 = version(client)
    assert version(client) == v0  # reads never change it

    r = client.post(f"/api/sessions/{SLUG}/{S3}/delete", headers=H).json()
    assert r["trash_version"] == version(client) != v0  # the deleting page takes it, other pages see it change
    after = ui_view(client, exam["id"])
    assert after == {"library": 2, "class": [S2, S1], "units": [S1, S2], "exam": [S1, S2],
                     "search": [S1, S2], "trash": [("session", S3)]}
    gone = client.get(f"/api/sessions/{SLUG}/{S3}")  # its own page: 404, and the trash names it (UI: "was deleted")
    assert gone.status_code == 404
    assert client.get(f"/api/sessions/{SLUG}/{S3}/transcript").status_code == 404

    v1 = version(client)
    restored = client.post(f"/api/trash/{r['id']}/restore", headers=H).json()
    assert restored["trash_version"] == version(client) != v1
    assert ui_view(client, exam["id"]) == before
    assert client.get(f"/api/sessions/{SLUG}/{S3}").status_code == 200

    tid = client.post(f"/api/sessions/{SLUG}/{S3}/delete", headers=H).json()["id"]
    v2 = version(client)
    d = client.post(f"/api/trash/{tid}/delete", headers=H, json={"confirm": trash.list_trash(cfg.data_dir)[0]["title"]}).json()
    assert d["trash_version"] == version(client) != v2
    assert ui_view(client, exam["id"])["trash"] == [] and client.get(f"/api/sessions/{SLUG}/{S3}").status_code == 404


def test_a_deleted_class_leaves_every_view_and_comes_back_on_restore(cfg, cls, client):
    exam = client.post(f"/api/classes/{SLUG}/exams", headers=H, json={
        "name": "Final", "units": [u["id"] for u in un.load_units(cls)]}).json()
    before = ui_view(client, exam["id"])
    v0 = version(client)
    tid = client.post(f"/api/classes/{SLUG}/delete", headers=H, json={"sessions": 3}).json()["id"]
    assert version(client) != v0
    assert ui_view(client, exam["id"]) == {"library": None, "class": 404, "units": 404, "exam": 404,
                                           "search": [], "trash": [("class", SLUG)]}
    assert client.get(f"/api/sessions/{SLUG}/{S1}").status_code == 404
    client.post(f"/api/trash/{tid}/restore", headers=H)
    assert ui_view(client, exam["id"]) == before


def test_terminal_deletes_reach_open_pages_through_the_status_poll(cfg, cls, client, capsys):
    v0 = version(client)
    assert cli.main(["delete", SLUG, S2, "--data-dir", str(cfg.data_dir)]) == 0  # another process, same files
    v1 = version(client)
    assert v1 != v0 and S2 not in [s["name"] for s in client.get(f"/api/classes/{SLUG}").json()["sessions"]]
    assert trash.purge_expired(cfg.data_dir, 0) == 0 and version(client) == v1  # nothing removed: no change
    assert cli.main(["restore", trash.list_trash(cfg.data_dir)[0]["id"], "--data-dir", str(cfg.data_dir)]) == 0
    assert version(client) != v1


def test_a_refused_delete_still_signals_and_leaves_no_trash_item(cfg, cls, client):
    v0 = version(client)
    assert client.post(f"/api/classes/{SLUG}/delete", headers=H, json={"sessions": 2}).status_code == 409
    assert version(client) != v0  # costs open pages one re-render; a missed change would leave them stale
    assert trash.list_trash(cfg.data_dir) == [] and (cls / "sessions" / S1).is_dir()
