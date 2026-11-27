"""Imports whose file time is their export time. No "recorded at the same time" flag between imports, no
schedule evidence from batch times, and the date from the file name is shown instead. Synthetic, temp folders."""
import hashlib
from datetime import date, datetime

import pytest

from lecture_recorder import autosort as sort
from lecture_recorder import dates
from lecture_recorder import duplicates as dup
from lecture_recorder import library as lib
from lecture_recorder import units as un
from test_duplicates import SLUG, cfg, lecture, signals  # noqa: F401  (fixtures)
from test_autosort import PHY, weekly
from test_autosort import cfg as cfg3  # noqa: F401  (fixture)

LATEST = date(2026, 10, 2)


def imported(cfg, name, file, created, duration=3000.0, **meta):
    src = {"kind": "import", "file": file, "bytes": 1000, "sha256": hashlib.sha256(file.encode()).hexdigest()}
    return lecture(cfg, name, created=created, duration=duration, source=src, date_from="file metadata",
                   imported="2026-10-02T17:20:00+00:00", **meta)


def batch(cfg, n=5):
    """Five 50-min lectures exported together: file times 9 s apart (the user's report)."""
    for i in range(n):
        s = 41 + 9 * i
        imported(cfg, f"2026-10-02_17-{16 + s // 60:02d}-{s % 60:02d}", f"Course 101 9-{10 + i}-26.m4a",
                 f"2026-10-02T17:{16 + s // 60:02d}:{s % 60:02d}+00:00")


# ------------------------------------------------------------------ duplicates
def test_batch_exported_imports_are_not_flagged(cfg):
    batch(cfg)
    assert dup.for_class(cfg.data_dir, SLUG) == {}


def test_two_imports_never_overlap_by_time_even_with_trusted_times(cfg):
    imported(cfg, "2026-09-01_09-00-00", "memo.m4a", "2026-09-01T09:00:00+00:00")
    lecture(cfg, "2026-09-01_09-05-00", created="2026-09-01T09:05:00+00:00", date_from="Zoom folder name",
            source={"kind": "import", "file": "zoom.m4a", "zoom": True})
    assert dup.for_class(cfg.data_dir, SLUG) == {}


def test_a_mic_recording_and_an_overlapping_import_are_flagged(cfg):
    lecture(cfg, "2026-09-01_09-00-00")  # mic
    imported(cfg, "2026-09-01_09-05-00", "memo.m4a", "2026-09-01T09:05:00+00:00")
    assert signals(cfg, "2026-09-01_09-00-00") == ["time"]


def test_a_mic_recording_and_an_app_capture_are_flagged(cfg):
    lecture(cfg, "2026-09-01_09-00-00")
    lecture(cfg, "2026-09-01_09-01-00", created="2026-09-01T09:01:00+00:00", source={"kind": "app", "app_name": "Zoom"})
    assert signals(cfg, "2026-09-01_09-00-00") == ["time"]


def test_a_mic_recording_and_a_batch_import_are_not_flagged_by_time(cfg):
    batch(cfg, 2)
    lecture(cfg, "2026-10-02_17-00-00", created="2026-10-02T17:00:00+00:00")  # overlaps both export times
    assert dup.for_class(cfg.data_dir, SLUG) == {}


def test_a_name_date_that_disagrees_with_the_file_time_drops_it(cfg):
    lecture(cfg, "2026-10-02_17-00-00", created="2026-10-02T17:00:00+00:00")
    imported(cfg, "2026-10-02_17-10-00", "Course 101 9-24-26.m4a", "2026-10-02T17:10:00+00:00")
    assert dup.for_class(cfg.data_dir, SLUG) == {}
    shown = {s["name"]: s["created"] for s in lib.list_sessions(cfg.data_dir, SLUG)}
    assert shown["2026-10-02_17-10-00"] == "2026-09-24"


def test_the_file_signal_still_links_imports(cfg):
    src = {"kind": "import", "file": "a.m4a", "bytes": 10, "sha256": "ab" * 32}
    lecture(cfg, "2026-10-02_17-16-41", created="2026-10-02T17:16:41+00:00", source=src, date_from="file metadata")
    lecture(cfg, "2026-10-02_17-16-50", created="2026-10-02T17:16:50+00:00", source=src, date_from="file metadata")
    assert signals(cfg, "2026-10-02_17-16-41") == ["file"]


# ------------------------------------------------------------------ the date shown
def test_batch_imports_show_and_sort_by_the_date_in_their_file_names(cfg):
    batch(cfg)
    rows = lib.list_sessions(cfg.data_dir, SLUG)
    assert [r["created"] for r in rows] == [f"2026-09-{d}" for d in (14, 13, 12, 11, 10)]  # newest first
    assert [s["created"] for s in un.class_sessions(cfg.data_dir / "classes" / SLUG)] == [f"2026-09-{d}" for d in range(10, 15)]
    detail = lib.session_detail(cfg.data_dir, SLUG, "2026-10-02_17-16-41")
    assert detail["created"] == "2026-09-10"


def test_a_lone_import_with_an_agreeing_name_keeps_its_file_time(cfg):
    imported(cfg, "2026-09-24_10-00-00", "Notes 9-24-26.m4a", "2026-09-24T10:00:00+00:00")
    start, shown = dates.class_times(cfg.data_dir / "classes" / SLUG)["2026-09-24_10-00-00"]
    assert start == datetime.fromisoformat("2026-09-24T10:00:00+00:00") and shown == "2026-09-24T10:00:00+00:00"


def test_nothing_on_disk_is_rewritten(cfg):
    batch(cfg)
    metas = sorted((cfg.data_dir / "classes" / SLUG / "sessions").glob("*/meta.json"))
    before = [p.read_bytes() for p in metas]
    lib.list_sessions(cfg.data_dir, SLUG), dup.for_class(cfg.data_dir, SLUG), un.class_sessions(cfg.data_dir / "classes" / SLUG)
    assert [p.read_bytes() for p in metas] == before


@pytest.mark.parametrize("name, expected", [
    ("Course 101 9-24-26.m4a", date(2026, 9, 24)),
    ("Course 101 9-24-2026.m4a", date(2026, 9, 24)),
    ("2026-09-24 lecture.m4a", date(2026, 9, 24)),
    ("Notes 10_1_26.m4a", date(2026, 10, 1)),
    ("Notes 9-24.m4a", date(2026, 9, 24)),        # no year: the most recent one not after the file time
    ("Notes 12-24.m4a", date(2025, 12, 24)),      # after it this year, so last year
    ("Notes 10-13-26.m4a", None),                 # after the file time: a wrong date is worse than none
    ("Notes 13-40-26.m4a", None),                 # impossible
    ("Notes 2-30.m4a", None),
    ("Course 101 lecture.m4a", None),              # a course number is no date
    ("New Recording 12.m4a", None),
    ("Notes 9-24 and 9-26.m4a", None),            # two dates without a year: unsure
    ("", None), (None, None),
])
def test_name_dates(name, expected):
    assert dates.name_date(name, LATEST) == expected


# ------------------------------------------------------------------ class suggestions (schedule)
def test_schedule_ignores_batch_file_times(cfg3):
    from test_autosort import lecture as lecture18
    # Four imports into Physics, all exported on Tuesday evenings within seconds of each other: no pattern.
    for i, (day, sec) in enumerate([("2026-09-01", 1), ("2026-09-01", 9), ("2026-09-08", 1), ("2026-09-08", 9)]):
        lecture18(cfg3, PHY, f"{day}_20-28-{sec:02d}", created=f"{day}T20:28:{sec:02d}+00:00", date_from="file metadata",
                  source={"kind": "import", "file": f"phys {i}.m4a"})
    assert sort.suggest(sort._classes(cfg3.data_dir), datetime.fromisoformat("2026-09-15T20:30:00+00:00")) is None
    weekly(cfg3, PHY, at="10-00", weeks=3)  # the real (mic) schedule still counts
    s = sort.suggest(sort._classes(cfg3.data_dir), datetime.fromisoformat("2026-09-22T10:05:00+00:00"))
    assert s and s["class"] == PHY and s["signals"] == ["schedule"]
