"""The "Record from an app" picker: apps with an audio session on the default output, Zoom first."""
from pathlib import Path

from lecture_recorder.recorder import winaudio

MEETING_APPS = ("zoom.exe",)  # listed first when present


def _root(pid: int, table: dict[int, tuple[int, str]]) -> int:
    """The top-most ancestor with the same exe name: e.g. the main Zoom.exe when audio plays from a
    helper Zoom.exe. Capture includes the whole tree under it, and its exit ends the recording."""
    exe = table.get(pid, (0, ""))[1].lower()
    seen = {pid}
    while True:
        parent = table.get(pid, (0, ""))[0]
        if parent in seen or parent not in table or table[parent][1].lower() != exe:
            return pid
        seen.add(parent)
        pid = parent


def list_audio_apps() -> list[dict]:
    """[{pid, exe, name, active}] one row per app, Zoom first, then playing apps, then by name.
    `active` = the app has an open audio stream right now (e.g. Zoom while in a meeting)."""
    table = winaudio.process_table()
    apps: dict[int, dict] = {}
    for pid, active in winaudio.audio_sessions():
        if pid not in table:
            continue
        root = _root(pid, table)
        row = apps.get(root)
        if row is None:
            exe = table[root][1]
            name = winaudio.friendly_name(winaudio.exe_path(root)) or Path(exe).stem
            row = apps[root] = {"pid": root, "exe": exe, "name": name, "active": False}
        row["active"] = row["active"] or active
    return sorted(apps.values(), key=lambda a: (a["exe"].lower() not in MEETING_APPS, not a["active"], a["name"].lower()))


def find_app(pid: int) -> dict | None:
    """The picker row for `pid` if it still has an audio session (validates what the UI sent)."""
    return next((a for a in list_audio_apps() if a["pid"] == pid), None)
