"""Zoom recordings: pick the files to import from a recording folder, read its date and topic, and
match the speaker names in Zoom's transcript (.vtt) to the diarized speakers. Local only.

Local recordings (Documents\\Zoom\\<YYYY-MM-DD HH.MM.SS Topic MeetingID>\\): audio*.m4a (audio only),
video*.mp4, chat.txt or meeting_saved_chat.txt, optionally "Audio Record\\" with one m4a per participant.
Before Zoom converts them, there are only double_click_to_convert_NN.zoom files.
Cloud downloads: GMT<YYYYMMDD-HHMMSS>_Recording*.mp4 / .m4a, *.transcript.vtt, *.cc.vtt, *chat*.txt.
"""
import re
from datetime import datetime, timezone
from pathlib import PurePosixPath

TRANSCRIPT_FILE = "zoom_transcript.vtt"
CHAT_FILE = "zoom_chat.txt"
MAX_TEXT_BYTES = 20 * 2**20  # transcript / chat uploads

UNCONVERTED = ("Zoom hasn't converted this recording yet: the folder only has double_click_to_convert "
               "files. Double-click one of them (or open Zoom > Meetings > Recorded > Convert), wait until "
               "the m4a/mp4 files appear, then import the folder again.")

_LOCAL_DIR = re.compile(r"^(\d{4})-(\d{2})-(\d{2}) (\d{2})\.(\d{2})\.(\d{2})(?: (.*?))?(?: \d{9,11})?$")
_CLOUD_NAME = re.compile(r"GMT(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})")


def _parts(name: str) -> tuple[str, ...]:
    return PurePosixPath(name.replace("\\", "/")).parts


def plan(files: list[tuple[str, int]]) -> dict:
    """Which files of a picked Zoom folder to import. `files`: (path relative to the picked folder's
    parent, e.g. "2024-05-01 10.00.00 Physics/audio123.m4a", size in bytes), as the browser lists them.

    Returns {folder, audio, transcript, chat, participants, others, title, when (aware datetime or None), error}.
    Audio: the largest top-level m4a (audio only), else the largest top-level mp4. `others`: further
    top-level recordings of that kind (e.g. a recording stopped and restarted), not imported, so the UI says so.
    """
    rows = [(_parts(n), int(s)) for n, s in files if n]
    tops = {p[0] for p, _ in rows if len(p) > 1}
    folder = tops.pop() if len(tops) == 1 and all(len(p) > 1 for p, _ in rows) else ""
    rel = [(p[1:] if folder else p, s) for p, s in rows]
    top = [(p[0], s) for p, s in rel if len(p) == 1]
    out = {"folder": folder, "audio": None, "transcript": None, "chat": None,
           "participants": sum(1 for p, _ in rel if len(p) > 1 and p[-1].lower().endswith(".m4a")),
           "others": 0, "title": None, "when": None, "error": None}

    def pick(pred):
        found = [(s, n) for n, s in top if pred(n.lower())]
        return max(found)[1] if found else None

    out["audio"] = pick(lambda n: n.endswith(".m4a")) or pick(lambda n: n.endswith(".mp4"))
    if out["audio"]:
        kind = PurePosixPath(out["audio"]).suffix.lower()
        out["others"] = sum(1 for n, _ in top if n.lower().endswith(kind)) - 1
    out["transcript"] = pick(lambda n: n.endswith(".transcript.vtt")) or pick(lambda n: n.endswith(".vtt"))
    out["chat"] = pick(lambda n: n.endswith(".txt") and "chat" in n)
    if out["audio"] is None:
        if any(n.lower().endswith(".zoom") for n, _ in top):
            out["error"] = UNCONVERTED
        else:
            out["error"] = "No Zoom audio or video (m4a or mp4) in this folder. Pick the meeting's own folder."
    out["title"], out["when"] = folder_date_title(folder, out["audio"] or "")
    if folder:
        out["audio"], out["transcript"], out["chat"] = (f"{folder}/{n}" if n else None
                                                        for n in (out["audio"], out["transcript"], out["chat"]))
    return out


def folder_date_title(folder: str, audio_name: str = "") -> tuple[str | None, datetime | None]:
    """(topic, start time) from a local recording folder name (local time) or a cloud file name (UTC)."""
    m = _LOCAL_DIR.match(folder)
    if m:
        y, mo, d, h, mi, s = map(int, m.groups()[:6])
        try:
            return (m.group(7) or "").strip() or None, datetime(y, mo, d, h, mi, s).astimezone()
        except ValueError:
            pass
    m = _CLOUD_NAME.search(audio_name) or _CLOUD_NAME.search(folder)
    if m:
        try:
            when = datetime(*map(int, m.groups()), tzinfo=timezone.utc).astimezone()
            return (folder.strip() or None), when
        except ValueError:
            pass
    return (folder.strip() or None), None


# ---------------------------------------------------------------- transcript (.vtt) speaker names
_TIME = r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})"
_CUE = re.compile(rf"^\s*{_TIME}\s*-->\s*{_TIME}")


def _secs(h, m, s, ms) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_vtt(text: str) -> list[dict]:
    """[{start, end, name, text}] from a WebVTT file; `name` from Zoom's "Name: text" lines, else None."""
    cues, lines = [], text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        m = _CUE.match(lines[i])
        i += 1
        if not m:
            continue
        g = m.groups()
        body = []
        while i < len(lines) and lines[i].strip():
            body.append(lines[i].strip())
            i += 1
        said = " ".join(body)
        name, sep, rest = said.partition(": ")
        has_name = bool(sep) and 0 < len(name) <= 80 and "-->" not in name
        cues.append({"start": _secs(*g[:4]), "end": _secs(*g[4:]), "name": name.strip() if has_name else None,
                     "text": rest if has_name else said})
    return cues


def speaker_names(cues: list[dict], segments: list[dict], min_share: float = 0.5) -> list[dict]:
    """For each diarized speaker, the Zoom name that overlaps most of its speech, if it covers at least
    `min_share` of the overlapped time: [{speaker, name, share}]. A hint only; labels are not changed."""
    named = sorted((c for c in cues if c["name"]), key=lambda c: c["start"])
    overlap: dict[str, dict[str, float]] = {}
    j = 0
    for seg in sorted(segments, key=lambda s: s["start"]):
        while j < len(named) and named[j]["end"] <= seg["start"]:
            j += 1  # cues are visited once overall, plus the few that span segment boundaries
        k = j
        while k < len(named) and named[k]["start"] < seg["end"]:
            o = min(seg["end"], named[k]["end"]) - max(seg["start"], named[k]["start"])
            if o > 0:
                per = overlap.setdefault(seg["speaker"], {})
                per[named[k]["name"]] = per.get(named[k]["name"], 0.0) + o
            k += 1
    out = []
    for spk, per in overlap.items():
        name, best = max(per.items(), key=lambda kv: kv[1])
        share = best / sum(per.values())
        if share >= min_share:
            out.append({"speaker": spk, "name": name, "share": round(share, 2)})
    return sorted(out, key=lambda r: r["speaker"])
