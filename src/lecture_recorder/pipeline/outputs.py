"""Plain-file transcript outputs in the session folder.

transcript.raw.json  machine output, never modified (refuses to overwrite an existing one)
transcript.json      the editable copy (created once; later edits belong to the user)
transcript.md/.txt   readable renderings
"""
import os
from pathlib import Path

from lecture_recorder.recorder.session import write_json_atomic

RAW, EDITABLE, MD, TXT = "transcript.raw.json", "transcript.json", "transcript.md", "transcript.txt"


class TranscriptExists(FileExistsError):
    pass


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def render_txt(doc: dict) -> str:
    return "".join(f"[{hms(s['start'])}] {s['speaker']} ({s['role']}): {s['text']}\n" for s in doc["segments"])


def render_md(doc: dict, title: str) -> str:
    lines = [f"# Transcript: {title}", ""]
    for s in doc["speakers"]:
        lines.append(f"- **{s['id']}** ({s['role']}): {s['speech_s'] / 60:.1f} min of speech")
    lines.append("")
    for s in doc["segments"]:
        lines += [f"**[{hms(s['start'])}] {s['speaker']} ({s['role']}):** {s['text']}", ""]
    return "\n".join(lines)


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_outputs(session_dir: Path, doc: dict) -> list[Path]:
    raw = session_dir / RAW
    if raw.exists():
        raise TranscriptExists(f"{raw} already exists; it is never overwritten without the user's confirmation")
    paths = []
    for name, text in ((MD, render_md(doc, session_dir.name)), (TXT, render_txt(doc))):
        if not (session_dir / name).exists():
            write_text_atomic(session_dir / name, text)
            paths.append(session_dir / name)
    if not (session_dir / EDITABLE).exists():
        write_json_atomic(session_dir / EDITABLE, doc)
        paths.append(session_dir / EDITABLE)
    write_json_atomic(raw, doc)  # written last: its presence means the job completed
    return [raw, *paths]
