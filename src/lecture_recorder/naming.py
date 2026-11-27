"""Lecture titles: validating a title the user types, and the default naming template.

A title lives only in meta.json `title`: the folder name and `source.file` never change.
Template placeholders: {class} (the class's name), {date} and {time} (the recording's start, from the
session folder name: for imports, the date read from the file), {n} (1-based position of the lecture
in its class by recording date), {original} (the imported file's name without extension; empty for
recordings made in the app). An empty template keeps the built-in names.
"""
import re
import string
import unicodedata
from datetime import datetime
from pathlib import Path

from lecture_recorder.recorder.session import SESSION_NAME_FMT, read_json, session_dirs

MAX_TITLE = 200
MAX_TEMPLATE = 200
PLACEHOLDERS = ("class", "date", "time", "n", "original")
# Characters that are never part of a title: control characters, line/paragraph separators and the
# bidirectional overrides/isolates (they can make a title display as different text), and lone UTF-16
# surrogates (a Windows file name can hold one; it can't be stored as UTF-8 in the index or a page).
_BAD = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029\u202a-\u202e\u2066-\u2069\ud800-\udfff]")
_TRIM = " \t-–—_·,:;|/"  # dangling separators left by an empty placeholder, e.g. "{class} - {original}"


class NamingError(ValueError):
    pass


def clean_title(raw) -> str:
    """The title to store, or NamingError. Trimmed, NFC-normalized, at most MAX_TITLE characters,
    no control characters; any other Unicode (accents, CJK, emoji) is allowed."""
    if not isinstance(raw, str):
        raise NamingError("The title must be text.")
    title = unicodedata.normalize("NFC", raw).strip()
    if not title:
        raise NamingError("The title can't be empty.")
    if _BAD.search(title):
        raise NamingError("The title can't contain line breaks, tabs or other control characters.")
    if len(title) > MAX_TITLE:
        raise NamingError(f"The title is too long ({len(title)} characters; at most {MAX_TITLE}).")
    return title


def check_template(template) -> str:
    """The template to save (trimmed; "" = built-in names), or NamingError with what to fix."""
    if not isinstance(template, str):
        raise NamingError("The template must be text.")
    template = unicodedata.normalize("NFC", template).strip()
    if len(template) > MAX_TEMPLATE:
        raise NamingError(f"The template is too long ({len(template)} characters; at most {MAX_TEMPLATE}).")
    if _BAD.search(template):
        raise NamingError("The template can't contain line breaks, tabs or other control characters.")
    _parse(template)
    return template


def _parse(template: str) -> list[tuple[str, str | None]]:
    allowed = ", ".join(f"{{{p}}}" for p in PLACEHOLDERS)
    try:
        parts = list(string.Formatter().parse(template))
    except ValueError:
        raise NamingError("A { or } has no partner. To show a brace itself, write {{ or }}.")
    for _, field, spec, conv in parts:
        if field is None:
            continue
        if field not in PLACEHOLDERS:
            raise NamingError(f"Unknown placeholder {{{field}}}. Use {allowed}." if field
                              else f"Empty placeholder {{}}. Use {allowed}.")
        if spec or conv:
            raise NamingError(f"Write the placeholder as {{{field}}}, without anything after it inside the braces.")
    return [(literal, field) for literal, field, _, _ in parts]


def render(template: str, values: dict[str, str]) -> str | None:
    """The title from a valid template, or None if it comes out empty (then the built-in name is used)."""
    out = "".join(literal + (_BAD.sub("", values.get(field, "")) if field else "")
                  for literal, field in _parse(template))
    out = re.sub(r"\s+", " ", out).strip(_TRIM)
    return out[:MAX_TITLE].strip() or None


def values(started: datetime, class_name: str, n: int, original: str = "") -> dict[str, str]:
    """Placeholder values (also used by the Settings preview, so it matches the titles saved)."""
    return {"class": class_name, "date": started.strftime("%Y-%m-%d"), "time": started.strftime("%H:%M"),
            "n": str(n), "original": Path(original).stem if original else ""}


def values_for(session_dir: Path, class_name: str, original: str = "") -> dict[str, str]:
    """Placeholder values for a session folder that was just created (named after its start date-time)."""
    started = datetime.strptime(session_dir.name[:19], SESSION_NAME_FMT)
    names = [d.name for d in session_dirs(session_dir.parent.parent)]
    n = names.index(session_dir.name) + 1 if session_dir.name in names else len(names) + 1
    return values(started, class_name, n, original)


def title_for(template: str, session_dir: Path, original: str = "") -> str | None:
    """The templated title for a new session, or None for the built-in name (no template, or empty result)."""
    if not template:
        return None
    class_dir = session_dir.parent.parent
    name = (read_json(class_dir / "class.json", {}) or {}).get("name") or class_dir.name
    return render(template, values_for(session_dir, str(name), original))
