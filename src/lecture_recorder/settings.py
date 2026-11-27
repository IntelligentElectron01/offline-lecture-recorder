"""User settings changed from the app, saved to config.toml in the app home.

Only a few simple values are editable (an allowlist); the file is edited line by line so the user's
other settings and comments stay as they are, then re-parsed and checked before it is replaced.
"""
import os
import re
import threading
import tomllib
import uuid
from pathlib import Path

from lecture_recorder.config import PROCESSING_MODES
from lecture_recorder.naming import NamingError, check_template


def _template_ok(v) -> bool:
    try:
        return check_template(v) == v
    except NamingError:
        return False


EDITABLE = {
    ("processing", "mode"): lambda v: isinstance(v, str) and v in PROCESSING_MODES,
    ("processing", "keep_awake"): lambda v: isinstance(v, bool),
    ("retention", "delete_original_after_opus"): lambda v: isinstance(v, bool),
    ("retention", "trash_days"): lambda v: isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 3650,
    ("server", "idle_quit_minutes"): lambda v: isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 1440,
    ("naming", "template"): _template_ok,  # saved already checked and trimmed (no control characters)
}


# Older names that config.toml may still hold: saving the new name also updates the old line if it is
# there, so the file never says two different things.
ALIASES = {("retention", "delete_original_after_opus"): ("retention", "delete_wav_after_opus")}


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


_lock = threading.Lock()  # read-modify-write of config.toml, one request at a time


def set_config_values(home: Path, values: dict[tuple[str, str], object]) -> None:
    with _lock:
        _set(home, values)


def _set(home: Path, values: dict[tuple[str, str], object]) -> None:
    for key, v in values.items():
        if key not in EDITABLE or not EDITABLE[key](v):
            raise ValueError(f"not an editable setting: {key} = {v!r}")
    path = Path(home) / "config.toml"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    if lines:
        old = tomllib.loads("\n".join(lines))
        values = {**values, **{ALIASES[k]: v for k, v in values.items()
                               if k in ALIASES and ALIASES[k][1] in old.get(ALIASES[k][0], {})}}
    for (section, name), v in values.items():
        line = f"{name} = {_toml_value(v)}"
        header = next((i for i, l in enumerate(lines) if re.fullmatch(rf"\s*\[{section}\]\s*(#.*)?", l)), None)
        if header is None:
            lines += ["", f"[{section}]", line]
            continue
        end = next((i for i in range(header + 1, len(lines)) if re.match(r"\s*\[", lines[i])), len(lines))
        found = next((i for i in range(header + 1, end) if re.match(rf"\s*{name}\s*=", lines[i])), None)
        if found is None:
            lines.insert(header + 1, line)
        else:
            lines[found] = line
    text = "\n".join(lines) + "\n"
    parsed = tomllib.loads(text)  # never write a file that no longer parses
    if any(parsed.get(section, {}).get(name) != v for (section, name), v in values.items()):
        raise ValueError("config.toml could not be updated safely (unexpected layout); edit it by hand")
    tmp = path.with_name(f"config.toml.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
