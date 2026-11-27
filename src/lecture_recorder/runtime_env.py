"""Network lockdown and environment setup. Import this BEFORE torch, pyannote, huggingface_hub.

The app never talks to the internet at runtime: telemetry is off and Hugging Face is forced
offline. Only scripts/download_models.py (run by the user during setup) may download.
"""
import os
import re
import sys
from pathlib import Path

SECRET_NAME = re.compile(r"TOKEN|SECRET|PASSWORD|API_?KEY|CREDENTIAL", re.I)
LOCKDOWN_ENV = {
    "PYANNOTE_METRICS_ENABLED": "0",
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_UPDATE_CHECK": "1",
    "DO_NOT_TRACK": "1",
    "TRANSFORMERS_OFFLINE": "1",
    # pyannote passes weights_only=False to torch.load; this forces the safe unpickler anyway.
    "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1",
}


def app_home() -> Path:
    """Folder holding .env, config.toml, data/ and models/.

    LECTURE_RECORDER_HOME if set; else the folder containing the running `.venv` (setup.bat
    always creates it there), so the launcher works from any terminal; else the current
    directory. Works the same for editable and normal installs; nothing is hard-coded.
    """
    if os.environ.get("LECTURE_RECORDER_HOME"):
        return Path(os.environ["LECTURE_RECORDER_HOME"]).resolve()
    prefix = Path(sys.prefix)
    if prefix.name == ".venv":
        return prefix.parent.resolve()
    return Path.cwd().resolve()


def apply_lockdown() -> None:
    """Force offline/telemetry-off settings. Overrides any values from the user's shell or .env."""
    from dotenv import dotenv_values

    # .env's other settings still apply, but secrets stay in the file: the app never needs HF_TOKEN (it runs
    # offline; only scripts/download_models.py reads it, from .env itself), and a token in os.environ would
    # be inherited by every child process.
    for k, v in dotenv_values(app_home() / ".env").items():
        if v is not None and k not in os.environ and not SECRET_NAME.search(k):
            os.environ[k] = v
    os.environ.update(LOCKDOWN_ENV)
    for key in [k for k in os.environ if k.startswith("OTEL_")]:
        del os.environ[key]  # no OpenTelemetry exporters may be configured
    os.environ["OTEL_SDK_DISABLED"] = "true"


def disable_library_telemetry() -> None:
    """Call after heavy imports are available."""
    try:
        import onnxruntime

        onnxruntime.disable_telemetry_events()
    except Exception:  # onnxruntime missing or built without telemetry: nothing to disable
        pass


def shown(path) -> str:
    """A path for terminal output people may paste into a bug report: the user-profile folder (which holds
    the account name) is shown as %USERPROFILE% (Windows) or ~."""
    text, home = str(path), str(Path.home())
    if home and text.lower().startswith(home.lower()) and (len(text) == len(home) or text[len(home)] in "\\/"):
        return ("%USERPROFILE%" if os.name == "nt" else "~") + text[len(home):]
    return text
