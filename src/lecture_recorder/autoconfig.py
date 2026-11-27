"""Machine auto-configuration: detect this computer, pick the processing settings, write config.auto.toml.

Runs at the end of setup, on the app's first start (no config.auto.toml yet), and on request
(`lecture-recorder configure [--dry-run]`). Fast (no model loads, no torch import, ~1 s) and
offline except the loopback Ollama check. Never downloads anything: missing models are reported with
the documented commands from README / docs/dependencies.md.

Precedence (config.load_config): built-in defaults < config.auto.toml (written here) < config.toml
(the user's). A value the user set in config.toml always wins; "auto" there means "use the pick".
config.toml is never written by this module.
"""
import json
import os
import shutil
import subprocess
import urllib.request
from datetime import datetime
from pathlib import Path

AUTO_FILE = "config.auto.toml"
MIN_VRAM_FOR_CUDA_MB = 4000  # transcription job peak ~2.3 GB on the GPU (3.3 GB device total, 1 GB desktop)
SUMMARY_VRAM_MB = 7000  # qwen3:8b at num_ctx 8192 measured 6.8 GB device total; below this Ollama splits it
MIN_RAM_GB, MIN_FREE_DISK_GB = 16, 20
TURBO, DISTIL = "large-v3-turbo", "distil-large-v3.5"
OLLAMA_MODELS = ("qwen3:8b", "qwen3-embedding:0.6b")
APPROVED_OLLAMA_VERSION = "0.35.0"  # listed in docs/dependencies.md; doctor warns on any other version


def _installer() -> str:
    from lecture_recorder.host import host

    return host().installer_name


def _script(name: str) -> str:
    from lecture_recorder.host import host

    return f"{host().venv_python} scripts{host().path_sep}{name}"


# Only the documented commands (docs/dependencies.md). Never run from here.
DOWNLOAD_HINTS = {
    TURBO: _script("download_models.py") + " whisper",
    DISTIL: _script("download_models.py") + " whisper",  # the same group has both
    "pyannote": _script("download_models.py") + f" pyannote  (needs your free Hugging Face token in .env; {_installer()} asks for it)",
    "qwen3:8b": "ollama pull qwen3:8b",
    "qwen3-embedding:0.6b": "ollama pull qwen3-embedding:0.6b",
    "ollama": f"Install Ollama v{APPROVED_OLLAMA_VERSION} from https://ollama.com/download ({_installer()} does this for you), then turn off its automatic updates",
}
# Settings this module may pick ([section] -> keys). Everything else stays the user's / the defaults.
AUTO_KEYS = {"transcription": ("model", "device", "compute_type", "batch_size", "cpu_threads")}


# ------------------------------------------------------------------ detection
def _ram_gb() -> float | None:
    from lecture_recorder.host import host

    return host().ram_gb()


def _has_battery() -> bool | None:
    from lecture_recorder.host import host

    return host().has_battery()


def _host_facts() -> dict:
    from lecture_recorder.host import host

    return host().facts()


def _gpu() -> dict | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    line = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else ""
    try:
        name, total = (x.strip() for x in line.rsplit(",", 1))
        return {"name": name, "vram_mb": int(float(total))}
    except ValueError:
        return None


def _cuda_usable() -> bool:
    from lecture_recorder.pipeline.gpu import cuda_available  # CTranslate2's device count: no model load

    return cuda_available()


def _ollama(host: str) -> dict:
    out = {"installed": shutil.which("ollama") is not None, "reachable": False, "models": [], "version": None}
    try:
        from lecture_recorder.summary.llm import check_local_host

        base, opener = check_local_host(host), urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(base + "/api/tags", timeout=1.5) as r:
            out["models"] = sorted(m.get("name", "") for m in json.load(r).get("models", []))
        out["reachable"] = True
        with opener.open(base + "/api/version", timeout=1.5) as r:
            out["version"] = str(json.load(r).get("version") or "") or None
    except Exception:
        pass
    return out


def detect(home: Path, models_dir: Path, ollama_host: str) -> dict:
    """What this computer has. Each probe fails soft (unknown), so detection never stops setup."""
    gpu = _gpu()
    try:
        disk = round(shutil.disk_usage(home).free / 2**30, 1)
    except OSError:
        disk = None
    return {
        "gpu": gpu,
        "cuda_usable": _cuda_usable() if gpu else False,
        "logical_cpus": os.cpu_count() or 1,
        "ram_gb": _ram_gb(),
        "disk_free_gb": disk,
        "battery": _has_battery(),
        "host": _host_facts(),
        "ollama": _ollama(ollama_host),
        "models": models_present(models_dir),
    }


def models_present(models_dir: Path) -> dict:
    """{'large-v3-turbo': bool, 'distil-large-v3.5': bool, 'pyannote': bool}: the model files on disk."""
    from lecture_recorder.pipeline.diarize import MODEL_DIR
    from lecture_recorder.pipeline.transcribe import MODEL_DIRS

    d = Path(models_dir)
    return {**{m: (d / sub / "model.bin").exists() for m, sub in MODEL_DIRS.items()},
            "pyannote": (d / MODEL_DIR / "config.yaml").exists()}


def pick_stt(cuda_usable: bool, vram_mb: int, models: dict, logical_cpus: int, perf_cores: int | None = None) -> dict:
    """The transcription picks: the one rule used by `configure` and by choose()'s "auto" values.

    large-v3-turbo with sequential decoding everywhere (the most accurate; on CPU int8 it runs at
    RTF ~0.25 on the reference laptop with the same accuracy, and distil batched was barely faster and
    less accurate). CUDA when CTranslate2 sees a GPU with >= 4 GB. distil only if
    turbo is not downloaded, then batched (sequential distil silently skips speech).
    """
    cuda = bool(cuda_usable) and (vram_mb or 0) >= MIN_VRAM_FOR_CUDA_MB
    model = TURBO if models.get(TURBO) or not models.get(DISTIL) else DISTIL
    return {"model": model, "device": "cuda" if cuda else "cpu", "compute_type": "int8_float16" if cuda else "int8",
            "batch_size": 8 if model == DISTIL else 0,
            # CTranslate2 and pyannote's torch threads on CPU: 4 measured within 5-20% of 8, and jobs run
            # in the background at below-normal priority, so a few cores stay free for the user.
            "cpu_threads": default_threads(logical_cpus, perf_cores)}


def default_threads(logical_cpus: int | None = None, perf_cores: int | None = None) -> int:
    if perf_cores:
        # Apple Silicon: the performance cores do the work and the efficiency cores stay free for the
        # rest of the Mac. Not measured here (no Mac was available): the cap of 6 is a cautious guess.
        return max(2, min(6, perf_cores))
    return max(1, min(4, (logical_cpus or os.cpu_count() or 2) // 2))


# ------------------------------------------------------------------ picks
def pick(profile: dict) -> dict:
    """Settings for this profile (pick_stt), plus notes and the commands for anything missing."""
    gpu = profile.get("gpu") or {}
    models = profile.get("models", {})
    facts = profile.get("host") or {}
    stt = pick_stt(profile.get("cuda_usable"), gpu.get("vram_mb", 0), models, profile.get("logical_cpus", 2),
                   facts.get("perf_cores"))
    cuda, model = stt["device"] == "cuda", stt["model"]
    settings = {"transcription": stt}
    notes, missing = [], []
    if gpu and not cuda:
        notes.append(f"GPU {gpu.get('name')} ({gpu.get('vram_mb')} MB) is not used for transcription "
                     + ("(CUDA not usable: update the NVIDIA driver)." if not profile.get("cuda_usable")
                        else f"(needs >= {MIN_VRAM_FOR_CUDA_MB} MB)."))
    if facts.get("os") == "macos":
        if facts.get("arch") not in (None, "arm64"):
            notes.append(f"This Mac is {facts.get('arch')}: only Apple Silicon (M1 or newer) is supported.")
        if facts.get("rosetta"):
            notes.append("Python is running under Rosetta (translated). Install the native arm64 Python 3.13 for full speed.")
        notes.append("Apple Silicon: transcription and speaker labels run on the CPU (int8; no GPU is used). The speed has not "
                     "been measured on a Mac yet: expect roughly an hour or more per lecture; process while plugged in.")
    elif not cuda:
        notes.append("Transcription runs on the CPU: a 90-min lecture takes roughly 1-3 hours (reference laptop ~70 min "
                     "with speaker labels); process at home, plugged in.")
    if model == DISTIL:
        notes.append("Using the fallback model (turbo not downloaded) with batched decoding; the missing-speech check "
                     "is approximate then.")
    if cuda and gpu.get("vram_mb", 0) < SUMMARY_VRAM_MB:
        notes.append("Summaries: the local AI model is larger than this GPU's memory, so Ollama runs part of it on "
                     "the CPU by itself (slower, same result).")
    if (profile.get("ram_gb") or 99) < MIN_RAM_GB:
        notes.append(f"{profile['ram_gb']} GB RAM: {MIN_RAM_GB} GB or more is recommended (the summary model needs ~6 GB).")
    if (profile.get("disk_free_gb") or 999) < MIN_FREE_DISK_GB:
        notes.append(f"Only {profile['disk_free_gb']} GB free disk space ({MIN_FREE_DISK_GB} GB recommended).")
    if profile.get("battery"):
        notes.append("Laptop with a battery: processing waits until it is plugged in (by design).")
    for key in (TURBO, "pyannote"):
        if not models.get(key):
            missing.append(DOWNLOAD_HINTS[key])
    oll = profile.get("ollama", {})
    if not oll.get("installed") and not oll.get("reachable"):
        missing.append(DOWNLOAD_HINTS["ollama"])
    elif not oll.get("reachable"):
        from lecture_recorder.host import host

        notes.append(f"Ollama is installed but not running: {host().ollama_start_hint}, then run configure again.")
    else:
        missing += [DOWNLOAD_HINTS[m] for m in OLLAMA_MODELS if m not in oll.get("models", [])]
    return {"settings": settings, "notes": notes, "missing": sorted(set(missing), key=missing.index)}


# ------------------------------------------------------------------ files
def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return json.dumps(str(v))  # a JSON string is a valid TOML basic string


def render(result: dict, profile: dict) -> str:
    gpu = profile.get("gpu") or {}
    lines = ["# Written by `lecture-recorder configure` for this computer. Do not edit: set your own values",
             "# in config.toml instead (they always win). Delete this file to re-detect on the next start.",
             f"# Detected {datetime.now().astimezone().isoformat(timespec='seconds')}: "
             f"GPU {gpu.get('name', 'none')} {gpu.get('vram_mb', 0)} MB (CUDA {'yes' if profile.get('cuda_usable') else 'no'}), "
             f"{profile.get('logical_cpus')} logical CPUs, {profile.get('ram_gb')} GB RAM, battery {profile.get('battery')}"]
    for section, values in result["settings"].items():
        lines += ["", f"[{section}]"] + [f"{k} = {_toml_value(v)}" for k, v in values.items()]
    return "\n".join(lines) + "\n"


def user_overrides(home: Path) -> dict:
    """Auto keys the user set to something other than "auto" in config.toml (config._merge's rule:
    those win over the picks). {} if config.toml is unreadable (load_config reports that itself)."""
    from lecture_recorder.config import _read_toml

    try:
        raw = _read_toml(Path(home) / "config.toml")
    except Exception:
        return {}
    return {f"{s}.{k}": raw[s][k] for s, keys in AUTO_KEYS.items() for k in keys
            if isinstance(raw.get(s), dict) and k in raw[s] and raw[s][k] != "auto"}


def configure(home: Path, dry_run: bool = False) -> dict:
    """Detect, pick, and (unless dry_run) write config.auto.toml atomically. Returns everything shown."""
    from lecture_recorder.config import load_config
    from lecture_recorder.pipeline.outputs import write_text_atomic

    cfg = load_config(home)
    profile = detect(cfg.home, cfg.models_dir, cfg.llm.ollama_host)
    result = pick(profile)
    result.update(profile=profile, overrides=user_overrides(cfg.home), path=str(cfg.home / AUTO_FILE),
                  written=False)
    if not dry_run:
        write_text_atomic(cfg.home / AUTO_FILE, render(result, profile))
        result["written"] = True
    return result


def describe(result: dict) -> list[str]:
    """Human-readable lines for the terminal (configure, doctor)."""
    out = []
    for section, values in result["settings"].items():
        for k, v in values.items():
            key = f"{section}.{k}"
            own = result["overrides"].get(key)
            out.append(f"  {key} = {v}" + (f"   (config.toml sets {own!r}, which is used)" if own is not None else ""))
    out += [f"  note: {n}" for n in result["notes"]]
    out += [f"  missing, run: {m}" for m in result["missing"]]
    return out


def read_auto(home: Path) -> dict:
    """The picks in config.auto.toml, for the Settings page and doctor ({} if not configured yet)."""
    from lecture_recorder.config import _read_toml

    return _read_toml(Path(home) / AUTO_FILE, generated=True)
