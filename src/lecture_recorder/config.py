"""App configuration: built-in defaults < `config.auto.toml` (this machine's picks, written by
`lecture-recorder configure`) < `config.toml` (the user's; "auto" there means "use the pick").

All paths in the config are relative to the app home, so a copied install keeps working.
"""
import dataclasses
import tomllib
from dataclasses import dataclass
from pathlib import Path

from lecture_recorder.runtime_env import app_home


@dataclass(frozen=True)
class RecordingConfig:
    sample_rate: int = 16000
    max_length_minutes: float = 120
    silence_pause_minutes: float = 10
    silence_dbfs: float = -50.0  # RMS below this counts as near-silence
    no_audio_dbfs: float = -60.0  # "No audio detected" warning: median 100 ms level below this ...
    no_audio_warn_s: float = 10.0  # ... over this much recorded audio
    pause_on_sleep: bool = True


@dataclass(frozen=True)
class TranscriptionConfig:
    device: str = "auto"  # auto | cuda | cpu
    model: str = "auto"  # auto (hardware tier) | large-v3-turbo | distil-large-v3.5
    compute_type: str = "auto"  # auto -> int8_float16 on CUDA, int8 on CPU
    language: str = "en"
    beam_size: int = 5
    batch_size: int = 0  # 0 = sequential decoding (turbo); > 0 = batched (only for distil)
    cpu_threads: int = 0  # CPU jobs: Whisper and pyannote threads; 0 = min(4, logical CPUs / 2), set at load


@dataclass(frozen=True)
class DiarizationConfig:
    min_speakers: int | None = None  # hints; not tuned yet
    max_speakers: int | None = None


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"  # loopback only; anything else is refused at startup
    port: int = 8765
    # Quit by itself after this many minutes with no open page and nothing recording, processing or queued
    #. 0 = never (default): a page in a hidden tab doesn't poll, so it would count as closed.
    idle_quit_minutes: int = 0

    def __post_init__(self):
        m = self.idle_quit_minutes
        if isinstance(m, bool) or not isinstance(m, int) or not 0 <= m <= 1440:
            raise ValueError(f"[server] idle_quit_minutes must be a whole number of minutes from 0 (never) to 1440, not {m!r}")


PROCESSING_MODES = ("manual", "auto")


@dataclass(frozen=True)
class ProcessingConfig:
    # "manual" (default): Stop/import only save the recording; the user presses Process now / Process all.
    # "auto": queued on Stop/import and run when plugged in. Either way, GPU jobs never start while
    # recording or on battery (machine.py).
    mode: str = "manual"
    # Ask Windows not to idle-sleep while a job is queued and running (per process, released when the
    # queue is empty or waits). Closing the lid or choosing Sleep still follows Windows' own settings.
    keep_awake: bool = True

    def __post_init__(self):
        if self.mode not in PROCESSING_MODES:
            raise ValueError(f"[processing] mode must be one of {PROCESSING_MODES}, not {self.mode!r}")
        if not isinstance(self.keep_awake, bool):
            raise ValueError(f"[processing] keep_awake must be true or false, not {self.keep_awake!r}")


@dataclass(frozen=True)
class RetentionConfig:
    # Delete a session's original audio (WAV, m4a, mp3, ...) once its Opus copy is verified. Off until
    # the user confirms (never delete recordings without explicit confirmation). None = not set:
    # the older name `delete_wav_after_opus` decides (it now covers every source).
    delete_original_after_opus: bool | None = None
    delete_wav_after_opus: bool = False  # alias only; code reads delete_original_after_opus
    # Recently deleted: permanently remove items older than this many days. 0 = never (default):
    # nothing is purged unless the user turns it on (Settings asks for a permanent-deletion confirmation).
    trash_days: int = 0

    def __post_init__(self):
        if self.delete_original_after_opus is None:
            object.__setattr__(self, "delete_original_after_opus", self.delete_wav_after_opus)
        for k in ("delete_original_after_opus", "delete_wav_after_opus"):
            if not isinstance(getattr(self, k), bool):
                raise ValueError(f"[retention] {k} must be true or false, not {getattr(self, k)!r}")
        if isinstance(self.trash_days, bool) or not isinstance(self.trash_days, int) or not 0 <= self.trash_days <= 3650:
            raise ValueError(f"[retention] trash_days must be a whole number of days from 0 (never) to 3650, "
                             f"not {self.trash_days!r}")


@dataclass(frozen=True)
class NamingConfig:
    # Default title of new lectures, e.g. "{class} {date} Lecture {n}"; placeholders in naming.py.
    # Empty (default): recordings are named by date and time, imports by their file name.
    template: str = ""

    def __post_init__(self):
        from lecture_recorder.naming import NamingError, check_template

        try:
            object.__setattr__(self, "template", check_template(self.template))
        except NamingError as e:
            raise ValueError(f"[naming] template: {e}") from None


@dataclass(frozen=True)
class LLMConfig:
    # Local Ollama only: anything but a loopback address is refused (summary/llm.py).
    ollama_host: str = "http://127.0.0.1:11434"
    summary_model: str = "qwen3:8b"
    embedding_model: str = "qwen3-embedding:0.6b"
    num_ctx: int = 8192  # set on every request; sized for an 8 GB GPU
    max_prompt_tokens: int = 5000  # whole prompt (instructions + transcript chunk), map-reduce above this
    num_predict: int = 2000  # output cap; prompt + output always fits in num_ctx
    timeout_s: float = 900
    # Concept search. num_ctx is fixed (changing it makes Ollama reload the model); each window
    # is far below the input cap, and truncate:false makes Ollama refuse anything longer.
    embed_num_ctx: int = 2048
    embed_max_input_tokens: int = 1500  # estimated, per input; longer windows are split before sending
    embed_batch: int = 16


@dataclass(frozen=True)
class SearchConfig:
    window_s: float = 45.0  # concept-search windows: ~45 s of transcript...
    step_s: float = 30.0  # ...starting every 30 s (15 s overlap)
    exact_limit: int = 200  # exact-search results shown (the total count is always reported)
    concept_k: int = 10


@dataclass(frozen=True)
class Config:
    home: Path
    data_dir: Path
    models_dir: Path
    recording: RecordingConfig
    transcription: TranscriptionConfig = TranscriptionConfig()
    diarization: DiarizationConfig = DiarizationConfig()
    server: ServerConfig = ServerConfig()
    retention: RetentionConfig = RetentionConfig()
    llm: LLMConfig = LLMConfig()
    search: SearchConfig = SearchConfig()
    processing: ProcessingConfig = ProcessingConfig()
    naming: NamingConfig = NamingConfig()


def load_config(home: Path | None = None) -> Config:
    home = (home or app_home()).resolve()
    raw = _merge(_read_toml(home / "config.auto.toml", generated=True), _read_toml(home / "config.toml"))
    paths = raw.get("paths", {})
    rec_fields = RecordingConfig.__dataclass_fields__
    rec = RecordingConfig(**{k: v for k, v in raw.get("recording", {}).items() if k in rec_fields})
    stt = _section(TranscriptionConfig, raw.get("transcription", {}))
    if not stt.cpu_threads:  # one value for Whisper and pyannote (autoconfig.default_threads)
        from lecture_recorder.autoconfig import default_threads
        from lecture_recorder.host import host

        stt = dataclasses.replace(stt, cpu_threads=default_threads(None, host().perf_cores()))
    dia = _section(DiarizationConfig, raw.get("diarization", {}))
    srv = _section(ServerConfig, raw.get("server", {}))
    ret = _section(RetentionConfig, raw.get("retention", {}))
    llm = _section(LLMConfig, raw.get("llm", {}))
    search = _section(SearchConfig, raw.get("search", {}))
    processing = _section(ProcessingConfig, raw.get("processing", {}))
    naming = _section(NamingConfig, raw.get("naming", {}))
    return Config(
        home=home,
        data_dir=home / paths.get("data_dir", "data"),
        models_dir=home / paths.get("models_dir", "models"),
        recording=rec,
        transcription=stt,
        diarization=dia,
        server=srv,
        retention=ret,
        llm=llm,
        search=search,
        processing=processing,
        naming=naming,
    )


def _read_toml(path: Path, generated: bool = False) -> dict:
    """A broken generated file (config.auto.toml) is ignored with a warning, so the app still starts and
    `configure` can rewrite it; a broken config.toml (the user's) still raises, as before."""
    if not path.exists():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        if not generated:
            raise
        import logging

        logging.getLogger(__name__).warning("ignoring %s (%s); run `lecture-recorder configure`", path.name, e)
        return {}


def _merge(auto: dict, user: dict) -> dict:
    """The user's values win; a user value of "auto" keeps the machine pick (if there is one).

    Picks that depend on another setting follow the user's choice of it instead of the stale pick:
    a user-set transcription model gets its own decoding (batched only for distil), a user-set
    device its own compute type."""
    out = {s: dict(v) for s, v in auto.items() if isinstance(v, dict)}
    for section, values in user.items():
        if not isinstance(values, dict):  # e.g. `transcription = "cpu"`: the user's file is broken, say so
            raise ValueError(f"config.toml: [{section}] must be a section (table), not {values!r}")
        merged = out.setdefault(section, {})
        for k, v in values.items():
            if v != "auto" or k not in merged:
                merged[k] = v
    stt, mine = out.get("transcription", {}), user.get("transcription", {}) if isinstance(user.get("transcription"), dict) else {}
    own = lambda k: k in mine and mine[k] != "auto"  # noqa: E731
    if own("model") and not own("batch_size") and "batch_size" in stt:
        stt["batch_size"] = 8 if stt["model"] == "distil-large-v3.5" else 0
    if own("device") and not own("compute_type") and "compute_type" in stt:
        stt["compute_type"] = "int8_float16" if stt["device"] == "cuda" else "int8"
    for k in ("batch_size", "cpu_threads"):  # "auto" without a pick = the built-in default (0)
        if stt.get(k) == "auto":
            stt[k] = 0
        if k in stt and (isinstance(stt[k], bool) or not isinstance(stt[k], int) or stt[k] < 0):
            raise ValueError(f"[transcription] {k} must be a whole number >= 0 or \"auto\", not {stt[k]!r}")
    return out


def _section(cls, values: dict):
    return cls(**{k: v for k, v in values.items() if k in cls.__dataclass_fields__})
