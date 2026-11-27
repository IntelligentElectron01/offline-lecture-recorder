"""Process one session: transcribe, free VRAM, diarize, merge, write plain files."""
from pathlib import Path

from lecture_recorder.config import Config, load_config
from lecture_recorder.pipeline.audio import find_session_audio, load_audio
from lecture_recorder.pipeline.diarize import diarize
from lecture_recorder.pipeline.gpu import gpu_job, vram_used_mb as _vram
from lecture_recorder.pipeline.merge import merge
from lecture_recorder.pipeline.progress import NULL
from lecture_recorder.pipeline.outputs import RAW, TranscriptExists, write_outputs
from lecture_recorder.pipeline.transcribe import choose, dropped_speech_report, transcribe
from lecture_recorder.recorder.platform import is_locked
from lecture_recorder.recorder.session import LOCK_FILE, now_iso
from lecture_recorder.runtime_env import disable_library_telemetry

SCHEMA_VERSION = 1


class NoUsableAudio(RuntimeError):
    """The file is (near-)silent or has no speech; meta.json["audio_check"] says why."""


def process_session(session_dir: Path, config: Config | None = None, progress=NULL) -> dict:
    cfg = config or load_config()
    session_dir = Path(session_dir)
    if is_locked(session_dir / LOCK_FILE):
        raise RuntimeError(f"{session_dir.name} is still recording")
    if (session_dir / RAW).exists():
        raise TranscriptExists(f"{session_dir / RAW} already exists")
    audio_path = find_session_audio(session_dir)
    disable_library_telemetry()
    progress.stage("prepare")

    audio = load_audio(audio_path)
    speech = check_audio(session_dir, audio, cfg.recording.no_audio_dbfs)  # CPU only, before the GPU lock
    stt_cfg = cfg.transcription
    choice = choose(cfg.models_dir, stt_cfg.model, stt_cfg.device, stt_cfg.compute_type)
    # distil chosen by "auto" (no pick, turbo not downloaded) is batched too: sequential distil skips speech
    batch = stt_cfg.batch_size or (8 if stt_cfg.model == "auto" and choice.model == "distil-large-v3.5" else 0)
    vram = {}
    progress.stage("gpu_wait")
    with gpu_job(cfg.home):  # serial: never two GPU jobs at once, across processes
        from lecture_recorder.summary.llm import release_llm_vram

        release_llm_vram(cfg.llm)  # an LLM left loaded by a killed summary job must not share the GPU
        vram["before_mb"] = _vram()
        stt = transcribe(audio, choice, stt_cfg.language, stt_cfg.beam_size, check_dropped=False,
                         batch_size=batch, cpu_threads=stt_cfg.cpu_threads, progress=progress)
        vram["after_stt_freed_mb"] = _vram()
        dia = diarize(audio, cfg.models_dir, device=choice.device, cpu_threads=stt_cfg.cpu_threads,
                      min_speakers=cfg.diarization.min_speakers, max_speakers=cfg.diarization.max_speakers,
                      progress=progress)
        vram["after_diarization_freed_mb"] = _vram()

    progress.stage("finish")
    stt["dropped_speech"] = dropped_speech_report(audio, stt["segments"], speech)  # CPU only: after the GPU lock
    # Batched decoding's word times don't line up with the VAD (63 s flagged on a clip it fully
    # transcribed), so the UI shows the result as approximate instead of as a warning.
    stt["dropped_speech"]["reliable"] = batch == 0
    segments, speakers = merge(stt["segments"], dia["turns"])
    doc = {
        "schema": SCHEMA_VERSION,
        "created": now_iso(),
        "audio": audio_path.name,
        "duration_s": stt["duration_s"],
        "stt": stt["settings"],
        "diarization": dia["settings"],
        "processing": {"timings": {"stt_load_s": stt["timings"]["load_s"], "transcribe_s": stt["timings"]["transcribe_s"],
                                   "diarization_load_s": dia["timings"]["load_s"], "diarize_s": dia["timings"]["diarize_s"]},
                       "vram_mb": vram,
                       "dropped_speech": stt["dropped_speech"]},
        "speakers": speakers,
        "turns": dia["turns"],
        "segments": segments,
    }
    write_outputs(session_dir, doc)
    _remember_in_meta(session_dir, doc["duration_s"], stt["dropped_speech"])
    _after_processing(session_dir, cfg, audio, audio_path)
    return doc


def check_audio(session_dir: Path, audio, near_silent_dbfs: float = -60.0) -> list[dict]:
    """Refuse a file with no usable audio (unless the user chose "Transcribe anyway"); returns the VAD
    speech timestamps, reused by the dropped-speech check. The result is kept in meta.json."""
    from lecture_recorder.pipeline.health import audio_health, speech_timestamps
    from lecture_recorder.recorder.session import update_meta

    speech = speech_timestamps(audio)
    health = audio_health(audio, speech, near_silent_dbfs)
    meta = update_meta(session_dir, audio_check={**health, "checked": now_iso()})  # raises if unreadable
    if not health["usable"] and not meta.get("audio_check_override"):
        raise NoUsableAudio(f"No usable audio: {health['reason']}")
    return speech


def _remember_in_meta(session_dir: Path, duration_s: float, dropped: dict) -> None:
    """Copy the small facts the session page shows into meta.json, so it never parses the (large)
    transcript.raw.json while polling."""
    from lecture_recorder.recorder.session import META_FILE, meta_lock, read_json, write_json_atomic

    with meta_lock(session_dir):  # never loses a rename made meanwhile in the app
        meta = read_json(session_dir / META_FILE, None)
        if not isinstance(meta, dict):
            return
        extra = {"dropped_speech": dropped}
        if meta.get("recorded_s") is None:  # replaces an import's length from the file header
            extra["duration_s"] = round(float(duration_s), 3)
        write_json_atomic(session_dir / META_FILE, {**meta, **extra})


def _after_processing(session_dir: Path, cfg: Config, audio=None, audio_path: Path | None = None) -> None:
    """Summary placeholder (replaced by the summary job) and the verified Opus copy of whatever audio was
    transcribed, reusing its decoded samples. Failures here never undo the transcript."""
    from lecture_recorder.library import write_summary_placeholder
    from lecture_recorder.retention import compress_session, record_error

    write_summary_placeholder(session_dir)
    try:
        # Read the setting now, not at job start: the user may have turned deletion off while this
        # (possibly long) job ran. config.toml lives in the app home.
        delete = load_config(cfg.home).retention.delete_original_after_opus
        compress_session(session_dir, delete_original=delete, audio=audio, source=audio_path)
    except Exception as e:  # keep the original; add why compression failed to the record
        record_error(session_dir, str(e))
