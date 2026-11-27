"""Is there usable audio? A CPU-only check that runs before any GPU work.

A muted or broken microphone records digital silence or 1-bit noise (~-96 dBFS); transcribing it
gives an empty transcript that looks like a failure of the app. Such files are flagged
"no usable audio" instead, with the reason, and the user can still transcribe them anyway.
"""
import numpy as np

from lecture_recorder.pipeline.audio import SAMPLE_RATE

NEAR_SILENT_DBFS = -60.0  # default; callers pass config [recording] no_audio_dbfs (the REC-bar warning's level)
MIN_SPEECH_S = 2.0  # less speech than max(this, MIN_SPEECH_SHARE of the file) = no usable audio
MIN_SPEECH_SHARE = 0.01
WARN_SILENT_SHARE = 0.5  # usable, but at least half of it near-silent: warn on the lecture page


def speech_timestamps(audio: np.ndarray) -> list[dict]:
    """Silero VAD (bundled with faster-whisper), in samples. ~10 s of CPU for 2 h of audio."""
    from faster_whisper.vad import get_speech_timestamps

    return get_speech_timestamps(audio, sampling_rate=SAMPLE_RATE)


def block_levels(audio: np.ndarray, block_s: float = 0.1) -> np.ndarray:
    n = int(SAMPLE_RATE * block_s)
    usable = len(audio) // n * n
    if not usable:
        return np.array([-120.0])
    b = audio[:usable].reshape(-1, n)
    rms = np.sqrt(np.einsum("ij,ij->i", b, b, dtype=np.float64) / n)  # no float64 copy of the whole file
    return np.where(rms > 1e-6, 20 * np.log10(np.maximum(rms, 1e-12)), -120.0)


def audio_health(audio: np.ndarray, speech: list[dict], near_silent_dbfs: float = NEAR_SILENT_DBFS) -> dict:
    duration = len(audio) / SAMPLE_RATE
    levels = block_levels(audio)
    median = float(np.median(levels))
    silent_share = float(np.mean(levels < near_silent_dbfs))
    speech_s = sum(r["end"] - r["start"] for r in speech) / SAMPLE_RATE
    usable = speech_s >= max(MIN_SPEECH_S, MIN_SPEECH_SHARE * duration)
    reason = warning = None
    if not usable:
        reason = (f"The recording is near-silent ({silent_share:.0%} of it below {near_silent_dbfs:.0f} dBFS): "
                  "the microphone probably recorded nothing. Check the microphone and its Windows settings."
                  if median < near_silent_dbfs else f"No speech was found ({speech_s:.1f} s of {duration:.0f} s).")
    elif silent_share >= WARN_SILENT_SHARE:
        warning = f"{silent_share:.0%} of this recording is near-silent: check the microphone before the next lecture."
    return {"usable": usable, "reason": reason, "warning": warning, "duration_s": round(duration, 1),
            "speech_s": round(speech_s, 1), "median_dbfs": round(median, 1), "near_silent_share": round(silent_share, 3)}
