"""Speech-to-text with faster-whisper. The model is unloaded (VRAM freed) before returning."""
import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lecture_recorder.pipeline.audio import SAMPLE_RATE
from lecture_recorder.pipeline.gpu import cuda_available, register_cuda_dlls
from lecture_recorder.pipeline.progress import NULL

MODEL_DIRS = {
    "large-v3-turbo": "faster-whisper-large-v3-turbo",
    "distil-large-v3.5": "faster-distil-large-v3.5",
}


@dataclass(frozen=True)
class SttChoice:
    model: str
    model_dir: Path
    device: str
    compute_type: str


def choose(models_dir: Path, model: str = "auto", device: str = "auto", compute_type: str = "auto") -> SttChoice:
    """"auto" values (no config.auto.toml pick yet) follow autoconfig.pick_stt, the same rule `configure`
    uses. A cuda device that is no longer usable falls back to the CPU instead of failing the job."""
    if "auto" in (model, device):
        from lecture_recorder.autoconfig import models_present, pick_stt
        from lecture_recorder.pipeline.gpu import nvidia_smi_memory

        from lecture_recorder.host import host

        usable = cuda_available()
        mem = nvidia_smi_memory() if usable else None
        stt = pick_stt(usable, mem["total_mb"] if mem else 0, models_present(models_dir), os.cpu_count() or 2,
                       host().perf_cores())
        model = stt["model"] if model == "auto" else model
        device = stt["device"] if device == "auto" else device
    if device == "cuda" and not cuda_available():  # e.g. a driver update, or the folder copied to another PC
        import logging

        logging.getLogger(__name__).warning("CUDA is not usable now: transcribing on the CPU (run `configure`)")
        device, compute_type = "cpu", "int8" if compute_type in ("auto", "int8_float16", "float16") else compute_type
    if model not in MODEL_DIRS:
        raise ValueError(f"unknown model {model!r}; choose one of {sorted(MODEL_DIRS)}")
    if compute_type == "auto":
        compute_type = "int8_float16" if device == "cuda" else "int8"
    model_dir = Path(models_dir) / MODEL_DIRS[model]
    if not (model_dir / "model.bin").exists():  # a clear message instead of a library error deep in the job
        from lecture_recorder.autoconfig import DOWNLOAD_HINTS

        raise FileNotFoundError(f"The speech model {model} is not downloaded ({model_dir}). Run: {DOWNLOAD_HINTS.get(model, DOWNLOAD_HINTS['large-v3-turbo'])}")
    return SttChoice(model, model_dir, device, compute_type)


def dropped_speech_report(audio: np.ndarray, segments: list[dict], speech: list[dict] | None = None) -> dict:
    gaps = dropped_speech(audio, segments, speech=speech)
    return {"total_s": round(sum(g["end"] - g["start"] for g in gaps), 2), "stretches": gaps}


def dropped_speech(audio: np.ndarray, segments: list[dict], min_gap_s: float = 1.0, pad_s: float = 0.3,
                   speech: list[dict] | None = None) -> list[dict]:
    """Stretches the VAD calls speech (>= min_gap_s long) that got no transcribed words.

    A model-independent check for silently skipped audio (seen with distil's long-form decoding on
    the synthetic fixture). Reported in every transcript; not proof of loss, but worth a look.
    `speech`: VAD timestamps already computed for this audio (health.speech_timestamps).
    """
    from lecture_recorder.pipeline.health import speech_timestamps

    step = 0.01
    n = int(len(audio) / SAMPLE_RATE / step) + 1
    heard = np.zeros(n, bool)
    for r in speech if speech is not None else speech_timestamps(audio):
        heard[int(r["start"] / SAMPLE_RATE / step): int(r["end"] / SAMPLE_RATE / step)] = True
    covered = np.zeros(n, bool)
    for seg in segments:
        for w in seg["words"] or [{"start": seg["start"], "end": seg["end"]}]:
            covered[max(0, int((w["start"] - pad_s) / step)): int((w["end"] + pad_s) / step) + 1] = True
    gaps, start = [], None
    for i, missing in enumerate(np.append(heard & ~covered, False)):
        if missing and start is None:
            start = i
        elif not missing and start is not None:
            if (i - start) * step >= min_gap_s:
                gaps.append({"start": round(start * step, 2), "end": round(i * step, 2)})
            start = None
    return gaps


def vocabulary_hints(vocabulary: list[str] | None, mode: str) -> dict:
    """faster-whisper 1.2.1 options for a class's vocabulary list.

    `hotwords` is put in front of every 30 s window (sequential and batched); `initial_prompt`
    reaches only the first window when condition_on_previous_text is False (the prompt resets),
    but every chunk in batched mode. Both are read from the installed code.
    """
    terms = [t.strip() for t in (vocabulary or []) if t and t.strip()]
    if not terms or mode == "none":
        return {}
    text = ", ".join(terms)
    hints = {}
    if mode in ("hotwords", "both"):
        hints["hotwords"] = text
    if mode in ("initial_prompt", "both"):
        hints["initial_prompt"] = f"Glossary: {text}."
    if not hints:
        raise ValueError(f"unknown vocabulary hint mode {mode!r}")
    return hints


def transcribe(audio: np.ndarray, choice: SttChoice, language: str = "en", beam_size: int = 5,
               check_dropped: bool = True, batch_size: int = 0, vocabulary: list[str] | None = None,
               hint_mode: str = "hotwords", condition_on_previous_text: bool = False, cpu_threads: int = 0,
               progress=NULL) -> dict:
    """Return {'segments': [...], 'settings': {...}, 'timings': {...}}. Words carry start/end/probability.

    batch_size > 0 uses faster-whisper's batched pipeline (each VAD chunk decoded on its own).
    `progress` (pipeline/progress.py) gets the seconds of audio decoded so far, per segment.
    """
    register_cuda_dlls()
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    settings = dict(
        language=language, beam_size=beam_size, vad_filter=True, word_timestamps=True,
        # False avoids repetition loops on long audio
        condition_on_previous_text=condition_on_previous_text,
        **vocabulary_hints(vocabulary, hint_mode),
    )
    t0 = time.perf_counter()
    progress.stage("load")
    model = WhisperModel(str(choice.model_dir), device=choice.device, compute_type=choice.compute_type,
                         cpu_threads=cpu_threads, local_files_only=True)  # 0 = CTranslate2's default (4)
    t1 = time.perf_counter()
    try:
        if batch_size > 0:
            segments_iter, info = BatchedInferencePipeline(model).transcribe(audio, batch_size=batch_size, **settings)
        else:
            segments_iter, info = model.transcribe(audio, **settings)
        segments = []
        progress.stage("transcribe", total=round(len(audio) / SAMPLE_RATE, 1), unit="s")
        for s in segments_iter:  # the generator does the work; consume it while the model is loaded
            words = [{"w": w.word, "start": round(w.start, 3), "end": round(w.end, 3),
                      "p": round(w.probability, 3)} for w in (s.words or [])]
            segments.append({"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip(),
                             "words": words})
            progress.update(s.end)
    finally:
        try:
            model.model.unload_model()  # frees VRAM now, not at garbage collection
        except Exception:
            pass
        del model
        gc.collect()
    t2 = time.perf_counter()
    return {
        "segments": segments,
        # CPU-only (~10 s per 2 h): the pipeline runs it after releasing the GPU lock (check_dropped=False)
        "dropped_speech": dropped_speech_report(audio, segments) if check_dropped else None,
        "settings": {"model": choice.model, "device": choice.device, "compute_type": choice.compute_type,
                     "batch_size": batch_size, **settings},
        "language": info.language,
        "duration_s": round(len(audio) / SAMPLE_RATE, 3),
        "timings": {"load_s": round(t1 - t0, 3), "transcribe_s": round(t2 - t1, 3)},
    }
