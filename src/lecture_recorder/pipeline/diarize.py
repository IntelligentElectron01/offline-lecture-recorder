"""Speaker diarization with pyannote community-1: safe loader, offline, in-memory waveform."""
import gc
import os
import time
from pathlib import Path

import numpy as np

from lecture_recorder.pipeline.audio import SAMPLE_RATE
from lecture_recorder.pipeline.progress import NULL

MODEL_DIR = "pyannote-community-1"


def diarize(audio: np.ndarray, models_dir: Path, device: str = "cuda",
            min_speakers: int | None = None, max_speakers: int | None = None, cpu_threads: int = 0,
            progress=NULL) -> dict:
    """Return {'turns': [{'start','end','label'}], 'timings': {...}}; non-overlapping (exclusive) turns.
    `progress` gets pyannote's own step counts through its `hook`."""
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("Hugging Face offline mode is not forced; import lecture_recorder first.")
    if not (Path(models_dir) / MODEL_DIR / "config.yaml").exists():  # before the (slow) torch import
        from lecture_recorder.autoconfig import DOWNLOAD_HINTS

        raise FileNotFoundError(f"The speaker model is not downloaded ({Path(models_dir) / MODEL_DIR}). Run: {DOWNLOAD_HINTS['pyannote']}")
    import torch

    from lecture_recorder.safe_load import load_diarization_pipeline

    # Per process (the job's own): the CPU side is light next to the GPU work, so don't take every core.
    torch.set_num_threads(cpu_threads or max(1, min(4, (os.cpu_count() or 2) // 2)))

    t0 = time.perf_counter()
    progress.stage("speakers")
    pipeline = load_diarization_pipeline(Path(models_dir) / MODEL_DIR)
    if device == "cuda" and torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    else:
        device = "cpu"
    t1 = time.perf_counter()
    try:
        waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None, :]
        hints = {k: v for k, v in (("min_speakers", min_speakers), ("max_speakers", max_speakers)) if v}
        hook = progress.diarization_hook()
        out = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE}, **hints, **({"hook": hook} if hook else {}))
        annotation = getattr(out, "exclusive_speaker_diarization", out)  # exclusive = made for transcripts
        turns = [{"start": round(seg.start, 3), "end": round(seg.end, 3), "label": str(label)}
                 for seg, _, label in annotation.itertracks(yield_label=True)]
    finally:
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    t2 = time.perf_counter()
    return {
        "turns": sorted(turns, key=lambda t: t["start"]),
        "settings": {"model": MODEL_DIR, "device": device, **hints},
        "timings": {"load_s": round(t1 - t0, 3), "diarize_s": round(t2 - t1, 3)},
    }
