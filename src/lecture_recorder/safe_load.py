"""Load model checkpoints with PyTorch's safe unpickler only.

pyannote.audio 4.0.7 calls torch.load(..., weights_only=False). The lockdown sets
TORCH_FORCE_WEIGHTS_ONLY_LOAD=1, which makes torch ignore that and use the restricted
(weights-only) unpickler. pyannote checkpoints need exactly these plain data types, which
are allowlisted here and nowhere else:
"""
from pathlib import Path

PYANNOTE_SAFE_GLOBALS = (
    "pyannote.audio.core.task.Specifications",  # dataclass describing model inputs/outputs
    "pyannote.audio.core.task.Problem",  # Enum
    "pyannote.audio.core.task.Resolution",  # Enum
)


def _allow_pyannote_globals() -> None:
    import importlib

    import torch

    objs = []
    for path in PYANNOTE_SAFE_GLOBALS:
        module, _, name = path.rpartition(".")
        objs.append(getattr(importlib.import_module(module), name))
    torch.serialization.add_safe_globals(objs)


def load_diarization_pipeline(model_dir: Path):
    """The only supported way to load the pyannote pipeline (local folder, offline, safe unpickling)."""
    import os

    if os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD") != "1":
        raise RuntimeError("Safe loading is not enforced; import lecture_recorder first.")
    _allow_pyannote_globals()
    from pyannote.audio import Pipeline

    return Pipeline.from_pretrained(str(model_dir))
