"""Download pinned models into <home>/models and verify the SHA-256 of EVERY file.

Usage:
  python scripts/download_models.py whisper    # ungated Whisper models (no token needed)
  python scripts/download_models.py pyannote   # gated diarization model (needs HF_TOKEN in .env)

Any mismatch, missing file, or error after the download moves that model's folder to
models/quarantine/<name>-<timestamp>/ (never loaded by the app) and exits non-zero.
Ollama models are pulled separately with `ollama pull` (see README).
"""
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

# Same home rule as the app: LECTURE_RECORDER_HOME, else the checkout this script lives in.
ROOT = Path(os.environ.get("LECTURE_RECORDER_HOME") or Path(__file__).resolve().parent.parent).resolve()
MODELS = ROOT / "models"

os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_UPDATE_CHECK"] = "1"
os.environ.pop("HF_HUB_OFFLINE", None)  # this script is the one place that may download

# Weight hashes were checked against the publisher's files (docs/dependencies.md). The small text
# files (configs/tokenizers) are pinned by commit and their hashes recorded from that commit.
_WHISPER_PREPROC = "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711"
_WHISPER_VOCAB = "c69260f2ab26d659b7c398f9a2b2b48ed0df16c3b47d7326782fd9cba71690c1"
SPECS = {
    "whisper": [
        (
            "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
            "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
            "faster-whisper-large-v3-turbo",
            {
                "model.bin": "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da",
                "config.json": "b0253ea6c0d3bea6b1e19e91a02acfd3b53f4467362efcb5a3e6b16c9b3a9b7e",
                "preprocessor_config.json": _WHISPER_PREPROC,
                "tokenizer.json": "297b13372ac43916285644fb9687add3cc62ee2a1adb60da3dc25cc94c1871fd",
                "vocabulary.json": _WHISPER_VOCAB,
            },
        ),
        (
            "distil-whisper/distil-large-v3.5-ct2",
            "9793ccc07920e0f830e1dba0343efcdf0ef8c903",
            "faster-distil-large-v3.5",
            {
                "model.bin": "c58b88b8585ffcd2135fddaaf421ce72cb223b32edea70d156aed1dea319a119",
                "config.json": "90c55f775cc4e0bb17293d0bf12f96557a486f20dea886fabd8e6075a3588b21",
                "preprocessor_config.json": _WHISPER_PREPROC,
                "tokenizer.json": "b3c8202bbf06d8ee4232c5984baa563784ac4737e2e7fdc42fa180200d3cfcdb",
                "vocabulary.json": _WHISPER_VOCAB,
            },
        ),
    ],
    "pyannote": [
        (
            "pyannote/speaker-diarization-community-1",
            "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
            "pyannote-community-1",
            {
                "config.yaml": "5ce2bfa9a938dc132cec1172592d65173cbb8f444ea1e4133f10f9391de155be",
                "segmentation/pytorch_model.bin": "7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e",
                "embedding/pytorch_model.bin": "6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929",
                "embedding/README.md": "fa9e5105ae95edb231d841476cdb91eef4be0621c372ed4f7d3421294b5f8ad7",
                "plda/plda.npz": "9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255",
                "plda/xvec_transform.npz": "325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f",
                "plda/README.md": "e1316dbbeb3261431478d48ceebbd4bba395c3587e7b80c254dbab00f1209d0a",
            },
        ),
    ],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quarantine(dest: Path, name: str) -> Path:
    q = MODELS / "quarantine" / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    q.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(dest), str(q))
    return q


def verify(dest: Path, expected: dict[str, str]) -> bool:
    ok = True
    for rel, want in expected.items():
        p = dest / rel
        got = sha256(p) if p.is_file() else None
        status = "MATCH" if got == want else ("MISSING" if got is None else "MISMATCH")
        print(f"  {rel}: {status}" + ("" if got == want else f" (got {got})"))
        ok &= got == want
    extra = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file() and ".cache" not in p.parts} - set(expected)
    if extra:
        print(f"  unexpected files: {sorted(extra)}")
        ok = False
    return ok


def main(group: str) -> int:
    from huggingface_hub import snapshot_download

    token = None
    if group == "pyannote":
        from dotenv import dotenv_values

        token = (dotenv_values(ROOT / ".env").get("HF_TOKEN") or "").strip() or None
        if not token:
            print("HF_TOKEN missing in .env. See .env.example.")
            return 2

    for repo, rev, local, expected in SPECS[group]:
        dest = MODELS / local
        print(f"Downloading {repo}@{rev[:8]} -> models/{local}")
        try:
            snapshot_download(repo, revision=rev, local_dir=dest, allow_patterns=list(expected), token=token)
            ok = verify(dest, expected)
        except Exception as e:  # network/auth/disk errors: never leave a partial model in place
            print(f"  error: {type(e).__name__}: {e}")
            ok = False
        if not ok:
            if dest.exists():
                q = quarantine(dest, local)
                print(f"  Moved to {q.relative_to(ROOT)}. These files will NOT be used.")
            return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in SPECS:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
