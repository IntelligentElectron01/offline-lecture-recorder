"""Compaction numbers (synthetic audio; CPU only; never data\\).

    .venv\\Scripts\\python.exe scripts\\bench\\measure_compact.py [--minutes 20] [--out .cache\\compact]

Tiles the synthetic TTS lecture (tests\\fixtures\\synthetic\\lecture_clean.wav; make it with
scripts\\dev\\make_synthetic_fixture.py) to --minutes, writes it as a Zoom-like m4a (AAC, stereo, 48 kHz, 72 kbps),
an mp3 (128 kbps, stereo, 44.1 kHz) and a mic-like WAV (48 kHz mono 16-bit), then compacts each in its own
process (decode + Opus encode + verification, as after a transcription job) and reports the time per hour of
audio, bytes before and after, the copy's bitrate, the dry run's estimate and the process's peak memory.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic" / "lecture_clean.wav"
FORMATS = {  # name: (codec, container, rate, layout, bit_rate)
    "audio.m4a": ("aac", "ipod", 48000, "stereo", 72000),
    "audio.mp3": ("libmp3lame", "mp3", 44100, "stereo", 128000),
    "audio.wav": ("pcm_s16le", "wav", 48000, "mono", 0),
}


def write(path: Path, x16k: np.ndarray, codec: str, fmt: str, rate: int, layout: str, bit_rate: int) -> None:
    import av

    with av.open(str(path), "w", format=fmt) as out:
        s = out.add_stream(codec, rate=rate)
        s.layout = layout
        if bit_rate:
            s.bit_rate = bit_rate
        rs = av.AudioResampler(format=s.format.name, layout=layout, rate=rate)
        step = 16000
        for i in range(0, len(x16k), step):
            f = av.AudioFrame.from_ndarray(x16k[None, i:i + step], format="flt", layout="mono")
            f.sample_rate, f.pts = 16000, i
            for g in rs.resample(f):
                for p in s.encode(g):
                    out.mux(p)
        for g in rs.resample(None):
            for p in s.encode(g):
                out.mux(p)
        for p in s.encode(None):
            out.mux(p)


def one(d: Path) -> dict:
    """Runs in a child process: compact one session folder, like the end of a transcription job."""
    import psutil

    import lecture_recorder  # noqa: F401  (offline lockdown)
    from lecture_recorder import retention as ret
    from lecture_recorder.pipeline.audio import audio_seconds

    src = ret.find_original(d)
    before, secs = src.stat().st_size, audio_seconds(src)
    rss0 = psutil.Process().memory_info().rss
    t0, c0 = time.perf_counter(), time.process_time()
    rec = ret.compress_session(d, delete_original=False)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    after = (d / ret.OPUS).stat().st_size
    mem = psutil.Process().memory_info()
    return {"file": src.name, "audio_s": secs, "verified": rec["verified"], "rms_diff_db": rec["check"]["rms_diff_db"],
            "wall_s": round(wall, 2), "cpu_s": round(cpu, 2), "wall_s_per_hour": round(wall / secs * 3600, 1),
            "bytes_before": before, "bytes_after": after, "ratio": round(after / before, 3),
            "opus_kbps": round(after * 8 / secs / 1000, 1), "estimate_bytes": int(secs * ret.EST_OPUS_BYTES_PER_S),
            "peak_mb": round(getattr(mem, "peak_wset", mem.rss) / 2**20), "start_rss_mb": round(rss0 / 2**20)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--out", type=Path, default=ROOT / ".cache" / "compact")
    ap.add_argument("--one", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        print(json.dumps(one(args.one)))
        return 0
    if not FIXTURE.exists():
        print(f"missing {FIXTURE}; run scripts\\dev\\make_synthetic_fixture.py first")
        return 1
    out = args.out.resolve()
    if out.is_relative_to(ROOT / "data"):
        print("refusing to write into data\\")
        return 1
    import soundfile as sf

    speech, rate = sf.read(str(FIXTURE), dtype="float32")
    assert rate == 16000 and speech.ndim == 1
    x = np.tile(speech, int(np.ceil(args.minutes * 60 * rate / len(speech))))[: int(args.minutes * 60 * rate)]
    results = []
    for name, spec in FORMATS.items():
        d = out / "sessions" / name.split(".")[1]
        d.mkdir(parents=True, exist_ok=True)
        for p in d.iterdir():
            p.unlink()  # this script's own synthetic files from an earlier run
        write(d / name, x, *spec)
        r = subprocess.run([sys.executable, __file__, "--one", str(d)], capture_output=True, text=True, timeout=1200)
        if r.returncode:
            print(r.stderr)
            return 1
        results.append(json.loads(r.stdout.strip().splitlines()[-1]))
        print(json.dumps(results[-1]))
    path = out / f"compact-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({"minutes": args.minutes, "results": results}, indent=2), encoding="utf-8")
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
