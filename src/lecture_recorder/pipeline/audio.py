"""Decode any supported audio file to a 16 kHz mono float32 waveform, in memory.

Uses PyAV (bundled with faster-whisper, with its own FFmpeg libraries), so wav/mp3/m4a/opus all
decode without a system FFmpeg. pyannote is given this same array, so torchcodec never decodes.
"""
from datetime import datetime
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
# The original first, the Opus copy (retention.py) last: transcription and the player use the original while
# it is kept, and the copy once it is gone (or an imported Opus file, which is stored as audio.opus).
AUDIO_NAMES = ("audio.wav", "audio.m4a", "audio.mp3", "audio.flac", "audio.ogg", "audio.opus")


def load_audio(path: Path) -> np.ndarray:
    from faster_whisper.audio import decode_audio

    audio = decode_audio(str(path), sampling_rate=SAMPLE_RATE)
    return np.ascontiguousarray(audio, dtype=np.float32)


def recorded_at(path: Path) -> datetime | None:
    """Recording start from the file's own metadata (e.g. a phone or tablet voice recorder's m4a: `creation_time`, UTC),
    in local time; None if absent, unreadable or implausible (MP4's 1904/1970 zero dates, the future)."""
    import av

    try:
        with av.open(str(path), metadata_errors="ignore") as c:
            tags = [c.metadata] + [s.metadata for s in list(c.streams)[:1]]
    except Exception:
        return None
    for t in tags:
        raw = t.get("creation_time") or t.get("date")
        if not raw:
            continue
        try:
            when = datetime.fromisoformat(str(raw).strip().replace("Z", "+00:00"))
        except ValueError:
            continue
        if when.tzinfo is None:
            when = when.astimezone()  # no zone given: read as local time
        when = when.astimezone()
        if 2000 <= when.year and when <= datetime.now().astimezone():
            return when
    return None


def find_session_audio(session_dir: Path) -> Path:
    for name in AUDIO_NAMES:
        if (session_dir / name).is_file():
            return session_dir / name
    if (session_dir / "audio.pcm").exists():
        raise FileNotFoundError(f"{session_dir} has an unfinished recording; run `lecture-recorder recover` first")
    raise FileNotFoundError(f"no audio file in {session_dir}")


def audio_seconds(path: Path) -> float | None:
    """The file's length from its headers (nothing is decoded), or None if it doesn't say."""
    import av

    try:
        with av.open(str(path), metadata_errors="ignore") as c:
            s = next((s for s in c.streams if s.type == "audio"), None)
            if s is not None and s.duration and s.time_base:
                return round(float(s.duration * s.time_base), 3)
            return round(c.duration / av.time_base, 3) if c.duration else None
    except Exception:  # only a hint for the lecture list and the duplicate check
        return None


def extract_audio(src: Path, dst_dir: Path) -> Path:
    """The audio track of a video file (e.g. a Zoom mp4) as `dst_dir/audio.m4a` (AAC copied as is, no
    re-encoding) or, for any other codec, `dst_dir/audio.wav` (16 kHz mono). The video is not kept.
    Written to a .part file first, so a failure leaves nothing half-written. ValueError: no audio track."""
    import os

    import av

    with av.open(str(src), metadata_errors="ignore") as inp:
        stream = next((s for s in inp.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("This file has no audio track.")
        aac = stream.codec_context.name == "aac"
        dst = dst_dir / ("audio.m4a" if aac else "audio.wav")
        part = dst.with_name(dst.name + ".part")
        try:
            with av.open(str(part), "w", format="ipod" if aac else "wav") as out:
                if aac:
                    o = out.add_stream_from_template(stream)
                    for pkt in inp.demux(stream):
                        if pkt.dts is not None:
                            pkt.stream = o
                            out.mux(pkt)
                else:
                    o = out.add_stream("pcm_s16le", rate=SAMPLE_RATE, layout="mono")
                    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
                    for frame in inp.decode(stream):
                        for f in resampler.resample(frame):
                            out.mux(o.encode(f))
                    for f in resampler.resample(None):
                        out.mux(o.encode(f))
                    out.mux(o.encode(None))
            os.replace(part, dst)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
    return dst
