"""Crash-safe audio: raw PCM appended while recording, rebuilt into a WAV afterwards."""
import hashlib
import os
import time
import wave
from pathlib import Path

from lecture_recorder.recorder.session import META_FILE, PCM_FILE, WAV_FILE, now_iso, read_json, write_json_atomic

SAMPLE_WIDTH = 2  # int16
CHUNK = 1 << 20


class PcmWriter:
    """Appends int16 PCM. Writes are unbuffered (a killed process loses nothing already written) and
    fsync'd about every `fsync_interval` seconds (a power loss loses at most that much)."""

    def __init__(self, path: Path, fsync_interval: float = 1.0):
        self._f = open(path, "ab", buffering=0)
        self._interval = fsync_interval
        self._last_sync = time.monotonic()
        self.bytes_written = path.stat().st_size

    def write(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            n = self._f.write(view)
            view = view[n:]
        self.bytes_written += len(data)
        if time.monotonic() - self._last_sync >= self._interval:
            self.sync()

    def sync(self) -> None:
        os.fsync(self._f.fileno())
        self._last_sync = time.monotonic()

    def close(self) -> None:
        if not self._f.closed:
            self.sync()
            self._f.close()


def _sha256_file(path: Path, limit: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        remaining = limit
        while remaining:
            block = f.read(min(CHUNK, remaining))
            if not block:
                break
            h.update(block)
            remaining -= len(block)
    return h.hexdigest()


def _sha256_wav_frames(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        while True:
            block = w.readframes(CHUNK // SAMPLE_WIDTH)
            if not block:
                break
            h.update(block)
    return h.hexdigest(), n


def finalize_session(session_dir: Path, recovered: bool = False) -> Path | None:
    """Turn `audio.pcm` into `audio.wav`. Safe to call repeatedly or after a crash at any point.

    The PCM file is removed only after the WAV is fsync'd and its audio bytes hash-match the PCM, so
    this is a lossless format change. Returns the WAV path, or None if the session has no audio.
    """
    pcm = session_dir / PCM_FILE
    wav_path = session_dir / WAV_FILE
    meta_path = session_dir / META_FILE
    if not pcm.exists():
        return wav_path if wav_path.exists() else None
    meta = read_json(meta_path)
    if not meta or "sample_rate" not in meta:
        # Without the true sample rate the WAV would play at the wrong speed; keep the PCM and retry later.
        raise RuntimeError(f"cannot read sample rate from {meta_path}; audio.pcm kept")

    rate = int(meta.get("sample_rate", 16000))
    channels = int(meta.get("channels", 1))
    frame_bytes = SAMPLE_WIDTH * channels
    usable = pcm.stat().st_size // frame_bytes * frame_bytes  # drop a torn final frame, if any

    tmp = session_dir / f"{WAV_FILE}.tmp"
    with open(pcm, "rb") as src, open(tmp, "wb") as raw:
        with wave.open(raw, "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(SAMPLE_WIDTH)
            w.setframerate(rate)
            remaining = usable
            while remaining:
                block = src.read(min(CHUNK, remaining))
                if not block:
                    break
                w.writeframesraw(block)
                remaining -= len(block)
        raw.flush()
        os.fsync(raw.fileno())

    wav_hash, nframes = _sha256_wav_frames(tmp)
    if nframes * frame_bytes != usable or wav_hash != _sha256_file(pcm, usable):
        raise RuntimeError(f"WAV verification failed for {session_dir}; audio.pcm kept")
    os.replace(tmp, wav_path)

    meta.update(
        state="stopped",
        recovered=bool(recovered or meta.get("recovered")),
        recorded_s=round(nframes / rate, 3),
        finalized=now_iso(),
    )
    if recovered and not meta.get("stop_reason"):
        meta["stop_reason"] = "recovered_after_crash"
    write_json_atomic(meta_path, meta)
    pcm.unlink()  # last step: a crash before this just repeats the (idempotent) conversion next time
    return wav_path
