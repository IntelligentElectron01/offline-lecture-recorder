"""Compact a transcribed session's audio to a verified Opus copy (`audio.opus`), whatever the source:
the app's own WAV, or an imported m4a/mp3/flac/ogg (an mp4's audio is already stored as m4a or wav).
The original is kept unless the user turns on `retention.delete_original_after_opus` (alias: the older
`delete_wav_after_opus`; off by default; never delete recordings without consent).

Verification: the Opus file decodes, its duration matches the original's within 50 ms, and its loudness
(RMS) is within 3 dB of the original's, so a silent or truncated encode is never accepted. The original
is deleted only after that check passed, and only if the Opus copy is smaller.
"""
import os
from pathlib import Path

import numpy as np

from lecture_recorder.pipeline.audio import AUDIO_NAMES, SAMPLE_RATE, audio_seconds, load_audio
from lecture_recorder.recorder.session import now_iso, read_json, write_json_atomic

WAV, OPUS, RECORD = "audio.wav", "audio.opus", "retention.json"
# Originals a session can hold besides its Opus copy (an imported Opus file is itself audio.opus).
ORIGINALS = tuple(n for n in AUDIO_NAMES if n != OPUS)
MAX_DURATION_DIFF_S = 0.05
MAX_RMS_DIFF_DB = 3.0
# The decode must cover what the file's header says, or it stopped early (a damaged file): the original is
# then kept, since another decoder might still recover the rest.
MAX_SHORT_DECODE_S = 1.0
# libsndfile Opus quality (0 = best/largest, 1 = smallest). 0.85 = ~43 kbps for 16 kHz mono speech, ~20 MB/h.
# The default (~27 kbps) cost turbo 5 of 154 words on a test recording; 0.85 cost 2.
OPUS_COMPRESSION_LEVEL = 0.85
# A source that is already Opus at or below this average bitrate is left as it is:
# the recorder's ~43 kbps plus VBR headroom, so re-encoding could only lose quality.
OPUS_KEEP_KBPS = 48
EST_OPUS_BYTES_PER_S = 5600  # dry-run estimate of the copy's size (~45 kbps)
NOT_SMALLER = "the original is already smaller than an Opus copy"
NEEDS_SETTING = "Opus above the app's bitrate; replacing it needs the delete-original setting"


def _rms_db(x: np.ndarray) -> float:
    # float64 sum without a float64 copy of the recording (np.square did: peak 944 -> 620 MB for 1 h)
    return 20 * np.log10(float(np.sqrt(np.einsum("i,i->", x, x, dtype=np.float64) / len(x))) + 1e-9)


def verify_opus(original: Path, opus_path: Path, ref: np.ndarray | None = None) -> dict:
    """`ref`: the original already decoded to 16 kHz mono float32 (saves decoding a long recording again).
    `complete`: the decode covers the original's header length (deleting the original needs this too)."""
    import soundfile as sf

    ref = load_audio(original) if ref is None else ref  # same as the encoder input
    header_s = audio_seconds(original)
    dur_ref = len(ref) / SAMPLE_RATE
    complete = header_s is None or dur_ref >= header_s - MAX_SHORT_DECODE_S
    try:
        got, rate = sf.read(str(opus_path), dtype="float32")
    except Exception as e:  # corrupt or truncated file: not verified (never raise past this point)
        return {"ok": False, "error": f"Opus file does not decode: {e}", "source_duration_s": round(dur_ref, 3),
                "complete": complete}
    if got.ndim > 1:
        got = got.mean(axis=1)
    dur_got = len(got) / rate
    rms_diff = abs(_rms_db(ref) - _rms_db(got)) if len(ref) and len(got) else float("inf")
    ok = abs(dur_ref - dur_got) <= MAX_DURATION_DIFF_S and rms_diff <= MAX_RMS_DIFF_DB
    return {"ok": bool(ok), "source_duration_s": round(dur_ref, 3), "opus_duration_s": round(dur_got, 3),
            "rms_diff_db": round(float(rms_diff), 2), "header_s": header_s, "complete": complete}


def find_original(d: Path) -> Path | None:
    """The session's original audio (not its Opus copy), or None."""
    return next((d / n for n in ORIGINALS if (d / n).is_file()), None)


def opus_kbps(path: Path) -> float | None:
    """Average bitrate if the file's audio is Opus (from its size and header length; nothing decoded), else None."""
    import av

    try:
        with av.open(str(path), metadata_errors="ignore") as c:
            s = next((s for s in c.streams if s.type == "audio"), None)
            if s is None or s.codec_context.name != "opus":
                return None
    except Exception:
        return None
    secs = audio_seconds(path)
    return path.stat().st_size * 8 / secs / 1000 if secs else None


def _already_opus(path: Path) -> str | None:
    """Why `path` is left as it is because it is Opus already (one rule for the dry run and the run)."""
    kbps = opus_kbps(path)
    if kbps is not None and kbps <= OPUS_KEEP_KBPS:
        return f"already Opus at {kbps:.0f} kbps"
    if kbps is None and path.name == OPUS:
        return "already Opus"  # length unknown: never re-encoded on a guess
    return None


def _encode(audio: np.ndarray, dst: Path) -> None:
    import soundfile as sf

    tmp = dst.with_name(f"{dst.name}.{os.getpid()}.tmp")
    try:
        sf.write(str(tmp), audio, SAMPLE_RATE, format="OGG", subtype="OPUS", compression_level=OPUS_COMPRESSION_LEVEL)
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _record(d: Path, **fields) -> dict:
    record = {"opus": OPUS if (d / OPUS).is_file() else None, "checked": now_iso(), **fields}
    write_json_atomic(d / RECORD, record)
    return record


def record_error(d: Path, error: str) -> None:
    """A failed compaction is added to the record, never replacing what it says was already deleted."""
    old = read_json(d / RECORD, None) or {"verified": False, "original_kept": True}
    write_json_atomic(d / RECORD, {**old, "error": error, "error_at": now_iso()})


def own_copy_done(d: Path) -> bool:
    """audio.opus is the app's verified copy and its original is gone (nothing left to compact)."""
    rec = read_json(d / RECORD, None) or {}
    gone = rec.get("source", WAV) != OPUS and not (d / rec.get("source", WAV)).exists()
    return bool(rec.get("verified")) and (rec.get("original_kept") is False or rec.get("wav_kept") is False or gone)


def compress_session(session_dir: Path, delete_original: bool = False, audio: np.ndarray | None = None,
                     source: Path | None = None) -> dict | None:
    """Write audio.opus (16 kHz mono speech) from the session's original and verify it.

    `audio`: `source` already decoded by load_audio. Returns the retention record, or None when there is
    nothing to compact. An existing audio.opus next to an original is the app's own copy: kept if it
    verifies, else made again. An imported Opus original above OPUS_KEEP_KBPS is replaced by its verified
    smaller copy only if `delete_original`. The original is deleted only if `delete_original` AND
    verification passed AND the decode was complete AND the copy is smaller.
    """
    d = Path(session_dir)
    src = source or find_original(d)
    opus = d / OPUS
    if src is not None and src.name == OPUS:
        src = None
    if src is None:  # only audio.opus: the app's own copy, or an imported Opus file
        if not opus.is_file() or own_copy_done(d):
            return None
        return _recompress_opus(d, opus, delete_original, audio)
    if why := _already_opus(src):  # an .ogg can hold Opus already
        return _record(d, source=src.name, verified=False, original_kept=True, skipped=why)
    if audio is None:
        audio = load_audio(src)
    made = not opus.exists()
    if made:
        _encode(audio, opus)
    check = verify_opus(src, opus, ref=audio)
    if not check["ok"] and not made:  # an earlier copy that no longer verifies: make it again
        _encode(audio, opus)  # atomic: the old copy is replaced by a complete new one
        made, check = True, verify_opus(src, opus, ref=audio)
    record = {"source": src.name, "verified": check["ok"], "check": check, "original_kept": True,
              "delete_original_after_opus": bool(delete_original)}
    if made and not check["ok"]:
        opus.unlink()  # this run's own failed copy: gone, so a later run can try again
    elif check["ok"] and opus.stat().st_size >= src.stat().st_size:
        if made:  # a copy that saves nothing is not kept (e.g. a low-bitrate mp3)
            opus.unlink()
        return _record(d, **{**record, "verified": False, "skipped": NOT_SMALLER})
    if delete_original and check["ok"] and check["complete"]:
        src.unlink()
        record.update(original_kept=False, original_deleted=now_iso())
    elif delete_original and check["ok"]:
        record["kept_why"] = "the original decoded shorter than its header says"
    return _record(d, **record)


def _recompress_opus(d: Path, opus: Path, delete_original: bool, audio) -> dict:
    if why := _already_opus(opus):
        return _record(d, source=OPUS, verified=False, original_kept=True, skipped=why)
    if not delete_original:  # the smaller copy would replace the original file of the same name
        return _record(d, source=OPUS, verified=False, original_kept=True, skipped=NEEDS_SETTING)
    audio = load_audio(opus) if audio is None else audio
    new = d / f"{OPUS}.new"
    try:
        _encode(audio, new)
        check = verify_opus(opus, new, ref=audio)
        if not (check["ok"] and check["complete"]) or new.stat().st_size >= opus.stat().st_size:
            return _record(d, source=OPUS, verified=False, check=check, original_kept=True,
                           skipped="no smaller copy" if check["ok"] and check["complete"] else "the smaller copy failed verification")
        os.replace(new, opus)  # one rename: the session always has exactly one complete audio.opus
    finally:
        new.unlink(missing_ok=True)  # also after a failed replace (e.g. the player holds the file open)
    return _record(d, source=OPUS, verified=True, check=check, original_kept=False, original_deleted=now_iso(),
                   delete_original_after_opus=True)


# ------------------------------------------------------------------ compact existing sessions (user action)
def _busy(d: Path, active: set[str]) -> bool:
    from lecture_recorder.move import MOVING
    from lecture_recorder.recorder.platform import is_locked
    from lecture_recorder.recorder.session import LOCK_FILE, META_FILE, PCM_FILE

    return (is_locked(d / LOCK_FILE) or (d / PCM_FILE).exists() or str(d.resolve()) in active
            or bool((read_json(d / META_FILE, {}) or {}).get(MOVING)))


def _active_sessions(data_dir: Path) -> set[str]:
    from lecture_recorder.pipeline.queue import ACTIVE, jobs

    return {j["session"] for j in jobs(data_dir) if j["state"] in ACTIVE}


def _recording(data_dir: Path) -> bool:
    from lecture_recorder.recorder.platform import is_locked
    from lecture_recorder.recorder.process import GLOBAL_LOCK

    return is_locked(Path(data_dir) / GLOBAL_LOCK)


def _plan_one(d: Path, encode: bool) -> tuple[dict | None, str | None]:
    """(item, None) for a session to compact, (None, reason) to list it as skipped, (None, None) to leave
    it out (already compact). Reads only headers and small JSON files."""
    from lecture_recorder.pipeline.queue import RAW
    from lecture_recorder.recorder.session import META_FILE

    src, opus = find_original(d), d / OPUS
    rec = read_json(d / RECORD, None) or {}
    if src is None and (not opus.is_file() or own_copy_done(d)):
        return None, None
    if not (d / RAW).exists():
        return None, "not transcribed yet (compacted after its transcription)"
    if src is not None and opus.is_file() and rec.get("verified") and rec.get("source", WAV) == src.name:
        size = src.stat().st_size
        return {"action": "delete original", "file": src.name, "bytes_before": size + opus.stat().st_size,
                "bytes_after": opus.stat().st_size}, None
    path = src or opus
    if rec.get("skipped") == NOT_SMALLER and rec.get("source") == path.name:
        return None, NOT_SMALLER  # measured on a real copy; the other skips are worked out again below
    if not encode:
        return None, ("Opus copy failed verification (`lecture-recorder compact` makes it again)" if opus.is_file()
                      else "no Opus copy yet (`lecture-recorder compact` makes one)")
    if why := _already_opus(path):
        return None, why
    secs = (read_json(d / META_FILE, {}) or {}).get("duration_s") or audio_seconds(path)
    if not secs:
        return None, "length unknown"
    size, est = path.stat().st_size, int(secs * EST_OPUS_BYTES_PER_S)
    if est >= size:
        return None, f"{NOT_SMALLER} would be"
    return {"action": "encode", "file": path.name, "bytes_before": size, "bytes_after": est}, None


def compact_plan(data_dir: Path, encode: bool = True) -> dict:
    """Dry run: which sessions would be compacted (Opus copy made if missing, verified, original deleted),
    bytes before and (estimated) after, and which are skipped and why. Changes nothing.

    `encode=False` lists only sessions that already have a verified Opus copy (the app's button: deleting
    is quick; encoding long recordings belongs in `lecture-recorder compact`). A session that is recording,
    queued, processing or moving is never touched. `token` identifies exactly this plan, so the deletion
    the user confirms is the deletion that runs."""
    import hashlib

    from lecture_recorder.recorder.session import iter_session_dirs

    active = _active_sessions(Path(data_dir))
    items, skipped = [], []
    for d in iter_session_dirs(Path(data_dir)):
        sid = f"{d.parent.parent.name}/{d.name}"
        if _busy(d, active):
            if find_original(d) is not None or (d / OPUS).is_file():
                skipped.append({"session": sid, "reason": "recording, queued or processing"})
            continue
        try:
            item, why = _plan_one(d, encode)
        except OSError as e:  # a file vanished or is locked: list it, never fail the whole plan
            item, why = None, f"could not read it: {e}"
        if item:
            items.append({"session": sid, **item})
        elif why:
            skipped.append({"session": sid, "reason": why})
    key = [(x["session"], x["action"], x["file"], x["bytes_before"]) for x in items]
    return {"items": items, "skipped": skipped, "bytes_before": sum(x["bytes_before"] for x in items),
            "bytes_after": sum(x["bytes_after"] for x in items),
            "token": hashlib.sha256(repr(key).encode()).hexdigest()[:16]}


def compact(data_dir: Path, token: str, encode: bool = True) -> dict:
    """Compact the sessions of the confirmed plan. The plan is recomputed and must match `token`. Right
    before each session: it must still be idle, and its Opus copy is verified (again) against the original;
    an original is deleted only if that passes. Stops (keeping the rest) as soon as a recording starts, so the
    recorder never competes with encoding."""
    plan = compact_plan(data_dir, encode)
    if plan["token"] != token:
        raise ValueError("The sessions changed since the dry run; review the new list and confirm again.")
    root = Path(data_dir)
    done, kept = [], []
    for item in plan["items"]:
        cls, name = item["session"].split("/")
        d = root / "classes" / cls / "sessions" / name
        try:
            if _recording(root):
                why = "a recording started; run compact again afterwards"
            elif _busy(d, _active_sessions(root)):
                why = "it started recording or processing"
            elif item["action"] == "delete original":
                why = _delete_original(d, d / item["file"])
            else:
                rec = compress_session(d, delete_original=True, source=d / item["file"]) or {}
                why = None if rec.get("original_kept") is False else (
                    rec.get("skipped") or rec.get("kept_why") or "Opus copy failed verification")
        except Exception as e:  # this session keeps its original; carry on with the others
            why = f"error: {e}"
        if why:
            kept.append({"session": item["session"], "reason": why})
        else:
            done.append({**item, "bytes_after": (d / OPUS).stat().st_size})
    return {"done": done, "kept": kept, "bytes_before": sum(x["bytes_before"] for x in done),
            "bytes_after": sum(x["bytes_after"] for x in done)}


def _delete_original(d: Path, src: Path) -> str | None:
    check = verify_opus(src, d / OPUS)
    if not check["ok"]:
        return "Opus copy failed re-verification"
    if not check["complete"]:
        return "the original decoded shorter than its header says"
    src.unlink()
    _record(d, source=src.name, verified=True, check=check, original_kept=False, original_deleted=now_iso(),
            deleted_by="compact")
    return None
