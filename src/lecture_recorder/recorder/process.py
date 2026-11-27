"""The recorder process: `python -m lecture_recorder.recorder --session-dir DIR`.

Runs separately from the UI so a UI crash never stops a recording. It is controlled through small
JSON command files in `<session>/control/` and reports through `<session>/status.json`.
"""
import argparse
import os
import signal
import sys
import time
import traceback
from pathlib import Path

from lecture_recorder.recorder.pcm import PcmWriter, finalize_session
from lecture_recorder.recorder.platform import KeepAwake, ProcessLock, make_power_events
from lecture_recorder.recorder.safeguards import Action, NoAudioMonitor, Safeguards, State
from lecture_recorder.recorder.session import (
    BOOKMARKS_FILE, CONTROL_DIR, LOCK_FILE, META_FILE, PCM_FILE, STATUS_FILE, now_iso, read_json, write_json_atomic,
)
from lecture_recorder.recorder.sources import AppSource, MicSource, SyntheticSource, dbfs

GLOBAL_LOCK = ".recording.lock"
STALL_S = 5.0  # no audio for this long while recording = device lost
STATUS_EVERY_S = 1.0  # the UI reads it every 2 s
COMMANDS_EVERY_S = 0.25  # pause/stop/bookmark latency; a directory listing 4x/s instead of every block
READ_TIMEOUT_S, PAUSED_READ_TIMEOUT_S = 0.1, 0.5  # paused: the source is stopped, so wake less often
EXIT_BUSY = 3


def data_dir_of(session_dir: Path) -> Path:
    return session_dir.parents[3]  # data/classes/<class>/sessions/<name>


def log(msg: str) -> None:
    print(f"{now_iso()} {msg}", flush=True)


class Recorder:
    def __init__(self, session_dir: Path, allow_simulated_events: bool = False):
        self.dir = session_dir
        self.meta_path = session_dir / META_FILE
        # The parent wrote meta.json a moment ago; a scanner may still hold it (this was the cause of a
        # rare "Recorder failed to start (exit 1)"). Wait well within the parent's 20 s start timeout.
        self.meta = read_json(self.meta_path, wait_s=10)
        if not self.meta:
            raise RuntimeError(f"{META_FILE} in {session_dir} is missing or stayed unreadable for 10 s")
        self.allow_sim = allow_simulated_events
        self.rate = int(self.meta["sample_rate"])
        self.sg = Safeguards(
            max_length_s=float(self.meta["max_length_s"]),
            silence_pause_s=float(self.meta["silence_pause_s"]),
            silence_dbfs=float(self.meta["silence_dbfs"]),
            pause_on_sleep=bool(self.meta.get("pause_on_sleep", True)),
        )
        self.no_audio = NoAudioMonitor(float(self.meta.get("no_audio_dbfs", -60.0)),
                                       float(self.meta.get("no_audio_warn_s", 10.0)))
        src = self.meta["source"]
        if src["kind"] == "synthetic":
            self.source = SyntheticSource(self.rate, src.get("pattern", "tone"), src.get("stall_after_s"))
        elif src["kind"] == "app":
            mic = MicSource(src.get("device_index"), self.rate) if src.get("include_mic") else None
            self.source = AppSource(int(src["pid"]), self.rate, mic)
        else:
            self.source = MicSource(src.get("device_index"), self.rate)
        self.keep_awake = KeepAwake()
        self.power = make_power_events()
        self.bookmarks = read_json(session_dir / BOOKMARKS_FILE, []) or []
        self.level = -120.0  # loudest block since the last status write (the REC bar's level meter)
        self.last_audio = time.monotonic()
        self.stop_requested = False
        self.dirty: set[str] = set()

    # ------------------------------------------------------------ bookkeeping
    def event(self, kind: str, reason: str) -> None:
        self.meta.setdefault("events", []).append(
            {"type": kind, "reason": reason, "audio_s": round(self.sg.recorded_s, 3), "at": now_iso()}
        )
        self.meta.update(state=self.sg.state.value, state_reason=self.sg.reason, recorded_s=round(self.sg.recorded_s, 3))
        if kind == "stop":
            self.meta["stop_reason"] = reason
        self.dirty.add(META_FILE)
        self.flush_files()
        log(f"{kind} ({reason}) at {self.sg.recorded_s:.1f}s")

    def flush_files(self) -> None:
        """Best-effort JSON saves. A file held open by another program must never stop audio
        capture, so failures are retried on the next loop instead of raised."""
        for name in list(self.dirty):
            obj = self.meta if name == META_FILE else self.bookmarks
            try:
                write_json_atomic(self.dir / name, obj)
                self.dirty.discard(name)
            except OSError as e:
                log(f"could not save {name} yet: {e}")

    def write_status(self, state: str | None = None) -> None:
        try:
            self._write_status(state)
        except OSError:
            pass  # status is informational; the next write replaces it

    def _write_status(self, state: str | None) -> None:
        write_json_atomic(self.dir / STATUS_FILE, {
            "pid": os.getpid(),
            "state": state or self.sg.state.value,
            "reason": self.sg.reason,
            "recorded_s": round(self.sg.recorded_s, 2),
            "level_dbfs": round(self.level, 1),
            "no_audio": self.no_audio.active,
            "no_audio_since_s": None if self.no_audio.since_s is None else round(self.no_audio.since_s, 1),
            "keep_awake": self.keep_awake.held,
            "power_events": getattr(self.power, "error", None) or "ok",
            "bookmarks": len(self.bookmarks),
            "overflows": getattr(self.source, "overflows", 0),  # mic buffer overruns (dropped input)
            "updated": time.time(),
        }, retries=2, fsync=False)  # informational: don't stall the audio loop or fsync every second

    def note_no_audio(self, change: str | None) -> None:
        """Each 'no audio' stretch is kept in meta.json (start/end in recorded-audio seconds)."""
        if change is None:
            return
        at = self.no_audio.since_s if change == "start" else self.sg.recorded_s
        self.meta.setdefault("events", []).append({"type": f"no_audio_{change}", "audio_s": round(at, 1), "at": now_iso()})
        self.dirty.add(META_FILE)
        log(f"no audio detected from {at:.1f}s" if change == "start" else f"audio back at {at:.1f}s")
        self.write_status()

    # ------------------------------------------------------------ actions
    def apply(self, action: Action | None) -> None:
        if action is None:
            return
        if action.kind == "pause":
            self.source.stop()
            self.keep_awake.release()
            self.writer.sync()
        elif action.kind == "resume":
            try:
                self.source.start()
            except Exception as e:
                log(f"could not reopen audio device: {e}")
                self.sg.state, self.sg.reason = State.PAUSED, "device_error"
                self.event("pause", "device_error")
                return
            self.keep_awake.hold()
            self.last_audio = time.monotonic()
            self.no_audio.reset()
        elif action.kind == "stop":
            self.source.stop()
            self.keep_awake.release()
        self.event(action.kind, action.reason)
        self.write_status()

    def handle_commands(self) -> None:
        ctl = self.dir / CONTROL_DIR
        if not ctl.is_dir():
            return
        for f in sorted(ctl.glob("*.json")):
            cmd = read_json(f)
            if cmd is None:
                continue  # temporarily unreadable; keep the file and retry next loop
            try:
                f.unlink()
            except OSError:
                continue
            name = cmd.get("cmd")
            if name in ("pause", "resume", "stop"):
                self.apply(self.sg.on_user(name))
            elif name == "bookmark":
                self.bookmarks.append({"audio_s": round(self.sg.recorded_s, 2), "label": str(cmd.get("label") or ""), "created": now_iso()})
                self.dirty.add(BOOKMARKS_FILE)
                self.flush_files()
                log(f"bookmark at {self.sg.recorded_s:.1f}s")
            elif name == "simulate_power" and self.allow_sim:
                self.power.events.put(str(cmd.get("event")))
            elif name == "simulate_clock_gap" and self.allow_sim:
                self.apply(self.sg.on_tick(time.time() + float(cmd.get("seconds", 60))))
            else:
                log(f"ignored command {cmd!r}")

    # ------------------------------------------------------------ main loop
    def run(self) -> int:
        self.writer = PcmWriter(self.dir / PCM_FILE)
        self.power.start()
        if getattr(self.power, "error", None):
            log(self.power.error)
        try:
            self.source.start()
        except Exception as e:
            log(f"could not open audio device: {e}")
            reason = "app_closed" if isinstance(e, ProcessLookupError) else "device_error"
            self.sg.state, self.sg.reason = State.STOPPED, reason
            self.event("stop", reason)
            return 2
        self.keep_awake.hold()
        self.event("start", "user")
        self.write_status()
        last_status = last_cmds = time.monotonic()

        while self.sg.state is not State.STOPPED:
            block = self.source.read(timeout=READ_TIMEOUT_S if self.sg.state is State.RECORDING else PAUSED_READ_TIMEOUT_S)
            if block is not None and self.sg.state is State.RECORDING:
                block = block[: self.sg.max_frames_allowed(self.rate)]
                self.writer.write(block.tobytes())
                level = dbfs(block)
                self.level = max(self.level, level)
                self.last_audio = time.monotonic()
                self.apply(self.sg.on_audio(len(block) / self.rate, level))
                self.note_no_audio(self.no_audio.on_audio(len(block) / self.rate, level, self.sg.recorded_s))
            if self.stop_requested:
                self.apply(self.sg.on_user("stop"))
            if getattr(self.source, "ended", None):
                log("the recorded app closed")
                self.apply(self.sg.on_source_ended(self.source.ended))
            if time.monotonic() - last_cmds >= COMMANDS_EVERY_S:
                self.handle_commands()
                last_cmds = time.monotonic()
            while not self.power.events.empty():
                ev = self.power.events.get()
                log(f"power event: {ev}")
                self.apply(self.sg.on_power(ev))
            self.apply(self.sg.on_tick(time.time()))
            if self.sg.state is State.RECORDING and time.monotonic() - self.last_audio > STALL_S:
                log("no audio from device")
                self.apply(self.sg.on_device_lost())
            if time.monotonic() - last_status >= STATUS_EVERY_S:
                self.flush_files()
                self.write_status()
                self.level = -120.0
                last_status = time.monotonic()
        return 0

    def close(self) -> None:
        getattr(self.source, "close", self.source.stop)()
        self.keep_awake.release()
        self.power.stop()
        if hasattr(self, "writer"):
            self.writer.close()


def acquire_with_retry(lock: ProcessLock, seconds: float = 2.0) -> bool:
    """`is_locked()` probes by briefly taking the lock, so a free lock can look busy for an instant."""
    end = time.monotonic() + seconds
    while not lock.acquire():
        if time.monotonic() > end:
            return False
        time.sleep(0.05)
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lecture_recorder.recorder")
    p.add_argument("--session-dir", required=True, type=Path)
    p.add_argument("--allow-simulated-events", action="store_true", help="tests only")
    args = p.parse_args(argv)
    session_dir = args.session_dir.resolve()

    global_lock = ProcessLock(data_dir_of(session_dir) / GLOBAL_LOCK)
    if not acquire_with_retry(global_lock):
        log("another recording is already running")
        return EXIT_BUSY
    session_lock = ProcessLock(session_dir / LOCK_FILE)
    if not acquire_with_retry(session_lock):
        global_lock.release()
        log("this session already has a recorder")
        return EXIT_BUSY

    try:
        rec = Recorder(session_dir, args.allow_simulated_events)
    except Exception:
        log("recorder could not start:\n" + traceback.format_exc())
        session_lock.release()
        global_lock.release()
        return 1

    def on_signal(*_):
        rec.stop_requested = True

    signal.signal(signal.SIGINT, on_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, on_signal)
    if os.name != "nt":  # macOS/Linux: closing the terminal or `kill` saves the recording instead of ending it abruptly
        signal.signal(signal.SIGTERM, on_signal)

    code = 1
    try:
        code = rec.run()
    except Exception:
        log("recorder crashed:\n" + traceback.format_exc())
    finally:
        rec.close()
        for _ in range(50):  # finalize reads meta from disk, so pending saves must land first
            rec.flush_files()
            if not rec.dirty:
                break
            time.sleep(0.1)
        try:
            rec.write_status("finalizing")
            wav = finalize_session(session_dir)
            log(f"saved {wav}")
        except Exception:
            log("finalize failed (audio.pcm kept; recovery will retry):\n" + traceback.format_exc())
            code = code or 1
        rec.meta = read_json(rec.meta_path, rec.meta)
        rec.write_status("stopped")
        session_lock.release()
        global_lock.release()
    return code


if __name__ == "__main__":
    sys.exit(main())
