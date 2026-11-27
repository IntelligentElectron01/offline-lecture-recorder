"""Start and control the recorder process; recover sessions left behind by a crash."""
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from lecture_recorder.config import Config, load_config
from lecture_recorder.recorder.pcm import finalize_session
from lecture_recorder.host import host
from lecture_recorder.recorder.platform import is_locked
from lecture_recorder.recorder.process import EXIT_BUSY, GLOBAL_LOCK
from lecture_recorder.recorder.session import (
    CONTROL_DIR, LOCK_FILE, LOG_FILE, META_FILE, PCM_FILE, STATUS_FILE,
    iter_session_dirs, new_session_dir, now_iso, read_json, sessions_root, write_json_atomic,
)

PERMISSION_REMINDER = "Reminder: you are responsible for getting permission before recording anyone."


class RecorderError(RuntimeError):
    pass


class RecordingHandle:
    def __init__(self, session_dir: Path, proc: subprocess.Popen | None = None):
        self.session_dir = session_dir
        self.proc = proc

    @property
    def name(self) -> str:
        return self.session_dir.name

    def is_running(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is None
        return is_locked(self.session_dir / LOCK_FILE)

    def status(self) -> dict:
        return read_json(self.session_dir / STATUS_FILE, {}) or {}

    def meta(self) -> dict:
        return read_json(self.session_dir / META_FILE, {}) or {}

    def send(self, cmd: str, **fields) -> None:
        ctl = self.session_dir / CONTROL_DIR
        ctl.mkdir(exist_ok=True)
        # Name sorts by time, so commands run in the order they were sent.
        write_json_atomic(ctl / f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}.json", {"cmd": cmd, **fields})

    def pause(self) -> None:
        self.send("pause")

    def resume(self) -> None:
        self.send("resume")

    def bookmark(self, label: str = "") -> None:
        self.send("bookmark", label=label)

    def wait_for(self, predicate, timeout: float) -> dict:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            s = self.status()
            if predicate(s):
                return s
            time.sleep(0.05)
        raise TimeoutError(f"recorder status did not reach the expected state: {self.status()}")

    def stop(self, timeout: float = 60) -> dict:
        """Ask the recorder to stop and wait until the WAV is saved. Returns the final meta."""
        if self.is_running():
            self.send("stop")
            end = time.monotonic() + timeout
            while self.is_running():
                if time.monotonic() > end:
                    raise TimeoutError("recorder did not stop in time")
                time.sleep(0.05)
        if (self.session_dir / PCM_FILE).exists():  # the process died before finalizing
            finalize_session(self.session_dir, recovered=True)
        return self.meta()


def active_recording(data_dir: Path) -> RecordingHandle | None:
    """The running recording, e.g. after the UI restarts. None if nothing is recording."""
    if not is_locked(data_dir / GLOBAL_LOCK):
        return None
    for d in iter_session_dirs(data_dir):
        if is_locked(d / LOCK_FILE):
            return RecordingHandle(d)
    return None


def recover_sessions(data_dir: Path) -> list[Path]:
    """Finalize every session whose recorder is gone but whose audio.pcm is still on disk, and finish any
    class move a crash interrupted."""
    from lecture_recorder.move import finish_moves

    try:
        finish_moves(data_dir)
    except TimeoutError:  # the queue lock stayed busy: the journal waits for the next start
        pass
    recovered = []
    for d in iter_session_dirs(data_dir):
        if (d / PCM_FILE).exists() and not is_locked(d / LOCK_FILE):
            finalize_session(d, recovered=True)
            recovered.append(d)
    return recovered


def start_recording(
    class_slug: str,
    *,
    config: Config | None = None,
    device: str | int | None = None,
    app: dict | None = None,
    include_mic: bool = False,
    synthetic: dict | None = None,
    max_length_minutes: float | None = None,
    silence_pause_minutes: float | None = None,
    allow_simulated_events: bool = False,
    startup_timeout: float = 20,
    process_on_stop: bool = False,
) -> RecordingHandle:
    """Create a date-time named session under the existing class `class_slug` and start recording into it.

    `app={"pid", "exe", "name"}` (a row from apps.list_audio_apps) records that app's audio instead of a
    mic, plus the microphone `device` if `include_mic`.
    `synthetic={"pattern": "tone"|"silence", "stall_after_s": ...}` records a test signal instead of a mic.
    `max_length_minutes` overrides the default (e.g. from a per-class setting).
    `process_on_stop` marks the session for automatic transcription (the app sets it), so a recording
    that ends while the app is closed is still queued the next time the app starts.
    """
    cfg = config or load_config()
    rc = cfg.recording
    data_dir = cfg.data_dir
    if not (sessions_root(data_dir, class_slug).parent / "class.json").is_file():  # validates the slug first
        # Never create a class folder here: it would show up as a class nobody created.
        raise RecorderError(f"There is no class {class_slug!r}. Create it in the app first (Library -> Add class); "
                            "`lecture-recorder classes` lists the existing ones.")
    if is_locked(data_dir / GLOBAL_LOCK):
        raise RecorderError("A recording is already running.")

    if synthetic is not None:
        source = {"kind": "synthetic", **synthetic}
        rate = rc.sample_rate
    elif app is not None:
        if why := host().app_capture_reason:
            raise RecorderError(why)
        rate = rc.sample_rate  # the app's audio is converted to any rate by Windows
        source = {"kind": "app", "label": f"app:{app['name']}", "app_name": app["name"], "exe": app["exe"],
                  "pid": int(app["pid"]), "include_mic": bool(include_mic)}
        if include_mic:
            from lecture_recorder.recorder.sources import pick_sample_rate, resolve_device

            index, name = resolve_device(device)
            if pick_sample_rate(index, rate) != rate:
                raise RecorderError(f"The microphone {name!r} can't record at {rate} Hz together with the app; "
                                    "record without the microphone or choose another one.")
            source.update(device_index=index, device_name=name)
    else:
        from lecture_recorder.recorder.sources import pick_sample_rate, resolve_device

        index, name = resolve_device(device)
        rate = pick_sample_rate(index, rc.sample_rate)
        source = {"kind": "mic", "device_index": index, "device_name": name}

    session_dir = new_session_dir(data_dir, class_slug)
    from lecture_recorder.naming import title_for

    write_json_atomic(session_dir / META_FILE, {
        "name": session_dir.name,
        "title": title_for(cfg.naming.template, session_dir) or session_dir.name,
        "class": class_slug,
        "created": now_iso(),
        "source": source,
        "sample_rate": rate,
        "channels": 1,
        "sample_format": "s16le",
        "max_length_s": 60 * (max_length_minutes if max_length_minutes is not None else rc.max_length_minutes),
        "silence_pause_s": 60 * (silence_pause_minutes if silence_pause_minutes is not None else rc.silence_pause_minutes),
        "silence_dbfs": rc.silence_dbfs,
        "no_audio_dbfs": rc.no_audio_dbfs,
        "no_audio_warn_s": rc.no_audio_warn_s,
        "pause_on_sleep": rc.pause_on_sleep,
        "process_on_stop": bool(process_on_stop),
        "state": "starting",
        "events": [],
    })

    cmd = [sys.executable, "-m", "lecture_recorder.recorder", "--session-dir", str(session_dir)]
    if allow_simulated_events:
        cmd.append("--allow-simulated-events")
    env = {**os.environ, "LECTURE_RECORDER_HOME": str(cfg.home), "PYTHONUNBUFFERED": "1"}
    with open(session_dir / LOG_FILE, "ab") as log:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env=env, **host().recorder_spawn_kwargs(),
        )

    handle = RecordingHandle(session_dir, proc)
    end = time.monotonic() + startup_timeout
    while time.monotonic() < end:
        if handle.status().get("state") in ("recording", "paused", "stopped", "finalizing"):
            return handle
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    if proc.poll() is None:
        proc.kill()
        proc.wait()
    log_text = (session_dir / LOG_FILE).read_text(encoding="utf-8", errors="replace")[-2000:]
    if proc.returncode == EXIT_BUSY:
        raise RecorderError("A recording is already running.")
    raise RecorderError(f"Recorder failed to start (exit {proc.returncode}):\n{log_text}")
