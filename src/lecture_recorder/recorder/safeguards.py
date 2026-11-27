"""Recording state machine: manual controls plus automatic pause/stop rules. Pure logic, no I/O.

Rules (SPEC "Recording safeguards"):
- Sleep or lid close pauses; waking does NOT resume (the user resumes).
- A wall-clock jump between ticks (missed sleep notification) is treated as sleep.
- `silence_pause_s` of continuous near-silence pauses (stays paused until the user resumes).
- `max_length_s` of recorded audio stops the session.
- Recording from an app: the app closing stops the session.
"""
from collections import deque
from dataclasses import dataclass
from enum import Enum


class State(str, Enum):
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass(frozen=True)
class Action:
    kind: str  # "pause" | "resume" | "stop"
    reason: str


class NoAudioMonitor:
    """'No audio detected' warning: the median block level over the last `window_s` of recorded audio
    is below `dbfs`. The median ignores clicks and short noises, so a dead or muted microphone
    (digital silence or 1-bit noise, ~-96 dBFS) is caught 10 s after start and again for any later
    stretch, while a working microphone in a room (lecture clip median -37 dBFS, never below
    -43 dBFS per 100 ms) never triggers it. Pure logic; the recorder reports it in status.json.
    """

    def __init__(self, dbfs: float = -60.0, window_s: float = 10.0):
        self.dbfs, self.window_s = dbfs, window_s
        self.blocks: deque[tuple[float, float]] = deque()
        self.span_s = 0.0
        self.active = False
        self.since_s: float | None = None  # recorded-audio time where the current stretch began

    def reset(self) -> None:
        """After a pause: judge the next `window_s` of audio afresh (a warning stays until audio returns)."""
        self.blocks.clear()
        self.span_s = 0.0

    def on_audio(self, seconds: float, level_dbfs: float, end_audio_s: float) -> str | None:
        """Returns "start" or "end" when the warning changes, else None."""
        self.blocks.append((seconds, level_dbfs))
        self.span_s += seconds
        while self.blocks and self.span_s - self.blocks[0][0] >= self.window_s - 1e-9:
            self.span_s -= self.blocks.popleft()[0]
        if self.span_s < self.window_s - 0.05:  # not enough audio yet to judge (float slack)
            return None
        levels = sorted(level for _, level in self.blocks)
        quiet = levels[len(levels) // 2] < self.dbfs
        if quiet and not self.active:
            quiet_s = sum(sec for sec, level in self.blocks if level < self.dbfs)  # ~ the stretch so far
            self.active, self.since_s = True, max(0.0, end_audio_s - quiet_s)
            return "start"
        if not quiet and self.active:
            self.active, self.since_s = False, None
            return "end"
        return None


PAUSE_EVENTS = {"suspend": "sleep", "lid_closed": "lid_closed"}
WAKE_EVENTS = {"resume_from_sleep", "lid_open"}


class Safeguards:
    def __init__(
        self,
        max_length_s: float,
        silence_pause_s: float,
        silence_dbfs: float,
        pause_on_sleep: bool = True,
        sleep_gap_s: float = 15.0,
    ):
        self.max_length_s = max_length_s
        self.silence_pause_s = silence_pause_s
        self.silence_dbfs = silence_dbfs
        self.pause_on_sleep = pause_on_sleep
        self.sleep_gap_s = sleep_gap_s
        self.state = State.RECORDING
        self.reason = "start"
        self.recorded_s = 0.0
        self.silent_s = 0.0
        self._last_tick: float | None = None

    def _go(self, kind: str, reason: str) -> Action:
        self.state = {"pause": State.PAUSED, "resume": State.RECORDING, "stop": State.STOPPED}[kind]
        self.reason = reason
        if kind == "resume":
            self.silent_s = 0.0
        return Action(kind, reason)

    def max_frames_allowed(self, rate: int) -> int:
        """Frames that may still be written before the length limit."""
        return max(0, round((self.max_length_s - self.recorded_s) * rate))

    def on_audio(self, seconds: float, dbfs: float) -> Action | None:
        if self.state is not State.RECORDING:
            return None
        self.recorded_s += seconds
        self.silent_s = self.silent_s + seconds if dbfs < self.silence_dbfs else 0.0
        if self.recorded_s >= self.max_length_s - 1e-9:
            return self._go("stop", "max_length")
        if self.silent_s >= self.silence_pause_s:
            return self._go("pause", "silence")
        return None

    def on_tick(self, wall_now: float) -> Action | None:
        last, self._last_tick = self._last_tick, wall_now
        if (
            last is not None
            and wall_now - last > self.sleep_gap_s
            and self.state is State.RECORDING
            and self.pause_on_sleep
        ):
            return self._go("pause", "sleep_detected_by_clock_gap")
        return None

    def on_power(self, event: str) -> Action | None:
        if event in PAUSE_EVENTS and self.state is State.RECORDING and self.pause_on_sleep:
            return self._go("pause", PAUSE_EVENTS[event])
        return None  # wake events never resume on their own

    def on_user(self, command: str) -> Action | None:
        if self.state is State.STOPPED:
            return None
        if command == "stop":
            return self._go("stop", "user")
        if command == "pause" and self.state is State.RECORDING:
            return self._go("pause", "user")
        if command == "resume" and self.state is State.PAUSED:
            if self.recorded_s >= self.max_length_s:
                return self._go("stop", "max_length")
            return self._go("resume", "user")
        return None

    def on_source_ended(self, reason: str) -> Action | None:
        """The recorded app closed: stop, whether recording or paused."""
        if self.state is not State.STOPPED:
            return self._go("stop", reason)
        return None

    def on_device_lost(self) -> Action | None:
        if self.state is State.RECORDING:
            return self._go("pause", "device_lost")
        return None
