"""Audio sources: a microphone (sounddevice), one app's audio output (Windows), and a synthetic
generator for tests.

All deliver mono int16 blocks of ~100 ms through `read()`.
"""
import queue
import threading
import time

import numpy as np


def dbfs(block: np.ndarray) -> float:
    if block.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean((block.astype(np.float64) / 32768.0) ** 2)))
    return 20 * np.log10(rms) if rms > 1e-6 else -120.0


class AudioSource:
    rate: int
    error: str | None = None

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def read(self, timeout: float) -> np.ndarray | None: ...


# ---------------------------------------------------------------- microphone
def list_input_devices() -> list[dict]:
    """Input devices of the default host API (MME on Windows: resamples to any rate, works with every mic)."""
    import sounddevice as sd

    default_in = sd.default.device[0]
    api = sd.query_hostapis(sd.default.hostapi) if sd.default.hostapi >= 0 else None
    allowed = set(api["devices"]) if api else None
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and (allowed is None or i in allowed):
            out.append({"index": i, "name": d["name"], "default_samplerate": d["default_samplerate"], "is_default": i == default_in})
    return out


def resolve_device(spec: str | int | None) -> tuple[int | None, str]:
    """Device index + name from an index, an exact/partial name, or None (system default)."""
    import sounddevice as sd

    devices = list_input_devices()
    if spec is None or spec == "":
        idx = sd.default.device[0]
        return (idx if idx >= 0 else None), (sd.query_devices(idx)["name"] if idx is not None and idx >= 0 else "default")
    if isinstance(spec, int) or str(spec).isdigit():
        idx = int(spec)
        return idx, sd.query_devices(idx)["name"]
    exact = [d for d in devices if d["name"] == spec]
    partial = [d for d in devices if str(spec).lower() in d["name"].lower()]
    match = (exact or partial or [None])[0]
    if match is None:
        raise ValueError(f"no input device matches {spec!r}")
    return match["index"], match["name"]


def pick_sample_rate(device: int | None, preferred: int) -> int:
    import sounddevice as sd

    try:
        sd.check_input_settings(device=device, samplerate=preferred, channels=1, dtype="int16")
        return preferred
    except Exception:
        return int(sd.query_devices(device, "input")["default_samplerate"])


class MicSource(AudioSource):
    def __init__(self, device: int | None, rate: int):
        self.device = device
        self.rate = rate
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None
        self.overflows = 0
        self.error = None

    def _callback(self, indata, frames, time_info, status):
        if status.input_overflow:
            self.overflows += 1
        self._q.put(indata[:, 0].copy())

    def start(self) -> None:
        import sounddevice as sd

        self.error = None
        self._q = queue.Queue()
        self._stream = sd.InputStream(
            device=self.device, samplerate=self.rate, channels=1, dtype="int16",
            blocksize=self.rate // 10, callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def read(self, timeout: float) -> np.ndarray | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


# ---------------------------------------------------------------- one app's audio (Windows process loopback)
class AppSource(AudioSource):
    """One app's audio output (its whole process tree), optionally mixed with a microphone.

    The level doesn't depend on the system volume, but the app's own volume, its slider in Windows'
    Volume Mixer and its mute do lower it. `ended` becomes "app_closed" when the app exits
    (a process-handle wait, no polling); it stays watched while paused.
    """

    MIC_BUFFER_S = 0.5  # the mic's clock drifts against the app's: keep at most this much mic audio

    def __init__(self, pid: int, rate: int, mic: "AudioSource | None" = None, loopback=None, watcher=None):
        from lecture_recorder.recorder import winaudio

        self.pid, self.rate, self.mic = pid, rate, mic
        self.error = None
        self.ended: str | None = None
        self._loopback = loopback or winaudio.ProcessLoopback(pid, rate)
        self._watcher_factory = watcher or winaudio.ExitWatcher
        self._watcher = None
        self._running = False
        self._exited = threading.Event()  # wakes a paused read() as soon as the app closes
        self._mic_buf = np.zeros(0, np.int16)

    @property
    def overflows(self) -> int:
        return getattr(self.mic, "overflows", 0)

    def _app_exited(self) -> None:
        self.ended = "app_closed"
        self._exited.set()

    def start(self) -> None:
        if self._watcher is None:
            self._watcher = self._watcher_factory(self.pid, self._app_exited)  # ProcessLookupError: already gone
        self._loopback.start()
        if self.mic is not None:
            self._mic_buf = np.zeros(0, np.int16)
            try:
                self.mic.start()
            except Exception:
                self._loopback.stop()
                raise
        self._running = True

    def stop(self) -> None:
        self._running = False
        self._loopback.stop()
        if self.mic is not None:
            self.mic.stop()

    def close(self) -> None:
        self.stop()
        if self._watcher is not None:
            self._watcher.close()
            self._watcher = None

    def _mix(self, app: np.ndarray) -> np.ndarray:
        parts = [self._mic_buf]
        while (b := self.mic.read(0)) is not None:
            parts.append(b)
        buf = np.concatenate(parts)[-round(self.MIC_BUFFER_S * self.rate):]
        take = buf[: app.size]
        self._mic_buf = buf[take.size:]
        mixed = app.astype(np.int32)
        mixed[: take.size] += take
        return np.clip(mixed, -32768, 32767).astype(np.int16)

    def read(self, timeout: float) -> np.ndarray | None:
        if not self._running:  # paused: nothing to read; wait like a stopped mic would
            self._exited.wait(timeout)
            return None
        block = self._loopback.read(timeout)
        if block is None or self.mic is None:
            return block
        return self._mix(block)


# ---------------------------------------------------------------- synthetic (tests, plumbing)
class SyntheticSource(AudioSource):
    """Real-time paced test signal. pattern: "tone" (speech-level bursts) or "silence" (-80 dBFS hiss).

    `stall_after_s` makes the source stop delivering after that many seconds, as a dead device would.
    """

    def __init__(self, rate: int = 16000, pattern: str = "tone", stall_after_s: float | None = None):
        self.rate = rate
        self.pattern = pattern
        self.stall_after_s = stall_after_s
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._n = 0
        self.error = None

    def _block(self) -> np.ndarray:
        n = self.rate // 10
        rng = np.random.default_rng(self._n)
        if self.pattern == "silence":
            x = rng.normal(0, 10 ** (-80 / 20), n)
        else:
            t = (np.arange(n) + self._n * n) / self.rate
            on = (self._n // 5) % 2 == 0  # 0.5 s on / 0.5 s off
            x = (0.1 * np.sin(2 * np.pi * 220 * t) if on else 0) + rng.normal(0, 0.003, n)
        self._n += 1
        return np.clip(x * 32767, -32768, 32767).astype(np.int16)

    def _run(self) -> None:
        start = time.monotonic()
        k = 0
        while not self._stop.is_set():
            k += 1
            if self.stall_after_s is not None and k * 0.1 > self.stall_after_s:
                self._stop.wait(0.1)
                continue
            delay = start + k * 0.1 - time.monotonic()
            if delay > 0 and self._stop.wait(delay):
                break
            self._q.put(self._block())

    def start(self) -> None:
        self._stop.clear()
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(1)
            self._thread = None

    def read(self, timeout: float) -> np.ndarray | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None
