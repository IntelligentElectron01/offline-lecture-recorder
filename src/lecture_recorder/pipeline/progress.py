"""Per-job progress: the job process writes small JSON snapshots that the app reads on its status poll.

The queue worker sets LECTURE_RECORDER_PROGRESS to `<data>/queue/progress/<job id>.json` for each job and
removes the file when the job ends. A snapshot holds only stage names and counters (never transcript text):
{"stage", "sub", "done", "total", "unit", "stage_started", "updated"} with wall-clock times in seconds.

Fractions come only from real counters (seconds of audio decoded, parts, passages, pyannote's own chunk
counts); a stage without one has no total and shows as indeterminate. Writes are throttled (a stage change
always writes, counter updates at most every `min_interval` s), not fsynced, and never raise: progress can
never fail a job.
"""
import os
import time
from pathlib import Path

ENV = "LECTURE_RECORDER_PROGRESS"
STALE_S = 30           # an older snapshot keeps its label but shows no time left
ETA_MIN_FRACTION = 0.05
ETA_MIN_ELAPSED_S = 20
_MAX_NAME = 32


class Progress:
    def __init__(self, path: Path, min_interval: float = 1.5, clock=time.monotonic, wall=time.time):
        self.path, self.min_interval, self.clock, self.wall = Path(path), min_interval, clock, wall
        self._state: dict = {}
        self._last: float | None = None

    def stage(self, name: str, total: float | None = None, unit: str | None = None, sub: str | None = None) -> None:
        self._state = {"stage": name, "sub": sub, "done": 0 if total else None, "total": total or None,
                       "unit": unit, "stage_started": self.wall()}
        self._write(force=True)

    def update(self, done: float) -> None:
        if not self._state.get("total"):
            return
        self._state["done"] = done
        self._write()

    def diarization_hook(self):
        """pyannote's `hook(step_name, artifact, file=, total=, completed=)`: segmentation and embedding
        batches are counted; clustering (after the embeddings) has no counter."""
        def hook(step_name, step_artifact=None, file=None, total=None, completed=None):
            try:
                if step_name in ("segmentation", "embeddings") and total:
                    if self._state.get("sub") != step_name:
                        self.stage("speakers", total=total, sub=step_name)
                    self.update(min(completed or 0, total))
                elif step_name == "embeddings" and step_artifact is not None:
                    self.stage("speakers", sub="clustering")
            except Exception:  # never let progress reporting break the diarization
                pass
        return hook

    def _write(self, force: bool = False) -> None:
        now = self.clock()
        if not force and self._last is not None and now - self._last < self.min_interval:
            return
        self._last = now
        from lecture_recorder.recorder.session import write_json_atomic

        try:
            write_json_atomic(self.path, {**self._state, "updated": self.wall()}, retries=3, fsync=False)
        except Exception:  # e.g. the app is reading it right now: the next update writes again
            try:
                self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp").unlink(missing_ok=True)
            except OSError:
                pass


class NullProgress:
    """Used when no progress file was asked for (tests, the CLI's own calls)."""

    def stage(self, *a, **k) -> None:
        pass

    def update(self, *a, **k) -> None:
        pass

    def diarization_hook(self):
        return None


NULL = NullProgress()


def from_env() -> Progress | NullProgress:
    p = os.environ.get(ENV)
    return Progress(Path(p)) if p else NULL


def _name(v) -> str | None:
    return v if isinstance(v, str) and 0 < len(v) <= _MAX_NAME else None


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def describe(raw, now: float) -> dict | None:
    """What the UI shows for a snapshot: the fraction only from a real counter, and a time left in this
    step only when f >= 5 %, the step has run >= 20 s and the snapshot is fresh (< 30 s old)."""
    if not isinstance(raw, dict) or not _name(raw.get("stage")):
        return None
    done, total = _num(raw.get("done")), _num(raw.get("total"))
    fraction = min(1.0, max(0.0, done / total)) if done is not None and total and total > 0 else None
    updated, started = _num(raw.get("updated")), _num(raw.get("stage_started"))
    stale = updated is None or now - updated > STALE_S
    eta = None
    if fraction is not None and fraction >= ETA_MIN_FRACTION and not stale and started is not None:
        elapsed = updated - started
        if elapsed >= ETA_MIN_ELAPSED_S:
            eta = max(0, round(elapsed * (1 - fraction) / fraction - (now - updated)))
    return {"stage": raw["stage"], "sub": _name(raw.get("sub")), "done": done, "total": total,
            "unit": _name(raw.get("unit")), "fraction": fraction, "eta_s": eta, "stale": stale}
