"""Machine state that decides whether a GPU job may start (project rules: efficiency).

Hard rule in every processing mode: never start a GPU job while recording or on battery. A job that
is already running finishes; the remaining jobs wait in the queue.

Power status comes from the host (Windows: GetSystemPowerStatus through ctypes; macOS: `pmset`). Linux, a
desktop without a battery, and an unknown status all count as plugged in.
"""
from pathlib import Path

ON_BATTERY, RECORDING = "on battery", "recording in progress"


def power_source() -> str:
    """'ac', 'battery', or 'unknown' (treated as plugged in). Asked of this computer's host (host/)."""
    from lecture_recorder.host import host

    return host().power_source()


def gpu_block_reason(data_dir: Path) -> str | None:
    """Why a GPU job may not start now, or None. Checked before every queued job and concept query."""
    from lecture_recorder.recorder.platform import is_locked
    from lecture_recorder.recorder.process import GLOBAL_LOCK

    if is_locked(Path(data_dir) / GLOBAL_LOCK):
        return RECORDING
    if power_source() == "battery":
        return ON_BATTERY
    return None
