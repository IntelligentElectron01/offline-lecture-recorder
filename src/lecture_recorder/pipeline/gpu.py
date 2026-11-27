"""One heavy job (transcription, speakers, summary, embedding) at a time, across processes, plus VRAM readings for
logs and benchmarks. On a Mac there is no GPU job, but the same lock keeps the machine to one heavy job."""
import os
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from lecture_recorder.recorder.platform import ProcessLock

GPU_LOCK = ".gpu.lock"


class GpuBusy(RuntimeError):
    pass


@contextmanager
def gpu_job(lock_dir: Path | None = None, wait_s: float = 3600, poll_s: float = 0.5):
    """Hold the GPU lock (`<app home>/.gpu.lock`) for the duration of the block.

    The lock lives in the app home, not the data folder, so jobs on any data folder (tests, the
    benchmark, the app) share it. The OS frees the lock if the holder dies, so a crash can never
    leave the GPU "busy".
    """
    from lecture_recorder.runtime_env import app_home

    lock = ProcessLock(Path(lock_dir or app_home()) / GPU_LOCK)
    deadline = time.monotonic() + wait_s
    while not lock.acquire():
        if time.monotonic() >= deadline:
            raise GpuBusy(f"another GPU job is still running (lock: {lock.path})")
        time.sleep(poll_s)
    try:
        yield
    finally:
        lock.release()


heavy_job = gpu_job  # the name that fits every platform; the lock file stays `.gpu.lock`

_dll_dirs = []


def register_cuda_dlls() -> None:
    """Let ctranslate2 find cuBLAS/cuDNN in the torch wheel's lib folder (no separate CUDA install).

    Importing torch does this as a side effect; this makes it explicit so the order of imports
    doesn't matter. Windows only; elsewhere the loader finds the libraries through rpath.
    """
    if os.name != "nt" or _dll_dirs:
        return
    import importlib.util

    spec = importlib.util.find_spec("torch")
    for root in (spec.submodule_search_locations or []) if spec else []:
        lib = Path(root) / "lib"
        if lib.is_dir():
            _dll_dirs.append(os.add_dll_directory(str(lib)))
            os.environ["PATH"] = f"{lib}{os.pathsep}{os.environ.get('PATH', '')}"


def cuda_available() -> bool:
    try:
        register_cuda_dlls()
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def nvidia_smi_memory() -> dict | None:
    """{'used_mb', 'free_mb', 'total_mb'} for GPU 0, or None without an NVIDIA driver."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.splitlines()[0]
        used, free, total = (int(v) for v in out.split(","))
        return {"used_mb": used, "free_mb": free, "total_mb": total}
    except Exception:
        return None


def vram_used_mb() -> int | None:
    m = nvidia_smi_memory()
    return m["used_mb"] if m else None


def settle_vram(baseline: int | None, wait_s: float = 20, slack_mb: int = 300) -> int | None:
    """VRAM after an unload; the driver/Ollama free it asynchronously, so wait (bounded) for it."""
    now = vram_used_mb()
    deadline = time.monotonic() + wait_s
    while baseline is not None and now is not None and now > baseline + slack_mb and time.monotonic() < deadline:
        time.sleep(0.5)
        now = vram_used_mb()
    return now


class VramPeak:
    """Polls nvidia-smi in a thread and keeps the highest device-wide 'used' value (MB). Each poll
    starts a process, so once a second (model loads take several seconds; peaks are not missed)."""

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.peak_mb: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            m = nvidia_smi_memory()
            if m:
                self.peak_mb = max(self.peak_mb or 0, m["used_mb"])
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
