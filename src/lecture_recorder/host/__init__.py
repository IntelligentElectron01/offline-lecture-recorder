"""The one place that knows which operating system this is.

Everything OS-specific the app needs sits behind `Host`: battery state, below-normal priority and spawn flags for
child processes, keep-awake, memory and core counts, whether recording from one app exists, and the wording that
names the OS's own things (setup script, "Start menu"). The rest of the code asks `host()` and never checks
`sys.platform`; tests install a fake with `use_host()`. File locks and the console-close handler stay in
recorder/platform.py.

The base class is the safe POSIX behaviour (Linux, and what the macOS host builds on); `windows.py` and `mac.py`
override what differs. Windows behaviour is exactly what the code did before the Mac port.
"""
import os
import subprocess
import sys
from contextlib import contextmanager


class Host:
    name = "posix"
    setup_command = "./setup.sh"
    installer_name = "./setup.sh"  # the one-step installer people are pointed to
    start_text = "run ./run.sh"  # how to start the app
    venv_python = ".venv/bin/python"
    path_sep = "/"
    ollama_start_hint = "start the Ollama app"
    # Why recording from one app's audio isn't offered here, or None when it is.
    app_capture_reason: str | None = "Recording from an app is only available on Windows. Use Zoom's own local recording and import it."
    app_capture_hidden = False  # True: the page doesn't show the option at all

    # ---- machine state
    def power_source(self) -> str:
        """'ac', 'battery', or 'unknown' (treated as plugged in)."""
        return "unknown"

    def has_battery(self) -> bool | None:
        return None

    def ram_gb(self) -> float | None:
        try:
            return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
        except (OSError, ValueError, AttributeError):
            return None

    def perf_cores(self) -> int | None:
        """Fast cores (Apple Silicon's performance cores); None where the OS doesn't tell them apart."""
        return None

    def facts(self) -> dict:
        """OS details `configure` and `doctor` show (no personal data)."""
        return {"os": self.name}

    # ---- processes
    def lower_own_priority(self) -> None:
        """This process only; no system setting."""
        try:
            os.nice(10)
        except (OSError, AttributeError):
            pass

    def job_spawn_kwargs(self) -> dict:
        """Popen arguments for a queue job: its own session, so Ctrl+C in the app's terminal doesn't end it. The
        job lowers its own priority when it starts (`job_child_setup`): preexec_fn isn't safe in a threaded server."""
        return {"start_new_session": True}

    def job_child_setup(self) -> None:
        self.lower_own_priority()

    def recorder_spawn_kwargs(self) -> dict:
        return {"start_new_session": True}

    def keep_awake(self, hold: bool, flags: int, owner) -> bool:
        """Ask the OS not to idle-sleep (hold) or stop asking. `flags`: the Windows request; `owner`: the
        KeepAwake object, for hosts that hold a helper process."""
        return False

    # ---- microphone
    def mic_hint(self) -> str | None:
        """OS-specific advice shown when the recorder hears nothing (e.g. a denied microphone permission)."""
        return None


_host: Host | None = None


def _detect() -> Host:
    if sys.platform == "win32":
        from lecture_recorder.host.windows import WindowsHost

        return WindowsHost()
    if sys.platform == "darwin":
        from lecture_recorder.host.mac import MacHost

        return MacHost()
    return Host()


def host() -> Host:
    global _host
    if _host is None:
        _host = _detect()
    return _host


@contextmanager
def use_host(fake: Host):
    """Tests: act as another OS for the block."""
    global _host
    old, _host = _host, fake
    try:
        yield fake
    finally:
        _host = old


def run_quiet(cmd: list[str], timeout: float = 5) -> str | None:
    """stdout of a short system command, or None if it is missing, fails or hangs."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None
