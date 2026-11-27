"""OS-specific pieces behind small interfaces: keep-awake, power/lid events, and process locks.

Windows has real implementations; other platforms get safe fallbacks (no keep-awake, no power
events: the recorder's wall-clock gap check still pauses after a sleep).
"""
import os
import queue
import sys
import threading
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"


# ---------------------------------------------------------------- console window closed
CLOSE_EVENTS = (2, 5, 6)  # CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT


def console_close_handler(on_close):
    """The console control handler: on close, logoff or shutdown run `on_close` (Windows waits for it,
    about 5 s for a closed window, then ends the process) and report the event handled. Ctrl+C and
    Ctrl+Break stay with Python (uvicorn stops, then the caller's cleanup runs)."""
    def handler(event: int) -> bool:
        if event not in CLOSE_EVENTS:
            return False
        try:
            on_close()
        except Exception:
            pass  # the process ends anyway; anything left unfinished is recovered at the next start
        return True
    return handler


def on_console_close(on_close):
    """Call `on_close` when the console window is closed (Windows). Returns the registered callback,
    which the caller must keep referenced; None where there is no console or no such event."""
    if not IS_WINDOWS:
        return None
    import ctypes
    from ctypes import wintypes

    routine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)(console_close_handler(on_close))
    return routine if ctypes.windll.kernel32.SetConsoleCtrlHandler(routine, True) else None


# ---------------------------------------------------------------- keep-awake
class KeepAwake:
    """Per-thread request that the system not idle-sleep. No system settings are changed.

    SetThreadExecutionState applies to the calling thread, so hold() and release() must be called
    from the same thread (the recorder's main loop; the queue worker's thread). The request also ends
    when the process exits. No ES_DISPLAY_REQUIRED: the screen may still turn off.
    """

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self):
        self.held = False

    def _set(self, flags: int) -> bool:
        from lecture_recorder.host import host

        return host().keep_awake(bool(flags & self.ES_SYSTEM_REQUIRED), flags, self)

    def hold(self) -> None:
        if not self.held:
            self._set(self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED)
            self.held = True

    def release(self) -> None:
        if self.held:
            self._set(self.ES_CONTINUOUS)
            self.held = False


# ---------------------------------------------------------------- power events
class PowerEvents:
    """Puts "suspend", "resume_from_sleep", "lid_closed", "lid_open" on `self.events`."""

    def __init__(self):
        self.events: queue.Queue[str] = queue.Queue()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class WindowsPowerEvents(PowerEvents):
    """Hidden top-level window (message-only windows don't get WM_POWERBROADCAST) on its own thread."""

    WM_POWERBROADCAST = 0x0218
    WM_CLOSE = 0x0010
    PBT_APMSUSPEND = 0x0004
    PBT_APMRESUMESUSPEND = 0x0007
    PBT_APMRESUMEAUTOMATIC = 0x0012
    PBT_POWERSETTINGCHANGE = 0x8013
    GUID_LIDSWITCH_STATE_CHANGE = "{BA3E0F4D-B817-4094-A2D1-D56379E6A0F3}"

    def __init__(self):
        super().__init__()
        self._hwnd = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._lid_seen_initial = False
        self.error: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="power-events", daemon=True)
        self._thread.start()
        self._ready.wait(5)

    def stop(self) -> None:
        if self._hwnd:
            import win32gui

            try:
                win32gui.PostMessage(self._hwnd, self.WM_CLOSE, 0, 0)
            except Exception:
                pass
        if self._thread:
            self._thread.join(2)

    def _run(self) -> None:
        try:
            import ctypes
            from ctypes import wintypes

            import win32api
            import win32gui

            class POWERBROADCAST_SETTING(ctypes.Structure):
                _fields_ = [("PowerSetting", ctypes.c_byte * 16), ("DataLength", wintypes.DWORD), ("Data", ctypes.c_ubyte * 1)]

            def wndproc(hwnd, msg, wparam, lparam):
                if msg == self.WM_POWERBROADCAST:
                    if wparam == self.PBT_APMSUSPEND:
                        self.events.put("suspend")
                    elif wparam in (self.PBT_APMRESUMESUSPEND, self.PBT_APMRESUMEAUTOMATIC):
                        self.events.put("resume_from_sleep")
                    elif wparam == self.PBT_POWERSETTINGCHANGE and lparam:
                        s = ctypes.cast(lparam, ctypes.POINTER(POWERBROADCAST_SETTING)).contents
                        if bytes(s.PowerSetting) == lid_guid_bytes and s.DataLength >= 1:
                            # Windows sends the current lid state right after registering; skip it.
                            if not self._lid_seen_initial:
                                self._lid_seen_initial = True
                            else:
                                self.events.put("lid_open" if s.Data[0] else "lid_closed")
                    return 1
                if msg == self.WM_CLOSE:
                    win32gui.DestroyWindow(hwnd)
                    win32gui.PostQuitMessage(0)
                    return 0
                return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

            lid_guid_bytes = _guid_bytes(self.GUID_LIDSWITCH_STATE_CHANGE)

            wc = win32gui.WNDCLASS()
            wc.hInstance = win32api.GetModuleHandle(None)
            wc.lpszClassName = f"LectureRecorderPower{os.getpid()}"
            wc.lpfnWndProc = wndproc
            atom = win32gui.RegisterClass(wc)
            self._hwnd = win32gui.CreateWindow(atom, "LectureRecorderPower", 0, 0, 0, 0, 0, 0, 0, wc.hInstance, None)

            guid_buf = (ctypes.c_byte * 16).from_buffer_copy(lid_guid_bytes)
            register = ctypes.windll.user32.RegisterPowerSettingNotification
            register.restype = ctypes.c_void_p
            register.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            handle = register(self._hwnd, ctypes.byref(guid_buf), 0)  # DEVICE_NOTIFY_WINDOW_HANDLE
            if not handle:
                self.error = "lid notifications unavailable"
            self._ready.set()
            win32gui.PumpMessages()
            if handle:
                ctypes.windll.user32.UnregisterPowerSettingNotification(ctypes.c_void_p(handle))
        except Exception as e:  # never take the recorder down; the clock-gap check still works
            self.error = f"power events unavailable: {e}"
            self._ready.set()


def _guid_bytes(guid: str) -> bytes:
    import uuid

    return uuid.UUID(guid).bytes_le


def make_power_events() -> PowerEvents:
    return WindowsPowerEvents() if IS_WINDOWS else PowerEvents()


# ---------------------------------------------------------------- process lock
class ProcessLock:
    """Exclusive lock on a file, held for the life of the process. The OS releases it when the
    process dies (even on `taskkill /F`), so a free lock means the owner is gone."""

    def __init__(self, path: Path):
        self.path = path
        self._f = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        f = open(self.path, "a+b")
        try:
            f.seek(0)
            if IS_WINDOWS:
                import msvcrt

                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            return False
        self._f = f
        return True

    def release(self) -> None:
        if self._f is None:
            return
        try:
            self._f.seek(0)
            if IS_WINDOWS:
                import msvcrt

                msvcrt.locking(self._f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._f.fileno(), fcntl.LOCK_UN)
        finally:
            self._f.close()
            self._f = None


def is_locked(path: Path) -> bool:
    """True if a live process holds the lock at `path`."""
    if not path.exists():
        return False
    probe = ProcessLock(path)
    if probe.acquire():
        probe.release()
        return False
    return True
