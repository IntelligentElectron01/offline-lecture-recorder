"""Windows host: exactly what the app did before the Mac port (Win32 calls through ctypes, no extra package)."""
import ctypes
import subprocess
import sys

from lecture_recorder.host import Host


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class WindowsHost(Host):
    name = "windows"
    setup_command = "setup.bat"
    installer_name = "INSTALL.bat"
    start_text = "double-click run.bat"
    venv_python = r".venv\Scripts\python.exe"
    path_sep = "\\"
    ollama_start_hint = "start it from the Start menu"

    @property
    def app_capture_reason(self) -> str | None:  # process loopback needs Windows 10 build 20348+
        from lecture_recorder.recorder import winaudio

        return winaudio.supported()

    def power_source(self) -> str:
        st = _SystemPowerStatus()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
            return "unknown"
        return {0: "battery", 1: "ac"}.get(st.ACLineStatus, "unknown")

    def has_battery(self) -> bool | None:
        st = _SystemPowerStatus()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
            return None
        return st.BatteryFlag not in (128, 255)  # 128 = no system battery, 255 = unknown

    def ram_gb(self) -> float | None:
        try:
            st = _MemoryStatus()
            st.dwLength = ctypes.sizeof(st)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return round(st.ullTotalPhys / 2**30, 1)
        except (OSError, ValueError, AttributeError):
            pass
        return None

    def facts(self) -> dict:
        return {"os": self.name, "build": sys.getwindowsversion().build}

    def lower_own_priority(self) -> None:
        try:
            k32 = ctypes.windll.kernel32
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL_PRIORITY_CLASS
        except Exception:  # only a courtesy to the rest of the machine
            pass

    def job_spawn_kwargs(self) -> dict:
        # Below-normal priority, own process group, no console: Ctrl+C or closing the app's window doesn't end a
        # job behind the queue's back; the app's Quit does.
        return {"creationflags": subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_NO_WINDOW}

    def job_child_setup(self) -> None:
        pass  # the parent's creation flags already did it

    def recorder_spawn_kwargs(self) -> dict:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
                "start_new_session": False}

    def keep_awake(self, hold: bool, flags: int, owner) -> bool:
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))
