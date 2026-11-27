"""macOS host (Apple Silicon). Everything here is plain system commands (`pmset`, `sysctl`, `caffeinate`) run
with a short timeout, so it can be tested on any computer by feeding it their output.

No system setting is changed. Per-app audio capture doesn't exist on a Mac and the page doesn't offer it.
"""
import os
import platform
import subprocess
import time

from lecture_recorder.host import Host, run_quiet

POWER_CACHE_S = 10.0  # pmset is a process start; the queue and the status poll ask often, the answer changes rarely


def parse_pmset(text: str | None) -> tuple[str, bool | None]:
    """(power source, has a battery) from `pmset -g batt`.

    Typical output:
        Now drawing from 'Battery Power'
         -InternalBattery-0 (id=...)	87%; discharging; 3:12 remaining present: true
    A desktop Mac says 'AC Power' and lists no InternalBattery."""
    if not text:
        return "unknown", None
    first = text.splitlines()[0].lower() if text.splitlines() else ""
    battery = "internalbattery" in text.lower()
    if "battery power" in first:
        return "battery", True
    if "ac power" in first or "ups power" in first:
        return "ac", battery
    return "unknown", battery or None


class MacHost(Host):
    name = "macos"
    start_text = "run ./run.sh (or double-click Lecture Recorder.command)"
    ollama_start_hint = "open the Ollama app from your Applications folder"
    app_capture_reason = ("Recording from one app isn't available on a Mac. Use Zoom's own local recording (Zoom settings, "
                          "Recording, \"Record to this Computer\") and import the folder with Import a recording.")
    app_capture_hidden = True

    def __init__(self):
        self._power: tuple[float, tuple[str, bool | None]] | None = None
        self._hw: dict = {}  # core and memory counts don't change while the app runs: asked once

    def _pmset(self) -> tuple[str, bool | None]:
        now = time.monotonic()
        if self._power is None or now - self._power[0] > POWER_CACHE_S:
            self._power = (now, parse_pmset(run_quiet(["pmset", "-g", "batt"], timeout=5)))
        return self._power[1]

    def power_source(self) -> str:
        return self._pmset()[0]

    def has_battery(self) -> bool | None:
        return self._pmset()[1]

    @staticmethod
    def _sysctl(key: str) -> str | None:
        out = run_quiet(["sysctl", "-n", key], timeout=5)
        return out.strip() if out and out.strip() else None

    def ram_gb(self) -> float | None:
        if "ram" not in self._hw:
            mem = self._sysctl("hw.memsize")
            self._hw["ram"] = round(int(mem) / 2**30, 1) if mem and mem.isdigit() else super().ram_gb()
        return self._hw["ram"]

    def perf_cores(self) -> int | None:
        if "perf" not in self._hw:
            self._hw["perf"] = None
            for key in ("hw.perflevel0.logicalcpu", "hw.physicalcpu"):
                v = self._sysctl(key)
                if v and v.isdigit() and int(v) > 0:
                    self._hw["perf"] = int(v)
                    break
        return self._hw["perf"]

    def facts(self) -> dict:
        return {"os": self.name, "arch": platform.machine(), "macos": platform.mac_ver()[0] or None,
                "chip": self._sysctl("machdep.cpu.brand_string"), "rosetta": self._sysctl("sysctl.proc_translated") == "1",
                "perf_cores": self.perf_cores()}

    # `caffeinate -i -w <pid>`: no idle sleep while this process lives, and macOS ends it when we exit.
    def keep_awake(self, hold: bool, flags: int, owner) -> bool:
        proc = getattr(owner, "_caffeinate", None)
        if not hold:
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)  # reaped, not left as a zombie until we exit
                except (OSError, subprocess.SubprocessError):
                    pass
                owner._caffeinate = None
            return True
        if proc is not None and proc.poll() is None:
            return True
        try:
            owner._caffeinate = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except OSError:
            return False

    def mic_hint(self) -> str | None:
        return ("If macOS asked to let Terminal (or Python) use the microphone and you chose Don't Allow, open System Settings, "
                "Privacy & Security, Microphone, switch it on for Terminal, then start the recording again. "
                "Also check the input device in System Settings, Sound, Input.")
