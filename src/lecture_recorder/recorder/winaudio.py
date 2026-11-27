"""Windows audio plumbing for "Record from an app" (ctypes only, no extra package).

- `ProcessLoopback`: WASAPI process loopback (Windows 10 build 20348+ / Windows 11). It reads one
  process tree's audio output locally, before the system volume, as 16-bit mono at the requested rate.
  It is passive: nothing is injected into, hooked in or sent to the other app.
- `audio_sessions()`: processes with an audio session on the default output (IAudioSessionManager2).
- `process_table()`, `exe_path()`, `friendly_name()`: names for the picker.
- `ExitWatcher`: waits on a process handle (an event, no polling) and calls back when it exits.

All COM calls go through raw vtables; indexes are from the Windows SDK headers.
"""
import ctypes
import ctypes.wintypes as wt
import sys
import threading

import numpy as np

HRESULT = ctypes.c_long
S_OK = 0
E_NOINTERFACE = 0x80004002 - (1 << 32)
PP = ctypes.POINTER(ctypes.c_void_p)
MIN_BUILD = 20348  # process loopback


class GUID(ctypes.Structure):
    _fields_ = [("Data1", wt.DWORD), ("Data2", wt.WORD), ("Data3", wt.WORD), ("Data4", ctypes.c_ubyte * 8)]

    @classmethod
    def from_str(cls, s: str) -> "GUID":
        import uuid

        return cls.from_buffer_copy(uuid.UUID(s).bytes_le)

    def same(self, other: "GUID") -> bool:
        return bytes(self) == bytes(other)


IID_IUnknown = GUID.from_str("00000000-0000-0000-C000-000000000046")
IID_IAgileObject = GUID.from_str("94EA2B94-E9CC-49E0-C0FF-EE64CA8F5B90")
IID_IActivateAudioInterfaceCompletionHandler = GUID.from_str("41D949AB-9862-444A-80F6-C261334DA5EB")
IID_IAudioClient = GUID.from_str("1CB9AD4C-DBFA-4C32-B178-C2F568A703B2")
IID_IAudioCaptureClient = GUID.from_str("C8ADBD64-E71E-48A0-A4DE-185C395CD317")
CLSID_MMDeviceEnumerator = GUID.from_str("BCDE0395-E52F-467C-8E3D-C4579291692E")
IID_IMMDeviceEnumerator = GUID.from_str("A95664D2-9614-4F35-A746-DE8DB63617E6")
IID_IAudioSessionManager2 = GUID.from_str("77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F")
IID_IAudioSessionControl2 = GUID.from_str("BFB7FF88-7239-4FC9-8FA2-07C950BE9C6D")

VT_BLOB = 65
CLSCTX_ALL = 23
LOOPBACK_DEVICE = r"VAD\Process_Loopback"
ACTIVATION_TYPE_PROCESS_LOOPBACK = 1
INCLUDE_TARGET_PROCESS_TREE = 0
STREAMFLAGS = (0x00020000  # LOOPBACK
               | 0x00040000  # EVENTCALLBACK
               | 0x08000000  # SRC_DEFAULT_QUALITY
               | 0x80000000)  # AUTOCONVERTPCM: the engine converts to our 16-bit mono rate
BUFFERFLAGS_SILENT = 0x2
SESSION_ACTIVE = 1


class _ActivationParams(ctypes.Structure):
    _fields_ = [("ActivationType", ctypes.c_int), ("TargetProcessId", wt.DWORD), ("ProcessLoopbackMode", ctypes.c_int)]


class _Blob(ctypes.Structure):
    _fields_ = [("cbSize", wt.ULONG), ("pBlobData", ctypes.c_void_p)]


class _PropVariant(ctypes.Structure):
    _fields_ = [("vt", wt.USHORT), ("r1", wt.WORD), ("r2", wt.WORD), ("r3", wt.WORD), ("blob", _Blob)]


class _WaveFormatEx(ctypes.Structure):
    _fields_ = [("wFormatTag", wt.WORD), ("nChannels", wt.WORD), ("nSamplesPerSec", wt.DWORD),
                ("nAvgBytesPerSec", wt.DWORD), ("nBlockAlign", wt.WORD), ("wBitsPerSample", wt.WORD), ("cbSize", wt.WORD)]


def supported() -> str | None:
    """None if process loopback is available here, else why not (shown in the UI)."""
    if sys.platform != "win32":
        return "Recording from an app needs Windows 10 (build 20348 or later) or Windows 11."
    build = sys.getwindowsversion().build
    if build < MIN_BUILD:
        return f"Recording from an app needs Windows 11 or Windows 10 build {MIN_BUILD}+ (this is build {build})."
    return None


# ---------------------------------------------------------------- COM helpers
def _dll(name: str):
    return ctypes.OleDLL(name) if name in ("ole32", "Mmdevapi") else ctypes.WinDLL(name, use_last_error=True)


def _method(ptr, index: int, *argtypes, restype=HRESULT):
    vtbl = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    fn = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[index])
    return lambda *a: fn(ptr, *a)


def _check(hr: int, what: str) -> int:
    if hr < 0:
        raise OSError(f"{what} failed (0x{hr & 0xFFFFFFFF:08X})")
    return hr


def _release(ptr) -> None:
    if ptr:
        _method(ptr, 2, restype=wt.ULONG)()


def com_init() -> None:
    """Multithreaded apartment on this thread; a thread already initialized another way keeps it."""
    try:
        _dll("ole32").CoInitializeEx(None, 0)
    except OSError:
        pass


_QI = ctypes.WINFUNCTYPE(HRESULT, ctypes.c_void_p, ctypes.POINTER(GUID), PP)
_REF = ctypes.WINFUNCTYPE(wt.ULONG, ctypes.c_void_p)
_DONE = ctypes.WINFUNCTYPE(HRESULT, ctypes.c_void_p, ctypes.c_void_p)


_ACTIVATIONS: list = []  # every handler (+ its params) lives as long as the process: see _activate_loopback


class _CompletionHandler:
    """Minimal agile COM object for ActivateAudioInterfaceAsync. Reference counting is a no-op, so the
    object is never freed (kept in _ACTIVATIONS): Windows may call it after a timeout, or still be
    returning from the callback after done.set(). A few hundred bytes per recording start."""

    def __init__(self):
        self.done = threading.Event()
        self.hr = 0
        self.client = ctypes.c_void_p()

        def qi(this, riid, ppv):
            if any(riid.contents.same(i) for i in (IID_IUnknown, IID_IAgileObject, IID_IActivateAudioInterfaceCompletionHandler)):
                ppv[0] = this
                return S_OK
            ppv[0] = None
            return E_NOINTERFACE

        def completed(this, op):
            try:
                act_hr, unk = HRESULT(), ctypes.c_void_p()
                hr = _method(op, 3, ctypes.POINTER(HRESULT), PP)(ctypes.byref(act_hr), ctypes.byref(unk))
                self.hr = hr if hr < 0 else act_hr.value
                if self.hr >= 0 and unk:
                    self.hr = _method(unk, 0, ctypes.POINTER(GUID), PP)(ctypes.byref(IID_IAudioClient), ctypes.byref(self.client))
                _release(unk)
            finally:
                self.done.set()
            return S_OK

        self._fns = (_QI(qi), _REF(lambda this: 1), _REF(lambda this: 1), _DONE(completed))
        self._vtbl = (ctypes.c_void_p * 4)(*[ctypes.cast(f, ctypes.c_void_p) for f in self._fns])
        self._obj = ctypes.c_void_p(ctypes.addressof(self._vtbl))
        self.ptr = ctypes.addressof(self._obj)


def _activate_loopback(pid: int, timeout: float = 5.0) -> ctypes.c_void_p:
    params = _ActivationParams(ACTIVATION_TYPE_PROCESS_LOOPBACK, pid, INCLUDE_TARGET_PROCESS_TREE)
    pv = _PropVariant(vt=VT_BLOB, blob=_Blob(ctypes.sizeof(params), ctypes.addressof(params)))
    handler, op = _CompletionHandler(), ctypes.c_void_p()
    _ACTIVATIONS.append((handler, params, pv))
    fn = _dll("Mmdevapi").ActivateAudioInterfaceAsync
    fn.argtypes = [wt.LPCWSTR, ctypes.POINTER(GUID), ctypes.POINTER(_PropVariant), ctypes.c_void_p, PP]
    fn(LOOPBACK_DEVICE, ctypes.byref(IID_IAudioClient), ctypes.byref(pv), handler.ptr, ctypes.byref(op))
    try:
        if not handler.done.wait(timeout):
            raise OSError("the app's audio could not be opened (no answer from Windows)")
    finally:
        _release(op)
    _check(handler.hr, "opening the app's audio")
    return handler.client


# ---------------------------------------------------------------- capture
class ProcessLoopback:
    """Capture process `pid` and its child processes as mono int16 at `rate`.

    `read(timeout)` returns at least `min_frames` (one 100 ms block by default, so the recorder loop
    runs 10x/s, not on every 10 ms engine event), or what arrived before the timeout, or None.
    """

    def __init__(self, pid: int, rate: int):
        self.pid, self.rate = pid, rate
        self.min_frames = rate // 10
        self._client = self._capture = None
        self._event = None
        self._k32 = _dll("kernel32")

    def start(self) -> None:
        com_init()
        client = _activate_loopback(self.pid)
        try:
            fmt = _WaveFormatEx(1, 1, self.rate, self.rate * 2, 2, 16, 0)  # WAVE_FORMAT_PCM, mono, 16-bit
            init = _method(client, 3, ctypes.c_int, wt.DWORD, ctypes.c_longlong, ctypes.c_longlong,
                           ctypes.POINTER(_WaveFormatEx), ctypes.c_void_p)
            _check(init(0, STREAMFLAGS, 2_000_000, 0, ctypes.byref(fmt), None), "IAudioClient.Initialize")  # 200 ms buffer
            self._k32.CreateEventW.restype = wt.HANDLE
            self._event = self._k32.CreateEventW(None, False, False, None)
            _check(_method(client, 13, wt.HANDLE)(self._event), "IAudioClient.SetEventHandle")
            cap = ctypes.c_void_p()
            _check(_method(client, 14, ctypes.POINTER(GUID), PP)(ctypes.byref(IID_IAudioCaptureClient), ctypes.byref(cap)),
                   "IAudioClient.GetService")
            self._capture = cap
            self._get = _method(cap, 3, ctypes.POINTER(ctypes.POINTER(ctypes.c_int16)), ctypes.POINTER(ctypes.c_uint32),
                                ctypes.POINTER(wt.DWORD), ctypes.c_void_p, ctypes.c_void_p)
            self._free = _method(cap, 4, ctypes.c_uint32)
            self._next = _method(cap, 5, ctypes.POINTER(ctypes.c_uint32))
            self._client = client
            _check(_method(client, 10)(), "IAudioClient.Start")
        except BaseException:
            self._client = client
            self.stop()
            raise

    def _drain(self, out: list) -> int:
        n, got = ctypes.c_uint32(), 0
        while True:
            _check(self._next(ctypes.byref(n)), "GetNextPacketSize")
            if not n.value:
                return got
            data, frames, flags = ctypes.POINTER(ctypes.c_int16)(), ctypes.c_uint32(), wt.DWORD()
            _check(self._get(ctypes.byref(data), ctypes.byref(frames), ctypes.byref(flags), None, None), "GetBuffer")
            if flags.value & BUFFERFLAGS_SILENT:
                out.append(np.zeros(frames.value, np.int16))
            else:
                out.append(np.ctypeslib.as_array(data, (frames.value,)).copy())
            self._free(frames.value)
            got += frames.value

    def read(self, timeout: float) -> np.ndarray | None:
        out: list[np.ndarray] = []
        got = 0
        wait = self._k32.WaitForSingleObject
        wait.argtypes = [wt.HANDLE, wt.DWORD]
        ms = max(1, int(timeout * 1000))
        while got < self.min_frames:
            if wait(self._event, ms) != 0:  # WAIT_OBJECT_0
                break
            got += self._drain(out)
        return np.concatenate(out) if out else None

    def stop(self) -> None:
        if self._client:
            try:
                _method(self._client, 11)()
            except OSError:
                pass
        _release(self._capture)
        _release(self._client)
        self._capture = self._client = None
        if self._event:
            self._k32.CloseHandle(wt.HANDLE(self._event))
            self._event = None


# ---------------------------------------------------------------- sessions and processes
def _enumerator():
    com_init()
    en = ctypes.c_void_p()
    _dll("ole32").CoCreateInstance(ctypes.byref(CLSID_MMDeviceEnumerator), None, CLSCTX_ALL,
                                   ctypes.byref(IID_IMMDeviceEnumerator), ctypes.byref(en))
    return en


def _default_render_device():
    en, dev = _enumerator(), ctypes.c_void_p()
    try:
        _check(_method(en, 4, ctypes.c_int, ctypes.c_int, PP)(0, 1, ctypes.byref(dev)), "GetDefaultAudioEndpoint")
    finally:
        _release(en)
    return dev


def _render_devices() -> list:
    """Every active output device (an app such as Zoom may play to a headset that isn't the default)."""
    en, coll, out = _enumerator(), ctypes.c_void_p(), []
    try:
        _check(_method(en, 3, ctypes.c_int, wt.DWORD, PP)(0, 1, ctypes.byref(coll)), "EnumAudioEndpoints")  # eRender, ACTIVE
        n = wt.UINT()
        _check(_method(coll, 3, ctypes.POINTER(wt.UINT))(ctypes.byref(n)), "GetCount")
        for i in range(n.value):
            dev = ctypes.c_void_p()
            if _method(coll, 4, wt.UINT, PP)(i, ctypes.byref(dev)) >= 0:
                out.append(dev)
    finally:
        _release(coll)
        _release(en)
    return out


def audio_sessions() -> list[tuple[int, bool]]:
    """(pid, active) for every app audio session on every active output (pid 0 = system sounds, skipped).
    A pid can appear once per device."""
    out = []
    for dev in _render_devices():
        try:
            out += _device_sessions(dev)
        except OSError:
            pass  # a device that went away meanwhile
        finally:
            _release(dev)
    return out


def _device_sessions(dev) -> list[tuple[int, bool]]:
    mgr, en = ctypes.c_void_p(), ctypes.c_void_p()
    out = []
    try:
        _check(_method(dev, 3, ctypes.POINTER(GUID), wt.DWORD, ctypes.c_void_p, PP)(
            ctypes.byref(IID_IAudioSessionManager2), CLSCTX_ALL, None, ctypes.byref(mgr)), "IMMDevice.Activate")
        _check(_method(mgr, 5, PP)(ctypes.byref(en)), "GetSessionEnumerator")
        n = ctypes.c_int()
        _check(_method(en, 3, ctypes.POINTER(ctypes.c_int))(ctypes.byref(n)), "GetCount")
        for i in range(n.value):
            ctl, ctl2 = ctypes.c_void_p(), ctypes.c_void_p()
            if _method(en, 4, ctypes.c_int, PP)(i, ctypes.byref(ctl)) < 0:
                continue
            try:
                if _method(ctl, 0, ctypes.POINTER(GUID), PP)(ctypes.byref(IID_IAudioSessionControl2), ctypes.byref(ctl2)) < 0:
                    continue
                pid, state = wt.DWORD(), ctypes.c_int()
                _method(ctl2, 14, ctypes.POINTER(wt.DWORD))(ctypes.byref(pid))  # GetProcessId
                _method(ctl2, 3, ctypes.POINTER(ctypes.c_int))(ctypes.byref(state))  # GetState
                if pid.value:
                    out.append((pid.value, state.value == SESSION_ACTIVE))
            finally:
                _release(ctl2)
                _release(ctl)
    finally:
        _release(en)
        _release(mgr)
    return out


class _ProcessEntry(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ProcessID", wt.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wt.DWORD), ("cntThreads", wt.DWORD),
                ("th32ParentProcessID", wt.DWORD), ("pcPriClassBase", ctypes.c_long), ("dwFlags", wt.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


def process_table() -> dict[int, tuple[int, str]]:
    """pid -> (parent pid, exe file name) for every running process (one snapshot)."""
    k32 = _dll("kernel32")
    k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if snap in (None, wt.HANDLE(-1).value):
        return {}
    out = {}
    try:
        e = _ProcessEntry(dwSize=ctypes.sizeof(_ProcessEntry))
        ok = k32.Process32FirstW(wt.HANDLE(snap), ctypes.byref(e))
        while ok:
            out[e.th32ProcessID] = (e.th32ParentProcessID, e.szExeFile)
            ok = k32.Process32NextW(wt.HANDLE(snap), ctypes.byref(e))
    finally:
        k32.CloseHandle(wt.HANDLE(snap))
    return out


PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000


def _open(pid: int, access: int):
    k32 = _dll("kernel32")
    k32.OpenProcess.restype = wt.HANDLE
    return k32.OpenProcess(access, False, pid)


def exe_path(pid: int) -> str | None:
    h = _open(pid, PROCESS_QUERY_LIMITED_INFORMATION)
    if not h:
        return None
    k32 = _dll("kernel32")
    try:
        buf, size = ctypes.create_unicode_buffer(32768), wt.DWORD(32768)
        return buf.value if k32.QueryFullProcessImageNameW(wt.HANDLE(h), 0, buf, ctypes.byref(size)) else None
    finally:
        k32.CloseHandle(wt.HANDLE(h))


def friendly_name(path: str | None) -> str | None:
    """The exe's "File description" (e.g. "Zoom Meetings"), or None."""
    if not path:
        return None
    try:
        import win32api

        lang, cp = win32api.GetFileVersionInfo(path, r"\VarFileInfo\Translation")[0]
        desc = win32api.GetFileVersionInfo(path, rf"\StringFileInfo\{lang:04x}{cp:04x}\FileDescription")
        return str(desc).strip() or None
    except Exception:
        return None


class ExitWatcher:
    """Calls `on_exit()` once when process `pid` ends. A thread waits on the process handle and a stop
    event together (WaitForMultipleObjects): no polling. Raises ProcessLookupError if `pid` is gone."""

    def __init__(self, pid: int, on_exit):
        self._k32 = _dll("kernel32")
        self._proc = _open(pid, SYNCHRONIZE)
        if not self._proc:
            raise ProcessLookupError(f"process {pid} is not running")
        self._k32.CreateEventW.restype = wt.HANDLE
        self._stop = self._k32.CreateEventW(None, True, False, None)
        self._on_exit = on_exit
        self._thread = threading.Thread(target=self._run, name="app-exit-watch", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        handles = (wt.HANDLE * 2)(self._proc, self._stop)
        self._k32.WaitForMultipleObjects.argtypes = [wt.DWORD, ctypes.POINTER(wt.HANDLE), wt.BOOL, wt.DWORD]
        if self._k32.WaitForMultipleObjects(2, handles, False, 0xFFFFFFFF) == 0:  # INFINITE; 0 = the process
            self._on_exit()

    def close(self) -> None:
        if self._stop:
            self._k32.SetEvent(wt.HANDLE(self._stop))
            self._thread.join(2)
            self._k32.CloseHandle(wt.HANDLE(self._stop))
            self._k32.CloseHandle(wt.HANDLE(self._proc))
            self._stop = self._proc = None
