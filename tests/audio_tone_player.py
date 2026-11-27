"""Test helper (not a test): play a quiet sine through the default output until killed or SECONDS pass.

usage: python audio_tone_player.py FREQ AMP SECONDS [--own-volume X]
--own-volume sets this player's own per-app volume (its slider in Windows' Volume Mixer, which Windows
remembers per app); it is set back to 1.0 before exiting. The system volume is never touched.
Prints PLAYING once the stream runs.
"""
import ctypes
import ctypes.wintypes as wt
import sys
import time

import numpy as np
import sounddevice as sd

from lecture_recorder.recorder import winaudio as W

freq, amp, secs = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
own = float(sys.argv[sys.argv.index("--own-volume") + 1]) if "--own-volume" in sys.argv else None
RATE = 48000
pos = 0


def cb(out, frames, t, status):
    global pos
    x = amp * np.sin(2 * np.pi * freq * np.arange(pos, pos + frames) / RATE)
    out[:] = np.repeat(x[:, None], out.shape[1], axis=1).astype(np.float32)
    pos += frames


def own_session_volume():
    dev = W._default_render_device()
    mgr, vol = ctypes.c_void_p(), ctypes.c_void_p()
    W._check(W._method(dev, 3, ctypes.POINTER(W.GUID), wt.DWORD, ctypes.c_void_p, W.PP)(
        ctypes.byref(W.IID_IAudioSessionManager2), W.CLSCTX_ALL, None, ctypes.byref(mgr)), "Activate")
    W._check(W._method(mgr, 4, ctypes.c_void_p, wt.DWORD, W.PP)(None, 0, ctypes.byref(vol)), "GetSimpleAudioVolume")
    W._release(mgr)
    W._release(dev)
    return W._method(vol, 3, ctypes.c_float, ctypes.c_void_p)  # SetMasterVolume(level, context)


with sd.OutputStream(samplerate=RATE, channels=2, dtype="float32", callback=cb):
    setvol = own_session_volume() if own is not None else None
    try:
        if setvol:
            setvol(own, None)
        print("PLAYING", flush=True)
        time.sleep(secs)
    finally:
        if setvol:
            setvol(1.0, None)
