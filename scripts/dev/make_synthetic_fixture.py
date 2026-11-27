"""Generate the synthetic fixture with Windows SAPI voices (pywin32). No downloads.

Writes (git-ignored) tests/fixtures/synthetic/:
  lecture_clean.wav   16 kHz mono int16: a "main" lecturer plus two short audience questions
  lecture_noisy.wav   the same speech with generated pink noise + mains hum at a fixed SNR
  truth.json          reference text, speaker turns with times, known phrase, SNR

Ground truth comes from the same script that drives the voices, and turn times are measured from
the generated audio (leading/trailing silence trimmed). Deterministic: same voices -> same files.
Synthetic audio is for plumbing, speed and memory only; accuracy is judged on real recordings.

Usage: python scripts/dev/make_synthetic_fixture.py [--out DIR] [--snr-db 10]
"""
import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

RATE = 16000
SAFT16kHz16BitMono = 18  # SpeechAudioFormatType
SVSFIsXML = 8
KNOWN_PHRASE = "the purple elephant theorem"

VOICES = {
    # speaker id: (SAPI voice name fragment, rate, pitch) -- pitch via SAPI XML <pitch absmiddle>
    "main": ("David", 0, 0),
    "audience_a": ("Zira", 1, 0),
    "audience_b": ("Zira", 2, -8),
}
ROLES = {"main": "main", "audience_a": "audience", "audience_b": "audience"}

SCRIPT = [
    ("main", "Good morning everyone. Today we are going to talk about heat engines, efficiency, and entropy. "
             "These ideas connect almost everything we covered in the first unit."),
    ("main", "A heat engine takes energy from a hot reservoir, turns part of it into work, and sends the rest "
             "to a cold reservoir. The fraction that becomes work is called the efficiency."),
    ("main", "No real engine can turn all of its heat into work. In this course I like to call that limit "
             f"{KNOWN_PHRASE}, so please remember that name for the exam."),
    ("audience_a", "Sorry, is that the same thing as the second law?"),
    ("main", "Yes, good question. It is a friendly name for one statement of the second law. The second law "
             "says that the total entropy of an isolated system never decreases."),
    ("main", "Let us look at the Carnot cycle. It has two isothermal steps and two adiabatic steps. "
             "Its efficiency is one minus the cold temperature divided by the hot temperature, "
             "with both temperatures measured in kelvin."),
    ("main", "So if the hot reservoir is six hundred kelvin and the cold reservoir is three hundred kelvin, "
             "the best possible efficiency is one half, or fifty percent."),
    ("audience_b", "Does that mean a real car engine is always below fifty percent?"),
    ("main", "For those temperatures, yes. Real engines lose energy to friction and to heat leaking out, "
             "so they always do worse than the Carnot limit."),
    ("main", "Entropy gives us a way to keep track of this. When heat flows from hot to cold without doing "
             "work, entropy increases, and that energy is no longer available to do useful work."),
    ("audience_a", "Could you repeat the formula for the change in entropy?"),
    ("main", "Of course. For a reversible process the change in entropy equals the heat added divided by "
             "the absolute temperature. Write that down, because we will use it in every homework problem."),
    ("main", "For next week, please read chapter four and try the practice problems at the end. "
             "That is all for today, thank you."),
]


def _find_voice(sapi, fragment: str):
    for token in sapi.GetVoices():
        if fragment.lower() in token.GetDescription().lower():
            return token
    raise SystemExit(f"SAPI voice containing {fragment!r} not found (Windows ships David and Zira).")


def speak(text: str, voice_name: str, rate: int, pitch: int) -> np.ndarray:
    """Synthesize `text` to 16 kHz mono int16 samples in memory."""
    import win32com.client

    voice = win32com.client.Dispatch("SAPI.SpVoice")
    voice.Voice = _find_voice(voice, voice_name)
    voice.Rate = rate
    stream = win32com.client.Dispatch("SAPI.SpMemoryStream")
    fmt = win32com.client.Dispatch("SAPI.SpAudioFormat")
    fmt.Type = SAFT16kHz16BitMono
    stream.Format = fmt
    voice.AudioOutputStream = stream
    xml = f'<pitch absmiddle="{pitch}">{text}</pitch>'
    voice.Speak(xml, SVSFIsXML)
    return np.frombuffer(bytes(stream.GetData()), dtype=np.int16).copy()


def trim(x: np.ndarray, threshold: int = 200) -> np.ndarray:
    idx = np.flatnonzero(np.abs(x.astype(np.int32)) > threshold)
    return x[idx[0]: idx[-1] + 1] if idx.size else x[:0]


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / RATE)
    spectrum[1:] /= np.sqrt(f[1:])
    spectrum[0] = 0
    x = np.fft.irfft(spectrum, n)
    return x / np.std(x)


def add_noise(clean: np.ndarray, speech_mask: np.ndarray, snr_db: float, seed: int = 1234) -> np.ndarray:
    """Pink noise + 60 Hz hum (with harmonics), scaled so speech power / noise power = snr_db."""
    rng = np.random.default_rng(seed)
    t = np.arange(clean.size) / RATE
    hum = sum(np.sin(2 * np.pi * 60 * k * t) / k for k in (1, 2, 3))
    noise = pink_noise(clean.size, rng) + 0.3 * hum / np.std(hum)
    x = clean.astype(np.float64)
    p_speech = np.mean(x[speech_mask] ** 2)
    noise *= np.sqrt(p_speech / (10 ** (snr_db / 10)) / np.mean(noise ** 2))
    return np.clip(np.round(x + noise), -32768, 32767).astype(np.int16)


def write_wav(path: Path, x: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(x.tobytes())


def build(out: Path, snr_db: float, text_only: bool = False) -> dict:
    """text_only: no audio and no voices (any OS): the script and its turns with times estimated from the word count."""
    out.mkdir(parents=True, exist_ok=True)
    gaps = [0.7, 0.9, 0.6, 0.8]  # seconds of silence between turns, cycled (deterministic)
    pieces = [np.zeros(int(0.5 * RATE), np.int16)]
    pos = pieces[0].size
    turns = []
    for i, (spk, text) in enumerate(SCRIPT):
        name, rate, pitch = VOICES[spk]
        audio = np.zeros(int(0.4 * len(text.split()) * RATE), np.int16) if text_only else trim(speak(text, name, rate, pitch))
        turns.append({"speaker": spk, "role": ROLES[spk], "start": round(pos / RATE, 3),
                      "end": round((pos + audio.size) / RATE, 3), "text": text})
        gap = np.zeros(int(gaps[i % len(gaps)] * RATE), np.int16)
        pieces += [audio, gap]
        pos += audio.size + gap.size
    clean = np.concatenate(pieces)
    mask = np.zeros(clean.size, bool)
    for t in turns:
        mask[int(t["start"] * RATE): int(t["end"] * RATE)] = True
    noisy = add_noise(clean, mask, snr_db)
    if not text_only:
        write_wav(out / "lecture_clean.wav", clean)
        write_wav(out / "lecture_noisy.wav", noisy)

    speech = {}
    for t in turns:
        speech[t["speaker"]] = speech.get(t["speaker"], 0) + t["end"] - t["start"]
    truth = {
        "generator": "scripts/dev/make_synthetic_fixture.py (text only)" if text_only else "scripts/dev/make_synthetic_fixture.py (Windows SAPI)",
        "sample_rate": RATE,
        "duration_s": round(clean.size / RATE, 3),
        "known_phrase": KNOWN_PHRASE,
        "noisy_snr_db": snr_db,
        "reference_text": " ".join(text for _, text in SCRIPT),
        "speech_s": {k: round(v, 3) for k, v in speech.items()},
        "main_speaker": "main",
        "turns": turns,
    }
    (out / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "synthetic")
    ap.add_argument("--snr-db", type=float, default=10.0)
    ap.add_argument("--text-only", action="store_true", help="write only truth.json (no audio, no voices; works on any OS)")
    args = ap.parse_args(argv)
    if sys.platform != "win32" and not args.text_only:
        raise SystemExit("The synthetic audio uses Windows SAPI voices (use --text-only for the script alone).")
    truth = build(args.out, args.snr_db, args.text_only)
    share = truth["speech_s"]["main"] / sum(truth["speech_s"].values())
    print(f"wrote {args.out}: {truth['duration_s']:.1f} s, {len(truth['turns'])} turns, "
          f"main speaker {share:.0%} of speech, noisy copy at {args.snr_db:g} dB SNR")
    return 0


if __name__ == "__main__":
    sys.exit(main())
