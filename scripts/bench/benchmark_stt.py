"""Benchmark: large-v3-turbo vs distil-large-v3.5 on the clean and noisy fixtures.

Reports WER (normalized text, jiwer), load and transcription time, real-time factor, and peak
RAM/VRAM. Each run is a separate process, so peaks are not mixed between models. Before running,
records `ollama ps` (must show no loaded model) and nvidia-smi free VRAM. Results are saved as
plain files in data/benchmarks/.

Usage:
  python scripts/bench/benchmark_stt.py                         synthetic fixture
  python scripts/bench/benchmark_stt.py --audio A.wav --reference ref.txt --label real

Synthetic results are for speed and memory only; accuracy is compared on real recordings.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ["large-v3-turbo", "distil-large-v3.5"]
SYNTHETIC_NOTE = "speed and memory only; accuracy is compared on real recordings"


# ------------------------------------------------------------------ one run (child process)
def run_one(model: str, audio_path: Path, reference: str, known_phrase: str | None) -> dict:
    import lecture_recorder  # noqa: F401  (offline lockdown)
    import jiwer
    import psutil

    from lecture_recorder.config import load_config
    from lecture_recorder.pipeline.audio import load_audio
    from lecture_recorder.pipeline.gpu import VramPeak, gpu_job, nvidia_smi_memory
    from lecture_recorder.pipeline.textnorm import normalize
    from lecture_recorder.pipeline.transcribe import choose, transcribe

    cfg = load_config()
    audio = load_audio(audio_path)
    with gpu_job(cfg.home):
        base = nvidia_smi_memory()  # before anything creates a CUDA context in this process
        choice = choose(cfg.models_dir, model=model, device="auto")
        with VramPeak() as peak:
            out = transcribe(audio, choice, cfg.transcription.language, cfg.transcription.beam_size)
    hyp = " ".join(s["text"] for s in out["segments"])
    dur = out["duration_s"]
    t = out["timings"]
    return {
        "model": model,
        "device": choice.device,
        "compute_type": choice.compute_type,
        "audio": audio_path.name,
        "duration_s": dur,
        "wer": round(jiwer.wer(normalize(reference), normalize(hyp)), 4),
        "dropped_speech_s": out["dropped_speech"]["total_s"],
        "known_phrase_found": (normalize(known_phrase) in normalize(hyp)) if known_phrase else None,
        "load_s": t["load_s"],
        "transcribe_s": t["transcribe_s"],
        "rtf": round(t["transcribe_s"] / dur, 4),
        "peak_ram_mb": round(psutil.Process().memory_info().peak_wset / 2**20) if hasattr(
            psutil.Process().memory_info(), "peak_wset") else None,
        "peak_vram_mb": (peak.peak_mb - base["used_mb"]) if base and peak.peak_mb else None,
    }


# ------------------------------------------------------------------ parent
def preflight() -> dict:
    info = {"time": datetime.now().astimezone().isoformat(timespec="seconds")}
    exe = shutil.which("ollama")
    if exe:
        r = subprocess.run([exe, "ps"], capture_output=True, text=True, timeout=30)
        info["ollama_ps"] = r.stdout.strip() or r.stderr.strip()
        info["ollama_models_loaded"] = max(0, len([l for l in r.stdout.splitlines() if l.strip()]) - 1)
    else:
        info["ollama_ps"] = "ollama not installed"
        info["ollama_models_loaded"] = 0
    from lecture_recorder.pipeline.gpu import nvidia_smi_memory

    info["nvidia_smi"] = nvidia_smi_memory()
    return info


def table(rows: list[dict], label_note: str | None) -> str:
    head = ("| Model | Audio | WER | Dropped speech (s) | Known phrase | Load (s) | Transcribe (s) | RTF | Peak RAM (MB) | Peak VRAM (MB) |\n"
            "|---|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['model']} | {r['audio']} | error: {r['error']} |  |  |  |  |  |  |  |")
            continue
        kp = {True: "yes", False: "no", None: "-"}[r["known_phrase_found"]]
        lines.append(f"| {r['model']} ({r['compute_type']}, {r['device']}) | {r['audio']} | {r['wer']:.1%} | {r['dropped_speech_s']:.1f} | {kp} | "
                     f"{r['load_s']:.1f} | {r['transcribe_s']:.1f} | {r['rtf']:.3f} | {r['peak_ram_mb']} | {r['peak_vram_mb']} |")
    if label_note:
        lines.append(f"\n*Synthetic results: {label_note}.*")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Speech-to-text benchmark")
    ap.add_argument("--audio", type=Path, nargs="*", help="audio files (default: synthetic clean + noisy)")
    ap.add_argument("--reference", type=Path, help="reference transcript (default: synthetic truth.json)")
    ap.add_argument("--label", default="synthetic")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--allow-loaded-ollama", action="store_true", help="run even if Ollama has a model in VRAM")
    ap.add_argument("--one", nargs=2, metavar=("MODEL", "AUDIO"), help=argparse.SUPPRESS)  # child mode
    ap.add_argument("--ref-json", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.one:
        ref = json.loads(args.ref_json.read_text(encoding="utf-8"))
        print("RESULT " + json.dumps(run_one(args.one[0], Path(args.one[1]), ref["text"], ref.get("known_phrase"))))
        return 0

    synth = ROOT / "tests" / "fixtures" / "synthetic"
    if not args.audio:
        if not (synth / "truth.json").exists():
            subprocess.run([sys.executable, str(ROOT / "scripts" / "dev" / "make_synthetic_fixture.py")], check=True)
        args.audio = [synth / "lecture_clean.wav", synth / "lecture_noisy.wav"]
        truth = json.loads((synth / "truth.json").read_text(encoding="utf-8"))
        ref = {"text": truth["reference_text"], "known_phrase": truth["known_phrase"]}
    else:
        if not args.reference:
            ap.error("--reference is required with --audio")
        ref = {"text": args.reference.read_text(encoding="utf-8"), "known_phrase": None}

    sys.path.insert(0, str(ROOT / "src"))
    import lecture_recorder  # noqa: F401
    from lecture_recorder.config import load_config

    out_dir = load_config().data_dir / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ref_json = out_dir / f".ref-{stamp}.json"
    ref_json.write_text(json.dumps(ref), encoding="utf-8")

    pre = preflight()
    print("Before benchmarking:")
    print("  ollama ps:\n    " + pre["ollama_ps"].replace("\n", "\n    "))
    smi = pre["nvidia_smi"]
    print(f"  nvidia-smi: {smi['free_mb']} MB free of {smi['total_mb']} MB" if smi else "  nvidia-smi: not available")
    if pre["ollama_models_loaded"] and not args.allow_loaded_ollama:
        print("Ollama has a model loaded in memory; stop it (`ollama stop <model>`) for a fair measurement.")
        return 2

    rows = []
    try:
        for model in args.models:
            for audio in args.audio:
                t0 = time.perf_counter()
                r = subprocess.run([sys.executable, __file__, "--one", model, str(audio), "--ref-json", str(ref_json)],
                                   capture_output=True, text=True, timeout=3600)
                line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
                if line:
                    rows.append(json.loads(line[7:]))
                else:
                    rows.append({"model": model, "audio": audio.name, "error": (r.stderr.strip().splitlines() or ["?"])[-1]})
                print(f"  {model} on {audio.name}: {time.perf_counter() - t0:.1f} s wall")
    finally:
        ref_json.unlink(missing_ok=True)

    note = SYNTHETIC_NOTE if args.label == "synthetic" else None
    md = table(rows, note)
    print()
    print(md)
    result = {"label": args.label, "note": note, "preflight": pre, "results": rows}
    (out_dir / f"stt-{args.label}-{stamp}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / f"stt-{args.label}-{stamp}.md").write_text(
        f"# STT benchmark ({args.label}, {pre['time']})\n\nollama ps:\n```\n{pre['ollama_ps']}\n```\n"
        f"nvidia-smi free VRAM: {smi['free_mb'] if smi else 'n/a'} MB of {smi['total_mb'] if smi else 'n/a'} MB\n\n{md}\n",
        encoding="utf-8")
    print(f"\nSaved {out_dir / f'stt-{args.label}-{stamp}.md'}")
    return 0 if all("error" not in r for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
