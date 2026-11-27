"""Measure GPU memory and speed of the summary model at candidate context sizes.

Usage: python scripts/bench/measure_llm_context.py [model] [num_ctx ...]
Fills each context size to ~85% with filler text, sets num_ctx explicitly, and reports
prompt tokens evaluated (vs sent), peak VRAM used (nvidia-smi), GPU/CPU split, and speed.
"""
import subprocess
import sys
import threading
import time

import lecture_recorder  # noqa: F401  (network lockdown)
import ollama

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3:8b"
SIZES = [int(s) for s in sys.argv[2:]] or [4096, 8192, 12288, 16384]
HOST = "http://127.0.0.1:11434"
SENTENCE = "The professor explained how enzymes lower activation energy in cellular reactions. "


def vram_used_mb() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(out.strip().splitlines()[0])


def main() -> None:
    client = ollama.Client(host=HOST)
    client.generate(model=MODEL, prompt="", keep_alive=0)  # unload so the baseline is clean
    time.sleep(2)
    baseline = vram_used_mb()
    per_sentence = client.generate(model=MODEL, prompt=SENTENCE * 20, options={"num_ctx": 4096, "num_predict": 1}, keep_alive=0).prompt_eval_count / 20
    print(f"model={MODEL} baseline_vram={baseline} MB total_gpu=8192 MB tokens/sentence~{per_sentence:.1f}")
    print("num_ctx | prompt_sent_est | prompt_evaluated | peak_vram_MB | model_on_gpu | eval tok/s")
    for n in SIZES:
        reps = int(n * 0.85 / per_sentence)
        prompt = SENTENCE * reps + "\nSummarize the text above in one sentence."
        peak = [0]
        stop = threading.Event()

        def poll():
            while not stop.is_set():
                peak[0] = max(peak[0], vram_used_mb())
                time.sleep(0.2)

        t = threading.Thread(target=poll)
        t.start()
        r = client.generate(model=MODEL, prompt=prompt, think=False,
                            options={"num_ctx": n, "num_predict": 64, "temperature": 0}, keep_alive="30s")
        ps = [m for m in client.ps().models if m.model.startswith(MODEL.split(":")[0])]
        stop.set()
        t.join()
        gpu_share = f"{ps[0].size_vram / ps[0].size:.0%}" if ps else "?"
        tps = r.eval_count / (r.eval_duration / 1e9) if r.eval_duration else 0
        print(f"{n:7d} | {int(reps * per_sentence):15d} | {r.prompt_eval_count:16d} | {peak[0] - baseline:12d} | {gpu_share:>12} | {tps:9.1f}")
        client.generate(model=MODEL, prompt="", keep_alive=0)
        time.sleep(2)


if __name__ == "__main__":
    main()
