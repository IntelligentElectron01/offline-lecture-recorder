"""Measure the efficiency budgets. Dev tool: needs psutil (dev lock).

  python scripts/bench/measure_resources.py               idle app + recorder + UI polling (no GPU work)
  python scripts/bench/measure_resources.py --jobs        also each GPU job on the synthetic fixture
                                                    (main session only: one GPU job at a time)

Everything runs on throwaway synthetic data in .cache/measure/data/ (recreated each run); the
real data/ folder is never touched. Results: .cache/measure/resources-<time>.json + a table (--out to change).
CPU % is total machine CPU (process CPU time / (wall time x logical CPUs)).
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
WORK = ROOT / ".cache" / "measure"
PORT = 8799
NCPU = psutil.cpu_count() or 1

import lecture_recorder  # noqa: E402,F401  (offline lockdown)
from lecture_recorder.pipeline.gpu import settle_vram, vram_used_mb, VramPeak  # noqa: E402


def ollama_loaded() -> list[str] | None:
    """Models Ollama holds, or None if it isn't running (then it holds none; a job would fail by itself)."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=3) as r:
            return [m["name"] for m in json.load(r)["models"]]
    except OSError:
        return None


def process_vram_mb(pids: list[int]) -> int | None:
    """Dedicated VRAM of these processes (Windows GPU counters; nvidia-smi shows N/A per process under
    WDDM). Device-wide "used" also moves with the desktop and games, so it can't show what the app holds.
    None if the counter can't be read (it is named in the system's language)."""
    ps = (r"(Get-Counter '\GPU Process Memory(*)\Dedicated Usage').CounterSamples | "
          "ForEach-Object { $_.InstanceName + '=' + [int64]$_.CookedValue }")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    found = re.findall(r"^pid_(\d+)_.*=(\d+)$", r.stdout, re.M)
    if r.returncode or not found:
        return None
    return sum(int(b) for pid, b in found if int(pid) in pids) // 2**20


def stage(name: str) -> None:
    """Progress on stderr, so a run that is stopped still shows how far it got."""
    print(f"{datetime.now():%H:%M:%S} {name}", file=sys.stderr, flush=True)


def tree(pid: int) -> list[psutil.Process]:
    p = psutil.Process(pid)
    return [p, *p.children(recursive=True)]


def cpu_seconds(procs: list[psutil.Process]) -> float:
    total = 0.0
    for p in procs:
        try:
            t = p.cpu_times()
            total += t.user + t.system
        except psutil.Error:
            pass
    return total


def sample_cpu(pid: int, seconds: float, during=None) -> dict:
    """Average CPU of the whole process tree over `seconds` (total machine % and % of one core), plus
    peak RSS (MB). The tree is re-listed every second, so helpers started during the window count, and
    a process that exits keeps the CPU time it was last seen with."""
    start: dict[int, float] = {}
    last: dict[int, float] = {}

    def poll(first: bool = False) -> int:
        rss = 0
        for p in tree(pid):
            try:
                t = p.cpu_times()
                used = t.user + t.system
                rss += p.memory_info().rss
            except psutil.Error:
                continue
            start.setdefault(p.pid, used if first else 0.0)  # started during the window: count from 0
            last[p.pid] = used
        return rss // 2**20

    rss = poll(first=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        if during:
            during()
        time.sleep(1)
        rss = max(rss, poll())
    wall = time.monotonic() - t0
    used = sum(last[k] - start[k] for k in last)
    return {"cpu_pct_total": round(100 * used / (wall * NCPU), 3), "cpu_pct_one_core": round(100 * used / wall, 2),
            "rss_mb": rss, "seconds": round(wall)}


def fresh_data() -> Path:
    from test_search import make_corpus

    data = WORK / "data"
    if data.exists():
        shutil.rmtree(data)  # our own throwaway folder (results next to it are kept)
    data.mkdir(parents=True)
    make_corpus(data)  # three synthetic transcribed lectures, so the idle app has something to index
    return data


_opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())  # holds the app's sign-in cookie


def get(url: str) -> None:
    with _opener.open(url, timeout=5) as r:
        r.read()


def measure_idle_app(data: Path, seconds: float) -> dict:
    exe = Path(sys.executable).with_name("lecture-recorder.exe")
    cmd = [str(exe) if exe.exists() else sys.executable, *([] if exe.exists() else ["-m", "lecture_recorder.cli"]),
           "serve", "--no-browser", "--port", str(PORT), "--data-dir", str(data)]
    vram0, models0 = vram_used_mb(), ollama_loaded()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        base = f"http://127.0.0.1:{PORT}"
        for _ in range(100):
            try:
                get(base + "/api/health")
                break
            except OSError:
                time.sleep(0.2)
        from lecture_recorder.runtime_env import app_home
        from lecture_recorder.web.auth import LOGIN_PARAM, read_login_key

        get(f"{base}/?{LOGIN_PARAM}={read_login_key(app_home(), PORT)}")  # sign in like the browser the app opens
        get(base + "/api/search?q=membrane")  # build the index once, as a user would
        time.sleep(5)  # settle after start-up
        hidden = sample_cpu(proc.pid, seconds)  # no client at all = UI hidden (it makes no requests)
        last = [0.0]

        def visible_client():  # what an open, visible tab does: the 2 s recording-status poll
            if time.monotonic() - last[0] >= 2:
                get(base + "/api/record/status")
                last[0] = time.monotonic()

        visible = sample_cpu(proc.pid, min(seconds, 30), during=visible_client)
        return {"hidden": hidden, "visible": visible, "vram_before_mb": vram0, "vram_idle_mb": vram_used_mb(),
                "app_vram_mb": process_vram_mb([p.pid for p in tree(proc.pid)]),
                "ollama_before": models0, "ollama_idle": ollama_loaded()}
    finally:
        proc.terminate()
        proc.wait(10)


def gpu_pids() -> set[int]:
    """Processes nvidia-smi lists as using the GPU (on Windows/WDDM this includes graphics users)."""
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=30)
    return {int(x) for x in r.stdout.split() if x.strip().isdigit()}


def measure_recorder(data: Path, seconds: float, app: dict | None = None) -> dict:
    """The recorder process with the synthetic source, or with `app` (record from an app)."""
    from lecture_recorder.config import load_config
    from lecture_recorder.recorder.control import start_recording
    import dataclasses

    cfg = dataclasses.replace(load_config(), data_dir=data)
    vram0 = vram_used_mb()
    # Synthetic tone source: the recorder loop, PCM writes, fsyncs and status are the same as with a mic
    # (no live microphone recording is made by this script).
    from lecture_recorder.library import create_class, get_class

    if not (get_class(data, "biology-101") or {}).get("has_class_file"):
        create_class(data, "Biology 101")
    h = start_recording("biology-101", config=cfg, app=app, synthetic=None if app else {"pattern": "tone"})
    try:
        time.sleep(2)
        rec = sample_cpu(h.proc.pid, seconds)
        h.pause()
        time.sleep(1)
        paused = sample_cpu(h.proc.pid, min(seconds, 20))
        pids = {p.pid for p in tree(h.proc.pid)}
        rec.update(vram_before_mb=vram0, vram_during_mb=vram_used_mb(), paused=paused,
                   gpu_processes=sorted(pids & gpu_pids()))  # device-wide VRAM also moves with other programs
        return rec
    finally:
        h.stop()


def measure_app_recorder(data: Path, seconds: float) -> dict | None:
    """Record from an app: this suite's own quiet tone player (tests/audio_tone_player.py), never
    another program. The system volume is not touched. None where process loopback isn't available."""
    from lecture_recorder.recorder.winaudio import supported

    if supported():
        return None
    player = subprocess.Popen([sys.executable, str(ROOT / "tests" / "audio_tone_player.py"), "440", "0.02",
                               str(seconds * 2 + 40)], stdout=subprocess.PIPE, text=True, cwd=ROOT)
    try:
        assert player.stdout.readline().strip() == "PLAYING"
        return measure_recorder(data, seconds, app={"pid": player.pid, "exe": "python.exe", "name": "Tone player"})
    finally:
        player.kill()


def measure_polling() -> dict:
    """Same checks as tests/test_efficiency.py (one helper): poll intervals, and no timers outside the
    visibility-aware poller. Hidden-tab behaviour was also checked live in the browser pane."""
    from test_efficiency import polling_facts

    return polling_facts()


def measure_unit_suggestions(data: Path, lectures: int = 30, windows: int = 180, dims: int = 1024) -> dict:
    """CPU-only boundary suggestion on a semester-sized class: 30 lectures x 180 stored windows (90 min each)
    of synthetic vectors in 6 topics. No model is loaded."""
    import hashlib
    import io

    import numpy as np

    from lecture_recorder.config import load_config
    from lecture_recorder.recorder.session import write_json_atomic
    from lecture_recorder.search.embed import EMB_META, EMB_NPY, text_hash
    from lecture_recorder.search.units_suggest import suggest_boundaries

    import dataclasses

    cfg = dataclasses.replace(load_config(), data_dir=data)
    c = data / "classes" / "semester"
    rng = np.random.default_rng(7)
    topics = rng.normal(size=(6, dims)).astype(np.float32)
    for i in range(lectures):
        d = c / "sessions" / f"2026-{9 + i // 28:02d}-{1 + i % 28:02d}_09-00-00"
        d.mkdir(parents=True)
        # realistic size: ~1,200 segments of ~14 words with word timestamps (about 1 MB per 90-min lecture)
        doc = {"segments": [{"start": 4.5 * k, "end": 4.5 * k + 4.4, "text": " ".join(f"word{k}x{j}" for j in range(14)),
                             "words": [{"w": f" word{k}x{j}", "start": 4.5 * k + 0.3 * j, "end": 4.5 * k + 0.3 * j + 0.28,
                                        "p": 0.9} for j in range(14)]} for k in range(1200)]}
        write_json_atomic(d / "transcript.json", doc)
        m = topics[i * 6 // lectures] + 0.8 * rng.normal(size=(windows, dims)).astype(np.float32)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        buf = io.BytesIO()
        np.save(buf, m.astype(np.float32))
        (d / EMB_NPY).write_bytes(buf.getvalue())
        write_json_atomic(d / EMB_META, {
            "state": "done", "model": cfg.llm.embedding_model, "window_s": cfg.search.window_s,
            "step_s": cfg.search.step_s, "text_hash": text_hash(doc), "dims": dims, "count": windows,
            "npy_sha256": hashlib.sha256(buf.getvalue()).hexdigest(),
            "windows": [{"start": 30.0 * k, "end": 30.0 * k + 45, "text": "word " * 140} for k in range(windows)]})
    me = psutil.Process()
    c0, t0 = sum(me.cpu_times()[:2]), time.monotonic()
    suggest_boundaries(c, cfg)  # first use: also indexes the 30 lectures (as the app's first search would)
    cold_s, cold_cpu = round(time.monotonic() - t0, 2), round(sum(me.cpu_times()[:2]) - c0, 2)
    c0, t0 = sum(me.cpu_times()[:2]), time.monotonic()
    out = suggest_boundaries(c, cfg)  # the usual case: index already current
    return {"lectures": lectures, "windows": lectures * windows, "cold_seconds": cold_s, "cold_cpu_s": cold_cpu,
            "seconds": round(time.monotonic() - t0, 2), "cpu_s": round(sum(me.cpu_times()[:2]) - c0, 2),
            "suggestions": len(out["suggestions"]),
            "vram_mb": process_vram_mb([me.pid])}


def run_sampled(cmd: list[str]) -> tuple[int, float, str]:
    """Run a job process to its end, sampling its CPU time every 0.5 s. Returns (exit code, CPU seconds,
    the end of its output). Output goes to a temp file, not a pipe: a pipe nobody reads while polling
    fills up (~4 KB; a failed job's traceback is bigger) and the job then blocks forever at exit."""
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True,
                             creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
        ps = psutil.Process(p.pid)
        cpu = 0.0
        while p.poll() is None:
            try:
                cpu = max(cpu, cpu_seconds([ps]))
            except psutil.Error:  # ended between poll() and sampling
                pass
            time.sleep(0.5)
        log.seek(0)
        return p.returncode, cpu, log.read()[-300:]


def measure_jobs(data: Path, audio: Path | None = None) -> list[dict]:
    """Each GPU job in its own process on the synthetic fixture, or on `audio` (e.g. a real recording;
    copied into the throwaway data folder, never moved). The job's last output line has no transcript text."""
    from lecture_recorder.machine import gpu_block_reason

    if audio is not None and not audio.is_file():
        raise SystemExit(f"--audio {audio} is not a file")
    fixture = audio or ROOT / "tests" / "fixtures" / "synthetic" / "lecture_clean.wav"
    if not fixture.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "dev" / "make_synthetic_fixture.py")], check=True)
    d = data / "classes" / "biology-101" / "sessions" / "2026-09-09_09-00-00"
    d.mkdir(parents=True)
    shutil.copy(fixture, d / f"audio{fixture.suffix.lower()}")
    (d / "meta.json").write_text(json.dumps({"name": d.name, "class": "biology-101"}), encoding="utf-8")
    out = []
    for kind in ("transcribe", "summarize", "embed"):
        reason = gpu_block_reason(data)
        if reason:
            raise SystemExit(f"cannot run GPU jobs now: {reason}")
        base = vram_used_mb()
        t0 = time.monotonic()
        stage(f"job {kind}")
        with VramPeak() as peak:
            code, cpu, tail = run_sampled([sys.executable, "-m", "lecture_recorder.pipeline", str(d),
                                           "--data-dir", str(data), "--task", kind])
        after = settle_vram(base, wait_s=20, slack_mb=150)
        out.append({"job": kind, "ok": code == 0, "seconds": round(time.monotonic() - t0, 1),
                    "job_process_cpu_s": round(cpu, 1),  # excludes Ollama's server (summary/embed work)
                    "vram_before_mb": base, "vram_peak_mb": peak.peak_mb,
                    "vram_after_mb": after, "vram_delta_mb": (after - base) if None not in (after, base) else None,
                    "ollama_after": ollama_loaded(), "tail": tail.strip().splitlines()[-1:] if code else []})
    if audio is not None:  # private audio: don't leave its copy, transcript or summary in the work folder
        shutil.rmtree(d)
    return out


def table(r: dict) -> str:
    rows = []
    ia, rc, pol = r["idle_app"], r["recorder"], r["polling"]
    rows.append(("Idle app, UI hidden: avg total CPU (60 s)", f"{ia['hidden']['cpu_pct_total']}%", "< 1%",
                 ia["hidden"]["cpu_pct_total"] < 1))
    rows.append(("Idle app, UI hidden: % of one core", f"{ia['hidden']['cpu_pct_one_core']}%", "< 0.5%",
                 ia["hidden"]["cpu_pct_one_core"] < 0.5))
    rows.append(("Idle app, UI visible (2 s poll): avg total CPU", f"{ia['visible']['cpu_pct_total']}%", "(info)", None))
    rows.append(("Idle app RSS", f"{ia['hidden']['rss_mb']} MB", "(info)", None))
    rows.append(("Idle: Ollama models loaded", "Ollama not running" if ia["ollama_idle"] is None else ", ".join(ia["ollama_idle"]) or "none",
                 "none", not ia["ollama_idle"]))
    dv = (ia["vram_idle_mb"] - ia["vram_before_mb"]) if None not in (ia["vram_idle_mb"], ia["vram_before_mb"]) else None
    av = ia.get("app_vram_mb")
    rows.append(("Idle: app VRAM (its own processes)", f"{av} MB" if av is not None else "n/a", "0 (+/-50 noise)",
                 abs(av) <= 50 if av is not None else None))
    rows.append(("Idle: device VRAM change (other programs move it too)", f"{dv:+} MB" if dv is not None else "n/a",
                 "(info)", None))
    rows.append(("Recorder: avg total CPU (60 s)", f"{rc['cpu_pct_total']}%", "< 2%", rc["cpu_pct_total"] < 2))
    rows.append(("Recorder: % of one core", f"{rc['cpu_pct_one_core']}%", "< 5%", rc["cpu_pct_one_core"] < 5))
    rows.append(("Recorder paused: avg total CPU", f"{rc['paused']['cpu_pct_total']}%", "(info)", None))
    ar = r.get("recorder_app")
    if ar:
        rows.append(("Recorder from an app: avg total CPU", f"{ar['cpu_pct_total']}%", "< 2%", ar["cpu_pct_total"] < 2))
        rows.append(("Recorder from an app: % of one core", f"{ar['cpu_pct_one_core']}%", "< 5%", ar["cpu_pct_one_core"] < 5))
        rows.append(("Recorder from an app, paused: % of one core", f"{ar['paused']['cpu_pct_one_core']}%", "(info)", None))
        rows.append(("Recorder from an app: GPU use", ", ".join(map(str, ar["gpu_processes"])) or "none", "none",
                     not ar["gpu_processes"]))
    gv = (rc["vram_during_mb"] - rc["vram_before_mb"]) if None not in (rc["vram_during_mb"], rc["vram_before_mb"]) else None
    rows.append(("Recorder GPU use (its processes in nvidia-smi's GPU list)", ", ".join(map(str, rc["gpu_processes"])) or "none",
                 "none", not rc["gpu_processes"]))
    rows.append(("Recorder: device VRAM change (other programs move it too)", f"{gv:+} MB" if gv is not None else "n/a",
                 "(info)", None))
    rows.append(("UI polling: fastest interval while visible", f"{pol['min_interval_ms'] / 1000:g} s", ">= 2 s",
                 (pol["min_interval_ms"] or 0) >= 2000 and pol["only_visibility_aware_timers"]))
    rows.append(("UI polling while hidden", "none" if pol["stops_when_hidden"] else "continues", "none",
                 pol["stops_when_hidden"]))
    us = r["unit_suggestions"]
    dv = us["vram_mb"]
    rows.append((f"Unit suggestions, {us['lectures']} lectures ({us['windows']} windows), CPU only",
                 f"{us['seconds']} s ({us['cpu_s']} CPU-s); first use incl. indexing {us['cold_seconds']} s; "
                 + (f"own VRAM {dv} MB" if dv is not None else ""),
                 "< 5 s, no GPU", us["seconds"] < 5 and us["cold_seconds"] < 30 and (dv is None or abs(dv) <= 50)))
    for j in r.get("jobs", []):
        rows.append((f"VRAM after {j['job']} job vs before", f"{j['vram_delta_mb']:+} MB (peak {j['vram_peak_mb']} MB, "
                     f"{j['seconds']} s)", "<= +150 MB", j["ok"] and j["vram_delta_mb"] is not None and j["vram_delta_mb"] <= 150
                     and not j["ollama_after"]))
    lines = ["| Budget | Measured | Limit | Pass |", "|---|---|---|---|"]
    lines += [f"| {a} | {b} | {c} | {'-' if ok is None else ('yes' if ok else '**NO**')} |" for a, b, c, ok in rows]
    return "\n".join(lines), all(ok is not False for *_, ok in rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", action="store_true", help="also run each GPU job on the synthetic fixture")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--audio", type=Path, help="run the jobs on this audio file instead of the synthetic fixture")
    ap.add_argument("--out", type=Path, default=WORK, help="folder for the results JSON (default .cache/measure; never data/)")
    args = ap.parse_args()
    data = fresh_data()
    r = {"when": datetime.now().astimezone().isoformat(timespec="seconds"), "logical_cpus": NCPU}
    for key, step in (("polling", measure_polling), ("idle_app", lambda: measure_idle_app(data, args.seconds)),
                      ("recorder", lambda: measure_recorder(data, args.seconds)),
                      ("recorder_app", lambda: measure_app_recorder(data, args.seconds)),
                      ("unit_suggestions", lambda: measure_unit_suggestions(data))):
        stage(key)
        r[key] = step()
    if args.jobs:
        r["jobs"] = measure_jobs(data, args.audio)
        r["jobs_audio"] = args.audio.suffix if args.audio else "synthetic"
    md, ok = table(r)
    out = args.out.resolve() / f"resources-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(md)
    print(f"\n{'All budgets met.' if ok else 'BUDGET EXCEEDED.'} Details: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
