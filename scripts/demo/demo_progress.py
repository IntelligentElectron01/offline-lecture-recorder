"""Synthetic processing-progress demo: a class with lectures in every processing state, and a fake
"running" job whose progress file moves through the real stages. No models, no GPU, no audio.

    .venv\\Scripts\\python.exe scripts\\demo\\demo_progress.py --data-dir .cache\\progress-demo [--waiting] [--loop]

Then start the app on that folder (launch.json "progress-demo"). The script holds the queue's worker
lock, so the app neither recovers nor runs the demo's jobs. --waiting also holds the recording lock, so the
queued jobs show "Waiting · while recording". Never point it at the real data folder.
"""
import argparse
import sys
import time
from pathlib import Path

import lecture_recorder  # noqa: F401  (offline lockdown)
from lecture_recorder.pipeline import queue as q
from lecture_recorder.pipeline.progress import Progress
from lecture_recorder.recorder.platform import ProcessLock
from lecture_recorder.recorder.process import GLOBAL_LOCK
from lecture_recorder.recorder.session import now_iso, write_json_atomic

SLUG = "demo-bio"
LECTURES = {  # name: title
    "2026-10-28_09-00-00": "Cell membranes",
    "2026-10-29_09-00-00": "Osmosis and diffusion",
    "2026-10-30_09-00-00": "Enzymes",
    "2026-10-31_09-00-00": "Photosynthesis",
}
RUNNING, QUEUED, SUMMARY_QUEUED, FAILED = LECTURES
AUDIO_S = 3600.0
SEG_S = 6


def transcript(n: int = 40) -> dict:
    segs = [{"id": i, "start": i * SEG_S, "end": i * SEG_S + SEG_S - 1, "speaker": "Speaker 1", "role": "Main",
             "text": f"Synthetic demo sentence number {i + 1} about cells.", "words": []} for i in range(n)]
    return {"speakers": [{"id": "Speaker 1", "role": "Main", "speech_s": n * SEG_S}], "segments": segs,
            "duration_s": n * SEG_S}


def seed(data: Path) -> dict[str, Path]:
    """(Re)set the demo: its lectures, and no jobs. Only the demo's own files are touched."""
    c = data / "classes" / SLUG
    c.mkdir(parents=True, exist_ok=True)
    write_json_atomic(c / "class.json", {"name": "Biology (demo)", "slug": SLUG})
    for p in q._queue_dir(data).glob("*.json"):
        p.unlink()
    dirs = {}
    for name, title in LECTURES.items():
        d = c / "sessions" / name
        d.mkdir(parents=True, exist_ok=True)
        for f in ("transcript.json", "transcript.raw.json", "summary.md"):
            (d / f).unlink(missing_ok=True)
        write_json_atomic(d / "meta.json", {"name": name, "title": title, "class": SLUG, "created": name[:10] + "T09:00:00+00:00",
                                            "duration_s": AUDIO_S})
        dirs[name] = d
    doc = transcript()
    write_json_atomic(dirs[SUMMARY_QUEUED] / "transcript.json", doc)
    write_json_atomic(dirs[SUMMARY_QUEUED] / "transcript.raw.json", doc)
    return dirs


def set_job(job: dict, **kw) -> dict:
    job.update(kw)
    write_json_atomic(job["_path"], {k: v for k, v in job.items() if k != "_path"})
    return job


def run_steps(prog: Progress, stage: str, total, unit: str | None, sub: str | None, seconds: float) -> None:
    """Report `total` units evenly over `seconds` (updates every 0.25 s; the writes are throttled to 1.5 s)."""
    prog.stage(stage, total=total, unit=unit, sub=sub)
    if not total:
        time.sleep(seconds)
        return
    t0 = time.monotonic()
    while (el := time.monotonic() - t0) < seconds:
        prog.update(round(total * el / seconds, 1) if unit == "s" else int(total * el / seconds))
        time.sleep(0.25)


def play(data: Path, dirs: dict[str, Path], speed: float, finish: str | None = None) -> None:
    q.enqueue(data, dirs[QUEUED])
    q.enqueue(data, dirs[SUMMARY_QUEUED], q.SUMMARIZE)
    failed = q.enqueue(data, dirs[FAILED])
    set_job(failed, state="failed", finished=now_iso(), error="demo: the speech model could not be loaded")

    def job_run(kind, steps, name=RUNNING, ok=True):
        job = set_job(q.enqueue(data, dirs[name], kind), state="running", started=now_iso(), started_ts=time.time())
        prog = Progress(q.progress_path(data, job["id"]))
        prog.path.parent.mkdir(parents=True, exist_ok=True)
        for st in steps:
            run_steps(prog, *st[:4], seconds=st[4] / speed)
        prog.path.unlink(missing_ok=True)
        if not ok:
            return set_job(job, state="failed", finished=now_iso(), error="demo: transcription failed")
        return set_job(job, state="done", finished=now_iso())

    job_run(q.TRANSCRIBE, [("prepare", None, None, None, 3), ("gpu_wait", None, None, None, 2), ("load", None, None, None, 4),
                           ("transcribe", AUDIO_S, "s", None, 45),
                           ("speakers", 120, None, "segmentation", 10), ("speakers", 40, None, "embeddings", 8),
                           ("speakers", None, None, "clustering", 5), ("finish", None, None, None, 3)])
    doc = transcript()
    write_json_atomic(dirs[RUNNING] / "transcript.json", doc)
    write_json_atomic(dirs[RUNNING] / "transcript.raw.json", doc)
    job_run(q.SUMMARIZE, [("gpu_wait", None, None, None, 2), ("map", 7, "part", None, 35), ("combine", None, None, None, 6)])
    (dirs[RUNNING] / "summary.md").write_text("# Summary: Cell membranes\n\n- A synthetic demo summary.\n", encoding="utf-8")
    job_run(q.EMBED, [("gpu_wait", None, None, None, 2), ("embed", 300, "passage", None, 25)])
    if finish:  # run the queued jobs too, so the queue empties ("All caught up", or 2 done + 1 failed)
        job_run(q.TRANSCRIBE, [("load", None, None, None, 2), ("transcribe", AUDIO_S, "s", None, 6)], QUEUED, ok=finish == "ok")
        if finish == "ok":
            for f in ("transcript.json", "transcript.raw.json"):
                write_json_atomic(dirs[QUEUED] / f, doc)
            job_run(q.SUMMARIZE, [("map", 7, "part", None, 4)], QUEUED)
        job_run(q.SUMMARIZE, [("map", 7, "part", None, 6)], SUMMARY_QUEUED)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--waiting", action="store_true", help="hold the recording lock: queued jobs wait")
    ap.add_argument("--loop", action="store_true", help="start over after each run")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--finish", choices=("ok", "failed"), help="also run the queued jobs, so the queue empties: all done, "
                    "or 2 done and 1 failed")
    args = ap.parse_args()
    data = args.data_dir.resolve()
    if ".cache" not in data.parts:
        sys.exit("refusing: the demo folder must be inside .cache (its queue is reset)")
    while True:
        worker = ProcessLock(q._queue_dir(data) / q.WORKER_LOCK)
        recording = ProcessLock(data / GLOBAL_LOCK)
        if not worker.acquire() or (args.waiting and not recording.acquire()):
            sys.exit("the demo folder is in use")
        try:
            play(data, seed(data), args.speed, args.finish)
        finally:
            worker.release()
            if args.waiting:
                recording.release()
        print("demo run finished")
        if not args.loop:
            return 0
        time.sleep(8)


if __name__ == "__main__":
    sys.exit(main())
