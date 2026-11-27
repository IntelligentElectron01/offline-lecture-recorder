"""python -m lecture_recorder.pipeline <session-or-exam-dir> [--data-dir DIR] [--task transcribe|summarize|embed|exam]: one job.

Used by the queue worker, which runs each job in its own process so all GPU memory is returned
to the system when the job ends, even if a library leaks.
"""
import argparse
import dataclasses
import sys
from pathlib import Path

import lecture_recorder  # noqa: F401  (applies the offline lockdown before any model library loads)
from lecture_recorder.config import load_config
from lecture_recorder.pipeline.run import process_session

EXIT_NO_AUDIO = 4  # the queue marks the job failed; the session shows "No usable audio"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lecture_recorder.pipeline")
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--data-dir", type=Path, default=None, help="override the data folder")
    ap.add_argument("--task", choices=("transcribe", "summarize", "embed", "exam"), default="transcribe")
    args = ap.parse_args(argv)
    from lecture_recorder.host import host

    host().job_child_setup()  # below-normal priority where the parent's spawn flags can't do it (macOS, Linux)
    cfg = load_config()
    if args.data_dir:
        cfg = dataclasses.replace(cfg, data_dir=args.data_dir.resolve())
    from lecture_recorder.machine import gpu_block_reason
    from lecture_recorder.pipeline.queue import EXIT_BLOCKED

    blocked = gpu_block_reason(cfg.data_dir)  # the queue checked too; this closes the gap until now
    if blocked:
        print(blocked)
        return EXIT_BLOCKED
    from lecture_recorder.pipeline.progress import from_env

    progress = from_env()  # the queue worker names the file; no-op otherwise
    if args.task == "summarize":
        from lecture_recorder.summary.session import summarize_session

        info = summarize_session(args.session_dir, cfg, progress=progress)
        print(f"summary {info['state']}" + (f": {info['warning']}" if info.get("warning") else ""))
        return 0
    if args.task == "exam":
        from lecture_recorder.summary.exam import generate_exam_material

        info = generate_exam_material(args.session_dir, cfg, progress=progress)
        print(f"exam material {info['state']}" + (f": {info['warning']}" if info.get("warning") else ""))
        return 0
    if args.task == "embed":
        from lecture_recorder.search.embed import embed_session

        info = embed_session(args.session_dir, cfg, progress=progress)
        print(f"embeddings {info['state']}: {info.get('count', info.get('windows', 0))} window(s)")
        return 0
    from lecture_recorder.pipeline.run import NoUsableAudio

    try:
        doc = process_session(args.session_dir, cfg, progress=progress)
    except NoUsableAudio as e:
        print(e)
        return EXIT_NO_AUDIO
    print(f"done: {len(doc['segments'])} segments, {len(doc['speakers'])} speaker(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
