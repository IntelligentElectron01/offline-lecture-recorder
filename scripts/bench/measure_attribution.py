"""How often a student's claim is stated as fact in a summary.

Each synthetic lecture (tests/attribution_cases.py) x labeling (correct, all-lecturer) is summarized
`--runs` times with the model unloaded in between (a fresh load, as in real jobs), and every claim is
checked by the LLM judge. Synthetic text only. GPU and local Ollama: main session only.

Usage: python scripts/bench/measure_attribution.py [--runs 5] [--old PATH_TO_OLD_summarize.py]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--old", type=Path, help="compare with the prompts of an older summarize.py")
    args = ap.parse_args()
    import lecture_recorder  # noqa: F401
    import lecture_recorder.summary.summarize as cur
    from attribution_cases import CASES, doc, judge
    from lecture_recorder.config import load_config
    from lecture_recorder.machine import gpu_block_reason
    from lecture_recorder.pipeline.gpu import gpu_job
    from lecture_recorder.summary.llm import LocalLLM

    cfg = load_config()
    if gpu_block_reason(cfg.data_dir):
        print(f"blocked: {gpu_block_reason(cfg.data_dir)}")
        return 3
    if args.old:
        spec = importlib.util.spec_from_file_location("old_summarize", args.old)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        for n in ("MAP_TASK", "CONDENSE_TASK", "GLANCE_TASK", "transcript_lines"):
            setattr(cur, n, getattr(old, n))
    rows = {}
    for case in sorted(CASES):
        for mislabel in (False, True):
            key = f"{case}/{'all-lecturer' if mislabel else 'labeled'}"
            ok_runs, stated, fails = 0, 0, set()
            for _ in range(args.runs):
                llm = LocalLLM.from_config(cfg.llm)
                try:
                    with gpu_job(cfg.home):
                        r = cur.summarize_doc(doc(case, mislabel), llm, title=CASES[case]["title"])
                        text = r["glance"] + "\n" + "\n".join(p["notes"] for p in r["parts"])
                        failed = [k for k, c in CASES[case]["claims"].items() if judge(llm, text, c)]
                        bad = len(failed)
                        fails.update(failed)
                        llm.unload()  # fresh load for the next run
                finally:
                    llm.close()
                stated += bad
                ok_runs += bad == 0
            rows[key] = {"runs_without_any_claim_as_fact": f"{ok_runs}/{args.runs}",
                         "claims_stated_as_fact": f"{stated}/{args.runs * len(CASES[case]['claims'])}",
                         "failed_claims": sorted(fails)}
            print(key, rows[key], flush=True)
    for lab in ("labeled", "all-lecturer"):
        sub = [v for k, v in rows.items() if k.endswith(lab)]
        ok = sum(int(v["runs_without_any_claim_as_fact"].split("/")[0]) for v in sub)
        print(f"TOTAL {lab}: clean runs {ok}/{args.runs * len(sub)}")
    print(json.dumps(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
