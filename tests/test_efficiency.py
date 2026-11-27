"""Efficiency rules (project rules: efficiency) that can be checked without measuring: polling cadence,
job priority, queue-file growth, and which work runs under the GPU lock. The measured budgets come
from scripts/bench/measure_resources.py (the efficiency budgets)."""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import lecture_recorder  # noqa: F401
from lecture_recorder.pipeline import queue as q
from lecture_recorder.recorder import process

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "src" / "lecture_recorder" / "web" / "static" / "app.js").read_text(encoding="utf-8")


def polling_facts(js: str = JS) -> dict:
    """UI polling as written in app.js (also used by scripts/bench/measure_resources.py)."""
    polls = [int(x) for x in re.findall(r"startPoll\((?:[^()]|\([^()]*\))*?,\s*(\d+)\)", js, flags=re.S)]
    only_visibility_aware = (len(re.findall(r"\bsetInterval\(", js)) == 2  # startPoll + its visibility handler
                             and "setTimeout(" not in js.replace("setTimeout(() => (t.hidden = true)", "")
                             and bool(re.search(r'addEventListener\("visibilitychange"', js))
                             and "document.hidden ? null" in js)
    return {"poll_intervals_ms": sorted(set(polls)), "min_interval_ms": min(polls) if polls else None,
            "stops_when_hidden": only_visibility_aware, "only_visibility_aware_timers": only_visibility_aware}


def test_ui_polls_every_2_s_or_slower_and_never_while_hidden():
    f = polling_facts()
    assert f["min_interval_ms"] and f["min_interval_ms"] >= 2000, f
    assert f["only_visibility_aware_timers"], f


def test_measure_script_never_blocks_on_a_jobs_long_output():
    """A failed summarize job's ~5.6 KB traceback filled an unread pipe; the job hung at exit for 20 min."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "bench"))
    from measure_resources import run_sampled

    t0 = time.monotonic()
    code, cpu, tail = run_sampled([sys.executable, "-c", "import sys; sys.stderr.write('x' * 200_000 + 'END'); sys.exit(1)"])
    assert code == 1 and tail.endswith("END") and time.monotonic() - t0 < 30


def test_ollama_down_fails_fast_with_a_clear_message():
    from lecture_recorder.summary.llm import LocalLLM, OllamaUnavailable

    llm = LocalLLM(host="http://127.0.0.1:9", model="m", num_ctx=2048, max_prompt_tokens=1000, num_predict=10)
    t0 = time.monotonic()
    try:
        with pytest.raises(OllamaUnavailable, match="Ollama is not reachable"):
            llm.loaded_models()  # the first call of every summary/embed job
        with pytest.raises(OllamaUnavailable, match="Is it installed and running"):
            llm.release_models(["m"])
    finally:
        llm.close()
    assert time.monotonic() - t0 < 15  # refused at once; connect timeout 5 s


def test_queue_jobs_run_at_below_normal_priority(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw)

        class R:
            returncode, stdout, stderr = 0, "", ""
        return R()

    monkeypatch.setattr(q, "_run_job", fake_run)
    monkeypatch.setattr(q, "gpu_block_reason", lambda d: None)
    d = tmp_path / "s"
    d.mkdir()
    q.enqueue(tmp_path, d)
    q.run_worker(tmp_path)
    if os.name == "nt":
        assert seen["creationflags"] & subprocess.BELOW_NORMAL_PRIORITY_CLASS
    else:  # macOS/Linux: its own session; the job lowers its own priority when it starts (host/)
        assert seen["start_new_session"] is True


def test_finished_jobs_older_than_a_day_are_archived(tmp_path):
    old = (datetime.now().astimezone() - timedelta(days=2)).isoformat(timespec="seconds")
    new = datetime.now().astimezone().isoformat(timespec="seconds")
    qd = tmp_path / "queue"
    qd.mkdir()
    for i, (state, finished) in enumerate([("done", old), ("done", new), ("failed", old)]):
        (qd / f"{i}.json").write_text(json.dumps({"id": str(i), "session": f"s{i}", "state": state, "created": old,
                                                  "finished": finished}), encoding="utf-8")
    q._archive_done(tmp_path)
    assert sorted(j["id"] for j in q.jobs(tmp_path)) == ["1", "2"]  # failed ones stay visible
    assert [p.name for p in (qd / "archive").iterdir()] == ["0.json"]  # kept as a log, not deleted


def test_recorder_wakes_rarely():
    assert process.STATUS_EVERY_S >= 1.0 and process.COMMANDS_EVERY_S >= 0.25
    assert process.PAUSED_READ_TIMEOUT_S >= 0.5


def test_cpu_only_speech_check_runs_after_the_gpu_lock():
    src = (ROOT / "src" / "lecture_recorder" / "pipeline" / "run.py").read_text(encoding="utf-8")
    block = src[src.index("with gpu_job(cfg.home)"):src.index("segments, speakers = merge")]
    assert "check_dropped=False" in block
    assert block.index("dropped_speech_report(") > block.index("after_diarization_freed_mb")
    assert re.search(r"\n    stt\[\"dropped_speech\"\] = dropped_speech_report", block)  # dedented: lock released


def test_status_file_is_not_fsynced_but_meta_is(tmp_path, monkeypatch):
    from lecture_recorder.recorder import session

    calls = []
    monkeypatch.setattr(session.os, "fsync", lambda fd: calls.append(fd))
    session.write_json_atomic(tmp_path / "status.json", {"a": 1}, fsync=False)
    assert calls == []
    session.write_json_atomic(tmp_path / "meta.json", {"a": 1})
    assert len(calls) == 1
    t0 = time.monotonic()
    assert session.read_json(tmp_path / "missing.json", "d", wait_s=5) == "d" and time.monotonic() - t0 < 0.5
