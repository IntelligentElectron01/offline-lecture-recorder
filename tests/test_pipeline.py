"""Speech pipeline, serial GPU queue, plain-file outputs.

Synthetic audio checks plumbing, speed and memory. Accuracy thresholds here are sanity floors for
TTS speech, not the model choice; that is checked on real recordings.
"""
import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import lecture_recorder  # noqa: F401  (lockdown)
from lecture_recorder.config import load_config
from lecture_recorder.pipeline import merge as M
from lecture_recorder.pipeline.outputs import RAW, TranscriptExists, write_outputs
from lecture_recorder.pipeline.textnorm import normalize

ROOT = Path(__file__).resolve().parents[1]
SESSION = "classes/m2-synthetic/sessions/2026-09-30_10-00-00"


# ------------------------------------------------------------------ fixture
@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "dev" / "make_synthetic_fixture.py"), "--out", str(out)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return out, json.loads((out / "truth.json").read_text(encoding="utf-8"))


def test_fixture_has_main_and_audience_voices_and_known_phrase(synthetic):
    out, truth = synthetic
    speech = truth["speech_s"]
    assert set(speech) == {"main", "audience_a", "audience_b"}
    assert max(speech, key=speech.get) == truth["main_speaker"] == "main"
    assert speech["main"] / sum(speech.values()) > 0.8
    assert truth["known_phrase"] in truth["reference_text"]
    assert all(t["end"] > t["start"] for t in truth["turns"])
    assert all(a["end"] <= b["start"] for a, b in zip(truth["turns"], truth["turns"][1:]))  # no overlap


def test_noisy_copy_is_at_the_stated_snr(synthetic):
    import soundfile as sf

    out, truth = synthetic
    clean, _ = sf.read(out / "lecture_clean.wav", dtype="float64")
    noisy, _ = sf.read(out / "lecture_noisy.wav", dtype="float64")
    mask = np.zeros(clean.size, bool)
    for t in truth["turns"]:
        mask[int(t["start"] * 16000): int(t["end"] * 16000)] = True
    snr = 10 * np.log10(np.mean(clean[mask] ** 2) / np.mean((noisy - clean) ** 2))
    assert abs(snr - truth["noisy_snr_db"]) < 0.5


def test_fixture_is_deterministic_and_git_ignored(synthetic, tmp_path):
    out, _ = synthetic
    subprocess.run([sys.executable, str(ROOT / "scripts" / "dev" / "make_synthetic_fixture.py"), "--out", str(tmp_path)],
                   check=True, capture_output=True, timeout=300)
    for name in ("lecture_clean.wav", "lecture_noisy.wav", "truth.json"):
        assert (tmp_path / name).read_bytes() == (out / name).read_bytes(), name
    for name in ("lecture_clean.wav", "truth.json"):
        assert subprocess.run(["git", "check-ignore", "-q", f"tests/fixtures/synthetic/{name}"], cwd=ROOT).returncode == 0


# ------------------------------------------------------------------ audio import without ffmpeg
def test_m4a_and_mp3_decode_with_bundled_pyav(tmp_path):
    """Imports decode through PyAV's own FFmpeg libraries; no system ffmpeg, nothing downloaded."""
    import av

    from lecture_recorder.pipeline.audio import load_audio

    tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(32000) / 16000)).astype(np.float32)
    for ext, codec in (("m4a", "aac"), ("mp3", "mp3")):
        path = tmp_path / f"clip.{ext}"
        with av.open(str(path), "w") as c:
            st = c.add_stream(codec, rate=16000 if codec == "aac" else 22050)
            st.layout = "mono"
            frame = av.AudioFrame.from_ndarray(tone[None, :], format="flt", layout="mono")
            frame.sample_rate = 16000
            for p in st.encode(frame):
                c.mux(p)
            for p in st.encode(None):
                c.mux(p)
        audio = load_audio(path)
        assert audio.dtype == np.float32 and abs(len(audio) / 16000 - 2.0) < 0.15, ext


# ------------------------------------------------------------------ merge (pure functions)
TURNS = [
    {"start": 0.0, "end": 10.0, "label": "A"},
    {"start": 10.5, "end": 12.0, "label": "B"},
    {"start": 12.5, "end": 30.0, "label": "A"},
    {"start": 31.0, "end": 33.0, "label": "C"},
]


def test_speaker_1_is_most_speech_and_main():
    ranked = M.rank_speakers(TURNS)
    assert [(s["id"], s["label"], s["role"]) for s in ranked] == [
        ("Speaker 1", "A", "Main"), ("Speaker 2", "C", "Audience"), ("Speaker 3", "B", "Audience")]


def test_words_go_to_max_overlap_or_nearest_turn():
    assert M.label_for(10.4, 11.0, TURNS) == "B"
    assert M.label_for(9.0, 10.6, TURNS) == "A"
    assert M.label_for(10.1, 10.2, TURNS) == "A"  # in a gap: nearest turn (A ends 0.15 s away, B starts 0.35 s away)
    assert M.label_for(10.3, 10.4, TURNS) == "B"
    assert M.label_for(30.6, 30.8, TURNS) == "C"
    assert M.label_for(1.0, 2.0, []) is None


def test_segments_split_where_speaker_changes():
    words = [{"w": " Hello", "start": 9.0, "end": 9.5, "p": 1}, {"w": " there.", "start": 9.6, "end": 9.9, "p": 1},
             {"w": " Question?", "start": 10.6, "end": 11.5, "p": 1}, {"w": " Yes.", "start": 12.6, "end": 13.0, "p": 1}]
    segs, speakers = M.merge([{"start": 9.0, "end": 13.0, "text": "", "words": words}], TURNS)
    assert [(s["speaker"], s["role"], s["text"]) for s in segs] == [
        ("Speaker 1", "Main", "Hello there."), ("Speaker 3", "Audience", "Question?"), ("Speaker 1", "Main", "Yes.")]
    assert [s["id"] for s in segs] == [0, 1, 2]


def test_no_diarization_turns_means_everything_is_speaker_1():
    segs, speakers = M.merge([{"start": 0, "end": 1, "text": "hi", "words": []}], [])
    assert segs[0]["speaker"] == "Speaker 1" and speakers[0]["role"] == "Main"


def test_role_accuracy_by_time_overlap():
    truth = [{"start": 0, "end": 10, "role": "main"}, {"start": 10, "end": 12, "role": "audience"}]
    segs = [{"start": 0, "end": 9, "role": "Main"}, {"start": 9, "end": 12, "role": "Audience"}]
    assert M.role_accuracy(segs, truth) == pytest.approx(11 / 12)


def test_role_accuracy_counts_overlapping_segments_once():
    truth = [{"start": 0, "end": 10, "role": "main"}]
    segs = [{"start": 0, "end": 6, "role": "Main"}, {"start": 0, "end": 6, "role": "Main"},
            {"start": 5, "end": 7, "role": "Main"}]
    assert M.role_accuracy(segs, truth) == pytest.approx(0.7)


def test_wer_normalization_treats_numbers_and_percent_as_words():
    assert normalize("600 Kelvin is 50%, or 1 half!") == normalize("six hundred kelvin is fifty percent or one half")


# ------------------------------------------------------------------ outputs
def _doc():
    return {"speakers": [{"id": "Speaker 1", "role": "Main", "speech_s": 60.0}],
            "segments": [{"start": 61.2, "end": 62.0, "speaker": "Speaker 1", "role": "Main", "text": "Hi."}]}


def test_outputs_are_plain_files_and_raw_is_never_overwritten(tmp_path):
    paths = write_outputs(tmp_path, _doc())
    assert {p.name for p in paths} == {"transcript.raw.json", "transcript.json", "transcript.md", "transcript.txt"}
    assert (tmp_path / "transcript.txt").read_text(encoding="utf-8") == "[00:01:01] Speaker 1 (Main): Hi.\n"
    (tmp_path / "transcript.json").write_text('{"edited": true}', encoding="utf-8")
    before = (tmp_path / RAW).read_bytes()
    with pytest.raises(TranscriptExists):
        write_outputs(tmp_path, _doc())
    assert (tmp_path / RAW).read_bytes() == before
    assert json.loads((tmp_path / "transcript.json").read_text()) == {"edited": True}


def test_user_edits_survive_reprocessing_after_raw_removed(tmp_path):
    write_outputs(tmp_path, _doc())
    (tmp_path / "transcript.json").write_text('{"edited": true}', encoding="utf-8")
    (tmp_path / RAW).unlink()
    write_outputs(tmp_path, _doc())
    assert json.loads((tmp_path / "transcript.json").read_text()) == {"edited": True}


def test_refuses_session_still_recording_or_unrecovered(tmp_path):
    from lecture_recorder.pipeline.run import process_session
    from lecture_recorder.recorder.platform import ProcessLock

    cfg = dataclasses.replace(load_config(), data_dir=tmp_path)
    s = tmp_path / SESSION
    s.mkdir(parents=True)
    (s / "audio.pcm").write_bytes(b"\0" * 32)
    lock = ProcessLock(s / "recorder.lock")
    assert lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="still recording"):
            process_session(s, cfg)
    finally:
        lock.release()
    with pytest.raises(FileNotFoundError, match="recover"):
        process_session(s, cfg)


# ------------------------------------------------------------------ serial GPU queue
GRAB_LOCK = """
import sys, lecture_recorder
from lecture_recorder.pipeline.gpu import gpu_job, GpuBusy
try:
    with gpu_job(sys.argv[1] if len(sys.argv) > 1 else None, wait_s=0.5):
        print("GOT")
except GpuBusy:
    print("BUSY")
"""


def test_only_one_gpu_job_at_a_time_across_processes(tmp_path):
    from lecture_recorder.pipeline.gpu import gpu_job

    run = lambda: subprocess.run([sys.executable, "-c", GRAB_LOCK, str(tmp_path)], capture_output=True, text=True,
                                 timeout=120).stdout.strip()
    with gpu_job(tmp_path):
        assert run() == "BUSY"
    assert run() == "GOT"  # released


def test_gpu_lock_is_shared_whatever_the_data_folder():
    """Jobs on a temp data folder (tests) and on data/ (app, benchmark) take the same lock."""
    from lecture_recorder.pipeline.gpu import gpu_job

    with gpu_job():  # default: <app home>/.gpu.lock
        r = subprocess.run([sys.executable, "-c", GRAB_LOCK], cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert r.stdout.strip() == "BUSY"
    src = (ROOT / "src" / "lecture_recorder" / "pipeline" / "run.py").read_text(encoding="utf-8")
    assert "gpu_job(cfg.home)" in src

def test_queue_dedupes_requeues_stale_and_records_failures(tmp_path, monkeypatch):
    from lecture_recorder.pipeline import queue
    from lecture_recorder.pipeline.__main__ import main as job_main
    from lecture_recorder.pipeline.queue import enqueue, jobs, run_worker
    from lecture_recorder.recorder.platform import ProcessLock

    def in_process(cmd, **kw):  # the real job code, in this process: it runs "plugged in" (conftest.py)
        try:
            return subprocess.CompletedProcess(cmd, job_main(cmd[3:]), "", "")
        except Exception as e:
            return subprocess.CompletedProcess(cmd, 1, "", str(e))

    monkeypatch.setattr(queue, "_run_job", in_process)
    s = tmp_path / SESSION
    s.mkdir(parents=True)  # no audio -> the job fails fast, before any model loads
    a = enqueue(tmp_path, s)
    assert enqueue(tmp_path, s)["id"] == a["id"]
    job = jobs(tmp_path)[0]
    job["state"] = "running"  # as if a worker crashed mid-job
    from lecture_recorder.pipeline.queue import _save

    _save(job)
    busy = ProcessLock(tmp_path / "queue" / ".worker.lock")  # another worker is running
    assert busy.acquire()
    try:
        assert run_worker(tmp_path) is None  # a second worker never runs in parallel
    finally:
        busy.release()
    done = run_worker(tmp_path)
    assert [j["state"] for j in done] == ["failed"]
    assert "no audio file" in done[0]["error"]
    assert enqueue(tmp_path, s)["id"] != a["id"]  # finished jobs can be queued again


def test_job_that_finished_before_a_worker_crash_is_marked_done(tmp_path, monkeypatch):
    from lecture_recorder.pipeline import queue
    from lecture_recorder.pipeline.queue import _save, enqueue, jobs, kind_of, run_worker

    s = tmp_path / SESSION
    s.mkdir(parents=True)
    enqueue(tmp_path, s)
    job = jobs(tmp_path)[0]
    job["state"] = "running"
    _save(job)
    (s / RAW).write_text("{}", encoding="utf-8")  # the child wrote its output, then the worker died
    calls = []
    monkeypatch.setattr(queue, "_run_job", lambda argv, **kw: calls.append(argv) or subprocess.CompletedProcess(argv, 0, "", ""))
    done = run_worker(tmp_path)
    assert [kind_of(j) for j in done] == ["summarize"]  # not transcribed again; only its summary runs
    # same-second jobs sort by a random id, so compare without order
    assert sorted((kind_of(j), j["state"]) for j in jobs(tmp_path)) == [("summarize", "done"), ("transcribe", "done")]


# ------------------------------------------------------------------ full pipeline on GPU
@pytest.fixture(scope="session")
def processed(synthetic, tmp_path_factory):
    """Run clean and noisy sessions through the real queue -> worker -> pipeline (one GPU job at a time)."""
    from lecture_recorder.pipeline.queue import enqueue, run_worker

    out, truth = synthetic
    data = tmp_path_factory.mktemp("data")
    sessions = {}
    for kind in ("clean", "noisy"):
        s = data / SESSION.replace("10-00-00", "10-00-00" if kind == "clean" else "11-00-00")
        s.mkdir(parents=True)
        shutil.copy(out / f"lecture_{kind}.wav", s / "audio.wav")
        enqueue(data, s)
        sessions[kind] = s
    done = run_worker(data)
    # each finished transcription queues its summary, then its search vectors, all in this worker run
    assert [(j.get("kind"), j["state"]) for j in done] ==         [("transcribe", "done")] * 2 + [("summarize", "done")] * 2 + [("embed", "done")] * 2,         [j.get("error") for j in done]
    return {k: (s, json.loads((s / RAW).read_text(encoding="utf-8"))) for k, s in sessions.items()}, truth


@pytest.mark.gpu
@pytest.mark.models
def test_pipeline_writes_plain_files_with_known_phrase(processed):
    sessions, truth = processed
    for kind, (s, doc) in sessions.items():
        for name in ("transcript.raw.json", "transcript.json", "transcript.md", "transcript.txt"):
            assert (s / name).is_file(), (kind, name)
        assert normalize(truth["known_phrase"]) in normalize((s / "transcript.txt").read_text(encoding="utf-8")), kind
        assert doc["stt"]["model"] == "large-v3-turbo" and doc["stt"]["compute_type"] == "int8_float16"
        assert doc["stt"]["vad_filter"] and doc["stt"]["word_timestamps"]
        assert all(w["end"] >= w["start"] for seg in doc["segments"] for w in seg["words"])
        assert doc["processing"]["dropped_speech"]["total_s"] < 2.0, kind


@pytest.mark.gpu
@pytest.mark.models
def test_vram_is_freed_after_each_model(processed):
    """Whisper is unloaded before pyannote loads, and pyannote is released before the job ends.
    Allowance: the CUDA contexts ctranslate2/torch keep for the rest of the process (~100-200 MB)."""
    sessions, _ = processed
    for kind, (_, doc) in sessions.items():
        v = doc["processing"]["vram_mb"]
        if v["before_mb"] is None:
            pytest.skip("nvidia-smi not available")
        assert v["after_stt_freed_mb"] - v["before_mb"] < 400, (kind, v)
        assert v["after_diarization_freed_mb"] - v["before_mb"] < 600, (kind, v)


@pytest.mark.gpu
@pytest.mark.models
def test_r3_main_is_speaker_1_and_roles_match_by_time(processed):
    sessions, truth = processed
    main_turns = [t for t in truth["turns"] if t["role"] == "main"]
    for kind, (_, doc) in sessions.items():
        # the transcript speaker covering most of the true main speaker's time must be Speaker 1 / Main
        cover: dict[str, float] = {}
        for t in main_turns:
            for s in doc["segments"]:
                ov = min(t["end"], s["end"]) - max(t["start"], s["start"])
                if ov > 0:
                    cover[s["speaker"]] = cover.get(s["speaker"], 0) + ov
        assert max(cover, key=cover.get) == "Speaker 1", (kind, cover)
        assert (doc["speakers"][0]["id"], doc["speakers"][0]["role"]) == ("Speaker 1", "Main")
        assert len(doc["speakers"]) >= 2, kind  # at least one audience voice found
        acc = M.role_accuracy(doc["segments"], truth["turns"])
        print(f"Speaker roles {kind}: role accuracy {acc:.1%}, speakers found {len(doc['speakers'])}")
        assert acc >= 0.85, (kind, acc)
        for t in truth["turns"]:
            if t["role"] == "audience":  # every audience question is mostly labeled Audience
                aud = M.role_time(doc["segments"], t["start"], t["end"], M.AUDIENCE)
                assert aud / (t["end"] - t["start"]) > 0.5, (kind, t["text"][:30])


@pytest.mark.gpu
@pytest.mark.models
def test_benchmark_script_reports_table(tmp_path):
    """Runs the benchmark on the synthetic fixture (both models, clean + noisy)."""
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "bench" / "benchmark_stt.py")], cwd=ROOT,
                       capture_output=True, text=True, timeout=1800)
    print(r.stdout[-2500:])
    assert "ollama ps" in r.stdout and "nvidia-smi" in r.stdout
    assert "speed and memory only; accuracy is compared on real recordings" in r.stdout
    rows = [l for l in r.stdout.splitlines() if l.startswith("| large-v3-turbo") or l.startswith("| distil")]
    assert len(rows) == 4, r.stdout + r.stderr[-2000:]
