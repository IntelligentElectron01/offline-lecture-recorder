"""Every transcribed lecture gets a verified Opus copy, whatever its source (wav, m4a, mp3, flac, ogg, an
mp4's audio), the original goes only with consent and after verification, everything that reads audio works
with only audio.opus left, and `compact` backfills older lectures. Synthetic files made with the bundled PyAV,
in temp folders only; never the real data/ folder.
"""
import dataclasses
import hashlib
import tomllib
from pathlib import Path

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient

import lecture_recorder  # noqa: F401  (lockdown)
from lecture_recorder import duplicates as dup
from lecture_recorder import importer as imp
from lecture_recorder import library as lib
from lecture_recorder import retention as ret
from lecture_recorder import trash
from lecture_recorder.config import RetentionConfig, load_config
from lecture_recorder.move import move_session
from lecture_recorder.pipeline import queue as q
from lecture_recorder.pipeline.audio import find_session_audio, load_audio
from lecture_recorder.recorder.platform import ProcessLock
from lecture_recorder.recorder.session import LOCK_FILE, read_json, write_json_atomic
from lecture_recorder.web.app import create_app
from test_web_app import BASE, H, write_tone_wav

RATE = 48000
ONSETS = (1.0, 3.25, 5.5)  # loud 1 kHz bursts after 0.3 s of silence: their start is easy to find
SECONDS = 7.0


def signal(seconds: float = SECONDS, rate: int = RATE) -> np.ndarray:
    """Speech-like (harmonics with a syllable-rate envelope, a little noise), stereo, with marked bursts."""
    t = np.arange(int(seconds * rate)) / rate
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) ** 2
    x = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((180, 360, 540, 900, 1400), 1)) * 0.12 * env
    x = (x + np.random.default_rng(1).normal(0, 0.01, len(t))).astype(np.float32)
    for on in ONSETS:
        i = int(on * rate)
        x[i - int(0.3 * rate):i] = 0
        n = int(0.08 * rate)
        x[i:i + n] = 0.8 * np.sin(2 * np.pi * 1000 * np.arange(n) / rate)
    return np.stack([x, 0.9 * x])


def write_coded(path: Path, codec: str, fmt: str, bit_rate: int, rate: int = RATE, layout: str = "stereo") -> Path:
    data = signal(rate=rate)
    data = data if layout == "stereo" else data[:1]
    with av.open(str(path), "w", format=fmt) as out:
        s = out.add_stream(codec, rate=rate)
        s.layout, s.bit_rate = layout, bit_rate
        for i in range(0, data.shape[1], 960):
            f = av.AudioFrame.from_ndarray(np.ascontiguousarray(data[:, i:i + 960]), format="fltp", layout=layout)
            f.sample_rate, f.pts = rate, i
            for p in s.encode(f):
                out.mux(p)
        for p in s.encode(None):
            out.mux(p)
    return path


def m4a(path):  # like a Zoom recording: AAC, stereo, 48 kHz, ~72 kbps
    return write_coded(path, "aac", "ipod", 72000)


def mp3(path, bit_rate=128000, rate=RATE, layout="stereo"):
    return write_coded(path, "libmp3lame", "mp3", bit_rate, rate, layout)


def onsets(audio: np.ndarray, rate: int = 16000) -> list[float]:
    out = []
    for on in ONSETS:
        lo = int((on - 0.15) * rate)
        out.append((lo + int(np.argmax(np.abs(audio[lo:lo + int(0.3 * rate)]) > 0.3))) / rate)
    return out


def pyav_decode(path: Path) -> np.ndarray:
    """Decoded the way a player does it (FFmpeg honors the Opus pre-skip and the AAC/MP3 encoder delay)."""
    with av.open(str(path)) as c:
        rs = av.AudioResampler(format="flt", layout="mono", rate=16000)
        frames = [f for fr in c.decode(audio=0) for f in rs.resample(fr)] + rs.resample(None)
    return np.concatenate([f.to_ndarray().reshape(-1) for f in frames])


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LECTURE_RECORDER_HOME", str(home))  # never the install's own config.toml
    return dataclasses.replace(load_config(home), data_dir=(home / "data").resolve())


def lecture(cfg, name: str, make=None, transcribed: bool = True, source: dict | None = None) -> Path:
    """A lecture in physics-101 whose audio `make(dir)` writes (default: an m4a)."""
    data = cfg.data_dir
    if not (data / "classes" / "physics-101" / "class.json").exists():
        lib.create_class(data, "Physics 101")
    d = data / "classes" / "physics-101" / "sessions" / name
    d.mkdir(parents=True)
    (make or (lambda d: m4a(d / "audio.m4a")))(d)
    write_json_atomic(d / "meta.json", {"name": name, "class": "physics-101", "created": "2026-09-01T09:00:00",
                                        "state": "stopped", "source": source or {"kind": "import", "file": "a.m4a"}})
    if transcribed:
        (d / q.RAW).write_text("{}", encoding="utf-8")
    return d


# ------------------------------------------------------------------ 1. every source gets a verified copy
@pytest.mark.parametrize("name,make", [
    ("audio.m4a", m4a), ("audio.mp3", mp3),
    ("audio.flac", lambda p: write_coded(p, "flac", "flac", 0, 48000, "mono")),
    ("audio.wav", lambda p: write_tone_wav(p, seconds=5)),
])
def test_every_source_gets_a_verified_smaller_opus_copy(tmp_path, name, make):
    make(tmp_path / name)
    before = sha(tmp_path / name)
    rec = ret.compress_session(tmp_path)
    assert rec["verified"] and rec["source"] == name and rec["original_kept"] is True
    assert rec["check"]["rms_diff_db"] <= 3 and abs(rec["check"]["source_duration_s"] - rec["check"]["opus_duration_s"]) <= 0.05
    assert (tmp_path / "audio.opus").stat().st_size < (tmp_path / name).stat().st_size
    assert sha(tmp_path / name) == before  # kept unchanged: the setting is off
    assert read_json(tmp_path / ret.RECORD)["source"] == name


@pytest.mark.parametrize("name,make", [("audio.m4a", m4a), ("audio.mp3", mp3)])
def test_transcript_times_line_up_with_opus_playback(tmp_path, name, make):
    """The transcript's times come from the original decoded by PyAV; the player gets audio.opus once the
    original is gone. A burst must start at the same moment in both (encoder delay and Opus pre-skip)."""
    make(tmp_path / name)
    transcript_clock = load_audio(tmp_path / name)  # exactly what transcription sees
    assert ret.compress_session(tmp_path, delete_original=True)["original_kept"] is False
    for got in (onsets(pyav_decode(tmp_path / "audio.opus")), onsets(load_audio(tmp_path / "audio.opus"))):
        assert np.allclose(got, onsets(transcript_clock), atol=0.005), (got, onsets(transcript_clock))
    assert np.allclose(onsets(transcript_clock), ONSETS, atol=0.01)  # and with the true times


def test_original_deleted_only_with_consent_and_after_verification(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    m4a(a / "audio.m4a")
    m4a(b / "audio.m4a")
    rec = ret.compress_session(a, delete_original=True)
    assert rec["original_kept"] is False and not (a / "audio.m4a").exists() and rec["original_deleted"]
    monkeypatch.setattr(ret, "verify_opus", lambda *a, **k: {"ok": False, "error": "simulated"})
    rec = ret.compress_session(b, delete_original=True)
    assert rec["verified"] is False and rec["original_kept"] is True and (b / "audio.m4a").exists()
    assert not (b / "audio.opus").exists()  # this run's failed copy is gone, so a later run can try again


def test_a_copy_that_saves_nothing_is_not_kept(tmp_path):
    mp3(tmp_path / "audio.mp3", bit_rate=8000, rate=8000, layout="mono")  # 8 kbps: smaller than any Opus copy
    before = sha(tmp_path / "audio.mp3")
    rec = ret.compress_session(tmp_path, delete_original=True)
    assert rec["skipped"] == "the original is already smaller than an Opus copy" and rec["original_kept"] is True
    assert sha(tmp_path / "audio.mp3") == before and not (tmp_path / "audio.opus").exists()


def test_already_opus_at_or_below_the_recorder_bitrate_is_left_alone(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    write_coded(a / "audio.opus", "libopus", "ogg", 24000)  # an imported .opus
    write_coded(b / "audio.ogg", "libopus", "ogg", 24000)  # Opus inside an .ogg
    before = sha(a / "audio.opus"), sha(b / "audio.ogg")
    for d in (a, b):
        rec = ret.compress_session(d, delete_original=True)
        assert rec["skipped"].startswith("already Opus at") and rec["original_kept"] is True
    assert (sha(a / "audio.opus"), sha(b / "audio.ogg")) == before and not (b / "audio.opus").exists()


def test_imported_opus_above_the_bitrate_is_replaced_only_with_consent(tmp_path):
    write_coded(tmp_path / "audio.opus", "libopus", "ogg", 128000)
    big = (tmp_path / "audio.opus").stat().st_size
    rec = ret.compress_session(tmp_path, delete_original=False)
    assert "needs the delete-original setting" in rec["skipped"] and (tmp_path / "audio.opus").stat().st_size == big
    rec = ret.compress_session(tmp_path, delete_original=True)
    assert rec["verified"] and rec["original_kept"] is False and (tmp_path / "audio.opus").stat().st_size < big
    assert not list(tmp_path.glob("*.new")) and ret.opus_kbps(tmp_path / "audio.opus") <= ret.OPUS_KEEP_KBPS


def test_the_users_own_mp4_outside_data_is_never_touched(cfg, tmp_path):
    lib.create_class(cfg.data_dir, "Physics 101")
    mine = write_coded(tmp_path / "meeting.mp4", "aac", "mp4", 96000)
    before = sha(mine)
    d = imp.create_session(cfg.data_dir, "physics-101", mine, "meeting.mp4", size=mine.stat().st_size, keep_source=True)
    assert (d / "audio.m4a").exists()
    rec = ret.compress_session(d, delete_original=True)
    assert rec["source"] == "audio.m4a" and rec["original_kept"] is False and (d / "audio.opus").exists()
    assert mine.exists() and sha(mine) == before


def test_after_transcription_every_source_is_compacted_with_the_older_setting_name(cfg):
    """The user's config.toml says `delete_wav_after_opus = true`; it now covers an m4a too."""
    from lecture_recorder.pipeline.run import _after_processing

    (cfg.home / "config.toml").write_text("[retention]\ndelete_wav_after_opus = true\n", encoding="utf-8")
    d = lecture(cfg, "2026-09-01_09-00-00")
    _after_processing(d, cfg, load_audio(d / "audio.m4a"), d / "audio.m4a")
    rec = read_json(d / ret.RECORD)
    assert rec["verified"] and rec["original_kept"] is False and not (d / "audio.m4a").exists()
    (cfg.home / "config.toml").write_text("[retention]\ndelete_wav_after_opus = true\ndelete_original_after_opus = false\n",
                                          encoding="utf-8")  # the new name wins
    e = lecture(cfg, "2026-09-02_09-00-00")
    _after_processing(e, cfg, None, e / "audio.m4a")
    assert (e / "audio.m4a").exists() and read_json(e / ret.RECORD)["verified"]


def test_setting_names(tmp_path):
    assert RetentionConfig().delete_original_after_opus is False
    assert RetentionConfig(delete_wav_after_opus=True).delete_original_after_opus is True
    assert RetentionConfig(delete_wav_after_opus=True, delete_original_after_opus=False).delete_original_after_opus is False
    with pytest.raises(ValueError):
        RetentionConfig(delete_original_after_opus="yes")


def test_saving_the_setting_also_updates_an_older_line(cfg):
    from lecture_recorder.settings import set_config_values

    (cfg.home / "config.toml").write_text("[retention]\ndelete_wav_after_opus = true  # mine\n", encoding="utf-8")
    set_config_values(cfg.home, {("retention", "delete_original_after_opus"): False})
    parsed = tomllib.loads((cfg.home / "config.toml").read_text(encoding="utf-8"))
    assert parsed["retention"] == {"delete_wav_after_opus": False, "delete_original_after_opus": False}
    assert load_config(cfg.home).retention.delete_original_after_opus is False


# ------------------------------------------------------------------ 3. everything that reads audio, with only audio.opus
@pytest.fixture
def opus_only(cfg):
    sha_m4a = None

    def make(d):
        nonlocal sha_m4a
        m4a(d / "audio.m4a")
        sha_m4a = dup.file_sha256(d / "audio.m4a")

    d = lecture(cfg, "2026-09-01_09-00-00", make)
    meta = read_json(d / "meta.json")
    write_json_atomic(d / "meta.json", {**meta, "source": {"kind": "import", "file": "a.m4a", "bytes": 1, "sha256": sha_m4a}})
    ret.compress_session(d, delete_original=True)
    assert [p.name for p in d.iterdir() if p.name.startswith("audio")] == ["audio.opus"]
    return d


def test_player_and_files_use_the_opus_copy(cfg, opus_only):
    detail = lib.session_detail(cfg.data_dir, "physics-101", opus_only.name)
    assert detail["audio"] == "audio.opus" and "audio.opus" in detail["files"] and lib.has_audio(opus_only)
    assert detail["retention"]["original_kept"] is False
    with TestClient(create_app(cfg, autoprocess=False, monitor=False), base_url=BASE) as c:
        r = c.get(f"/files/physics-101/{opus_only.name}/audio.opus", headers={"Range": "bytes=0-99"})
        assert r.status_code == 206 and r.headers["content-type"] == "audio/ogg" and r.content[:4] == b"OggS"


def test_original_is_preferred_while_kept(tmp_path):
    m4a(tmp_path / "audio.m4a")
    ret.compress_session(tmp_path)
    assert find_session_audio(tmp_path).name == "audio.m4a"  # transcription and the player use the original
    (tmp_path / "audio.m4a").unlink()
    assert find_session_audio(tmp_path).name == "audio.opus"


def test_transcribing_again_and_anyway_work_from_the_opus_copy(cfg, opus_only, monkeypatch):
    """process --anyway / a re-transcription after the transcript was removed: the pipeline reads audio.opus,
    and the finished job leaves the copy and its record alone. Models are faked (no GPU)."""
    import lecture_recorder.pipeline.run as run
    from lecture_recorder.pipeline.transcribe import SttChoice

    (opus_only / q.RAW).unlink()
    lib.transcribe_anyway(opus_only)
    record, opus = read_json(opus_only / ret.RECORD), sha(opus_only / "audio.opus")
    seen = []
    real_load = run.load_audio
    monkeypatch.setattr(run, "load_audio", lambda p: seen.append(Path(p).name) or real_load(p))
    monkeypatch.setattr(run, "choose", lambda *a: SttChoice("turbo", cfg.models_dir, "cpu", "int8"))
    monkeypatch.setattr("lecture_recorder.summary.llm.release_llm_vram", lambda *a: None)
    seg = {"id": 0, "start": ONSETS[0], "end": ONSETS[0] + 1, "text": "Hello.", "words": []}
    monkeypatch.setattr(run, "transcribe", lambda audio, *a, **k: {
        "segments": [seg], "duration_s": len(audio) / 16000, "settings": {}, "timings": {"load_s": 0, "transcribe_s": 0}})
    monkeypatch.setattr(run, "diarize", lambda *a, **k: {
        "turns": [{"start": 0.0, "end": SECONDS, "label": "SPEAKER_00"}], "settings": {}, "timings": {"load_s": 0, "diarize_s": 0}})
    doc = run.process_session(opus_only, dataclasses.replace(cfg, home=cfg.home))
    assert seen == ["audio.opus"] and doc["audio"] == "audio.opus" and abs(doc["duration_s"] - SECONDS) < 0.1
    assert sha(opus_only / "audio.opus") == opus and read_json(opus_only / ret.RECORD) == record


def test_recently_deleted_restore_and_move_keep_the_opus_copy(cfg, opus_only):
    opus = sha(opus_only / "audio.opus")
    trash.delete_session(cfg.data_dir, "physics-101", opus_only.name)
    assert not opus_only.exists()
    (tid,) = [x["id"] for x in trash.list_trash(cfg.data_dir)]
    trash.restore(cfg.data_dir, tid)
    assert sha(opus_only / "audio.opus") == opus
    lib.create_class(cfg.data_dir, "Chem 200")
    r = move_session(cfg.data_dir, "physics-101", opus_only.name, "chem-200")
    moved = cfg.data_dir / "classes" / "chem-200" / "sessions" / r["session"]
    assert sha(moved / "audio.opus") == opus and find_session_audio(moved).name == "audio.opus"
    assert lib.session_detail(cfg.data_dir, "chem-200", r["session"])["audio"] == "audio.opus"


def test_same_file_duplicate_check_still_finds_a_compacted_import(cfg, opus_only):
    src = read_json(opus_only / "meta.json")["source"]
    hit = dup.find_source(cfg.data_dir, src["sha256"], "again.m4a", 123)
    assert hit and hit["session"] == opus_only.name


# ------------------------------------------------------------------ 4. backfill: compact
@pytest.fixture
def library(cfg):
    """One lecture per case the plan must handle."""
    s = {
        "m4a": lecture(cfg, "2026-09-01_09-00-00"),
        "mp3": lecture(cfg, "2026-09-02_09-00-00", lambda d: mp3(d / "audio.mp3")),
        "wav_with_copy": lecture(cfg, "2026-09-03_09-00-00", lambda d: write_tone_wav(d / "audio.wav", seconds=5)),
        "untranscribed": lecture(cfg, "2026-09-04_09-00-00", transcribed=False),
        "recording": lecture(cfg, "2026-09-05_09-00-00"),
        "queued": lecture(cfg, "2026-09-06_09-00-00"),
        "done": lecture(cfg, "2026-09-07_09-00-00"),
        "tiny_mp3": lecture(cfg, "2026-09-08_09-00-00", lambda d: mp3(d / "audio.mp3", 8000, 8000, "mono")),
    }
    ret.compress_session(s["wav_with_copy"])
    ret.compress_session(s["done"], delete_original=True)
    q.enqueue(cfg.data_dir, s["queued"], q.SUMMARIZE)  # any job of the lecture counts
    return s


def test_dry_run_lists_bytes_and_reasons_and_changes_nothing(cfg, library):
    snapshot = {p: sha(p) for p in cfg.data_dir.rglob("audio.*")}
    lock = ProcessLock(library["recording"] / LOCK_FILE)
    assert lock.acquire()
    try:
        plan = ret.compact_plan(cfg.data_dir)
    finally:
        lock.release()
    assert {p: sha(p) for p in cfg.data_dir.rglob("audio.*")} == snapshot
    items = {x["session"].split("/")[1]: x for x in plan["items"]}
    assert {k: (v["action"], v["file"]) for k, v in items.items()} == {
        library["m4a"].name: ("encode", "audio.m4a"), library["mp3"].name: ("encode", "audio.mp3"),
        library["wav_with_copy"].name: ("delete original", "audio.wav")}
    for x in plan["items"]:
        assert x["bytes_after"] < x["bytes_before"]
    assert plan["bytes_before"] == sum(x["bytes_before"] for x in plan["items"])
    reasons = {x["session"].split("/")[1]: x["reason"] for x in plan["skipped"]}
    assert reasons == {library["untranscribed"].name: "not transcribed yet (compacted after its transcription)",
                       library["recording"].name: "recording, queued or processing",
                       library["queued"].name: "recording, queued or processing",
                       library["tiny_mp3"].name: "the original is already smaller than an Opus copy would be"}
    app_plan = ret.compact_plan(cfg.data_dir, encode=False)  # the app's button: only lectures that have a copy
    assert [x["session"].split("/")[1] for x in app_plan["items"]] == [library["wav_with_copy"].name]


def test_compact_runs_the_confirmed_plan_only(cfg, library):
    lock = ProcessLock(library["recording"] / LOCK_FILE)
    assert lock.acquire()
    try:
        plan = ret.compact_plan(cfg.data_dir)
        with pytest.raises(ValueError):
            ret.compact(cfg.data_dir, "not-the-token")
        sizes = sum(x["bytes_before"] for x in plan["items"])
        r = ret.compact(cfg.data_dir, plan["token"])
        assert ret.compact_plan(cfg.data_dir)["items"] == []  # nothing left; the done ones aren't listed again
    finally:
        lock.release()
    assert len(r["done"]) == 3 and r["kept"] == [] and r["bytes_before"] == sizes and r["bytes_after"] < sizes / 2
    for k in ("m4a", "mp3", "wav_with_copy"):
        d = library[k]
        assert [p.name for p in d.glob("audio.*")] == ["audio.opus"] and read_json(d / ret.RECORD)["verified"]
    for k in ("untranscribed", "recording", "queued", "tiny_mp3"):
        assert any(p.name != "audio.opus" for p in library[k].glob("audio.*")), k


def test_a_lecture_that_becomes_busy_after_the_dry_run_is_kept(cfg, library, monkeypatch):
    plan = ret.compact_plan(cfg.data_dir)
    real = ret._active_sessions
    calls = []

    def later(data_dir):  # the plan saw it idle; it is queued by the time its turn comes
        calls.append(1)
        return real(data_dir) | ({str(library["m4a"].resolve())} if len(calls) > 1 else set())

    monkeypatch.setattr(ret, "_active_sessions", later)
    r = ret.compact(cfg.data_dir, plan["token"])
    assert {x["session"].split("/")[1]: x["reason"] for x in r["kept"]} == {library["m4a"].name: "it started recording or processing"}
    assert (library["m4a"] / "audio.m4a").exists() and not (library["m4a"] / "audio.opus").exists()


def test_cli_compact_dry_run_and_typed_delete(cfg, library, monkeypatch, capsys):
    from lecture_recorder.cli import main

    monkeypatch.setattr("lecture_recorder.cli._cfg", lambda data_dir=None: cfg)
    assert main(["compact"]) == 2  # the setting is off
    on = dataclasses.replace(cfg, retention=RetentionConfig(delete_original_after_opus=True))
    monkeypatch.setattr("lecture_recorder.cli._cfg", lambda data_dir=None: on)
    del library["recording"]  # nothing holds its lock here: it is compacted like the others
    snapshot = {p: sha(p) for p in cfg.data_dir.rglob("audio.*")}
    assert main(["compact", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "encode" in out and "(estimate)" in out and "Nothing has been changed" in out
    monkeypatch.setattr("builtins.input", lambda prompt="": "yes")
    assert main(["compact"]) == 1
    assert {p: sha(p) for p in cfg.data_dir.rglob("audio.*")} == snapshot
    monkeypatch.setattr("builtins.input", lambda prompt="": "DELETE")
    lowered = []
    monkeypatch.setattr("lecture_recorder.cli.lower_own_priority", lambda: lowered.append(1))
    assert main(["cleanup-wav"]) == 0 and lowered == [1]  # the older command name runs the same, below normal priority
    assert "Compacted 4 lecture(s)" in capsys.readouterr().out
    assert not (library["m4a"] / "audio.m4a").exists() and (library["m4a"] / "audio.opus").exists()


def test_app_button_deletes_only_originals_that_have_a_copy(cfg, library):
    on = dataclasses.replace(cfg, retention=RetentionConfig(delete_original_after_opus=True))
    with TestClient(create_app(on, autoprocess=False, monitor=False), base_url=BASE) as c:
        plan = c.get("/api/compact").json()
        assert [x["action"] for x in plan["items"]] == ["delete original"]
        assert c.post("/api/compact", headers=H, json={"token": plan["token"]}).status_code == 400
        r = c.post("/api/compact", headers=H, json={"token": plan["token"], "confirm": True}).json()
    assert len(r["done"]) == 1 and not (library["wav_with_copy"] / "audio.wav").exists()
    assert (library["m4a"] / "audio.m4a").exists()  # encoding is left to `lecture-recorder compact`


def test_settings_api_uses_the_new_name_and_accepts_the_old(cfg):
    with TestClient(create_app(cfg, autoprocess=False, monitor=False), base_url=BASE) as c:
        assert c.get("/api/processing").json()["delete_original_after_opus"] is False
        assert c.put("/api/settings", headers=H, json={"delete_original_after_opus": True}).status_code == 400
        r = c.put("/api/settings", headers=H, json={"delete_original_after_opus": True, "confirm_permanent_deletion": True})
        assert r.json()["delete_original_after_opus"] is True
        assert c.put("/api/settings", headers=H, json={"delete_wav_after_opus": False}).json()["delete_original_after_opus"] is False


# ------------------------------------------------------------------ review fixes
def test_a_skip_that_needed_the_setting_is_planned_again_once_it_is_on(cfg):
    d = lecture(cfg, "2026-09-01_09-00-00", lambda d: write_coded(d / "audio.opus", "libopus", "ogg", 128000))
    assert ret.compress_session(d, delete_original=False)["skipped"] == ret.NEEDS_SETTING
    (item,) = ret.compact_plan(cfg.data_dir)["items"]
    assert item["action"] == "encode" and item["file"] == "audio.opus"


def test_an_error_never_overwrites_what_was_deleted(cfg, opus_only):
    ret.record_error(opus_only, "simulated: config.toml unreadable")
    rec = read_json(opus_only / ret.RECORD)
    assert rec["original_kept"] is False and rec["verified"] and rec["error"].startswith("simulated")
    assert ret.own_copy_done(opus_only) and ret.compress_session(opus_only, delete_original=True) is None
    write_json_atomic(opus_only / ret.RECORD, {"verified": True, "source": "audio.m4a"})  # a record without the flag
    assert ret.own_copy_done(opus_only)  # its original is gone: the app's own copy is never re-encoded


def test_an_original_that_decodes_short_is_kept(tmp_path, monkeypatch):
    m4a(tmp_path / "audio.m4a")
    monkeypatch.setattr(ret, "audio_seconds", lambda p: SECONDS + 30)  # the header promises 30 s more
    rec = ret.compress_session(tmp_path, delete_original=True)
    assert rec["verified"] and rec["original_kept"] is True and "shorter" in rec["kept_why"]
    assert (tmp_path / "audio.m4a").exists() and ret._delete_original(tmp_path, tmp_path / "audio.m4a") is not None
    assert (tmp_path / "audio.m4a").exists()


def test_a_copy_that_no_longer_verifies_is_made_again(cfg):
    d = lecture(cfg, "2026-09-01_09-00-00")
    ret.compress_session(d)
    (d / "audio.opus").write_bytes(b"OggS" + b"\0" * 100)  # damaged since
    write_json_atomic(d / ret.RECORD, {**read_json(d / ret.RECORD), "verified": False})
    (item,) = ret.compact_plan(cfg.data_dir)["items"]
    assert item["action"] == "encode"
    r = ret.compact(cfg.data_dir, ret.compact_plan(cfg.data_dir)["token"])
    assert len(r["done"]) == 1 and not (d / "audio.m4a").exists()
    assert read_json(d / ret.RECORD)["verified"] and (d / "audio.opus").stat().st_size > 1000


def test_compact_stops_when_a_recording_starts(cfg, library, monkeypatch):
    plan = ret.compact_plan(cfg.data_dir)
    monkeypatch.setattr(ret, "_recording", lambda data_dir: True)
    r = ret.compact(cfg.data_dir, plan["token"])
    assert r["done"] == [] and {x["reason"] for x in r["kept"]} == {"a recording started; run compact again afterwards"}
    assert (library["m4a"] / "audio.m4a").exists() and (library["wav_with_copy"] / "audio.wav").exists()


def test_a_failed_swap_of_an_imported_opus_leaves_no_new_file(tmp_path, monkeypatch):
    write_coded(tmp_path / "audio.opus", "libopus", "ogg", 128000)
    before = sha(tmp_path / "audio.opus")
    real = ret.os.replace

    def held_open(src, dst):  # Windows: the player holds audio.opus open
        if Path(dst).name == "audio.opus" and Path(src).name.endswith(".new"):
            raise PermissionError("in use")
        return real(src, dst)

    monkeypatch.setattr(ret.os, "replace", held_open)
    with pytest.raises(PermissionError):
        ret.compress_session(tmp_path, delete_original=True)
    assert sha(tmp_path / "audio.opus") == before and sorted(p.name for p in tmp_path.iterdir()) == ["audio.opus"]
