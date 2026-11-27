"""Machine auto-configuration (mocked hardware profiles) and the raised Opus quality."""
import time

import pytest

import lecture_recorder  # noqa: F401  (lockdown)
from lecture_recorder import autoconfig as ac
from lecture_recorder.config import load_config

ALL_MODELS = {"large-v3-turbo": True, "distil-large-v3.5": True, "pyannote": True}
OLLAMA_OK = {"installed": True, "reachable": True, "models": ["qwen3-embedding:0.6b", "qwen3:8b"]}


def profile(**kw) -> dict:
    p = {"gpu": None, "cuda_usable": False, "logical_cpus": 8, "ram_gb": 32, "disk_free_gb": 200,
         "battery": False, "ollama": OLLAMA_OK, "models": ALL_MODELS}
    return {**p, **kw}


def t(result: dict) -> dict:
    return result["settings"]["transcription"]


def test_no_gpu_uses_turbo_int8_on_cpu_sequential():
    r = ac.pick(profile())
    assert t(r) == {"model": "large-v3-turbo", "device": "cpu", "compute_type": "int8", "batch_size": 0, "cpu_threads": 4}
    assert any("CPU" in n for n in r["notes"]) and r["missing"] == []


def test_4gb_gpu_uses_cuda_and_warns_summaries_split():
    r = ac.pick(profile(gpu={"name": "GTX 1650", "vram_mb": 4096}, cuda_usable=True))
    assert t(r)["device"] == "cuda" and t(r)["compute_type"] == "int8_float16" and t(r)["model"] == "large-v3-turbo"
    assert any("Ollama runs part of it on the CPU" in n for n in r["notes"])


def test_small_or_unusable_gpu_falls_back_to_cpu():
    assert t(ac.pick(profile(gpu={"name": "MX150", "vram_mb": 2048}, cuda_usable=True)))["device"] == "cpu"
    r = ac.pick(profile(gpu={"name": "RTX 3060", "vram_mb": 12288}, cuda_usable=False))
    assert t(r)["device"] == "cpu" and any("driver" in n for n in r["notes"])


def test_8gb_gpu_no_notes_about_the_gpu():
    r = ac.pick(profile(gpu={"name": "Example GPU 8GB", "vram_mb": 8192}, cuda_usable=True))
    assert t(r)["device"] == "cuda" and not any("GPU" in n or "Ollama" in n for n in r["notes"])


def test_battery_note_and_threads_sized_to_the_machine():
    r = ac.pick(profile(battery=True, logical_cpus=2))
    assert any("plugged in" in n for n in r["notes"]) and t(r)["cpu_threads"] == 1
    assert t(ac.pick(profile(logical_cpus=32)))["cpu_threads"] == 4


def test_ollama_missing_or_stopped_and_models_missing_give_approved_commands_only():
    r = ac.pick(profile(ollama={"installed": False, "reachable": False, "models": []},
                        models={"large-v3-turbo": False, "distil-large-v3.5": False, "pyannote": False}))
    assert ac.DOWNLOAD_HINTS["ollama"] in r["missing"] and ac.DOWNLOAD_HINTS["pyannote"] in r["missing"]
    assert ac.DOWNLOAD_HINTS["large-v3-turbo"] in r["missing"] and t(r)["model"] == "large-v3-turbo"
    assert set(r["missing"]) <= set(ac.DOWNLOAD_HINTS.values())  # nothing outside the approved list
    stopped = ac.pick(profile(ollama={"installed": True, "reachable": False, "models": []}))
    assert any("not running" in n for n in stopped["notes"]) and stopped["missing"] == []
    partial = ac.pick(profile(ollama={"installed": True, "reachable": True, "models": ["qwen3:8b"]}))
    assert partial["missing"] == ["ollama pull qwen3-embedding:0.6b"]


def test_only_distil_downloaded_uses_batched_decoding():
    r = ac.pick(profile(models={"large-v3-turbo": False, "distil-large-v3.5": True, "pyannote": True}))
    assert t(r)["model"] == "distil-large-v3.5" and t(r)["batch_size"] == 8
    assert any("approximate" in n for n in r["notes"])


def test_low_ram_and_disk_notes():
    notes = ac.pick(profile(ram_gb=8, disk_free_gb=5))["notes"]
    assert any("RAM" in n for n in notes) and any("disk" in n for n in notes)


def test_configure_writes_auto_file_and_never_overrides_user_values(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "detect", lambda *a: profile(gpu={"name": "X", "vram_mb": 8192}, cuda_usable=True))
    user = '[transcription]\nmodel = "auto"\ndevice = "cpu"   # mine\n'
    (tmp_path / "config.toml").write_text(user, encoding="utf-8")
    dry = ac.configure(tmp_path, dry_run=True)
    assert not (tmp_path / ac.AUTO_FILE).exists() and dry["written"] is False
    r = ac.configure(tmp_path)
    assert (tmp_path / "config.toml").read_text(encoding="utf-8") == user  # the user's file is untouched
    assert r["overrides"] == {"transcription.device": "cpu"}
    cfg = load_config(tmp_path)
    assert cfg.transcription.device == "cpu"  # the user's value wins
    assert cfg.transcription.model == "large-v3-turbo"  # the pick
    assert cfg.transcription.compute_type == "int8"  # follows the user's device, not the stale cuda pick
    assert ac.read_auto(tmp_path)["transcription"]["batch_size"] == 0
    assert any("config.toml sets 'cpu'" in line for line in ac.describe(r))


def test_without_auto_file_defaults_still_work(tmp_path):
    cfg = load_config(tmp_path)
    assert cfg.transcription.model == "auto" and cfg.transcription.batch_size == 0


@pytest.mark.gpu  # real probes: nvidia-smi, CUDA, loopback Ollama
def test_detection_is_fast():
    t0 = time.perf_counter()
    p = ac.detect(load_config().home, load_config().models_dir, "http://127.0.0.1:11434")
    assert time.perf_counter() - t0 < 2.0
    assert {"gpu", "cuda_usable", "logical_cpus", "ram_gb", "disk_free_gb", "battery", "ollama", "models"} <= set(p)


def test_cli_configure_dry_run_prints_picks(tmp_path, monkeypatch, capsys):
    from lecture_recorder.cli import main

    monkeypatch.setenv("LECTURE_RECORDER_HOME", str(tmp_path))
    monkeypatch.setattr(ac, "detect", lambda *a: profile())
    assert main(["configure", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "transcription.device = cpu" in out and "dry run" in out and not (tmp_path / ac.AUTO_FILE).exists()


@pytest.mark.gpu  # real detection in the child: nvidia-smi, CUDA probe, loopback Ollama
def test_settings_api_shows_picks_and_redetects(tmp_path, monkeypatch):
    import dataclasses

    from fastapi.testclient import TestClient

    from lecture_recorder.web.app import create_app

    monkeypatch.setattr(ac, "detect", lambda *a: profile())
    home = tmp_path
    ac.configure(home)
    cfg = dataclasses.replace(load_config(home), data_dir=home / "data")
    with TestClient(create_app(cfg, autoprocess=False, monitor=False), base_url="http://127.0.0.1:8765") as c:
        assert c.get("/api/processing").json()["machine"]["picks"]["device"] == "cpu"
        assert c.post("/api/configure").status_code == 403  # state change without the app header
        before = (tmp_path / ac.AUTO_FILE).stat().st_mtime_ns
        r = c.post("/api/configure", headers={"X-Lecture-Recorder": "1"})  # real detection, in a child process
        assert r.status_code == 200 and r.json()["picks"]["model"] == "large-v3-turbo"
    assert (tmp_path / ac.AUTO_FILE).stat().st_mtime_ns != before  # written into this app's home, nowhere else

# ------------------------------------------------------------------ B: Opus quality
def test_opus_copy_uses_the_raised_quality(tmp_path):
    import numpy as np
    import soundfile as sf

    from lecture_recorder.retention import compress_session

    rng = np.random.default_rng(0)
    sf.write(str(tmp_path / "audio.wav"), (0.1 * rng.standard_normal(16000 * 5)).astype(np.float32), 16000, subtype="PCM_16")
    rec = compress_session(tmp_path)
    kbps = (tmp_path / "audio.opus").stat().st_size * 8 / 5 / 1000
    assert rec["verified"] and 30 < kbps < 70, kbps  # ~43 kbps at OPUS_COMPRESSION_LEVEL (old default ~27)


# ------------------------------------------------------------------ regressions
def test_broken_auto_file_is_ignored_and_configure_repairs_it(tmp_path, monkeypatch):
    (tmp_path / ac.AUTO_FILE).write_text("[transcription\nmodel = ", encoding="utf-8")
    assert load_config(tmp_path).transcription.model == "auto"  # the app still starts
    monkeypatch.setattr(ac, "detect", lambda *a: profile())
    ac.configure(tmp_path)
    assert load_config(tmp_path).transcription.device == "cpu"
    (tmp_path / "config.toml").write_text("[broken", encoding="utf-8")
    with pytest.raises(Exception):  # the user's own file still fails loudly
        load_config(tmp_path)


def test_auto_without_a_pick_means_the_default_and_bad_numbers_are_refused(tmp_path):
    (tmp_path / "config.toml").write_text('[transcription]\nbatch_size = "auto"\ncpu_threads = "auto"\n', encoding="utf-8")
    t_cfg = load_config(tmp_path).transcription
    assert t_cfg.batch_size == 0 and t_cfg.cpu_threads == ac.default_threads()  # one value for Whisper and pyannote
    (tmp_path / "config.toml").write_text('[transcription]\ncpu_threads = "lots"\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_user_model_or_device_brings_its_own_decoding_and_compute_type(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "detect", lambda *a: profile(models={"large-v3-turbo": False, "distil-large-v3.5": True,
                                                                  "pyannote": True}))
    ac.configure(tmp_path)  # pick: distil, batched, cpu/int8
    (tmp_path / "config.toml").write_text('[transcription]\nmodel = "large-v3-turbo"\ndevice = "cuda"\n', encoding="utf-8")
    t_cfg = load_config(tmp_path).transcription
    assert (t_cfg.model, t_cfg.batch_size, t_cfg.device, t_cfg.compute_type) == ("large-v3-turbo", 0, "cuda", "int8_float16")
    (tmp_path / "config.toml").write_text('[transcription]\nmodel = "large-v3-turbo"\nbatch_size = 4\n', encoding="utf-8")
    assert load_config(tmp_path).transcription.batch_size == 4  # explicitly the user's: kept


def test_picked_cuda_falls_back_to_cpu_when_cuda_is_gone(monkeypatch, tmp_path):
    import lecture_recorder.pipeline.transcribe as tr

    monkeypatch.setattr(tr, "cuda_available", lambda: False)
    (tmp_path / tr.MODEL_DIRS["large-v3-turbo"]).mkdir()
    (tmp_path / tr.MODEL_DIRS["large-v3-turbo"] / "model.bin").write_bytes(b"")
    c = tr.choose(tmp_path, "large-v3-turbo", "cuda", "int8_float16")
    assert (c.device, c.compute_type) == ("cpu", "int8")


def test_non_table_section_in_config_toml_fails_loudly(tmp_path):
    (tmp_path / "config.toml").write_text('transcription = "cpu"\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_auto_distil_without_a_pick_is_batched(tmp_path, monkeypatch):
    import dataclasses

    import lecture_recorder.pipeline.run as run
    from lecture_recorder.pipeline.transcribe import SttChoice

    seen = {}
    monkeypatch.setattr(run, "choose", lambda *a: SttChoice("distil-large-v3.5", tmp_path, "cpu", "int8"))
    monkeypatch.setattr(run, "check_audio", lambda *a: [])
    monkeypatch.setattr(run, "load_audio", lambda p: __import__("numpy").zeros(16000, "float32"))

    def stop(audio, choice, *a, **k):
        seen.update(k)
        raise RuntimeError("stop here")

    monkeypatch.setattr(run, "transcribe", stop)
    d = tmp_path / "s"
    d.mkdir()
    (d / "audio.wav").write_bytes(b"")
    cfg = dataclasses.replace(load_config(tmp_path), data_dir=tmp_path)
    with pytest.raises(RuntimeError, match="stop here"):
        run.process_session(d, cfg)
    assert seen["batch_size"] == 8
