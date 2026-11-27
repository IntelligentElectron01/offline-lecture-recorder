"""`lecture-recorder` launcher.

  lecture-recorder doctor   check Python, GPU/CPU path, models, and Ollama
  lecture-recorder serve    start the local app on 127.0.0.1 and open it in the browser
  lecture-recorder devices  list microphones
  lecture-recorder apps     list apps playing audio (for record --app)
  lecture-recorder record   record from the terminal (p/r/b/Enter); --app records one app's audio
  lecture-recorder import-zoom <folder> --class <slug>  import a Zoom recording folder (local or cloud)
  lecture-recorder recover  save recordings left unfinished by a crash
  lecture-recorder process  transcribe + label speakers + summarize a session (serial queue)
  lecture-recorder summarize  (re)queue only the summary for a session
  lecture-recorder reindex  rebuild the search index from the session folders (--embed also computes
                            missing concept-search vectors through the GPU queue)
  lecture-recorder search   search transcripts from the terminal (exact phrase, or --concept)
  lecture-recorder compact  give transcribed lectures a verified Opus copy and delete the original
                            (WAV, m4a, mp3, ...; setting must be on; dry run first, then you type
                            DELETE). `cleanup-wav` is the same command.
  lecture-recorder classes  list classes (a folder without class settings is marked)
  lecture-recorder delete <class> [<lecture>]  move a lecture, or a whole class, to Recently deleted
  lecture-recorder move <class> <lecture> <target-class>  move a lecture to another class
  lecture-recorder restore <id>  put an item from Recently deleted back where it was
  lecture-recorder trash list | empty  show Recently deleted, or empty it (you type DELETE)
"""
import argparse
import sys
import json

from lecture_recorder import __version__
from lecture_recorder.runtime_env import app_home, shown



def doctor() -> int:
    """Checks everything a new install needs, one line each, with what to do. Never downloads or writes."""
    import os

    ok = True

    def line(good, text: str, optional: bool = False) -> None:
        nonlocal ok
        ok &= bool(good) or optional
        print(f"[{'ok' if good else ('--' if optional else '!!')}] {text}")

    home = app_home()
    print(f"lecture-recorder {__version__} | home: {shown(home)}")
    from lecture_recorder.host import host

    here, setup = host(), host().setup_command
    line(sys.version_info[:2] == (3, 13), f"Python {sys.version.split()[0]} (3.13 required: run {setup} with Python 3.13)")
    if here.name == "windows":  # Windows' 260-character path limit
        line(len(str(home)) <= 100, f"Folder path length {len(str(home))} (setup needs <= 100 characters; spaces and accents are fine)."
             f" If it is too long, move this folder to a shorter path (for example into your Documents folder) and run {setup} again")
        line("onedrive" not in str(home).lower(), "This folder is not inside OneDrive (recordings and models are large and OneDrive would try "
             f"to upload them: move the folder outside OneDrive, e.g. to %LOCALAPPDATA%\\LectureRecorder, and run {setup} again)", optional=True)
        build = here.facts()["build"]
        line(build >= 19041, f"Windows build {build} ({'Windows 11' if build >= 22000 else 'Windows 10'})"
             + ("" if build >= 19041 else ": too old, install Windows Update (Windows 10 version 2004 or later is needed)"))
    elif here.name == "macos":
        facts = here.facts()
        line(facts.get("arch") == "arm64", f"Mac: {facts.get('chip') or 'unknown chip'}, {facts.get('arch')}"
             + ("" if facts.get("arch") == "arm64" else ": only Apple Silicon (M1 or newer) is supported"))
        major = int((facts.get("macos") or "0").split(".")[0] or 0)
        line(major >= 14, f"macOS {facts.get('macos') or 'unknown'}" + ("" if major >= 14 else ": macOS 14 (Sonoma) or newer is needed"))
        line(not facts.get("rosetta"), "Python runs natively on Apple Silicon"
             + (" (it is running under Rosetta: install the arm64 Python 3.13 from python.org)" if facts.get("rosetta") else ""), optional=True)
        line(True, "Microphone: macOS asks the first time you record. If you chose Don't Allow, switch it on in System Settings, "
             "Privacy & Security, Microphone")
    try:
        from lecture_recorder.config import load_config

        cfg = load_config(home)
    except Exception as e:  # a broken config.toml: say which and stop (nothing else can be checked reliably)
        line(False, f"config.toml could not be read: {e}. Fix it, or rename it and run {setup} to get a fresh one.")
        print("Some checks failed. See docs/troubleshooting.md.")
        return 1

    import importlib.util

    from lecture_recorder.autoconfig import (APPROVED_OLLAMA_VERSION, AUTO_FILE, DOWNLOAD_HINTS, OLLAMA_MODELS, describe,
                                             detect, read_auto, user_overrides)

    found = detect(cfg.home, cfg.models_dir, cfg.llm.ollama_host)  # nvidia-smi + CTranslate2 probe; no torch import
    line(importlib.util.find_spec("torch") is not None, f"PyTorch installed (run {setup} again if not)")
    gpu = found["gpu"] or {}
    ram = found.get("ram_gb")
    if ram is not None and ram < 8:
        line(False, f"{ram} GB of RAM: transcription and search work, but summaries (a 6 GB model) will be very slow or may fail; "
             "8 GB or more is needed, 16 GB recommended", optional=True)
    free = found.get("disk_free_gb")
    if free is not None and free < 20:
        line(False, f"Only {free} GB free on this drive: models need about 12 GB and recordings grow over time; free up space or "
             "move this folder to a bigger drive", optional=True)
    if gpu and found["cuda_usable"]:
        line(True, f"GPU: {gpu['name']} ({gpu['vram_mb']} MB), CUDA usable")
    elif gpu:
        line(True, f"GPU {gpu['name']} found but CUDA is not usable (update the NVIDIA driver): CPU path")
    else:
        line(True, "No NVIDIA GPU: CPU path (same speech model, several times slower; AMD and Intel graphics are not used)")
    for name, present in found["models"].items():
        hint = "" if present else f" missing: run  {DOWNLOAD_HINTS.get(name, DOWNLOAD_HINTS['large-v3-turbo'])}"
        line(present, f"Model {name}{hint}", optional=name == "distil-large-v3.5")  # distil: optional fallback
    if not found["models"]["pyannote"]:
        try:
            from dotenv import dotenv_values

            has_token = bool((dotenv_values(cfg.home / ".env").get("HF_TOKEN") or "").strip())
        except Exception:
            has_token = False
        line(has_token, "HF_TOKEN in .env (your own free Hugging Face Read token, needed once to download the speaker model; "
             f"see docs/installation.md, \"Hugging Face token\"; {here.installer_name} asks for it for you)")
    oll = found["ollama"]
    if oll["reachable"]:
        ver = oll.get("version")
        if ver == APPROVED_OLLAMA_VERSION:
            line(True, f"Ollama {ver} (the approved version)")
        else:  # a warning, not a failure: summaries still work, but this version has not been tested with the app
            line(False, f"Ollama {ver or 'version unknown'} is not the approved version {APPROVED_OLLAMA_VERSION}"
                 ": not tested with the app (docs/dependencies.md). Turn off Ollama's auto-update", optional=True)
        for m in OLLAMA_MODELS:
            line(m in oll["models"], f"Ollama model {m}" + ("" if m in oll["models"] else f" missing: run  ollama pull {m}"))
    elif oll["installed"]:
        line(False, f"Ollama is installed but not running at {cfg.llm.ollama_host}: {here.ollama_start_hint}")
    else:
        line(False, f"Ollama not found: {DOWNLOAD_HINTS['ollama']}")

    data = cfg.data_dir
    parent = next((q for q in (data, *data.parents) if q.exists()), None)
    line(parent is not None and os.access(parent, os.W_OK),
         f"Data folder {shown(data)} {'exists' if data.exists() else 'will be created'} and is writable")
    from lecture_recorder.web.app import port_state

    try:
        state = port_state(cfg.server.host, cfg.server.port)
    except ValueError as e:  # a non-loopback host in config.toml: refused, never probed
        state = "busy"
        line(False, f"[server] host: {e}")
    line(state != "busy", f"Port {cfg.server.port}: " + {"free": "free", "this app": "Lecture Recorder is already running there",
                                                         "busy": "used by another program (set [server] port in config.toml)"}[state])
    picks = read_auto(home)
    if picks:
        line(True, f"Settings for this computer ({AUTO_FILE}):")
        print("\n".join(describe({"settings": picks, "overrides": user_overrides(home), "notes": [], "missing": []})))
    else:
        line(False, f"Not configured for this computer yet: run  {here.venv_python} -m lecture_recorder.cli configure"
             " (until then each job picks settings itself)", optional=True)
    if ok:
        print(f"All checks passed. Next: {here.start_text} to start the app.")
    else:
        print("Some checks failed. Each [!!] line above says what to do; then run doctor again. "
              f"{here.installer_name} repeats every install step and skips the ones that are done. See docs/troubleshooting.md.")
    return 0 if ok else 1


def configure(dry_run: bool = False, as_json: bool = False) -> int:
    import time

    from lecture_recorder.autoconfig import configure as run, describe

    t0 = time.perf_counter()
    r = run(app_home(), dry_run=dry_run)
    if as_json:
        print(json.dumps({k: r[k] for k in ("settings", "overrides", "notes", "missing", "written")}))
        return 0
    gpu = r["profile"]["gpu"] or {}
    print(f"This computer: GPU {gpu.get('name', 'none')} {gpu.get('vram_mb', '')}{' MB' if gpu else ''}"
          f" (CUDA {'usable' if r['profile']['cuda_usable'] else 'not usable'}), {r['profile']['logical_cpus']} logical CPUs, "
          f"{r['profile']['ram_gb']} GB RAM, {r['profile']['disk_free_gb']} GB free, battery: {r['profile']['battery']}")
    print("Picks" + (" (dry run, nothing written):" if dry_run else f" (written to {shown(r['path'])}):"))
    print("\n".join(describe(r)))
    print(f"Detected in {time.perf_counter() - t0:.1f} s. Your own values in config.toml always win.")
    return 0


def devices() -> int:
    from lecture_recorder.recorder.sources import list_input_devices

    for d in list_input_devices():
        print(f"{'*' if d['is_default'] else ' '} {d['index']:>3}  {d['name']}")
    print("* = system default. Use: lecture-recorder record --device <number or name>")
    return 0


def apps() -> int:
    from lecture_recorder.host import host

    if why := host().app_capture_reason:
        print(why)
        return 2
    from lecture_recorder.recorder.apps import list_audio_apps

    rows = list_audio_apps()
    for a in rows:
        print(f"{a['pid']:>7}  {a['name']}  ({a['exe']}){'' if a['active'] else '  - not playing now'}")
    print("Use: lecture-recorder record --class <slug> --app <pid or name>" if rows else "No app has audio open.")
    return 0


def _pick_app(spec: str) -> dict | None:
    from lecture_recorder.recorder.apps import list_audio_apps

    rows = list_audio_apps()
    if spec.isdigit():
        return next((a for a in rows if a["pid"] == int(spec)), None)
    s = spec.lower()
    return next((a for a in rows if s in (a["name"].lower(), a["exe"].lower(), a["exe"].lower().removesuffix(".exe"))), None)         or next((a for a in rows if s in a["name"].lower() or s in a["exe"].lower()), None)


def import_zoom(args) -> int:
    """Import a Zoom recording folder: the audio-only m4a (else the mp4's audio), plus the transcript and
    chat if present. The original folder is only read, never changed."""
    from pathlib import Path

    from lecture_recorder import duplicates as dup
    from lecture_recorder import importer as imp
    from lecture_recorder import zoom
    from lecture_recorder.config import load_config
    from lecture_recorder.library import real_class

    cfg = load_config()
    if args.data_dir:
        cfg = _with_data_dir(cfg, args.data_dir)
    folder = Path(args.folder)
    if not folder.is_dir() or not real_class(cfg.data_dir, args.class_slug):
        print("Give an existing Zoom recording folder and an existing class (see `classes`).")
        return 2
    files = [(f"{folder.name}/{p.relative_to(folder).as_posix()}", p.stat().st_size) for p in folder.rglob("*") if p.is_file()]
    plan = zoom.plan(files)
    if plan["error"]:
        print(plan["error"])
        return 2
    src = lambda rel: folder.parent / rel  # noqa: E731
    size = src(plan["audio"]).stat().st_size
    sha = dup.file_sha256(src(plan["audio"]))
    seen = None if args.anyway else dup.find_source(cfg.data_dir, sha, Path(plan["audio"]).name, size)
    if seen:
        print(f"Already imported as {seen['title']} ({seen['class']}/{seen['session']}). Nothing was imported; "
              "use --anyway to import it again.")
        return 2
    tmps = {}
    try:
        video = Path(plan["audio"]).suffix.lower() in imp.VIDEO_EXTS  # read in place: only its audio is kept
        audio_tmp = src(plan["audio"]) if video else imp.copy_to_temp(src(plan["audio"]), cfg.data_dir, args.class_slug)
        if not video:
            tmps["audio"] = audio_tmp
        for key, name in (("transcript", zoom.TRANSCRIPT_FILE), ("chat", zoom.CHAT_FILE)):
            if plan[key] and src(plan[key]).stat().st_size <= zoom.MAX_TEXT_BYTES:
                tmps[name] = imp.copy_to_temp(src(plan[key]), cfg.data_dir, args.class_slug)
        d = imp.create_session(cfg.data_dir, args.class_slug, audio_tmp, Path(plan["audio"]).name,
                               size=size, when=plan["when"],
                               date_from="Zoom folder name" if plan["when"] else None, title=plan["title"],
                               source={"zoom": True, "folder": folder.name[:255], "sha256": sha},
                               extras={k: v for k, v in tmps.items() if k != "audio"}, keep_source=video,
                               template=cfg.naming.template)
    except imp.ImportFailed as e:
        print(e)
        return 2
    finally:
        for t in tmps.values():
            t.unlink(missing_ok=True)
    print(f"Imported {plan['audio']} -> {d}")
    for key in ("transcript", "chat"):
        if plan[key]:
            print(f"  kept Zoom {key}: {plan[key]}")
    if plan["others"]:
        print(f"  The folder has {plan['others']} more recording file(s) (e.g. the recording was restarted); only the "
              "largest was imported. Import the others with the app's Import a recording.")
    if plan["participants"]:
        print(f"  {plan['participants']} per-participant audio file(s) were not imported (the mixed recording is used).")
    return 0


def _with_data_dir(cfg, data_dir):
    import dataclasses
    from pathlib import Path

    return dataclasses.replace(cfg, data_dir=Path(data_dir).resolve())


def recover() -> int:
    from lecture_recorder.config import load_config
    from lecture_recorder.recorder.control import recover_sessions

    done = recover_sessions(load_config().data_dir)
    for d in done:
        print(f"recovered {d}")
    print(f"{len(done)} session(s) recovered.")
    return 0


def record(args) -> int:
    """Terminal recorder: Enter = stop, p = pause, r = resume, b = bookmark."""
    from lecture_recorder.config import load_config
    from lecture_recorder.recorder.control import PERMISSION_REMINDER, recover_sessions, start_recording

    from lecture_recorder.recorder.control import RecorderError

    cfg = load_config()
    for d in recover_sessions(cfg.data_dir):
        print(f"recovered unfinished session {d.name}")
    print(PERMISSION_REMINDER)
    app = None
    if args.app:
        from lecture_recorder.host import host

        if why := host().app_capture_reason:
            print(why)
            return 2
        app = _pick_app(args.app)
        if app is None:
            print(f"No app matching {args.app!r} has audio open. `lecture-recorder apps` lists them.")
            return 2
        print(f"Recording {app['name']}'s audio (the system volume doesn't matter; keep the app's own volume up "
              f"and unmuted in Windows' Volume Mixer).{' Your microphone is mixed in.' if args.with_mic else ''}")
    try:
        h = start_recording(args.class_slug, config=cfg, device=args.device, app=app, include_mic=args.with_mic,
                            max_length_minutes=args.max_minutes)
    except (RecorderError, ValueError) as e:
        print(e)
        return 2
    print(f"● REC  {h.session_dir}")
    print("Commands: p = pause, r = resume, b = bookmark, Enter or q = stop")
    try:
        while h.is_running():
            line = input().strip().lower()
            if line in ("", "q"):
                break
            if line == "p":
                h.pause()
            elif line == "r":
                h.resume()
            elif line == "b":
                h.bookmark()
            s = h.status()
            print(f"{'● REC' if s.get('state') == 'recording' else '❚❚ ' + str(s.get('state'))}  {s.get('recorded_s', 0):.0f}s  ({s.get('reason')})"
                  f"  level {s.get('level_dbfs', -120):.0f} dBFS" + ("  ⚠ No audio detected — check your microphone" if s.get("no_audio") else ""))
    except (EOFError, KeyboardInterrupt):
        pass
    meta = h.stop()
    print(f"Saved {h.session_dir / 'audio.wav'} ({meta.get('recorded_s', 0):.0f}s, stop: {meta.get('stop_reason')})")
    return 0


def process(args, kind: str = "transcribe") -> int:
    """Queue sessions and run the worker until the queue is empty (one GPU job at a time).
    A finished transcription queues its summary, which this same run also processes."""
    from pathlib import Path

    from lecture_recorder.config import load_config
    from lecture_recorder.pipeline.queue import enqueue, kind_of, run_worker

    from lecture_recorder.library import has_audio
    from lecture_recorder.recorder.session import SESSION_NAME_RE

    cfg = load_config()
    bad = [d for d in args.sessions if not (SESSION_NAME_RE.match(Path(d).name) and Path(d).is_dir()
                                            and (kind != "transcribe" or has_audio(Path(d))))]
    if bad:
        print("Not a session folder" + ("" if kind != "transcribe" else " with audio") + ": " + ", ".join(bad))
        return 2
    for d in args.sessions:
        if getattr(args, "anyway", False):
            from lecture_recorder.library import transcribe_anyway

            transcribe_anyway(Path(d))
        job = enqueue(cfg.data_dir, Path(d), kind)
        print(f"queued {kind_of(job)} {job['session']} ({job['state']})")
    blocked: list[str] = []
    done = run_worker(cfg.data_dir, blocked=blocked, keep_awake=lambda: cfg.processing.keep_awake)
    if done is None:
        print("Another worker is already processing the queue; your sessions will be picked up by it.")
        return 0
    for job in done:
        print(f"{job['state']}: {kind_of(job)} {job['session']}")
        if job.get("error"):
            print("  " + job["error"].strip().splitlines()[-1])
    if blocked:
        print(f"Waiting: {blocked[0]}. The remaining jobs stay queued; run this again when that has changed.")
        return 3
    return 0 if all(j["state"] == "done" for j in done) else 1


def _cfg(data_dir=None):
    import dataclasses
    from pathlib import Path

    from lecture_recorder.config import load_config

    cfg = load_config()
    return dataclasses.replace(cfg, data_dir=Path(data_dir).resolve()) if data_dir else cfg


def reindex(args) -> int:
    from lecture_recorder.search.index import SearchIndex

    cfg = _cfg(args.data_dir)
    idx = SearchIndex(cfg.data_dir, cfg)
    stats = idx.rebuild()
    from lecture_recorder.duplicates import sketch_scope
    from lecture_recorder.library import UNSORTED

    sketch_scope(cfg.data_dir, UNSORTED)  # Unsorted's scope is every class: all lectures' duplicate sketches
    st = idx.concept_status()
    print(f"Index rebuilt from files: {stats['added']} session(s). Concept search ready for {st['indexed']}, "
          f"missing or stale for {st['pending']} ({st['windows']} windows).")
    if args.embed and st["pending"]:
        from lecture_recorder.pipeline.queue import EMBED, enqueue, kind_of, run_worker

        for d in idx.pending_embedding():
            enqueue(cfg.data_dir, d, EMBED)
        blocked: list[str] = []
        done = run_worker(cfg.data_dir, blocked=blocked, keep_awake=lambda: cfg.processing.keep_awake)
        if done is None:
            print("Another worker is already processing the queue; it will index them.")
            return 0
        for job in done:
            print(f"{job['state']}: {kind_of(job)} {job['session']}")
        idx.sync()
        if blocked:
            print(f"Waiting: {blocked[0]}. The remaining jobs stay queued.")
            return 3
        return 0 if all(j["state"] == "done" for j in done) else 1
    return 0


def search(args) -> int:
    from lecture_recorder.pipeline.outputs import hms
    from lecture_recorder.search.index import SearchIndex

    cfg = _cfg(args.data_dir)
    idx = SearchIndex(cfg.data_dir, cfg)
    if not args.concept:
        r = idx.exact(args.query, args.class_slug)
        for x in r["results"]:
            c = x["context"]
            print(f"{x['class']}/{x['session']}  {hms(x['start'])}  {c['before']}[{c['match']}]{c['after']}")
        print(r.get("message") or f"{r['total']} match(es) in {r['sessions']} session(s).")
        return 0
    from lecture_recorder.pipeline.gpu import GpuBusy
    from lecture_recorder.search.embed import embed_query

    if not idx.concept_status()["windows"]:
        print("No lectures are indexed for concept search yet (lecture-recorder reindex --embed).")
        return 0
    from lecture_recorder.machine import gpu_block_reason

    blocked = gpu_block_reason(cfg.data_dir)
    if blocked:
        print(f"Search by meaning is paused ({blocked}): it needs the GPU. Exact search works now.")
        return 3
    try:
        vec = embed_query(args.query, cfg)
    except GpuBusy:
        print("The GPU is busy with a processing job; try again shortly (exact search works now).")
        return 3
    for x in idx.concept(vec, args.class_slug):
        print(f"#{x['rank']} {x['score']:.3f}  {x['class']}/{x['session']}  {hms(x['start'])}-{hms(x['end'])}  "
              f"{x['text'][:160]}")
    return 0


def lower_own_priority() -> None:
    """This process only (no system setting): the host's below-normal priority."""
    from lecture_recorder.host import host

    host().lower_own_priority()


def compact(args) -> int:
    """Dry run first; compact only after the user types DELETE (and only if the setting is on)."""
    from lecture_recorder.retention import compact as run, compact_plan

    cfg = _cfg(args.data_dir)
    if not cfg.retention.delete_original_after_opus:
        print("Deleting originals is off. Turn on \"Delete the original after a verified Opus copy\" in the app's "
              "settings (or [retention] delete_original_after_opus = true in config.toml) first.")
        return 2
    mb = lambda b: f"{b / 2**20:.1f} MB"  # noqa: E731
    plan = compact_plan(cfg.data_dir)
    for x in plan["items"]:
        print(f"{x['action']:<16}{x['session']}/{x['file']}  {mb(x['bytes_before'])} -> {mb(x['bytes_after'])}"
              + (" (estimate)" if x["action"] == "encode" else ""))
    for x in plan["skipped"]:
        print(f"{'keep':<16}{x['session']}  ({x['reason']})")
    print(f"Dry run: {len(plan['items'])} lecture(s), {mb(plan['bytes_before'])} now, about {mb(plan['bytes_after'])} "
          f"after; {len(plan['skipped'])} kept. Nothing has been changed.")
    if args.dry_run or not plan["items"]:
        return 0
    try:
        answer = input("Deleting the originals is permanent. Type DELETE to compact these lectures: ")
    except EOFError:
        answer = ""
    if answer.strip() != "DELETE":
        print("Cancelled. Nothing was deleted.")
        return 1
    lower_own_priority()  # minutes of encoding on one core: below normal, like the queue's jobs
    r = run(cfg.data_dir, plan["token"])
    for x in r["kept"]:
        print(f"{'kept':<16}{x['session']}  ({x['reason']})")
    print(f"Compacted {len(r['done'])} lecture(s): {mb(r['bytes_before'])} -> {mb(r['bytes_after'])}.")
    return 0


def classes(args) -> int:
    from lecture_recorder.library import list_classes

    cs = list_classes(_cfg(args.data_dir).data_dir)
    for c in cs:
        mark = "" if c["has_class_file"] else "  (folder without class settings: register it in the app)"
        print(f"{c['slug']:<24} {c['name']}  ({c['sessions']} lecture(s)){mark}")
    if not cs:
        print("No classes yet. Add one in the app (Library -> Add class).")
    return 0


def delete(args) -> int:
    """Move a lecture or a class to Recently deleted (restorable; nothing is deleted for good)."""
    from lecture_recorder import trash
    from lecture_recorder.search.index import SearchIndex

    cfg = _cfg(args.data_dir)
    idx = SearchIndex(cfg.data_dir, cfg)
    try:
        if args.lecture:
            r = trash.delete_session(cfg.data_dir, args.class_slug, args.lecture, index=idx)
        else:
            r = trash.delete_class(cfg.data_dir, args.class_slug, index=idx)
    except trash.TrashError as e:
        print(e)
        return 2
    what = f"lecture {r['title']}" if r["kind"] == "session" else f"class {r['title']} ({r['sessions']} lecture(s))"
    print(f"Moved {what} to Recently deleted. Undo with:  lecture-recorder restore {r['id']}")
    return 0


def move(args) -> int:
    from lecture_recorder import trash
    from lecture_recorder.move import move_session
    from lecture_recorder.search.index import SearchIndex

    cfg = _cfg(args.data_dir)
    try:
        r = move_session(cfg.data_dir, args.class_slug, args.lecture, args.target, prefer=args.name,
                         index=SearchIndex(cfg.data_dir, cfg))
    except trash.TrashError as e:
        print(e)
        return 2
    back = "" if r["session"] == r["from_session"] else f" --name {r['from_session']}"
    print(f"Moved {r['title']} to {r['class_name']} ({r['class']}/{r['session']}). Undo with:  lecture-recorder move "
          f"{r['class']} {r['session']} {r['from_class']}{back}")
    return 0


def restore(args) -> int:
    from lecture_recorder import trash
    from lecture_recorder.search.index import SearchIndex

    cfg = _cfg(args.data_dir)
    try:
        r = trash.restore(cfg.data_dir, args.id, index=SearchIndex(cfg.data_dir, cfg))
    except trash.TrashError as e:
        print(e)
        return 2
    print(f"Restored {r['class']}" + (f"/{r['session']}" if r["session"] else "")
          + (f"; {r['requeued']} job(s) queued again" if r["requeued"] else "") + ".")
    return 0


def trash_cmd(args) -> int:
    from lecture_recorder import trash

    cfg = _cfg(args.data_dir)
    items = trash.list_trash(cfg.data_dir)
    for r in items:
        what = (f"lecture  {r['class']}/{r['session']}  {r['title']}" if r["kind"] == "session"
                else f"class    {r['class']}  {r['title']} ({r['sessions']} lecture(s))")
        print(f"{r['id']}  {r['deleted'][:16].replace('T', ' ')}  {what}" + ("  [original place taken]" if r["occupied"] else ""))
    print(f"{len(items)} item(s) in Recently deleted.")
    if args.action == "list" or not items:
        return 0
    try:
        answer = input(f"Permanently delete these {len(items)} item(s)? This cannot be undone. Type DELETE: ")
    except EOFError:
        answer = ""
    if answer.strip() != "DELETE":
        print("Cancelled. Nothing was deleted.")
        return 1
    print(f"Permanently deleted {trash.empty_trash(cfg.data_dir, [r['id'] for r in items])} item(s).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lecture-recorder", description="Local lecture recorder.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check the installation")
    conf = sub.add_parser("configure", help="detect this computer and pick processing settings (config.auto.toml)")
    conf.add_argument("--dry-run", action="store_true", help="only show the picks; write nothing")
    conf.add_argument("--json", action="store_true", help=argparse.SUPPRESS)  # for the app's child process
    srv = sub.add_parser("serve", help="start the local app")
    srv.add_argument("--port", type=int, default=None, help="port (default from config.toml, 8765)")
    srv.add_argument("--data-dir", default=None, help="use another data folder (e.g. for a demo)")
    srv.add_argument("--no-browser", action="store_true", help="don't open the browser")
    sub.add_parser("devices", help="list microphones")
    sub.add_parser("recover", help="save recordings left unfinished by a crash")
    rec = sub.add_parser("record", help="record from the terminal")
    rec.add_argument("--class", dest="class_slug", required=True, help="an existing class (see `classes`)")
    rec.add_argument("--device", default=None, help="microphone number or name (see `devices`)")
    rec.add_argument("--max-minutes", type=float, default=None, help="auto-stop length (default 120)")
    rec.add_argument("--app", default=None, help="record this app's audio instead of a microphone (pid or name, see `apps`)")
    rec.add_argument("--with-mic", action="store_true", help="with --app: also record your microphone")
    sub.add_parser("apps", help="list apps playing audio (for record --app)")
    iz = sub.add_parser("import-zoom", help="import a Zoom recording folder (local or cloud download)")
    iz.add_argument("folder")
    iz.add_argument("--class", dest="class_slug", required=True, help="an existing class (see `classes`)")
    iz.add_argument("--data-dir", default=None, help="use another data folder")
    iz.add_argument("--anyway", action="store_true", help="import even if this file was imported before")
    proc = sub.add_parser("process", help="transcribe and label speakers for session folders")
    proc.add_argument("sessions", nargs="+", help="session folder(s), e.g. data/classes/<class>/sessions/<name>")
    proc.add_argument("--anyway", action="store_true", help="transcribe even if flagged as having no usable audio")
    summ = sub.add_parser("summarize", help="summarize transcribed session folders (local LLM)")
    summ.add_argument("sessions", nargs="+", help="session folder(s)")
    rei = sub.add_parser("reindex", help="rebuild the search index from the session folders")
    rei.add_argument("--embed", action="store_true", help="also compute missing concept-search vectors (GPU queue)")
    rei.add_argument("--data-dir", default=None, help="use another data folder")
    se = sub.add_parser("search", help="search transcripts (exact phrase, or --concept)")
    se.add_argument("query")
    se.add_argument("--concept", action="store_true", help="search by meaning instead of exact words")
    se.add_argument("--class", dest="class_slug", default=None, help="only this class")
    se.add_argument("--data-dir", default=None, help="use another data folder")
    cw = sub.add_parser("compact", aliases=["cleanup-wav"],
                        help="verified Opus copies for transcribed lectures, then delete the originals (dry run, then confirm)")
    cw.add_argument("--dry-run", action="store_true", help="only list what would be deleted")
    cw.add_argument("--data-dir", default=None, help="use another data folder")
    cl = sub.add_parser("classes", help="list classes")
    cl.add_argument("--data-dir", default=None, help="use another data folder")
    de = sub.add_parser("delete", help="move a lecture (or a whole class) to Recently deleted")
    de.add_argument("class_slug", metavar="class")
    de.add_argument("lecture", nargs="?", default=None, help="lecture folder name; omit to delete the whole class")
    de.add_argument("--data-dir", default=None, help="use another data folder")
    mv = sub.add_parser("move", help="move a lecture to another class")
    mv.add_argument("class_slug", metavar="class")
    mv.add_argument("lecture", help="lecture folder name")
    mv.add_argument("target", metavar="target-class")
    mv.add_argument("--name", default=None, help="folder name to use in the target class, if free")
    mv.add_argument("--data-dir", default=None, help="use another data folder")
    rs = sub.add_parser("restore", help="restore an item from Recently deleted")
    rs.add_argument("id", help="the item id (see `trash list`)")
    rs.add_argument("--data-dir", default=None, help="use another data folder")
    tr = sub.add_parser("trash", help="Recently deleted: list, or empty it for good (typed confirmation)")
    tr.add_argument("action", choices=("list", "empty"))
    tr.add_argument("--data-dir", default=None, help="use another data folder")
    args = parser.parse_args(argv)
    if args.cmd == "doctor":
        return doctor()
    if args.cmd == "configure":
        return configure(args.dry_run, args.json)
    if args.cmd == "devices":
        return devices()
    if args.cmd == "recover":
        return recover()
    if args.cmd == "record":
        if args.with_mic and not args.app:
            parser.error("--with-mic needs --app")
        return record(args)
    if args.cmd == "apps":
        return apps()
    if args.cmd == "import-zoom":
        return import_zoom(args)
    if args.cmd == "process":
        return process(args)
    if args.cmd == "summarize":
        return process(args, kind="summarize")
    if args.cmd == "serve":
        return serve(args)
    if args.cmd in ("compact", "cleanup-wav"):
        return compact(args)
    if args.cmd == "reindex":
        return reindex(args)
    if args.cmd == "search":
        return search(args)
    if args.cmd in ("classes", "delete", "move", "restore", "trash"):
        return {"classes": classes, "delete": delete, "move": move, "restore": restore, "trash": trash_cmd}[args.cmd](args)
    return 1


def run_configure_child(home=None, as_json: bool = False):
    """`configure` in a short child process, so the long-running server never loads the CUDA DLLs that
    the detection probes (measured +24 MB RSS in the server otherwise)."""
    import os
    import subprocess

    env = {**os.environ, "LECTURE_RECORDER_HOME": str(home or app_home())}  # the caller's install, explicitly
    r = subprocess.run([sys.executable, "-m", "lecture_recorder.cli", "configure"] + (["--json"] if as_json else []),
                       capture_output=as_json, text=True, timeout=120, env=env)
    if as_json:
        if r.returncode != 0:
            raise RuntimeError((r.stderr.strip().splitlines() or ["configure failed"])[-1])
        return json.loads(r.stdout.strip().splitlines()[-1])
    return r.returncode


def serve(args) -> int:
    import dataclasses
    from pathlib import Path

    from lecture_recorder.config import load_config
    from lecture_recorder.web.app import serve as run_server

    from lecture_recorder.autoconfig import AUTO_FILE

    if not (app_home() / AUTO_FILE).exists():  # first start: pick settings for this computer (~1 s)
        run_configure_child()
    cfg = load_config()
    if args.data_dir:
        cfg = dataclasses.replace(cfg, data_dir=Path(args.data_dir).resolve())
    if args.port:
        cfg = dataclasses.replace(cfg, server=dataclasses.replace(cfg.server, port=args.port))
    try:
        return run_server(cfg, open_browser=not args.no_browser)
    except ValueError as e:
        print(e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
