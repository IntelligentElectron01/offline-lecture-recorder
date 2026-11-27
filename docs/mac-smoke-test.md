# Mac smoke test (about 15 minutes)

For a person with an Apple Silicon Mac (M1 or newer, macOS 14 Sonoma or newer). The Mac port was written and unit-tested
without a Mac, so this checklist is how it gets its first real check. Tick each line; write down anything that differs,
with the exact message. Don't post a screenshot or log without removing your user name and folder names from it.

Use a real class folder only if you want to; the test needs just a minute of speech. Nothing here uploads anything.

## 1. Install (5-10 minutes of waiting)
- [ ] Python 3.13 for Apple Silicon is installed (python.org, "macOS 64-bit universal2 installer"). `python3.13 -c "import platform; print(platform.machine())"` prints `arm64`.
- [ ] In the project folder, `./setup.sh` finishes without an error and ends with the "Next steps" text. (If macOS blocks the scripts: `chmod +x setup.sh run.sh *.command` once.)
- [ ] `.venv/bin/python -m lecture_recorder.cli doctor` prints a "Mac: <chip>, arm64" line, a "macOS 14.x" line, and no `[!!]` for Python or the folder. Missing models and Ollama are expected at this point.
- [ ] `.venv/bin/python -m lecture_recorder.cli configure --dry-run` picks `device = "cpu"`, `compute_type = "int8"`, and a thread count no larger than your performance-core count.

## 2. Models and Ollama (long downloads)
- [ ] `.venv/bin/python scripts/download_models.py whisper` finishes with every file checksum-verified.
- [ ] Your own Hugging Face Read token is in `.env` and you accepted the model terms (see [installation](installation.md)); `.venv/bin/python scripts/download_models.py pyannote` finishes. Then blank the token in `.env`.
- [ ] `./setup.sh` offered to install Ollama; after you answered yes it printed a matching checksum and installed `Ollama.app` in `~/Applications` (no password asked). Open it once, turn OFF "Auto-download updates" in its menu-bar settings.
- [ ] `ollama pull qwen3:8b` and `ollama pull qwen3-embedding:0.6b` finish. If the `ollama` command is not found, tell us how you reached it (menu-bar app prompt, or the path inside the app). `doctor` now shows Ollama 0.35.0 and both models as `[ok]`.
- [ ] **Not checked by the project (it had no Mac): the Ollama download's Apple signature.** In Terminal: `codesign -dv --verbose=4 ~/Applications/Ollama.app` (the Authority lines should name "Ollama Inc."), `spctl -a -vv ~/Applications/Ollama.app` (should say "accepted" and "source=Notarized Developer ID"), `xcrun stapler validate ~/Applications/Ollama.app` (should say "The validate action worked!"). Report the output.

## 3. Start the app
- [ ] `./run.sh` (or double-click `Lecture Recorder.command`) opens the browser at 127.0.0.1:8765 with the Library page. The Terminal window stays open.
- [ ] Opening `./run.sh` again in another Terminal window opens the running app again (it does not start a second one).
- [ ] The **Record from** choice (microphone or app) is NOT shown: only the microphone is offered.

## 4. Record one minute
- [ ] Add a class, choose it, press **Start recording**. macOS asks to let Terminal (or Python) use the microphone: choose **Allow**.
- [ ] The REC bar shows, the level meter moves when you speak, and there is no "No audio detected" warning.
- [ ] Speak for about one minute, then **Stop**. The lecture appears marked **Not processed**, and a `.wav` exists under `data/classes/<class>/sessions/<date>/`.
- [ ] Optional: start another recording, and when macOS asks, choose **Don't Allow**. After about 10 seconds the REC bar should say "No audio detected" and mention System Settings, Privacy & Security, Microphone. (If macOS already remembers an earlier answer, reset it in that settings page.)

## 5. Import a file and process it
- [ ] **Import a recording** with a short `.m4a` (a Voice Memo copied to the Mac works) or a Zoom recording folder. It becomes its own lecture with a sensible date.
- [ ] Plugged in: press **Process now** on the lecture. It shows progress (transcribing, then speakers), then a transcript with Speaker 1 and the text you spoke. Note the time it took and the length of the audio.
- [ ] On battery, **Process now** waits and the page says so ("on battery"). (Skip if your Mac is a desktop.)
- [ ] The summary appears after the transcript (Ollama must be running).
- [ ] The lecture's folder now has `audio.opus` next to the original audio.

## 6. Search and the rest
- [ ] Search for a word you said: the lecture is listed with the context, and clicking it jumps to that moment in the audio.
- [ ] Concept search finds the lecture with words you did not say (needs the summary step to have finished and the vectors to be built: **Index them now**).
- [ ] **Delete** the test lecture, then restore it from **Recently deleted**. **Quit** in the app closes it; the Terminal window ends.

## 7. Report
Send back: the chip and macOS version, which lines failed (with the exact text), and the processing time against the audio length from step 5.
