# Troubleshooting

- **First step for any problem:** `.venv\Scripts\lecture-recorder.exe doctor`. Every line says what to do.
- **"Python isn't 3.13":** install Python 3.13 (64-bit) from python.org, tick *Add python.exe to PATH*, delete `.venv`, and run `INSTALL.bat` again.
- **The folder path is too long:** move the folder to a shorter path (for example into `Documents`) and run `INSTALL.bat` again.
- **No GPU detected, or it is slow:** update the NVIDIA driver. Without an NVIDIA GPU the app uses the CPU path automatically; AMD and Intel graphics are not used.
- **Ollama not reachable:** start Ollama from the Start menu, then press **Summarize** (or **Try again**) on the lecture page.
- **"Not signed in" or a blank page:** close the tab and open the app with `run.bat` (or the Desktop shortcut).
- **Model hash mismatch:** don't use the files. Delete `models\quarantine` and try again later; if it repeats, report it.
- **The speaker model won't download:** check that you clicked *Agree and access repository* on the model page and that the token is a *Read* token.
- **Asking for help:** `doctor` shows your user folder as `%USERPROFILE%`, but other messages and log files can contain your user name and folder names. Replace those, and anything else personal, before you post them anywhere.
