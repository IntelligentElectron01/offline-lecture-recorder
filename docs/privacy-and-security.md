# Privacy and security

- **Everything stays on your computer.** The app makes no network calls at runtime other than to Ollama on `127.0.0.1`; telemetry is switched off in the libraries it uses, and Hugging Face is forced offline. Models are downloaded once, during installation, and checked against known checksums.
- **Only you can use the app.** It listens on `127.0.0.1` only, and every request needs the sign-in cookie that only the browser the app opened has. Another program on your computer cannot read your lectures through it. (Programs running as *you* can still read your files in `data\`; that is how any local file works.)
- **Your files:** everything you record lives in `data\` inside this folder as plain audio, text and Markdown. Back up `data\` to keep your lectures; moving the whole folder to another computer (after running setup there) keeps everything. Models are in `models\`, your own settings in `config.toml`, and your token (if you kept it) in `.env`.
- **Permission to record** is your responsibility (see the notice in the README and the Zoom notes in [usage](usage.md)).
