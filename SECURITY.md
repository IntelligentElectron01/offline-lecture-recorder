# Security policy

## Reporting a vulnerability

Please report security problems privately through GitHub: open the repository's **Security** tab and choose **Report a vulnerability** (GitHub private vulnerability reporting). Do not open a public issue for a vulnerability.

Include what you found, how to reproduce it and which version you tested. Do not include real recordings, transcripts or tokens.

## Scope

The app is meant to run on your own computer only. It listens on `127.0.0.1`, requires a sign-in cookie that only the browser it opened holds, makes no network calls at run time other than to Ollama on loopback, and verifies every model download and dependency against pinned hashes. Reports about any way around those properties are in scope. See [docs/privacy-and-security.md](docs/privacy-and-security.md).

## Supported versions

Only the latest release is supported.
