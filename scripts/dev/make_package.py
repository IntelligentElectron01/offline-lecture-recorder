"""Build dist/LectureRecorder-<version>.zip: the folder to give to another user.

Usage: .venv\\Scripts\\python.exe scripts\\dev\\make_package.py

Ships what git would (tracked + non-ignored files), minus developer-only files. Never includes
.venv, data, models, .env, config.toml or recordings (they are git-ignored). The zip has one
top folder, LectureRecorder/, so unzipping gives a short, clean path.
"""
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOP = "LectureRecorder"
# What the user zip contains: these root files, the program, the install scripts and the runtime locks.
ROOT_FILES = {"README.md", "LICENSE", "CHANGELOG.md", "CONTRIBUTING.md", "SECURITY.md", "pyproject.toml", "INSTALL.bat", "setup.bat", "setup.sh", "run.bat", "run.sh",
              "Lecture Recorder.command", "Setup.command", "config.example.toml", ".env.example"}
PREFIXES = ("src/", "requirements/", "docs/", "scripts/install.ps1", "scripts/download_models.py")


def wanted(f: str) -> bool:
    return (f in ROOT_FILES or f.startswith(PREFIXES)) and not f.startswith(("requirements/dev", "docs/real-clip"))


def main() -> int:
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, check=True, encoding="utf-8",
    ).stdout.splitlines()
    files = sorted(f for f in listed if wanted(f) and (ROOT / f).is_file())
    required = {"INSTALL.bat", "setup.bat", "run.bat", "README.md", "scripts/install.ps1", "scripts/download_models.py",
                "requirements/runtime-win-py313.lock", "pyproject.toml", ".env.example", "config.example.toml"}
    if missing := required - set(files):
        print(f"missing from package: {sorted(missing)}")
        return 1
    version = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]["version"]
    out = ROOT / "dist" / f"LectureRecorder-{version}.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(ROOT / f, f"{TOP}/{f}")
    print(f"{out.relative_to(ROOT)}: {len(files)} files, {out.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
