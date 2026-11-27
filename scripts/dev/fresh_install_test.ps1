# Portability check: install from an EMPTY folder with a brand-new .venv, exactly as another user would.
# Usage: powershell -File scripts\dev\fresh_install_test.ps1 -Target <empty or non-existent folder>
# Copies only files git would ship (tracked + non-ignored); never .venv, data, models, .env.
param([Parameter(Mandatory = $true)][string]$Target)
$ErrorActionPreference = "Stop"
$src = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

if ((Test-Path $Target) -and (Get-ChildItem $Target -Force | Select-Object -First 1)) { throw "Target must be empty: $Target" }
New-Item -ItemType Directory -Force $Target | Out-Null
$Target = (Resolve-Path $Target).Path

Push-Location $src
$files = git ls-files --cached --others --exclude-standard
Pop-Location
foreach ($f in $files) {
    $dest = Join-Path $Target $f
    New-Item -ItemType Directory -Force (Split-Path -Parent $dest) | Out-Null
    Copy-Item (Join-Path $src $f) $dest
}
"[1] copied $($files.Count) shippable files into empty folder; .venv present: $(Test-Path "$Target\.venv")"

Push-Location $Target
try {
    git init -q
    cmd /c "`"$Target\setup.bat`"" | Select-Object -Last 2  # full path: cmd may not search the current dir
    if ($LASTEXITCODE -ne 0) { throw "setup.bat failed" }
    $venvPy = Join-Path $Target ".venv\Scripts\python.exe"
    "[2] setup.bat (normal install) ok; launcher: " + (& "$Target\.venv\Scripts\lecture-recorder.exe" --version)
    $where = & $venvPy -c "import lecture_recorder, pathlib; print(pathlib.Path(lecture_recorder.__file__).parent)"
    "[3] package imported from: $where"
    if ($where -notmatch "site-packages") { throw "not a normal (non-editable) install" }

    & $venvPy -m pip install --require-hashes --no-deps --only-binary=:all: -r requirements\dev-win-py313.lock --quiet
    if ($LASTEXITCODE -ne 0) { throw "dev lock install failed" }
    & $venvPy -m pytest -q -p no:cacheprovider -m "not models and not ollama and not private and not network"
    if ($LASTEXITCODE -ne 0) { throw "tests failed in fresh install" }

    $hits = Get-ChildItem $Target -Recurse -File -Exclude *.pyc | Where-Object { $_.FullName -notmatch '\\(\.venv|\.git)\\' } |
        Select-String -SimpleMatch -Pattern $src -List
    "[4] files in fresh copy referencing the original project path: $(@($hits).Count)"
    if (@($hits).Count) { $hits | ForEach-Object { "   $($_.Path)" }; throw "machine-specific path found" }
    "FRESH INSTALL TEST PASSED"
}
finally { Pop-Location }
