# One-click installer for a Windows PC. Run via INSTALL.bat. Safe to re-run: finished steps are skipped.
# Steps: Python 3.13 -> app environment (setup.bat) -> speech models -> speaker model (needs a free
# Hugging Face token) -> Ollama + summary models -> check -> desktop shortcut.
# -CheckOnly: report what is already installed and change nothing.
param([switch]$CheckOnly)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$vpy = Join-Path $root ".venv\Scripts\python.exe"

# Pinned Ollama installer (see docs/dependencies.md).
$OllamaUrl = "https://github.com/ollama/ollama/releases/download/v0.35.0/OllamaSetup.exe"
$OllamaSha = "92e45c3518daf7d4ea60480896ee7bdb1bed71381593cfc9f2d7c1ceb941efde"
$OllamaModels = @("qwen3:8b", "qwen3-embedding:0.6b")

function Step($n, $text) { Write-Host ""; Write-Host "[$n/6] $text" -ForegroundColor Cyan }
function Ok($text) { Write-Host "  OK  $text" -ForegroundColor Green }
function Fail($text) { Write-Host ""; Write-Host "  STOPPED: $text" -ForegroundColor Red; Write-Host "  Fix that, then double-click INSTALL.bat again; finished steps are skipped."; exit 1 }
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}
# Run a probe command silently and return its exit code (Windows PowerShell 5.1 turns native stderr into
# terminating errors under $ErrorActionPreference = Stop, so relax it just here).
function Probe([scriptblock]$sb) {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { & $sb 2>$null | Out-Null; return $LASTEXITCODE } finally { $ErrorActionPreference = $old }
}
function Has-Python313 {
    if ((Get-Command py -ErrorAction SilentlyContinue) -and (Probe { py -3.13 -c "" }) -eq 0) { return $true }
    if ((Get-Command python -ErrorAction SilentlyContinue) -and (Probe { python -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,13) else 1)" }) -eq 0) { return $true }
    return $false
}
function Find-Ollama {
    $c = Get-Command ollama -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    $p = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path $p) { return $p }
    return $null
}
# Ollama updates itself by default, and a new version has not been checked by this project. The setting lives in
# Ollama's own tray menu, so the person does it; we say exactly where and wait for them.
function Ollama-Autoupdate-Step {
    Write-Host ""
    Write-Host "  ============================================================" -ForegroundColor Yellow
    Write-Host "  REQUIRED STEP: turn OFF Ollama's automatic updates" -ForegroundColor Yellow
    Write-Host "  ============================================================" -ForegroundColor Yellow
    Write-Host "   1. Find the Ollama icon (a llama) in the system tray, bottom right of the taskbar (click ^ to show hidden icons)."
    Write-Host "   2. Click it, then Settings."
    Write-Host "   3. Turn off 'Auto-download updates'. Also make sure 'Expose Ollama to the network' is OFF."
    Write-Host "  Why: this installer uses a specific, checksum-verified Ollama version; the app's check (doctor) warns on any other one."
    $null = Read-Host "  Press Enter when you have done this (you can change it later too)"
}
function Offer-Blank-Token {
    # The app never needs the Hugging Face token after the speaker model is downloaded.
    $envFile = Join-Path $root ".env"
    if (-not (Test-Path $envFile)) { return }
    if (-not (Select-String -Path $envFile -Pattern '^\s*HF_TOKEN\s*=\s*\S' -Quiet)) { return }
    Write-Host ""
    Write-Host "  Your Hugging Face token is saved in .env. The app does not need it any more (it works offline)."
    $a = Read-Host "  Remove it from .env now? (recommended) [Y/n]"
    if ($a -match '^(n|no)$') { Write-Host "  Kept. You can delete the value after HF_TOKEN= in .env yourself later."; return }
    $lines = @(Get-Content $envFile | ForEach-Object { if ($_ -match '^\s*HF_TOKEN\s*=') { "HF_TOKEN=" } else { $_ } })
    Set-Content -Path $envFile -Value $lines -Encoding ASCII
    Ok "token removed from .env (create a new one at huggingface.co/settings/tokens if you ever need to download again)"
}
function Ollama-Up { try { Invoke-WebRequest "http://127.0.0.1:11434" -UseBasicParsing -TimeoutSec 3 | Out-Null; return $true } catch { return $false } }

Write-Host "Lecture Recorder installer" -ForegroundColor White
Write-Host "Everything runs on this computer. This will install (only what is missing):"
Write-Host "  - Python 3.13 (via Windows' winget)"
Write-Host "  - the app's libraries, in a private folder inside this one"
Write-Host "  - speech and speaker models (~5 GB) and Ollama + 2 summary models (~6 GB), roughly 20-40 min on a fast connection"
Write-Host "  - a 'Lecture Recorder' shortcut on your Desktop"
Write-Host "You will be asked for ONE thing: your own free Hugging Face access token (explained when we get there)."
Write-Host "You will also be asked to turn off Ollama's automatic updates in its own settings (one minute)."
if (-not $CheckOnly) { Read-Host "Press Enter to start (or close this window to cancel)" | Out-Null }

# 0. Preflight: things that would otherwise fail 30 minutes in
if ($root.Length -gt 100) { Fail "This folder's path is too long ($root). Move the LectureRecorder folder somewhere with a shorter path (for example into your Documents folder, or into $env:LOCALAPPDATA), and run INSTALL.bat again." }
if ($root -match '\.zip\\|\\Temp\\|\\INetCache\\') { Fail "This folder looks like it is still inside the zip file or a temporary folder. Right-click the zip -> Extract All, move the LectureRecorder folder into your Documents folder, and run INSTALL.bat from there." }
$arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
if ($arch -ne "AMD64") { Fail "This needs a 64-bit Intel/AMD Windows PC (found: $arch). ARM and 32-bit Windows are not supported yet." }
if ([Environment]::OSVersion.Version.Build -lt 19041) { Fail "This needs Windows 10 (version 2004 or later) or Windows 11. Run Windows Update first." }
$freeGB = [math]::Round((Get-PSDrive -Name $root.Substring(0, 1)).Free / 1GB, 1)
if ($freeGB -lt 20) { Fail "Only $freeGB GB free on drive $($root.Substring(0, 1)):, and about 20 GB is needed. Free up space or move the folder to a bigger drive." }
if ($root -match '\\OneDrive') { Write-Host "  Warning: this folder is inside OneDrive. Recordings and models are large and OneDrive will try to upload them. A folder outside OneDrive is better, for example $env:LOCALAPPDATA\LectureRecorder (you can move this folder there and re-run INSTALL.bat)." -ForegroundColor Yellow }
if (-not $CheckOnly) {
    try { Invoke-WebRequest "https://huggingface.co" -UseBasicParsing -TimeoutSec 15 | Out-Null; Invoke-WebRequest "https://pypi.org" -UseBasicParsing -TimeoutSec 15 | Out-Null }
    catch { Fail "Can't reach the internet (huggingface.co / pypi.org). Check your connection, VPN or school/work firewall, then run INSTALL.bat again." }
}

# 1. Python
Step 1 "Python 3.13"
if (-not (Has-Python313)) {
    $local = Join-Path $env:LOCALAPPDATA "Programs\Python\Python313"
    if (Test-Path "$local\python.exe") { $env:Path = "$local;$local\Scripts;" + $env:Path }
}
if (Has-Python313) { Ok "found" }
elseif ($CheckOnly) { Write-Host "  missing" -ForegroundColor Yellow }
else {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { Fail "Python 3.13 is missing and winget is not available. Install Python 3.13 from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then run INSTALL.bat again." }
    winget install -e --id Python.Python.3.13 --scope user --accept-package-agreements --accept-source-agreements
    Refresh-Path
    $local = Join-Path $env:LOCALAPPDATA "Programs\Python\Python313"
    if (Test-Path "$local\python.exe") { $env:Path = "$local;$local\Scripts;" + $env:Path }
    if (-not (Has-Python313)) { Fail "Python 3.13 did not install. Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then run INSTALL.bat again." }
    Ok "installed"
}

# 2. App environment
Step 2 "App libraries (private .venv in this folder)"
$haveApp = (Test-Path $vpy) -and ((Probe { & $vpy -c "import lecture_recorder" }) -eq 0)
if ($haveApp) { Ok "already set up" }
elseif ($CheckOnly) { Write-Host "  missing" -ForegroundColor Yellow }
else {
    cmd /c "`"$root\setup.bat`""
    if ($LASTEXITCODE -ne 0) { Fail "setup.bat failed (see the messages above)." }
    Ok "installed"
}
if ($CheckOnly -and -not $haveApp) { Write-Host "Check only: stopping here (the remaining checks need the app libraries)."; exit 0 }

# 3. Speech models
Step 3 "Speech-to-text models (no account needed)"
$whisperDirs = "faster-whisper-large-v3-turbo", "faster-distil-large-v3.5"
if (-not ($whisperDirs | Where-Object { -not (Test-Path (Join-Path $root "models\$_\model.bin")) })) { Ok "already downloaded" }
elseif ($CheckOnly) { Write-Host "  missing" -ForegroundColor Yellow }
else {
    & $vpy scripts\download_models.py whisper
    if ($LASTEXITCODE -ne 0) { Fail "Speech model download failed. Check your internet connection and free disk space (~20 GB)." }
    Ok "downloaded and checksum-verified"
}

# 4. Speaker model (gated: needs a Hugging Face token)
Step 4 "Speaker-separation model (needs a free Hugging Face token)"
if (Test-Path (Join-Path $root "models\pyannote-community-1\config.yaml")) { Ok "already downloaded"; if (-not $CheckOnly) { Offer-Blank-Token } }
elseif ($CheckOnly) { Write-Host "  missing" -ForegroundColor Yellow }
else {
    $envFile = Join-Path $root ".env"
    $token = ""
    if (Test-Path $envFile) {
        $m = Select-String -Path $envFile -Pattern '^\s*HF_TOKEN\s*=\s*(\S+)' | Select-Object -First 1
        if ($m) { $token = $m.Matches[0].Groups[1].Value }
    }
    $opened = $false
    for ($try = 1; $try -le 3; $try++) {
        if (-not $token) {
            if (-not $opened) {
                Write-Host "  Do these once (a browser page will open):"
                Write-Host "   1. Sign in or sign up (free) at huggingface.co"
                Write-Host "   2. On the page that opens, click 'Agree and access repository'"
                Write-Host "   3. Go to huggingface.co/settings/tokens -> Create new token -> type 'Read' -> copy it"
                Start-Process "https://huggingface.co/pyannote/speaker-diarization-community-1"
                Start-Process "https://huggingface.co/settings/tokens"
                $opened = $true
            }
            $sec = Read-Host "  Paste the token here and press Enter (input is hidden)" -AsSecureString
            $token = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)).Trim()
            if ($token -notmatch '^hf_\S+$') { Write-Host "  That doesn't look like a token (they start with hf_). Try again." -ForegroundColor Yellow; $token = ""; continue }
            $lines = @(); if (Test-Path $envFile) { $lines = @(Get-Content $envFile | Where-Object { $_ -notmatch '^\s*HF_TOKEN\s*=' }) }
            Set-Content -Path $envFile -Value (@("HF_TOKEN=$token") + $lines) -Encoding ASCII
        }
        & $vpy scripts\download_models.py pyannote
        if ($LASTEXITCODE -eq 0) { Ok "downloaded and checksum-verified"; Offer-Blank-Token; break }
        if ($try -eq 3) { Fail "Speaker model download failed. Check that you clicked 'Agree and access repository' on the model page and that the token is a Read token." }
        Write-Host "  Download failed. Most often the 'Agree and access repository' button was not clicked, or the token is wrong." -ForegroundColor Yellow
        Read-Host "  Fix that, then press Enter to retry (or paste a new token after pressing Enter)" | Out-Null
        $token = ""
        $sec = $null
    }
}

# 5. Ollama
Step 5 "Ollama and summary models"
$ollama = Find-Ollama
if ($ollama) { Ok "Ollama found" }
elseif ($CheckOnly) { Write-Host "  Ollama missing (the installer will ask you to turn off its automatic updates)" -ForegroundColor Yellow }
else {
    $tmp = Join-Path $env:TEMP "OllamaSetup.exe"
    Write-Host "  Downloading Ollama (pinned version, checked against a known SHA-256)..."
    Invoke-WebRequest $OllamaUrl -OutFile $tmp -UseBasicParsing
    if ((Get-FileHash $tmp -Algorithm SHA256).Hash.ToLower() -ne $OllamaSha) { Remove-Item $tmp -Force; Fail "The Ollama installer's checksum did not match, so it was NOT run. Try again later." }
    $sig = Get-AuthenticodeSignature $tmp
    if ($sig.Status -ne "Valid" -or $sig.SignerCertificate.Subject -notmatch "O=Ollama Inc") { Remove-Item $tmp -Force; Fail "The Ollama installer's signature is not valid, so it was NOT run." }
    Write-Host "  Installing Ollama (a Windows prompt may ask permission)..."
    Start-Process $tmp -ArgumentList "/SILENT" -Wait
    Remove-Item $tmp -Force
    $ollama = Find-Ollama
    if (-not $ollama) { Fail "Ollama did not install. Install it from https://ollama.com/download and run INSTALL.bat again." }
    Ok "Ollama installed"
}
if ($ollama -and -not $CheckOnly) {
    Ollama-Autoupdate-Step
    if (-not (Ollama-Up)) {
        $app = Join-Path (Split-Path $ollama) "ollama app.exe"
        if (Test-Path $app) { Start-Process $app } else { Start-Process $ollama -ArgumentList "serve" -WindowStyle Hidden }
        for ($i = 0; $i -lt 30 -and -not (Ollama-Up); $i++) { Start-Sleep -Seconds 2 }
    }
    if (-not (Ollama-Up)) { Fail "Ollama is not responding. Open 'Ollama' from the Start menu, wait a few seconds, and run INSTALL.bat again." }
    foreach ($m in $OllamaModels) {
        Write-Host "  Pulling $m ..."
        & $ollama pull $m
        if ($LASTEXITCODE -ne 0) { Fail "Could not download model $m. Check your internet connection and disk space." }
    }
    Ok "summary models ready"
}

# 6. Check + shortcut
Step 6 "Final check"
& $vpy -m lecture_recorder.cli doctor
if (-not $CheckOnly) {
    try {
        $lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "Lecture Recorder.lnk"
        $s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
        $s.TargetPath = Join-Path $root "run.bat"; $s.WorkingDirectory = $root; $s.Save()
        Ok "Desktop shortcut 'Lecture Recorder' created"
    } catch { Write-Host "  (Could not create the shortcut; just double-click run.bat instead.)" }
    Write-Host ""
    Write-Host "All set. Double-click 'Lecture Recorder' on your Desktop (or run.bat) to start." -ForegroundColor Green
}
