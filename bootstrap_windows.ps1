$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$LeaderHost = if ($env:LEADER_HOST) { $env:LEADER_HOST } else { "leader-macbook-pro.taila5426e.ts.net" }
$LeaderPort = if ($env:LEADER_PORT) { [int]$env:LEADER_PORT } else { 8787 }
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir    = Join-Path $ProjectDir ".venv"

if (-not $env:PIP_NO_CACHE_DIR) { $env:PIP_NO_CACHE_DIR = "1" }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Step {
    param([string]$Message)
    Write-Host "[bootstrap:windows] $Message"
}

function Refresh-Path {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath    = [Environment]::GetEnvironmentVariable("Path", "User")
    $parts = @($machinePath, $userPath) | Where-Object { $_ -and $_.Trim() }
    $env:Path = ($parts -join ";")
}

function Get-CommandPath {
    param([string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $cmd) { return $null }
    # NOTE: no ?? operator here — that requires PowerShell 7+
    if ($cmd.Source) { return $cmd.Source }
    if ($cmd.Path)   { return $cmd.Path }
    return $null
}

# ---------------------------------------------------------------------------
# Python spec helpers
# ---------------------------------------------------------------------------

function New-PythonSpec {
    param(
        [string]$Executable,
        [string[]]$LauncherArgs = @()
    )
    return [pscustomobject]@{ Executable = $Executable; LauncherArgs = $LauncherArgs }
}

function Test-PythonSpec {
    param([object]$PythonSpec)
    if (-not $PythonSpec) { return $false }

    $exe = $PythonSpec.Executable
    if (-not (Test-Path $exe -ErrorAction SilentlyContinue)) {
        $exe = Get-CommandPath $exe
    }
    if (-not $exe) { return $false }

    try {
        # f-strings need Python 3.6+ which is fine, but avoid them here for clarity
        $argList = @($PythonSpec.LauncherArgs) + @("-c", "import sys; print(str(sys.version_info.major) + '.' + str(sys.version_info.minor))")
        $version = (& "$exe" @argList 2>&1 | Select-Object -First 1)
        return ($LASTEXITCODE -eq 0 -and "$version".Trim() -eq "3.11")
    } catch {
        return $false
    }
}

# ---------------------------------------------------------------------------
# Safe execution
# ---------------------------------------------------------------------------

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )
    if (-not (Test-Path $FilePath -ErrorAction SilentlyContinue)) {
        throw "Executable not found: $FilePath"
    }
    $output = & "$FilePath" @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host ($output | Out-String)
        throw "Command failed (exit $LASTEXITCODE): $FilePath $($Arguments -join ' ')"
    }
    return $output
}

function Invoke-Python311 {
    param(
        [object]$PythonSpec,
        [string[]]$Arguments
    )
    if (-not (Test-PythonSpec $PythonSpec)) {
        throw "Python 3.11 spec is invalid. Close PowerShell, open a new window, and re-run the script."
    }
    $exe = $PythonSpec.Executable
    if (-not (Test-Path $exe -ErrorAction SilentlyContinue)) {
        $exe = Get-CommandPath $exe
    }
    $argList = @($PythonSpec.LauncherArgs) + $Arguments
    $output  = & "$exe" @argList 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host ($output | Out-String)
        throw "Python 3.11 command failed (exit $LASTEXITCODE)"
    }
    return $output
}

# ---------------------------------------------------------------------------
# Python 3.11 discovery
# ---------------------------------------------------------------------------

function Get-Python311RegistryCandidates {
    $paths = @(
        "HKCU:\Software\Python\PythonCore\3.11\InstallPath",
        "HKLM:\Software\Python\PythonCore\3.11\InstallPath",
        "HKLM:\Software\WOW6432Node\Python\PythonCore\3.11\InstallPath"
    )
    $candidates = @()
    foreach ($path in $paths) {
        $key = Get-Item -Path $path -ErrorAction SilentlyContinue
        if (-not $key) { continue }
        $executable  = [string]$key.GetValue("ExecutablePath")
        $installPath = [string]$key.GetValue("")
        if ($executable)  { $candidates += $executable }
        if ($installPath) { $candidates += (Join-Path $installPath "python.exe") }
    }
    return $candidates
}

function Get-Python311DirectoryCandidates {
    $roots = @(
        (Join-Path $env:LocalAppData "Programs\Python"),
        $env:ProgramFiles,
        ${env:ProgramFiles(x86)}
    ) | Where-Object { $_ -and (Test-Path $_ -ErrorAction SilentlyContinue) }

    $candidates = @()
    foreach ($root in $roots) {
        $dirs = Get-ChildItem -Path $root -Directory -Filter "Python311*" -ErrorAction SilentlyContinue
        foreach ($dir in $dirs) {
            $candidates += (Join-Path $dir.FullName "python.exe")
        }
    }
    return $candidates
}

function Find-Python311 {
    # 1. Prefer py launcher with -3.11
    $pyLaunchers = @(
        (Get-CommandPath "py.exe"),
        (Join-Path $env:SystemRoot "py.exe"),
        (Join-Path $env:LocalAppData "Programs\Python\Launcher\py.exe")
    ) | Where-Object { $_ -and (Test-Path $_ -ErrorAction SilentlyContinue) } | Select-Object -Unique

    foreach ($py in $pyLaunchers) {
        $spec = New-PythonSpec -Executable $py -LauncherArgs @("-3.11")
        if (Test-PythonSpec $spec) {
            Write-Step "Found Python 3.11 via py launcher: $py -3.11"
            return $spec
        }
    }

    # 2. Fall back to direct python executables
    $candidates = @(
        (Get-CommandPath "python3.11"),
        (Get-CommandPath "python.exe"),
        "$env:LocalAppData\Programs\Python\Python311\python.exe",
        "$env:ProgramFiles\Python311\python.exe",
        "${env:ProgramFiles(x86)}\Python311\python.exe"
    )
    $candidates += Get-Python311RegistryCandidates
    $candidates += Get-Python311DirectoryCandidates
    $candidates = $candidates |
        Where-Object { $_ -and (Test-Path $_ -ErrorAction SilentlyContinue) } |
        Where-Object { $_ -notlike "*\Microsoft\WindowsApps\*" } |
        Select-Object -Unique

    foreach ($candidate in $candidates) {
        $spec = New-PythonSpec -Executable $candidate
        if (Test-PythonSpec $spec) {
            Write-Step "Found Python 3.11: $candidate"
            return $spec
        }
    }

    return $null
}

function Ensure-Python311 {
    $python = Find-Python311
    if ($python) { return $python }

    Write-Step "Python 3.11 not found — installing via winget..."
    Ensure-Winget

    $winget = Get-CommandPath "winget.exe"
    if (-not $winget) { throw "winget not available. Install App Installer from the Microsoft Store." }

    & "$winget" install --id Python.Python.3.11 --exact --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) { throw "winget failed to install Python 3.11." }

    Refresh-Path
    Start-Sleep -Seconds 5

    $python = Find-Python311
    if ($python) { return $python }

    throw @"
Python 3.11 was installed but cannot be found in this PowerShell session.
ACTION REQUIRED: Close this window, open a NEW PowerShell window, and run the script again.
To verify first: py -3.11 --version
"@
}

# ---------------------------------------------------------------------------
# winget / tool helpers
# ---------------------------------------------------------------------------

function Ensure-Winget {
    if (Get-CommandPath "winget.exe") {
        Write-Step "winget already present"
        return
    }
    Write-Step "winget not found — installing Microsoft App Installer..."
    $installer = Join-Path $env:TEMP "Microsoft.DesktopAppInstaller.msixbundle"
    try {
        Invoke-WebRequest -Uri "https://aka.ms/getwinget" -OutFile $installer -UseBasicParsing
    } catch {
        throw "Failed to download App Installer: $_. Install it from the Microsoft Store manually."
    }
    Add-AppxPackage -Path $installer
    Refresh-Path
    if (-not (Get-CommandPath "winget.exe")) {
        throw "winget installation did not complete. Install App Installer from the Microsoft Store and re-run."
    }
}

function Ensure-WingetCommand {
    param(
        [string]$Command,
        [string]$PackageId
    )
    if (Get-CommandPath $Command) {
        Write-Step "$Command already present"
        return
    }
    Write-Step "Installing $PackageId via winget..."
    $winget = Get-CommandPath "winget.exe"
    & "$winget" install --id $PackageId --exact --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) { throw "winget failed to install $PackageId" }
    Refresh-Path
}

# ---------------------------------------------------------------------------
# Tailscale
# ---------------------------------------------------------------------------

function Find-Tailscale {
    $cmd = Get-CommandPath "tailscale.exe"
    if ($cmd) { return $cmd }
    $candidates = @(
        "$env:ProgramFiles\Tailscale\tailscale.exe",
        "${env:ProgramFiles(x86)}\Tailscale\tailscale.exe"
    ) | Where-Object { $_ -and (Test-Path $_ -ErrorAction SilentlyContinue) }
    if ($candidates.Count -gt 0) { return $candidates[0] }
    return $null
}

function Test-TailscaleRunning {
    param([string]$TailscaleExe)
    try {
        $status = & "$TailscaleExe" status --json 2>&1
        return (($status -join "`n") -match '"BackendState"\s*:\s*"Running"')
    } catch {
        return $false
    }
}

function Ensure-Tailscale {
    $tailscale = Find-Tailscale
    if (-not $tailscale) {
        Write-Step "Installing Tailscale via winget..."
        $winget = Get-CommandPath "winget.exe"
        & "$winget" install --id Tailscale.Tailscale --exact --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -ne 0) { throw "winget failed to install Tailscale." }
        Refresh-Path
        $tailscale = Find-Tailscale
    } else {
        Write-Step "Tailscale already installed: $tailscale"
    }

    if (-not $tailscale) {
        throw "Tailscale installed but tailscale.exe not found. Open a new PowerShell and re-run."
    }

    $svc = Get-Service -Name "Tailscale" -ErrorAction SilentlyContinue
    if ($svc -and $svc.Status -ne "Running") {
        Write-Step "Starting Tailscale service..."
        Start-Service -Name "Tailscale" -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 3
    }

    if (Test-TailscaleRunning $tailscale) {
        Write-Step "Tailscale is authenticated and running"
        return $tailscale
    }

    Write-Step "Tailscale needs authentication — opening browser..."
    $authOutput = (& "$tailscale" up --timeout=5s 2>&1 | Out-String)
    Write-Host $authOutput

    $match = [regex]::Match($authOutput, "https://\S+")
    if ($match.Success) { Start-Process $match.Value } else { & "$tailscale" up }

    Write-Step "Waiting for Tailscale authentication (complete it in your browser)..."
    $waited = 0
    while (-not (Test-TailscaleRunning $tailscale)) {
        Start-Sleep -Seconds 5
        $waited += 5
        if ($waited % 30 -eq 0) { Write-Step "Still waiting... ($waited seconds elapsed)" }
        if ($waited -ge 300)    { throw "Tailscale auth timed out after 5 minutes. Run 'tailscale up' manually then re-run this script." }
    }

    Write-Step "Tailscale authenticated"
    return $tailscale
}

# ---------------------------------------------------------------------------
# GPU status
# ---------------------------------------------------------------------------

function Write-GpuStatus {
    $nvidia = Get-CommandPath "nvidia-smi.exe"
    if (-not $nvidia) {
        Write-Step "No NVIDIA GPU detected — CPU-only PyTorch will be installed"
        return
    }
    $gpu = (& "$nvidia" --query-gpu=name --format=csv,noheader 2>&1 | Select-Object -First 1)
    Write-Step "NVIDIA GPU detected: $gpu"
    $cudaPath = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $cudaPath) {
        Write-Step "CUDA toolkit directory present"
    } else {
        Write-Step "CUDA toolkit not found — CPU-only PyTorch will be installed"
    }
}

# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------

function Ensure-Torch {
    param([string]$VenvPython)

    if ($env:SKIP_TORCH_INSTALL -eq "1") {
        Write-Step "Skipping PyTorch install (SKIP_TORCH_INSTALL=1)"
        return
    }

    $probe = & "$VenvPython" -c "import torch, torchvision; print(torch.__version__)" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Step "PyTorch already present: $($probe | Select-Object -First 1)"
        return
    }

    Write-Step "Upgrading pip before PyTorch install..."
    Invoke-Checked -FilePath $VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

    $nvidiaPresent = ($null -ne (Get-CommandPath "nvidia-smi.exe"))
    if ($nvidiaPresent) {
        Write-Step "Installing PyTorch with CUDA 12.1 index..."
        Invoke-Checked -FilePath $VenvPython -Arguments @(
            "-m", "pip", "install", "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        )
    } else {
        Write-Step "Installing PyTorch CPU-only..."
        Invoke-Checked -FilePath $VenvPython -Arguments @(
            "-m", "pip", "install", "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        )
    }

    $probe = & "$VenvPython" -c "import torch, torchvision; print(torch.__version__)" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host ($probe | Out-String)
        throw "PyTorch import failed after install. See error above."
    }
    Write-Step "PyTorch installed: $($probe | Select-Object -First 1)"
}

# ---------------------------------------------------------------------------
# Virtual environment
# ---------------------------------------------------------------------------

function Ensure-Venv {
    param([object]$PythonSpec)

    $venvPython = Join-Path $VenvDir "Scripts\python.exe"
    $reqFile    = Join-Path $ProjectDir "requirements.txt"

    if (-not (Test-Path $venvPython -ErrorAction SilentlyContinue)) {
        Write-Step "Creating virtual environment at $VenvDir..."
        Invoke-Python311 -PythonSpec $PythonSpec -Arguments @("-m", "venv", $VenvDir)
        if (-not (Test-Path $venvPython -ErrorAction SilentlyContinue)) {
            throw "Venv creation failed — $venvPython not found after venv step."
        }
    } else {
        Write-Step "Virtual environment already exists"
    }

    Write-Step "Upgrading pip in venv..."
    Invoke-Checked -FilePath $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

    if (Test-Path $reqFile) {
        Write-Step "Installing requirements.txt..."
        Invoke-Checked -FilePath $venvPython -Arguments @("-m", "pip", "install", "-r", $reqFile)
    } else {
        Write-Step "WARNING: requirements.txt not found at $reqFile — skipping"
    }

    Ensure-Torch -VenvPython $venvPython

    Write-Step "Installing project package (editable)..."
    Invoke-Checked -FilePath $venvPython -Arguments @("-m", "pip", "install", "-e", $ProjectDir)

    Write-Step "Verifying dml_cluster imports..."
    $check = & "$venvPython" -c "import dml_cluster.hardware, dml_cluster.worker; print('OK')" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host ($check | Out-String)
        throw "Import verification failed — see error above."
    }
    Write-Step "Import verification passed"

    return $venvPython
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

function Main {
    Set-Location $ProjectDir
    Write-Step "Project directory : $ProjectDir"
    Write-Step "Leader target     : ${LeaderHost}:${LeaderPort}"

    Ensure-Winget
    Ensure-WingetCommand "curl.exe" "cURL.cURL"
    Ensure-WingetCommand "git.exe"  "Git.Git"

    $python     = Ensure-Python311
    $displayExe = ($python.Executable + " " + ($python.LauncherArgs -join " ")).Trim()
    Write-Step "Using Python 3.11 : $displayExe"

    Ensure-Tailscale | Out-Null
    Write-GpuStatus

    $venvPython = Ensure-Venv -PythonSpec $python

    Write-Step "Detected hardware:"
    Invoke-Checked -FilePath $venvPython -Arguments @("-m", "dml_cluster.hardware")

    Write-Step "Starting worker..."
    & "$venvPython" -m dml_cluster.worker --leader $LeaderHost --port $LeaderPort --project-dir $ProjectDir
}

Main