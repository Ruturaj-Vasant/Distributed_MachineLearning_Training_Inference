$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$LeaderHost = if ($env:LEADER_HOST) { $env:LEADER_HOST } else { "ruturajs-macbook-pro.taila5426e.ts.net" }
$LeaderPort = if ($env:LEADER_PORT) { [int]$env:LEADER_PORT } else { 8787 }
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectDir ".venv"

function Write-Step {
    param([string]$Message)
    Write-Host "[bootstrap:windows] $Message"
}

function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

function Get-CommandPath {
    param([string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Ensure-Winget {
    if (Get-CommandPath "winget.exe") {
        Write-Step "winget already present"
        return
    }

    Write-Step "winget missing; installing Microsoft App Installer MSIX bundle"
    $installer = Join-Path $env:TEMP "Microsoft.DesktopAppInstaller.msixbundle"
    Invoke-WebRequest -Uri "https://aka.ms/getwinget" -OutFile $installer
    Add-AppxPackage -Path $installer
    Refresh-Path

    if (-not (Get-CommandPath "winget.exe")) {
        throw "winget installation did not finish. Install App Installer from Microsoft Store, then re-run this script."
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

    Write-Step "Installing $PackageId via winget"
    winget install --id $PackageId --exact --accept-source-agreements --accept-package-agreements
    Refresh-Path
}

function Find-Python311 {
    $candidates = @(
        (Get-CommandPath "python3.11.exe"),
        (Get-CommandPath "python.exe"),
        "$env:LocalAppData\Programs\Python\Python311\python.exe",
        "$env:ProgramFiles\Python311\python.exe"
    ) | Where-Object { $_ -and (Test-Path $_) }

    foreach ($candidate in $candidates) {
        try {
            $version = & $candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($version -eq "3.11") { return $candidate }
        } catch {
        }
    }

    $py = Get-CommandPath "py.exe"
    if ($py) {
        try {
            & $py -3.11 -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) { return "$py -3.11" }
        } catch {
        }
    }

    return $null
}

function Invoke-Python311 {
    param(
        [string]$PythonSpec,
        [string[]]$Arguments
    )
    if ($PythonSpec.EndsWith(" -3.11")) {
        $exe = $PythonSpec.Substring(0, $PythonSpec.Length - 6)
        & $exe -3.11 @Arguments
    } else {
        & $PythonSpec @Arguments
    }
}

function Ensure-Python311 {
    $python = Find-Python311
    if ($python) {
        Write-Step "Python 3.11 already present"
        return $python
    }

    Write-Step "Installing Python 3.11 via winget"
    winget install --id Python.Python.3.11 --exact --accept-source-agreements --accept-package-agreements
    Refresh-Path

    $python = Find-Python311
    if (-not $python) {
        throw "Python 3.11 installation completed but python was not found on PATH. Open a new PowerShell and re-run this script."
    }
    return $python
}

function Find-Tailscale {
    $cmd = Get-CommandPath "tailscale.exe"
    if ($cmd) { return $cmd }

    $candidates = @(
        "$env:ProgramFiles\Tailscale\tailscale.exe",
        "${env:ProgramFiles(x86)}\Tailscale\tailscale.exe"
    ) | Where-Object { $_ -and (Test-Path $_) }

    if ($candidates.Count -gt 0) { return $candidates[0] }
    return $null
}

function Test-TailscaleRunning {
    param([string]$TailscaleExe)
    try {
        $status = & $TailscaleExe status --json 2>$null
        return ($status -join "`n") -match '"BackendState"\s*:\s*"Running"'
    } catch {
        return $false
    }
}

function Ensure-Tailscale {
    $installedNow = $false
    $tailscale = Find-Tailscale
    if ($tailscale) {
        Write-Step "Tailscale CLI already present"
    } else {
        Write-Step "Installing Tailscale via winget"
        winget install --id Tailscale.Tailscale --exact --accept-source-agreements --accept-package-agreements
        Refresh-Path
        $installedNow = $true
        $tailscale = Find-Tailscale
    }

    if (-not $tailscale) {
        throw "Tailscale installation completed but tailscale.exe was not found. Open a new PowerShell and re-run this script."
    }

    $service = Get-Service -Name "Tailscale" -ErrorAction SilentlyContinue
    if ($service -and $service.Status -ne "Running") {
        Write-Step "Starting Tailscale service"
        Start-Service -Name "Tailscale" -ErrorAction SilentlyContinue
    }

    if (Test-TailscaleRunning $tailscale) {
        Write-Step "Tailscale is authenticated"
        return $tailscale
    }

    if ($installedNow) {
        Write-Step "Tailscale was just installed."
    }
    Write-Step "Authenticate Tailscale in the browser. If the browser does not open, use the URL printed below."
    $authOutput = (& $tailscale up --timeout=1s 2>&1 | Out-String)
    Write-Host $authOutput
    $match = [regex]::Match($authOutput, "https://\S+")
    if ($match.Success) {
        Start-Process $match.Value
    } else {
        & $tailscale up
    }

    Write-Step "Waiting for Tailscale authentication to finish"
    while (-not (Test-TailscaleRunning $tailscale)) {
        Start-Sleep -Seconds 5
    }
    Write-Step "Tailscale is authenticated"
    return $tailscale
}

function Write-GpuStatus {
    $nvidia = Get-CommandPath "nvidia-smi.exe"
    if (-not $nvidia) {
        Write-Step "No NVIDIA GPU detected through nvidia-smi"
        return
    }

    $gpu = (& $nvidia --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
    Write-Step "NVIDIA GPU detected: $gpu"

    $cudaPath = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $cudaPath) {
        Write-Step "CUDA toolkit directory already present"
    } else {
        Write-Step "CUDA toolkit installation is deferred until the PyTorch training milestone."
    }
}

function Ensure-Venv {
    param([string]$PythonSpec)
    $venvPython = Join-Path $VenvDir "Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        Write-Step "Creating virtual environment at $VenvDir"
        Invoke-Python311 -PythonSpec $PythonSpec -Arguments @("-m", "venv", $VenvDir)
    } else {
        Write-Step "Virtual environment already present"
    }

    Write-Step "Installing project package into virtual environment"
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r (Join-Path $ProjectDir "requirements.txt")
    Ensure-Torch $venvPython
    & $venvPython -m pip install -e $ProjectDir
    return $venvPython
}

function Ensure-Torch {
    param([string]$VenvPython)
    if ($env:SKIP_TORCH_INSTALL -eq "1") {
        Write-Step "Skipping PyTorch install because SKIP_TORCH_INSTALL=1"
        return
    }

    & $VenvPython -c "import torch, torchvision" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Step "PyTorch and torchvision already present"
        return
    }

    Write-Step "Installing PyTorch build selected for this machine"
    & $VenvPython -m dml_cluster.torch_install --install
}

function Main {
    Set-Location $ProjectDir
    Ensure-Winget
    Ensure-WingetCommand "curl.exe" "cURL.cURL"
    Ensure-WingetCommand "git.exe" "Git.Git"
    $python = Ensure-Python311
    Ensure-Tailscale | Out-Null
    Write-GpuStatus
    $venvPython = Ensure-Venv $python

    Write-Step "Detected hardware:"
    & $venvPython -m dml_cluster.hardware
    Write-Step "Starting worker inside virtual environment"
    & $venvPython -m dml_cluster.worker --leader $LeaderHost --port $LeaderPort --project-dir $ProjectDir
}

Main
