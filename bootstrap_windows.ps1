$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$LeaderHost = if ($env:LEADER_HOST) { $env:LEADER_HOST } else { "leader-macbook-pro.taila5426e.ts.net" }
$LeaderPort = if ($env:LEADER_PORT) { [int]$env:LEADER_PORT } else { 8787 }
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectDir ".venv"
if (-not $env:PIP_NO_CACHE_DIR) { $env:PIP_NO_CACHE_DIR = "1" }

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

function New-PythonSpec {
    param(
        [string]$Executable,
        [string[]]$LauncherArgs = @()
    )
    return [pscustomobject]@{
        Executable = $Executable
        LauncherArgs = $LauncherArgs
    }
}

function Test-PythonSpec {
    param([object]$PythonSpec)
    if (-not $PythonSpec -or -not (Test-Path $PythonSpec.Executable)) {
        return $false
    }

    try {
        $args = @($PythonSpec.LauncherArgs) + @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        $version = (& $PythonSpec.Executable @args 2>$null | Select-Object -First 1)
        return ($LASTEXITCODE -eq 0 -and $version -eq "3.11")
    } catch {
        return $false
    }
}

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )

    if (-not (Test-Path $FilePath)) {
        throw "Executable not found: $FilePath"
    }

    & "$FilePath" @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

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
        $executable = [string]$key.GetValue("ExecutablePath")
        $installPath = [string]$key.GetValue("")
        if ($executable) { $candidates += $executable }
        if ($installPath) { $candidates += (Join-Path $installPath "python.exe") }
    }
    return $candidates
}

function Get-Python311DirectoryCandidates {
    $roots = @(
        (Join-Path $env:LocalAppData "Programs\Python"),
        $env:ProgramFiles,
        ${env:ProgramFiles(x86)}
    ) | Where-Object { $_ -and (Test-Path $_) }

    $candidates = @()
    foreach ($root in $roots) {
        $dirs = Get-ChildItem -Path $root -Directory -Filter "Python311*" -ErrorAction SilentlyContinue
        foreach ($dir in $dirs) {
            $candidates += (Join-Path $dir.FullName "python.exe")
        }
    }
    return $candidates
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
    $pyCandidates = @(
        (Get-CommandPath "py.exe"),
        (Join-Path $env:SystemRoot "py.exe"),
        (Join-Path $env:LocalAppData "Programs\Python\Launcher\py.exe")
    ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique

    foreach ($py in $pyCandidates) {
        $spec = New-PythonSpec -Executable $py -LauncherArgs @("-3.11")
        if (Test-PythonSpec $spec) { return $spec }
    }

    $candidates = @(
        (Get-CommandPath "python3.11.exe"),
        (Get-CommandPath "python.exe"),
        "$env:LocalAppData\Programs\Python\Python311\python.exe",
        "$env:ProgramFiles\Python311\python.exe",
        "${env:ProgramFiles(x86)}\Python311\python.exe"
    )
    $candidates += Get-Python311RegistryCandidates
    $candidates += Get-Python311DirectoryCandidates
    $candidates = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique

    foreach ($candidate in $candidates) {
        if ($candidate -like "*\Microsoft\WindowsApps\*") {
            continue
        }
        $spec = New-PythonSpec -Executable $candidate
        if (Test-PythonSpec $spec) { return $spec }
    }

    return $null
}

function Invoke-Python311 {
    param(
        [object]$PythonSpec,
        [string[]]$Arguments
    )
    if (-not (Test-PythonSpec $PythonSpec)) {
        throw "Python 3.11 was found but could not be executed. Close PowerShell, open a new PowerShell, and re-run this script."
    }
    $args = @($PythonSpec.LauncherArgs) + $Arguments
    & $PythonSpec.Executable @args
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.11 command failed with exit code ${LASTEXITCODE}: $($PythonSpec.Executable) $($args -join ' ')"
    }
}

function Ensure-Python311 {
    $pythonPath = "$env:LocalAppData\Programs\Python\Python311\python.exe"

    if (Test-Path $pythonPath) {
        Write-Step "Using Python 3.11 from: $pythonPath"
        return New-PythonSpec -Executable $pythonPath
    }

    # fallback to py launcher
    $py = Get-CommandPath "py.exe"
    if ($py) {
        $spec = New-PythonSpec -Executable $py -LauncherArgs @("-3.11")
        if (Test-PythonSpec $spec) {
            Write-Step "Using Python via py launcher"
            return $spec
        }
    }

    throw "Python 3.11 not found, but it SHOULD be. Something is very wrong."
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
    param([object]$PythonSpec)
    $venvPython = Join-Path $VenvDir "Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        Write-Step "Creating virtual environment at $VenvDir"
        Invoke-Python311 -PythonSpec $PythonSpec -Arguments @("-m", "venv", $VenvDir)
    } else {
        Write-Step "Virtual environment already present"
    }

    Write-Step "Installing project package into virtual environment"
    Invoke-Checked -FilePath $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-Checked -FilePath $venvPython -Arguments @("-m", "pip", "install", "-r", (Join-Path $ProjectDir "requirements.txt"))
    Ensure-Torch $venvPython
    Invoke-Checked -FilePath $venvPython -Arguments @("-m", "pip", "install", "-e", $ProjectDir)
    Invoke-Checked -FilePath $venvPython -Arguments @("-c", "import dml_cluster.hardware, dml_cluster.worker")
    return $venvPython
}

function Ensure-Torch {
    param([string]$VenvPython)
    if ($env:SKIP_TORCH_INSTALL -eq "1") {
        Write-Step "Skipping PyTorch install because SKIP_TORCH_INSTALL=1"
        return
    }

    & "$VenvPython" -c "import torch, torchvision" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Step "PyTorch and torchvision already present"
        return
    }

    Write-Step "Installing PyTorch build selected for this machine"
    Invoke-Checked -FilePath $VenvPython -Arguments @("-m", "dml_cluster.torch_install", "--install")
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
    Invoke-Checked -FilePath $venvPython -Arguments @("-m", "dml_cluster.hardware")
    Write-Step "Starting worker inside virtual environment"
    & "$venvPython" -m dml_cluster.worker --leader $LeaderHost --port $LeaderPort --project-dir $ProjectDir
}

Main
