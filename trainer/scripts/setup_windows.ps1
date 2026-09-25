param(
    [Parameter(Mandatory = $true)]
    [string]$TorchIndexUrl,
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "3.12"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$EnvironmentPath = Join-Path $RepoRoot ".venv"
$PythonPath = Join-Path $EnvironmentPath "Scripts\python.exe"

& nvidia-smi
if ($LASTEXITCODE -ne 0) {
    throw "nvidia-smi failed. A working NVIDIA driver is required."
}

if (-not (Test-Path $PythonPath)) {
    & $PythonLauncher "-$PythonVersion" -m venv $EnvironmentPath
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the Python environment."
    }
}

& $PythonPath -m pip install --upgrade pip
& $PythonPath -m pip install torch --index-url $TorchIndexUrl
& $PythonPath -m pip install -e "$RepoRoot\trainer[dev]"
& $PythonPath -c "import torch; assert torch.cuda.is_available(); print(torch.__version__); print(torch.cuda.get_device_name(0))"

Write-Host "Trainer environment ready at $EnvironmentPath"

