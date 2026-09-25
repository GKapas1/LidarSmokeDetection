param(
    [string]$RunName = "local-default"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonPath = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$ConfigPath = Join-Path $RepoRoot "trainer\configs\local.toml"

if (-not (Test-Path $PythonPath)) {
    throw "Run trainer\scripts\setup_windows.ps1 first."
}

Push-Location $RepoRoot
try {
    & $PythonPath -m smoke_trainer.cli train --config $ConfigPath --run-name $RunName
} finally {
    Pop-Location
}
