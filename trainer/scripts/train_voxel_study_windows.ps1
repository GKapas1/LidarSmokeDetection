param(
    [string]$RunPrefix = "voxel-study-v1",
    [switch]$PrepareOnly
)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Run trainer\scripts\setup_windows.ps1 first." }
$Arguments = @("--prefix", $RunPrefix, "--device", "cuda")
if ($PrepareOnly) { $Arguments += "--prepare-only" }
& $Python (Join-Path $PSScriptRoot "run_voxel_study.py") @Arguments
if ($LASTEXITCODE -ne 0) { throw "Voxel study failed. Rerun the same command to resume checkpointed runs." }
