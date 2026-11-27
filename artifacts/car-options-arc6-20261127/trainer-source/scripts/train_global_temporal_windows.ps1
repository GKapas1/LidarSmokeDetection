param(
    [string]$RunName = "global-temporal-default",
    [switch]$SkipChecksums
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonPath = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$DataRoot = Join-Path $RepoRoot "data\training\unified_v1"
$ManifestPath = Join-Path $DataRoot "dataset_manifest.json"
$SchemaPath = Join-Path $DataRoot "dataset_schema.json"
$ChunksPath = Join-Path $DataRoot "chunks"
$SplitPath = Join-Path $RepoRoot "experiments\splits\pilot_temporal_v1.json"
$ConfigPath = Join-Path $RepoRoot "trainer\configs\global_temporal.toml"
$RunPath = Join-Path $RepoRoot (Join-Path "runs" $RunName)

function Invoke-Checked([scriptblock]$Command, [string]$Description) {
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

if (-not (Test-Path $PythonPath)) {
    throw "Run trainer\scripts\setup_windows.ps1 first."
}
foreach ($Path in @($ManifestPath, $SchemaPath, $ChunksPath, $ConfigPath)) {
    if (-not (Test-Path $Path)) {
        throw "Required training input is missing: $Path"
    }
}
if (Test-Path $RunPath) {
    throw "Run directory already exists: $RunPath. Choose a new -RunName."
}

Push-Location $RepoRoot
try {
    $Verification = @()
    if (-not $SkipChecksums) {
        $Verification = @("--verify-checksums")
    }

    # Recreate the split on the Windows copy. This validates the manifest and
    # records its exact hash, so a stale split can never be used accidentally.
    Invoke-Checked {
        & $PythonPath -m smoke_trainer.cli inspect --manifest $ManifestPath @Verification
    } "Dataset inspection"
    Invoke-Checked {
        & $PythonPath -m smoke_trainer.cli split --manifest $ManifestPath --output $SplitPath @Verification
    } "Chronological split creation"
    Invoke-Checked {
        & $PythonPath -m smoke_trainer.cli train --config $ConfigPath --run-name $RunName --device cuda
    } "GT training"
    Invoke-Checked {
        & $PythonPath -m smoke_trainer.cli evaluate `
            --bundle (Join-Path $RunPath "bundle.pt") `
            --manifest $ManifestPath `
            --split $SplitPath `
            --partition test `
            --output (Join-Path $RunPath "test_metrics.json") `
            --device cuda
    } "GT test evaluation"
} finally {
    Pop-Location
}

Write-Host "GT run complete: $RunPath"
Write-Host "Bundle: $(Join-Path $RunPath 'bundle.pt')"
Write-Host "Test metrics: $(Join-Path $RunPath 'test_metrics.json')"
