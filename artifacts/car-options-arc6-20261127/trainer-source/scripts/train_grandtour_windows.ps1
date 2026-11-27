param(
    [string]$RunName = "grandtour-local"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonPath = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$ManifestPath = Join-Path $RepoRoot "data\training\unified_v1\dataset_manifest.json"
$SplitPath = Join-Path $RepoRoot "experiments\splits\grandtour_temporal_v1.json"
$ConfigPath = Join-Path $RepoRoot "trainer\configs\grandtour_local.toml"
$RunPath = Join-Path $RepoRoot (Join-Path "runs" $RunName)

if (-not (Test-Path $PythonPath)) {
    throw "Run trainer\scripts\setup_windows.ps1 first."
}
if (Test-Path $RunPath) {
    throw "Run directory already exists: $RunPath. Choose a new -RunName."
}

Push-Location $RepoRoot
try {
    # Rebuild the split after copying data so its manifest hash and frame counts are verified.
    & $PythonPath -m smoke_trainer.cli split `
        --manifest $ManifestPath `
        --output $SplitPath `
        --source-domain grandtour_ros1 `
        --verify-checksums

    & $PythonPath -m smoke_trainer.cli train --config $ConfigPath --run-name $RunName

    & $PythonPath -m smoke_trainer.cli evaluate `
        --bundle (Join-Path $RunPath "bundle.pt") `
        --manifest $ManifestPath `
        --split $SplitPath `
        --partition test `
        --output (Join-Path $RunPath "test_metrics.json")
} finally {
    Pop-Location
}
