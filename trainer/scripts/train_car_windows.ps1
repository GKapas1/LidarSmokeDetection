param(
    [string]$RunName = "car-v4-pointnet"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonPath = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$ManifestPath = Join-Path $RepoRoot "data\training\car_binary_v4\dataset_manifest.json"
$SplitPath = Join-Path $RepoRoot "experiments\splits\car_whole_recording_v1.json"
$ConfigPath = Join-Path $RepoRoot "trainer\configs\car_global_pointnet.toml"
$RunPath = Join-Path $RepoRoot (Join-Path "runs" $RunName)

if (-not (Test-Path $PythonPath -PathType Leaf)) {
    throw "Set up the CUDA trainer environment using docs\CAR_WINDOWS_TRAINING.md."
}
foreach ($Path in @($ManifestPath, $SplitPath, $ConfigPath)) {
    if (-not (Test-Path $Path -PathType Leaf)) {
        throw "Missing training input: $Path. See docs\CAR_WINDOWS_TRAINING.md."
    }
}
if (Test-Path $RunPath) {
    throw "Run already exists at $RunPath. Choose a new -RunName or resume via the trainer CLI."
}

& $PythonPath -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see a CUDA device'; print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) {
    throw "CUDA environment check failed with exit code $LASTEXITCODE."
}

# The config verifies dataset checksums and loads the committed whole-recording split.
& $PythonPath -m smoke_trainer.cli train --config $ConfigPath --run-name $RunName --device cuda
if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE."
}

& $PythonPath -m smoke_trainer.cli evaluate `
    --bundle (Join-Path $RunPath "bundle.pt") `
    --manifest $ManifestPath `
    --split $SplitPath `
    --partition test `
    --output (Join-Path $RunPath "test_metrics.json") `
    --device cuda
if ($LASTEXITCODE -ne 0) {
    throw "Held-out evaluation failed with exit code $LASTEXITCODE."
}

Write-Host "Run complete at $RunPath"
Write-Host "Held-out metrics at $(Join-Path $RunPath 'test_metrics.json')"
