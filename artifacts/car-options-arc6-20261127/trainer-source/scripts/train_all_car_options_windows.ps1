param(
    [string[]]$Models = @("l1", "g1", "gt", "voxel"),
    [string[]]$Recipes = @(),
    [string]$RunPrefix = "car-all-options-v1"
)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonPath = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Dataset = Join-Path $RepoRoot "data\training\car_training_options_v1"
$Setup = Join-Path $RepoRoot "runs\$RunPrefix-setup"
if (-not (Test-Path $PythonPath -PathType Leaf)) { throw "Run trainer\scripts\setup_windows.ps1 first." }
foreach ($Path in @("dataset_manifest.json", "experiments.json", "split_car_only.json", "split_with_arc5.json")) {
    if (-not (Test-Path (Join-Path $Dataset $Path) -PathType Leaf)) { throw "Missing $Path in $Dataset" }
}
& $PythonPath -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; import smoke_trainer.augmentation; print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "CUDA trainer environment check failed." }
& $PythonPath -m smoke_trainer.cli inspect --manifest (Join-Path $Dataset "dataset_manifest.json") --verify-checksums
if ($LASTEXITCODE -ne 0) { throw "Dataset verification failed." }
$Arguments = @("--dataset", $Dataset, "--output", $Setup, "--run-prefix", $RunPrefix, "--models") + $Models
if ($Recipes.Count -gt 0) { $Arguments += @("--recipes") + $Recipes }
& $PythonPath (Join-Path $PSScriptRoot "prepare_all_car_options.py") @Arguments
if ($LASTEXITCODE -ne 0) { throw "Experiment preparation failed." }
$Runs = Get-Content (Join-Path $Setup "runs.json") -Raw | ConvertFrom-Json
foreach ($Run in $Runs) {
    Write-Host "`n=== $($Run.model): $($Run.recipe) ==="
    & $PythonPath -m smoke_trainer.cli train --config $Run.config --run-name $Run.run_name --device cuda
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $($Run.run_name). Existing runs were retained." }
    $RunPath = Join-Path $RepoRoot ("runs\" + $Run.run_name)
    & $PythonPath -m smoke_trainer.cli evaluate --bundle (Join-Path $RunPath "bundle.pt") `
        --manifest (Join-Path $Dataset "dataset_manifest.json") --split $Run.split --partition test `
        --output (Join-Path $RunPath "test_metrics.json") --device cuda
    if ($LASTEXITCODE -ne 0) { throw "Test evaluation failed: $($Run.run_name)." }
}
Write-Host "Completed $($Runs.Count) runs. Configurations and splits: $Setup"
