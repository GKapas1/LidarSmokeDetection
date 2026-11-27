param(
    [string[]]$Recipes = @(),
    [string]$RunPrefix = "car-options-v1"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonPath = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Dataset = Join-Path $RepoRoot "data\training\car_training_options_v1"
$Inventory = Get-Content (Join-Path $Dataset "experiments.json") -Raw | ConvertFrom-Json
if ($Recipes.Count -eq 0) { $Recipes = $Inventory.recipes }
foreach ($Recipe in $Recipes) {
    if ($Inventory.recipes -notcontains $Recipe) { throw "Unknown recipe $Recipe" }
    $Run = Join-Path $RepoRoot ("runs\" + $RunPrefix + "-" + $Recipe)
    if (Test-Path $Run) { throw "Run already exists at $Run" }
}
& $PythonPath -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; import smoke_trainer.augmentation; print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "Trainer environment check failed" }
& $PythonPath -m smoke_trainer.cli inspect --manifest (Join-Path $Dataset "dataset_manifest.json") --verify-checksums
if ($LASTEXITCODE -ne 0) { throw "Dataset verification failed" }
foreach ($Recipe in $Recipes) {
    $Config = Join-Path $Dataset ("configs\" + $Recipe + ".toml")
    & $PythonPath -m smoke_trainer.cli train --config $Config --run-name ($RunPrefix + "-" + $Recipe) --device cuda
    if ($LASTEXITCODE -ne 0) { throw "Training failed for $Recipe" }
}
