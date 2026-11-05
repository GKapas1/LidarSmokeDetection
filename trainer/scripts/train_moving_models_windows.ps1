param(
    [string]$ExperimentName = "moving-models-v1",
    [switch]$SkipChecksums
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonPath = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$DataRoot = Join-Path $RepoRoot "data\training\unified_v1"
$ManifestPath = Join-Path $DataRoot "dataset_manifest.json"
$SchemaPath = Join-Path $DataRoot "dataset_schema.json"
$ChunksPath = Join-Path $DataRoot "chunks"
$SplitPath = Join-Path $RepoRoot "experiments\splits\grandtour_temporal_v1.json"
$RunsRoot = Join-Path $RepoRoot "runs"
$MovingDomain = "grandtour_ros1"

$Models = @(
    [PSCustomObject]@{
        Key = "l1"
        Name = "L1 local voxel MLP"
        ConfigPath = Join-Path $RepoRoot "trainer\configs\grandtour_local.toml"
    },
    [PSCustomObject]@{
        Key = "g1"
        Name = "G1 global PointNet"
        ConfigPath = Join-Path $RepoRoot "trainer\configs\grandtour_global_pointnet.toml"
    },
    [PSCustomObject]@{
        Key = "gt"
        Name = "GT global temporal PointNet-GRU"
        ConfigPath = Join-Path $RepoRoot "trainer\configs\grandtour_global_temporal.toml"
    }
)

function Invoke-Checked([scriptblock]$Command, [string]$Description) {
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

if (-not (Test-Path $PythonPath)) {
    throw "Run trainer\scripts\setup_windows.ps1 first."
}
foreach ($Path in @($ManifestPath, $SchemaPath, $ChunksPath)) {
    if (-not (Test-Path $Path)) {
        throw "Required training input is missing: $Path"
    }
}
foreach ($Model in $Models) {
    if (-not (Test-Path $Model.ConfigPath)) {
        throw "Model configuration is missing: $($Model.ConfigPath)"
    }
    $RunPath = Join-Path $RunsRoot "$ExperimentName-$($Model.Key)"
    if (Test-Path $RunPath) {
        throw "Run directory already exists: $RunPath. Choose a new -ExperimentName."
    }
}

$ComparisonJson = Join-Path $RunsRoot "$ExperimentName-comparison.json"
$ComparisonCsv = Join-Path $RunsRoot "$ExperimentName-comparison.csv"
if ((Test-Path $ComparisonJson) -or (Test-Path $ComparisonCsv)) {
    throw "Comparison output already exists for $ExperimentName. Choose a new -ExperimentName."
}

Push-Location $RepoRoot
try {
    $Verification = @()
    if (-not $SkipChecksums) {
        $Verification = @("--verify-checksums")
    }

    # The split contains only the moving GrandTour stream. Every model therefore
    # trains and tests under the same moving-sensor distribution.
    Invoke-Checked {
        & $PythonPath -m smoke_trainer.cli inspect --manifest $ManifestPath @Verification
    } "Dataset inspection"
    Invoke-Checked {
        & $PythonPath -m smoke_trainer.cli split `
            --manifest $ManifestPath `
            --output $SplitPath `
            --source-domain $MovingDomain `
            @Verification
    } "Moving-data split creation"

    foreach ($Model in $Models) {
        $RunName = "$ExperimentName-$($Model.Key)"
        $RunPath = Join-Path $RunsRoot $RunName
        Write-Host "`n=== Training $($Model.Name) on moving data only ==="
        Invoke-Checked {
            & $PythonPath -m smoke_trainer.cli train `
                --config $Model.ConfigPath `
                --run-name $RunName `
                --device cuda
        } "$($Model.Name) training"
        Invoke-Checked {
            & $PythonPath -m smoke_trainer.cli evaluate `
                --bundle (Join-Path $RunPath "bundle.pt") `
                --manifest $ManifestPath `
                --split $SplitPath `
                --partition test `
                --output (Join-Path $RunPath "test_metrics.json") `
                --device cuda
        } "$($Model.Name) test evaluation"
    }

    $Comparison = foreach ($Model in $Models) {
        $RunName = "$ExperimentName-$($Model.Key)"
        $RunPath = Join-Path $RunsRoot $RunName
        $ReportPath = Join-Path $RunPath "test_metrics.json"
        $Report = Get-Content $ReportPath -Raw | ConvertFrom-Json
        $Metrics = $Report.metrics.overall
        [PSCustomObject][ordered]@{
            model = $Model.Name
            run_name = $RunName
            training_domain = $MovingDomain
            test_report = $ReportPath
            average_precision = $Metrics.average_precision
            roc_auc = $Metrics.roc_auc
            brier_score = $Metrics.brier_score
            f1 = $Metrics.f1
            precision = $Metrics.precision
            recall = $Metrics.recall
            false_positive_rate = $Metrics.false_positive_rate
            threshold = $Metrics.threshold
            points = $Metrics.points
            positives = $Metrics.positives
        }
    }
    $Comparison | ConvertTo-Json -Depth 3 | Set-Content -Path $ComparisonJson -Encoding utf8
    $Comparison | Export-Csv -Path $ComparisonCsv -NoTypeInformation -Encoding utf8

    Write-Host "`n=== Moving-data held-out test comparison ==="
    $Comparison | Format-Table model, average_precision, roc_auc, f1, recall, false_positive_rate -AutoSize
    Write-Host "Comparison JSON: $ComparisonJson"
    Write-Host "Comparison CSV:  $ComparisonCsv"
} finally {
    Pop-Location
}
