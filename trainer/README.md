# Smoke trainer

This package provides the first three end-to-end training milestones for the LiDAR
smoke thesis: canonical dataset validation, a reproducible chronological split,
local voxel and global point features, compact local and PointNet-style models,
checkpointing, calibration, full-frame evaluation, portable inference, and
end-to-end timing.

The current models are the L1 local voxel baseline, G1 global single-scan
PointNet-style classifier, and GT global temporal PointNet-GRU classifier.

## Environment

Use a stable Python 3.11 or newer release. Create an environment in the repository
and install the PyTorch build appropriate for the machine before installing this
package. Do not use a Python release candidate.

Linux CPU development:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e './trainer[dev]'
```

Windows PowerShell, without administrator privileges:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install the CUDA wheel selected at https://pytorch.org/get-started/locally/
python -m pip install torch --index-url <PYTORCH-CUDA-INDEX>
python -m pip install -e ".\trainer[dev]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The Windows installation stays under the user's repository and profile. A working
NVIDIA driver is the only system-level prerequisite. The full CUDA toolkit is not
needed for the prebuilt PyTorch wheel. If `py` is unavailable, Python's standard
Windows installer can install the launcher and interpreter for the current user.

Copy the repository and `data/training/unified_v1/` to the Windows PC. Original ROS
bags and labeler intermediates are not required for training. Run the split command
with `--verify-checksums` after copying the data.

The repository also includes a no-admin setup wrapper. Copy the CUDA index URL shown
by the official PyTorch selector and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\trainer\scripts\setup_windows.ps1 `
  -TorchIndexUrl https://download.pytorch.org/whl/<CUDA-VARIANT>

powershell -ExecutionPolicy Bypass -File .\trainer\scripts\train_windows.ps1 `
  -RunName local-default
```

## Windows GPU setup for GT

On the Windows machine, copy the repository code and the existing dataset so this
layout is present (the original ROS bags are not needed):

```text
LidarSmokeDetection\
  data\training\unified_v1\
    chunks\
    dataset_manifest.json
    dataset_schema.json
```

Install the NVIDIA driver first. Then select the CUDA wheel index matching that
driver from the official PyTorch selector and run this once from the repository
root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\trainer\scripts\setup_windows.ps1 `
  -TorchIndexUrl https://download.pytorch.org/whl/<CUDA-VARIANT>
```

The setup verifies that PyTorch can see the GPU. Start the complete GT run with:

```powershell
powershell -ExecutionPolicy Bypass -File .\trainer\scripts\train_global_temporal_windows.ps1 `
  -RunName global-temporal-v1
```

The script checks the expected dataset files, verifies the manifest and chunk hashes,
recreates the chronological split for this exact dataset copy, trains GT on CUDA, and
writes the held-out test report to `runs\global-temporal-v1\test_metrics.json`.
For a local copy whose checksums have already been verified, add `-SkipChecksums` to
avoid the extra file-hash pass. Do not use that option after moving or changing data.

To train and compare all three models on one identical chronological split, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\trainer\scripts\train_all_models_windows.ps1 `
  -ExperimentName comparison-v1
```

This runs L1 (local voxel MLP), G1 (single-scan global PointNet), and GT (five-scan
global PointNet-GRU), evaluates each only on the shared held-out test partition, and
writes both `runs\comparison-v1-comparison.json` and
`runs\comparison-v1-comparison.csv`. The PowerShell output also displays the primary
comparison table: average precision, ROC-AUC, F1, recall, and false-positive rate.

## Moving-data-only comparison

To compare the models without any stationary frames in either training or testing,
run this separate experiment:

```powershell
powershell -ExecutionPolicy Bypass -File .\trainer\scripts\train_moving_models_windows.ps1 `
  -ExperimentName moving-only-v1
```

It creates a split limited to `grandtour_ros1`, then trains L1, G1, and GT with the
matching GrandTour-only configurations. Its comparison outputs are
`runs\moving-only-v1-comparison.json` and `.csv`. Do not compare its absolute scores
directly with the mixed-domain run: it uses a different training distribution and a
GrandTour-only split, intentionally answering a different question. GT requires a
five-scan causal context, so its report omits the first four frames of each contiguous
test segment; use the exported `points` count when comparing it with L1 and G1.

## GrandTour-only experiment

To measure the moving-sensor baseline without stationary ROS 2 frames, use the
dedicated Windows wrapper after copying `data/training/unified_v1/`. It rebuilds a
chronological split containing only `grandtour_ros1`, trains the local MLP, and
then evaluates the previously untouched GrandTour test partition. The resulting
files are written under `runs/<RunName>/`, including `test_metrics.json`.

```powershell
powershell -ExecutionPolicy Bypass -File .\trainer\scripts\train_grandtour_windows.ps1 `
  -RunName grandtour-local-v1
```

For a manual run on any platform, add `--source-domain grandtour_ros1` to the
`smoke-train split` command and use `trainer/configs/grandtour_local.toml` for
training. The configuration rejects a split that was not created for
`grandtour_ros1`. The split leaves twenty frames on each side of every
chronological partition boundary unused to reduce temporal leakage.

## First run

From the repository root:

```bash
smoke-train inspect \
  --manifest data/training/unified_v1/dataset_manifest.json

smoke-train split \
  --manifest data/training/unified_v1/dataset_manifest.json \
  --output experiments/splits/pilot_temporal_v1.json

smoke-train train --config trainer/configs/local.toml --run-name local-default
```

## G1 global PointNet-style model

G1 consumes every finite point in a scan as `[x, y, z, range, intensity]`. A shared
point MLP forms per-point embeddings, global max pooling forms a scan summary, and a
small head predicts one smoke logit per original point. Its default configuration is:

```bash
smoke-train train \
  --config trainer/configs/global_pointnet.toml \
  --run-name global-pointnet-default
```

The canonical G1 configuration enables a compact, identity-initialized PointNet
input transform over XYZ. Its orthogonality penalty discourages non-rigid learned
warps. Training also applies random yaw rotations, reflecting changes in vehicle
heading; pitch and roll augmentation are deliberately disabled until their real
operating range has been measured. Neither mechanism registers scans across time or
makes smoke deformation disappear; those are temporal-model questions.

## GT global temporal model

GT encodes each of five consecutive scans with a shared G1-style PointNet encoder,
then feeds their global scan summaries into a causal GRU. Its segmentation head uses
the current point embedding, current scan summary, and GRU state to predict only the
final scan's points. It never consumes future frames, crosses a chunk or split
boundary, or carries state over a timestamp gap larger than `maximum_time_gap_s`.

```bash
smoke-train train \
  --config trainer/configs/global_temporal.toml \
  --run-name global-temporal-default
```

GT uses the same direct `[x, y, z, range, intensity]` point inputs as G1, with an
identity-initialized per-scan PointNet input transform. Its default training
augmentation applies one shared random yaw rotation to the whole five-scan window,
preserving real inter-scan motion. Online inference maintains a bounded cache and
resets automatically at stream/chunk changes, non-increasing timestamps, and time
gaps over the configured maximum. A one-frame `smoke-train predict` request warms
that cache from immediately preceding frames in the selected chunk.

The split command reads each chunk once to record exact class counts. Add
`--verify-checksums` when the data have been copied to another computer.

Each run contains the resolved configuration and split, train-only normalization,
last and best checkpoints, training history, calibrated selection metrics, a
calibration-fit diagram, and `bundle.pt`. The bundle uses standard PyTorch state
dictionaries and loads on CPU or CUDA independently of where it was trained. Training
does not inspect the test partition; run the explicit evaluation command only after
the compared recipes have been fixed.

Evaluate a bundle again or create aligned per-point predictions:

```bash
smoke-train evaluate \
  --bundle runs/local-default/bundle.pt \
  --manifest data/training/unified_v1/dataset_manifest.json \
  --split experiments/splits/pilot_temporal_v1.json \
  --partition test \
  --output runs/local-default/test-repeat.json

smoke-train predict \
  --bundle runs/local-default/bundle.pt \
  --manifest data/training/unified_v1/dataset_manifest.json \
  --chunk 1 --frame 0 \
  --output runs/local-default/example-prediction.npz
```

For a quick pipeline check, `train` accepts `--max-train-frames`,
`--max-eval-frames`, and `--max-epochs`. Such a run is diagnostic only because the
limited prefixes may not contain both classes. Normal experiments use complete split
partitions.

Run the test suite with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest trainer/tests -q
```

Disabling external pytest plugins keeps a sourced ROS installation from injecting
its launch-testing plugins into this independent environment.
