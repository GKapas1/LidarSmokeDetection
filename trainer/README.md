# Smoke trainer

This package provides the first three end-to-end training milestones for the LiDAR
smoke thesis: canonical dataset validation, a reproducible chronological split,
local voxel and global point features, compact local and PointNet-style models,
checkpointing, calibration, full-frame evaluation, portable inference, and
end-to-end timing.

The models are the L1 local voxel baseline, G1 global single-scan
PointNet-style classifier, GT global temporal PointNet-GRU classifier, and the
local voxel PointNet described below.

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

## Windows: all car and ARC-5 options

The car-options dataset supports a 4-model by 8-recipe CUDA experiment: L1,
G1, GT, and the local voxel PointNet, each with car-only or clean ARC-5 data,
and with baseline, distant-smoke copying, background masking, or both.

On the Windows PC, extract the existing `car_training_options_v1_windows.zip`
at the repository root. Copy the **current** `trainer/` source and scripts after
extracting it: the older archive contains an earlier trainer. The dataset must be
at `data/training/car_training_options_v1/`. Install the NVIDIA driver and set
up the repository environment as described in `docs/CAR_WINDOWS_TRAINING.md`.
Then run from the repository root:

```powershell
.\trainer\scripts\train_all_car_options_windows.ps1 -RunPrefix car-all-options-v1
```

This runs 32 experiments sequentially, each with up to 30 epochs, 1,400
optimizer steps per epoch and four hours. It checks CUDA and dataset hashes,
refuses to overwrite runs, generates configurations under
`runs/<RunPrefix>-setup/`, and saves each model and its held-out car test report
under `runs/<RunPrefix>-<model>-<recipe>/`. For a smaller selection:

```powershell
.\trainer\scripts\train_all_car_options_windows.ps1 `
  -Models voxel,gt `
  -Recipes a_baseline_with_arc5,d_both_with_arc5 `
  -RunPrefix car-local-temporal-v1
```

Model keys are `l1`, `g1`, `gt`, and `voxel`. Recipe names are listed in
`data/training/car_training_options_v1/experiments.json`; see
`artifacts/car-training-options-v1/README.md` for their construction and label
caveats. After an interruption, use a new prefix for a subset of unfinished
runs, or resume a specific run with `smoke-train train --resume-run` and its
generated configuration.

Single-scan models use the original split, including every fourth clean ARC-5
scan. GT instead uses 254 consecutive clean ARC-5 training frames, yielding 250
five-frame targets; its car selection, calibration and test recordings stay the
same. GT evaluation excludes the first four scans of each contiguous segment,
so its reported test points differ from the single-scan models. Frame-level
smoke copying and context masking in GT modify only the target scan: without
registration they do not model coherent smoke movement across the history.
Treat those GT augmented runs as a temporal robustness experiment and compare
them with the GT baseline, rather than assuming realistic synthetic motion.

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

## Local voxel PointNet

This model divides each scan into fixed-size, axis-aligned voxels and uses a
shared point encoder and max pooling to predict one smoke logit for each occupied
voxel. The prediction is assigned to every point in that voxel. Inputs are each
point's XYZ offset from its voxel's point centroid and its intensity; absolute
position and range are excluded. Supervision and evaluation use original point
labels, excluding ignored and invalid points. Mixed-label voxels share one
prediction and contribute the loss of each labeled point.

```bash
smoke-train train --config trainer/configs/voxel_pointnet.toml --run-name voxel-pointnet
```

Change `voxel_size_m` to choose the physical neighborhood size. Local coordinates
remove direct dependence on sensor position, but the voxel grid can change when a
scan moves relative to its boundaries. Point density and intensity can also vary
with distance, so distance independence must be measured on held-out data.
The existing comparison scripts still run L1, G1, and GT only.

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

## Nine-run voxel study

Run the three voxel variants with seeds 20260920, 20260921, and 20260922 on
Windows CUDA:

```powershell
.\trainer\scripts\train_voxel_study_windows.ps1 -RunPrefix voxel-study-v1
```

Copy the current `trainer/` code to Windows first. The workflow uses both
`data/training/car_training_options_v1/` and `data/training/unified_v1/`,
including their manifests and chunks. It verifies dataset checksums and prepares
nine configurations under `runs/<RunPrefix>-setup/`. Add `-PrepareOnly` to
generate configurations without training.

The baseline uses the existing voxel PointNet and chunk-based shuffling.
The interleaved variant assigns 50% of updates to smoke car recordings, 35% to
clean car recordings, and 15% to clean ARC-5, dividing each category equally
among its recordings. It samples frames with replacement and shuffles the
complete update schedule. Each epoch has 1,400 updates. Its chunk cache retains
the active training chunks in RAM.

The context variant uses the same interleaved schedule and combines each
central cell's embedding with max-pooled embeddings from its 26 neighboring
cells. Predictions remain on the 10 cm grid. The complete neighborhood spans
30 cm along each axis. Grid coordinates are used for neighbor lookup, while
the neural encoder consumes centroid-relative XYZ and intensity. All variants
use clean ARC-5 and hold augmentation fixed with yaw rotations only.

Each run saves its bundle, car test report, and `stationary_diagnostic.json`.
The diagnostic uses the stationary recordings in `unified_v1` and reports
recall and false-positive rate by distance at the saved threshold and at a
threshold derived from a 1% false-positive-rate target on car selection
negatives. It reports the achieved car selection rate because tied scores can
make it lower than the target. The stationary recordings serve as development
diagnostics for this study. Their sparse 5–8 m smoke observations cover
approximately 5.0–5.9 m. Repeated review of ARC-6 results makes subsequent ARC-6
comparisons exploratory.

Rerun the same command to resume checkpointed training and complete missing
evaluations. Completed bundles and reports are reused. A final
`comparison.json` under the setup directory collects all nine reports.
