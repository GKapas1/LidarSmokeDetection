# Smoke trainer

This package provides the first end-to-end training milestone for the LiDAR smoke
thesis: canonical dataset validation, a reproducible chronological split, local
voxel features, a compact MLP, checkpointing, calibration, full-frame evaluation,
portable inference, and end-to-end timing.

The current model is the local single-scan baseline. Global PointNet and temporal
models will use the same split, evaluation, checkpoint, and predictor contracts.

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

For a quick pipeline check, `train` accepts `--max-train-frames` and
`--max-eval-frames`. Such a run is diagnostic only because the limited prefixes may
not contain both classes. Normal experiments use complete split partitions.

Run the test suite with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest trainer/tests -q
```

Disabling external pytest plugins keeps a sourced ROS installation from injecting
its launch-testing plugins into this independent environment.
