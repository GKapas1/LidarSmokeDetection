# Car v4 training handoff for Windows

Run the single-scan PointNet baseline on CUDA using all training frames assigned in the saved whole-recording split. Complete calibration and held-out evaluation, then report the metrics and artifact paths. The same export contains all five smoky car drives and all five clean moving controls.

## Copy the dataset and prepare CUDA

1. Pull the repository changes onto the Windows PC. Transfer `car_binary_v4_windows.zip` from the Linux workspace separately. From the repository root, extract it into `data\training`. The resulting path must be `data\training\car_binary_v4\dataset_manifest.json`, alongside `dataset_schema.json` and the ten files in `chunks\`. If transferring the directory directly, preserve the manifest and chunk bytes so the split and dataset hashes still match.
2. Check that `nvidia-smi` sees the GPU. If needed, run the existing setup script with the CUDA wheel index from the PyTorch installer selector for that PC. Use the actual index URL in place of `<CUDA-INDEX-URL>`.

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\trainer\scripts\setup_windows.ps1 -TorchIndexUrl "<CUDA-INDEX-URL>"
   .\.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
   ```

The transfer contains about 812 MiB of data. The manifest identifies 3,587 frames and 71,722,176 points. Raw MCAPs and offline labeling intermediates are unnecessary for training. Any Linux source paths in the manifest are provenance. The trainer loads the relative chunk paths.

## Run and report

From the repository root, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\trainer\scripts\train_car_windows.ps1 -RunName car-v4-pointnet
```

The script requires CUDA and uses [car_global_pointnet.toml](../trainer/configs/car_global_pointnet.toml) and the committed [whole-recording split](../experiments/splits/car_whole_recording_v1.json). Keep the saved split for this baseline. The generic split command makes a chronological split within recordings and would change this experiment's evaluation.

| Role | Recordings | Frames |
| --- | --- | ---: |
| Training | Hall smoke 1, corridor smoke 1, hall clean 1, corridor clean 1 | 1,530 |
| Selection | Hall smoke 2, hall clean 2 | 717 |
| Calibration | Corridor smoke 2, tree clean 1 | 572 |
| Test | Tree smoke 1, tree clean 2 | 768 |

The run uses up to 30 epochs, with early stopping and a four-hour training budget. If interrupted after a checkpoint, use the same environment and config with the CLI below. Then evaluate the resulting bundle if `test_metrics.json` is missing.

```powershell
..venv\Scripts\python.exe -m smoke_trainer.cli train --config trainer/configs/car_global_pointnet.toml --resume-run runs/car-v4-pointnet --device cuda
..venv\Scripts\python.exe -m smoke_trainer.cli evaluate --bundle runs/car-v4-pointnet/bundle.pt --manifest data/training/car_binary_v4/dataset_manifest.json --split experiments/splits/car_whole_recording_v1.json --partition test --output runs/car-v4-pointnet/test_metrics.json --device cuda
```

Keep `history.json`, `bundle.pt`, `selection_metrics.json`, and `test_metrics.json` under the run directory. Report best epoch, test average precision, precision, recall, F1, false positive rate, and test support at the threshold from calibration. Label `255` excludes points from loss and scoring. Report the scores as agreement with the automatic v4 labels on the held-out drives. The tree smoke drive tests transfer to a different smoky layout, and tree clean 1 contributes to calibration. Independent manual evaluation is a separate task.
