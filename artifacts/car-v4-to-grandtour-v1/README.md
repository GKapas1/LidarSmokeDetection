# Car v4 to GrandTour investigation

Results from 2026-11-24, code base commit `2224d98e05c6ccc7e1b67bd5d71d5f3480909830` plus the two car configurations and GrandTour split included in this change. The training machine used Windows, an NVIDIA RTX 3090, Python 3.12 and PyTorch 2.7.0+cu126. See `environment-pip-freeze.txt` for the installed packages; the editable package path and resolved `config.json` paths reflect that machine and are not portable.

## Protocol

All three models used `data/training/car_binary_v4`, the committed whole-recording split `experiments/splits/car_whole_recording_v1.json`, train-only fitted normalization, up to 30 epochs and early stopping after five epochs without improved selection average precision. L1, G1, and GT stopped after 6, 19, and 13 epochs; their selected epochs were 1, 14, and 8. No diagnostic frame or epoch limits were applied. The split contains 1,530 training frames; L1 and G1 reported 1,401 processed training frames per epoch because frames with no usable supervised points are skipped. GT used 1,387 supervised temporal windows per epoch. These counts are in the histories.

Probability calibration is fitted on the car calibration partition; the operating threshold is selected on the car selection partition. GrandTour and the car test recordings were not used for either step. This corrects the earlier chat description that called the threshold calibrated on the calibration partition. Car labels are automatic v4 labels; scores indicate agreement with those labels, not an independent human ground truth.

The external test uses every `grandtour_ros1` chunk in `data/training/unified_v1` and no stationary chunks. The saved split `experiments/splits/grandtour_all_external_test_v1.json` has all 3,609 GrandTour frames in `roles.test`. It was derived from the existing chronological split: its inherited `fractions`, `streams.boundaries`, and `purge_frames_each_side` fields describe that earlier split and do not describe this all-test assignment. The authoritative frame assignments are `roles` and `excluded`; the split hash covers the entire saved file.

GT requires five consecutive same-chunk scans within 0.25 s and therefore scores 3,581 GrandTour frames, 28 fewer than L1 and G1, and 760 rather than 768 car test frames. Comparisons are not exactly on identical points. Ignore label 255 is excluded. The test results use the car-selected thresholds; no GrandTour threshold tuning was performed.

## Results

| Model | Best epoch | Car test AP | Car test F1 | GrandTour AP | GrandTour precision | GrandTour recall | GrandTour F1 | GrandTour FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L1 | 1 | 0.68219 | 0.56278 | 0.06848 | 0.01469 | 0.61430 | 0.02870 | 0.19430 |
| G1 | 14 | 0.68474 | 0.63366 | 0.05422 | 0.04836 | 0.38753 | 0.08598 | 0.03597 |
| GT | 8 | 0.54238 | 0.55998 | 0.03510 | 0.04250 | 0.35958 | 0.07602 | 0.03786 |

The GrandTour evaluation has 221,980 positive and 47,283,015 scored points for L1/G1; GT has 218,274 positive and 46,916,532 scored points. The reports retain confusion counts, ROC-AUC, thresholds, prevalence and calibration parameters. These results identify a cross-domain performance problem but do not establish its cause.

## Files and transfer

Each model directory contains the portable `bundle.pt`, best and last training checkpoints, resolved configuration, train split, normalizer, epoch history, selection and test reports, and calibration plot. `best.pt` and `last.pt` include optimizer state; `bundle.pt` is intended for inference. Data and `.venv` remain excluded from Git. The preserved manifests and schemas in `datasets/` document provenance, hashes and label contracts, but do not contain point clouds. Their source paths are historical provenance and are not needed by the trainer.

On the other machine, transfer the complete `data/training/car_binary_v4/` and `data/training/unified_v1/` directories separately, keeping the manifest and chunks byte-identical. The external split is bound to the SHA-256 of the whole unified manifest, even though it uses only the GrandTour chunks. Install the trainer as documented in `trainer/README.md`. Check the copied data with:

```bash
python -m smoke_trainer.cli inspect --manifest data/training/car_binary_v4/dataset_manifest.json --verify-checksums
python -m smoke_trainer.cli inspect --manifest data/training/unified_v1/dataset_manifest.json --verify-checksums
```

For example, reproduce G1 GrandTour evaluation from the repository root, substituting `--device cuda` if available:

```bash
python -m smoke_trainer.cli evaluate --bundle artifacts/car-v4-to-grandtour-v1/car-v4-pointnet/bundle.pt --manifest data/training/unified_v1/dataset_manifest.json --split experiments/splits/grandtour_all_external_test_v1.json --partition test --output runs/car-v4-pointnet/grandtour_test_repeat.json --device cpu
```

Use `car-v4-l1` or `car-v4-gt` in the bundle path for the other models. To reproduce the car test, use the car manifest, `experiments/splits/car_whole_recording_v1.json` and the `test` partition. Configurations for new training runs are in `trainer/configs/car_local.toml`, `car_global_pointnet.toml`, and `car_global_temporal.toml`.

Potential next steps: inspect false positives by recording, range, intensity, and geometry; compare frame-matched results across models; examine differences in acquisition and labeling; and review representative predictions manually before attributing causes or changing thresholds.