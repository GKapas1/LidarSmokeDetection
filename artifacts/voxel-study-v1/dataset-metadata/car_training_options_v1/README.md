# Car and ARC-5 training options v1

The shared dataset is `data/training/car_training_options_v1/`. It contains all ten car surface-v2 recordings and the two ARC-5 chunks covering 95–195 seconds. The 88–95 and 195–204 second transition buffers are excluded from this package.

Eight configurations implement the four car recipes, each with a paired run that includes outdoor clean ARC-5 scans. All use G1, the same architecture, seed, optimizer settings, car recording assignments and a ceiling of 1,400 optimizer steps per epoch. Early stopping uses car selection AP, with at most 30 epochs and a four-hour limit per run. The number of supervised points can vary with augmentation. These are training preparation artifacts.

| Recipe prefix | Training treatment |
| --- | --- |
| `a_baseline` | Original car scans |
| `b_distant_smoke` | Copy small smoke patches to supported free space at 3–8 m |
| `c_background_mask` | Randomly remove a sector of distant scan context |
| `d_both` | Apply smoke copying and context masking together |

Each prefix has `_car_only` and `_with_arc5` configurations in the dataset's `configs/` directory. The complete inventory is in `experiments.json`.

The car training pool has 1,530 scans. The with-ARC5 pool adds every fourth scan from the outdoor core, approximately 2.5 Hz, for 250 extra clean scans and 1,780 scans in total. All 1,000 core ARC-5 scans are present in the package. The other 750 ARC-5 scans have excluded split assignments. The added 250 scans contribute 3,231,170 clean-labeled returns. Each epoch shuffles the selected pool and stops at 1,400 supervised optimizer steps, subject to the time limit. This controls the optimizer-step budget while allowing scan selection to vary across epochs.

Car selection, calibration and test assignments are inherited from the existing whole-recording split. Normalization is fitted on each run's training split before geometric augmentation. Selection and calibration operate on the original car scans. Training records per-domain scan counts and augmentation point counts in `history.json`. ARC-6 supplies a later external evaluation.

Smoke copying is attempted on 50% of training scans. A seed is selected from existing smoke closer than 3 m, and nearby smoke points within 0.35 m form a patch. The chosen target distance is retained through up to 16 placement attempts. A destination requires at least eight measured returns in an angular cone behind the entire patch, with no closer measured obstruction and at least 0.25 m clearance from measured geometry. Ignored returns participate in obstruction checks. Sampling the copied patch with squared source-to-target range ratio approximates the reduction in angular footprint. Successful copies contain at least eight points, all at 3–8 m. Original source smoke is retained.

This translation preserves the patch's physical size and raw intensities. It approximates a distant observation. Real distant smoke can differ in attenuation, beam interactions, return availability and temporal evolution. Some frames have no valid distant placement. The audit and epoch histories report actual copied point counts.

Context masking is attempted on 50% of training scans. Its range threshold is sampled uniformly from 8–12 m and its sector width from 90–180 degrees, with random azimuth. Points beyond the threshold within that sector are removed. The decision uses geometry alone and applies to both clean and smoky recordings. The configured mask starts beyond the synthetic smoke range. The original datasets retain every source point and label. These augmentations currently support the G1 training entry point.

The car labels are the existing surface-v2 pilot. They include 2,440 smoke-to-background revisions from surface matches and reviewed overrides. The automatic revisions still require annotation review. The ARC-5 clean assumption follows the user's video review. LiDAR route review supports the buffered outdoor core, with independently verified video-timer synchronization still pending. Some outdoor scans can observe shared building facades or interiors through openings.

Extract `car_training_options_v1_windows.zip` at the repository root. It supplies the complete shared dataset, configurations, updated trainer sources, tests and Windows scripts. The updated trainer code is required for these recipes. Use the CUDA environment described in `docs/CAR_WINDOWS_TRAINING.md`.

Run all eight configurations in PowerShell from the repository root.

```powershell
.\trainer\scripts\train_car_options_windows.ps1
```

Run an individual option with a distinct run prefix when needed.

```powershell
.\trainer\scripts\train_car_options_windows.ps1 `
  -Recipes d_both_with_arc5 `
  -RunPrefix car-options-v1
```

The launcher verifies the dataset and CUDA environment, then runs the supplied configurations sequentially. Each training run creates its own portable model bundle, checkpoints, history, car selection report, calibration plot and car test report under `runs/`. Choose a new run prefix for a repeat. Checkpoint resume also verifies the augmentation configuration and optimizer-step limit.

To prepare the shared dataset again from the original car and ARC-5 exports, use an unused output directory.

```bash
trainer/.venv/bin/python trainer/scripts/prepare_car_options.py
```

The preparation script verifies source checksums and validates all chunks and splits. The small curated files under `artifacts/car-training-options-v1/` preserve the configs, split assignments and audit reports. `dataset_verification.json` checks all source car chunk hashes, car role assignments, complete split coverage and all eight configuration paths. `augmentation_verification.json` records seeded augmentation checks on sampled training scans, including original smoke retention and synthetic 5–8 m examples. `augmentation-preview.png` shows one real training scan before and after augmentation. All 31 trainer tests passed on Linux, including the training entry point, model packaging, resume checks and existing callers. The PowerShell launcher received static review.
