# ARC-5 outdoor clean supplement v1

Prepared from the raw MID-360 recording in `data/grandtour/arc5/` for the experiment that adds clean GrandTour observations to car training.

The user identified the smoke-free outdoor passage at approximately 88–204 seconds on the recording timer. Seconds in this export are relative to the first raw Livox header timestamp, matching the existing ARC-5 pipeline. The first header and bag timestamps both equal `1731945633000429630` ns. The LiDAR trajectory and sampled boundary scans support the outdoor interval. Independent synchronization against the video timer remains unverified.

The candidate export retains 88–204 seconds, with 1,161 scans and 15,488,560 raw points. Its supplied split selects 95–195 seconds for training, buffering the approximate doorway transitions. The 161 boundary scans have excluded assignments.

| Training subset | Count |
| --- | ---: |
| Scans | 1,000 |
| Raw returns | 13,378,969 |
| Clean labels, class 0 | 12,923,344 |
| Ignored labels, class 255 | 455,625 |
| Smoke labels, class 1 | 0 |
| Clean returns below 1 m | 29,942 |
| Clean returns at 1–3 m | 7,715,600 |
| Clean returns at 3–5 m | 2,070,244 |
| Clean returns at 5–8 m | 1,118,865 |
| Clean returns above 8 m | 1,988,693 |

The close returns include the ground. These counts measure observations and include repeated views. The first training scan is at 95.099752188 seconds and the last at 194.999878407 seconds.

The clean assumption comes from the user's video review. Finite returns between 0.3 and 25 m inclusive receive label 0 when another in-range return is within 0.5 m in the same frame. Other returns receive label 255. The range limits match car clean controls, and the isolated-return radius matches the existing ARC-6 labeling pipeline. All raw XYZ, intensity, tag, line, point order and point timing are preserved. The export uses original sensor coordinates and the trainer's canonical schema.

The full candidate interval has zero invalid feature values, 421,386 out-of-range returns and 102,576 isolated returns. The existing indoor reference crop is absent from the label calculation. The route plot shows sensor position relative to that crop. Outdoor scans can still observe shared building facades or interiors through openings.

The portable dataset is `data/training/arc5_outdoor_clean_v1/`. Its `split.json` assigns the two middle chunks to training and the first and last chunks to excluded. Its selection, calibration and test assignments are empty. Use this as a clean supplement alongside the car data. When constructing a combined manifest, carry these role assignments into the combined split and update its manifest and chunk hashes. The dataset alone has no positive training examples or model-selection recordings.

The data contains full-rate scans at approximately 10 Hz. Keeping those scans allows temporal training and later controlled sampling. The two 500-scan training chunks support 992 five-scan temporal windows at a maximum interscan gap of 0.25 seconds.

Reproduce from the repository root with a labeler environment installed as described in `labeler/README.md`. Choose an unused output directory when rerunning.

```bash
labeler/.venv/bin/python labeler/scripts/prepare_arc5_outdoor_clean.py \
  --session data/grandtour/arc5 \
  --config labeler/config/grandtour/arc5_outdoor_clean.toml \
  --output data/training/arc5_outdoor_clean_v1
```

Verify the copied dataset with the trainer environment.

```bash
python -m smoke_trainer.cli inspect \
  --manifest data/training/arc5_outdoor_clean_v1/dataset_manifest.json \
  --verify-checksums
```

The transfer archive is `data/training/arc5_outdoor_clean_v1_windows.zip`, with an adjacent SHA-256 file. Extract it under `data/training/` on Windows. The archive includes the dataset, split, frame provenance CSV, labeling configuration and this README. Provenance source paths describe the original Linux preparation and the trainer resolves chunks relative to the manifest.

Review material in this directory includes `route-review.png`, `boundary-scans.png`, `export_verification.json` and `trainer_verification.json`. Every exported raw feature and timestamp was checked against the source recording, and every label was independently recomputed using a radius-neighbor query. The trainer's schema, hashes, split statistics, local and global adapters and temporal window loader were verified.
