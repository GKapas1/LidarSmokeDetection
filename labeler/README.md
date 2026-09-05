# Smoke LiDAR Labeler

Offline pseudo-labeling for Livox MID-360 recordings. Produces per-point labels and preserves ROS frame boundaries for training a single-frame or temporal smoke detector.

**Python:** 3.11+ · **Input:** MCAP bags containing `/livox/lidar` as `livox_ros_driver2/msg/CustomMsg`.

## Recording requirements

Keep the LiDAR pose and room geometry unchanged throughout a recording group. Record the clean bags before introducing smoke; avoid people or moved objects in the scene.

| Bag name | Purpose |
| --- | --- |
| `clean_pos01_ref_001` | Clean directional reference |
| `clean_pos01_control_001` | Separate clean recording for calibration and validation |
| `smoke_pos01_low_001` | Smoke recording to label |
| `smoke_pos01_*` | Additional smoke conditions, processed automatically |

Each bag directory contains its `.mcap` file(s) and `metadata.yaml`. Change `pos01` consistently for another position.

## Install

Keep `pyproject.toml` inside `labeler/`. Run:

```bash
cd ~/Smoke/LidarSmokeDetection/labeler
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
chmod +x scripts/run_raw_session.sh
```

## Generate a dataset

Run these commands in the same terminal. Set `DATA_ROOT` to the directory containing your recording group and use a new output name for each run.

```bash
REPO="$HOME/Smoke/LidarSmokeDetection"
cd "$REPO/labeler"

DATA_ROOT="$REPO/data/towel_test/raw_bags/20260901_lab01"
SESSION_ID="20260901_lab01_pos01_v040"
OUTPUT="$REPO/data/towel_test/labeled_sets/$SESSION_ID"

.venv/bin/smoke-label inspect \
  "$DATA_ROOT/clean_pos01_ref_001" --topic /livox/lidar

./scripts/run_raw_session.sh \
  "$DATA_ROOT" pos01 "$SESSION_ID" "$OUTPUT"
```

The script processes the clean control and every `smoke_pos01_*` bag. Arguments are `DATA_DIR POSITION SESSION_ID OUTPUT_DIR`. Set `OUTPUT` to another destination, such as `data/towel_test/labeled_sets/$SESSION_ID`, when needed; no script edit is necessary.

Run the script itself—do not paste its internal `$1` or `BASH_SOURCE` setup into the terminal. After opening a new terminal, define `OUTPUT` again before using the review commands below.

## Check the result

```bash
.venv/bin/python -m json.tool "$OUTPUT/dataset_summary.json"

QT_QPA_PLATFORM=xcb CloudCompare \
  "$OUTPUT/qc/smoke_pos01_low_001_preview.ply"
```

Change the preview filename to match your recording.

- Check `usable_band`: unsupported ranges receive no training labels.
- Inspect the held-out clean false-positive and ignore rates together.
- Check warnings and `truncated_at_configured_point_limit`. Increase limits if required data was omitted.
- Preview colors: green = unaffected, red = smoke impacted, grey = ignore. Investigate red points on unrelated surfaces or in clean-control data.

## How labeling works

1. Sample the clean reference and group returns into angular cells. Store the median range in stable cells; reject sparse cells and cells with excessive depth variation.
2. Use the first half of the separate clean control to calibrate range-dependent thresholds: the 99th percentile of absolute residuals for normal variation and the 99.9th percentile of early-return residuals for smoke candidates, subject to configured minimums.
3. Compare each target point with its expected clean range. Disable bands with insufficient calibration or smoke thresholds above 0.25 m. Use the second half of the clean control for held-out validation.
4. Save point labels with original frame boundaries and timestamps, then create temporal-window metadata. Computation is vectorized across recordings; five-frame windows do not determine the labels.

| Label | Meaning |
| ---: | --- |
| `0` | Consistent with normal clean-range variation |
| `1` | Significantly closer than the expected clean surface: smoke-impact candidate |
| `255` | Invalid, unsupported, or uncertain; exclude from training loss |

## Outputs and training

| Output | Use |
| --- | --- |
| `recordings/*.npz` | Point features, labels, validity masks, confidence, and frame indexing |
| `windows.jsonl` | Five consecutive input frames, with supervision on the newest frame |
| `dataset_schema.json` | Array names, types, shapes, and semantics |
| `dataset_summary.json` | Counts, thresholds, validation results, and warnings |
| `effective_config.json` | Exact merged configuration, including defaults |
| `run_provenance.json` | Run arguments, UTC start time, code revision/hashes, and dependency versions |
| `clean_reference.npz` | Reference and calibration used by the labeler |
| `qc/` | Sampled previews, recording summaries, and teacher diagnostics |

Each frame is stored once. Slice its arrays using `frame_ptr[k]:frame_ptr[k+1]`. A single-frame model can use the target frame alone; temporal models use the window indices. Windows crossing invalid timing gaps are skipped.

Use sensor-available features such as XYZ, reflectivity, and relative point timing as model inputs. Clean-reference values, residuals, labels, and teacher confidence must not become input features. The live predictor will not require a clean reference.

Split by complete sessions or positions, not random points or overlapping windows. The generated split assignments remain unassigned until you choose them.

## Configuration and maintenance

Edit `config/raw_dataset.toml`:

| Setting | Default |
| --- | --- |
| Reference / calibration point stride | `2` / `2` |
| Reference / calibration point cap | `8,000,000` each |
| Target stride / point cap | `1` / `30,000,000` |
| Angular cell size | `0.30°` |
| Valid input range | `0.10–30.0 m` |
| Window length / maximum frame gap | `5` / `0.25 s` |

Sampling reduces reference/calibration cost. To use more clean data, reduce its stride and raise its cap. Target points are retained at stride 1 up to the configured cap, keeping complete frames. Decoding, reference construction, and compressed output can take minutes on a laptop.

Source roles: `bag.py` reads MCAP; `core.py` builds references and labels; `raw_dataset.py` exports recordings and windows; `pipeline.py` supplies configuration, inspection, and previews; `cli.py` exposes commands.

Run tests from `labeler/`:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
