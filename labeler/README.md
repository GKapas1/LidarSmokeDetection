# Smoke LiDAR Labeler

Offline per-point pseudo-labeling for Livox MID-360 recordings. The package keeps
original frame boundaries and produces final labels suitable for point-cloud model
training.

It provides two independent workflows:

- stationary ROS 2 MCAP recordings containing
  `livox_ros_driver2/msg/CustomMsg` on `/livox/lidar`;
- moving-sensor GrandTour ROS 1 sessions containing MID-360 `PointCloud2`, DLIO
  poses, and mission calibration.

Python 3.11 or newer is required.

## Installation

```bash
cd ~/Smoke/LidarSmokeDetection/labeler
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
```

Run the tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Labels

| Value | Name | Training meaning |
| ---: | --- | --- |
| `0` | Unimpacted | Include as the negative class |
| `1` | Smoke impacted | Include as the positive class |
| `254` | Excluded | Internal review code; converted to `255` by the training exporter |
| `255` | Unknown | Ignore in the training loss |

The derived labeling files retain `automatic_label` and final `label` separately.
The final label includes accepted human-review corrections. A model trains against
`label`; `automatic_label` is teacher metadata and must not be a model input.

## Stationary ROS 2 workflow

A stationary recording group consists of a clean reference, an independent clean
control, and one or more smoke recordings. The LiDAR pose and room geometry must
remain fixed within a group.

| Bag name | Purpose |
| --- | --- |
| `clean_pos01_ref_001` | Build the directional clean reference |
| `clean_pos01_control_001` | Calibrate thresholds and measure clean false positives |
| `smoke_pos01_*` | Generate smoke-impact labels |

From `labeler/`, inspect and process a recording group with:

```bash
DATA_ROOT="$HOME/Smoke/LidarSmokeDetection/data/towel_test/raw_bags/20260901_lab01"
OUTPUT="$HOME/Smoke/LidarSmokeDetection/data/towel_test/labeled_sets/20260901_lab01_pos01"

.venv/bin/smoke-label inspect \
  "$DATA_ROOT/clean_pos01_ref_001" --topic /livox/lidar

./scripts/run_raw_session.sh \
  "$DATA_ROOT" pos01 20260901_lab01_pos01 "$OUTPUT"
```

The stationary algorithm builds stable angular clean-range cells, calibrates
range-dependent thresholds on the independent control recording, and labels target
returns by their early-return residual. Its outputs include frame-preserving NPZ
recordings, temporal-window metadata, schemas, provenance, summaries, and sampled
CloudCompare previews.

## GrandTour input contract

Place these four bags together under each of `data/grandtour/arc5` and
`data/grandtour/arc6`:

- `*_livox.bag`
- `*_livox_undist.bag`
- `*_dlio.bag`
- `*_tf_minimal.bag`

The implemented sessions are ARC-5 `2024-11-18-16-59-23` and ARC-6
`2024-11-18-17-13-09`. The recordings originate from the
[GrandTour dataset](https://grand-tour.leggedrobotics.com/); raw and derived data
remain local and are excluded from Git.

The reader uses:

| Source | Topic/frame | Use |
| --- | --- | --- |
| Raw MID-360 | `/boxi/livox/points`, `livox_lidar` | Model inputs and original point identity |
| Deskewed MID-360 | `/boxi/livox/points_undistorted` | Offline reference geometry |
| DLIO odometry | `/boxi/dlio/lidar_map_odometry` | Scan positioning |
| Minimal TF | `box_base`, `hesai_lidar`, `livox_lidar` | Hesai-to-MID-360 transform chain |

Raw and deskewed clouds are required to have identical message timestamps, point
counts, ordering, intensity, tag, line, and per-point timestamps. Only deskewed XYZ
may differ. The loader rejects a mismatch instead of guessing correspondences.

Published Hesai/IMU-derived poses are used only for positioning MID-360 scans. All
reference points, labels, and exported model features remain MID-360 observations.

## GrandTour workflow

Run these commands from `labeler/`.

Inspect the source schema and paired-cloud identity:

```bash
.venv/bin/smoke-label grandtour-inspect \
  ../data/grandtour/arc5 --sample-frames 5
```

Create optional map previews for CloudCompare alignment checks:

```bash
.venv/bin/smoke-label grandtour-preview \
  ../data/grandtour/arc5 \
  --output ../data/grandtour/prepared/arc5_preview.ply
```

Build the stable ARC-5 clean reference. Frames before 300 seconds build the map;
the later interval remains disjoint for clean validation.

```bash
.venv/bin/smoke-label grandtour-reference \
  ../data/grandtour/arc5 \
  --config config/grandtour/arc5_reference.toml \
  --output ../data/grandtour/prepared/reference
```

The reviewed rigid ARC-6-to-ARC-5 transform is stored in
`config/grandtour/arc6_to_arc5.toml`. Generate a 60-second review pilot and then
the complete session:

```bash
.venv/bin/smoke-label grandtour-pilot \
  ../data/grandtour/arc6 \
  --config config/grandtour/arc6_pilot.toml \
  --reference ../data/grandtour/prepared/reference/arc5_clean_reference.npz \
  --output ../data/grandtour/prepared/pilot

.venv/bin/smoke-label grandtour-session \
  ../data/grandtour/arc6 \
  --config config/grandtour/arc6_full.toml \
  --reference ../data/grandtour/prepared/reference/arc5_clean_reference.npz \
  --output ../data/grandtour/prepared/arc6_full_complete
```

The full run is divided into seven bounded time chunks. Each chunk contains its
labeling NPZ, summary, combined CloudCompare preview, smoke-only preview, hard
negatives, and isolated returns. Preview colors are green for unimpacted, red for
smoke impacted, grey for unknown, and purple for explicit exclusions.

The ARC-6 review configuration records the accepted hallway smoke region, the
person at the smoke source as an unimpacted hard negative, the expanded room crop,
and a 0.5 m same-frame isolation rule for solitary sensor errors.

## Clean validation

Validate the final rule against the ARC-5 interval excluded from reference-map
construction:

```bash
.venv/bin/smoke-label grandtour-validate-clean \
  ../data/grandtour/arc5 \
  --reference-config config/grandtour/arc5_reference.toml \
  --label-config config/grandtour/arc6_full.toml \
  --reference ../data/grandtour/prepared/reference/arc5_clean_reference.npz \
  --output ../data/grandtour/prepared/validation
```

The current result covers 13,490,744 points in 1,033 clean frames. The effective
reviewed rule produces 316 false-positive points, or 0.00234%. The output directory
contains the JSON report and CloudCompare PLYs for both the effective reviewed rule
and the broader automatic-distance diagnostic.

## Training export

Export the reviewed labels without teacher-only features:

```bash
.venv/bin/smoke-label grandtour-export-training \
  ../data/grandtour/prepared/arc6_full_complete/arc6_full_manifest.json \
  --output ../data/grandtour/training/arc6_reviewed_v1
```

The export contains seven NPZ chunks with 3,609 frames and 47,306,636 points:

| Label | Points |
| --- | ---: |
| Unimpacted (`0`) | 47,061,035 |
| Smoke impacted (`1`) | 221,980 |
| Ignore (`255`) | 23,621 |

Model input arrays are `xyz`, `intensity`, `tag`, `line`, and `relative_time_ns`.
`xyz` remains in the original `livox_lidar` sensor frame. Frame reconstruction uses
`frame_ptr`, `frame_index`, and `frame_time_ns`.

The exporter deliberately omits `world_xyz`, nearest-reference distance, automatic
labels, clean-map identifiers, and review metadata. These fields are useful for
offline auditing but would leak information unavailable to a live predictor.

### Canonical stationary + GrandTour export

Convert both current datasets to the exact same trainer-facing contract:

```bash
.venv/bin/smoke-label unified-export \
  --stationary ../data/towel_test/labeled_sets/20260901_lab01_pos01_v040 \
  --grandtour ../data/grandtour/training/arc6_reviewed_v1/training_manifest.json \
  --output ../data/training/unified_v1
```

The command preserves every source recording or GrandTour chunk as a separate
file and writes `dataset_manifest.json` plus `dataset_schema.json`. It standardizes
stationary `reflectivity` and GrandTour `intensity` as raw `float32` `intensity`,
and standardizes all point timing as `float32` seconds in `point_offset_s`.

Use the same validated loader for every chunk:

```python
import json
from pathlib import Path

from smoke_labeler.unified_dataset import load_unified_chunk

root = Path("../data/training/unified_v1")
manifest = json.loads((root / "dataset_manifest.json").read_text())
for item in manifest["chunks"]:
    chunk = load_unified_chunk(root / item["path"])
    train_inputs = (
        chunk["xyz"], chunk["intensity"], chunk["tag"],
        chunk["line"], chunk["point_offset_s"],
    )
    target = chunk["label"]
    loss_mask = target != 255
```

Metadata such as `source_domain` and `condition` supports auditing and split
construction but is forbidden as a network input. Splits remain unassigned so a
training implementation can choose and record a group-safe split policy.

## Implementation map

| Module | Responsibility |
| --- | --- |
| `bag.py` | Stationary ROS 2 MCAP decoding |
| `core.py` | Stationary directional references and labels |
| `raw_dataset.py` | Stationary dataset and temporal-window export |
| `geometry.py` | Pose construction, interpolation, and point transforms |
| `grandtour.py` | ROS 1 PointCloud2 decoding, paired MID-360 loading, TF, and DLIO poses |
| `grandtour_pipeline.py` | GrandTour reference building, labeling, validation, QC, and training export |
| `unified_dataset.py` | Canonical conversion, schema validation, and shared training loader |
| `cli.py` | Command-line interface |

See [GRANDTOUR_IMPLEMENTATION.md](GRANDTOUR_IMPLEMENTATION.md) for the exact
implemented geometry, review decisions, and artifact contract.

## Scope of the labels

These labels identify anomalous returned points consistent with smoke impact. They
do not reconstruct the full smoke volume or directly label missing returns. A
structure-matching return also does not prove that its beam experienced no smoke
attenuation. The training export is therefore per-return pseudo-ground truth for
the detector defined by this project.
