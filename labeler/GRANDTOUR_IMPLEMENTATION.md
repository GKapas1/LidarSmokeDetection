# GrandTour implementation record

This document records the implemented ARC-5/ARC-6 labeling pipeline. Runtime
commands and installation instructions are in [README.md](README.md).

## Source sessions

| Role | Session | Local directory |
| --- | --- | --- |
| Clean reference and held-out clean validation | ARC-5 `2024-11-18-16-59-23` | `data/grandtour/arc5` |
| Reviewed smoke labeling | ARC-6 `2024-11-18-17-13-09` | `data/grandtour/arc6` |

Each directory contains raw MID-360, point-corresponding deskewed MID-360, DLIO
odometry, and minimal-TF ROS 1 bags. The native reader preserves PointCloud2 row
layout, endianness, source point index, integer nanosecond time, intensity, Livox
tag, and scan line.

The observed MID-360 point layout is `x`, `y`, `z`, `intensity`, `tag`, `line`, and
`timestamp` with a 26-byte point step. Raw and deskewed frames have matching
message timestamps, point counts, point order, non-XYZ fields, and per-point times.
Raw XYZ remains the model input; deskewed XYZ is used only for offline geometry.

## Transform convention

`T_A_B` maps coordinates expressed in frame B into frame A. For an ARC-6
deskewed scan at time `t`:

```text
p_arc5 = T_arc5_arc6 @ T_arc6_hesai(t) @ T_hesai_livox @ p_livox
```

`T_arc6_hesai(t)` comes from `/boxi/dlio/lidar_map_odometry` and is interpolated
with linear translation and quaternion Slerp. `T_hesai_livox` is composed from
the mission's `box_base` static transforms. Timestamps outside pose coverage and
interpolation gaps larger than 250 ms are rejected.

The reviewed cross-session rigid transform is stored in
`config/grandtour/arc6_to_arc5.toml`. It was recovered from a CloudCompare manual
alignment and validated across sampled frames. Its translation norm is 1.81565 m,
rotation is 0.59224 degrees, and 97.59% of the sampled ARC-6 preview lies within
0.15 m of ARC-5 after alignment.

## ARC-5 reference

`config/grandtour/arc5_reference.toml` builds a 5 cm voxel reference from every
second frame and every fourth point during the first 300 seconds. A voxel must be
observed in at least three distinct scans. The reviewed crop is:

```text
minimum = [-3.5, -9.5, -2.0] m
maximum = [16.0, 12.5, 3.0] m
```

The resulting reference contains 179,833 stable voxels. ARC-5 data after 300
seconds is excluded from reference construction and reserved for validation.

## ARC-6 labeling rule

Nearest-reference distance is evaluated using deskewed, pose-aligned XYZ. The
base automatic thresholds are 0.08 m for structure and 0.25 m for smoke
candidates. Human review established the final ARC-6 decisions recorded in
`config/grandtour/arc6_full.toml`:

- the hallway smoke region is `[-0.4, 1.5, 0.0]` to `[3.3, 4.0, 1.6]` m;
- within that accepted region, a distance of at least 0.08 m is smoke impacted;
- valid reviewed background outside that region is unimpacted;
- the person at the smoke source is retained as a named unimpacted hard negative;
- a valid return with no same-frame neighbor within 0.5 m is ignored as an
  isolated sensor return.

Internal labels are `STRUCTURE=0`, `SMOKE_CANDIDATE=1`, `EXCLUDED=254`, and
`UNKNOWN=255`. The labeling NPZ retains automatic and final labels, reason codes,
human-review state, source identity, world coordinates, and reference distance.

## Final artifacts

The full ARC-6 run spans 0–362 seconds in seven bounded chunks. Of 3,617 raw
frames, 3,609 have published DLIO pose coverage. The reviewed result contains:

| Final label | Points | Fraction |
| --- | ---: | ---: |
| Unimpacted | 47,061,035 | 99.4808% |
| Smoke impacted | 221,980 | 0.4692% |
| Unknown/ignored | 23,621 | 0.0499% |

The labeling manifest is
`data/grandtour/prepared/arc6_full_complete/arc6_full_manifest.json`. Every chunk
has a compressed labeling NPZ plus combined, smoke-only, hard-negative, and
isolated-return CloudCompare PLYs.

## Clean validation

The held-out ARC-5 interval contains 1,033 frames and 13,490,744 covered points.
The final reviewed rule produces 316 false-positive smoke labels (0.00234%) in 12
frames. The global 0.25 m automatic-distance diagnostic produces 3,822 points
(0.02833%). Results and review clouds are stored under
`data/grandtour/prepared/validation`.

## Training contract

`grandtour-export-training` converts internal label `254` to training-ignore
`255` and writes independently loadable sensor-frame chunks. Allowed model inputs
are:

- raw `xyz` in `livox_lidar`;
- `intensity`;
- Livox `tag` and `line`;
- `relative_time_ns`.

The target is the corrected final `label`. Frame and source-point indices are
included as dataset structure. World coordinates, reference distances, automatic
labels, review state, and clean-map identifiers are explicitly forbidden as model
inputs. The schema and hashes are recorded in
`data/grandtour/training/arc6_reviewed_v1/training_manifest.json` and
`dataset_schema.json`.

## Current boundary

The repository implements offline source inspection, reference construction,
alignment loading, reviewed labeling, clean validation, CloudCompare QC export,
and model-safe NPZ export. It does not yet contain neural-network training code or
a live ROS 2 inference node.
