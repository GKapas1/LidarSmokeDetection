# LiDAR Smoke Detection

Master's thesis project for detecting smoke-impacted Livox MID-360 returns with a
neural network and publishing the result in ROS 2.

## Current status

| Component | Status |
| --- | --- |
| [Offline labeler](labeler/README.md) | Supports stationary ROS 2 Livox recordings and moving-sensor GrandTour ROS 1 sessions |
| ARC-5 clean reference | Built from pose-aligned MID-360 scans, with a disjoint clean interval reserved for validation |
| ARC-6 labeled dataset | Complete: 3,609 frames and 47,306,636 points with reviewed per-point labels |
| [Training export](data/README.md) | Complete: seven frame-preserving, sensor-frame NPZ chunks with model-safe inputs |
| Neural network and ROS 2 predictor | Not implemented in this repository yet |

The finalized ARC-6 labels contain 47,061,035 unimpacted points, 221,980
smoke-impacted points, and 23,621 points ignored during training. Validation on
1,033 held-out ARC-5 clean frames produced 316 effective false-positive points
out of 13,490,744 covered points (0.00234%).

## Label meanings

| Value | Meaning |
| ---: | --- |
| `0` | Unimpacted return, including reviewed hard negatives such as a person near the smoke source |
| `1` | Smoke-impacted return accepted by the reviewed labeling rule |
| `255` | Invalid, isolated, outside verified coverage, or otherwise ignored by the training loss |

The labeler retains automatic geometric labels and review metadata in its derived
labeling artifacts. The training export contains only the corrected final `label`
and sensor-available MID-360 inputs. This prevents clean-map and alignment data from
leaking into a model that must operate from live sensor measurements.

## Repository layout

| Path | Contents |
| --- | --- |
| `labeler/` | Python package, commands, configuration, tests, and labeler documentation |
| `data/` | Local raw/derived-data layout and artifact documentation |
| `GKapas_ThesisProposal.pdf` | Thesis proposal and research context |

Large bags, point clouds, derived arrays, and training artifacts remain local and
are excluded from Git. Start with the [labeler workflow](labeler/README.md) and
[data layout](data/README.md).
