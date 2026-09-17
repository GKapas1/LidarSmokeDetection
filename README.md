# LiDAR Smoke Detection

Master's thesis project for detecting smoke-impacted Livox MID-360 returns with a
neural network and publishing the result in ROS 2.

## Current status

| Component | Status |
| --- | --- |
| [Offline labeler](labeler/README.md) | Supports stationary ROS 2 Livox recordings and moving-sensor GrandTour ROS 1 sessions |
| ARC-5 clean reference | Built from pose-aligned MID-360 scans, with a disjoint clean interval reserved for validation |
| ARC-6 labeled dataset | Complete: 3,609 frames and 47,306,636 points with reviewed per-point labels |
| [Unified training dataset](data/README.md) | Complete: stationary ROS 2 and ARC-6 data share one validated, model-safe NPZ contract |
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

The canonical training dataset combines both current acquisition domains without
mixing their recording boundaries. Every chunk has the same array names, dtypes,
time units, and label meanings. The labeler retains automatic geometric labels and
review metadata only in its derived labeling artifacts; the unified export contains
the corrected final `label` and sensor-available MID-360 inputs.
The current unified dataset contains 5,081 frames and 76,782,092 points across
nine independently loadable chunks.

## Repository layout

| Path | Contents |
| --- | --- |
| `labeler/` | Python package, commands, configuration, tests, and labeler documentation |
| `data/` | Local raw/derived-data layout and artifact documentation |
| `GKapas_ThesisProposal.pdf` | Thesis proposal and research context |

Large bags, point clouds, derived arrays, and training artifacts remain local and
are excluded from Git. Start with the [labeler workflow](labeler/README.md) and
[data layout](data/README.md).
