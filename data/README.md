# Data layout

Large sensor recordings and generated datasets are stored under `data/` locally
and are excluded from Git. The repository tracks the code and TOML configurations
needed to reproduce them.

## Stationary ROS 2 recordings

Stationary experiments use MCAP bag directories containing `metadata.yaml`. A
recording group normally lives under `data/towel_test/raw_bags/<session>/` and
contains one clean reference, one independent clean control, and one or more
`smoke_pos01_*` bags. Generated stationary datasets live under
`data/towel_test/labeled_sets/`. MCAP/DB3 payloads and generated NPZ/PLY artifacts
are ignored; small metadata, schemas, manifests, and QC summaries may be tracked.

## GrandTour source files

The implemented ARC-5 and ARC-6 workflow expects four ROS 1 bags per session:

```text
data/grandtour/
├── arc5/
│   ├── 2024-11-18-16-59-23_livox.bag
│   ├── 2024-11-18-16-59-23_livox_undist.bag
│   ├── 2024-11-18-16-59-23_dlio.bag
│   └── 2024-11-18-16-59-23_tf_minimal.bag
└── arc6/
    └── corresponding 2024-11-18-17-13-09 bags
```

The raw MID-360 topic is `/boxi/livox/points`; the point-corresponding deskewed
cloud is `/boxi/livox/points_undistorted`. DLIO poses and the mission calibration
are used only to position MID-360 observations for offline reference comparison.

## Generated GrandTour artifacts

```text
data/grandtour/
├── prepared/
│   ├── reference/               # ARC-5 stable voxel reference
│   ├── validation/              # held-out ARC-5 clean validation
│   └── arc6_full_complete/      # reviewed labels, QC PLYs, and full manifest
└── training/
    └── arc6_reviewed_v1/        # model-safe NPZ chunks and schema
```

The authoritative training export is described by
`training/arc6_reviewed_v1/training_manifest.json`. Its seven NPZ chunks preserve
all 3,609 frame boundaries and contain:

- `xyz` in the original `livox_lidar` sensor frame;
- `intensity`, `tag`, `line`, and `relative_time_ns`;
- the corrected final `label`;
- frame and original point indices.

Teacher-only fields such as ARC-5 world coordinates, nearest-reference distance,
automatic labels, and review reasons remain in `prepared/arc6_full_complete` and
are deliberately absent from the training export.

The GrandTour recordings originate from the
[GrandTour dataset](https://grand-tour.leggedrobotics.com/). Raw downloads and
derived point data remain outside version control; repository commits contain
only the adapter, reviewed configuration, and documentation.

See the [labeler README](../labeler/README.md) for the complete reproduction
commands and validation procedure.
