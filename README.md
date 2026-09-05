# LiDAR Smoke Detection

Master's thesis project: create labeled Livox MID-360 data and train a neural network to identify potentially smoke-impacted LiDAR returns in a live ROS 2 pipeline.

| Component | Status |
| --- | --- |
| [Labeler](labeler/README.md) | Working offline pseudo-labeler for stationary raw Livox recordings |
| [Data](data/README.md) | Recording metadata, dataset manifests, and QC summaries; large files stored separately |
| NN predictor | Planned: single-frame or short temporal input, predicting labels for the newest frame |

## Current workflow

1. Record a clean reference, separate clean control, and smoke recordings with the sensor and scene fixed.
2. Run the labeler on `/livox/lidar` (`livox_ros_driver2/msg/CustomMsg`) in MCAP bags.
3. Review clean-control statistics and colored previews before accepting the pseudo-labels.
4. Use the saved frames and labels for model training. Split complete sessions across training and evaluation.

The current labeler detects returns significantly closer than expected clean surfaces. It does not provide complete smoke-volume ground truth or support sensor motion. The future predictor will use live sensor features without a clean reference.

Start with the [labeler installation and commands](labeler/README.md). See [data setup](data/README.md) before running on a fresh clone.
