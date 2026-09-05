import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from smoke_labeler.bag import BagPoints
from smoke_labeler.pipeline import DEFAULT_CONFIG
from smoke_labeler.raw_dataset import run_raw_dataset


def make_raw_points(smoky=False, source="synthetic.mcap"):
    frames = 8
    points_per_frame = 300
    base_angles = np.linspace(-0.4, 0.4, points_per_frame, dtype=np.float32)
    angles = np.tile(base_angles, frames)
    ranges = np.full(len(angles), 5.0, dtype=np.float32)
    if smoky:
        ranges[np.arange(len(ranges)) % 20 == 0] = 2.0
    xyz = np.column_stack(
        (ranges * np.cos(angles), ranges * np.sin(angles), np.zeros(len(angles)))
    ).astype(np.float32)
    frame_index = np.repeat(np.arange(frames, dtype=np.int32), points_per_frame)
    frame_time = np.arange(frames, dtype=np.float64) * 0.1
    point_offset = np.tile(np.linspace(0.0, 0.09, points_per_frame), frames)
    frame_ptr = np.arange(0, len(xyz) + 1, points_per_frame, dtype=np.int64)
    return BagPoints(
        xyz=xyz,
        reflectivity=np.full(len(xyz), 30.0, np.float32),
        tag=np.zeros(len(xyz), np.uint8),
        line=np.zeros(len(xyz), np.uint8),
        time_s=frame_time[frame_index] + point_offset,
        frame_index=frame_index,
        topic="/livox/lidar",
        message_type="livox_ros_driver2/msg/CustomMsg",
        source_files=[source],
        frame_ptr=frame_ptr,
        frame_time_s=frame_time,
    )


class RawDatasetTests(unittest.TestCase):
    def test_builds_frame_preserving_dataset(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["sampling"].update(
            reference_point_stride=1,
            calibration_point_stride=1,
            target_point_stride=1,
            max_reference_points=100000,
            max_calibration_points=100000,
            max_target_points=100000,
        )
        config["reference"].update(angular_resolution_deg=1.0, min_returns_per_cell=3)
        config["calibration"].update(
            range_band_edges_m=[0.0, 10.0],
            minimum_calibration_samples_per_band=100,
        )
        config["dataset"].update(sequence_length=5, maximum_frame_gap_s=0.25)
        config["output"]["preview_max_points"] = 500

        side_effect = [
            make_raw_points(source="clean_ref.mcap"),
            make_raw_points(source="clean_control_cal.mcap"),
            make_raw_points(source="clean_control.mcap"),
            make_raw_points(smoky=True, source="smoke_low.mcap"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            with patch("smoke_labeler.raw_dataset.read_bag_points", side_effect=side_effect):
                summary = run_raw_dataset(
                    config,
                    "clean_ref",
                    "clean_control",
                    ["smoke_low"],
                    directory,
                    "/livox/lidar",
                    "lab01_pos01",
                )

            output = Path(directory)
            self.assertTrue((output / "dataset_summary.json").exists())
            self.assertTrue((output / "dataset_schema.json").exists())
            self.assertEqual(json.loads((output / "effective_config.json").read_text()), config)
            provenance = json.loads((output / "run_provenance.json").read_text())
            self.assertEqual(provenance["invocation"]["session_id"], "lab01_pos01")
            self.assertIn("core.py", provenance["source_sha256"])
            self.assertEqual(summary["run_provenance_file"], "run_provenance.json")
            self.assertTrue((output / "windows.jsonl").exists())
            smoke_file = output / "recordings/smoke_low.npz"
            self.assertTrue(smoke_file.exists())
            data = np.load(smoke_file)
            self.assertEqual(data["frame_ptr"].tolist(), list(range(0, 2401, 300)))
            self.assertGreater(np.count_nonzero(data["label"] == 1), 0)
            self.assertNotIn("expected_clean_range_m", data.files)
            self.assertNotIn("range_residual_m", data.files)

            rows = [json.loads(line) for line in (output / "windows.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(row["target_frame"] >= 4 for row in rows))
            self.assertEqual(summary["sequence_length"], 5)
            self.assertEqual(summary["windows"], 8)


if __name__ == "__main__":
    unittest.main()
