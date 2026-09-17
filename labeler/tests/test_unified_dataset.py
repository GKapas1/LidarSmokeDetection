from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from smoke_labeler.unified_dataset import ARRAY_NAMES, export_unified_dataset, load_unified_chunk


class UnifiedDatasetTests(unittest.TestCase):
    def test_both_domains_have_identical_trainer_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stationary = root / "stationary"
            (stationary / "recordings").mkdir(parents=True)
            np.savez_compressed(
                stationary / "recordings" / "smoke.npz",
                xyz=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
                reflectivity=np.asarray([12, 34], dtype=np.float32),
                tag=np.asarray([1, 2], dtype=np.uint8),
                line=np.asarray([3, 4], dtype=np.uint8),
                point_offset_s=np.asarray([0.01, 0.02], dtype=np.float32),
                label=np.asarray([0, 1], dtype=np.uint8),
                frame_index=np.asarray([0, 1], dtype=np.int32),
                frame_ptr=np.asarray([0, 1, 2], dtype=np.int64),
                frame_time_s=np.asarray([20.0, 20.1], dtype=np.float64),
            )
            (stationary / "dataset_summary.json").write_text(json.dumps({
                "session_id": "fixed",
                "recordings": [{
                    "recording": "smoke",
                    "condition": "clean_control",
                    "data_file": "recordings/smoke.npz",
                }],
            }))

            grandtour_chunk = root / "grandtour.npz"
            np.savez_compressed(
                grandtour_chunk,
                xyz=np.asarray([[7, 8, 9], [10, 11, 12]], dtype=np.float32),
                intensity=np.asarray([56, 78], dtype=np.float32),
                tag=np.asarray([5, 6], dtype=np.uint8),
                line=np.asarray([7, 8], dtype=np.uint8),
                relative_time_ns=np.asarray([30_000_000, 40_000_000], dtype=np.int64),
                label=np.asarray([1, 255], dtype=np.uint8),
                frame_index=np.asarray([0, 1], dtype=np.int32),
                frame_ptr=np.asarray([0, 1, 2], dtype=np.int64),
                frame_time_ns=np.asarray([1_000_000_000, 1_100_000_000], dtype=np.int64),
            )
            grandtour_manifest = root / "grandtour.json"
            grandtour_manifest.write_text(json.dumps({
                "status": "complete",
                "session_id": "moving",
                "chunks": [{"chunk_index": 0, "path": str(grandtour_chunk)}],
            }))

            result = export_unified_dataset(
                [stationary], [grandtour_manifest], root / "unified"
            )
            loaded = [
                load_unified_chunk(root / "unified" / item["path"])
                for item in result["chunks"]
            ]

            self.assertEqual(len(loaded), 2)
            self.assertEqual(set(loaded[0]), set(ARRAY_NAMES))
            self.assertEqual(set(loaded[1]), set(ARRAY_NAMES))
            np.testing.assert_array_equal(loaded[0]["intensity"], [12, 34])
            np.testing.assert_array_equal(loaded[0]["label"], [0, 0])
            np.testing.assert_allclose(loaded[1]["point_offset_s"], [0.03, 0.04])
            np.testing.assert_allclose(loaded[0]["frame_time_s"], [0.0, 0.1])
            np.testing.assert_allclose(loaded[1]["frame_time_s"], [0.0, 0.1])
            self.assertNotIn("world_xyz", loaded[1])
            self.assertEqual(result["labels"]["smoke_impacted"], 1)
            self.assertEqual(
                result["chunks"][0]["label_policy"],
                "known_clean_valid_points_are_unimpacted",
            )


if __name__ == "__main__":
    unittest.main()
