from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from smoke_labeler.grandtour import pointcloud2_array
from smoke_labeler.grandtour_pipeline import (
    EXCLUDED,
    REASON_EXCLUDED_REGION,
    REASON_OUTSIDE_COVERAGE,
    REASON_OUTSIDE_CONFIRMED_SMOKE_REGION,
    SMOKE_CANDIDATE,
    STRUCTURE,
    UNKNOWN,
    _time_chunks,
    _isolated_return_mask,
    classify_reference_distances,
    export_grandtour_training_dataset,
)


class PointCloudDecoderTests(unittest.TestCase):
    def test_decodes_fields_and_skips_row_padding(self):
        dtype = np.dtype(
            {
                "names": ["x", "tag"],
                "formats": ["<f4", "u1"],
                "offsets": [0, 4],
                "itemsize": 8,
            }
        )
        first = np.zeros(2, dtype=dtype)
        first["x"] = [1.5, 2.5]
        first["tag"] = [3, 4]
        second = np.zeros(2, dtype=dtype)
        second["x"] = [5.5, 6.5]
        second["tag"] = [7, 8]
        message = SimpleNamespace(
            height=2,
            width=2,
            point_step=8,
            row_step=20,
            is_bigendian=False,
            fields=[
                SimpleNamespace(name="x", offset=0, datatype=7, count=1),
                SimpleNamespace(name="tag", offset=4, datatype=2, count=1),
            ],
            data=first.tobytes() + b"PAD!" + second.tobytes() + b"PAD!",
        )

        decoded = pointcloud2_array(message)

        np.testing.assert_array_equal(decoded["x"], [1.5, 2.5, 5.5, 6.5])
        np.testing.assert_array_equal(decoded["tag"], [3, 4, 7, 8])

    def test_rejects_truncated_data(self):
        message = SimpleNamespace(
            height=1,
            width=2,
            point_step=4,
            row_step=8,
            is_bigendian=False,
            fields=[SimpleNamespace(name="x", offset=0, datatype=7, count=1)],
            data=b"1234",
        )
        with self.assertRaisesRegex(ValueError, "truncated"):
            pointcloud2_array(message)


class ReferenceClassificationTests(unittest.TestCase):
    def test_only_spatially_isolated_return_is_rejected(self):
        xyz = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [5.0, 5.0, 5.0]])
        mask = _isolated_return_mask(xyz, np.ones(3, dtype=bool), 0.5)
        np.testing.assert_array_equal(mask, [False, False, True])

    def test_full_session_chunks_include_short_final_interval(self):
        self.assertEqual(
            _time_chunks(0.0, 349.0, 60.0),
            [(0.0, 60.0), (60.0, 120.0), (120.0, 180.0),
             (180.0, 240.0), (240.0, 300.0), (300.0, 349.0)],
        )

    def test_margin_and_human_confirmed_exclusion_override_smoke(self):
        world = np.asarray(
            [
                [0.1, 5.0, 1.0],
                [5.0, 5.0, 1.0],
                [2.0, 2.0, 1.0],
            ],
            dtype=np.float64,
        )
        final, automatic, reason, region_id, verified, hard_negative_id = classify_reference_distances(
            world_xyz=world,
            nearest_distance_m=np.asarray([0.4, 0.4, 0.4], dtype=np.float32),
            input_valid=np.ones(3, dtype=bool),
            crop_min=np.asarray([0.0, 0.0, 0.0]),
            crop_max=np.asarray([10.0, 10.0, 2.0]),
            coverage_margin_m=0.5,
            structure_threshold_m=0.08,
            smoke_threshold_m=0.25,
            excluded_regions=[
                {"name": "person", "min_m": [1.5, 1.5, 0.5], "max_m": [2.5, 2.5, 1.5]}
            ],
            confirmed_smoke_regions=[
                {"name": "band", "min_m": [4.5, 4.5, 0.5], "max_m": [5.5, 5.5, 1.5]}
            ],
        )

        np.testing.assert_array_equal(final, [UNKNOWN, SMOKE_CANDIDATE, EXCLUDED])
        np.testing.assert_array_equal(automatic, [UNKNOWN, SMOKE_CANDIDATE, SMOKE_CANDIDATE])
        np.testing.assert_array_equal(
            reason, [REASON_OUTSIDE_COVERAGE, 0, REASON_EXCLUDED_REGION]
        )
        np.testing.assert_array_equal(region_id, [-1, -1, 0])
        np.testing.assert_array_equal(verified, [False, True, True])
        np.testing.assert_array_equal(hard_negative_id, [-1, -1, -1])

    def test_smoke_outside_confirmed_region_is_withheld(self):
        final, automatic, reason, _, _, _ = classify_reference_distances(
            world_xyz=np.asarray([[7.0, 7.0, 1.0]]),
            nearest_distance_m=np.asarray([0.4], dtype=np.float32),
            input_valid=np.ones(1, dtype=bool),
            crop_min=np.asarray([0.0, 0.0, 0.0]),
            crop_max=np.asarray([10.0, 10.0, 2.0]),
            coverage_margin_m=0.5,
            structure_threshold_m=0.08,
            smoke_threshold_m=0.25,
            excluded_regions=[],
            confirmed_smoke_regions=[
                {"name": "band", "min_m": [4.5, 4.5, 0.5], "max_m": [5.5, 5.5, 1.5]}
            ],
        )
        self.assertEqual(automatic[0], SMOKE_CANDIDATE)
        self.assertEqual(final[0], UNKNOWN)
        self.assertEqual(reason[0], REASON_OUTSIDE_CONFIRMED_SMOKE_REGION)

    def test_reviewed_background_and_person_are_unimpacted(self):
        final, automatic, reason, _, verified, hard_negative_id = classify_reference_distances(
            world_xyz=np.asarray([[7.0, 7.0, 1.0], [2.0, 2.0, 1.0]]),
            nearest_distance_m=np.asarray([0.4, 0.4], dtype=np.float32),
            input_valid=np.ones(2, dtype=bool),
            crop_min=np.asarray([0.0, 0.0, 0.0]),
            crop_max=np.asarray([10.0, 10.0, 2.0]),
            coverage_margin_m=0.5,
            structure_threshold_m=0.08,
            smoke_threshold_m=0.25,
            excluded_regions=[],
            confirmed_smoke_regions=[
                {"name": "band", "min_m": [1.5, 1.5, 0.5], "max_m": [2.5, 2.5, 1.5]}
            ],
            reviewed_background_unimpacted=True,
            hard_negative_regions=[
                {"name": "person", "min_m": [1.5, 1.5, 0.5], "max_m": [2.5, 2.5, 1.5]}
            ],
        )
        np.testing.assert_array_equal(automatic, [SMOKE_CANDIDATE, SMOKE_CANDIDATE])
        np.testing.assert_array_equal(final, [0, 0])
        self.assertTrue(verified.all())
        np.testing.assert_array_equal(hard_negative_id, [-1, 0])


class TrainingExportTests(unittest.TestCase):
    def test_export_contains_sensor_inputs_but_no_teacher_features(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_npz = root / "labels.npz"
            np.savez_compressed(
                source_npz,
                session_id=np.array("arc6"),
                sensor_xyz=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
                world_xyz=np.asarray([[7, 8, 9], [10, 11, 12]], dtype=np.float32),
                intensity=np.asarray([2, 3], dtype=np.float32),
                tag=np.asarray([0, 1], dtype=np.uint8),
                line=np.asarray([1, 2], dtype=np.uint8),
                point_time_ns=np.asarray([1010, 2020], dtype=np.uint64),
                frame_index=np.asarray([0, 1], dtype=np.int32),
                source_point_index=np.asarray([0, 0], dtype=np.uint32),
                frame_ptr=np.asarray([0, 1, 2], dtype=np.int64),
                frame_time_ns=np.asarray([1000, 2000], dtype=np.int64),
                label=np.asarray([STRUCTURE, EXCLUDED], dtype=np.uint8),
                nearest_reference_distance_m=np.asarray([0.01, 0.5], dtype=np.float32),
            )
            source_manifest = root / "source.json"
            source_manifest.write_text(json.dumps({
                "status": "complete",
                "session_id": "arc6",
                "chunks": [{
                    "labels_npz": str(source_npz),
                    "time_selection_s": {"start": 0.0, "end": 1.0},
                }],
            }))

            result = export_grandtour_training_dataset(source_manifest, root / "training")

            with np.load(result["chunks"][0]["path"]) as exported:
                self.assertIn("xyz", exported.files)
                self.assertIn("relative_time_ns", exported.files)
                self.assertNotIn("world_xyz", exported.files)
                self.assertNotIn("nearest_reference_distance_m", exported.files)
                np.testing.assert_array_equal(exported["relative_time_ns"], [10, 20])
                np.testing.assert_array_equal(exported["label"], [STRUCTURE, UNKNOWN])


if __name__ == "__main__":
    unittest.main()
