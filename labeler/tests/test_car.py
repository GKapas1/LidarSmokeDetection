import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from smoke_labeler.car import (
    _sha256, build_car_reference, discover_recordings, timing_stats,
    validate_transform, voxel_cloud, write_cloud,
)
from smoke_labeler.geometry import transform_points
from smoke_labeler.car_registration import _rigid_fit, _trimmed_icp, _voxel_indices
from scipy.spatial import cKDTree


class CarPreparationTests(unittest.TestCase):
    def test_aliases_preserve_sources_and_ignore_old_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ["mid360_hall_clean_staionary1", "mid360_corridor_clean_stationary1ű",
                     "mid360_tree_smoke1", "mid360_drivetest_002"]
            for name in names:
                (root / name).mkdir()
                (root / name / "test.mcap").touch()
            entries, ignored = discover_recordings(root)
            self.assertEqual({e["id"] for e in entries}, {
                "hall_clean_stationary1", "corridor_clean_stationary1", "tree_smoke1"})
            self.assertEqual(ignored, ["mid360_drivetest_002"])
            self.assertTrue(all(Path(e["source_directory"]).exists() for e in entries))
            (root / "mid360_hall_clean_stationary1").mkdir()
            (root / "mid360_hall_clean_stationary1" / "test.mcap").touch()
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                discover_recordings(root)

    def test_support_counts_frames_not_points(self):
        xyz = np.array([[1, 1, 1]] * 5 + [[2, 2, 2]] * 3, dtype=np.float32)
        frame = np.array([0] * 5 + [0, 1, 2])
        points, intensity, support = voxel_cloud(xyz, np.arange(8), frame)
        np.testing.assert_array_equal(points, [[2, 2, 2]])
        np.testing.assert_array_equal(support, [3])

    def test_rigid_transform_direction_and_no_scale(self):
        matrix = np.array([[0, -1, 0, 2], [1, 0, 0, 3], [0, 0, 1, .12], [0, 0, 0, 1]])
        np.testing.assert_allclose(transform_points(validate_transform(matrix), np.array([[1, 0, 0]])),
                                   [[2, 4, .12]], atol=1e-6)
        for bad in [np.diag([2, 1, 1, 1]), np.diag([-1, 1, 1, 1]), np.full((4, 4), np.nan), None]:
            with self.assertRaises(ValueError):
                validate_transform(bad)

    def test_integer_timing_detects_duplicates_and_uses_intervals(self):
        epoch = 1_794_000_000_000_000_000
        stats = timing_stats([epoch, epoch + 100_000_000, epoch + 200_000_000])
        self.assertEqual(stats["rate_hz"], 10)
        self.assertEqual(timing_stats([epoch, epoch])["nonincreasing_timestamps"], 1)

    def test_reference_requires_review_and_tracks_exact_cloud(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xyz = np.array([[1, 0, 0], [2, 0, 0]], dtype=np.float32)
            np.savez(root / "cloud.npz", xyz=xyz, intensity=[20, 40], support_scans=[3, 4])
            manifest = {"layouts": {"hall": "alignment.json"}, "recordings": [
                {"id": "hall_clean_stationary1", "layout": "hall", "role": "reference"}]}
            review = {"layout_matches_smoke_confirmed": False, "references": [{
                "id": "hall_clean_stationary1", "cloud": "cloud.npz", "cloud_sha256": _sha256(root / "cloud.npz"),
                "alignment_reviewed": False, "T_layout_lidar": np.eye(4).tolist(),
                "exclude_boxes_sensor_frame": [{"min_m": [.5, -1, -1], "max_m": [1.5, 1, 1]}]}]}
            (root / "manifest.json").write_text(json.dumps(manifest))
            def save():
                (root / "alignment.json").write_text(json.dumps(review))
            save()
            with self.assertRaisesRegex(ValueError, "layout consistency"):
                build_car_reference(root, "hall", root / "output")
            review["layout_matches_smoke_confirmed"] = True
            save()
            with self.assertRaisesRegex(ValueError, "not been reviewed"):
                build_car_reference(root, "hall", root / "output")
            review["references"][0]["alignment_reviewed"] = True
            save()
            result = build_car_reference(root, "hall", root / "output")
            self.assertEqual(result["points"], 1)
            with np.load(root / "output/reference.npz") as cloud:
                np.testing.assert_array_equal(cloud["xyz"], [[2, 0, 0]])
            with self.assertRaisesRegex(ValueError, "not empty"):
                build_car_reference(root, "hall", root / "output")
            with (root / "cloud.npz").open("ab") as f:
                f.write(b"changed")
            with self.assertRaisesRegex(ValueError, "changed since"):
                build_car_reference(root, "hall", root / "other")

    def test_ply_preserves_float_intensity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.ply"
            write_cloud(path, np.array([[1, 2, 3]]), [42.5])
            header, data = path.read_bytes().split(b"end_header\n", 1)
            self.assertIn(b"element vertex 1", header)
            np.testing.assert_array_equal(np.frombuffer(data, dtype="<f4"), [1, 2, 3, 42.5])

    def test_registration_helpers_recover_rigid_pose(self):
        rng = np.random.default_rng(7)
        source = rng.normal(size=(1000, 3))
        expected = np.array([
            [0, -1, 0, 0.2], [1, 0, 0, -0.1], [0, 0, 1, 0.05], [0, 0, 0, 1]
        ], dtype=np.float64)
        target = transform_points(expected, source).astype(np.float64)
        recovered = _rigid_fit(source, target)
        np.testing.assert_allclose(recovered, expected, atol=1e-6)
        keep = _voxel_indices(source, 0.1)
        self.assertGreater(len(keep), 100)
        initial = expected.copy()
        initial[:3, 3] += [0.03, -0.02, 0.01]
        refined, converged, iterations = _trimmed_icp(
            source, initial, target, cKDTree(target)
        )
        self.assertTrue(converged)
        self.assertGreater(iterations, 0)
        np.testing.assert_allclose(refined, expected, atol=2e-4)
