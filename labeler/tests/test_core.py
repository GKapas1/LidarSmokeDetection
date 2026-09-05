import unittest

import numpy as np

from smoke_labeler.core import (
    IGNORE,
    SMOKE_IMPACTED,
    UNAFFECTED,
    build_directional_reference,
    calibrate_early_return_thresholds,
    label_points,
    query_reference,
)


def rays_at_ranges(ranges):
    angles = np.linspace(-0.01, 0.01, len(ranges), dtype=np.float32)
    ranges = np.asarray(ranges, dtype=np.float32)
    return np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles), np.zeros(len(ranges))))


class CoreTests(unittest.TestCase):
    def test_early_return_is_labeled_smoke(self):
        clean_xyz = rays_at_ranges(np.full(200, 5.0))
        reference = build_directional_reference(clean_xyz, np.full(200, 20.0), 1.0)

        held_out = query_reference(reference, rays_at_ranges(np.full(100, 5.0)), 3, 0.12)
        calibration = calibrate_early_return_thresholds(
            held_out["expected_range"], held_out["range_residual"], held_out["valid_reference"],
            np.array([0.0, 10.0]), 0.99, 0.05, 10,
        )
        smoke_xyz = rays_at_ranges(np.array([5.0, 2.0, 5.02]))
        query = query_reference(reference, smoke_xyz, 3, 0.12)
        labels, confidence, _ = label_points(query, calibration)

        self.assertEqual(labels.tolist(), [UNAFFECTED, SMOKE_IMPACTED, UNAFFECTED])
        self.assertGreater(confidence[1], 0.9)

    def test_depth_discontinuity_cell_is_ignored(self):
        clean_xyz = rays_at_ranges(np.tile([2.0, 5.0], 100))
        reference = build_directional_reference(clean_xyz, np.ones(200), 5.0)
        query = query_reference(reference, rays_at_ranges([3.0]), 3, 0.12)
        calibration = calibrate_early_return_thresholds(
            np.array([3.0]), np.array([0.0]), np.array([True]), np.array([0.0, 10.0]), 0.99, 0.05, 1,
        )
        labels, _, _ = label_points(query, calibration)
        self.assertEqual(labels[0], IGNORE)

    def test_directional_labels_have_uncertain_interval(self):
        clean_xyz = rays_at_ranges(np.full(300, 5.0))
        reference = build_directional_reference(clean_xyz, np.full(300, 20.0), 1.0)
        held_out = query_reference(reference, rays_at_ranges(np.full(200, 5.0)), 3, 0.12)
        calibration = calibrate_early_return_thresholds(
            held_out["expected_range"], held_out["range_residual"], held_out["valid_reference"],
            np.array([0.0, 10.0]), 0.999, 0.05, 10,
            normal_quantile=0.99, minimum_normal_residual_m=0.02,
            maximum_usable_early_return_m=0.25,
        )
        query = query_reference(reference, rays_at_ranges([5.0, 4.97, 4.8]), 3, 0.12)
        labels, _, _ = label_points(query, calibration)
        self.assertEqual(labels.tolist(), [UNAFFECTED, IGNORE, SMOKE_IMPACTED])


if __name__ == "__main__":
    unittest.main()
