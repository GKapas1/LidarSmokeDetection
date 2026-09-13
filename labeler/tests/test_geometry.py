from __future__ import annotations

import unittest

import numpy as np

from smoke_labeler.geometry import PoseTrajectory, pose_matrix, transform_points


class GeometryTests(unittest.TestCase):
    def test_pose_interpolation_uses_translation_and_slerp(self):
        trajectory = PoseTrajectory(
            time_ns=np.array([1_000_000_000, 1_100_000_000], dtype=np.int64),
            translation=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            quaternion_xyzw=np.array(
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
            ),
            world_frame="world",
            body_frame="body",
        )
        transform = trajectory.at(1_050_000_000)
        point = transform_points(transform, np.array([[1.0, 0.0, 0.0]]))
        np.testing.assert_allclose(point, [[1.0, 1.0, 0.0]], atol=1e-6)

    def test_pose_interpolation_rejects_large_gap(self):
        trajectory = PoseTrajectory(
            np.array([0, 1_000_000_000]),
            np.zeros((2, 3)),
            np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
            "world",
            "body",
        )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            trajectory.at(500_000_000, maximum_gap_ns=100_000_000)

    def test_pose_matrix_normalizes_quaternion(self):
        transform = pose_matrix(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0, 2.0]))
        np.testing.assert_allclose(transform, np.array([
            [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]
        ]))


if __name__ == "__main__":
    unittest.main()
