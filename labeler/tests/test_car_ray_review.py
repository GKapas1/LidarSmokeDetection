import unittest

import numpy as np

from smoke_labeler.car_ray_review import query_directional_ranges


class DirectionalReviewTests(unittest.TestCase):
    def setUp(self):
        y, z = np.meshgrid(np.arange(-0.12, 0.13, 0.03), np.arange(-0.12, 0.13, 0.03))
        self.wall = np.column_stack([np.full(y.size, 10.), y.ravel(), z.ravel()])

    def test_distinguishes_front_and_back_at_equal_map_distance(self):
        points = np.array([[10., 0, 0], [8., 0, 0], [12., 0, 0], [0, 10., 0]])
        expected, residual, _, _, category = query_directional_ranges(self.wall, points, np.zeros(3))
        np.testing.assert_array_equal(category, [1, 2, 3, 0])
        np.testing.assert_allclose(residual[:3], [0, -2, 2], atol=0.001)
        self.assertTrue(np.isnan(expected[-1]))

    def test_mixed_depths_and_sparse_support_stay_unknown(self):
        wall2 = self.wall.copy() * 1.2
        wall2[:, 1:] += 0.002
        for reference in (np.vstack([self.wall, wall2]), self.wall[:2], np.empty((0, 3))):
            result = query_directional_ranges(reference, np.array([[14., 0, 0]]), np.zeros(3))
            self.assertEqual(result[-1][0], 0)

    def test_translation_and_rotation_do_not_change_result(self):
        origin = np.array([3., -2., 0.5])
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        result = query_directional_ranges(self.wall @ rotation.T + origin,
            np.array([[12., 0, 0]]) @ rotation.T + origin, origin)
        self.assertEqual(result[-1][0], 3)


if __name__ == "__main__":
    unittest.main()
