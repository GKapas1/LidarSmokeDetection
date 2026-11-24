import unittest

import numpy as np

from smoke_labeler.car_label import _repeat_candidates


class CarLabelTemporalTests(unittest.TestCase):
    def test_requires_local_returns_in_both_adjacent_frames(self):
        xyz = np.array([[0.00, 0, 0], [0.05, 0, 0], [0.10, 0, 0]])
        previous = {"time": 900_000_000, "candidates": xyz}
        current = {"time": 1_000_000_000, "candidates": np.array([[0, 0, 0], [1, 0, 0]])}
        following = {"time": 1_100_000_000, "candidates": xyz}

        np.testing.assert_array_equal(_repeat_candidates(current, previous, following), [True, False])
        np.testing.assert_array_equal(_repeat_candidates(current, previous, None), [False, False])
        following["time"] = 1_200_000_000
        np.testing.assert_array_equal(_repeat_candidates(current, previous, following), [False, False])


if __name__ == "__main__":
    unittest.main()
