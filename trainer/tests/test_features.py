from __future__ import annotations

import unittest

import numpy as np

from smoke_trainer.data import Frame, FrameRef
from smoke_trainer.features import FEATURE_NAMES, fit_normalizer, voxelize_frame


def frame(xyz, intensity, labels) -> Frame:
    return Frame(
        ref=FrameRef(0, 0),
        source_domain="test",
        session_id="test",
        recording_id="test",
        condition="smoke",
        timestamp_s=0.0,
        xyz=np.asarray(xyz, dtype=np.float32),
        intensity=np.asarray(intensity, dtype=np.float32),
        label=np.asarray(labels, dtype=np.uint8),
    )


class FeatureTests(unittest.TestCase):
    def test_voxelization_preserves_point_order_and_uses_ignored_points(self):
        source = frame(
            [[0.01, 0, 0], [1.01, 0, 0], [0.02, 0, 0]],
            [10, 30, 20],
            [0, 1, 255],
        )
        result = voxelize_frame(source, 0.1)
        self.assertEqual(result.features.shape, (2, len(FEATURE_NAMES)))
        np.testing.assert_array_equal(result.inverse, [0, 1, 0])
        self.assertAlmostEqual(float(result.features[0, 1]), 15.0)
        np.testing.assert_array_equal(result.labels, [0, 1, 255])

    def test_normalizer_uses_only_supplied_frames(self):
        training = frame([[0, 0, 0]], [10], [0])
        outlier = frame([[1000, 0, 0]], [9999], [1])
        normalizer = fit_normalizer([training], 0.1)
        self.assertAlmostEqual(float(normalizer.mean[1]), 10.0)
        transformed = normalizer.transform(voxelize_frame(outlier, 0.1).features)
        self.assertGreater(float(transformed[0, 1]), 1000.0)


if __name__ == "__main__":
    unittest.main()

