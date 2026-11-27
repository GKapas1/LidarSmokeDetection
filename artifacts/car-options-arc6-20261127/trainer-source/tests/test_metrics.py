from __future__ import annotations

import unittest

import numpy as np

from smoke_trainer.evaluate import fit_calibration
from smoke_trainer.metrics import average_precision, best_f1_threshold, metric_summary, roc_auc


class MetricTests(unittest.TestCase):
    def test_perfect_ranking(self):
        labels = np.asarray([0, 1, 0, 1], dtype=np.uint8)
        scores = np.asarray([0.1, 0.9, 0.2, 0.8])
        self.assertAlmostEqual(average_precision(labels, scores), 1.0)
        self.assertAlmostEqual(roc_auc(labels, scores), 1.0)
        threshold = best_f1_threshold(labels, scores)
        report = metric_summary(labels, scores, threshold=threshold)
        self.assertAlmostEqual(report["f1"], 1.0)

    def test_tied_scores_do_not_depend_on_label_order(self):
        scores = np.asarray([0.5, 0.5, 0.5, 0.5])
        first = average_precision(np.asarray([1, 1, 0, 0], dtype=np.uint8), scores)
        second = average_precision(np.asarray([0, 1, 0, 1], dtype=np.uint8), scores)
        self.assertAlmostEqual(first, 0.5)
        self.assertAlmostEqual(second, 0.5)

    def test_calibrator_has_positive_slope(self):
        logits = np.asarray([-3, -2, -1, 1, 2, 3], dtype=np.float32)
        labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.uint8)
        calibration = fit_calibration(logits, labels)
        self.assertGreater(calibration.slope, 0.0)
        probabilities = calibration.apply(logits)
        self.assertTrue(np.all(np.diff(probabilities) >= 0))
        self.assertLess(float(probabilities[0]), float(probabilities[-1]))


if __name__ == "__main__":
    unittest.main()
