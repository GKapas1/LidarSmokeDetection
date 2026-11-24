import unittest

from smoke_labeler.car_clean_audit import _runs, classify_frame


def row(**overrides):
    value = dict(
        correction_translation_m=0.1, correction_rotation_deg=0.5,
        paired_mismatch_fraction=0.01, behind_reference_fraction=0.01,
        outside_reference_footprint_fraction=0.01,
        median_m=0.1, within_0_20=0.8, p90_m=0.3,
    )
    value.update(overrides)
    return value


class CleanAuditTests(unittest.TestCase):
    def test_consecutive_ranges(self):
        self.assertEqual(_runs([1, 2, 4, 7, 8, 9]), [[1, 2], [4, 4], [7, 9]])

    def test_statuses_preserve_reason(self):
        self.assertEqual(classify_frame(row())[0], "keep_candidate")
        status, reasons = classify_frame(row(
            paired_mismatch_fraction=.04, behind_reference_fraction=.03))
        self.assertEqual(status, "review")
        self.assertIn("possible_paired_displacement", reasons)
        status, reasons = classify_frame(row(
            paired_mismatch_fraction=.06, behind_reference_fraction=.06))
        self.assertEqual(status, "reject_candidate")
        self.assertIn("strong_paired_displacement", reasons)
        status, reasons = classify_frame(row(correction_translation_m=.6))
        self.assertEqual(status, "reject_candidate")
        self.assertIn("implausible_pose_correction", reasons)
        status, reasons = classify_frame(row(
            outside_reference_footprint_fraction=.10))
        self.assertEqual(status, "review")
        self.assertIn("outside_reference_footprint", reasons)
        status, _ = classify_frame(row(
            outside_reference_footprint_fraction=.30, median_m=.30))
        self.assertEqual(status, "reject_candidate")


if __name__ == "__main__":
    unittest.main()
