import unittest
import numpy as np

from smoke_labeler.object_surfaces import (
    Panel, fit_trunk, floor_coordinates, panel_evidence, revise_labels, trunk_evidence,
)


class ObjectSurfaceTests(unittest.TestCase):
    def test_floor_coordinates_preserve_beam_lengths(self):
        points = np.array([[2., -3., .4], [-1., 4., 2.], [.1, .3, -.2]])
        floor = {"floor_normal": [-.02, .03, 1.], "floor_offset_m": .3}
        transformed = floor_coordinates(points, floor)
        np.testing.assert_allclose(np.linalg.norm(transformed[1:] - transformed[0], axis=1),
                                   np.linalg.norm(points[1:] - points[0], axis=1))

    def panel(self, blocks):
        uv = np.array([[y, z] for y in [-.03, 0, .03] for z in [-.03, 0, .03]])
        anchors = np.column_stack((np.full(len(uv), 2.), uv))
        return Panel(np.array([2., 0, 0]), np.array([1., 0, 0]),
                     np.array([0., 1, 0]), np.array([0., 0, 1.]),
                     np.array([-.2, -.2]), np.array([.2, .2]),
                     np.tile(anchors, (len(blocks), 1)), np.repeat(blocks, len(uv)))

    def test_independent_temporal_support_and_finite_face(self):
        panel = self.panel([0, 1, 2, 3])
        result = panel.support(np.array([[2., 0, 0], [2., .5, 0]]), block=0)
        np.testing.assert_array_equal(result, [True, False])
        panel = self.panel([0, 0, 1, 1])
        self.assertFalse(panel.support(np.array([[2., 0, 0]]), block=0)[0])

    def test_surface_early_return_and_behind_surface(self):
        points = np.array([[2., 0, 0], [1.6, 0, 0], [2.4, 0, 0], [1.6, 1., 0]])
        surface, ambiguous, early = panel_evidence(points, np.zeros_like(points),
                                                    [self.panel([0, 1, 2, 3])], block=0)
        np.testing.assert_array_equal(surface, [True, False, False, False])
        np.testing.assert_array_equal(early, [False, True, False, False])

    def test_existing_false_smoke_can_be_corrected(self):
        labels = np.array([1, 1, 255, 0, 255], dtype=np.uint8)
        surface = np.array([1, 0, 0, 0, 1], bool)
        ambiguous = np.array([0, 1, 0, 0, 0], bool)
        smoke = np.array([0, 0, 1, 1, 1], bool)
        result = revise_labels(labels, surface, ambiguous, smoke)
        np.testing.assert_array_equal(result, [0, 1, 1, 0, 0])
        np.testing.assert_array_equal(labels, [1, 1, 255, 0, 255])

    def test_trunk_fit_and_surface_do_not_fill_cylinder_interior(self):
        angles = np.tile(np.linspace(0, 2 * np.pi, 40, endpoint=False), 8)
        blocks = np.repeat(np.arange(8), 40)
        points = np.column_stack((-.9 + .05 * np.cos(angles),
                                  -.5 + .05 * np.sin(angles),
                                  np.full(len(angles), .3)))
        model = fit_trunk(points, blocks, [[-1., -.6], [-.8, -.4]])
        self.assertTrue(model["accepted"])
        surface, _ = trunk_evidence(np.array([[-.85, -.5, .3], [-.9, -.5, .3],
                                              [-.85, -.5, 1.]]), model, [.1, .55])
        np.testing.assert_array_equal(surface, [True, False, False])

    def test_trunk_with_one_temporal_subset_is_rejected(self):
        points = np.tile([[-.9, -.5, .3]], (30, 1))
        model = fit_trunk(points, np.zeros(30, int), [[-1., -.6], [-.8, -.4]])
        self.assertFalse(model["accepted"])


if __name__ == "__main__":
    unittest.main()
