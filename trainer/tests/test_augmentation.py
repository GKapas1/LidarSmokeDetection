from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from smoke_trainer.augmentation import augment_training_frame, validate_augmentation
from smoke_trainer.data import Frame, FrameRef, UnifiedDataset, create_split_plan
from smoke_trainer.train import train_model

from helpers import make_dataset


SMOKE = dict(probability=1.0, target_min_m=5.0, target_max_m=7.0,
             patch_radius_m=0.35, clearance_m=0.25, ray_tolerance_rad=0.02,
             minimum_points=8, attempts=16)


def scene(obstructed=False):
    rng = np.random.default_rng(33)
    smoke = rng.normal([1.8, 0, 0.2], 0.07, (2000, 3))
    y, z = np.meshgrid(np.linspace(-2, 2, 64), np.linspace(-2, 2, 64))
    wall = np.column_stack((np.full(y.size, 12), y.ravel(), z.ravel()))
    xyz = np.concatenate((smoke, wall)).astype(np.float32)
    label = np.r_[np.ones(len(smoke), np.uint8), np.zeros(len(wall), np.uint8)]
    if obstructed:
        xyz = np.concatenate((xyz, (wall / 4).astype(np.float32)))
        label = np.r_[label, np.full(len(wall), 255, np.uint8)]
    return Frame(FrameRef(0, 0), "car_ros2", "test", "smoke", "smoke", 0.0,
                 xyz, np.full(len(xyz), 10, np.float32), label)


class AugmentationTests(unittest.TestCase):
    def test_copy_preserves_originals_and_places_thinned_smoke_in_free_space(self):
        source = scene()
        config = {"smoke_relocation": SMOKE}
        validate_augmentation(config, "global_pointnet")
        result, counts = augment_training_frame(source, config, np.random.default_rng(5))
        copied = counts["copied_smoke_points"]
        self.assertGreaterEqual(copied, 8)
        self.assertLess(copied, 2000)
        np.testing.assert_array_equal(result.xyz[:len(source.xyz)], source.xyz)
        np.testing.assert_array_equal(result.label[:len(source.label)], source.label)
        self.assertTrue(np.all(result.label[-copied:] == 1))
        ranges = np.linalg.norm(result.xyz[-copied:], axis=1)
        self.assertTrue(np.all((ranges >= 5) & (ranges <= 7)))
        repeated, _ = augment_training_frame(source, config, np.random.default_rng(5))
        np.testing.assert_array_equal(result.xyz, repeated.xyz)

    def test_ignored_foreground_returns_block_synthetic_placement(self):
        source = scene(obstructed=True)
        result, counts = augment_training_frame(source, {"smoke_relocation": SMOKE}, np.random.default_rng(5))
        self.assertEqual(counts["copied_smoke_points"], 0)
        self.assertIs(result, source)

    def test_context_mask_uses_geometry_and_keeps_near_points(self):
        source = scene()
        options = {"background_mask": dict(probability=1.0, range_min_m=8.0,
                   range_max_m=8.0, width_min_deg=180.0, width_max_deg=180.0)}
        result, counts = augment_training_frame(source, options, np.random.default_rng(5))
        self.assertGreater(counts["masked_context_points"], 0)
        np.testing.assert_array_equal(result.xyz[:2000], source.xyz[:2000])
        changed_labels = replace(source, label=1 - source.label)
        second, _ = augment_training_frame(changed_labels, options, np.random.default_rng(5))
        np.testing.assert_array_equal(result.xyz, second.xyz)
        self.assertEqual(len(source.xyz), 6096)

    def test_disabled_augmentation_preserves_frame_and_rng(self):
        source = scene()
        rng = np.random.default_rng(7)
        state = repr(rng.bit_generator.state)
        result, counts = augment_training_frame(source, {}, rng)
        self.assertIs(result, source)
        self.assertEqual(repr(rng.bit_generator.state), state)
        self.assertEqual(sum(counts.values()), 0)

    def test_training_entry_point_applies_mask_and_optimizer_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = make_dataset(root / "dataset")
            split = root / "split.json"
            create_split_plan(UnifiedDataset(manifest), split, purge_frames=1)
            config = {
                "config_path": "test", "data": {"manifest": str(manifest), "split": str(split)},
                "model": {"name": "global_pointnet", "adapter": "global_points", "point_widths": [4], "head_widths": [4]},
                "training": dict(seed=8, device="cpu", epochs=1, learning_rate=.01,
                                 weight_decay=0, positive_weight=1, patience=1, max_hours=.1,
                                 max_optimizer_steps_per_epoch=2),
                "augmentation": {"background_mask": dict(probability=1.0, range_min_m=1.4,
                                   range_max_m=1.4, width_min_deg=359.0, width_max_deg=359.0)},
                "output": {"root": str(root / "runs")},
            }
            run = train_model(config, run_name="augmented")
            history = json.loads((run / "history.json").read_text())
            self.assertEqual(history[0]["training_frames"], 2)
            self.assertGreater(history[0]["augmentation"]["masked_context_points"], 0)
            self.assertTrue((run / "bundle.pt").is_file())
            config["augmentation"] = {}
            with self.assertRaisesRegex(ValueError, "augmentation differs"):
                train_model(config, resume_run=run)
