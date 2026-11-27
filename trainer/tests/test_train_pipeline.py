from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from smoke_trainer.data import UnifiedDataset, create_split_plan
from smoke_trainer.predict import GlobalTemporalPredictor, LocalPredictor, SmokePredictor
from smoke_trainer.train import train_model

from helpers import make_dataset


class TrainPipelineTests(unittest.TestCase):
    def test_tiny_voxel_pointnet_trains_and_predicts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = make_dataset(root / "dataset")
            dataset = UnifiedDataset(manifest)
            split = root / "split.json"
            create_split_plan(dataset, split, purge_frames=1)
            config = {
                "config_path": "test",
                "data": {"manifest": str(manifest), "split": str(split), "voxel_size_m": 0.1},
                "model": {"name": "voxel_pointnet", "adapter": "voxel_points", "point_widths": [8, 12], "head_widths": [6]},
                "training": {"seed": 8, "device": "cpu", "epochs": 1, "learning_rate": 0.01,
                             "weight_decay": 0.0, "positive_weight": 1.0, "patience": 1, "max_hours": 0.1},
                "output": {"root": str(root / "runs")},
            }
            run = train_model(config, run_name="tiny-voxel")
            predictor = SmokePredictor.load(run / "bundle.pt", device="cpu")
            source = dataset.frame(dataset.chunks[0], dataset.load_chunk(0), 0)
            probabilities, valid = predictor.predict(source)
            self.assertEqual(predictor.adapter.name, "voxel_points")
            self.assertEqual(probabilities.shape, source.label.shape)
            self.assertTrue(valid.all())
            self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))

    def test_tiny_dataset_trains_packages_and_predicts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = make_dataset(root / "dataset")
            dataset = UnifiedDataset(manifest)
            split = root / "split.json"
            create_split_plan(dataset, split, purge_frames=1)
            config = {
                "config_path": "test",
                "data": {
                    "manifest": str(manifest),
                    "split": str(split),
                    "voxel_size_m": 0.1,
                    "verify_checksums": True,
                },
                "model": {"name": "local_mlp", "hidden_widths": [8, 4]},
                "training": {
                    "seed": 7,
                    "device": "cpu",
                    "epochs": 2,
                    "learning_rate": 0.01,
                    "weight_decay": 0.0,
                    "positive_weight": 1.0,
                    "patience": 2,
                    "max_hours": 0.1,
                },
                "output": {"root": str(root / "runs")},
            }
            run = train_model(config, run_name="tiny")

            self.assertTrue((run / "best.pt").is_file())
            self.assertTrue((run / "bundle.pt").is_file())
            self.assertTrue((run / "selection_metrics.json").is_file())
            self.assertTrue((run / "calibration_fit.png").is_file())
            bundle = __import__("torch").load(run / "bundle.pt", weights_only=False)
            self.assertEqual(bundle["format_version"], 2)
            self.assertEqual(bundle["adapter_name"], "local_voxel")
            self.assertEqual(bundle["input_schema"], bundle["feature_names"])

            predictor = SmokePredictor.load(run / "bundle.pt", device="cpu")
            arrays = dataset.load_chunk(0)
            source = dataset.frame(dataset.chunks[0], arrays, 0)
            probabilities, valid = predictor.predict(source)
            self.assertEqual(probabilities.shape, source.label.shape)
            self.assertTrue(valid.all())
            self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
            timing = predictor.benchmark(source, repeats=2)
            self.assertEqual(timing["points"], len(source.xyz))

            resumed = train_model(config, resume_run=run)
            self.assertEqual(resumed.resolve(), run.resolve())
            history = __import__("json").loads((run / "history.json").read_text())
            self.assertEqual([row["epoch"] for row in history], [1, 2])

            compatibility = LocalPredictor.load(run / "bundle.pt", device="cpu")
            compatibility_probabilities, _ = compatibility.predict(source)
            np.testing.assert_allclose(probabilities, compatibility_probabilities)

    def test_tiny_global_pointnet_trains_packages_and_predicts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = make_dataset(root / "dataset")
            dataset = UnifiedDataset(manifest)
            split = root / "split.json"
            create_split_plan(dataset, split, purge_frames=1)
            config = {
                "config_path": "test",
                "data": {
                    "manifest": str(manifest),
                    "split": str(split),
                    "verify_checksums": True,
                },
                "model": {
                    "name": "global_pointnet",
                    "adapter": "global_points",
                    "point_widths": [8, 12],
                    "head_widths": [6],
                    "input_alignment": "tnet",
                    "alignment_widths": [4, 8],
                    "alignment_regularization": 0.001,
                },
                "training": {
                    "seed": 8,
                    "device": "cpu",
                    "epochs": 1,
                    "learning_rate": 0.01,
                    "weight_decay": 0.0,
                    "positive_weight": 1.0,
                    "patience": 1,
                    "max_hours": 0.1,
                    "yaw_augmentation_deg": 180.0,
                    "pitch_augmentation_deg": 0.0,
                    "roll_augmentation_deg": 0.0,
                },
                "output": {"root": str(root / "runs")},
            }
            run = train_model(config, run_name="tiny-global")
            predictor = SmokePredictor.load(run / "bundle.pt", device="cpu")
            arrays = dataset.load_chunk(0)
            source = dataset.frame(dataset.chunks[0], arrays, 0)
            probabilities, valid = predictor.predict(source)

            self.assertEqual(predictor.adapter.name, "global_points")
            self.assertEqual(probabilities.shape, source.label.shape)
            self.assertTrue(valid.all())
            self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))

    def test_tiny_global_temporal_model_trains_packages_and_predicts_causally(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = make_dataset(root / "dataset")
            dataset = UnifiedDataset(manifest)
            split = root / "split.json"
            create_split_plan(dataset, split, purge_frames=1)
            config = {
                "config_path": "test",
                "data": {
                    "manifest": str(manifest),
                    "split": str(split),
                    "verify_checksums": True,
                    "history_length": 3,
                    "maximum_time_gap_s": 0.25,
                },
                "model": {
                    "name": "global_temporal_pointnet",
                    "adapter": "global_temporal",
                    "history_length": 3,
                    "point_widths": [8, 12],
                    "head_widths": [6],
                    "gru_hidden_width": 5,
                    "input_alignment": "tnet",
                    "alignment_widths": [4, 8],
                    "alignment_regularization": 0.001,
                },
                "training": {
                    "seed": 9,
                    "device": "cpu",
                    "epochs": 1,
                    "learning_rate": 0.01,
                    "weight_decay": 0.0,
                    "positive_weight": 1.0,
                    "patience": 1,
                    "max_hours": 0.1,
                    "max_optimizer_steps_per_epoch": 2,
                    "yaw_augmentation_deg": 180.0,
                    "pitch_augmentation_deg": 0.0,
                    "roll_augmentation_deg": 0.0,
                },
                "augmentation": {"background_mask": {"probability": 1.0, "range_min_m": 1.4,
                    "range_max_m": 1.4, "width_min_deg": 359.0, "width_max_deg": 359.0}},
                "output": {"root": str(root / "runs")},
            }
            run = train_model(config, run_name="tiny-gt")
            history = __import__("json").loads((run / "history.json").read_text())
            self.assertEqual(history[0]["training_windows"], 2)
            self.assertGreater(history[0]["augmentation"]["masked_context_points"], 0)
            predictor = SmokePredictor.load(run / "bundle.pt", device="cpu")
            self.assertIsInstance(predictor, GlobalTemporalPredictor)
            arrays = dataset.load_chunk(0)
            first = dataset.frame(dataset.chunks[0], arrays, 0)
            second = dataset.frame(dataset.chunks[0], arrays, 1)
            first_probabilities, first_valid = predictor.predict(first)
            second_probabilities, second_valid = predictor.predict(second)

            self.assertEqual(predictor.adapter.name, "global_temporal")
            self.assertEqual(predictor.last_history_length, 2)
            self.assertTrue(first_valid.all())
            self.assertTrue(second_valid.all())
            self.assertEqual(first_probabilities.shape, first.label.shape)
            self.assertEqual(second_probabilities.shape, second.label.shape)
            predictor.predict(first)
            self.assertEqual(predictor.last_history_length, 1)


if __name__ == "__main__":
    unittest.main()
