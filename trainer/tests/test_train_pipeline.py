from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from smoke_trainer.data import UnifiedDataset, create_split_plan
from smoke_trainer.predict import LocalPredictor
from smoke_trainer.train import train_local_model

from helpers import make_dataset


class TrainPipelineTests(unittest.TestCase):
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
            run = train_local_model(config, run_name="tiny")

            self.assertTrue((run / "best.pt").is_file())
            self.assertTrue((run / "bundle.pt").is_file())
            self.assertTrue((run / "selection_metrics.json").is_file())
            self.assertTrue((run / "calibration_fit.png").is_file())

            predictor = LocalPredictor.load(run / "bundle.pt", device="cpu")
            arrays = dataset.load_chunk(0)
            source = dataset.frame(dataset.chunks[0], arrays, 0)
            probabilities, valid = predictor.predict(source)
            self.assertEqual(probabilities.shape, source.label.shape)
            self.assertTrue(valid.all())
            self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
            timing = predictor.benchmark(source, repeats=2)
            self.assertEqual(timing["points"], len(source.xyz))


if __name__ == "__main__":
    unittest.main()
