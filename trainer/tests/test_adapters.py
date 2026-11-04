from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from smoke_trainer.adapters import GlobalPointAdapter, LocalVoxelAdapter, create_adapter_for_model
from smoke_trainer.data import Frame, FrameRef, UnifiedDataset
from smoke_trainer.evaluate import Calibration, frame_logits
from smoke_trainer.features import fit_normalizer
from smoke_trainer.models import LocalMLP
from smoke_trainer.predict import LocalPredictor, SmokePredictor

from helpers import make_dataset


class AdapterTests(unittest.TestCase):
    def _source_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = UnifiedDataset(make_dataset(root))
            arrays = dataset.load_chunk(0)
            return dataset.frame(dataset.chunks[0], arrays, 0)

    def test_local_adapter_preserves_l1_point_logits(self):
        source = self._source_frame()
        adapter = LocalVoxelAdapter()
        normalizer = adapter.fit_normalizer([source], {"voxel_size_m": 0.1})
        torch.manual_seed(3)
        model = LocalMLP(11, (8, 4)).eval()

        legacy_logits, legacy_labels = frame_logits(
            model, source, normalizer, 0.1, torch.device("cpu")
        )
        prepared = adapter.prepare(source, {"voxel_size_m": 0.1})
        with torch.inference_mode():
            logits = adapter.expand_logits(
                model(torch.from_numpy(normalizer.transform(prepared.features))), prepared
            ).numpy()

        np.testing.assert_allclose(logits, legacy_logits)
        np.testing.assert_array_equal(prepared.labels, legacy_labels)

    def test_legacy_l1_bundle_loads_through_generic_predictor(self):
        source = self._source_frame()
        normalizer = fit_normalizer([source], 0.1)
        torch.manual_seed(4)
        model = LocalMLP(11, (8, 4)).eval()
        legacy_bundle = {
            "format_version": 1,
            "model_name": "local_mlp",
            "model_parameters": {"name": "local_mlp", "hidden_widths": [8, 4]},
            "model_state": model.state_dict(),
            "feature_names": list(normalizer.feature_names),
            "normalizer": normalizer.to_dict(),
            "voxel_size_m": 0.1,
            "calibration": Calibration(1.0, 0.0).to_dict(),
            "threshold": 0.5,
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "legacy.pt"
            torch.save(legacy_bundle, path)
            generic = SmokePredictor.load(path, device="cpu")
            legacy = LocalPredictor.load(path, device="cpu")
            generic_probabilities, generic_valid = generic.predict(source)
            legacy_probabilities, legacy_valid = legacy.predict(source)

        self.assertEqual(generic.adapter.name, "local_voxel")
        np.testing.assert_allclose(generic_probabilities, legacy_probabilities)
        np.testing.assert_array_equal(generic_valid, legacy_valid)

    def test_local_adapter_is_inferred_for_legacy_config(self):
        adapter = create_adapter_for_model({"name": "local_mlp", "hidden_widths": [8, 4]})
        self.assertEqual(adapter.name, "local_voxel")

    def test_global_adapter_excludes_nonfinite_inputs_but_preserves_point_order(self):
        source = Frame(
            ref=FrameRef(0, 0),
            source_domain="test",
            session_id="test",
            recording_id="test",
            condition="smoke",
            timestamp_s=0.0,
            xyz=np.asarray([[1, 0, 0], [np.nan, 1, 0], [0, 2, 0]], dtype=np.float32),
            intensity=np.asarray([10, 20, np.inf], dtype=np.float32),
            label=np.asarray([0, 1, 255], dtype=np.uint8),
        )
        prepared = GlobalPointAdapter().prepare(source, {})
        self.assertEqual(prepared.features.shape, (1, 5))
        np.testing.assert_array_equal(prepared.valid, [True, False, False])
        np.testing.assert_array_equal(prepared.point_index, [0, -1, -1])

    def test_global_rotation_augmentation_preserves_range_and_intensity(self):
        adapter = GlobalPointAdapter()
        features = np.asarray([[1, 2, 3, np.sqrt(14), 17]], dtype=np.float32)
        augmented = adapter.augment_features(
            features,
            {
                "yaw_augmentation_deg": 180.0,
                "pitch_augmentation_deg": 0.0,
                "roll_augmentation_deg": 0.0,
            },
            np.random.default_rng(4),
        )
        self.assertFalse(np.allclose(augmented[:, :3], features[:, :3]))
        np.testing.assert_allclose(augmented[:, 3:], features[:, 3:])
        np.testing.assert_allclose(
            np.linalg.norm(augmented[:, :3], axis=1), augmented[:, 3]
        )


if __name__ == "__main__":
    unittest.main()
