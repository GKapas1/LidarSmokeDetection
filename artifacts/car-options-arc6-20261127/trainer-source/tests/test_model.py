from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from smoke_trainer.models import GlobalPointNet, GlobalTemporalPointNet, LocalMLP, VoxelPointNet


class ModelTests(unittest.TestCase):
    def test_voxel_pointnet_is_local_and_permutation_equivariant(self):
        torch.manual_seed(5)
        model = VoxelPointNet(5, (8, 12), (6,)).eval()
        features = torch.tensor([[0.1, 0, 0, 1, 0], [-0.1, 0, 0, 2, 0],
                                 [0, 0, 0, 3, 1]], dtype=torch.float32)
        output = model(features)
        torch.testing.assert_close(output[0], output[1])
        permutation = torch.tensor([2, 0, 1])
        torch.testing.assert_close(model(features[permutation]), output[permutation])
        changed = features.clone()
        changed[2, 3] = 1000
        torch.testing.assert_close(model(changed)[:2], output[:2])
        self.assertEqual(model(features[:0]).shape, (0,))

    def test_local_model_returns_one_logit_per_voxel(self):
        model = LocalMLP(11, (8, 4))
        output = model(torch.zeros((7, 11)))
        self.assertEqual(tuple(output.shape), (7,))

    def test_ignored_points_have_no_loss_or_gradient_effect(self):
        voxel_logits = torch.tensor([0.2, -0.4], requires_grad=True)
        inverse = torch.tensor([0, 0, 1])
        labels = torch.tensor([1.0, 255.0, 0.0])
        mask = labels != 255
        loss = F.binary_cross_entropy_with_logits(voxel_logits[inverse][mask], labels[mask])
        loss.backward()
        first_gradient = voxel_logits.grad.clone()

        voxel_logits.grad.zero_()
        labels[1] = 0.0
        loss = F.binary_cross_entropy_with_logits(
            voxel_logits[inverse][torch.tensor([True, False, True])],
            labels[torch.tensor([True, False, True])],
        )
        loss.backward()
        torch.testing.assert_close(voxel_logits.grad, first_gradient)

    def test_global_pointnet_returns_one_logit_per_point(self):
        model = GlobalPointNet(
            5,
            point_widths=(8, 12),
            head_widths=(6,),
            input_alignment="tnet",
            alignment_widths=(4, 8),
        )
        output = model(torch.zeros((7, 5)))
        self.assertEqual(tuple(output.shape), (7,))

    def test_global_pointnet_is_permutation_equivariant(self):
        torch.manual_seed(11)
        model = GlobalPointNet(
            5,
            point_widths=(8, 12),
            head_widths=(6,),
            input_alignment="tnet",
            alignment_widths=(4, 8),
        ).eval()
        features = torch.randn((9, 5))
        permutation = torch.tensor([6, 0, 8, 1, 4, 7, 3, 5, 2])
        original = model(features)
        permuted = model(features[permutation])
        torch.testing.assert_close(permuted, original[permutation])

    def test_input_tnet_starts_as_identity_and_has_no_initial_penalty(self):
        model = GlobalPointNet(
            5,
            point_widths=(8, 12),
            head_widths=(6,),
            input_alignment="tnet",
            alignment_widths=(4, 8),
        )
        features = torch.randn((5, 5))
        model(features)
        self.assertIsNotNone(model._last_transform)
        torch.testing.assert_close(model._last_transform, torch.eye(3))
        self.assertEqual(float(model.regularization_loss().detach()), 0.0)

    def test_global_temporal_pointnet_returns_current_point_logits(self):
        model = GlobalTemporalPointNet(
            5,
            point_widths=(8, 12),
            head_widths=(6,),
            history_length=3,
            gru_hidden_width=5,
            input_alignment="tnet",
            alignment_widths=(4, 8),
        )
        output = model((torch.zeros((4, 5)), torch.zeros((7, 5))))
        self.assertEqual(tuple(output.shape), (7,))

    def test_global_temporal_pointnet_is_current_scan_permutation_equivariant(self):
        torch.manual_seed(13)
        model = GlobalTemporalPointNet(
            5,
            point_widths=(8, 12),
            head_widths=(6,),
            history_length=3,
            gru_hidden_width=5,
            input_alignment="tnet",
            alignment_widths=(4, 8),
        ).eval()
        history = torch.randn((5, 5))
        current = torch.randn((9, 5))
        permutation = torch.tensor([6, 0, 8, 1, 4, 7, 3, 5, 2])
        original = model((history, current))
        permuted = model((history, current[permutation]))
        torch.testing.assert_close(permuted, original[permutation])

    def test_global_temporal_pointnet_accepts_startup_history(self):
        model = GlobalTemporalPointNet(
            5,
            point_widths=(8, 12),
            head_widths=(6,),
            history_length=3,
            gru_hidden_width=5,
        )
        self.assertEqual(tuple(model((torch.zeros((2, 5)),)).shape), (2,))


if __name__ == "__main__":
    unittest.main()
