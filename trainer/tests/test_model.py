from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from smoke_trainer.models import LocalMLP


class ModelTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

