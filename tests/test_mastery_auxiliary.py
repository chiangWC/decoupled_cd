from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from models.mastery_auxiliary import MasteryAuxiliaryObjective


class MasteryAuxiliaryObjectiveTests(unittest.TestCase):
    def test_historical_monotonic_bce_formula_and_fixed_scale_buffer(self) -> None:
        objective = MasteryAuxiliaryObjective(aux_weight=0.5)
        mastery_logits = torch.tensor([[0.0, 2.0], [-1.0, 1.0]])
        difficulty_logits = torch.tensor([[1.0, -2.0], [0.5, -0.5]])
        q_mask = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
        labels = torch.tensor([1.0, 0.0])

        loss = objective(
            mastery_logits,
            difficulty_logits,
            q_mask,
            labels,
            epoch=1,
            total_epochs=10,
        )

        expected_scale = F.softplus(torch.tensor(2.0))
        expected_logits = expected_scale * (
            q_mask * (torch.sigmoid(mastery_logits) - torch.sigmoid(difficulty_logits))
        ).sum(dim=1) / q_mask.sum(dim=1).clamp_min(1.0)
        expected = 0.5 * F.binary_cross_entropy_with_logits(expected_logits, labels)
        self.assertTrue(torch.allclose(loss, expected))
        self.assertAlmostEqual(float(objective.scale), 2.126928, places=6)
        self.assertEqual(dict(objective.named_parameters()), {})
        self.assertIn("scale", dict(objective.named_buffers()))

    def test_detach_item_difficulty_blocks_only_difficulty_gradient(self) -> None:
        objective = MasteryAuxiliaryObjective(
            aux_weight=1.0,
            detach_item_difficulty=True,
        )
        mastery_logits = torch.tensor([[0.0, 0.5]], requires_grad=True)
        difficulty_logits = torch.tensor([[0.2, -0.2]], requires_grad=True)

        loss = objective(
            mastery_logits,
            difficulty_logits,
            torch.ones_like(mastery_logits),
            torch.ones(1),
            epoch=1,
            total_epochs=5,
        )
        loss.backward()

        self.assertIsNotNone(mastery_logits.grad)
        self.assertGreater(float(mastery_logits.grad.abs().sum()), 0.0)
        self.assertIsNone(difficulty_logits.grad)

    def test_warmup_linearly_scales_weight_over_leading_epoch_fraction(self) -> None:
        objective = MasteryAuxiliaryObjective(
            aux_weight=0.8,
            warmup_fraction=0.4,
        )

        self.assertTrue(math.isclose(objective.effective_weight(1, 10), 0.2))
        self.assertTrue(math.isclose(objective.effective_weight(2, 10), 0.4))
        self.assertTrue(math.isclose(objective.effective_weight(4, 10), 0.8))
        self.assertTrue(math.isclose(objective.effective_weight(8, 10), 0.8))

    def test_zero_weight_returns_detached_zero_without_touching_backbone_graph(self) -> None:
        objective = MasteryAuxiliaryObjective(aux_weight=0.0)
        mastery_logits = torch.tensor([[0.0]], requires_grad=True)
        difficulty_logits = torch.tensor([[0.0]], requires_grad=True)

        loss = objective(
            mastery_logits,
            difficulty_logits,
            torch.ones(1, 1),
            torch.ones(1),
            epoch=1,
            total_epochs=1,
        )

        self.assertEqual(float(loss), 0.0)
        self.assertFalse(loss.requires_grad)
        self.assertIsNone(loss.grad_fn)
        self.assertIsNone(mastery_logits.grad)
        self.assertIsNone(difficulty_logits.grad)

    def test_invalid_hyperparameters_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "aux_weight"):
            MasteryAuxiliaryObjective(aux_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "warmup_fraction"):
            MasteryAuxiliaryObjective(aux_weight=1.0, warmup_fraction=1.1)


if __name__ == "__main__":
    unittest.main()
