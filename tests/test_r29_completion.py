from __future__ import annotations

import unittest

import torch

from models.r29_completion import R29CompletionCDM


def inputs() -> dict[str, torch.Tensor]:
    q_matrix = torch.tensor([[1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0]], dtype=torch.float32)
    evidence = torch.zeros(5, 3, 6)
    evidence[:, :, 0] = torch.tensor(
        [[2, 0, 1], [1, 2, 0], [0, 1, 3], [2, 0, 2], [1, 1, 0]], dtype=torch.float32
    )
    evidence[:, :, 1] = torch.tensor(
        [[1, 0, 1], [1, 1, 0], [0, 0, 2], [2, 0, 1], [0, 1, 0]], dtype=torch.float32
    )
    return {
        "q_matrix": q_matrix,
        "concept_graph": torch.eye(3),
        "student_exercise_mask": torch.ones(5, 4),
        "response_matrix": torch.rand(5, 4),
        "student_tkc_mask": (evidence[..., 0] > 0).float(),
        "student_ukc_mask": (evidence[..., 0] == 0).float(),
        "student_concept_evidence": evidence,
        "exercise_evidence": torch.ones(4, 6),
        "target_student_ids": torch.tensor([0, 1, 2, 3, 4]),
        "target_exercise_ids": torch.tensor([0, 1, 2, 3, 0]),
        "use_student_subset": True,
    }


class R29CompletionTest(unittest.TestCase):
    def build(self, mode: str) -> R29CompletionCDM:
        torch.manual_seed(42)
        return R29CompletionCDM(
            num_students=5,
            num_exercises=4,
            num_concepts=3,
            concept_dim=8,
            state_completer=mode,
        )

    def test_meta_and_capacity_are_exact_parameter_controls(self) -> None:
        meta = self.build("meta_implicit")
        capacity = self.build("capacity_control")
        self.assertEqual(meta.active_parameter_count(), capacity.active_parameter_count())
        self.assertEqual(meta.common_initialization_hash(), capacity.common_initialization_hash())

    def test_meta_outputs_and_gradients(self) -> None:
        model = self.build("meta_implicit")
        output = model(**inputs())
        self.assertEqual(output.framework_state.shape, (5, 3, 8))
        self.assertEqual(output.mastery.shape, (5, 3))
        self.assertEqual(output.probs.shape, (5,))
        torch.nn.functional.binary_cross_entropy(output.probs, torch.rand(5)).backward()
        diagnostic_only = ("mastery_head.", "reconstruction_head.", "completer.reliability_head.")
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith(diagnostic_only)
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_framework_state_is_the_only_student_specific_prediction_path(self) -> None:
        model = self.build("meta_implicit").eval()
        arguments = inputs()
        first = model(**arguments).probs.detach()
        altered = {key: value.clone() if torch.is_tensor(value) else value for key, value in arguments.items()}
        altered["student_concept_evidence"].zero_()
        second = model(**altered).probs.detach()
        self.assertFalse(torch.allclose(first, second))
        self.assertFalse(any("student_embedding" in name for name, _ in model.named_parameters()))

    def test_rejected_candidates_cannot_run(self) -> None:
        for mode in ("peer_completion", "wasserstein_flow"):
            with self.assertRaisesRegex(ValueError, "rejected"):
                self.build(mode)


if __name__ == "__main__":
    unittest.main()
