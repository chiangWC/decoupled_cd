import unittest

import torch

from models.hetero_propagation import HeterogeneousGraphPropagation, _build_exercise_component, _logit


class ExerciseComponentAggregationTest(unittest.TestCase):
    def test_unrelated_history_does_not_shrink_concept_behavior_signal(self) -> None:
        exercise_messages = torch.tensor(
            [
                [1.0],
                [10.0],
                [100.0],
                [1000.0],
            ],
            dtype=torch.float32,
        )
        q_matrix = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        concept_0_only_history = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        concept_0_with_extra_unrelated_history = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)

        only_history_output = _build_exercise_component(
            weighted_exercises=concept_0_only_history,
            q_matrix=q_matrix,
            exercise_messages=exercise_messages,
        )
        extra_history_output = _build_exercise_component(
            weighted_exercises=concept_0_with_extra_unrelated_history,
            q_matrix=q_matrix,
            exercise_messages=exercise_messages,
        )

        self.assertAlmostEqual(only_history_output[0, 0, 0].item(), 5.5, places=6)
        self.assertAlmostEqual(extra_history_output[0, 0, 0].item(), 5.5, places=6)
        self.assertAlmostEqual(extra_history_output[0, 1, 0].item(), 550.0, places=6)


class GraphModeValidationTest(unittest.TestCase):
    def test_single_graph_mode_rejects_legacy_dual_graph_inputs(self) -> None:
        propagation = HeterogeneousGraphPropagation(concept_dim=2, graph_mode="single")

        with self.assertRaisesRegex(ValueError, "Single graph mode does not accept"):
            propagation(
                concept_embeddings=torch.eye(2, dtype=torch.float32),
                exercise_embeddings=torch.ones(1, 2, dtype=torch.float32),
                q_matrix=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                concept_graph=torch.eye(2, dtype=torch.float32),
                prerequisite_graph=torch.eye(2, dtype=torch.float32),
                similarity_graph=torch.eye(2, dtype=torch.float32),
                student_exercise_mask=torch.ones(1, 1, dtype=torch.float32),
                response_matrix=torch.ones(1, 1, dtype=torch.float32),
                student_tkc_mask=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                student_ukc_mask=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            )

    def test_dual_graph_mode_requires_both_legacy_dual_graph_inputs(self) -> None:
        propagation = HeterogeneousGraphPropagation(concept_dim=2, graph_mode="dual")

        with self.assertRaisesRegex(ValueError, "Dual graph mode requires both"):
            propagation(
                concept_embeddings=torch.eye(2, dtype=torch.float32),
                exercise_embeddings=torch.ones(1, 2, dtype=torch.float32),
                q_matrix=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                concept_graph=torch.eye(2, dtype=torch.float32),
                prerequisite_graph=torch.eye(2, dtype=torch.float32),
                similarity_graph=None,
                student_exercise_mask=torch.ones(1, 1, dtype=torch.float32),
                response_matrix=torch.ones(1, 1, dtype=torch.float32),
                student_tkc_mask=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                student_ukc_mask=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            )


class StudentGatePriorInitializationTest(unittest.TestCase):
    def test_new_student_gate_prior_names_preserve_old_initialization_ratio(self) -> None:
        propagation = HeterogeneousGraphPropagation(
            concept_dim=2,
            student_gate_prior_alpha=2.0,
            student_gate_prior_beta=1.0,
        )

        self.assertAlmostEqual(propagation.student_fusion_gate[-1].bias.item(), _logit(2.0 / 3.0), places=6)

    def test_legacy_alpha_beta_aliases_still_map_to_student_gate_prior(self) -> None:
        propagation = HeterogeneousGraphPropagation(concept_dim=2, alpha=3.0, beta=1.0)

        self.assertAlmostEqual(propagation.student_fusion_gate[-1].bias.item(), _logit(0.75), places=6)


if __name__ == "__main__":
    unittest.main()
