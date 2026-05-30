import unittest

import torch

from models.decoupled_cdm import DecoupledCDM
from models.ensemble_cdm import DecoupledCDMEnsemble


def _subset_forward_inputs() -> dict[str, torch.Tensor | None]:
    student_exercise_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    response_matrix = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    student_concept_evidence = torch.zeros(4, 3, 6, dtype=torch.float32)
    student_concept_evidence[..., 0] = torch.tensor(
        [[2.0, 1.0, 1.0], [1.0, 2.0, 0.0], [1.0, 0.0, 1.0], [0.0, 2.0, 2.0]],
        dtype=torch.float32,
    )
    student_concept_evidence[..., 1] = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    student_concept_evidence[..., 3] = student_concept_evidence[..., 1] / student_concept_evidence[
        ..., 0
    ].clamp_min(1.0)
    student_concept_evidence[..., 4] = torch.log1p(student_concept_evidence[..., 0])
    student_concept_evidence[..., 5] = (student_concept_evidence[..., 0] > 0).to(torch.float32)
    return {
        "q_matrix": torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        "concept_graph": torch.tensor(
            [[1.0, 0.2, 0.0], [0.1, 1.0, 0.3], [0.0, 0.4, 1.0]],
            dtype=torch.float32,
        ),
        "student_exercise_mask": student_exercise_mask,
        "response_matrix": response_matrix,
        "student_tkc_mask": torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
            dtype=torch.float32,
        ),
        "student_ukc_mask": torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "student_concept_evidence": student_concept_evidence,
        "target_student_ids": torch.tensor([3, 1, 3, 0, 2], dtype=torch.long),
        "target_exercise_ids": torch.tensor([4, 2, 3, 0, 1], dtype=torch.long),
    }


class CognitiveDifficultyAdapterTest(unittest.TestCase):
    def test_adapter_output_layer_starts_at_zero(self) -> None:
        model = DecoupledCDM(num_students=2, num_exercises=3, num_concepts=2, concept_dim=4)

        final_layer = model.cognitive_difficulty_adapter[-1]
        torch.testing.assert_close(final_layer.weight, torch.zeros_like(final_layer.weight))
        torch.testing.assert_close(final_layer.bias, torch.zeros_like(final_layer.bias))

    def test_adapter_does_not_backpropagate_into_difficulty_input(self) -> None:
        model = DecoupledCDM(num_students=2, num_exercises=3, num_concepts=2, concept_dim=4)
        with torch.no_grad():
            model.cognitive_difficulty_adapter[-1].weight.fill_(1.0)
            model.cognitive_difficulty_adapter[-1].bias.fill_(0.0)

        match_inputs = torch.randn(5, 16, requires_grad=True)
        difficulty = torch.randn(5, requires_grad=True)
        adapter_inputs = torch.cat([match_inputs.detach(), difficulty.detach().unsqueeze(-1)], dim=-1)
        output = model.cognitive_difficulty_adapter(adapter_inputs).sum()

        output.backward()

        self.assertIsNone(match_inputs.grad)
        self.assertIsNone(difficulty.grad)


class DualTowerCDMEnsembleTest(unittest.TestCase):
    def test_requires_positive_secondary_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "secondary_concept_dim"):
            DecoupledCDMEnsemble(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                secondary_concept_dim=0,
            )

    def test_runtime_attrs_sync_to_both_towers(self) -> None:
        model = DecoupledCDMEnsemble(
            num_students=2,
            num_exercises=3,
            num_concepts=2,
            concept_dim=4,
            secondary_concept_dim=5,
            concept_evidence_prior_residual=True,
            concept_evidence_prior_max_logit=0.5,
        )

        model.concept_evidence_prior_residual = False
        model.concept_evidence_prior_max_logit = 0.25
        model._sync_runtime_attrs()

        self.assertFalse(model.primary.concept_evidence_prior_residual)
        self.assertFalse(model.secondary.concept_evidence_prior_residual)
        self.assertEqual(model.primary.concept_evidence_prior_max_logit, 0.25)
        self.assertEqual(model.secondary.concept_evidence_prior_max_logit, 0.25)

    def test_student_subset_forward_matches_full_forward(self) -> None:
        torch.manual_seed(7)
        model = DecoupledCDMEnsemble(
            num_students=4,
            num_exercises=5,
            num_concepts=3,
            concept_dim=4,
            secondary_concept_dim=5,
        )
        model.eval()
        inputs = _subset_forward_inputs()

        full_output = model(**inputs, use_student_subset=False)
        subset_output = model(**inputs, use_student_subset=True)

        torch.testing.assert_close(subset_output.probs, full_output.probs)
        torch.testing.assert_close(subset_output.primary_probs, full_output.primary_probs)
        torch.testing.assert_close(subset_output.secondary_probs, full_output.secondary_probs)
        unique_students = torch.unique(inputs["target_student_ids"], sorted=True)
        torch.testing.assert_close(subset_output.tkc_weight, full_output.tkc_weight[unique_students])


class StudentSubsetForwardTest(unittest.TestCase):
    def test_student_subset_forward_matches_full_forward(self) -> None:
        torch.manual_seed(11)
        model = DecoupledCDM(
            num_students=4,
            num_exercises=5,
            num_concepts=3,
            concept_dim=4,
            high_concept_logit_adapter=True,
            pairwise_history_interaction_adapter=True,
            gs_difficulty_adapter=True,
            interpretable_readout_expert_adapter=True,
            student_conditioned_ukc_readout_residual=True,
            concept_evidence_readout_residual=True,
            concept_evidence_readout_min_count=1,
        )
        model.eval()
        inputs = _subset_forward_inputs()

        full_output = model(**inputs, use_student_subset=False)
        subset_output = model(**inputs, use_student_subset=True)

        torch.testing.assert_close(subset_output.probs, full_output.probs)
        torch.testing.assert_close(subset_output.cognitive_probs, full_output.cognitive_probs)
        torch.testing.assert_close(subset_output.guess_probs, full_output.guess_probs)
        torch.testing.assert_close(subset_output.slip_probs, full_output.slip_probs)
        unique_students = torch.unique(inputs["target_student_ids"], sorted=True)
        torch.testing.assert_close(subset_output.tkc_weight, full_output.tkc_weight[unique_students])


class InterpretableReadoutExpertAdapterTest(unittest.TestCase):
    def test_expert_output_layers_start_at_zero(self) -> None:
        model = DecoupledCDM(
            num_students=2,
            num_exercises=3,
            num_concepts=2,
            concept_dim=4,
            interpretable_readout_expert_adapter=True,
            interpretable_readout_expert_count=3,
        )

        self.assertEqual(len(model.interpretable_readout_experts), 3)
        for expert in model.interpretable_readout_experts:
            final_layer = expert[-1]
            torch.testing.assert_close(final_layer.weight, torch.zeros_like(final_layer.weight))
            torch.testing.assert_close(final_layer.bias, torch.zeros_like(final_layer.bias))

    def test_expert_count_requires_at_least_two(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                interpretable_readout_expert_count=1,
            )


class StudentConditionedUkcReadoutResidualTest(unittest.TestCase):
    def test_residual_head_output_layer_starts_at_zero(self) -> None:
        model = DecoupledCDM(
            num_students=2,
            num_exercises=3,
            num_concepts=2,
            concept_dim=4,
            student_conditioned_ukc_readout_residual=True,
        )

        self.assertIsNotNone(model.student_conditioned_ukc_readout_residual_head)
        final_layer = model.student_conditioned_ukc_readout_residual_head[-1]
        torch.testing.assert_close(final_layer.weight, torch.zeros_like(final_layer.weight))
        torch.testing.assert_close(final_layer.bias, torch.zeros_like(final_layer.bias))

    def test_residual_only_fires_for_none_seen_targets_with_tkc_neighbors(self) -> None:
        model = DecoupledCDM(
            num_students=2,
            num_exercises=2,
            num_concepts=2,
            concept_dim=2,
            student_conditioned_ukc_readout_residual=True,
        )
        head = model.student_conditioned_ukc_readout_residual_head
        self.assertIsNotNone(head)
        with torch.no_grad():
            head[-1].weight.zero_()
            head[-1].bias.fill_(1.0)

        q_vectors = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        output = model._build_student_conditioned_ukc_readout_residual(
            q_vectors=q_vectors,
            concept_graph=torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
            q_matrix=torch.eye(2, dtype=torch.float32),
            student_exercise_mask=torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32),
            student_tkc_mask=torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32),
            target_student_ids=torch.tensor([0, 0, 1], dtype=torch.long),
            tkc_states=torch.tensor(
                [
                    [[2.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ],
                dtype=torch.float32,
            ),
            ukc_states=torch.zeros(2, 2, 2, dtype=torch.float32),
            student_state=torch.zeros(3, 2, dtype=torch.float32),
            q_repr=torch.zeros(3, 2, dtype=torch.float32),
            concept_summary=(
                torch.ones(3, 1, dtype=torch.float32),
                torch.zeros(3, 2, dtype=torch.float32),
                torch.zeros(3, 2, dtype=torch.float32),
            ),
            difficulty=torch.zeros(3, dtype=torch.float32),
        )

        torch.testing.assert_close(output, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32))


class ConceptEvidenceReadoutResidualTest(unittest.TestCase):
    def test_residual_head_output_layer_starts_at_zero(self) -> None:
        model = DecoupledCDM(
            num_students=2,
            num_exercises=3,
            num_concepts=2,
            concept_dim=4,
            concept_evidence_readout_residual=True,
        )

        self.assertIsNotNone(model.concept_evidence_readout_residual_head)
        final_layer = model.concept_evidence_readout_residual_head[-1]
        torch.testing.assert_close(final_layer.weight, torch.zeros_like(final_layer.weight))
        torch.testing.assert_close(final_layer.bias, torch.zeros_like(final_layer.bias))

    def test_residual_rejects_invalid_trigger_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_count"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                concept_evidence_readout_min_count=2,
                concept_evidence_readout_max_count=1,
            )
        with self.assertRaisesRegex(ValueError, "min_seen_ratio"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                concept_evidence_readout_min_seen_ratio=1.1,
            )
        with self.assertRaisesRegex(ValueError, "max_logit"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                concept_evidence_readout_max_logit=0.0,
            )

    def test_residual_only_fires_for_fully_seen_multiconcept_targets(self) -> None:
        model = DecoupledCDM(
            num_students=2,
            num_exercises=3,
            num_concepts=3,
            concept_dim=4,
            concept_evidence_readout_residual=True,
            concept_evidence_readout_min_count=2,
            concept_evidence_readout_min_seen_ratio=1.0,
            concept_evidence_readout_max_logit=0.5,
        )
        head = model.concept_evidence_readout_residual_head
        self.assertIsNotNone(head)
        with torch.no_grad():
            head[-1].weight.zero_()
            head[-1].bias.fill_(1.0)

        q_vectors = torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        student_concept_evidence = torch.zeros(2, 3, 6, dtype=torch.float32)
        student_concept_evidence[0, 0] = torch.tensor(
            [3.0, 2.0, 1.0, 2.0 / 3.0, 1.3862944, 1.0],
            dtype=torch.float32,
        )
        student_concept_evidence[0, 1] = torch.tensor(
            [2.0, 2.0, 0.0, 1.0, 1.0986123, 1.0],
            dtype=torch.float32,
        )
        student_concept_evidence[1, 0] = torch.tensor(
            [4.0, 1.0, 3.0, 0.25, 1.6094379, 1.0],
            dtype=torch.float32,
        )

        output = model._build_concept_evidence_readout_residual(
            q_vectors=q_vectors,
            target_student_ids=torch.tensor([0, 1, 0], dtype=torch.long),
            student_concept_evidence=student_concept_evidence,
            concept_summary=(
                torch.tensor([[2.0], [2.0], [1.0]], dtype=torch.float32),
                torch.zeros(3, 4, dtype=torch.float32),
                torch.zeros(3, 4, dtype=torch.float32),
            ),
            difficulty=torch.zeros(3, dtype=torch.float32),
        )

        self.assertGreater(float(output[0]), 0.0)
        torch.testing.assert_close(output[1:], torch.zeros(2, dtype=torch.float32))

    def test_residual_can_be_scoped_to_single_concept_targets(self) -> None:
        model = DecoupledCDM(
            num_students=2,
            num_exercises=2,
            num_concepts=2,
            concept_dim=4,
            concept_evidence_readout_residual=True,
            concept_evidence_readout_min_count=1,
            concept_evidence_readout_max_count=1,
            concept_evidence_readout_min_seen_ratio=1.0,
            concept_evidence_readout_max_logit=0.5,
        )
        head = model.concept_evidence_readout_residual_head
        self.assertIsNotNone(head)
        with torch.no_grad():
            head[-1].weight.zero_()
            head[-1].bias.fill_(1.0)

        q_vectors = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        student_concept_evidence = torch.zeros(1, 2, 6, dtype=torch.float32)
        student_concept_evidence[0, 0] = torch.tensor(
            [3.0, 2.0, 1.0, 2.0 / 3.0, 1.3862944, 1.0],
            dtype=torch.float32,
        )
        student_concept_evidence[0, 1] = torch.tensor(
            [2.0, 2.0, 0.0, 1.0, 1.0986123, 1.0],
            dtype=torch.float32,
        )

        output = model._build_concept_evidence_readout_residual(
            q_vectors=q_vectors,
            target_student_ids=torch.tensor([0, 0], dtype=torch.long),
            student_concept_evidence=student_concept_evidence,
            concept_summary=(
                torch.tensor([[1.0], [2.0]], dtype=torch.float32),
                torch.zeros(2, 4, dtype=torch.float32),
                torch.zeros(2, 4, dtype=torch.float32),
            ),
            difficulty=torch.zeros(2, dtype=torch.float32),
        )

        self.assertGreater(float(output[0]), 0.0)
        torch.testing.assert_close(output[1], torch.tensor(0.0, dtype=torch.float32))


class ConceptEvidencePriorResidualTest(unittest.TestCase):
    def test_prior_rejects_invalid_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "prior_strength"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                concept_evidence_prior_strength=0.0,
            )
        with self.assertRaisesRegex(ValueError, "confidence_cap"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                concept_evidence_prior_confidence_cap=0.0,
            )
        with self.assertRaisesRegex(ValueError, "max_count"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                concept_evidence_prior_min_count=2,
                concept_evidence_prior_max_count=1,
            )
        with self.assertRaisesRegex(ValueError, "min_confidence"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                concept_evidence_prior_min_confidence=1.1,
            )
        with self.assertRaisesRegex(ValueError, "min_abs_mastery"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                concept_evidence_prior_min_abs_mastery=1.1,
            )
        with self.assertRaisesRegex(ValueError, "positive_scale"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                concept_evidence_prior_positive_scale=-0.1,
            )
        with self.assertRaisesRegex(ValueError, "negative_scale"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                concept_evidence_prior_negative_scale=-0.1,
            )
        with self.assertRaisesRegex(ValueError, "concept_evidence_prior_apply_mode"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                concept_evidence_prior_apply_mode="unknown",
            )

    def test_prior_uses_smoothed_target_concept_accuracy(self) -> None:
        model = DecoupledCDM(
            num_students=2,
            num_exercises=3,
            num_concepts=3,
            concept_dim=4,
            concept_evidence_prior_residual=True,
            concept_evidence_prior_min_count=2,
            concept_evidence_prior_max_count=2,
            concept_evidence_prior_min_seen_ratio=1.0,
            concept_evidence_prior_max_logit=0.5,
            concept_evidence_prior_strength=2.0,
            concept_evidence_prior_confidence_cap=20.0,
        )
        q_vectors = torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        student_concept_evidence = torch.zeros(2, 3, 6, dtype=torch.float32)
        student_concept_evidence[0, 0] = torch.tensor([3.0, 3.0, 0.0, 1.0, 1.3862944, 1.0])
        student_concept_evidence[0, 1] = torch.tensor([3.0, 3.0, 0.0, 1.0, 1.3862944, 1.0])
        student_concept_evidence[0, 2] = torch.tensor([3.0, 3.0, 0.0, 1.0, 1.3862944, 1.0])
        student_concept_evidence[1, 0] = torch.tensor([3.0, 0.0, 3.0, 0.0, 1.3862944, 1.0])

        output = model._build_concept_evidence_prior_residual(
            q_vectors=q_vectors,
            target_student_ids=torch.tensor([0, 1, 0, 0], dtype=torch.long),
            student_concept_evidence=student_concept_evidence,
            concept_summary=(
                torch.tensor([[2.0], [2.0], [1.0], [3.0]], dtype=torch.float32),
                torch.zeros(4, 4, dtype=torch.float32),
                torch.zeros(4, 4, dtype=torch.float32),
            ),
        )

        self.assertGreater(float(output[0]), 0.0)
        torch.testing.assert_close(output[1:], torch.zeros(3, dtype=torch.float32))

    def test_prior_stability_gates_mask_weak_evidence(self) -> None:
        model = DecoupledCDM(
            num_students=1,
            num_exercises=2,
            num_concepts=2,
            concept_dim=4,
            concept_evidence_prior_residual=True,
            concept_evidence_prior_min_count=1,
            concept_evidence_prior_min_seen_ratio=1.0,
            concept_evidence_prior_min_confidence=0.75,
            concept_evidence_prior_min_abs_mastery=0.5,
            concept_evidence_prior_max_logit=0.5,
        )
        q_vectors = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        student_concept_evidence = torch.zeros(1, 2, 6, dtype=torch.float32)
        student_concept_evidence[0, 0] = torch.tensor([20.0, 20.0, 0.0, 1.0, 3.0, 1.0])

        output = model._build_concept_evidence_prior_residual(
            q_vectors=q_vectors,
            target_student_ids=torch.tensor([0, 0], dtype=torch.long),
            student_concept_evidence=student_concept_evidence,
            concept_summary=(
                torch.tensor([[1.0], [2.0]], dtype=torch.float32),
                torch.zeros(2, 4, dtype=torch.float32),
                torch.zeros(2, 4, dtype=torch.float32),
            ),
        )

        self.assertGreater(float(output[0]), 0.0)
        torch.testing.assert_close(output[1], torch.tensor(0.0, dtype=torch.float32))

    def test_prior_direction_scales_adjust_signed_evidence(self) -> None:
        model = DecoupledCDM(
            num_students=2,
            num_exercises=2,
            num_concepts=1,
            concept_dim=4,
            concept_evidence_prior_residual=True,
            concept_evidence_prior_min_count=1,
            concept_evidence_prior_min_seen_ratio=1.0,
            concept_evidence_prior_max_logit=0.5,
            concept_evidence_prior_positive_scale=0.0,
            concept_evidence_prior_negative_scale=0.5,
        )
        q_vectors = torch.ones(2, 1, dtype=torch.float32)
        student_concept_evidence = torch.zeros(2, 1, 6, dtype=torch.float32)
        student_concept_evidence[0, 0] = torch.tensor([20.0, 20.0, 0.0, 1.0, 3.0, 1.0])
        student_concept_evidence[1, 0] = torch.tensor([20.0, 0.0, 20.0, 0.0, 3.0, 1.0])

        output = model._build_concept_evidence_prior_residual(
            q_vectors=q_vectors,
            target_student_ids=torch.tensor([0, 1], dtype=torch.long),
            student_concept_evidence=student_concept_evidence,
            concept_summary=(
                torch.ones(2, 1, dtype=torch.float32),
                torch.zeros(2, 4, dtype=torch.float32),
                torch.zeros(2, 4, dtype=torch.float32),
            ),
        )

        torch.testing.assert_close(output[0], torch.tensor(0.0, dtype=torch.float32))
        self.assertLess(float(output[1]), 0.0)

    def test_prior_apply_mode_respects_training_state(self) -> None:
        model = DecoupledCDM(
            num_students=1,
            num_exercises=1,
            num_concepts=1,
            concept_dim=4,
            concept_evidence_prior_residual=True,
            concept_evidence_prior_apply_mode="eval_only",
        )

        model.train()
        self.assertFalse(model._should_apply_concept_evidence_prior())
        model.eval()
        self.assertTrue(model._should_apply_concept_evidence_prior())


class HistoryEvidenceLogitPriorResidualTest(unittest.TestCase):
    def test_logit_prior_rejects_invalid_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "logit_prior_min_count"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                history_evidence_logit_prior_min_count=0,
            )
        with self.assertRaisesRegex(ValueError, "logit_prior_max_logit"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                history_evidence_logit_prior_max_logit=0.0,
            )
        with self.assertRaisesRegex(ValueError, "prior_weight"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                history_evidence_logit_prior_prior_weight=0.0,
            )
        with self.assertRaisesRegex(ValueError, "logit_prior_location"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                history_evidence_logit_prior_location="unknown",
            )
        DecoupledCDM(
            num_students=2,
            num_exercises=3,
            num_concepts=2,
            concept_dim=4,
            history_evidence_logit_prior_location="loss_only",
        )

    def test_logit_prior_is_bounded_signed_and_triggered(self) -> None:
        model = DecoupledCDM(
            num_students=2,
            num_exercises=3,
            num_concepts=3,
            concept_dim=4,
            history_evidence_logit_prior_residual=True,
            history_evidence_logit_prior_min_count=2,
            history_evidence_logit_prior_min_seen_ratio=0.5,
            history_evidence_logit_prior_max_logit=0.25,
            history_evidence_logit_prior_weight_student=1.0,
            history_evidence_logit_prior_weight_exercise=1.0,
            history_evidence_logit_prior_weight_target_concept=1.0,
            history_evidence_logit_prior_weight_concept=0.0,
            history_evidence_logit_prior_weight_mastery=1.0,
        )
        q_vectors = torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        student_concept_evidence = torch.zeros(2, 3, 6, dtype=torch.float32)
        student_concept_evidence[0, 0] = torch.tensor([4.0, 4.0, 0.0, 1.0, 1.6094379, 1.0])
        student_concept_evidence[0, 1] = torch.tensor([4.0, 4.0, 0.0, 1.0, 1.6094379, 1.0])
        student_concept_evidence[1, 0] = torch.tensor([4.0, 0.0, 4.0, 0.0, 1.6094379, 1.0])
        student_concept_evidence[1, 1] = torch.tensor([4.0, 0.0, 4.0, 0.0, 1.6094379, 1.0])
        exercise_evidence = torch.tensor(
            [
                [4.0, 4.0, 0.0, 1.0, 1.6094379, 1.0],
                [4.0, 0.0, 4.0, 0.0, 1.6094379, 1.0],
                [4.0, 2.0, 2.0, 0.5, 1.6094379, 1.0],
            ],
            dtype=torch.float32,
        )
        student_exercise_mask = torch.ones(2, 3, dtype=torch.float32)
        response_matrix = torch.tensor(
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
            dtype=torch.float32,
        )

        output = model._build_history_evidence_logit_prior_residual(
            q_vectors=q_vectors,
            target_student_ids=torch.tensor([0, 1, 0], dtype=torch.long),
            target_exercise_ids=torch.tensor([0, 1, 2], dtype=torch.long),
            student_concept_evidence=student_concept_evidence,
            exercise_evidence=exercise_evidence,
            student_exercise_mask=student_exercise_mask,
            response_matrix=response_matrix,
            concept_summary=(
                torch.tensor([[2.0], [2.0], [1.0]], dtype=torch.float32),
                torch.zeros(3, 4, dtype=torch.float32),
                torch.zeros(3, 4, dtype=torch.float32),
            ),
        )

        self.assertGreater(float(output[0]), 0.0)
        self.assertLess(float(output[1]), 0.0)
        torch.testing.assert_close(output[2:], torch.zeros(1, dtype=torch.float32))
        self.assertLessEqual(float(output.abs().max()), 0.25)


if __name__ == "__main__":
    unittest.main()
