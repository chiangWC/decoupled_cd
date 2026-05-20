import unittest

import torch

from models.decoupled_cdm import DecoupledCDM
from models.ensemble_cdm import DecoupledCDMEnsemble


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


class CognitiveReadoutHeadEnsembleTest(unittest.TestCase):
    def test_head_count_requires_positive_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "cognitive_readout_head_count"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                cognitive_readout_head_count=0,
            )

    def test_multiple_readout_heads_are_averaged_inside_single_model(self) -> None:
        model = DecoupledCDM(
            num_students=2,
            num_exercises=3,
            num_concepts=2,
            concept_dim=4,
            cognitive_readout_head_count=3,
        )
        self.assertEqual(len(model.cognitive_match_extra_mlps), 2)
        with torch.no_grad():
            heads = [model.cognitive_match_mlp, *model.cognitive_match_extra_mlps]
            for bias, head in zip((1.0, 3.0, 5.0), heads, strict=True):
                head[-1].weight.zero_()
                head[-1].bias.fill_(bias)

        logits = model._build_cognitive_readout_logits(match_inputs=torch.randn(4, 16))

        torch.testing.assert_close(logits, torch.full((4,), 3.0))


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

    def test_requires_valid_secondary_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "secondary_weight"):
            DecoupledCDMEnsemble(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                secondary_weight=1.1,
            )

    def test_blend_uses_secondary_weight(self) -> None:
        model = DecoupledCDMEnsemble(
            num_students=2,
            num_exercises=3,
            num_concepts=2,
            concept_dim=4,
            secondary_concept_dim=5,
            secondary_weight=0.25,
        )

        blended = model._blend(torch.tensor([0.2, 0.8]), torch.tensor([0.6, 0.4]))

        torch.testing.assert_close(blended, torch.tensor([0.3, 0.7]))

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


class EvidenceCalibratedBehaviorGateModelTest(unittest.TestCase):
    def test_evidence_gate_output_layer_starts_at_zero(self) -> None:
        model = DecoupledCDM(
            num_students=2,
            num_exercises=3,
            num_concepts=2,
            concept_dim=4,
            evidence_calibrated_behavior_gate=True,
        )

        layer = model.propagation.evidence_behavior_gate_residual
        self.assertIsNotNone(layer)
        torch.testing.assert_close(layer.weight, torch.zeros_like(layer.weight))
        torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias))

    def test_evidence_gate_rejects_invalid_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                evidence_behavior_gate_max_logit=0.0,
            )


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


class ConceptEvidenceCalibratedReadoutTest(unittest.TestCase):
    def test_calibrated_readout_output_layer_starts_at_zero(self) -> None:
        model = DecoupledCDM(
            num_students=2,
            num_exercises=3,
            num_concepts=2,
            concept_dim=4,
            concept_evidence_calibrated_readout=True,
        )

        self.assertIsNotNone(model.concept_evidence_calibrated_readout_head)
        final_layer = model.concept_evidence_calibrated_readout_head[-1]
        torch.testing.assert_close(final_layer.weight, torch.zeros_like(final_layer.weight))
        torch.testing.assert_close(final_layer.bias, torch.zeros_like(final_layer.bias))

    def test_calibrated_readout_rejects_invalid_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_count"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                concept_evidence_calibrated_readout_min_count=2,
                concept_evidence_calibrated_readout_max_count=1,
            )
        with self.assertRaisesRegex(ValueError, "min_seen_ratio"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                concept_evidence_calibrated_readout_min_seen_ratio=1.1,
            )
        with self.assertRaisesRegex(ValueError, "max_logit"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                concept_evidence_calibrated_readout_max_logit=0.0,
            )

    def test_calibrated_readout_is_bounded_and_triggered_by_seen_ratio(self) -> None:
        model = DecoupledCDM(
            num_students=2,
            num_exercises=3,
            num_concepts=3,
            concept_dim=4,
            concept_evidence_calibrated_readout=True,
            concept_evidence_calibrated_readout_min_count=2,
            concept_evidence_calibrated_readout_min_seen_ratio=1.0,
            concept_evidence_calibrated_readout_max_logit=0.25,
        )
        head = model.concept_evidence_calibrated_readout_head
        self.assertIsNotNone(head)
        with torch.no_grad():
            head[-1].weight.zero_()
            head[-1].bias.fill_(10.0)

        q_vectors = torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        student_concept_evidence = torch.zeros(2, 3, 6, dtype=torch.float32)
        student_concept_evidence[0, 0] = torch.tensor([3.0, 3.0, 0.0, 1.0, 1.3862944, 1.0])
        student_concept_evidence[0, 1] = torch.tensor([3.0, 2.0, 1.0, 2.0 / 3.0, 1.3862944, 1.0])
        student_concept_evidence[1, 0] = torch.tensor([3.0, 0.0, 3.0, 0.0, 1.3862944, 1.0])

        output = model._build_concept_evidence_calibrated_readout(
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

        self.assertGreater(float(output[0]), 0.24)
        self.assertLessEqual(float(output[0]), 0.25)
        torch.testing.assert_close(output[1:], torch.zeros(2, dtype=torch.float32))

class HistoryEvidenceFusionReadoutTest(unittest.TestCase):
    def test_fusion_readout_rejects_invalid_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "fusion_min_count"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                history_evidence_fusion_min_count=0,
            )
        with self.assertRaisesRegex(ValueError, "fusion_max_logit"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                history_evidence_fusion_max_logit=0.0,
            )
        with self.assertRaisesRegex(ValueError, "student_confidence_cap"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                history_evidence_fusion_student_confidence_cap=0.0,
            )
        with self.assertRaisesRegex(ValueError, "history_evidence_fusion_feature_set"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                history_evidence_fusion_feature_set="direct",
            )

    def test_fusion_readout_is_zero_init_bounded_and_triggered(self) -> None:
        model = DecoupledCDM(
            num_students=2,
            num_exercises=3,
            num_concepts=3,
            concept_dim=4,
            history_evidence_fusion_readout=True,
            history_evidence_fusion_min_count=2,
            history_evidence_fusion_min_seen_ratio=0.5,
            history_evidence_fusion_max_logit=0.25,
        )
        q_vectors = torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        student_concept_evidence = torch.zeros(2, 3, 6, dtype=torch.float32)
        student_concept_evidence[0, 0] = torch.tensor([3.0, 3.0, 0.0, 1.0, 1.3862944, 1.0])
        student_concept_evidence[0, 1] = torch.tensor([3.0, 2.0, 1.0, 2.0 / 3.0, 1.3862944, 1.0])
        student_concept_evidence[1, 0] = torch.tensor([3.0, 0.0, 3.0, 0.0, 1.3862944, 1.0])
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
        kwargs = {
            "q_vectors": q_vectors,
            "target_student_ids": torch.tensor([0, 1, 0], dtype=torch.long),
            "target_exercise_ids": torch.tensor([0, 1, 2], dtype=torch.long),
            "student_concept_evidence": student_concept_evidence,
            "exercise_evidence": exercise_evidence,
            "student_exercise_mask": student_exercise_mask,
            "response_matrix": response_matrix,
            "concept_summary": (
                torch.tensor([[2.0], [2.0], [1.0]], dtype=torch.float32),
                torch.zeros(3, 4, dtype=torch.float32),
                torch.zeros(3, 4, dtype=torch.float32),
            ),
            "difficulty": torch.zeros(3, dtype=torch.float32),
        }

        zero_output = model._build_history_evidence_fusion_readout(**kwargs)
        torch.testing.assert_close(zero_output, torch.zeros(3, dtype=torch.float32))

        with torch.no_grad():
            model.history_evidence_fusion_readout_head[-1].bias.fill_(3.0)
        output = model._build_history_evidence_fusion_readout(**kwargs)
        self.assertGreater(float(output[0]), 0.24)
        self.assertGreater(float(output[1]), 0.24)
        torch.testing.assert_close(output[2:], torch.zeros(1, dtype=torch.float32))
        self.assertLessEqual(float(output.abs().max()), 0.25)

    def test_fusion_cogonly_feature_set_masks_student_and_exercise_inputs(self) -> None:
        full_model = DecoupledCDM(
            num_students=2,
            num_exercises=3,
            num_concepts=3,
            concept_dim=4,
            history_evidence_fusion_readout=True,
            history_evidence_fusion_feature_set="full",
            history_evidence_fusion_min_count=2,
            history_evidence_fusion_min_seen_ratio=0.5,
            history_evidence_fusion_max_logit=1.0,
        )
        cogonly_model = DecoupledCDM(
            num_students=2,
            num_exercises=3,
            num_concepts=3,
            concept_dim=4,
            history_evidence_fusion_readout=True,
            history_evidence_fusion_feature_set="cogonly",
            history_evidence_fusion_min_count=2,
            history_evidence_fusion_min_seen_ratio=0.5,
            history_evidence_fusion_max_logit=1.0,
        )
        with torch.no_grad():
            for model in (full_model, cogonly_model):
                first_layer = model.history_evidence_fusion_readout_head[0]
                final_layer = model.history_evidence_fusion_readout_head[-1]
                first_layer.weight.zero_()
                first_layer.bias.zero_()
                first_layer.weight[0, 10:17] = 1.0
                final_layer.weight.zero_()
                final_layer.bias.zero_()
                final_layer.weight[0, 0] = 1.0

        q_vectors = torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        student_concept_evidence = torch.zeros(2, 3, 6, dtype=torch.float32)
        student_concept_evidence[0, 0] = torch.tensor([3.0, 3.0, 0.0, 1.0, 1.3862944, 1.0])
        student_concept_evidence[0, 1] = torch.tensor([3.0, 2.0, 1.0, 2.0 / 3.0, 1.3862944, 1.0])
        student_concept_evidence[1, 0] = torch.tensor([3.0, 0.0, 3.0, 0.0, 1.3862944, 1.0])
        exercise_evidence = torch.tensor(
            [
                [4.0, 4.0, 0.0, 1.0, 1.6094379, 1.0],
                [4.0, 0.0, 4.0, 0.0, 1.6094379, 1.0],
                [4.0, 2.0, 2.0, 0.5, 1.6094379, 1.0],
            ],
            dtype=torch.float32,
        )
        kwargs = {
            "q_vectors": q_vectors,
            "target_student_ids": torch.tensor([0, 1], dtype=torch.long),
            "target_exercise_ids": torch.tensor([0, 1], dtype=torch.long),
            "student_concept_evidence": student_concept_evidence,
            "exercise_evidence": exercise_evidence,
            "student_exercise_mask": torch.ones(2, 3, dtype=torch.float32),
            "response_matrix": torch.tensor(
                [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
                dtype=torch.float32,
            ),
            "concept_summary": (
                torch.tensor([[2.0], [2.0]], dtype=torch.float32),
                torch.zeros(2, 4, dtype=torch.float32),
                torch.zeros(2, 4, dtype=torch.float32),
            ),
            "difficulty": torch.zeros(2, dtype=torch.float32),
        }

        full_output = full_model._build_history_evidence_fusion_readout(**kwargs)
        cogonly_output = cogonly_model._build_history_evidence_fusion_readout(**kwargs)

        self.assertGreater(float(full_output[0]), 0.0)
        torch.testing.assert_close(cogonly_output, torch.zeros(2, dtype=torch.float32))


class HistoryEvidenceLinearReadoutTest(unittest.TestCase):
    def test_linear_readout_rejects_invalid_feature_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "history_evidence_linear_feature_set"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                history_evidence_linear_feature_set="direct",
            )

    def test_linear_readout_is_zero_init_bounded_and_signed(self) -> None:
        model = DecoupledCDM(
            num_students=2,
            num_exercises=3,
            num_concepts=3,
            concept_dim=4,
            history_evidence_linear_readout=True,
            history_evidence_linear_min_count=2,
            history_evidence_linear_min_seen_ratio=0.5,
            history_evidence_linear_max_logit=0.25,
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
        kwargs = {
            "q_vectors": q_vectors,
            "target_student_ids": torch.tensor([0, 1, 0], dtype=torch.long),
            "target_exercise_ids": torch.tensor([0, 1, 2], dtype=torch.long),
            "student_concept_evidence": student_concept_evidence,
            "exercise_evidence": exercise_evidence,
            "student_exercise_mask": student_exercise_mask,
            "response_matrix": response_matrix,
            "concept_summary": (
                torch.tensor([[2.0], [2.0], [1.0]], dtype=torch.float32),
                torch.zeros(3, 4, dtype=torch.float32),
                torch.zeros(3, 4, dtype=torch.float32),
            ),
        }

        zero_output = model._build_history_evidence_linear_readout(**kwargs)
        torch.testing.assert_close(zero_output, torch.zeros(3, dtype=torch.float32))

        with torch.no_grad():
            model.history_evidence_linear_weights.fill_(1.0)
        output = model._build_history_evidence_linear_readout(**kwargs)
        self.assertGreater(float(output[0]), 0.0)
        self.assertLess(float(output[1]), 0.0)
        torch.testing.assert_close(output[2:], torch.zeros(1, dtype=torch.float32))
        self.assertLessEqual(float(output.abs().max()), 0.25)

    def test_linear_cogonly_feature_set_masks_student_and_exercise_inputs(self) -> None:
        full_model = DecoupledCDM(
            num_students=2,
            num_exercises=2,
            num_concepts=1,
            concept_dim=4,
            history_evidence_linear_readout=True,
            history_evidence_linear_feature_set="full",
            history_evidence_linear_max_logit=1.0,
        )
        cogonly_model = DecoupledCDM(
            num_students=2,
            num_exercises=2,
            num_concepts=1,
            concept_dim=4,
            history_evidence_linear_readout=True,
            history_evidence_linear_feature_set="cogonly",
            history_evidence_linear_max_logit=1.0,
        )
        with torch.no_grad():
            for model in (full_model, cogonly_model):
                model.history_evidence_linear_weights[:] = torch.tensor([0.0, 1.0, 1.0])

        q_vectors = torch.ones(2, 1, dtype=torch.float32)
        student_concept_evidence = torch.tensor(
            [
                [[4.0, 2.0, 2.0, 0.5, 1.6094379, 1.0]],
                [[4.0, 2.0, 2.0, 0.5, 1.6094379, 1.0]],
            ],
            dtype=torch.float32,
        )
        exercise_evidence = torch.tensor(
            [[8.0, 8.0, 0.0, 1.0, 2.1972246, 1.0], [8.0, 8.0, 0.0, 1.0, 2.1972246, 1.0]],
            dtype=torch.float32,
        )
        kwargs = {
            "q_vectors": q_vectors,
            "target_student_ids": torch.tensor([0, 1], dtype=torch.long),
            "target_exercise_ids": torch.tensor([0, 1], dtype=torch.long),
            "student_concept_evidence": student_concept_evidence,
            "exercise_evidence": exercise_evidence,
            "student_exercise_mask": torch.ones(2, 2, dtype=torch.float32),
            "response_matrix": torch.ones(2, 2, dtype=torch.float32),
            "concept_summary": (
                torch.ones(2, 1, dtype=torch.float32),
                torch.zeros(2, 4, dtype=torch.float32),
                torch.zeros(2, 4, dtype=torch.float32),
            ),
        }

        full_output = full_model._build_history_evidence_linear_readout(**kwargs)
        cogonly_output = cogonly_model._build_history_evidence_linear_readout(**kwargs)

        self.assertGreater(float(full_output[0]), 0.0)
        torch.testing.assert_close(cogonly_output, torch.zeros(2, dtype=torch.float32))


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


class HistoryEvidenceOutputCalibrationTest(unittest.TestCase):
    def test_output_calibration_rejects_invalid_apply_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "output_calibration_apply_mode"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                history_evidence_output_calibration_apply_mode="unknown",
            )

    def test_output_calibration_head_starts_at_zero(self) -> None:
        model = DecoupledCDM(
            num_students=2,
            num_exercises=3,
            num_concepts=2,
            concept_dim=4,
            history_evidence_output_calibration=True,
        )

        self.assertIsNotNone(model.history_evidence_output_calibration_head)
        final_layer = model.history_evidence_output_calibration_head[-1]
        torch.testing.assert_close(final_layer.weight, torch.zeros_like(final_layer.weight))
        torch.testing.assert_close(final_layer.bias, torch.zeros_like(final_layer.bias))

    def test_output_calibration_apply_mode_respects_training_state(self) -> None:
        model = DecoupledCDM(
            num_students=2,
            num_exercises=3,
            num_concepts=2,
            concept_dim=4,
            history_evidence_output_calibration=True,
            history_evidence_output_calibration_apply_mode="train_only",
        )

        model.train()
        self.assertTrue(model._should_apply_history_evidence_output_calibration())
        model.eval()
        self.assertFalse(model._should_apply_history_evidence_output_calibration())

        model.history_evidence_output_calibration_apply_mode = "eval_only"
        self.assertTrue(model._should_apply_history_evidence_output_calibration())


class ExerciseEvidencePriorResidualTest(unittest.TestCase):
    def test_prior_rejects_invalid_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_count"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                exercise_evidence_prior_min_count=0,
            )
        with self.assertRaisesRegex(ValueError, "max_logit"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                exercise_evidence_prior_max_logit=0.0,
            )
        with self.assertRaisesRegex(ValueError, "strength"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                exercise_evidence_prior_strength=0.0,
            )
        with self.assertRaisesRegex(ValueError, "confidence_cap"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                exercise_evidence_prior_confidence_cap=0.0,
            )

    def test_prior_uses_smoothed_exercise_ease_with_count_trigger(self) -> None:
        model = DecoupledCDM(
            num_students=2,
            num_exercises=4,
            num_concepts=2,
            concept_dim=4,
            exercise_evidence_prior_residual=True,
            exercise_evidence_prior_min_count=3,
            exercise_evidence_prior_max_logit=0.25,
            exercise_evidence_prior_strength=2.0,
            exercise_evidence_prior_confidence_cap=20.0,
        )
        exercise_evidence = torch.tensor(
            [
                [4.0, 4.0, 0.0, 1.0, 1.6094379, 1.0],
                [4.0, 0.0, 4.0, 0.0, 1.6094379, 1.0],
                [2.0, 2.0, 0.0, 1.0, 1.0986123, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        output = model._build_exercise_evidence_prior_residual(
            target_exercise_ids=torch.tensor([0, 1, 2, 3], dtype=torch.long),
            exercise_evidence=exercise_evidence,
            q_vectors=torch.zeros(4, 2, dtype=torch.float32),
        )

        self.assertGreater(float(output[0]), 0.0)
        self.assertLess(float(output[1]), 0.0)
        torch.testing.assert_close(output[2:], torch.zeros(2, dtype=torch.float32))
        self.assertLessEqual(float(output.abs().max()), 0.25)


class StudentEvidenceAbilityPriorResidualTest(unittest.TestCase):
    def test_student_ability_prior_rejects_invalid_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_attempts"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                student_evidence_ability_prior_min_attempts=0,
            )
        with self.assertRaisesRegex(ValueError, "max_logit"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                student_evidence_ability_prior_max_logit=0.0,
            )

    def test_student_ability_prior_uses_global_history_with_count_trigger(self) -> None:
        model = DecoupledCDM(
            num_students=3,
            num_exercises=4,
            num_concepts=2,
            concept_dim=4,
            student_evidence_ability_prior_residual=True,
            student_evidence_ability_prior_min_attempts=3,
            student_evidence_ability_prior_max_logit=0.25,
            student_evidence_ability_prior_strength=2.0,
            student_evidence_ability_prior_confidence_cap=20.0,
        )
        student_exercise_mask = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        response_matrix = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        output = model._build_student_evidence_ability_prior_residual(
            target_student_ids=torch.tensor([0, 1, 2], dtype=torch.long),
            student_exercise_mask=student_exercise_mask,
            response_matrix=response_matrix,
            q_vectors=torch.zeros(3, 2, dtype=torch.float32),
        )

        self.assertGreater(float(output[0]), 0.0)
        self.assertLess(float(output[1]), 0.0)
        torch.testing.assert_close(output[2:], torch.zeros(1, dtype=torch.float32))
        self.assertLessEqual(float(output.abs().max()), 0.25)


class ExerciseEvidenceDifficultyAdapterTest(unittest.TestCase):
    def test_adapter_rejects_invalid_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "adapter_min_count"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                exercise_evidence_difficulty_adapter_min_count=0,
            )
        with self.assertRaisesRegex(ValueError, "adapter_max_logit"):
            DecoupledCDM(
                num_students=2,
                num_exercises=3,
                num_concepts=2,
                concept_dim=4,
                exercise_evidence_difficulty_adapter_max_logit=0.0,
            )

    def test_adapter_is_zero_init_and_learns_global_ease_slope(self) -> None:
        model = DecoupledCDM(
            num_students=2,
            num_exercises=4,
            num_concepts=2,
            concept_dim=4,
            exercise_evidence_difficulty_adapter=True,
            exercise_evidence_difficulty_adapter_min_count=3,
            exercise_evidence_difficulty_adapter_max_logit=0.25,
            exercise_evidence_difficulty_adapter_strength=2.0,
            exercise_evidence_difficulty_adapter_confidence_cap=20.0,
        )
        exercise_evidence = torch.tensor(
            [
                [4.0, 4.0, 0.0, 1.0, 1.6094379, 1.0],
                [4.0, 0.0, 4.0, 0.0, 1.6094379, 1.0],
                [2.0, 2.0, 0.0, 1.0, 1.0986123, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        target_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        q_vectors = torch.zeros(4, 2, dtype=torch.float32)

        zero_output = model._build_exercise_evidence_difficulty_adapter(
            target_exercise_ids=target_ids,
            exercise_evidence=exercise_evidence,
            q_vectors=q_vectors,
        )
        torch.testing.assert_close(zero_output, torch.zeros(4, dtype=torch.float32))

        with torch.no_grad():
            model.exercise_evidence_difficulty_adapter_scale.fill_(1.0)
        learned_output = model._build_exercise_evidence_difficulty_adapter(
            target_exercise_ids=target_ids,
            exercise_evidence=exercise_evidence,
            q_vectors=q_vectors,
        )
        self.assertGreater(float(learned_output[0]), 0.0)
        self.assertLess(float(learned_output[1]), 0.0)
        torch.testing.assert_close(learned_output[2:], torch.zeros(2, dtype=torch.float32))
        self.assertLessEqual(float(learned_output.abs().max()), 0.25)


class ExerciseEvidenceDifficultyInitTest(unittest.TestCase):
    def test_initializes_difficulty_with_inverse_exercise_ease(self) -> None:
        model = DecoupledCDM(num_students=2, num_exercises=4, num_concepts=2, concept_dim=4)
        with torch.no_grad():
            model.exercise_difficulty.weight.zero_()
        exercise_evidence = torch.tensor(
            [
                [4.0, 4.0, 0.0, 1.0, 1.6094379, 1.0],
                [4.0, 0.0, 4.0, 0.0, 1.6094379, 1.0],
                [2.0, 2.0, 0.0, 1.0, 1.0986123, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        initialized_count = model.initialize_exercise_difficulty_from_evidence(
            exercise_evidence=exercise_evidence,
            min_count=3,
            max_abs_logit=0.25,
            strength=2.0,
            confidence_cap=20.0,
        )

        weights = model.exercise_difficulty.weight[:, 0].detach()
        self.assertEqual(initialized_count, 2)
        self.assertLess(float(weights[0]), 0.0)
        self.assertGreater(float(weights[1]), 0.0)
        torch.testing.assert_close(weights[2:], torch.zeros(2, dtype=torch.float32))
        self.assertLessEqual(float(weights.abs().max()), 0.25)

    def test_difficulty_init_rejects_invalid_evidence_shape(self) -> None:
        model = DecoupledCDM(num_students=2, num_exercises=4, num_concepts=2, concept_dim=4)
        with self.assertRaisesRegex(ValueError, "one row per exercise"):
            model.initialize_exercise_difficulty_from_evidence(
                exercise_evidence=torch.zeros(3, 6, dtype=torch.float32),
            )


if __name__ == "__main__":
    unittest.main()
