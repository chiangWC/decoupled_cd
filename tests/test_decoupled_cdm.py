import unittest

import torch

from models.decoupled_cdm import DecoupledCDM


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

    def test_prior_uses_smoothed_target_concept_accuracy(self) -> None:
        model = DecoupledCDM(
            num_students=2,
            num_exercises=3,
            num_concepts=3,
            concept_dim=4,
            concept_evidence_prior_residual=True,
            concept_evidence_prior_min_count=2,
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
            ],
            dtype=torch.float32,
        )
        student_concept_evidence = torch.zeros(2, 3, 6, dtype=torch.float32)
        student_concept_evidence[0, 0] = torch.tensor([3.0, 3.0, 0.0, 1.0, 1.3862944, 1.0])
        student_concept_evidence[0, 1] = torch.tensor([3.0, 3.0, 0.0, 1.0, 1.3862944, 1.0])
        student_concept_evidence[1, 0] = torch.tensor([3.0, 0.0, 3.0, 0.0, 1.3862944, 1.0])

        output = model._build_concept_evidence_prior_residual(
            q_vectors=q_vectors,
            target_student_ids=torch.tensor([0, 1, 0], dtype=torch.long),
            student_concept_evidence=student_concept_evidence,
            concept_summary=(
                torch.tensor([[2.0], [2.0], [1.0]], dtype=torch.float32),
                torch.zeros(3, 4, dtype=torch.float32),
                torch.zeros(3, 4, dtype=torch.float32),
            ),
        )

        self.assertGreater(float(output[0]), 0.0)
        torch.testing.assert_close(output[1:], torch.zeros(2, dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()
