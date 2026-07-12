import inspect
import math
import unittest
from unittest import mock

import torch
import torch.nn as nn

from models.unified_decoupled_cdm import UnifiedDecoupledCDM
from models.unified_v2_components import (
    GlobalConceptPriorCompleter,
    ObservedMasteryEstimator,
    CoverageAwareStateComposer,
    MonotonicDiagnosisDecoder,
    TestedKnowledgeEvidenceEncoder,
    UntestedKnowledgeInferenceNetwork,
    assemble_mastery,
    smoothed_evidence_logits,
)
from models.unified_v2_spec import UnifiedArchitectureSpec


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def set_effective_weight(layer: nn.Module, value: float) -> None:
    with torch.no_grad():
        layer.raw_weight.fill_(inverse_softplus(value))


class UnifiedComponentTests(unittest.TestCase):
    def test_observed_estimator_uses_train_only_initial_logits(self):
        evidence = torch.tensor(
            [[[4.0, 3.0], [0.0, 0.0]], [[2.0, 0.0], [5.0, 4.0]]]
        )
        initial = smoothed_evidence_logits(evidence, alpha=1.0)
        module = ObservedMasteryEstimator(2, 2, initial_logits=initial)
        state = module(evidence)
        self.assertTrue(
            torch.equal(state.observed_mask, evidence[..., 0] > 0)
        )
        torch.testing.assert_close(state.mastery, initial.sigmoid())

    def test_zero_attempt_cells_are_marked_missing(self):
        evidence = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
        state = ObservedMasteryEstimator(1, 2)(evidence)

        self.assertTrue(
            torch.equal(state.observed_mask, torch.tensor([[False, True]]))
        )

    def test_observed_reliability_is_capped_attempt_fraction(self):
        evidence = torch.tensor(
            [[[0.0, 0.0], [5.0, 3.0], [20.0, 10.0], [30.0, 20.0]]]
        )
        state = ObservedMasteryEstimator(1, 4)(evidence)

        torch.testing.assert_close(
            state.reliability,
            torch.tensor([[0.0, 0.25, 1.0, 1.0]]),
        )

    def test_prior_completer_broadcasts_learnable_concept_vector(self):
        initial_prior = torch.tensor([0.2, 0.7, 0.9])
        module = GlobalConceptPriorCompleter(3, initial_prior=initial_prior)

        completed = module(num_students=4)

        self.assertIsInstance(module.logits, nn.Parameter)
        self.assertEqual(tuple(completed.shape), (4, 3))
        torch.testing.assert_close(completed[0], initial_prior)
        for student in completed[1:]:
            torch.testing.assert_close(student, completed[0])

    def test_hard_assembly_has_no_learned_blending(self):
        observed = torch.tensor([[0.2, 0.8]])
        missing = torch.tensor([[0.9, 0.1]])
        mask = torch.tensor([[True, False]])
        torch.testing.assert_close(
            assemble_mastery(observed, missing, mask),
            torch.tensor([[0.2, 0.1]]),
        )

    def test_observed_output_gradient_never_reaches_missing_completer(self):
        completer = GlobalConceptPriorCompleter(2)
        observed = torch.tensor([[0.2, 0.8]], requires_grad=True)
        mastery = assemble_mastery(
            observed,
            completer(num_students=1),
            torch.tensor([[True, False]]),
        )

        mastery[0, 0].backward()

        torch.testing.assert_close(
            completer.logits.grad,
            torch.zeros_like(completer.logits),
        )

    def test_m3_prefers_direct_state_at_full_reliability(self):
        composer = CoverageAwareStateComposer(dim=4)
        tkc = torch.tensor(
            [[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]]
        )
        ukc = torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]]]
        )
        prior = torch.zeros(2, 4)
        tkc_mask = torch.tensor([[1.0, 0.0]])
        state, weight = composer(
            tkc_states=tkc,
            ukc_states=ukc,
            concept_prior=prior,
            tkc_mask=tkc_mask,
            direct_reliability=torch.tensor([[1.0, 0.0]]),
            inferred_reliability=torch.tensor([[0.0, 1.0]]),
        )
        self.assertTrue(torch.allclose(state[:, 0], tkc[:, 0], atol=1e-5))

    def test_m3_uses_inference_for_zero_coverage_reachable_concept(self):
        composer = CoverageAwareStateComposer(dim=4)
        tkc = torch.zeros(1, 2, 4)
        ukc = torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]]]
        )
        state, weight = composer(
            tkc_states=tkc,
            ukc_states=ukc,
            concept_prior=torch.zeros(2, 4),
            tkc_mask=torch.tensor([[1.0, 0.0]]),
            direct_reliability=torch.zeros(1, 2),
            inferred_reliability=torch.tensor([[0.0, 1.0]]),
        )
        self.assertTrue(torch.allclose(state[:, 1], ukc[:, 1], atol=1e-5))

    def test_m3_does_not_depend_on_dataset_name(self):
        self.assertNotIn(
            "dataset",
            inspect.signature(CoverageAwareStateComposer.forward).parameters,
        )

    def test_m3_unreachable_ukc_has_only_prior_candidate(self):
        composer = CoverageAwareStateComposer(dim=2)
        prior = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        state, weight = composer(
            tkc_states=torch.zeros(1, 2, 2),
            ukc_states=prior.unsqueeze(0),
            concept_prior=prior,
            tkc_mask=torch.tensor([[1.0, 0.0]]),
            direct_reliability=torch.tensor([[0.4, 0.0]]),
            inferred_reliability=torch.zeros(1, 2),
        )

        self.assertEqual(float(weight[0, 1, 1]), 0.0)
        self.assertEqual(float(weight[0, 1, 2]), 1.0)
        self.assertTrue(torch.equal(state[0, 1], prior[1]))

    def test_m3_all_calibration_parameters_receive_nonzero_gradients(self):
        composer = CoverageAwareStateComposer(dim=2)
        state, _ = composer(
            tkc_states=torch.tensor(
                [[[1.0, 2.0], [2.0, 3.0], [0.0, 0.0], [0.0, 0.0]]]
            ),
            ukc_states=torch.tensor(
                [[[0.0, 0.0], [0.0, 0.0], [3.0, 4.0], [4.0, 5.0]]]
            ),
            concept_prior=torch.zeros(4, 2),
            tkc_mask=torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            direct_reliability=torch.tensor([[0.2, 0.3, 0.0, 0.0]]),
            inferred_reliability=torch.tensor([[0.0, 0.0, 0.25, 0.35]]),
        )

        state.sum().backward()

        for name, parameter in composer.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertTrue(torch.all(parameter.grad != 0), parameter.grad)

    def test_m3_optimizer_step_changes_partial_inference_weight(self):
        composer = CoverageAwareStateComposer(dim=2)
        optimizer = torch.optim.SGD(composer.parameters(), lr=0.1)
        inputs = {
            "tkc_states": torch.zeros(1, 1, 2),
            "ukc_states": torch.ones(1, 1, 2),
            "concept_prior": torch.zeros(1, 2),
            "tkc_mask": torch.zeros(1, 1),
            "direct_reliability": torch.zeros(1, 1),
            "inferred_reliability": torch.tensor([[0.25]]),
        }
        _, before = composer(**inputs)

        optimizer.zero_grad()
        (-before[..., 1].sum()).backward()
        optimizer.step()
        _, after = composer(**inputs)

        self.assertFalse(torch.equal(before[..., 1], after[..., 1]))

    def test_m3_calibration_has_no_jacobian_null_direction(self):
        composer = CoverageAwareStateComposer(dim=2)
        _, weights = composer(
            tkc_states=torch.zeros(1, 4, 2),
            ukc_states=torch.zeros(1, 4, 2),
            concept_prior=torch.zeros(4, 2),
            tkc_mask=torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            direct_reliability=torch.tensor([[0.2, 0.7, 0.0, 0.0]]),
            inferred_reliability=torch.tensor([[0.0, 0.0, 0.25, 0.65]]),
        )
        selected_weights = torch.stack(
            [
                weights[0, 0, 0],
                weights[0, 1, 0],
                weights[0, 2, 1],
                weights[0, 3, 1],
            ]
        )
        parameters = tuple(composer.parameters())
        jacobian_rows = []
        for weight in selected_weights:
            gradients = torch.autograd.grad(
                weight,
                parameters,
                retain_graph=True,
            )
            jacobian_rows.append(
                torch.cat([gradient.reshape(-1) for gradient in gradients])
            )
        jacobian = torch.stack(jacobian_rows)

        self.assertEqual(tuple(jacobian.shape), (4, 4))
        self.assertEqual(int(torch.linalg.matrix_rank(jacobian)), 4)

    def test_b0_always_emits_student_concept_mastery(self):
        model = UnifiedDecoupledCDM(
            num_students=3,
            num_exercises=2,
            num_concepts=3,
            dim=4,
            architecture=UnifiedArchitectureSpec(inference="prior", composer="mask"),
        )
        q = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
        history = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        tkc = (history @ q > 0).float()
        tensors = {
            "q_matrix": q,
            "concept_graph": torch.eye(3),
            "student_exercise_mask": history,
            "response_matrix": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
            "student_tkc_mask": tkc,
            "student_ukc_mask": 1.0 - tkc,
            "student_concept_evidence": None,
            "target_student_ids": torch.tensor([0, 1, 2]),
            "target_exercise_ids": torch.tensor([0, 1, 1]),
        }
        output = model(**tensors)
        self.assertEqual(tuple(output.mastery.shape), (3, 3))
        self.assertTrue(torch.isfinite(output.mastery).all())

    def test_graph_inference_wires_m2_into_state_map(self):
        model = UnifiedDecoupledCDM(
            num_students=2,
            num_exercises=1,
            num_concepts=2,
            dim=4,
            architecture=UnifiedArchitectureSpec(
                inference="graph",
                composer="mask",
            ),
        )
        with torch.no_grad():
            first_encoder = model.evidence_encoder.encoder[0]
            second_encoder = model.evidence_encoder.encoder[2]
            first_encoder.weight.zero_()
            first_encoder.bias.zero_()
            first_encoder.weight[0, 2] = 1.0
            second_encoder.weight.copy_(torch.eye(4))
            second_encoder.bias.zero_()
            for layer in model.inference_network.layers:
                layer.weight.copy_(torch.eye(4))

        output = model(
            q_matrix=torch.tensor([[1.0, 0.0]]),
            concept_graph=torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
            student_exercise_mask=torch.ones(2, 1),
            response_matrix=torch.tensor([[0.0], [1.0]]),
            student_tkc_mask=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            student_ukc_mask=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            student_concept_evidence=None,
            target_student_ids=torch.tensor([0, 1]),
            target_exercise_ids=torch.tensor([0, 0]),
        )
        self.assertFalse(
            torch.equal(output.ukc_states[0, 1], output.ukc_states[1, 1])
        )
        self.assertTrue(
            torch.equal(
                output.student_state,
                (output.tkc_states + output.ukc_states).mean(dim=1),
            )
        )
        self.assertIsNone(output.source_weights)

    def test_coverage_composer_wires_m3_and_exposes_source_weights(self):
        model = UnifiedDecoupledCDM(
            num_students=1,
            num_exercises=1,
            num_concepts=2,
            dim=4,
            architecture=UnifiedArchitectureSpec(
                inference="graph",
                composer="coverage",
            ),
        )
        output = model(
            q_matrix=torch.tensor([[1.0, 0.0]]),
            concept_graph=torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
            student_exercise_mask=torch.ones(1, 1),
            response_matrix=torch.ones(1, 1),
            student_tkc_mask=torch.tensor([[1.0, 0.0]]),
            student_ukc_mask=torch.tensor([[0.0, 1.0]]),
            student_concept_evidence=None,
            target_student_ids=torch.tensor([0]),
            target_exercise_ids=torch.tensor([0]),
        )

        self.assertIsNotNone(output.source_weights)
        self.assertEqual(tuple(output.source_weights.shape), (1, 2, 3))
        candidates = torch.stack(
            [
                output.tkc_states,
                output.ukc_states,
                model.inference_network.concept_prior.unsqueeze(0),
            ],
            dim=-2,
        )
        expected_state_map = (
            output.source_weights.unsqueeze(-1) * candidates
        ).sum(dim=-2)
        self.assertTrue(
            torch.allclose(
                output.student_state,
                expected_state_map.mean(dim=1),
            )
        )
        self.assertIsNotNone(output.mastery)

    def test_decoder_is_monotone_in_target_mastery(self):
        decoder = MonotonicDiagnosisDecoder(num_exercises=1, num_concepts=1, dim=4)
        low = decoder.decode_from_mastery(torch.tensor([[0.2]]), torch.tensor([[1.0]]), torch.tensor([0]))
        high = decoder.decode_from_mastery(torch.tensor([[0.8]]), torch.tensor([[1.0]]), torch.tensor([0]))
        self.assertGreaterEqual(float(high), float(low))

    def test_neuralcdm_decoder_is_monotone_at_random_and_boundary_mastery(self):
        torch.manual_seed(42)
        decoder = MonotonicDiagnosisDecoder(
            num_exercises=3, num_concepts=4, dim=4
        )
        self.assertEqual(len(decoder.interaction_layers), 3)
        q = torch.tensor(
            [[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]]
        )
        exercise_ids = torch.tensor([0, 1])
        ordered_pairs = [
            (torch.zeros(2, 4), torch.ones(2, 4)),
            (torch.rand(2, 4), torch.rand(2, 4)),
        ]
        for left, right in ordered_pairs:
            lower = torch.minimum(left, right)
            upper = torch.maximum(left, right)
            low_probs = decoder.decode_from_mastery(lower, q, exercise_ids)
            high_probs = decoder.decode_from_mastery(upper, q, exercise_ids)
            self.assertTrue(torch.all(high_probs >= low_probs - 1e-8))
            for row_tensor, concept_tensor in q.nonzero():
                row = int(row_tensor)
                concept = int(concept_tensor)
                coordinate_high = lower.clone()
                coordinate_high[row, concept] = upper[row, concept]
                changed = decoder.decode_from_mastery(
                    coordinate_high, q, exercise_ids
                )
                self.assertGreaterEqual(
                    float(changed[row]), float(low_probs[row]) - 1e-8
                )

    def test_neuralcdm_required_mastery_jacobian_is_nonnegative(self):
        torch.manual_seed(42)
        decoder = MonotonicDiagnosisDecoder(
            num_exercises=2, num_concepts=4, dim=4
        )
        self.assertEqual(len(decoder.interaction_layers), 3)
        q = torch.tensor(
            [[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]]
        )
        mastery = torch.tensor(
            [[0.2, 0.4, 0.6, 0.8], [0.8, 0.6, 0.4, 0.2]],
            requires_grad=True,
        )
        probabilities = decoder.decode_from_mastery(
            mastery, q, torch.tensor([0, 1])
        )
        for row in range(2):
            gradient = torch.autograd.grad(
                probabilities[row], mastery, retain_graph=True
            )[0]
            required = q[row].bool()
            self.assertTrue(torch.all(gradient[row, required] >= -1e-8))

    def test_neuralcdm_non_q_mastery_is_exactly_invariant(self):
        decoder = MonotonicDiagnosisDecoder(
            num_exercises=1, num_concepts=3, dim=4
        )
        self.assertEqual(len(decoder.interaction_layers), 3)
        q = torch.tensor([[1.0, 1.0, 0.0]])
        first = decoder.decode_from_mastery(
            torch.tensor([[0.3, 0.7, 0.0]]), q, torch.tensor([0])
        )
        second = decoder.decode_from_mastery(
            torch.tensor([[0.3, 0.7, 1.0]]), q, torch.tensor([0])
        )
        self.assertTrue(torch.equal(first, second))

    def test_neuralcdm_item_concept_difficulty_is_item_specific(self):
        decoder = MonotonicDiagnosisDecoder(
            num_exercises=2, num_concepts=2, dim=4
        )
        with torch.no_grad():
            decoder.item_concept_difficulty.weight[0].fill_(
                torch.logit(torch.tensor(0.2))
            )
            decoder.item_concept_difficulty.weight[1].fill_(
                torch.logit(torch.tensor(0.8))
            )
            decoder.exercise_discrimination.weight.fill_(inverse_softplus(1.0))
        probabilities = decoder.decode_from_mastery(
            torch.full((2, 2), 0.5),
            torch.ones(2, 2),
            torch.tensor([0, 1]),
        )
        self.assertNotEqual(float(probabilities[0]), float(probabilities[1]))

    def test_neuralcdm_expresses_nonlinear_multiconcept_logit_interaction(self):
        decoder = MonotonicDiagnosisDecoder(
            num_exercises=1, num_concepts=2, dim=4
        )
        with torch.no_grad():
            decoder.item_concept_difficulty.weight.fill_(-20.0)
            decoder.exercise_discrimination.weight.fill_(inverse_softplus(1.0))
            for layer in decoder.interaction_layers:
                layer.raw_weight.fill_(-20.0)
                layer.bias.fill_(-20.0)
            first, second, output = decoder.interaction_layers
            first.raw_weight[0, :2].fill_(inverse_softplus(4.0))
            first.bias[0] = -6.0
            second.raw_weight[0, 0] = inverse_softplus(4.0)
            second.bias[0] = -2.0
            output.raw_weight[0, 0] = inverse_softplus(4.0)
            output.bias[0] = -2.0
        corners = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        )
        probabilities = decoder.decode_from_mastery(
            corners, torch.ones(4, 2), torch.zeros(4, dtype=torch.long)
        )
        logits = torch.logit(probabilities)
        mixed = logits[3] - logits[1] - logits[2] + logits[0]
        self.assertGreater(abs(float(mixed)), 1e-2)

    def test_neuralcdm_initialization_is_reproducible_nonsaturated_and_trainable(self):
        torch.manual_seed(42)
        first = MonotonicDiagnosisDecoder(
            num_exercises=3, num_concepts=8, dim=4
        )
        torch.manual_seed(42)
        second = MonotonicDiagnosisDecoder(
            num_exercises=3, num_concepts=8, dim=4
        )
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]))
        self.assertGreater(
            float(first.interaction_layers[0].effective_weight.std()), 0.0
        )

        q = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            ]
        )
        mastery = torch.tensor(
            [
                [0.2, 0.8, 0.4, 0.6, 0.3, 0.7, 0.5, 0.9],
                [0.8, 0.2, 0.6, 0.4, 0.7, 0.3, 0.5, 0.1],
                [0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.9, 0.1],
            ],
            requires_grad=True,
        )
        probabilities = first.decode_from_mastery(
            mastery, q, torch.tensor([0, 1, 2])
        )
        self.assertTrue(torch.all((probabilities > 0.05) & (probabilities < 0.95)))
        probabilities.sum().backward()
        required = q.bool()
        self.assertTrue(torch.isfinite(mastery.grad).all())
        self.assertGreater(float(mastery.grad[required].abs().sum()), 0.0)
        beta_grad = first.item_concept_difficulty.weight.grad
        self.assertTrue(torch.isfinite(beta_grad).all())
        self.assertGreater(float(beta_grad[required].abs().sum()), 0.0)
        discrimination_grad = first.exercise_discrimination.weight.grad
        self.assertTrue(torch.isfinite(discrimination_grad).all())
        self.assertTrue(torch.all(discrimination_grad.abs().sum(dim=1) > 0))
        for layer in first.interaction_layers:
            self.assertTrue(torch.isfinite(layer.raw_weight.grad).all())
            self.assertTrue(torch.isfinite(layer.bias.grad).all())
            self.assertGreater(float(layer.raw_weight.grad.abs().sum()), 0.0)
            self.assertGreater(float(layer.bias.grad.abs().sum()), 0.0)

    def test_mastery_tensor_feeds_the_only_neuralcdm_prediction_path(self):
        model = UnifiedDecoupledCDM(
            num_students=2,
            num_exercises=2,
            num_concepts=2,
            dim=4,
            architecture=UnifiedArchitectureSpec(inference="prior", composer="mask"),
        )
        target_students = torch.tensor([0, 1])
        with mock.patch.object(
            model.decoder,
            "decode_from_mastery",
            wraps=model.decoder.decode_from_mastery,
        ) as decode:
            output = model(
                q_matrix=torch.eye(2),
                concept_graph=torch.eye(2),
                student_exercise_mask=torch.eye(2),
                response_matrix=torch.eye(2),
                student_tkc_mask=torch.eye(2),
                student_ukc_mask=1.0 - torch.eye(2),
                student_concept_evidence=None,
                target_student_ids=target_students,
                target_exercise_ids=torch.tensor([0, 1]),
            )
        captured_mastery = decode.call_args.args[0]
        self.assertEqual(tuple(output.mastery.shape), (2, 2))
        self.assertGreater(output.mastery.numel(), 0)
        self.assertTrue(torch.equal(captured_mastery, output.mastery[target_students]))
        self.assertEqual(output.probs.data_ptr(), output.cognitive_probs.data_ptr())
        names = {name for name, _ in model.decoder.named_parameters()}
        self.assertTrue(any(name.startswith("item_concept_difficulty") for name in names))
        self.assertNotIn("concept_difficulty.weight", names)
        self.assertNotIn("exercise_bias.weight", names)
        forbidden = ("guess", "slip", "legacy", "residual")
        source = inspect.getsource(MonotonicDiagnosisDecoder).lower()
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_old_linear_m4_checkpoint_is_rejected(self):
        decoder = MonotonicDiagnosisDecoder(
            num_exercises=2, num_concepts=3, dim=4
        )
        legacy = {
            "mastery_head.weight": torch.zeros(1, 4),
            "mastery_head.bias": torch.zeros(1),
            "concept_difficulty.weight": torch.zeros(3, 1),
            "exercise_discrimination.weight": torch.zeros(2, 1),
            "exercise_bias.weight": torch.zeros(2, 1),
        }
        with self.assertRaisesRegex(RuntimeError, "Missing key|Unexpected key"):
            decoder.load_state_dict(legacy)

    def test_m1_changes_with_student_responses(self):
        encoder = TestedKnowledgeEvidenceEncoder(dim=4, evidence_cap=20.0)
        with torch.no_grad():
            first = encoder.encoder[0]
            second = encoder.encoder[2]
            first.weight.zero_()
            first.weight[:, 0].fill_(1.0)
            first.weight[:, 1].fill_(-1.0)
            first.weight[:, 2].fill_(1.0)
            first.bias.fill_(1.0)
            second.weight.copy_(torch.eye(4))
            second.bias.zero_()
        q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        history = torch.ones(1, 2)
        common = {
            "q_matrix": q,
            "student_exercise_mask": history,
            "student_tkc_mask": torch.ones(1, 2),
        }
        left = encoder(response_matrix=torch.zeros(1, 2), **common)
        right = encoder(response_matrix=torch.ones(1, 2), **common)
        self.assertFalse(torch.equal(left.tkc_states, right.tkc_states))

    def test_m2_is_student_conditioned(self):
        module = UntestedKnowledgeInferenceNetwork(
            num_concepts=3,
            dim=4,
            layers=2,
        )
        graph = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
        )
        tkc_mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        ukc_mask = 1.0 - tkc_mask
        states = torch.zeros(2, 3, 4)
        states[0, 0] = 1.0
        states[1, 0] = -1.0
        out = module(
            tkc_states=states,
            tkc_mask=tkc_mask,
            ukc_mask=ukc_mask,
            concept_graph=graph,
            direct_reliability=tkc_mask,
        )
        self.assertFalse(torch.equal(out.ukc_states[0, 1], out.ukc_states[1, 1]))

    def test_m2_propagates_from_source_to_destination(self):
        module = UntestedKnowledgeInferenceNetwork(
            num_concepts=3,
            dim=2,
            layers=1,
        )
        with torch.no_grad():
            module.layers[0].weight.copy_(torch.eye(2))
            module.concept_prior.fill_(7.0)

        out = module(
            tkc_states=torch.tensor([[[1.0, -1.0], [0.0, 0.0], [0.0, 0.0]]]),
            tkc_mask=torch.tensor([[1.0, 0.0, 0.0]]),
            ukc_mask=torch.tensor([[0.0, 1.0, 1.0]]),
            concept_graph=torch.tensor(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
            direct_reliability=torch.tensor([[1.0, 0.0, 0.0]]),
        )

        self.assertTrue(
            torch.equal(
                out.reachable_mask,
                torch.tensor([[False, True, False]]),
            )
        )
        self.assertTrue(
            torch.allclose(
                out.ukc_states[0, 1],
                torch.tanh(torch.tensor([1.0, -1.0])),
            )
        )
        self.assertTrue(
            torch.equal(out.ukc_states[0, 2], module.concept_prior[2])
        )

    def test_m2_structural_reachability_ignores_zero_message_values(self):
        module = UntestedKnowledgeInferenceNetwork(
            num_concepts=3,
            dim=2,
            layers=2,
        )
        with torch.no_grad():
            for layer in module.layers:
                layer.weight.copy_(torch.eye(2))
            module.concept_prior.copy_(
                torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            )

        out = module(
            tkc_states=torch.zeros(2, 3, 2),
            tkc_mask=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            ukc_mask=torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
            concept_graph=torch.tensor(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
            ),
            direct_reliability=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
        )

        expected_reachability = torch.tensor(
            [[False, True, True], [False, False, False]]
        )
        self.assertTrue(torch.equal(out.reachable_mask, expected_reachability))
        self.assertTrue(
            torch.equal(
                out.inferred_reliability.gt(0),
                expected_reachability,
            )
        )
        self.assertTrue(
            torch.all(out.inferred_reliability[expected_reachability] < 1.0)
        )
        self.assertTrue(torch.equal(out.ukc_states[0, 1:], torch.zeros(2, 2)))
        self.assertTrue(
            torch.equal(
                out.ukc_states[1, 1:],
                module.concept_prior[1:],
            )
        )

    def test_m2_reliability_tracks_source_evidence_and_hop_decay(self):
        module = UntestedKnowledgeInferenceNetwork(
            num_concepts=4,
            dim=2,
            layers=2,
        )
        tkc_mask = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        )
        out = module(
            tkc_states=torch.zeros(2, 4, 2),
            tkc_mask=tkc_mask,
            ukc_mask=1.0 - tkc_mask,
            concept_graph=torch.tensor(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            direct_reliability=torch.tensor(
                [[0.25, 0.0, 0.0, 0.0], [0.75, 0.0, 0.0, 0.0]]
            ),
        )

        self.assertTrue(
            torch.equal(
                out.reachable_mask,
                torch.tensor(
                    [
                        [False, True, True, False],
                        [False, True, True, False],
                    ]
                ),
            )
        )
        reachable_reliability = out.inferred_reliability[:, 1:3]
        self.assertTrue(torch.all(reachable_reliability > 0.0))
        self.assertTrue(torch.all(reachable_reliability < 1.0))
        self.assertTrue(
            torch.all(reachable_reliability[1] > reachable_reliability[0])
        )
        self.assertTrue(
            torch.all(
                reachable_reliability[:, 0] > reachable_reliability[:, 1]
            )
        )
        self.assertTrue(
            torch.equal(
                out.inferred_reliability[:, 3],
                torch.zeros(2),
            )
        )

    def test_m2_unreachable_ukc_uses_prior_not_static_broadcast(self):
        module = UntestedKnowledgeInferenceNetwork(
            num_concepts=3,
            dim=4,
            layers=1,
        )
        graph = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        tkc_mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        ukc_mask = 1.0 - tkc_mask
        out = module(
            tkc_states=torch.randn(2, 3, 4),
            tkc_mask=tkc_mask,
            ukc_mask=ukc_mask,
            concept_graph=graph,
            direct_reliability=tkc_mask,
        )
        self.assertTrue(
            torch.equal(
                out.ukc_states[:, 2],
                module.concept_prior[2].expand(2, -1),
            )
        )


if __name__ == "__main__":
    unittest.main()
