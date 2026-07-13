from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch

from models import R28CompletionCDM


def tensors() -> dict[str, torch.Tensor]:
    students, exercises, concepts = 5, 7, 4
    q_matrix = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=torch.float32,
    )
    evidence = torch.zeros(students, concepts, 6)
    evidence[0, 0, :2] = torch.tensor([2.0, 1.0])
    evidence[0, 1, :2] = torch.tensor([1.0, 1.0])
    evidence[1, 2, :2] = torch.tensor([3.0, 1.0])
    evidence[2, 3, :2] = torch.tensor([1.0, 0.0])
    evidence[3, 0, :2] = torch.tensor([2.0, 2.0])
    evidence[4, 1, :2] = torch.tensor([3.0, 2.0])
    attempts = evidence[..., 0]
    student_exercise_mask = torch.zeros(students, exercises)
    response_matrix = torch.zeros(students, exercises)
    observed_pairs = [(0, 0, 1.0), (0, 4, 0.0), (1, 2, 0.0), (2, 3, 0.0), (3, 0, 1.0), (4, 5, 1.0)]
    for student_id, exercise_id, label in observed_pairs:
        student_exercise_mask[student_id, exercise_id] = 1.0
        response_matrix[student_id, exercise_id] = label
    item_attempts = student_exercise_mask.sum(dim=0)
    item_correct = (student_exercise_mask * response_matrix).sum(dim=0)
    exercise_evidence = torch.stack(
        [
            item_attempts,
            item_correct,
            item_attempts - item_correct,
            item_correct / item_attempts.clamp_min(1.0),
            torch.log1p(item_attempts),
            (item_attempts > 0).float(),
        ],
        dim=-1,
    )
    return {
        "q_matrix": q_matrix,
        "concept_graph": torch.zeros(concepts, concepts),
        "student_exercise_mask": student_exercise_mask,
        "response_matrix": response_matrix,
        "student_tkc_mask": (attempts > 0).float(),
        "student_ukc_mask": (attempts == 0).float(),
        "student_concept_evidence": evidence,
        "exercise_evidence": exercise_evidence,
        "target_student_ids": torch.tensor([0, 1, 0, 4]),
        "target_exercise_ids": torch.tensor([0, 2, 4, 6]),
        "use_student_subset": True,
    }


def build(mode: str, objective: str = "none") -> R28CompletionCDM:
    torch.manual_seed(42)
    return R28CompletionCDM(
        num_students=5,
        num_exercises=7,
        num_concepts=4,
        concept_dim=8,
        state_completer=mode,
        completion_objective=objective,
    )


class R28CompletionTests(unittest.TestCase):
    def test_output_contract_and_gradients(self) -> None:
        for mode in (
            "relational",
            "direct_prior",
            "capacity_mlp",
            "partial_vae",
            "difficulty_set",
            "difficulty_capacity",
            "poe_ability",
            "poe_capacity",
            "hierarchical_bayes",
            "hierarchical_capacity",
            "cohort_conditioned",
            "cohort_capacity",
            "bipolar_prototype",
            "bipolar_capacity",
        ):
            with self.subTest(mode=mode):
                model = build(mode)
                output = model(**tensors())
                self.assertEqual(output.probs.shape, (4,))
                self.assertEqual(output.framework_state.shape, (3, 4, 8))
                self.assertEqual(output.mastery.shape, (3, 4))
                self.assertEqual(output.state_reliability.shape, (3, 4))
                self.assertEqual(output.architecture_fingerprint, model.architecture_fingerprint)
                output.framework_state.retain_grad()
                output.probs.sum().backward()
                self.assertIsNotNone(output.framework_state.grad)
                self.assertGreater(torch.count_nonzero(output.framework_state.grad).item(), 0)

    def test_concept_graph_is_not_a_required_path(self) -> None:
        model = build("relational")
        inputs = tensors()
        zero = model(**inputs).probs
        inputs["concept_graph"] = torch.rand_like(inputs["concept_graph"])
        nonzero = model(**inputs).probs
        torch.testing.assert_close(zero, nonzero)

    def test_common_initialization_and_capacity_are_clean(self) -> None:
        models = {mode: build(mode) for mode in ("relational", "direct_prior", "capacity_mlp")}
        self.assertEqual(len({model.common_initialization_hash() for model in models.values()}), 1)
        full = models["relational"].active_parameter_count()
        capacity = models["capacity_mlp"].active_parameter_count()
        self.assertLessEqual(abs(full - capacity) / full, 0.10)
        self.assertEqual(full, capacity)

    def test_no_free_student_id_parameter_or_prediction_bypass(self) -> None:
        model = build("relational")
        names = [name for name, _ in model.named_parameters()]
        self.assertFalse(any("student_embedding" in name or "student_id" in name for name in names))
        inputs = tensors()
        inputs["target_student_ids"] = torch.tensor([0, 1])
        inputs["target_exercise_ids"] = torch.tensor([0, 0])
        output = model(**inputs)
        self.assertFalse(torch.equal(output.framework_state[0], output.framework_state[1]))

    def test_direct_prior_unobserved_state_is_static_concept_prior(self) -> None:
        output = build("direct_prior")(**tensors())
        torch.testing.assert_close(output.framework_state[0, 3], output.framework_state[1, 3])
        torch.testing.assert_close(output.framework_state[1, 3], output.framework_state[2, 3])

    def test_masked_reconstruction_is_scalar_and_training_only(self) -> None:
        model = build("relational", objective="masked_reconstruction")
        model.completion_mask_frac = 0.99
        model.train()
        torch.manual_seed(1)
        training = model(**tensors())
        self.assertIsNotNone(training.completion_reconstruction_loss)
        self.assertEqual(training.completion_reconstruction_loss.ndim, 0)
        model.eval()
        evaluation = model(**tensors())
        self.assertIsNone(evaluation.completion_reconstruction_loss)

    def test_partial_vae_is_stochastic_only_during_training(self) -> None:
        model = build("partial_vae", objective="masked_reconstruction")
        model.completion_mask_frac = 0.99
        model.train()
        torch.manual_seed(7)
        first = model(**tensors())
        torch.manual_seed(8)
        second = model(**tensors())
        self.assertFalse(torch.equal(first.framework_state, second.framework_state))
        self.assertIsNotNone(first.completion_reconstruction_loss)
        self.assertIn("posterior_kl", first.module_diagnostics)
        model.eval()
        deterministic_a = model(**tensors()).framework_state
        deterministic_b = model(**tensors()).framework_state
        torch.testing.assert_close(deterministic_a, deterministic_b)

    def test_partial_vae_capacity_matches_control_at_campaign_width(self) -> None:
        torch.manual_seed(42)
        full = R28CompletionCDM(
            num_students=5,
            num_exercises=7,
            num_concepts=4,
            concept_dim=64,
            state_completer="partial_vae",
        )
        torch.manual_seed(42)
        control = R28CompletionCDM(
            num_students=5,
            num_exercises=7,
            num_concepts=4,
            concept_dim=64,
            state_completer="capacity_mlp",
        )
        difference = abs(full.active_parameter_count() - control.active_parameter_count())
        self.assertLessEqual(difference / full.active_parameter_count(), 0.10)

    def test_difficulty_set_has_exact_capacity_control(self) -> None:
        full = build("difficulty_set")
        control = build("difficulty_capacity")
        self.assertEqual(full.active_parameter_count(), control.active_parameter_count())
        self.assertEqual(full.common_initialization_hash(), control.common_initialization_hash())
        full_state = full(**tensors()).framework_state
        control_state = control(**tensors()).framework_state
        self.assertFalse(torch.equal(full_state, control_state))

    def test_difficulty_set_consumes_train_only_item_calibration(self) -> None:
        model = build("difficulty_set")
        model.eval()
        inputs = tensors()
        baseline = model(**inputs).framework_state
        changed_inputs = tensors()
        changed_inputs["exercise_evidence"][:, 3] = 1.0 - changed_inputs["exercise_evidence"][:, 3]
        changed = model(**changed_inputs).framework_state
        self.assertFalse(torch.equal(baseline, changed))

    def test_product_of_experts_has_exact_capacity_control(self) -> None:
        full = build("poe_ability")
        control = build("poe_capacity")
        self.assertEqual(full.active_parameter_count(), control.active_parameter_count())
        self.assertEqual(full.common_initialization_hash(), control.common_initialization_hash())
        full.eval()
        control.eval()
        self.assertFalse(
            torch.equal(
                full(**tensors()).framework_state,
                control(**tensors()).framework_state,
            )
        )

    def test_product_of_experts_is_stochastic_only_during_training(self) -> None:
        model = build("poe_ability")
        model.train()
        torch.manual_seed(11)
        first = model(**tensors()).framework_state
        torch.manual_seed(12)
        second = model(**tensors()).framework_state
        self.assertFalse(torch.equal(first, second))
        model.eval()
        deterministic_a = model(**tensors()).framework_state
        deterministic_b = model(**tensors()).framework_state
        torch.testing.assert_close(deterministic_a, deterministic_b)

    def test_hierarchical_bayes_has_exact_capacity_control(self) -> None:
        full = build("hierarchical_bayes")
        control = build("hierarchical_capacity")
        anchor = build("difficulty_capacity")
        self.assertEqual(full.active_parameter_count(), control.active_parameter_count())
        self.assertEqual(full.active_parameter_count(), anchor.active_parameter_count())
        self.assertEqual(full.common_initialization_hash(), control.common_initialization_hash())
        full.eval()
        control.eval()
        anchor.eval()
        self.assertFalse(
            torch.equal(
                full(**tensors()).framework_state,
                control(**tensors()).framework_state,
            )
        )
        torch.testing.assert_close(
            control(**tensors()).framework_state,
            anchor(**tensors()).framework_state,
        )

    def test_hierarchical_bayes_consumes_local_counts_for_missing_state(self) -> None:
        model = build("hierarchical_bayes")
        model.eval()
        inputs = tensors()
        baseline = model(**inputs).framework_state
        changed_inputs = tensors()
        changed_inputs["student_concept_evidence"][0, 0, 1] = 2.0
        changed = model(**changed_inputs).framework_state
        self.assertFalse(torch.equal(baseline[0], changed[0]))

    def test_cohort_completion_has_exact_capacity_control(self) -> None:
        full = build("cohort_conditioned")
        control = build("cohort_capacity")
        anchor = build("difficulty_capacity")
        self.assertEqual(full.active_parameter_count(), control.active_parameter_count())
        self.assertEqual(full.common_initialization_hash(), control.common_initialization_hash())
        self.assertLessEqual(
            abs(full.active_parameter_count() - anchor.active_parameter_count())
            / full.active_parameter_count(),
            0.10,
        )
        full.eval()
        control.eval()
        self.assertFalse(
            torch.equal(
                full(**tensors()).framework_state,
                control(**tensors()).framework_state,
            )
        )

    def test_cohort_completion_borrows_peer_concept_evidence(self) -> None:
        model = build("cohort_conditioned")
        model.eval()
        inputs = tensors()
        baseline = model(**inputs).framework_state
        changed_inputs = tensors()
        # Change a peer's train-only evidence while preserving the target
        # student's own response history.
        changed_inputs["student_concept_evidence"][3, 0, 1] = 0.0
        changed = model(**changed_inputs).framework_state
        self.assertFalse(torch.equal(baseline[0], changed[0]))

    def test_bipolar_prototype_has_exact_capacity_control(self) -> None:
        full = build("bipolar_prototype")
        control = build("bipolar_capacity")
        anchor = build("difficulty_capacity")
        self.assertEqual(full.active_parameter_count(), control.active_parameter_count())
        self.assertEqual(full.common_initialization_hash(), control.common_initialization_hash())
        self.assertLessEqual(
            abs(full.active_parameter_count() - anchor.active_parameter_count())
            / full.active_parameter_count(),
            0.10,
        )
        full.eval()
        control.eval()
        self.assertFalse(
            torch.equal(
                full(**tensors()).framework_state,
                control(**tensors()).framework_state,
            )
        )

    def test_bipolar_prototype_uses_response_polarity(self) -> None:
        model = build("bipolar_prototype")
        model.eval()
        inputs = tensors()
        baseline = model(**inputs).framework_state
        changed_inputs = tensors()
        changed_inputs["response_matrix"][0, 0] = 0.0
        changed = model(**changed_inputs).framework_state
        self.assertFalse(torch.equal(baseline[0], changed[0]))

    def test_missing_evidence_and_gated_followups_fail_explicitly(self) -> None:
        model = build("relational")
        inputs = tensors()
        inputs["student_concept_evidence"] = None
        with self.assertRaisesRegex(ValueError, "train-history"):
            model(**inputs)
        with self.assertRaisesRegex(NotImplementedError, "not activated"):
            R28CompletionCDM(
                num_students=5,
                num_exercises=7,
                num_concepts=4,
                state_completer="exposure_dr",
            )

    def test_registry_freezes_dynamic_pool_and_rejected_directions(self) -> None:
        root = Path(__file__).resolve().parents[1]
        registry = json.loads((root / "configs/r28_registry.json").read_text())
        modules = json.loads((root / "configs/r28_module_registry.json").read_text())
        self.assertEqual(registry["seed"], 42)
        self.assertEqual(
            registry["active_pool"],
            ["assist_09", "assist_17", "moocradar", "nips34", "xes3g5m"],
        )
        self.assertEqual(modules["active_framework_boxes"], 2)
        self.assertEqual(modules["candidate_module_slots"], 1)
        self.assertEqual(modules["qualified_paper_module_count"], 0)
        self.assertIn("orcdf_response_graph", modules["do_not_repeat"])
        self.assertIn("set_transformer_query_completion", modules["do_not_repeat"])


if __name__ == "__main__":
    unittest.main()
