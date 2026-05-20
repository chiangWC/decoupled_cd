import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
import torch.nn as nn

from data.datasets import StepDataBundle
from trainers.engine import (
    _checkpoint_distillation_loss,
    _dual_tower_branch_bce_loss,
    _history_evidence_alignment_reliability_weights,
    _history_evidence_output_alignment_loss,
    _is_checkpoint_metric_improved,
    _resolve_linear_epoch_weight,
    _standardized_mse_alignment_loss,
    _standardized_pairwise_rank_alignment_loss,
    train_model,
)


class _BiasOnlyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, *, target_student_ids: torch.Tensor, **kwargs):
        probs = torch.sigmoid(self.logit).expand(target_student_ids.size(0))
        return SimpleNamespace(probs=probs)


class _PriorToggleModel(_BiasOnlyModel):
    def __init__(self) -> None:
        super().__init__()
        self.concept_evidence_prior_residual = True
        self.concept_evidence_prior_max_logit = 0.9
        self.seen_prior_enabled: list[bool] = []
        self.seen_prior_max_logit: list[float] = []

    def forward(self, *, target_student_ids: torch.Tensor, **kwargs):
        self.seen_prior_enabled.append(bool(self.concept_evidence_prior_residual))
        self.seen_prior_max_logit.append(float(self.concept_evidence_prior_max_logit))
        return super().forward(target_student_ids=target_student_ids, **kwargs)


class _DualBranchOutput:
    def __init__(self) -> None:
        self.primary_probs = torch.tensor([0.2, 0.8], dtype=torch.float32)
        self.secondary_probs = torch.tensor([0.4, 0.6], dtype=torch.float32)


def _build_toy_bundle(num_interactions: int = 5, labels: list[float] | None = None) -> StepDataBundle:
    resolved_labels = labels or [float(index % 2) for index in range(num_interactions)]
    interactions = pd.DataFrame(
        [
            {"stu_id": 1, "exer_id": 11 + index, "label": resolved_labels[index], "cpt_seq": "A"}
            for index in range(num_interactions)
        ]
    )
    return StepDataBundle(
        interactions=interactions,
        history_interactions=interactions.copy(),
        split_name="train",
        allow_target_in_history=True,
        q_matrix=pd.DataFrame([{"exer_id": 11 + index, "cpt_seq": "A"} for index in range(num_interactions)]),
        student_id_map={"1": 0},
        exercise_id_map={str(11 + index): index for index in range(num_interactions)},
        concept_id_map={"A": 0},
        q_matrix_tensor=torch.ones(num_interactions, 1, dtype=torch.float32),
        concept_graph=torch.ones(1, 1, dtype=torch.float32),
        student_exercise_mask=torch.ones(1, num_interactions, dtype=torch.float32),
        student_tkc_mask=torch.ones(1, 1, dtype=torch.float32),
        student_ukc_mask=torch.zeros(1, 1, dtype=torch.float32),
        response_matrix_tensor=torch.tensor([[row["label"] for row in interactions.to_dict("records")]], dtype=torch.float32),
        interaction_student_ids=torch.zeros(num_interactions, dtype=torch.long),
        interaction_exercise_ids=torch.arange(num_interactions, dtype=torch.long),
        interaction_labels=torch.tensor(interactions["label"].tolist(), dtype=torch.float32),
    )


class TrainingModeValidationTest(unittest.TestCase):
    def test_full_batch_rejects_batch_size_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not consume --batch-size"):
            train_model(
                train_bundle=_build_toy_bundle(),
                model=_BiasOnlyModel(),
                epochs=1,
                batch_size=2,
                training_mode="full_batch",
            )

    def test_recompute_minibatch_requires_batch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --batch-size"):
            train_model(
                train_bundle=_build_toy_bundle(),
                model=_BiasOnlyModel(),
                epochs=1,
                training_mode="recompute_minibatch",
            )

    def test_weight_decay_must_be_non_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "weight_decay"):
            train_model(
                train_bundle=_build_toy_bundle(),
                model=_BiasOnlyModel(),
                epochs=1,
                weight_decay=-0.1,
            )

    def test_swa_start_epoch_must_be_non_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "swa_start_epoch"):
            train_model(
                train_bundle=_build_toy_bundle(),
                model=_BiasOnlyModel(),
                epochs=1,
                swa_start_epoch=-1,
            )

    def test_ema_start_epoch_must_be_non_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "ema_start_epoch"):
            train_model(
                train_bundle=_build_toy_bundle(),
                model=_BiasOnlyModel(),
                epochs=1,
                ema_start_epoch=-1,
            )

    def test_ema_decay_must_be_less_than_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "ema_decay"):
            train_model(
                train_bundle=_build_toy_bundle(),
                model=_BiasOnlyModel(),
                epochs=1,
                ema_start_epoch=1,
                ema_decay=1.0,
            )

    def test_swa_and_ema_cannot_both_be_enabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot both be enabled"):
            train_model(
                train_bundle=_build_toy_bundle(),
                model=_BiasOnlyModel(),
                epochs=1,
                swa_start_epoch=1,
                ema_start_epoch=1,
            )

    def test_checkpoint_selection_metric_must_be_supported(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint_selection_metric"):
            train_model(
                train_bundle=_build_toy_bundle(),
                model=_BiasOnlyModel(),
                epochs=1,
                checkpoint_selection_metric="precision",
            )

    def test_checkpoint_selection_start_epoch_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint_selection_start_epoch"):
            train_model(
                train_bundle=_build_toy_bundle(),
                model=_BiasOnlyModel(),
                epochs=1,
                checkpoint_selection_start_epoch=0,
            )

    def test_checkpoint_selection_window_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint_selection_window"):
            train_model(
                train_bundle=_build_toy_bundle(),
                model=_BiasOnlyModel(),
                epochs=1,
                checkpoint_selection_window=0,
            )

    def test_concept_evidence_prior_train_start_epoch_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "concept_evidence_prior_train_start_epoch"):
            train_model(
                train_bundle=_build_toy_bundle(),
                model=_BiasOnlyModel(),
                epochs=1,
                concept_evidence_prior_train_start_epoch=0,
            )

    def test_concept_evidence_prior_train_warmup_epochs_must_be_non_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "concept_evidence_prior_train_warmup_epochs"):
            train_model(
                train_bundle=_build_toy_bundle(),
                model=_BiasOnlyModel(),
                epochs=1,
                concept_evidence_prior_train_warmup_epochs=-1,
            )

    def test_cognitive_alignment_final_weight_must_be_non_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "cognitive_alignment_final_weight"):
            train_model(
                train_bundle=_build_toy_bundle(),
                model=_BiasOnlyModel(),
                epochs=1,
                history_evidence_cognitive_alignment_final_weight=-0.1,
            )

    def test_cognitive_alignment_anneal_window_must_be_valid(self) -> None:
        invalid_cases = [
            ("history_evidence_cognitive_alignment_anneal_start_epoch", 0, "anneal_start_epoch"),
            ("history_evidence_cognitive_alignment_anneal_end_epoch", 2, "anneal_end_epoch"),
        ]
        for field_name, value, message in invalid_cases:
            with self.subTest(field_name=field_name):
                kwargs = {
                    "history_evidence_cognitive_alignment_anneal_start_epoch": 3,
                    "history_evidence_cognitive_alignment_anneal_end_epoch": 4,
                    field_name: value,
                }
                with self.assertRaisesRegex(ValueError, message):
                    train_model(
                        train_bundle=_build_toy_bundle(),
                        model=_BiasOnlyModel(),
                        epochs=1,
                        **kwargs,
                    )

    def test_cognitive_alignment_weight_must_be_non_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "cognitive_alignment_weight"):
            train_model(
                train_bundle=_build_toy_bundle(),
                model=_BiasOnlyModel(),
                epochs=1,
                history_evidence_cognitive_alignment_weight=-0.1,
            )

    def test_cognitive_alignment_confidence_weighting_parameters_must_be_valid(self) -> None:
        invalid_cases = [
            ("history_evidence_cognitive_alignment_confidence_power", -0.1, "confidence_power"),
            ("history_evidence_cognitive_alignment_confidence_cap", 0.0, "confidence_cap"),
            ("history_evidence_cognitive_alignment_confidence_floor", 1.1, "confidence_floor"),
            ("history_evidence_cognitive_alignment_residual_power", -0.1, "residual_power"),
            ("history_evidence_cognitive_alignment_residual_floor", 1.1, "residual_floor"),
        ]
        for field_name, value, message in invalid_cases:
            with self.subTest(field_name=field_name):
                kwargs = {field_name: value}
                with self.assertRaisesRegex(ValueError, message):
                    train_model(
                        train_bundle=_build_toy_bundle(),
                        model=_BiasOnlyModel(),
                        epochs=1,
                        **kwargs,
                    )

    def test_cognitive_rank_alignment_config_must_be_valid(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank_alignment_weight"):
            train_model(
                train_bundle=_build_toy_bundle(),
                model=_BiasOnlyModel(),
                epochs=1,
                history_evidence_cognitive_rank_alignment_weight=-0.1,
            )
        with self.assertRaisesRegex(ValueError, "rank_alignment_pair_count"):
            train_model(
                train_bundle=_build_toy_bundle(),
                model=_BiasOnlyModel(),
                epochs=1,
                history_evidence_cognitive_rank_alignment_pair_count=0,
            )

    def test_output_alignment_weight_must_be_non_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "output_alignment_weight"):
            train_model(
                train_bundle=_build_toy_bundle(),
                model=_BiasOnlyModel(),
                epochs=1,
                history_evidence_output_alignment_weight=-0.1,
            )

    def test_output_alignment_confidence_weighting_parameters_must_be_valid(self) -> None:
        invalid_cases = [
            ("history_evidence_output_alignment_confidence_power", -0.1, "confidence_power"),
            ("history_evidence_output_alignment_confidence_cap", 0.0, "confidence_cap"),
            ("history_evidence_output_alignment_confidence_floor", 1.1, "confidence_floor"),
        ]
        for field_name, value, message in invalid_cases:
            with self.subTest(field_name=field_name):
                kwargs = {field_name: value}
                with self.assertRaisesRegex(ValueError, message):
                    train_model(
                        train_bundle=_build_toy_bundle(),
                        model=_BiasOnlyModel(),
                        epochs=1,
                        **kwargs,
                    )

    def test_checkpoint_distillation_config_must_be_valid(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint_distillation_weight"):
            train_model(
                train_bundle=_build_toy_bundle(),
                model=_BiasOnlyModel(),
                epochs=1,
                checkpoint_distillation_weight=-0.1,
            )
        with self.assertRaisesRegex(ValueError, "checkpoint_distillation_start_epoch"):
            train_model(
                train_bundle=_build_toy_bundle(),
                model=_BiasOnlyModel(),
                epochs=1,
                checkpoint_distillation_start_epoch=0,
            )
        with self.assertRaisesRegex(ValueError, "checkpoint_distillation_loss"):
            train_model(
                train_bundle=_build_toy_bundle(),
                model=_BiasOnlyModel(),
                epochs=1,
                checkpoint_distillation_loss="rank",
            )
        with self.assertRaisesRegex(ValueError, "checkpoint_distillation_targets"):
            train_model(
                train_bundle=_build_toy_bundle(num_interactions=3),
                model=_BiasOnlyModel(),
                epochs=1,
                checkpoint_distillation_weight=0.1,
            )
        with self.assertRaisesRegex(ValueError, "label shape"):
            train_model(
                train_bundle=_build_toy_bundle(num_interactions=3),
                model=_BiasOnlyModel(),
                epochs=1,
                checkpoint_distillation_targets=torch.ones(2),
            )

    def test_dual_tower_branch_bce_weight_must_be_non_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "dual_tower_branch_bce_weight"):
            train_model(
                train_bundle=_build_toy_bundle(),
                model=_BiasOnlyModel(),
                epochs=1,
                dual_tower_branch_bce_weight=-0.1,
            )


class CognitiveAlignmentLossTest(unittest.TestCase):
    def test_checkpoint_metric_improvement_uses_metric_direction(self) -> None:
        self.assertTrue(_is_checkpoint_metric_improved(metric="auc", value=0.8, best_value=0.7))
        self.assertFalse(_is_checkpoint_metric_improved(metric="auc", value=0.6, best_value=0.7))
        self.assertTrue(_is_checkpoint_metric_improved(metric="brier", value=0.17, best_value=0.18))
        self.assertFalse(_is_checkpoint_metric_improved(metric="brier", value=0.19, best_value=0.18))

    def test_linear_epoch_weight_resolves_schedule(self) -> None:
        self.assertEqual(
            _resolve_linear_epoch_weight(start_weight=0.1, final_weight=None, epoch=2, epochs=4),
            0.1,
        )
        self.assertAlmostEqual(
            _resolve_linear_epoch_weight(start_weight=0.1, final_weight=0.04, epoch=3, epochs=4),
            0.06,
            places=6,
        )
        self.assertEqual(
            _resolve_linear_epoch_weight(
                start_weight=0.1,
                final_weight=0.04,
                epoch=3,
                epochs=8,
                anneal_start_epoch=4,
                anneal_end_epoch=6,
            ),
            0.1,
        )
        self.assertAlmostEqual(
            _resolve_linear_epoch_weight(
                start_weight=0.1,
                final_weight=0.04,
                epoch=5,
                epochs=8,
                anneal_start_epoch=4,
                anneal_end_epoch=6,
            ),
            0.07,
            places=6,
        )
        self.assertEqual(
            _resolve_linear_epoch_weight(
                start_weight=0.1,
                final_weight=0.04,
                epoch=6,
                epochs=8,
                anneal_start_epoch=4,
                anneal_end_epoch=6,
            ),
            0.04,
        )

    def test_standardized_alignment_loss_ignores_constant_target(self) -> None:
        prediction = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        target = torch.ones(3, dtype=torch.float32)

        loss = _standardized_mse_alignment_loss(prediction, target)

        torch.testing.assert_close(loss, torch.tensor(0.0))

    def test_alignment_reliability_weights_use_target_concept_attempt_confidence(self) -> None:
        q_vectors = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        student_concept_evidence = torch.tensor([[[10.0, 8.0], [2.0, 1.0]]], dtype=torch.float32)

        weights = _history_evidence_alignment_reliability_weights(
            q_vectors=q_vectors,
            target_student_ids=torch.tensor([0, 0], dtype=torch.long),
            student_concept_evidence=student_concept_evidence,
            confidence_power=1.0,
            confidence_cap=10.0,
            confidence_floor=0.2,
        )

        torch.testing.assert_close(weights, torch.tensor([1.0, 0.68], dtype=torch.float32))

    def test_weighted_alignment_loss_ignores_zero_weight_targets(self) -> None:
        prediction = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        target = torch.tensor([3.0, 3.0, 1.0], dtype=torch.float32)

        loss = _standardized_mse_alignment_loss(
            prediction,
            target,
            sample_weights=torch.zeros(3, dtype=torch.float32),
        )

        torch.testing.assert_close(loss, torch.tensor(0.0))

    def test_weighted_alignment_loss_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_weights"):
            _standardized_mse_alignment_loss(
                torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
                torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32),
                sample_weights=torch.ones(2, dtype=torch.float32),
            )

    def test_residual_focused_alignment_loss_emphasizes_mismatches(self) -> None:
        prediction = torch.tensor([-1.0, -0.5, 0.5, 1.0], dtype=torch.float32)
        target = torch.tensor([-1.0, -0.5, 0.5, -1.0], dtype=torch.float32)

        unfocused_loss = _standardized_mse_alignment_loss(prediction, target)
        focused_loss = _standardized_mse_alignment_loss(
            prediction,
            target,
            residual_focus_power=1.0,
            residual_focus_floor=0.0,
        )

        self.assertGreater(float(focused_loss), float(unfocused_loss))

    def test_pairwise_rank_alignment_loss_prefers_matching_order(self) -> None:
        target = torch.linspace(-1.0, 1.0, steps=16)
        aligned_prediction = target.clone()
        reversed_prediction = -target

        torch.manual_seed(0)
        aligned_loss = _standardized_pairwise_rank_alignment_loss(
            prediction=aligned_prediction,
            target=target,
            pair_count=4096,
        )
        torch.manual_seed(0)
        reversed_loss = _standardized_pairwise_rank_alignment_loss(
            prediction=reversed_prediction,
            target=target,
            pair_count=4096,
        )

        self.assertLess(float(aligned_loss), float(reversed_loss))

    def test_output_alignment_loss_prefers_matching_output_logits(self) -> None:
        class _FixedPriorModel:
            history_evidence_logit_prior_residual = True
            history_evidence_logit_prior_location = "loss_only"

            def _build_history_evidence_logit_prior_residual(self, **kwargs):
                del kwargs
                return torch.linspace(-1.0, 1.0, steps=16)

        target_student_ids = torch.zeros(16, dtype=torch.long)
        target_exercise_ids = torch.arange(16, dtype=torch.long)
        tensors = {
            "q_matrix": torch.ones(16, 1, dtype=torch.float32),
            "student_concept_evidence": torch.ones(1, 1, 2, dtype=torch.float32),
            "exercise_evidence": torch.ones(16, 2, dtype=torch.float32),
            "student_exercise_mask": torch.ones(1, 16, dtype=torch.float32),
            "response_matrix": torch.ones(1, 16, dtype=torch.float32),
        }
        prior = torch.linspace(-1.0, 1.0, steps=16)
        aligned_output = SimpleNamespace(probs=torch.sigmoid(prior))
        reversed_output = SimpleNamespace(probs=torch.sigmoid(-prior))

        aligned_loss = _history_evidence_output_alignment_loss(
            model=_FixedPriorModel(),
            output=aligned_output,
            tensors=tensors,
            target_student_ids=target_student_ids,
            target_exercise_ids=target_exercise_ids,
            weight=1.0,
        )
        reversed_loss = _history_evidence_output_alignment_loss(
            model=_FixedPriorModel(),
            output=reversed_output,
            tensors=tensors,
            target_student_ids=target_student_ids,
            target_exercise_ids=target_exercise_ids,
            weight=1.0,
        )

        self.assertLess(float(aligned_loss), float(reversed_loss))

    def test_output_alignment_loss_uses_confidence_weights(self) -> None:
        class _FixedPriorModel:
            history_evidence_logit_prior_residual = True
            history_evidence_logit_prior_location = "loss_only"

            def _build_history_evidence_logit_prior_residual(self, **kwargs):
                del kwargs
                return torch.tensor([-1.0, 1.0, -0.5, 0.5], dtype=torch.float32)

        target_student_ids = torch.zeros(4, dtype=torch.long)
        target_exercise_ids = torch.arange(4, dtype=torch.long)
        q_matrix = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        student_concept_evidence = torch.tensor([[[10.0, 8.0], [2.0, 1.0]]], dtype=torch.float32)
        tensors = {
            "q_matrix": q_matrix,
            "student_concept_evidence": student_concept_evidence,
            "exercise_evidence": torch.ones(4, 2, dtype=torch.float32),
            "student_exercise_mask": torch.ones(1, 4, dtype=torch.float32),
            "response_matrix": torch.ones(1, 4, dtype=torch.float32),
        }
        output_logits = torch.tensor([-0.5, 0.5, -1.0, 1.0], dtype=torch.float32)

        loss = _history_evidence_output_alignment_loss(
            model=_FixedPriorModel(),
            output=SimpleNamespace(probs=torch.sigmoid(output_logits)),
            tensors=tensors,
            target_student_ids=target_student_ids,
            target_exercise_ids=target_exercise_ids,
            weight=0.7,
            confidence_power=1.0,
            confidence_cap=10.0,
            confidence_floor=0.2,
        )

        prior = torch.tensor([-1.0, 1.0, -0.5, 0.5], dtype=torch.float32)
        expected_weights = _history_evidence_alignment_reliability_weights(
            q_vectors=q_matrix,
            target_student_ids=target_student_ids,
            student_concept_evidence=student_concept_evidence,
            confidence_power=1.0,
            confidence_cap=10.0,
            confidence_floor=0.2,
        )
        expected = _standardized_mse_alignment_loss(
            output_logits,
            prior,
            sample_weights=expected_weights,
        ) * 0.7

        torch.testing.assert_close(loss, expected)

    def test_checkpoint_distillation_loss_uses_soft_teacher_probabilities(self) -> None:
        output = SimpleNamespace(probs=torch.tensor([0.25, 0.75], dtype=torch.float32))
        target_probs = torch.tensor([0.20, 0.80], dtype=torch.float32)

        loss = _checkpoint_distillation_loss(
            output=output,
            target_probs=target_probs,
            weight=0.5,
        )

        expected = torch.nn.functional.binary_cross_entropy(output.probs, target_probs) * 0.5
        torch.testing.assert_close(loss, expected)

    def test_checkpoint_distillation_loss_can_use_standardized_logit_shape(self) -> None:
        target_probs = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float32)
        aligned = SimpleNamespace(probs=target_probs.clone())
        reversed_output = SimpleNamespace(probs=torch.flip(target_probs, dims=(0,)))

        aligned_loss = _checkpoint_distillation_loss(
            output=aligned,
            target_probs=target_probs,
            weight=1.0,
            loss_type="standardized_logit_mse",
        )
        reversed_loss = _checkpoint_distillation_loss(
            output=reversed_output,
            target_probs=target_probs,
            weight=1.0,
            loss_type="standardized_logit_mse",
        )

        self.assertLess(float(aligned_loss), float(reversed_loss))

    def test_dual_tower_branch_bce_loss_averages_branch_losses(self) -> None:
        labels = torch.tensor([0.0, 1.0], dtype=torch.float32)
        output = _DualBranchOutput()

        loss = _dual_tower_branch_bce_loss(output=output, labels=labels, weight=0.2)

        expected = (
            torch.nn.functional.binary_cross_entropy(output.primary_probs, labels)
            + torch.nn.functional.binary_cross_entropy(output.secondary_probs, labels)
        ) * 0.1
        torch.testing.assert_close(loss, expected)


class RecomputeMinibatchTrainingTest(unittest.TestCase):
    def test_recompute_minibatch_uses_multiple_optimizer_steps(self) -> None:
        result = train_model(
            train_bundle=_build_toy_bundle(num_interactions=5),
            model=_BiasOnlyModel(),
            epochs=1,
            batch_size=2,
            training_mode="recompute_minibatch",
        )

        self.assertEqual(len(result.history), 1)
        self.assertEqual(result.history[0]["training_mode"], "recompute_minibatch")
        self.assertEqual(result.history[0]["optimizer_steps"], 3.0)

    def test_concept_evidence_prior_can_start_after_early_training_epochs(self) -> None:
        model = _PriorToggleModel()

        train_model(
            train_bundle=_build_toy_bundle(num_interactions=4, labels=[1.0, 1.0, 1.0, 1.0]),
            model=model,
            epochs=2,
            concept_evidence_prior_train_start_epoch=2,
        )

        self.assertEqual(model.seen_prior_enabled, [False, True])
        self.assertTrue(model.concept_evidence_prior_residual)

    def test_concept_evidence_prior_train_warmup_scales_max_logit_temporarily(self) -> None:
        model = _PriorToggleModel()

        result = train_model(
            train_bundle=_build_toy_bundle(num_interactions=4, labels=[1.0, 1.0, 1.0, 1.0]),
            model=model,
            epochs=4,
            concept_evidence_prior_train_start_epoch=2,
            concept_evidence_prior_train_warmup_epochs=3,
        )

        self.assertEqual(model.seen_prior_enabled, [False, True, True, True])
        for observed, expected in zip(model.seen_prior_max_logit, [0.9, 0.3, 0.6, 0.9]):
            self.assertAlmostEqual(observed, expected)
        for observed, expected in zip(
            [row["concept_evidence_prior_train_scale"] for row in result.history],
            [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        ):
            self.assertAlmostEqual(observed, expected)
        self.assertTrue(model.concept_evidence_prior_residual)
        self.assertEqual(model.concept_evidence_prior_max_logit, 0.9)

    def test_train_model_restores_best_validation_state(self) -> None:
        train_bundle = _build_toy_bundle(num_interactions=4, labels=[1.0, 1.0, 1.0, 1.0])
        valid_bundle = _build_toy_bundle(num_interactions=4)
        model = _BiasOnlyModel()
        observed_logits: list[float] = []
        val_aucs = iter([0.8, 0.7])

        def _fake_evaluate_model(*, bundle: StepDataBundle, model: _BiasOnlyModel, device: str):
            del bundle, device
            observed_logits.append(float(model.logit.detach().cpu().item()))
            return {
                "loss": 0.0,
                "auc": next(val_aucs),
                "acc": 0.0,
                "rmse": 0.0,
                "brier": 0.0,
                "ece": 0.0,
            }

        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "best.pt"
            with patch("trainers.engine.evaluate_model", side_effect=_fake_evaluate_model):
                result = train_model(
                    train_bundle=train_bundle,
                    valid_bundle=valid_bundle,
                    model=model,
                    epochs=2,
                    learning_rate=1.0,
                    checkpoint_path=str(checkpoint_path),
                )

            self.assertEqual(result.best_epoch, 1)
            self.assertEqual(result.best_checkpoint_path, str(checkpoint_path))
            self.assertEqual(len(observed_logits), 2)
            self.assertGreater(observed_logits[1], observed_logits[0])
            self.assertAlmostEqual(float(model.logit.detach().cpu().item()), observed_logits[0], places=6)

            checkpoint_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.assertAlmostEqual(float(checkpoint_state["logit"].item()), observed_logits[0], places=6)

    def test_checkpoint_selection_window_uses_recent_metric_average(self) -> None:
        train_bundle = _build_toy_bundle(num_interactions=4, labels=[1.0, 1.0, 1.0, 1.0])
        valid_bundle = _build_toy_bundle(num_interactions=4)
        model = _BiasOnlyModel()
        observed_logits: list[float] = []
        val_aucs = iter([0.7, 0.9, 0.85])

        def _fake_evaluate_model(*, bundle: StepDataBundle, model: _BiasOnlyModel, device: str):
            del bundle, device
            observed_logits.append(float(model.logit.detach().cpu().item()))
            return {
                "loss": 0.0,
                "auc": next(val_aucs),
                "acc": 0.0,
                "rmse": 0.0,
                "brier": 0.0,
                "ece": 0.0,
            }

        with patch("trainers.engine.evaluate_model", side_effect=_fake_evaluate_model):
            result = train_model(
                train_bundle=train_bundle,
                valid_bundle=valid_bundle,
                model=model,
                epochs=3,
                learning_rate=1.0,
                checkpoint_selection_window=2,
            )

        self.assertEqual(result.best_epoch, 3)
        self.assertAlmostEqual(result.best_validation_score, 0.875)
        self.assertAlmostEqual(result.history[0]["val_auc_selection_score"], 0.7)
        self.assertAlmostEqual(result.history[1]["val_auc_selection_score"], 0.8)
        self.assertAlmostEqual(result.history[2]["val_auc_selection_score"], 0.875)
        self.assertAlmostEqual(float(model.logit.detach().cpu().item()), observed_logits[2], places=6)

    def test_swa_loads_single_run_averaged_state(self) -> None:
        train_bundle = _build_toy_bundle(num_interactions=4, labels=[1.0, 1.0, 1.0, 1.0])
        valid_bundle = _build_toy_bundle(num_interactions=4)
        model = _BiasOnlyModel()
        observed_logits: list[float] = []
        val_aucs = iter([0.8, 0.7])

        def _fake_evaluate_model(*, bundle: StepDataBundle, model: _BiasOnlyModel, device: str):
            del bundle, device
            observed_logits.append(float(model.logit.detach().cpu().item()))
            return {
                "loss": 0.0,
                "auc": next(val_aucs),
                "acc": 0.0,
                "rmse": 0.0,
                "brier": 0.0,
                "ece": 0.0,
            }

        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "best.pt"
            swa_checkpoint_path = Path(tmpdir) / "swa.pt"
            with patch("trainers.engine.evaluate_model", side_effect=_fake_evaluate_model):
                result = train_model(
                    train_bundle=train_bundle,
                    valid_bundle=valid_bundle,
                    model=model,
                    epochs=2,
                    learning_rate=1.0,
                    checkpoint_path=str(checkpoint_path),
                    swa_start_epoch=1,
                    swa_checkpoint_path=str(swa_checkpoint_path),
                )

            expected_swa_logit = sum(observed_logits) / len(observed_logits)
            self.assertEqual(result.best_epoch, 1)
            self.assertEqual(result.swa_epoch_count, 2)
            self.assertEqual(result.swa_checkpoint_path, str(swa_checkpoint_path))
            self.assertAlmostEqual(float(model.logit.detach().cpu().item()), expected_swa_logit, places=6)

            checkpoint_state = torch.load(swa_checkpoint_path, map_location="cpu", weights_only=True)
            self.assertAlmostEqual(float(checkpoint_state["logit"].item()), expected_swa_logit, places=6)

    def test_ema_loads_single_run_averaged_state(self) -> None:
        train_bundle = _build_toy_bundle(num_interactions=4, labels=[1.0, 1.0, 1.0, 1.0])
        valid_bundle = _build_toy_bundle(num_interactions=4)
        model = _BiasOnlyModel()
        observed_logits: list[float] = []
        val_aucs = iter([0.8, 0.7])

        def _fake_evaluate_model(*, bundle: StepDataBundle, model: _BiasOnlyModel, device: str):
            del bundle, device
            observed_logits.append(float(model.logit.detach().cpu().item()))
            return {
                "loss": 0.0,
                "auc": next(val_aucs),
                "acc": 0.0,
                "rmse": 0.0,
                "brier": 0.0,
                "ece": 0.0,
            }

        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "best.pt"
            ema_checkpoint_path = Path(tmpdir) / "ema.pt"
            with patch("trainers.engine.evaluate_model", side_effect=_fake_evaluate_model):
                result = train_model(
                    train_bundle=train_bundle,
                    valid_bundle=valid_bundle,
                    model=model,
                    epochs=2,
                    learning_rate=1.0,
                    checkpoint_path=str(checkpoint_path),
                    ema_start_epoch=1,
                    ema_decay=0.5,
                    ema_checkpoint_path=str(ema_checkpoint_path),
                )

            expected_ema_logit = (0.5 * observed_logits[0]) + (0.5 * observed_logits[1])
            self.assertEqual(result.best_epoch, 1)
            self.assertEqual(result.ema_epoch_count, 2)
            self.assertEqual(result.ema_checkpoint_path, str(ema_checkpoint_path))
            self.assertAlmostEqual(float(model.logit.detach().cpu().item()), expected_ema_logit, places=6)

            checkpoint_state = torch.load(ema_checkpoint_path, map_location="cpu", weights_only=True)
            self.assertAlmostEqual(float(checkpoint_state["logit"].item()), expected_ema_logit, places=6)


if __name__ == "__main__":
    unittest.main()
