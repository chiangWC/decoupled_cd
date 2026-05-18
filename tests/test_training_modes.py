import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
import torch.nn as nn

from data.datasets import StepDataBundle
from trainers.engine import _standardized_mse_alignment_loss, train_model


class _BiasOnlyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, *, target_student_ids: torch.Tensor, **kwargs):
        probs = torch.sigmoid(self.logit).expand(target_student_ids.size(0))
        return SimpleNamespace(probs=probs)


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

    def test_cognitive_alignment_weight_must_be_non_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "cognitive_alignment_weight"):
            train_model(
                train_bundle=_build_toy_bundle(),
                model=_BiasOnlyModel(),
                epochs=1,
                history_evidence_cognitive_alignment_weight=-0.1,
            )


class CognitiveAlignmentLossTest(unittest.TestCase):
    def test_standardized_alignment_loss_ignores_constant_target(self) -> None:
        prediction = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        target = torch.ones(3, dtype=torch.float32)

        loss = _standardized_mse_alignment_loss(prediction, target)

        torch.testing.assert_close(loss, torch.tensor(0.0))


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


if __name__ == "__main__":
    unittest.main()
