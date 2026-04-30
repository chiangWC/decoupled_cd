import unittest
from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn as nn

from data.datasets import StepDataBundle
from trainers.engine import train_model


class _BiasOnlyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, *, target_student_ids: torch.Tensor, **kwargs):
        probs = torch.sigmoid(self.logit).expand(target_student_ids.size(0))
        return SimpleNamespace(probs=probs)


def _build_toy_bundle(num_interactions: int = 5) -> StepDataBundle:
    interactions = pd.DataFrame(
        [
            {"stu_id": 1, "exer_id": 11 + index, "label": float(index % 2), "cpt_seq": "A"}
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


if __name__ == "__main__":
    unittest.main()
