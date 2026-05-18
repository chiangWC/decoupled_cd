import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn as nn

from data import (
    build_exercise_evidence_tensor,
    build_student_concept_evidence_tensor,
    prepare_experiment_split_bundles,
    prepare_step_data_bundle,
)
from data.datasets import StepDataBundle
from trainers.engine import evaluate_model


class _NeverCalledModel(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("history-visibility guard should fail before model.forward()")


class _ConstantProbModel(nn.Module):
    def forward(self, *, target_student_ids: torch.Tensor, **kwargs):
        batch_size = target_student_ids.size(0)
        probs = torch.full((batch_size,), 0.5, dtype=torch.float32)
        return SimpleNamespace(probs=probs)


class HistoryVisibilityBundleTest(unittest.TestCase):
    def test_student_concept_evidence_tensor_uses_history_rows_only(self) -> None:
        interactions = pd.DataFrame(
            [
                {"stu_id": 1, "exer_id": 11, "cpt_seq": "A,B", "label": 1},
                {"stu_id": 1, "exer_id": 12, "cpt_seq": "A", "label": 0},
                {"stu_id": 2, "exer_id": 13, "cpt_seq": "B", "label": 1},
            ]
        )

        evidence = build_student_concept_evidence_tensor(
            interactions=interactions,
            student_id_map={"1": 0, "2": 1},
            concept_id_map={"A": 0, "B": 1},
        )

        self.assertEqual(tuple(evidence.shape), (2, 2, 6))
        torch.testing.assert_close(
            evidence[0, 0],
            torch.tensor([2.0, 1.0, 1.0, 0.5, math.log1p(2.0), 1.0], dtype=torch.float32),
        )
        torch.testing.assert_close(
            evidence[0, 1],
            torch.tensor([1.0, 1.0, 0.0, 1.0, math.log1p(1.0), 1.0], dtype=torch.float32),
        )
        torch.testing.assert_close(evidence[1, 0], torch.zeros(6, dtype=torch.float32))

    def test_exercise_evidence_tensor_uses_history_rows_only(self) -> None:
        interactions = pd.DataFrame(
            [
                {"stu_id": 1, "exer_id": 11, "cpt_seq": "A", "label": 1},
                {"stu_id": 2, "exer_id": 11, "cpt_seq": "A", "label": 0},
                {"stu_id": 1, "exer_id": 12, "cpt_seq": "B", "label": 1},
            ]
        )

        evidence = build_exercise_evidence_tensor(
            interactions=interactions,
            exercise_id_map={"11": 0, "12": 1, "13": 2},
        )

        self.assertEqual(tuple(evidence.shape), (3, 6))
        torch.testing.assert_close(
            evidence[0],
            torch.tensor([2.0, 1.0, 1.0, 0.5, math.log1p(2.0), 1.0], dtype=torch.float32),
        )
        torch.testing.assert_close(
            evidence[1],
            torch.tensor([1.0, 1.0, 0.0, 1.0, math.log1p(1.0), 1.0], dtype=torch.float32),
        )
        torch.testing.assert_close(evidence[2], torch.zeros(6, dtype=torch.float32))

    def test_prepare_experiment_split_bundles_marks_eval_splits_as_history_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_path = root / "train.csv"
            valid_path = root / "valid.csv"
            test_path = root / "test.csv"
            q_matrix_path = root / "q_matrix.csv"

            pd.DataFrame(
                [
                    {"stu_id": 1, "exer_id": 11, "cpt_seq": "A", "label": 1},
                    {"stu_id": 2, "exer_id": 12, "cpt_seq": "B", "label": 0},
                ]
            ).to_csv(train_path, index=False)
            pd.DataFrame(
                [
                    {"stu_id": 1, "exer_id": 13, "cpt_seq": "A", "label": 0},
                ]
            ).to_csv(valid_path, index=False)
            pd.DataFrame(
                [
                    {"stu_id": 2, "exer_id": 14, "cpt_seq": "B", "label": 1},
                ]
            ).to_csv(test_path, index=False)
            pd.DataFrame(
                [
                    {"exer_id": 11, "cpt_seq": "A"},
                    {"exer_id": 12, "cpt_seq": "B"},
                    {"exer_id": 13, "cpt_seq": "A"},
                    {"exer_id": 14, "cpt_seq": "B"},
                ]
            ).to_csv(q_matrix_path, index=False)

            bundles = prepare_experiment_split_bundles(
                train_interactions_path=train_path,
                valid_interactions_path=valid_path,
                test_interactions_path=test_path,
                q_matrix_path=q_matrix_path,
            )

        self.assertTrue(bundles["train"].allow_target_in_history)
        self.assertFalse(bundles["valid"].allow_target_in_history)
        self.assertFalse(bundles["test"].allow_target_in_history)
        self.assertEqual(bundles["valid"].split_name, "valid")
        self.assertEqual(bundles["test"].split_name, "test")
        self.assertTrue(bundles["valid"].history_interactions.equals(bundles["train"].interactions))
        self.assertTrue(torch.equal(bundles["valid"].student_exercise_mask, bundles["train"].student_exercise_mask))
        self.assertTrue(
            torch.equal(
                bundles["valid"].student_concept_evidence_tensor,
                bundles["train"].student_concept_evidence_tensor,
            )
        )
        self.assertTrue(torch.equal(bundles["valid"].exercise_evidence_tensor, bundles["train"].exercise_evidence_tensor))

    def test_prepare_step_data_bundle_marks_full_bundle_as_target_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            interactions_path = root / "interactions.csv"
            q_matrix_path = root / "q_matrix.csv"

            pd.DataFrame(
                [
                    {"stu_id": 1, "exer_id": 11, "cpt_seq": "A", "label": 1},
                ]
            ).to_csv(interactions_path, index=False)
            pd.DataFrame([{"exer_id": 11, "cpt_seq": "A"}]).to_csv(q_matrix_path, index=False)

            bundle = prepare_step_data_bundle(interactions_path=interactions_path, q_matrix_path=q_matrix_path)

        self.assertEqual(bundle.split_name, "full")
        self.assertTrue(bundle.allow_target_in_history)
        self.assertTrue(bundle.history_interactions.equals(bundle.interactions))


class HistoryVisibilityEvaluationGuardTest(unittest.TestCase):
    def test_evaluate_model_rejects_target_overlap_for_history_only_bundle(self) -> None:
        interactions = pd.DataFrame([{"stu_id": 1, "exer_id": 11, "label": 1, "cpt_seq": "A"}])
        bundle = StepDataBundle(
            interactions=interactions,
            history_interactions=interactions.copy(),
            split_name="valid",
            allow_target_in_history=False,
            q_matrix=pd.DataFrame([{"exer_id": 11, "cpt_seq": "A"}]),
            student_id_map={"1": 0},
            exercise_id_map={"11": 0},
            concept_id_map={"A": 0},
            q_matrix_tensor=torch.tensor([[1.0]], dtype=torch.float32),
            concept_graph=torch.tensor([[1.0]], dtype=torch.float32),
            student_exercise_mask=torch.tensor([[1.0]], dtype=torch.float32),
            student_tkc_mask=torch.tensor([[1.0]], dtype=torch.float32),
            student_ukc_mask=torch.tensor([[0.0]], dtype=torch.float32),
            response_matrix_tensor=torch.tensor([[1.0]], dtype=torch.float32),
            interaction_student_ids=torch.tensor([0], dtype=torch.long),
            interaction_exercise_ids=torch.tensor([0], dtype=torch.long),
            interaction_labels=torch.tensor([1.0], dtype=torch.float32),
        )

        with self.assertRaisesRegex(ValueError, "reuses target interactions inside propagation history"):
            evaluate_model(bundle=bundle, model=_NeverCalledModel(), device="cpu")

    def test_evaluate_model_allows_train_bundle_overlap(self) -> None:
        interactions = pd.DataFrame([{"stu_id": 1, "exer_id": 11, "label": 1, "cpt_seq": "A"}])
        bundle = StepDataBundle(
            interactions=interactions,
            history_interactions=interactions.copy(),
            split_name="train",
            allow_target_in_history=True,
            q_matrix=pd.DataFrame([{"exer_id": 11, "cpt_seq": "A"}]),
            student_id_map={"1": 0},
            exercise_id_map={"11": 0},
            concept_id_map={"A": 0},
            q_matrix_tensor=torch.tensor([[1.0]], dtype=torch.float32),
            concept_graph=torch.tensor([[1.0]], dtype=torch.float32),
            student_exercise_mask=torch.tensor([[1.0]], dtype=torch.float32),
            student_tkc_mask=torch.tensor([[1.0]], dtype=torch.float32),
            student_ukc_mask=torch.tensor([[0.0]], dtype=torch.float32),
            response_matrix_tensor=torch.tensor([[1.0]], dtype=torch.float32),
            interaction_student_ids=torch.tensor([0], dtype=torch.long),
            interaction_exercise_ids=torch.tensor([0], dtype=torch.long),
            interaction_labels=torch.tensor([1.0], dtype=torch.float32),
        )

        metrics = evaluate_model(bundle=bundle, model=_ConstantProbModel(), device="cpu")

        self.assertAlmostEqual(metrics["loss"], 0.6931471805599453, places=6)


if __name__ == "__main__":
    unittest.main()
