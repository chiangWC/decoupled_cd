import unittest
from pathlib import Path
import tempfile

import pandas as pd
import torch

from data import prepare_experiment_split_bundles
from scripts.evaluate_history_hiding_stress import build_hidden_bundle, mask_train_history_interactions


class HistoryHidingStressTest(unittest.TestCase):
    def test_mask_train_history_interactions_is_deterministic(self) -> None:
        train_frame = pd.DataFrame(
            [
                {"stu_id": 1, "exer_id": 11, "cpt_seq": "A", "label": 1},
                {"stu_id": 1, "exer_id": 12, "cpt_seq": "B", "label": 0},
                {"stu_id": 2, "exer_id": 13, "cpt_seq": "A", "label": 1},
                {"stu_id": 2, "exer_id": 14, "cpt_seq": "B", "label": 0},
            ]
        )

        first = mask_train_history_interactions(train_frame, hide_ratio=0.5, seed=13)
        second = mask_train_history_interactions(train_frame, hide_ratio=0.5, seed=13)

        self.assertTrue(first.equals(second))
        self.assertLessEqual(len(first), len(train_frame))

    def test_hidden_bundle_keeps_exercise_evidence_when_requested(self) -> None:
        train_frame = pd.DataFrame(
            [
                {"stu_id": 1, "exer_id": 11, "cpt_seq": "A", "label": 1},
                {"stu_id": 1, "exer_id": 12, "cpt_seq": "B", "label": 0},
                {"stu_id": 2, "exer_id": 11, "cpt_seq": "A", "label": 0},
            ]
        )
        valid_frame = pd.DataFrame([{"stu_id": 1, "exer_id": 13, "cpt_seq": "A", "label": 1}])
        test_frame = pd.DataFrame([{"stu_id": 2, "exer_id": 14, "cpt_seq": "B", "label": 0}])
        q_matrix = pd.DataFrame(
            [
                {"exer_id": 11, "cpt_seq": "A"},
                {"exer_id": 12, "cpt_seq": "B"},
                {"exer_id": 13, "cpt_seq": "A"},
                {"exer_id": 14, "cpt_seq": "B"},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_path = root / "train.csv"
            valid_path = root / "valid.csv"
            test_path = root / "test.csv"
            q_matrix_path = root / "q_matrix.csv"
            train_frame.to_csv(train_path, index=False)
            valid_frame.to_csv(valid_path, index=False)
            test_frame.to_csv(test_path, index=False)
            q_matrix.to_csv(q_matrix_path, index=False)
            bundles = prepare_experiment_split_bundles(
                train_interactions_path=train_path,
                valid_interactions_path=valid_path,
                test_interactions_path=test_path,
                q_matrix_path=q_matrix_path,
            )

        masked_history = train_frame.iloc[[0]].reset_index(drop=True)
        hidden_bundle = build_hidden_bundle(
            bundle=bundles["test"],
            masked_history=masked_history,
            keep_exercise_evidence=True,
        )

        self.assertEqual(len(hidden_bundle.history_interactions), 1)
        self.assertFalse(torch.equal(hidden_bundle.student_exercise_mask, bundles["test"].student_exercise_mask))
        self.assertFalse(
            torch.equal(hidden_bundle.student_concept_evidence_tensor, bundles["test"].student_concept_evidence_tensor)
        )
        self.assertTrue(torch.equal(hidden_bundle.exercise_evidence_tensor, bundles["test"].exercise_evidence_tensor))


if __name__ == "__main__":
    unittest.main()
