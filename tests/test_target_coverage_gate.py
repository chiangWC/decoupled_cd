from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from scripts.analyze_target_coverage_gate import (
    build_analysis_frame,
    load_aligned_predictions,
    load_train_valid,
    residualize_two_way,
    two_way_fixed_effects,
)


class TargetCoverageGateTest(unittest.TestCase):
    def test_loader_unions_long_q_and_never_requires_test(self) -> None:
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            pd.DataFrame(
                {
                    "stu_id": [0, 1],
                    "exer_id": [10, 11],
                    "cpt_seq": ["1", "3"],
                    "label": [1, 0],
                }
            ).to_csv(data_dir / "train.csv", index=False)
            pd.DataFrame(
                {
                    "stu_id": [0],
                    "exer_id": [12],
                    "cpt_seq": ["1"],
                    "label": [1],
                }
            ).to_csv(data_dir / "valid.csv", index=False)
            # A malformed test artifact demonstrates that Gate A never opens it.
            (data_dir / "test.csv").write_text(
                "this,is,not,a,dataset\n", encoding="utf-8"
            )
            pd.DataFrame(
                {
                    "exer_id": [10, 10, 11, 12, 12],
                    "cpt_seq": [1, 2, 3, 1, 3],
                }
            ).to_csv(data_dir / "Q_matrix.csv", index=False)

            train, valid, q_map, _ = load_train_valid(data_dir)

        self.assertEqual(len(train), 2)
        self.assertEqual(len(valid), 1)
        self.assertEqual(q_map["10"], frozenset({"1", "2"}))
        self.assertEqual(q_map["12"], frozenset({"1", "3"}))

    def test_build_analysis_uses_target_overlap_and_global_coverage(self) -> None:
        train = pd.DataFrame(
            {
                "stu_id": ["0", "1"],
                "exer_id": ["10", "11"],
                "label": [1, 0],
            }
        )
        aligned = pd.DataFrame(
            {
                "stu_id": ["0", "1"],
                "exer_id": ["12", "12"],
                "label": [1, 0],
                "full_prob": [0.8, 0.2],
                "external_prob": [0.6, 0.4],
            }
        )
        q_map = {
            "10": frozenset({"1", "2"}),
            "11": frozenset({"3"}),
            "12": frozenset({"1", "3"}),
        }

        frame = build_analysis_frame(train, aligned, q_map)

        self.assertEqual(frame["target_coverage"].tolist(), [0.5, 0.5])
        self.assertEqual(frame["target_seen_concept_count"].tolist(), [1, 1])
        np.testing.assert_allclose(
            frame["student_global_coverage"].to_numpy(), [2 / 3, 1 / 3]
        )
        self.assertTrue((frame["full_log_loss_advantage"] > 0).all())

    def test_prediction_alignment_rejects_reordered_rows(self) -> None:
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            valid = pd.DataFrame(
                {"stu_id": ["0", "1"], "exer_id": ["10", "11"], "label": [0, 1]}
            )
            full = valid.assign(prob=[0.2, 0.8])
            external = valid.iloc[::-1].assign(prob=[0.7, 0.3])
            full_path = data_dir / "full.csv"
            external_path = data_dir / "external.csv"
            full.to_csv(full_path, index=False)
            external.to_csv(external_path, index=False)

            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_aligned_predictions(valid, full_path, external_path)

    def test_two_way_fixed_effect_recovers_interaction_signal(self) -> None:
        rows = []
        for student in range(30):
            for item in range(6):
                uncovered = ((student * (item + 1)) % 5) / 4
                outcome = 0.4 * uncovered + 0.03 * student - 0.05 * item
                rows.append(
                    {
                        "stu_id": str(student),
                        "exer_id": str(item),
                        "uncovered_fraction": uncovered,
                        "external_log_loss": outcome,
                    }
                )
        frame = pd.DataFrame(rows)

        result = two_way_fixed_effects(
            frame,
            ["external_log_loss"],
            bootstrap=300,
            seed=7,
        )[0]

        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["beta_uncovered_fraction"], 0.4, places=10)
        self.assertGreater(result["ci_low"], 0.39)
        self.assertLess(result["ci_high"], 0.41)

    def test_two_way_fixed_effect_marks_constant_coverage_unidentified(self) -> None:
        frame = pd.DataFrame(
            {
                "stu_id": ["0", "0", "1", "1"],
                "exer_id": ["10", "11", "10", "11"],
                "uncovered_fraction": np.ones(4),
                "external_log_loss": [0.2, 0.4, 0.5, 0.3],
            }
        )

        result = two_way_fixed_effects(
            frame,
            ["external_log_loss"],
            bootstrap=20,
            seed=11,
        )[0]

        self.assertEqual(result["status"], "not_identified")

    def test_residualization_removes_both_group_means(self) -> None:
        student = np.array([0, 0, 1, 1, 2, 2])
        item = np.array([0, 1, 0, 1, 0, 1])
        values = np.array([1.0, 2.0, 3.0, 7.0, 5.0, 4.0])
        residual, _ = residualize_two_way(values, student, item)

        for codes in (student, item):
            for code in np.unique(codes):
                self.assertAlmostEqual(
                    float(residual[codes == code].mean()), 0.0, places=10
                )


if __name__ == "__main__":
    unittest.main()
