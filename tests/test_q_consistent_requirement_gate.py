from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.analyze_target_coverage_gate import load_train_valid
from scripts.evaluate_q_consistent_requirement_gate import (
    build_q_consistent_coverage,
    evaluate_gate,
    external_win_flags,
    load_aligned_probabilities,
    target_mask,
)


class QConsistentRequirementGateTest(unittest.TestCase):
    def test_q_union_defines_both_history_and_target_coverage(self) -> None:
        train = pd.DataFrame(
            {
                "stu_id": ["0", "1"],
                "exer_id": ["10", "12"],
                "cpt_seq": ["1", "3"],
                "label": [1, 0],
            }
        )
        valid = pd.DataFrame(
            {
                "stu_id": ["0", "1"],
                "exer_id": ["11", "13"],
                "cpt_seq": ["2", "4"],
                "label": [1, 0],
            }
        )
        q_map = {
            "10": frozenset({"1", "2"}),
            "11": frozenset({"2"}),
            "12": frozenset({"3"}),
            "13": frozenset({"4"}),
        }

        coverage = build_q_consistent_coverage(train, valid, q_map)

        self.assertEqual(coverage["target_coverage"].tolist(), [1.0, 0.0])
        self.assertEqual(coverage["coverage_bucket"].tolist(), ["full", "zero"])

    def test_target_masks_follow_registered_semantics(self) -> None:
        frame = pd.DataFrame(
            {
                "label": [0, 1, 0, 1],
                "coverage_bucket": ["zero", "zero", "low", "full"],
                "coverage_group": [
                    "low_coverage",
                    "low_coverage",
                    "low_coverage",
                    "full_coverage",
                ],
            }
        )
        self.assertEqual(target_mask(frame, "bucket:zero").tolist(), [True, True, False, False])
        self.assertEqual(target_mask(frame, "low_coverage").tolist(), [True, True, True, False])

    def test_loader_rejects_reordered_predictions(self) -> None:
        valid = pd.DataFrame(
            {
                "stu_id": ["0", "1"],
                "exer_id": ["10", "11"],
                "label": [0, 1],
                "source_row_id": ["a", "b"],
            }
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            valid.iloc[::-1].assign(prob=[0.8, 0.2]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_aligned_probabilities(valid, path, role="control")

    def test_data_loader_never_opens_test_and_unions_long_q(self) -> None:
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            pd.DataFrame(
                {"stu_id": [0], "exer_id": [10], "cpt_seq": [1], "label": [1]}
            ).to_csv(data_dir / "train.csv", index=False)
            pd.DataFrame(
                {"stu_id": [0], "exer_id": [11], "cpt_seq": [2], "label": [0]}
            ).to_csv(data_dir / "valid.csv", index=False)
            pd.DataFrame(
                {"exer_id": [10, 10, 11], "cpt_seq": [1, 2, 2]}
            ).to_csv(data_dir / "Q_matrix.csv", index=False)
            (data_dir / "test.csv").write_text("malformed\n", encoding="utf-8")

            _, _, q_map, _ = load_train_valid(data_dir)

        self.assertEqual(q_map["10"], frozenset({"1", "2"}))

    def test_external_win_boundary_is_exact(self) -> None:
        external = {"S": 0.80, "H": 0.79, "T": 0.75}
        equal_target = {"S": 0.80, "H": 0.79, "T": 0.75}
        self.assertEqual(
            external_win_flags(equal_target, external),
            {"ordinary": False, "strict": False},
        )
        strict = {"S": 0.80, "H": 0.79, "T": 0.750001}
        self.assertEqual(
            external_win_flags(strict, external),
            {"ordinary": True, "strict": True},
        )
        ordinary = {"S": 0.798, "H": 0.788, "T": 0.750001}
        self.assertEqual(
            external_win_flags(ordinary, external),
            {"ordinary": True, "strict": False},
        )

    def test_preregistered_gate_threshold_boundaries(self) -> None:
        def result(name: str, target_gain: float, ci_low: float) -> dict:
            return {
                "dataset": name,
                "external_win": {"ordinary": True, "strict": False},
                "delta_auc_full_minus_control": {
                    "S": -0.001,
                    "H": 0.0,
                    "T": target_gain,
                },
                "target_bootstrap": {"ci_low": ci_low},
            }

        passing = evaluate_gate(
            [result("a", 0.003, 1e-9), result("b", 0.002, -0.1)]
        )
        self.assertTrue(passing["passed"])

        zero_ci = evaluate_gate(
            [result("a", 0.003, 0.0), result("b", 0.002, -0.1)]
        )
        self.assertFalse(zero_ci["passed"])
        self.assertFalse(zero_ci["checks"]["at_least_one_winning_ci_low_gt_0"])

        regression = result("a", 0.003, 1e-9)
        regression["delta_auc_full_minus_control"]["S"] = -0.0010001
        failed = evaluate_gate([regression, result("b", 0.002, -0.1)])
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["checks"]["no_winning_axis_regression_gt_0.001"])


if __name__ == "__main__":
    unittest.main()
