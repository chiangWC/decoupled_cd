from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from utils.response_credit_evaluation import (
    BOOTSTRAP_REPLICATES,
    MIN_VALID_BOOTSTRAP_REPLICATES,
    compute_stage1_gate,
    compute_stage2_gate,
    evaluate_credit_predictions,
    student_clustered_joint_bootstrap,
)


def _frame(rows_per_student: int = 4, students: int = 120) -> pd.DataFrame:
    rows = []
    for student in range(students):
        for row in range(rows_per_student):
            label = (student + row) % 2
            base = 0.75 if label else 0.25
            rows.append(
                {
                    "source_row_id": f"r-{student}-{row}",
                    "stu_id": f"s{student}",
                    "label": label,
                    "in_c": True,
                    "in_c_strict": student < 20,
                    "in_t": row < 2,
                    "prob_full": base,
                    "prob_direct": 0.70 if label else 0.30,
                    "prob_capacity": 0.68 if label else 0.32,
                }
            )
    return pd.DataFrame(rows)


class TestResponseCreditEvaluation(unittest.TestCase):
    def test_control_envelope_uses_best_control_per_metric(self) -> None:
        frame = _frame()
        summary, bootstrap = evaluate_credit_predictions(
            frame,
            dataset="MOOCRadar",
            split_kind="holdout",
            bootstrap_replicates=20,
        )
        c = summary["metrics"]["C"]
        expected_auc = c["full"]["auc"] - max(
            c["direct"]["auc"], c["capacity"]["auc"]
        )
        expected_brier = c["full"]["brier"] - min(
            c["direct"]["brier"], c["capacity"]["brier"]
        )
        self.assertEqual(
            summary["control_envelope_deltas"]["C"][
                "auc_vs_control_envelope"
            ],
            expected_auc,
        )
        self.assertEqual(
            summary["control_envelope_deltas"]["C"][
                "brier_vs_control_envelope"
            ],
            expected_brier,
        )
        self.assertEqual(len(bootstrap.valid), 20)

    def test_joint_bootstrap_uses_minimum_of_both_contrasts(self) -> None:
        result = student_clustered_joint_bootstrap(
            _frame(),
            stratum="synthetic:C",
            replicates=50,
            seed=2024,
        )
        self.assertEqual(len(result.sampled_student_hashes), 50)
        self.assertEqual(int(result.valid.sum()), 50)
        self.assertTrue(np.all(result.joint_min_delta[result.valid] >= 0.0))
        self.assertEqual(
            result.sampling_manifest_sha256,
            student_clustered_joint_bootstrap(
                _frame(), stratum="synthetic:C", replicates=50, seed=2024
            ).sampling_manifest_sha256,
        )

    def test_stage1_requires_full_control_envelope_and_valid_bootstrap(self) -> None:
        payloads = []
        for dataset in ("MOOCRadar", "NIPS34"):
            summary, _ = evaluate_credit_predictions(
                _frame(),
                dataset=dataset,
                split_kind="holdout",
                bootstrap_replicates=20,
            )
            for scope in ("C", "T"):
                summary["control_envelope_deltas"][scope][
                    "auc_vs_control_envelope"
                ] = 0.003
            summary["control_envelope_deltas"]["overall"][
                "auc_vs_control_envelope"
            ] = 0.0
            for scope in ("overall", "C", "T"):
                summary["control_envelope_deltas"][scope][
                    "brier_vs_control_envelope"
                ] = 0.0
            summary["slice_prevalence"]["C"] = {
                "rows": 600,
                "students": 120,
            }
            summary["joint_c_bootstrap"] = {
                "requested_replicates": BOOTSTRAP_REPLICATES,
                "valid_replicates": MIN_VALID_BOOTSTRAP_REPLICATES,
                "invalid_replicates": (
                    BOOTSTRAP_REPLICATES - MIN_VALID_BOOTSTRAP_REPLICATES
                ),
                "invalid_indices": [],
                "sampling_manifest_sha256": "hash",
                "confidence_interval_95": [0.001, 0.005],
                "minimum_valid_replicates": MIN_VALID_BOOTSTRAP_REPLICATES,
            }
            payloads.append(summary)
        gate = compute_stage1_gate(payloads)
        self.assertTrue(gate["stage1_passed"])

        payloads[0]["joint_c_bootstrap"]["valid_replicates"] = 1799
        gate = compute_stage1_gate(payloads)
        self.assertFalse(gate["stage1_passed"])
        self.assertFalse(
            gate["checks"]["MOOCRadar_bootstrap_valid_ge_1800"]
        )

    def test_stage2_refuses_to_run_after_failed_stage1(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "forbidden"):
            compute_stage2_gate([], stage1={"stage1_passed": False})


if __name__ == "__main__":
    unittest.main()
