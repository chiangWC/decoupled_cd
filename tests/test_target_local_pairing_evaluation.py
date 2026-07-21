from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from utils import target_local_pairing_evaluation as evaluation


def _table(students: int = 24, items: int = 8) -> pd.DataFrame:
    rows = []
    for student_index in range(students):
        for item_index in range(items):
            label = (student_index + item_index) % 2
            index = student_index * items + item_index
            base = 0.15 + 0.70 * label
            rows.append(
                {
                    "source_row_id": f"row-{index:04d}",
                    "stu_id": f"student-{student_index:03d}",
                    "exer_id": f"item-{item_index:03d}",
                    "optimizer_item_frequency": float(items - item_index),
                    "coverage_bucket": (
                        "exact_zero"
                        if item_index % 3 == 0
                        else "partial"
                        if item_index % 3 == 1
                        else "full"
                    ),
                    "label": label,
                    "prob_real_pair": np.clip(
                        base + 0.01 * np.sin(index), 0.01, 0.99
                    ),
                    "prob_strong_opms": np.clip(
                        base - 0.06 * (2 * label - 1), 0.01, 0.99
                    ),
                    "prob_perm_pair": np.clip(
                        base - 0.04 * (2 * label - 1), 0.01, 0.99
                    ),
                    "prob_late_fusion": np.clip(
                        base - 0.02 * (2 * label - 1), 0.01, 0.99
                    ),
                }
            )
    return pd.DataFrame(rows)


class TestTargetLocalPairingEvaluation(unittest.TestCase):
    def test_alignment_preserves_real_order(self) -> None:
        table = _table(students=3, items=4)
        real = table.loc[
            :,
            [
                "source_row_id",
                "stu_id",
                "exer_id",
                "prob_real_pair",
            ],
        ]
        order = real["source_row_id"].tolist()

        def variant(name: str) -> pd.DataFrame:
            return (
                table.loc[
                    :,
                    ["source_row_id", "stu_id", "exer_id", f"prob_{name}"],
                ]
                .rename(columns={f"prob_{name}": "prob"})
                .sample(frac=1.0, random_state=len(name))
            )

        aligned = evaluation.align_prediction_frames(
            real_pair=real,
            strong_opms=variant("strong_opms"),
            perm_pair=variant("perm_pair"),
            late_fusion=variant("late_fusion"),
            labels=table.loc[
                :, ["source_row_id", "stu_id", "exer_id", "label"]
            ].sample(frac=1.0, random_state=91),
        )
        self.assertEqual(aligned["source_row_id"].tolist(), order)
        self.assertEqual(
            aligned.attrs["real_row_order_sha256"],
            evaluation.row_order_sha256(order),
        )
        np.testing.assert_allclose(
            aligned["prob_perm_pair"], table["prob_perm_pair"]
        )

    def test_duplicate_and_missing_ids_are_rejected(self) -> None:
        table = _table(students=3, items=4)
        duplicate = table.copy()
        duplicate.loc[1, "source_row_id"] = duplicate.loc[0, "source_row_id"]
        with self.assertRaisesRegex(ValueError, "duplicate source_row_id"):
            evaluation.validate_prediction_table(duplicate)

        strong = table.loc[
            :, ["source_row_id", "prob_strong_opms"]
        ].iloc[:-1]
        with self.assertRaisesRegex(ValueError, "row-ID set differs"):
            evaluation.align_prediction_frames(
                real_pair=table.loc[
                    :,
                    [
                        "source_row_id",
                        "stu_id",
                        "exer_id",
                        "prob_real_pair",
                    ],
                ],
                strong_opms=strong,
                perm_pair=table.loc[
                    :, ["source_row_id", "prob_perm_pair"]
                ],
                late_fusion=table.loc[
                    :, ["source_row_id", "prob_late_fusion"]
                ],
                labels=table.loc[:, ["source_row_id", "label"]],
            )

    def test_joint_bootstrap_reuses_each_student_sample(self) -> None:
        table = _table()
        result = evaluation.evaluate_target_local_pairing(table)
        summary = result.summary
        bootstrap = summary["student_cluster_bootstrap"]
        self.assertTrue(bootstrap["identified"])
        self.assertEqual(bootstrap["requested"], 2000)
        self.assertEqual(
            bootstrap["used"] + bootstrap["invalid_single_class"], 2000
        )
        replicate = bootstrap["replicate_deltas"]
        expected_joint = np.min(
            np.stack(
                [replicate[name] for name in evaluation.CONTROL_NAMES]
            ),
            axis=0,
        )
        np.testing.assert_allclose(replicate["joint_min"], expected_joint)
        self.assertEqual(
            summary["joint_min_delta_ci"],
            bootstrap["joint_min_delta_ci"],
        )
        self.assertEqual(
            summary["dataset_effect"],
            min(
                value["auc"]
                for value in summary["deltas_vs_control"].values()
            ),
        )

    def test_hash_and_metrics_are_deterministic(self) -> None:
        table = _table()
        expected = evaluation.row_order_sha256(table["source_row_id"])
        first = evaluation.evaluate_target_local_pairing(
            table, expected_real_order_sha256=expected
        )
        second = evaluation.evaluate_target_local_pairing(table)
        self.assertEqual(
            first.summary["student_cluster_bootstrap"][
                "sample_index_sha256"
            ],
            second.summary["student_cluster_bootstrap"][
                "sample_index_sha256"
            ],
        )
        for metrics in first.summary["metrics"].values():
            self.assertEqual(set(metrics), {"auc", "brier", "acc", "rmse"})
        with self.assertRaisesRegex(ValueError, "row-order"):
            evaluation.evaluate_target_local_pairing(
                table, expected_real_order_sha256="wrong"
            )


    def test_coverage_and_target_item_sensitivity_are_nongating(self) -> None:
        table = _table(students=24, items=24)
        result = evaluation.evaluate_target_local_pairing(table)
        coverage = result.summary["coverage_descriptive_slices"]
        self.assertFalse(coverage["gating"])
        self.assertEqual(
            set(coverage["buckets"]), {"exact_zero", "partial", "full"}
        )
        for bucket in coverage["buckets"].values():
            self.assertTrue(bucket["identified"])
            self.assertEqual(
                set(bucket["metrics"]), set(evaluation.MODEL_NAMES)
            )
        top = result.summary["top1pct_train_frequency_item_removal"]
        self.assertFalse(top["gating"])
        self.assertEqual(top["removed_items"], ["item-000"])
        item_bootstrap = result.summary["item_cluster_bootstrap"]
        self.assertTrue(item_bootstrap["identified"])
        self.assertFalse(item_bootstrap["gating"])
        leave = result.summary["leave_one_item_out"]
        self.assertFalse(leave["gating"])
        self.assertEqual(leave["identified_item_count"], 24)
        self.assertEqual(len(result.leave_one_item_out), 24)

    def test_item_bootstrap_can_be_not_identified(self) -> None:
        result = evaluation.evaluate_target_local_pairing(
            _table(students=24, items=8)
        )
        item_bootstrap = result.summary["item_cluster_bootstrap"]
        self.assertFalse(item_bootstrap["identified"])
        self.assertIn("fewer_than_20", item_bootstrap["reason"])

    def test_fixed_model_donor_sensitivity_has_no_p_value(self) -> None:
        table = _table(students=3, items=4)
        labels = table["label"].to_numpy()
        base = table["prob_perm_pair"].to_numpy()
        matrix = np.stack(
            [
                np.clip(base + replicate * 1e-5, 0.0, 1.0)
                for replicate in range(200)
            ]
        )
        summary = evaluation.summarize_donor_sensitivity(
            labels=labels,
            real_probabilities=table["prob_real_pair"].to_numpy(),
            donor_probabilities=matrix,
            mapping_hashes=[
                f"mapping-{replicate:03d}" for replicate in range(200)
            ],
            source_row_ids=table["source_row_id"],
        )
        self.assertEqual(summary["unique_mapping_hash_count"], 200)
        self.assertEqual(len(summary["auc"]["values"]), 200)
        self.assertEqual(len(summary["brier"]["values"]), 200)
        self.assertAlmostEqual(
            summary["reference_real_auc"],
            evaluation._metrics(
                labels, table["prob_real_pair"].to_numpy()
            )["auc"],
        )
        self.assertAlmostEqual(
            summary["reference_real_brier"],
            evaluation._metrics(
                labels, table["prob_real_pair"].to_numpy()
            )["brier"],
        )
        np.testing.assert_allclose(
            summary["donor_minus_real_auc"]["values"],
            np.asarray(summary["auc"]["values"])
            - summary["reference_real_auc"],
        )
        np.testing.assert_allclose(
            summary["donor_minus_real_brier"]["values"],
            np.asarray(summary["brier"]["values"])
            - summary["reference_real_brier"],
        )
        self.assertEqual(summary["change_direction"], "donor_minus_real")
        self.assertFalse(summary["p_value_computed"])
        self.assertEqual(len(summary["probability_sha256"]), 200)
        self.assertNotIn("p_value", summary)


if __name__ == "__main__":
    unittest.main()
