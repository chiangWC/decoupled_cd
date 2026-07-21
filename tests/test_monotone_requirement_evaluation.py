from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from utils import monotone_requirement_evaluation as evaluation


def _artifacts(students: int = 20) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    rows = []
    for student in range(students):
        for item in range(12):
            index = student * 12 + item
            label = (student + item) % 2
            q_count = 2 if item < 10 else 3 if item == 10 else 4
            noise = 0.28 * np.sin(index * 0.73)
            sign = 2 * label - 1
            rows.append({
                "row_id": f"source-line-{index:05d}-s{student}-i{item}",
                "student_id": f"s{student:03d}",
                "item_id": f"i{item:02d}",
                "q_pair_id": f"pair-{item:02d}",
                "q_count": q_count,
                "eligible": True,
                "label": label,
                "full": np.clip(0.5 + 0.20 * sign + noise, 0.01, 0.99),
                "pooled": np.clip(0.5 + 0.08 * sign + noise, 0.01, 0.99),
                "capacity": np.clip(0.5 + 0.10 * sign + noise, 0.01, 0.99),
                "standard_ncd": np.clip(0.5 + 0.12 * sign + noise, 0.01, 0.99),
            })
    table = pd.DataFrame(rows)
    identity = ["row_id", "student_id", "item_id", "q_pair_id", "q_count", "eligible"]
    manifests = {}
    for offset, variant in enumerate(evaluation.VARIANT_NAMES):
        frame = table.loc[:, identity].assign(prob=table[variant])
        if variant != "full":
            frame = frame.sample(frac=1.0, random_state=100 + offset)
        manifests[variant] = frame
    labels = table.loc[:, ["row_id", "label"]].sample(frac=1.0, random_state=999)
    return manifests, labels


def _gate_summary(delta: float, ci_low: float) -> dict:
    def section(value: float, brier: float = 0.0005) -> dict:
        return {"envelope": {"delta_auc": value, "brier_regression": brier}}
    return {
        "slices": {
            "q2": section(delta),
            "eligible_pooled": section(0.001),
            "q2_without_top5_items": section(0.001),
            "q2_without_top5_q_pairs": section(0.001),
        },
        "student_cluster_bootstrap": {
            "identified": True,
            "conservative_envelope": {"ci_low": ci_low},
        },
    }


class TestMonotoneRequirementEvaluation(unittest.TestCase):
    def test_outcome_free_alignment_is_one_to_one_and_ordered(self) -> None:
        manifests, labels = _artifacts(students=4)
        expected_order = manifests["full"]["row_id"].tolist()
        aligned = evaluation.align_prediction_manifests(
            predictions=manifests, labels=labels
        )
        self.assertEqual(aligned["row_id"].tolist(), expected_order)
        self.assertEqual(
            aligned.attrs["row_order_sha256"],
            evaluation.row_order_sha256(expected_order),
        )
        duplicate = {name: frame.copy() for name, frame in manifests.items()}
        duplicate["capacity"].iloc[1, duplicate["capacity"].columns.get_loc("row_id")] = duplicate["capacity"].iloc[0]["row_id"]
        with self.assertRaisesRegex(ValueError, "duplicate outcome-free"):
            evaluation.align_prediction_manifests(predictions=duplicate, labels=labels)
        leaked = {name: frame.copy() for name, frame in manifests.items()}
        leaked["pooled"]["label"] = 0
        with self.assertRaisesRegex(ValueError, "not outcome-free"):
            evaluation.align_prediction_manifests(predictions=leaked, labels=labels)

    def test_auc_without_both_classes_is_an_explicit_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "AUC requires both"):
            evaluation.binary_metrics([1, 1], [0.2, 0.8])

    def test_q2_metrics_slices_removals_and_conservative_bootstrap(self) -> None:
        manifests, labels = _artifacts()
        aligned = evaluation.align_prediction_manifests(
            predictions=manifests, labels=labels
        )
        summary = evaluation.evaluate_requirement_surface(
            aligned,
            dataset="ASSIST17",
            expected_row_order_sha256=aligned.attrs["row_order_sha256"],
        )
        self.assertEqual(summary["q2_row_count"], 200)
        self.assertEqual(
            set(summary["slices"]["q2"]["metrics"]["full"]),
            set(evaluation.METRIC_NAMES),
        )
        self.assertTrue(summary["slices"]["q3"]["identified"])
        self.assertTrue(summary["slices"]["q4"]["identified"])
        self.assertEqual(summary["top5_item_removal"]["removed_ids"], [f"i{i:02d}" for i in range(5)])
        bootstrap = summary["student_cluster_bootstrap"]
        self.assertTrue(bootstrap["identified"])
        self.assertEqual(bootstrap["requested"], 2000)
        controls = np.stack([
            bootstrap["replicate_delta_auc_vs_control"][name]
            for name in evaluation.CONTROL_NAMES
        ])
        np.testing.assert_allclose(
            bootstrap["replicate_envelope_delta_auc"],
            controls.min(axis=0),
        )
        pair = summary["q_pair_cluster_bootstrap_sensitivity"]
        self.assertTrue(pair["identified"])
        self.assertFalse(pair["gating"])
        self.assertFalse(summary["bootstrap_is_multi_seed"])

    def test_aggregate_applies_every_preregistered_gate(self) -> None:
        summaries = {
            "ASSIST17": _gate_summary(0.011, 0.002),
            "MOOCRadar": _gate_summary(0.006, -0.001),
        }
        passed = evaluation.aggregate_stage1_gate(
            summaries,
            surface_noncollapse={"ASSIST17": True, "MOOCRadar": True},
        )
        self.assertTrue(passed["passed"])
        self.assertTrue(all(passed["checks"].values()))
        summaries["MOOCRadar"]["slices"]["q2_without_top5_q_pairs"]["envelope"]["delta_auc"] = 0.0
        failed = evaluation.aggregate_stage1_gate(
            summaries,
            surface_noncollapse={"ASSIST17": True, "MOOCRadar": False},
        )
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["checks"]["both_top5_q_pair_removed_q2_delta_positive"])
        self.assertFalse(failed["checks"]["both_full_surfaces_noncollapsed"])


if __name__ == "__main__":
    unittest.main()
