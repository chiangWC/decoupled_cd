from __future__ import annotations

import unittest

from scripts.unified_campaign import evaluate_candidate


class UnifiedCampaignTests(unittest.TestCase):
    cohort = "c" * 64
    baseline_fingerprint = "b" * 64
    candidate_fingerprint = "a" * 64

    def rows(
        self,
        fingerprint: str,
        *,
        count: int = 4,
        zero_deltas: tuple[float, ...] | None = None,
        ordinary_deltas: tuple[float, ...] | None = None,
    ) -> list[dict[str, object]]:
        zero_deltas = zero_deltas or (0.0,) * count
        ordinary_deltas = ordinary_deltas or (0.0,) * count
        return [
            {
                "dataset_id": f"dataset-{index}",
                "cohort_sha256": self.cohort,
                "architecture_fingerprint": fingerprint,
                "standard_overall_auc": 0.8,
                "holdout_overall_auc": 0.79,
                "zero_auc": 0.7 + zero_deltas[index],
                "ordinary_doa": 0.6 + ordinary_deltas[index],
                "weighted_doa": 0.61,
            }
            for index in range(count)
        ]

    def test_ceil_two_thirds_datasets_must_improve(self) -> None:
        baseline = self.rows(self.baseline_fingerprint)
        candidate = self.rows(
            self.candidate_fingerprint,
            zero_deltas=(0.001, 0.0002, 0.0001, 0.0),
            ordinary_deltas=(0.01, 0.01, 0.01, 0.0),
        )

        decision = evaluate_candidate(baseline, candidate)

        self.assertEqual(decision["required_improvements"], 3)
        self.assertTrue(decision["pass"])

    def test_any_overall_or_weighted_doa_regression_fails_global_gate(self) -> None:
        baseline = self.rows(self.baseline_fingerprint, count=3)
        candidate = self.rows(
            self.candidate_fingerprint,
            count=3,
            zero_deltas=(0.001, 0.001, 0.001),
            ordinary_deltas=(0.01, 0.01, 0.01),
        )
        candidate[0]["standard_overall_auc"] = 0.7999999999999999
        candidate[1]["holdout_overall_auc"] = 0.7899999999999999
        candidate[2]["weighted_doa"] = 0.6099999999999999

        decision = evaluate_candidate(baseline, candidate)

        self.assertFalse(decision["pass"])
        self.assertIn("standard_overall_auc_non_regression", decision["failed_gates"])
        self.assertIn("holdout_overall_auc_non_regression", decision["failed_gates"])
        self.assertIn("weighted_doa_non_regression", decision["failed_gates"])

    def test_one_zero_auc_delta_must_reach_raw_point_zero_zero_one(self) -> None:
        baseline = self.rows(self.baseline_fingerprint, count=3)
        candidate = self.rows(
            self.candidate_fingerprint,
            count=3,
            zero_deltas=(0.0009, 0.0008, 0.0007),
            ordinary_deltas=(0.01, 0.01, 0.01),
        )

        decision = evaluate_candidate(baseline, candidate)

        self.assertFalse(decision["pass"])
        self.assertIn("zero_auc_delta_at_least_0.001", decision["failed_gates"])
        self.assertEqual(
            decision["deltas"]["dataset-0"]["zero_auc"],
            candidate[0]["zero_auc"] - baseline[0]["zero_auc"],
        )

    def test_mixed_architecture_fingerprints_are_rejected(self) -> None:
        baseline = self.rows(self.baseline_fingerprint, count=3)
        candidate = self.rows(
            self.candidate_fingerprint,
            count=3,
            zero_deltas=(0.001, 0.001, 0.001),
            ordinary_deltas=(0.01, 0.01, 0.01),
        )
        candidate[-1]["architecture_fingerprint"] = "d" * 64

        with self.assertRaisesRegex(ValueError, "mixed.*fingerprint"):
            evaluate_candidate(baseline, candidate)

    def test_mismatched_dataset_sets_and_cohort_hashes_are_rejected(self) -> None:
        baseline = self.rows(self.baseline_fingerprint, count=3)
        candidate = self.rows(self.candidate_fingerprint, count=3)
        candidate[-1]["dataset_id"] = "replacement"
        with self.assertRaisesRegex(ValueError, "dataset sets"):
            evaluate_candidate(baseline, candidate)

        candidate = self.rows(self.candidate_fingerprint, count=3)
        candidate[-1]["cohort_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "cohort"):
            evaluate_candidate(baseline, candidate)

    def test_test_metrics_and_paths_are_rejected(self) -> None:
        baseline = self.rows(self.baseline_fingerprint, count=3)
        candidate = self.rows(self.candidate_fingerprint, count=3)
        candidate[0]["test_auc"] = 0.99
        with self.assertRaisesRegex(ValueError, "test"):
            evaluate_candidate(baseline, candidate)

        candidate = self.rows(self.candidate_fingerprint, count=3)
        candidate[0]["validation_metrics_path"] = "/private/test/metrics.json"
        with self.assertRaisesRegex(ValueError, "test"):
            evaluate_candidate(baseline, candidate)

        candidate = self.rows(self.candidate_fingerprint, count=3)
        candidate[0]["latestTestMetricPath"] = "/private/latestTestMetrics.json"
        with self.assertRaisesRegex(ValueError, "test"):
            evaluate_candidate(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
