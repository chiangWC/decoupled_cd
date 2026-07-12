from __future__ import annotations

import math
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
        weighted_deltas: tuple[float, ...] | None = None,
        parameter_count: int = 123,
    ) -> list[dict[str, object]]:
        zero_deltas = zero_deltas or (0.0,) * count
        ordinary_deltas = ordinary_deltas or (0.0,) * count
        weighted_deltas = weighted_deltas or (0.0,) * count
        return [
            {
                "dataset_id": f"dataset-{index}",
                "cohort_sha256": self.cohort,
                "architecture_fingerprint": fingerprint,
                "standard_overall_auc": 0.8,
                "holdout_overall_auc": 0.79,
                "zero_auc": 0.7 + zero_deltas[index],
                "ordinary_doa": 0.6 + ordinary_deltas[index],
                "weighted_doa": 0.61 + weighted_deltas[index],
                "parameter_count": parameter_count,
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

    def test_only_overall_auc_regressions_fail_global_nonregression_gates(self) -> None:
        baseline = self.rows(self.baseline_fingerprint, count=3)
        candidate = self.rows(
            self.candidate_fingerprint,
            count=3,
            zero_deltas=(0.001, 0.001, 0.001),
            ordinary_deltas=(0.01, 0.01, 0.01),
        )
        candidate[0]["standard_overall_auc"] = 0.7999999999999999
        candidate[1]["holdout_overall_auc"] = 0.7899999999999999
        candidate[2]["weighted_doa"] = 0.59

        decision = evaluate_candidate(baseline, candidate)

        self.assertFalse(decision["pass"])
        self.assertIn("standard_overall_auc_non_regression", decision["failed_gates"])
        self.assertIn("holdout_overall_auc_non_regression", decision["failed_gates"])
        self.assertNotIn("weighted_doa_non_regression", decision["gates"])

    def test_doa_regression_is_ranked_but_never_a_hard_gate(self) -> None:
        baseline = self.rows(self.baseline_fingerprint, count=3)
        candidate = self.rows(
            self.candidate_fingerprint,
            count=3,
            zero_deltas=(0.001, 0.001, 0.001),
            ordinary_deltas=(-0.03, -0.02, -0.01),
            weighted_deltas=(-0.03, -0.02, -0.01),
        )

        decision = evaluate_candidate(baseline, candidate)

        self.assertTrue(decision["pass"])
        self.assertNotIn("weighted_doa_non_regression", decision["gates"])
        self.assertAlmostEqual(
            decision["ranking"]["mean_weighted_doa_delta"], -0.02
        )
        self.assertEqual(decision["ranking"]["negative_parameter_count"], -369.0)

    def test_parameter_count_is_required_and_must_be_a_nonnegative_integer(self) -> None:
        baseline = self.rows(self.baseline_fingerprint, count=3)
        candidate = self.rows(self.candidate_fingerprint, count=3)
        candidate[0].pop("parameter_count")
        with self.assertRaisesRegex(ValueError, "parameter_count"):
            evaluate_candidate(baseline, candidate)
        for invalid in (-1, 1.5, True):
            candidate = self.rows(self.candidate_fingerprint, count=3)
            candidate[0]["parameter_count"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "parameter_count"
            ):
                evaluate_candidate(baseline, candidate)

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

    def test_identity_hashes_require_lowercase_canonical_sha256(self) -> None:
        invalid_values = ("A" * 64, "a" * 63, "g" * 64)
        for field in ("architecture_fingerprint", "cohort_sha256"):
            for value in invalid_values:
                with self.subTest(field=field, value=value):
                    baseline = self.rows(self.baseline_fingerprint, count=3)
                    candidate = self.rows(self.candidate_fingerprint, count=3)
                    for row in candidate:
                        row[field] = value
                    if field == "cohort_sha256":
                        for row in baseline:
                            row[field] = value
                    with self.assertRaisesRegex(ValueError, "lowercase.*64-hex"):
                        evaluate_candidate(baseline, candidate)

    def test_mismatched_dataset_sets_are_rejected_and_cohort_hash_is_a_gate(self) -> None:
        baseline = self.rows(self.baseline_fingerprint, count=3)
        candidate = self.rows(self.candidate_fingerprint, count=3)
        candidate[-1]["dataset_id"] = "replacement"
        with self.assertRaisesRegex(ValueError, "dataset sets"):
            evaluate_candidate(baseline, candidate)

        candidate = self.rows(self.candidate_fingerprint, count=3)
        candidate[-1]["cohort_sha256"] = "d" * 64
        for row in candidate:
            row["cohort_sha256"] = "d" * 64
        decision = evaluate_candidate(baseline, candidate)
        self.assertFalse(decision["pass"])
        self.assertEqual(
            decision["failed_gates"],
            [
                "zero_auc_improved_two_thirds",
                "zero_auc_delta_at_least_0.001",
                "same_frozen_cohort",
            ],
        )

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

        candidate = self.rows(self.candidate_fingerprint, count=3)
        candidate[0]["validation_metrics_path"] = "/private/test.csv"
        with self.assertRaisesRegex(ValueError, "test"):
            evaluate_candidate(baseline, candidate)

    def test_test_token_detection_accepts_latest_and_contest(self) -> None:
        baseline = self.rows(self.baseline_fingerprint, count=3)
        candidate = self.rows(
            self.candidate_fingerprint,
            count=3,
            zero_deltas=(0.001, 0.001, 0.001),
            ordinary_deltas=(0.01, 0.01, 0.01),
        )
        for row in baseline + candidate:
            row["latest_validation_path"] = "/private/contest/validation.json"

        decision = evaluate_candidate(baseline, candidate)

        self.assertTrue(decision["pass"])

    def test_metrics_must_be_actual_floats_in_unit_interval(self) -> None:
        invalid_values = (True, 1, -0.0000000000000001, 1.0000000000000002)
        for value in invalid_values:
            with self.subTest(value=value):
                baseline = self.rows(self.baseline_fingerprint, count=3)
                candidate = self.rows(self.candidate_fingerprint, count=3)
                candidate[0]["zero_auc"] = value
                with self.assertRaisesRegex(
                    ValueError,
                    r"float.*\[0, 1\]|not numeric",
                ):
                    evaluate_candidate(baseline, candidate)

    def test_nonfinite_computed_delta_is_rejected_before_json_output(self) -> None:
        baseline = self.rows(self.baseline_fingerprint, count=3)
        candidate = self.rows(self.candidate_fingerprint, count=3)
        baseline[0]["zero_auc"] = -float.fromhex("0x1.fffffffffffffp+1023")
        candidate[0]["zero_auc"] = float.fromhex("0x1.fffffffffffffp+1023")

        with self.assertRaisesRegex(ValueError, r"finite|\[0, 1\]"):
            evaluate_candidate(baseline, candidate)

    def test_all_decision_deltas_and_rankings_are_finite(self) -> None:
        baseline = self.rows(self.baseline_fingerprint, count=3)
        candidate = self.rows(
            self.candidate_fingerprint,
            count=3,
            zero_deltas=(0.001, 0.001, 0.001),
            ordinary_deltas=(0.01, 0.01, 0.01),
        )

        decision = evaluate_candidate(baseline, candidate)

        computed = [
            value
            for metrics in decision["deltas"].values()
            for value in metrics.values()
        ]
        computed.extend(decision["ranking"].values())
        self.assertTrue(all(type(value) is float for value in computed))
        self.assertTrue(all(math.isfinite(value) for value in computed))


if __name__ == "__main__":
    unittest.main()
