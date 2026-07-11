from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from scripts.run_unified_validation import (
    ELIGIBLE_DATASET_IDS,
    architecture_fingerprint,
    assemble_candidate_rows,
    build_train_command,
    can_reach_primary_cohort,
    parse_gpu_inventory,
    select_gpu_index,
    validate_smoke_summary,
)


class UnifiedValidationRunnerTests(unittest.TestCase):
    def test_script_entrypoint_imports_from_project_root(self):
        completed = subprocess.run(
            [sys.executable, "scripts/run_unified_validation.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_eligible_pool_excludes_nips_without_exact_zero_rows(self):
        self.assertEqual(
            ELIGIBLE_DATASET_IDS,
            ("ASSIST09", "ASSIST17", "MOOCRadar", "XES3G5M"),
        )
        self.assertNotIn("NIPS34", ELIGIBLE_DATASET_IDS)

    def test_training_command_maps_evaluation_input_to_valid(self):
        command = build_train_command(
            dataset_id="ASSIST17",
            split_id="holdout",
            architecture="b0",
            data_root=Path("/datasets"),
            output=Path("/artifacts/train-summary.json"),
            device="cuda:0",
        )
        valid_index = command.index("--valid-interactions") + 1
        evaluation_index = command.index("--test-interactions") + 1
        self.assertEqual(command[evaluation_index], command[valid_index])
        self.assertEqual(Path(command[evaluation_index]).name, "valid.csv")
        self.assertNotIn("test.csv", command)
        self.assertIn("--seed", command)
        self.assertEqual(command[command.index("--seed") + 1], "42")

    def test_gpu_selection_prefers_idle_then_allows_under_half_memory(self):
        snapshots = parse_gpu_inventory(
            "1, 6000, 24000, 0\n"
            "0, 10, 24000, 0\n"
            "2, 100, 24000, 30\n"
            "3, 13000, 24000, 0\n"
        )
        self.assertEqual(select_gpu_index(snapshots), 0)
        self.assertEqual(select_gpu_index([snapshots[2]]), 2)
        with self.assertRaisesRegex(RuntimeError, "under half memory"):
            select_gpu_index([snapshots[3]])

    def test_smoke_summary_requires_mastery_loss_fingerprint_and_gpu_peak(self):
        fingerprint = architecture_fingerprint("m2-m3")
        summary = {
            "architecture_fingerprint": fingerprint,
            "mastery_shape": [3, 4],
            "final_loss": 0.4,
            "peak_gpu_memory_gb": 0.2,
        }
        validate_smoke_summary(
            summary,
            expected_fingerprint=fingerprint,
            require_gpu_peak=True,
        )
        for field, value in (
            ("mastery_shape", [0, 4]),
            ("final_loss", float("nan")),
            ("architecture_fingerprint", "0" * 64),
            ("peak_gpu_memory_gb", None),
        ):
            broken = dict(summary)
            broken[field] = value
            with self.assertRaises(ValueError):
                validate_smoke_summary(
                    broken,
                    expected_fingerprint=fingerprint,
                    require_gpu_peak=True,
                )

    def test_candidate_rows_require_every_frozen_dataset_and_both_splits(self):
        cohort_hash = "a" * 64
        fingerprint = architecture_fingerprint("m2")
        summaries = []
        for dataset_id in ("ASSIST09", "ASSIST17", "MOOCRadar"):
            for split_id, overall_auc in (
                ("standard", 0.70),
                ("holdout", 0.71),
            ):
                summaries.append(
                    {
                        "dataset_id": dataset_id,
                        "split_id": split_id,
                        "architecture_fingerprint": fingerprint,
                        "seed": 42,
                        "overall_auc": overall_auc,
                        "zero_auc": 0.61,
                        "ordinary_doa": 0.62,
                        "weighted_doa": 0.63,
                        "mastery_shape": [2, 3],
                        "final_loss": 0.4,
                    }
                )
        rows = assemble_candidate_rows(
            summaries,
            cohort_dataset_ids=["ASSIST09", "ASSIST17", "MOOCRadar"],
            cohort_sha256=cohort_hash,
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["standard_overall_auc"], 0.70)
        self.assertEqual(rows[0]["holdout_overall_auc"], 0.71)
        self.assertEqual(rows[0]["zero_auc"], 0.61)
        with self.assertRaisesRegex(ValueError, "missing validation split"):
            assemble_candidate_rows(
                summaries[:-1],
                cohort_dataset_ids=["ASSIST09", "ASSIST17", "MOOCRadar"],
                cohort_sha256=cohort_hash,
            )

    def test_candidate_stops_when_three_successes_are_unreachable(self):
        self.assertTrue(can_reach_primary_cohort(successes=1, remaining=2))
        self.assertFalse(can_reach_primary_cohort(successes=1, remaining=1))


if __name__ == "__main__":
    unittest.main()
