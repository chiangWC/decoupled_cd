from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.unified_baseline_audit import audit_baseline_rows
from scripts.unified_dataset_audit import canonical_sha256


class UnifiedA0ExplorationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_ids = ("ASSIST17", "MOOCRadar", "XES3G5M")
        datasets = {}
        baseline_rows = []
        for index, dataset_id in enumerate(self.dataset_ids):
            standard = {
                "data_sha256": f"{index + 1:x}" * 64,
                "q_sha256": "a" * 64,
                "prediction_order_sha256": f"{index + 4:x}" * 64,
            }
            holdout = {
                "data_sha256": f"{index + 7:x}" * 64,
                "q_sha256": "a" * 64,
                "prediction_order_sha256": f"{index + 10:x}" * 64,
            }
            record = {
                "dataset_id": dataset_id,
                "eligible": True,
                "zero_count": 2_000 + index,
                "standard": standard,
                "holdout": holdout,
            }
            record["audit_sha256"] = canonical_sha256(record)
            datasets[dataset_id] = record
            for split, metric, value in (
                ("standard", "auc", 0.70 + index / 100),
                ("holdout", "auc", 0.69 + index / 100),
                ("holdout", "zero_auc", 0.60 + index / 100),
            ):
                source = Path(f"/audited/{dataset_id}-{split}-{metric}.json")
                baseline_rows.append({
                    "dataset_id": dataset_id,
                    "model": "KaNCD",
                    "seed": 42,
                    "split_seed": 2024,
                    "split": split,
                    "metric": metric,
                    "value": value,
                    "data_sha256": datasets[dataset_id][split]["data_sha256"],
                    "q_sha256": "a" * 64,
                    "prediction_sha256": "b" * 64,
                    "prediction_order_sha256": datasets[dataset_id][split][
                        "prediction_order_sha256"
                    ],
                    "config_sha256": "c" * 64,
                    "checkpoint_sha256": "d" * 64,
                    "source_path": str(source),
                })
        self.dataset_audit = {
            "schema_version": 1,
            "datasets": datasets,
            "eligible_dataset_ids": list(self.dataset_ids),
        }
        self.dataset_audit["audit_sha256"] = canonical_sha256(self.dataset_audit)
        self.baseline_audit = audit_baseline_rows(
            baseline_rows, self.dataset_audit
        )

    def test_builds_provisional_registry_and_real_external_guards(self) -> None:
        from scripts.unified_a0_exploration import build_exploration_payloads

        registry, guards = build_exploration_payloads(
            self.dataset_audit, self.baseline_audit
        )

        self.assertEqual(registry["dataset_ids"], list(self.dataset_ids))
        self.assertEqual(len(registry["cohort_sha256"]), 64)
        self.assertEqual([row["dataset_id"] for row in guards["rows"]], list(self.dataset_ids))
        self.assertEqual(
            set(guards["rows"][0]),
            {
                "dataset_id", "cohort_sha256", "standard_overall_auc",
                "holdout_overall_auc", "zero_auc", "comparator_sources",
            },
        )

    def test_missing_external_zero_guard_is_baseline_incomplete(self) -> None:
        from scripts.unified_a0_exploration import build_exploration_payloads

        broken = audit_baseline_rows(
            [
                row
                for row in self.baseline_audit["accepted_rows"]
                if not (
                    row["dataset_id"] == "ASSIST17"
                    and row["split"] == "holdout"
                    and row["metric"] == "zero_auc"
                )
            ],
            self.dataset_audit,
        )

        with self.assertRaisesRegex(ValueError, "baseline-incomplete: ASSIST17"):
            build_exploration_payloads(self.dataset_audit, broken)

    def test_controller_cli_exposes_init_and_run_validation(self) -> None:
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "unified_validation_controller.py"),
                "--help",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("init", completed.stdout)
        self.assertIn("run-validation", completed.stdout)


if __name__ == "__main__":
    unittest.main()
