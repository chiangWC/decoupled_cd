from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.audit_history_gate_anchors import (
    DATASETS,
    audit_history_gate_anchors,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_factorized_requirement_factorial.sh"


class HistoryGateRunnerTest(unittest.TestCase):
    def test_plan_contains_only_four_holdout_wo_both_jobs(self) -> None:
        completed = subprocess.run(
            [
                "bash",
                str(RUNNER),
                "--data-root",
                "/synthetic/knofield",
                "--stage",
                "history_gate",
                "--devices",
                "cpu",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        plan_rows = [
            line for line in completed.stdout.splitlines()
            if line[:2].isdigit() and "\t" in line
        ]
        self.assertEqual(len(plan_rows), 4)
        for dataset, row in zip(DATASETS, plan_rows, strict=True):
            self.assertEqual(row.split("\t")[1:], [dataset, "holdout", "wo_both"])
        self.assertEqual(
            completed.stdout.count(
                "--evidence-representation-mode calibrated_summary_control"
            ),
            4,
        )
        self.assertEqual(
            completed.stdout.count(
                "--target-requirement-mode factorized_item_control"
            ),
            4,
        )
        self.assertEqual(completed.stdout.count("--evaluation-stage validation"), 4)
        self.assertNotIn("test.csv", completed.stdout)

    def test_rejected_full_factorial_stage_is_unavailable(self) -> None:
        completed = subprocess.run(
            [
                "bash",
                str(RUNNER),
                "--data-root",
                "/synthetic/knofield",
                "--stage",
                "full_factorial",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("history_gate", completed.stderr)
        self.assertNotIn("requires --gate-approved", completed.stderr)

    def test_anchor_audit_locks_validation_only_matched_controls(self) -> None:
        with TemporaryDirectory() as directory:
            artifact_root = Path(directory)
            for dataset in DATASETS:
                stem = artifact_root / f"{dataset}_holdout_wo_requirement"
                checkpoint = Path(f"{stem}_best.pt")
                summary = {
                    "evaluation_stage": "validation",
                    "evidence_representation_mode": "calibrated_history",
                    "target_requirement_mode": "factorized_item_control",
                    "seed": 42,
                    "test_metrics": None,
                    "valid_interactions": f"/data/{dataset}/valid.csv",
                    "test_interactions": f"/data/{dataset}/valid.csv",
                    "best_checkpoint_path": str(checkpoint),
                    "architecture_fingerprint": "architecture",
                    "ablation_variant_fingerprint": "variant",
                    "initialization_hash": f"init-{dataset}",
                }
                stem.with_suffix(".json").write_text(json.dumps(summary))
                checkpoint.write_bytes(b"checkpoint")
                Path(f"{stem}_predictions.csv").write_text("prob\n0.5\n")

            result = audit_history_gate_anchors(artifact_root)

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["anchors"]), 4)
        self.assertEqual(result["architecture_fingerprint"], "architecture")

    def test_anchor_audit_rejects_test_metrics(self) -> None:
        with TemporaryDirectory() as directory:
            artifact_root = Path(directory)
            for dataset in DATASETS:
                stem = artifact_root / f"{dataset}_holdout_wo_requirement"
                summary = {
                    "evaluation_stage": "validation",
                    "evidence_representation_mode": "calibrated_history",
                    "target_requirement_mode": "factorized_item_control",
                    "seed": 42,
                    "test_metrics": {} if dataset == "assist17" else None,
                    "valid_interactions": f"/data/{dataset}/valid.csv",
                    "test_interactions": f"/data/{dataset}/valid.csv",
                    "best_checkpoint_path": f"{stem}_best.pt",
                    "architecture_fingerprint": "architecture",
                    "ablation_variant_fingerprint": "variant",
                    "initialization_hash": f"init-{dataset}",
                }
                stem.with_suffix(".json").write_text(json.dumps(summary))
                Path(f"{stem}_best.pt").write_bytes(b"checkpoint")
                Path(f"{stem}_predictions.csv").write_text("prob\n0.5\n")

            with self.assertRaisesRegex(ValueError, "test metrics"):
                audit_history_gate_anchors(artifact_root)


if __name__ == "__main__":
    unittest.main()
