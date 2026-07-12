from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.unified_baseline_audit import audit_baseline_sources
from scripts.unified_dataset_audit import canonical_sha256


class UnifiedBaselineAdapterTests(unittest.TestCase):
    def test_manifest_binds_exact_model_configuration(self) -> None:
        from scripts.unified_baseline_adapter import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            valid = root / "valid.csv"
            train.touch()
            valid.touch()
            output = root / "manifest.json"
            main([
                "manifest", "--model", "ORCDF", "--dataset-id", "ASSIST17",
                "--split-id", "standard", "--train-file", str(train),
                "--valid-file", str(valid), "--config", "epochs=8",
                "--config", "latent_dim=32", "--output", str(output),
            ])
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["configuration"], {"epochs": "8", "latent_dim": "32"}
        )

    def test_internal_evaluator_exposes_bound_prediction_output(self) -> None:
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "evaluate_coverage_slice.py"),
                "--help",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--prediction-output", completed.stdout)

    def test_finalizer_binds_validation_order_and_real_artifact_hashes(self) -> None:
        from scripts.unified_baseline_adapter import finalize_baseline_rows

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            valid = root / "valid.csv"
            q_path = root / "Q_matrix.csv"
            train.write_text(
                "stu_id,exer_id,cpt_seq,label\n0,0,0,1\n1,1,1,0\n",
                encoding="utf-8",
            )
            valid.write_text(
                "stu_id,exer_id,cpt_seq,label\n0,1,1,0\n1,0,0,1\n",
                encoding="utf-8",
            )
            q_path.write_text("exer_id,cpt_seq\n0,0\n1,1\n", encoding="utf-8")
            predictions = root / "predictions.csv"
            predictions.write_text("prob,label\n0.2,0\n0.8,1\n", encoding="utf-8")
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            manifest = root / "job-manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": 1,
                "model": "KaNCD",
                "dataset_id": "ASSIST17",
                "split": "holdout",
                "seed": 42,
                "split_seed": 2024,
                "evaluation_role": "validation_alias",
                "train_file": str(train),
                "valid_file": str(valid),
                "test_file": str(valid),
            }), encoding="utf-8")
            split = {
                "path": str(root),
                "train_path": str(train),
                "valid_path": str(valid),
                "q_path": str(q_path),
                "data_sha256": "a" * 64,
                "q_sha256": "b" * 64,
                "prediction_order_sha256": "c" * 64,
            }
            record = {
                "dataset_id": "ASSIST17",
                "eligible": True,
                "standard": split,
                "holdout": split,
            }
            record["audit_sha256"] = canonical_sha256(record)
            audit = {
                "schema_version": 1,
                "datasets": {"ASSIST17": record},
                "eligible_dataset_ids": ["ASSIST17"],
            }
            audit["audit_sha256"] = canonical_sha256(audit)
            audit_path = root / "dataset-audit.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            output = root / "baseline-rows.json"
            bound = root / "bound-predictions.csv"

            rows = finalize_baseline_rows(
                dataset_id="ASSIST17",
                split_id="holdout",
                model="KaNCD",
                dataset_audit_path=audit_path,
                manifest_path=manifest,
                prediction_path=predictions,
                checkpoint_path=checkpoint,
                bound_prediction_path=bound,
                output_path=output,
            )

            self.assertEqual([row["metric"] for row in rows], ["auc", "zero_auc"])
            self.assertTrue(bound.is_file())
            with bound.open(newline="", encoding="utf-8") as handle:
                bound_rows = list(csv.DictReader(handle))
            self.assertEqual(
                [(row["stu_id"], row["exer_id"]) for row in bound_rows],
                [("0", "1"), ("1", "0")],
            )
            accepted = audit_baseline_sources([output], audit)
            self.assertEqual(accepted["accepted_count"], 2)
            self.assertEqual(accepted["rejected_count"], 0)

    def test_finalizer_rejects_non_validation_alias_manifest(self) -> None:
        from scripts.unified_baseline_adapter import validate_job_manifest

        manifest = {
            "seed": 42,
            "split_seed": 2024,
            "evaluation_role": "test",
            "valid_file": "/data/valid.csv",
            "test_file": "/data/test.csv",
        }
        with self.assertRaisesRegex(ValueError, "validation_alias"):
            validate_job_manifest(manifest, valid_path=Path("/data/valid.csv"))


if __name__ == "__main__":
    unittest.main()
