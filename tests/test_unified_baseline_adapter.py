from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.unified_baseline_audit import audit_baseline_sources
from scripts.unified_dataset_audit import (
    _split_record,
    canonical_sha256,
    canonical_split_source_hashes,
)


class UnifiedBaselineAdapterTests(unittest.TestCase):
    def _finalizer_fixture(self, root: Path) -> dict[str, Path | str]:
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
        (root / "test.csv").write_text(valid.read_text(encoding="utf-8"))
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
        split, _ = _split_record(root)
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
        return {
            "dataset_id": "ASSIST17",
            "split_id": "holdout",
            "model": "KaNCD",
            "dataset_audit_path": audit_path,
            "manifest_path": manifest,
            "prediction_path": predictions,
            "checkpoint_path": checkpoint,
            "bound_prediction_path": root / "bound-predictions.csv",
            "output_path": root / "baseline-rows.json",
        }

    def test_adapter_direct_entrypoint_imports_from_project_root(self) -> None:
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "unified_baseline_adapter.py"),
                "--help",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("manifest", completed.stdout)
        self.assertIn("finalize", completed.stdout)

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
            arguments = self._finalizer_fixture(root)
            rows = finalize_baseline_rows(**arguments)

            self.assertEqual([row["metric"] for row in rows], ["auc", "zero_auc"])
            bound = Path(arguments["bound_prediction_path"])
            self.assertTrue(bound.is_file())
            with bound.open(newline="", encoding="utf-8") as handle:
                bound_rows = list(csv.DictReader(handle))
            self.assertEqual(
                [(row["stu_id"], row["exer_id"]) for row in bound_rows],
                [("0", "1"), ("1", "0")],
            )
            audit = json.loads(Path(arguments["dataset_audit_path"]).read_text())
            accepted = audit_baseline_sources([Path(arguments["output_path"])], audit)
            self.assertEqual(accepted["accepted_count"], 2)
            self.assertEqual(accepted["rejected_count"], 0)

    def test_finalizer_rejects_train_mutation_after_audit(self) -> None:
        from scripts.unified_baseline_adapter import finalize_baseline_rows

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = self._finalizer_fixture(root)
            (root / "train.csv").write_text(
                "stu_id,exer_id,cpt_seq,label\n0,0,0,0\n1,1,1,0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "data_sha256"):
                finalize_baseline_rows(**arguments)

    def test_finalizer_rejects_validation_content_mutation_after_audit(self) -> None:
        from scripts.unified_baseline_adapter import finalize_baseline_rows

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = self._finalizer_fixture(root)
            (root / "valid.csv").write_text(
                "stu_id,exer_id,cpt_seq,label\n0,1,0,0\n1,0,0,1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "data_sha256"):
                finalize_baseline_rows(**arguments)

    def test_finalizer_rejects_validation_order_mutation_after_audit(self) -> None:
        from scripts.unified_baseline_adapter import finalize_baseline_rows

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = self._finalizer_fixture(root)
            (root / "valid.csv").write_text(
                "stu_id,exer_id,cpt_seq,label\n1,0,0,0\n0,1,1,1\n",
                encoding="utf-8",
            )
            audit_path = Path(arguments["dataset_audit_path"])
            audit = json.loads(audit_path.read_text())
            actual = canonical_split_source_hashes(
                train_path=root / "train.csv",
                valid_path=root / "valid.csv",
                q_path=root / "Q_matrix.csv",
            )
            record = audit["datasets"]["ASSIST17"]
            for split_id in ("standard", "holdout"):
                record[split_id]["data_sha256"] = actual["data_sha256"]
            record["audit_sha256"] = canonical_sha256(
                {key: value for key, value in record.items() if key != "audit_sha256"}
            )
            audit["audit_sha256"] = canonical_sha256(
                {key: value for key, value in audit.items() if key != "audit_sha256"}
            )
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prediction_order_sha256"):
                finalize_baseline_rows(**arguments)

    def test_finalizer_rejects_q_mutation_after_audit(self) -> None:
        from scripts.unified_baseline_adapter import finalize_baseline_rows

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = self._finalizer_fixture(root)
            (root / "Q_matrix.csv").write_text(
                "exer_id,cpt_seq\n0,0\n1,0\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "q_sha256"):
                finalize_baseline_rows(**arguments)

    def test_finalizer_rejects_fabricated_audit_hashes(self) -> None:
        from scripts.unified_baseline_adapter import finalize_baseline_rows

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = self._finalizer_fixture(root)
            audit_path = Path(arguments["dataset_audit_path"])
            audit = json.loads(audit_path.read_text())
            record = audit["datasets"]["ASSIST17"]
            for split_id in ("standard", "holdout"):
                record[split_id]["data_sha256"] = "a" * 64
                record[split_id]["q_sha256"] = "b" * 64
                record[split_id]["prediction_order_sha256"] = "c" * 64
            record["audit_sha256"] = canonical_sha256(
                {key: value for key, value in record.items() if key != "audit_sha256"}
            )
            audit["audit_sha256"] = canonical_sha256(
                {key: value for key, value in audit.items() if key != "audit_sha256"}
            )
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                finalize_baseline_rows(**arguments)

    def test_finalizer_rejects_validation_alias_resolving_elsewhere(self) -> None:
        from scripts.unified_baseline_adapter import finalize_baseline_rows

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = self._finalizer_fixture(root)
            manifest_path = Path(arguments["manifest_path"])
            manifest = json.loads(manifest_path.read_text())
            other = root / "other-valid.csv"
            other.write_text((root / "valid.csv").read_text(), encoding="utf-8")
            manifest["test_file"] = str(other)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation_alias"):
                finalize_baseline_rows(**arguments)

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
