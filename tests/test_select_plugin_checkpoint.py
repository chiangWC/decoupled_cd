from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.plugin_campaign import compute_frozen_config_id
from scripts.select_plugin_checkpoint import SelectionError, select_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTOR = PROJECT_ROOT / "scripts" / "select_plugin_checkpoint.py"


class SelectPluginCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "candidates").mkdir()
        self.manifest = self.root / "candidates" / "manifest.jsonl"
        self.doa_csv = self.root / "valid_doa.csv"
        self.artifact_hashes: dict[str, dict[str, str]] = {}
        self.backbone_config = {
            "latent_dim": 32,
            "gcn_layers": 2,
            "keep_prob": 0.9,
            "if_type": "ncd",
            "mode": "all",
            "flip_ratio": 0.1,
            "ssl_temp": 0.5,
            "ssl_weight": 0.01,
            "prednet_len1": 512,
            "prednet_len2": 256,
            "dropout": 0.2,
        }
        self.protocol = {
            "split": "valid",
            "seed": 42,
            "doa_seed": 42,
            "min_responses": 3,
            "max_pairs_per_concept": 100_000,
            "split_seed": 2024,
            "q_matrix_sha256": hashlib.sha256(b"q-matrix").hexdigest(),
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_candidates(self, rows: list[dict[str, object]]) -> None:
        manifest_rows = []
        for row in rows:
            epoch = int(row["epoch"])
            candidate_dir = self.root / "candidates" / f"epoch-{epoch:03d}"
            candidate_dir.mkdir()
            checkpoint = candidate_dir / "checkpoint.pth"
            checkpoint.write_bytes(f"checkpoint-{epoch}".encode())
            mastery = candidate_dir / "mastery.npy"
            mastery.write_bytes(f"mastery-{epoch}".encode())
            id_maps = candidate_dir / "id_maps.json"
            id_maps.write_text(json.dumps({"epoch": epoch}), encoding="utf-8")
            self.artifact_hashes[str(row["model_name"])] = {
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "mastery_sha256": hashlib.sha256(mastery.read_bytes()).hexdigest(),
                "id_maps_sha256": hashlib.sha256(id_maps.read_bytes()).hexdigest(),
            }
            hashes = self.artifact_hashes[str(row["model_name"])]
            manifest_rows.append({
                "epoch": epoch,
                "model_name": row["model_name"],
                "checkpoint_path": f"candidates/epoch-{epoch:03d}/checkpoint.pth",
                "mastery_path": f"candidates/epoch-{epoch:03d}/mastery.npy",
                "id_maps_path": f"candidates/epoch-{epoch:03d}/id_maps.json",
                "validation": row["validation"],
                "plugin_config": row["plugin_config"],
                "backbone_config": row.get("backbone_config", self.backbone_config),
                "protocol": row.get("protocol", self.protocol),
                **hashes,
                "test_metrics": {"auc": 1.0},
            })
        self.manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in manifest_rows),
            encoding="utf-8",
        )

    def write_doa(
        self,
        rows: list[dict[str, object]],
        *,
        include_audit: bool = True,
    ) -> None:
        with self.doa_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "model",
                    "holdout_doa",
                    "holdout_doa_weighted",
                    "holdout_num_concepts_evaluated",
                    "holdout_num_pairs",
                    "split",
                    "doa_seed",
                    "min_responses",
                    "max_pairs_per_concept",
                    "split_seed",
                    "mastery_sha256",
                    "id_maps_sha256",
                ],
            )
            writer.writeheader()
            audited_rows = []
            for row in rows:
                artifact_hashes = self.artifact_hashes.get(str(row["model"]), {})
                audit = {
                    field: artifact_hashes[field]
                    for field in ("mastery_sha256", "id_maps_sha256")
                    if field in artifact_hashes
                }
                audited_rows.append({
                    "holdout_num_concepts_evaluated": 2,
                    "holdout_num_pairs": 10,
                    **({
                        "split": "valid",
                        "doa_seed": 42,
                        "min_responses": 3,
                        "max_pairs_per_concept": 100_000,
                        "split_seed": 2024,
                        **audit,
                    } if include_audit else {}),
                    **row,
                })
            writer.writerows(audited_rows)

    def test_baseline_selects_highest_validation_auc_then_earliest_epoch_at_lambda_zero(self) -> None:
        self.write_candidates([
            {"epoch": 1, "model_name": "base-e1", "validation": {"auc": 0.80, "acc": 0.7, "rmse": 0.4}, "plugin_config": {"aux_weight": 0.0}},
            {"epoch": 2, "model_name": "base-e2", "validation": {"auc": 0.82, "acc": 0.7, "rmse": 0.4}, "plugin_config": {"aux_weight": 0.0}},
            {"epoch": 3, "model_name": "base-e3", "validation": {"auc": 0.82, "acc": 0.8, "rmse": 0.3}, "plugin_config": {"aux_weight": 0.0}},
            {"epoch": 4, "model_name": "not-base", "validation": {"auc": 0.99, "acc": 0.9, "rmse": 0.2}, "plugin_config": {"aux_weight": 0.5}},
        ])
        self.write_doa([
            {"model": "base-e1", "holdout_doa": 0.50, "holdout_doa_weighted": 0.51},
            {"model": "base-e2", "holdout_doa": 0.52, "holdout_doa_weighted": 0.53},
            {"model": "base-e3", "holdout_doa": 0.54, "holdout_doa_weighted": 0.55},
            {"model": "not-base", "holdout_doa": 0.90, "holdout_doa_weighted": 0.90},
        ])
        output = self.root / "baseline-selection.json"

        selected = select_checkpoint(
            mode="baseline",
            manifest_path=self.manifest,
            doa_csv_path=self.doa_csv,
            output_path=output,
        )

        self.assertEqual(selected["epoch"], 2)
        self.assertEqual(selected["validation"]["auc"], 0.82)
        self.assertEqual(selected["validation_doa"]["holdout_doa_weighted"], 0.53)
        self.assertEqual(selected["constraint_deltas"]["auc"], 0.0)
        self.assertNotIn("test_metrics", json.dumps(selected))
        checkpoint = self.root / selected["checkpoint_path"]
        self.assertEqual(
            selected["checkpoint_sha256"],
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            selected["id_maps_sha256"],
            self.artifact_hashes[selected["model_name"]]["id_maps_sha256"],
        )
        self.assertEqual(selected["backbone_config"], self.backbone_config)
        self.assertEqual(json.loads(output.read_text()), selected)

    def test_plugin_filters_constraints_then_ranks_doa_auc_and_earlier_epoch(self) -> None:
        baseline = self.root / "baseline-selection.json"
        baseline.write_text(json.dumps({
            "validation": {"auc": 0.90},
            "validation_doa": {"holdout_doa": 0.50, "holdout_doa_weighted": 0.55},
            "protocol": self.protocol,
        }))
        self.write_candidates([
            {"epoch": 1, "model_name": "low-auc", "validation": {"auc": 0.897, "acc": 0.7, "rmse": 0.4}, "plugin_config": {"aux_weight": 0.1}},
            {"epoch": 2, "model_name": "low-weighted", "validation": {"auc": 0.91, "acc": 0.7, "rmse": 0.4}, "plugin_config": {"aux_weight": 0.1}},
            {"epoch": 3, "model_name": "winner-early", "validation": {"auc": 0.91, "acc": 0.7, "rmse": 0.4}, "plugin_config": {"aux_weight": 0.1}},
            {"epoch": 4, "model_name": "winner-late", "validation": {"auc": 0.91, "acc": 0.8, "rmse": 0.3}, "plugin_config": {"aux_weight": 0.2}},
            {"epoch": 5, "model_name": "lower-doa", "validation": {"auc": 0.95, "acc": 0.9, "rmse": 0.2}, "plugin_config": {"aux_weight": 0.3}},
        ])
        self.write_doa([
            {"model": "low-auc", "holdout_doa": 0.99, "holdout_doa_weighted": 0.80},
            {"model": "low-weighted", "holdout_doa": 0.99, "holdout_doa_weighted": 0.54},
            {"model": "winner-early", "holdout_doa": 0.70, "holdout_doa_weighted": 0.56},
            {"model": "winner-late", "holdout_doa": 0.70, "holdout_doa_weighted": 0.57},
            {"model": "lower-doa", "holdout_doa": 0.69, "holdout_doa_weighted": 0.80},
        ])

        selected = select_checkpoint(
            mode="plugin",
            manifest_path=self.manifest,
            doa_csv_path=self.doa_csv,
            output_path=self.root / "plugin-selection.json",
            baseline_selection_path=baseline,
        )

        self.assertEqual(selected["model_name"], "winner-early")
        self.assertAlmostEqual(selected["constraint_deltas"]["auc"], 0.01)
        self.assertAlmostEqual(
            selected["constraint_deltas"]["holdout_doa_weighted"],
            0.01,
        )
        self.assertEqual(selected["constraints"]["auc_tolerance"], 0.002)
        self.assertRegex(selected["frozen_config_id"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            selected["frozen_config_id"],
            compute_frozen_config_id(
                checkpoint_sha256=selected["checkpoint_sha256"],
                id_maps_sha256=selected["id_maps_sha256"],
                plugin_config=selected["plugin_config"],
                backbone_config=selected["backbone_config"],
                protocol=selected["protocol"],
            ),
        )

    def test_plugin_without_feasible_candidate_exits_nonzero(self) -> None:
        baseline = self.root / "baseline-selection.json"
        baseline.write_text(json.dumps({
            "validation": {"auc": 0.90},
            "validation_doa": {"holdout_doa": 0.50, "holdout_doa_weighted": 0.55},
            "protocol": self.protocol,
        }))
        self.write_candidates([
            {"epoch": 1, "model_name": "infeasible", "validation": {"auc": 0.80, "acc": 0.7, "rmse": 0.4}, "plugin_config": {"aux_weight": 0.1}},
        ])
        self.write_doa([
            {"model": "infeasible", "holdout_doa": 0.90, "holdout_doa_weighted": 0.90},
        ])
        output = self.root / "must-not-exist.json"

        with self.assertRaises(SelectionError):
            select_checkpoint(
                mode="plugin",
                manifest_path=self.manifest,
                doa_csv_path=self.doa_csv,
                output_path=output,
                baseline_selection_path=baseline,
            )
        completed = subprocess.run(
            [
                sys.executable,
                str(SELECTOR),
                "--mode", "plugin",
                "--manifest", str(self.manifest),
                "--valid-doa-csv", str(self.doa_csv),
                "--baseline-selection", str(baseline),
                "--output-json", str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(output.exists())

    def test_duplicate_model_names_are_rejected(self) -> None:
        self.write_candidates([
            {"epoch": 1, "model_name": "duplicate", "validation": {"auc": 0.80, "acc": 0.7, "rmse": 0.4}, "plugin_config": {"aux_weight": 0.0}},
            {"epoch": 2, "model_name": "duplicate", "validation": {"auc": 0.81, "acc": 0.7, "rmse": 0.4}, "plugin_config": {"aux_weight": 0.0}},
        ])
        self.write_doa([
            {"model": "duplicate", "holdout_doa": 0.50, "holdout_doa_weighted": 0.51},
        ])

        with self.assertRaisesRegex(SelectionError, "duplicate"):
            select_checkpoint(
                mode="baseline",
                manifest_path=self.manifest,
                doa_csv_path=self.doa_csv,
                output_path=self.root / "selection.json",
            )

    def test_nonfinite_or_empty_holdout_metrics_are_rejected(self) -> None:
        self.write_candidates([
            {"epoch": 1, "model_name": "invalid-doa", "validation": {"auc": 0.80, "acc": 0.7, "rmse": 0.4}, "plugin_config": {"aux_weight": 0.0}},
        ])
        self.write_doa([
            {
                "model": "invalid-doa",
                "holdout_doa": "nan",
                "holdout_doa_weighted": 0.51,
                "holdout_num_concepts_evaluated": 0,
                "holdout_num_pairs": 0,
            },
        ])

        with self.assertRaises(SelectionError):
            select_checkpoint(
                mode="baseline",
                manifest_path=self.manifest,
                doa_csv_path=self.doa_csv,
                output_path=self.root / "selection.json",
            )

    def test_non_valid_manifest_protocol_is_rejected(self) -> None:
        invalid_protocol = {**self.protocol, "split": "test"}
        self.write_candidates([
            {"epoch": 1, "model_name": "wrong-split", "validation": {"auc": 0.80, "acc": 0.7, "rmse": 0.4}, "plugin_config": {"aux_weight": 0.0}, "protocol": invalid_protocol},
        ])
        self.write_doa([
            {"model": "wrong-split", "holdout_doa": 0.50, "holdout_doa_weighted": 0.51},
        ])

        with self.assertRaisesRegex(SelectionError, "valid"):
            select_checkpoint(
                mode="baseline",
                manifest_path=self.manifest,
                doa_csv_path=self.doa_csv,
                output_path=self.root / "selection.json",
            )

    def test_manifest_protocol_rejects_unapproved_seed_values(self) -> None:
        wrong_seed = {**self.protocol, "seed": 7}
        self.write_candidates([
            {"epoch": 1, "model_name": "wrong-seed", "validation": {"auc": 0.80, "acc": 0.7, "rmse": 0.4}, "plugin_config": {"aux_weight": 0.0}, "protocol": wrong_seed},
        ])
        self.write_doa([
            {"model": "wrong-seed", "holdout_doa": 0.50, "holdout_doa_weighted": 0.51},
        ])

        with self.assertRaisesRegex(SelectionError, "seed"):
            select_checkpoint(
                mode="baseline",
                manifest_path=self.manifest,
                doa_csv_path=self.doa_csv,
                output_path=self.root / "selection.json",
            )

    def test_doa_csv_rejects_missing_or_wrong_protocol_and_artifact_binding(self) -> None:
        self.write_candidates([
            {"epoch": 1, "model_name": "candidate", "validation": {"auc": 0.80, "acc": 0.7, "rmse": 0.4}, "plugin_config": {"aux_weight": 0.0}},
        ])
        base_row = {
            "model": "candidate",
            "holdout_doa": 0.50,
            "holdout_doa_weighted": 0.51,
        }
        cases = (
            ("missing audit", {}, False),
            ("wrong split", {"split": "test"}, True),
            ("wrong doa seed", {"doa_seed": 2024}, True),
            ("wrong minimum", {"min_responses": 1}, True),
            ("wrong max pairs", {"max_pairs_per_concept": 99}, True),
            ("wrong split seed", {"split_seed": 7}, True),
            ("wrong mastery", {"mastery_sha256": "0" * 64}, True),
            ("wrong id maps", {"id_maps_sha256": "0" * 64}, True),
        )

        for label, override, include_audit in cases:
            with self.subTest(label=label):
                self.write_doa(
                    [{**base_row, **override}],
                    include_audit=include_audit,
                )
                with self.assertRaises(SelectionError):
                    select_checkpoint(
                        mode="baseline",
                        manifest_path=self.manifest,
                        doa_csv_path=self.doa_csv,
                        output_path=self.root / "selection.json",
                    )

    def test_manifest_artifact_hashes_are_checked_against_actual_files(self) -> None:
        self.write_candidates([
            {
                "epoch": 1,
                "model_name": "candidate",
                "validation": {"auc": 0.80, "acc": 0.7, "rmse": 0.4},
                "plugin_config": {"aux_weight": 0.0},
            },
        ])
        self.write_doa([
            {
                "model": "candidate",
                "holdout_doa": 0.50,
                "holdout_doa_weighted": 0.51,
            },
        ])
        rows = [json.loads(line) for line in self.manifest.read_text().splitlines()]
        rows[0]["checkpoint_sha256"] = "0" * 64
        self.manifest.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(SelectionError, "checkpoint_sha256"):
            select_checkpoint(
                mode="baseline",
                manifest_path=self.manifest,
                doa_csv_path=self.doa_csv,
                output_path=self.root / "selection.json",
            )


if __name__ == "__main__":
    unittest.main()
