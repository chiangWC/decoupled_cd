from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.plugin_campaign import compute_recipe_id, sha256_file
from scripts.select_joint_plugin_recipe import select_joint_recipe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTOR = PROJECT_ROOT / "scripts" / "select_joint_plugin_recipe.py"


class SelectJointPluginRecipeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.standard_root = self.root / "standard"
        self.holdout_root = self.root / "holdout"
        (self.standard_root / "candidates").mkdir(parents=True)
        (self.holdout_root / "candidates").mkdir(parents=True)
        self.standard_manifest = self.standard_root / "candidates" / "manifest.jsonl"
        self.holdout_manifest = self.holdout_root / "candidates" / "manifest.jsonl"
        self.holdout_doa = self.root / "holdout-valid-doa.csv"
        self.standard_baseline = self.root / "standard-baseline.json"
        self.holdout_baseline = self.root / "holdout-baseline.json"
        self.output_dir = self.root / "selection"
        self.backbone = {"latent_dim": 32, "gcn_layers": 2}
        self.recipe = {"aux_weight": 0.1, "decouple": True}
        self.holdout_assignments_sha256 = "9" * 64
        common_protocol = {
            "split": "valid",
            "seed": 42,
            "doa_seed": 42,
            "min_responses": 3,
            "max_pairs_per_concept": 100_000,
            "split_seed": 2024,
            "q_matrix_sha256": hashlib.sha256(b"q-matrix").hexdigest(),
            "dataset_name": "fixture",
        }
        self.standard_protocol = {
            **common_protocol,
            "data_protocol": "standard",
        }
        self.holdout_protocol = {
            **common_protocol,
            "data_protocol": "holdout",
        }
        self.standard_baseline.write_text(json.dumps({
            "validation": {"auc": 0.80},
            "protocol": self.standard_protocol,
            "dataset": "fixture",
        }), encoding="utf-8")
        self.holdout_baseline.write_text(json.dumps({
            "validation": {"auc": 0.80},
            "validation_doa": {
                "holdout_doa": 0.50,
                "holdout_doa_weighted": 0.60,
            },
            "protocol": self.holdout_protocol,
            "dataset": "fixture",
            "holdout_assignments_sha256": self.holdout_assignments_sha256,
        }), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_manifest(
        self,
        *,
        root: Path,
        manifest: Path,
        protocol: dict[str, object],
        rows: list[dict[str, object]],
    ) -> dict[str, dict[str, str]]:
        artifact_hashes: dict[str, dict[str, str]] = {}
        manifest_rows: list[dict[str, object]] = []
        for index, row in enumerate(rows, start=1):
            model_name = str(row.get("model_name", f"candidate-{index}"))
            candidate_dir = root / "candidates" / model_name
            candidate_dir.mkdir()
            checkpoint = candidate_dir / "checkpoint.pth"
            checkpoint.write_bytes(f"checkpoint-{root.name}-{model_name}".encode())
            mastery = candidate_dir / "mastery.npy"
            mastery.write_bytes(f"mastery-{root.name}-{model_name}".encode())
            id_maps = candidate_dir / "id_maps.json"
            id_maps.write_text(json.dumps({"model": model_name}), encoding="utf-8")
            recipe = dict(row.get("recipe", self.recipe))
            backbone = dict(row.get("backbone", self.backbone))
            hashes = {
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "mastery_sha256": hashlib.sha256(mastery.read_bytes()).hexdigest(),
                "id_maps_sha256": hashlib.sha256(id_maps.read_bytes()).hexdigest(),
            }
            artifact_hashes[model_name] = hashes
            relative_dir = f"candidates/{model_name}"
            manifest_rows.append({
                "epoch": int(row.get("epoch", index)),
                "model_name": model_name,
                "checkpoint_path": f"{relative_dir}/checkpoint.pth",
                "mastery_path": f"{relative_dir}/mastery.npy",
                "id_maps_path": f"{relative_dir}/id_maps.json",
                **hashes,
                "validation": {
                    "auc": float(row["auc"]),
                    "acc": 0.7,
                    "rmse": 0.4,
                },
                "plugin_config": recipe,
                "recipe_config": recipe,
                "recipe_id": compute_recipe_id(recipe, backbone),
                "backbone_config": backbone,
                "protocol": protocol,
            })
        manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in manifest_rows),
            encoding="utf-8",
        )
        return artifact_hashes

    def write_fixture(
        self,
        *,
        standard_auc: float = 0.802,
        holdout_auc: float = 0.803,
        holdout_weighted_doa: float = 0.60,
        holdout_doa: float = 0.51,
        same_recipe: bool = True,
    ) -> None:
        self.write_manifest(
            root=self.standard_root,
            manifest=self.standard_manifest,
            protocol=self.standard_protocol,
            rows=[{"model_name": "standard-plugin", "auc": standard_auc}],
        )
        holdout_recipe = self.recipe if same_recipe else {
            **self.recipe,
            "aux_weight": 0.2,
        }
        hashes = self.write_manifest(
            root=self.holdout_root,
            manifest=self.holdout_manifest,
            protocol=self.holdout_protocol,
            rows=[{
                "model_name": "holdout-plugin",
                "auc": holdout_auc,
                "recipe": holdout_recipe,
            }],
        )
        artifact_hashes = hashes["holdout-plugin"]
        with self.holdout_doa.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
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
                "dataset",
                "holdout_assignments_sha256",
            ])
            writer.writeheader()
            writer.writerow({
                "model": "holdout-plugin",
                "holdout_doa": holdout_doa,
                "holdout_doa_weighted": holdout_weighted_doa,
                "holdout_num_concepts_evaluated": 2,
                "holdout_num_pairs": 10,
                "split": "valid",
                "doa_seed": 42,
                "min_responses": 3,
                "max_pairs_per_concept": 100_000,
                "split_seed": 2024,
                "mastery_sha256": artifact_hashes["mastery_sha256"],
                "id_maps_sha256": artifact_hashes["id_maps_sha256"],
                "dataset": "fixture",
                "holdout_assignments_sha256": self.holdout_assignments_sha256,
            })

    def select(self, *, margin: float = 0.001) -> dict[str, object]:
        return select_joint_recipe(
            standard_manifest_path=self.standard_manifest,
            holdout_manifest_path=self.holdout_manifest,
            holdout_doa_csv_path=self.holdout_doa,
            standard_baseline_path=self.standard_baseline,
            holdout_baseline_path=self.holdout_baseline,
            output_dir=self.output_dir,
            auc_safety_margin=margin,
        )

    def assert_needs_adapter_without_children(self) -> dict[str, object]:
        result = self.select()
        self.assertEqual(result["decision"], "needs_adapter")
        self.assertTrue((self.output_dir / "joint_diagnostics.json").is_file())
        for name in (
            "joint_selection.json",
            "standard_selection.json",
            "holdout_selection.json",
        ):
            self.assertFalse((self.output_dir / name).exists())
        return result

    def test_joint_selector_requires_same_recipe_and_safety_margin(self) -> None:
        self.write_fixture()

        result = self.select()

        self.assertEqual(result["decision"], "shared")
        self.assertGreaterEqual(result["min_validation_auc_delta"], 0.001)
        self.assertGreater(result["holdout_validation_doa_delta"], 0)
        self.assertEqual(result["recipe_id"], compute_recipe_id(self.recipe, self.backbone))
        for name in (
            "joint_selection.json",
            "standard_selection.json",
            "holdout_selection.json",
        ):
            self.assertTrue((self.output_dir / name).is_file())
        self.assertNotEqual(
            result["standard_frozen_config_id"],
            result["holdout_frozen_config_id"],
        )
        self.assertEqual(
            result["input_sha256"]["standard_manifest"],
            sha256_file(self.standard_manifest),
        )
        self.assertEqual(
            result["input_sha256"]["holdout_manifest"],
            sha256_file(self.holdout_manifest),
        )
        self.assertEqual(
            result["input_sha256"]["holdout_doa_csv"],
            sha256_file(self.holdout_doa),
        )

    def test_negative_standard_auc_fails_hard_gate(self) -> None:
        self.write_fixture(standard_auc=0.799)
        result = self.assert_needs_adapter_without_children()
        self.assertEqual(result["hard_feasible_count"], 0)

    def test_negative_holdout_auc_fails_hard_gate(self) -> None:
        self.write_fixture(holdout_auc=0.799)
        result = self.assert_needs_adapter_without_children()
        self.assertEqual(result["hard_feasible_count"], 0)

    def test_negative_weighted_doa_fails_hard_gate(self) -> None:
        self.write_fixture(holdout_weighted_doa=0.599)
        result = self.assert_needs_adapter_without_children()
        self.assertEqual(result["hard_feasible_count"], 0)

    def test_ordinary_doa_must_increase_strictly(self) -> None:
        self.write_fixture(holdout_doa=0.50)
        result = self.assert_needs_adapter_without_children()
        self.assertEqual(result["hard_feasible_count"], 0)

    def test_different_recipe_ids_cannot_form_joint_candidate(self) -> None:
        self.write_fixture(same_recipe=False)
        result = self.assert_needs_adapter_without_children()
        self.assertEqual(result["joint_candidate_count"], 0)

    def test_hard_feasible_candidate_below_margin_needs_adapter(self) -> None:
        self.write_fixture(standard_auc=0.8005, holdout_auc=0.802)

        result = self.assert_needs_adapter_without_children()

        self.assertEqual(result["hard_feasible_count"], 1)
        self.assertAlmostEqual(result["best_hard_feasible_min_auc_delta"], 0.0005)

    def test_needs_adapter_cli_exits_three(self) -> None:
        self.write_fixture(standard_auc=0.8005, holdout_auc=0.802)

        completed = subprocess.run(
            [
                sys.executable,
                str(SELECTOR),
                "--standard-manifest", str(self.standard_manifest),
                "--holdout-manifest", str(self.holdout_manifest),
                "--holdout-valid-doa-csv", str(self.holdout_doa),
                "--standard-baseline-selection", str(self.standard_baseline),
                "--holdout-baseline-selection", str(self.holdout_baseline),
                "--output-dir", str(self.output_dir),
                "--auc-safety-margin", "0.001",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 3)


if __name__ == "__main__":
    unittest.main()
