from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.plugin_campaign import compute_recipe_id, sha256_file
from scripts.select_joint_plugin_recipe import select_joint_recipe
from scripts.select_plugin_checkpoint import SelectionError


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
            "selection_mode": "baseline",
            "validation": {"auc": 0.80},
            "protocol": self.standard_protocol,
            "dataset": "fixture",
            "plugin_config": {"aux_weight": 0.0},
        }), encoding="utf-8")
        self.holdout_baseline.write_text(json.dumps({
            "selection_mode": "baseline",
            "validation": {"auc": 0.80},
            "validation_doa": {
                "holdout_doa": 0.50,
                "holdout_doa_weighted": 0.60,
            },
            "protocol": self.holdout_protocol,
            "dataset": "fixture",
            "holdout_assignments_sha256": self.holdout_assignments_sha256,
            "plugin_config": {"aux_weight": 0.0},
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
        self.write_holdout_doa(hashes, [{
            "model": "holdout-plugin",
            "holdout_doa": holdout_doa,
            "holdout_doa_weighted": holdout_weighted_doa,
        }])

    def write_holdout_doa(
        self,
        artifact_hashes: dict[str, dict[str, str]],
        rows: list[dict[str, object]],
    ) -> None:
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
            for row in rows:
                model_name = str(row["model"])
                hashes = artifact_hashes[model_name]
                writer.writerow({
                    **row,
                    "holdout_num_concepts_evaluated": 2,
                    "holdout_num_pairs": 10,
                    "split": "valid",
                    "doa_seed": 42,
                    "min_responses": 3,
                    "max_pairs_per_concept": 100_000,
                    "split_seed": 2024,
                    "mastery_sha256": hashes["mastery_sha256"],
                    "id_maps_sha256": hashes["id_maps_sha256"],
                    "dataset": "fixture",
                    "holdout_assignments_sha256": self.holdout_assignments_sha256,
                })

    def select(
        self,
        *,
        margin: float = 0.001,
        output_dir: Path | None = None,
        standard_baseline: Path | None = None,
        holdout_baseline: Path | None = None,
    ) -> dict[str, object]:
        return select_joint_recipe(
            standard_manifest_path=self.standard_manifest,
            holdout_manifest_path=self.holdout_manifest,
            holdout_doa_csv_path=self.holdout_doa,
            standard_baseline_path=standard_baseline or self.standard_baseline,
            holdout_baseline_path=holdout_baseline or self.holdout_baseline,
            output_dir=output_dir or self.output_dir,
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
        standard_child = json.loads(
            (self.output_dir / "standard_selection.json").read_text()
        )
        holdout_child = json.loads(
            (self.output_dir / "holdout_selection.json").read_text()
        )
        self.assertEqual(
            set(standard_child["input_sha256"]),
            {"standard_manifest", "standard_baseline_selection"},
        )
        self.assertFalse(
            any(key.startswith("holdout_") for key in standard_child["input_sha256"])
        )
        self.assertEqual(
            set(holdout_child["input_sha256"]),
            {
                "holdout_manifest",
                "holdout_doa_csv",
                "holdout_baseline_selection",
            },
        )

    def test_joint_selector_rejects_non_baseline_or_nonzero_aux_reference(self) -> None:
        self.write_fixture()
        cases = (
            ("selection_mode", self.standard_baseline, "plugin", 0.0),
            ("aux_weight", self.holdout_baseline, "baseline", 0.1),
            ("finite", self.standard_baseline, "baseline", float("nan")),
        )
        for case_number, (expected, baseline, selection_mode, aux_weight) in enumerate(
            cases,
            start=1,
        ):
            with self.subTest(expected=expected):
                payload = json.loads(baseline.read_text())
                payload["selection_mode"] = selection_mode
                payload["plugin_config"]["aux_weight"] = aux_weight
                invalid_baseline = self.root / f"invalid-baseline-{case_number}.json"
                invalid_baseline.write_text(json.dumps(payload), encoding="utf-8")
                output_dir = self.root / f"invalid-selection-{case_number}"

                with self.assertRaisesRegex(SelectionError, expected):
                    self.select(
                        output_dir=output_dir,
                        standard_baseline=(
                            invalid_baseline
                            if baseline == self.standard_baseline
                            else self.standard_baseline
                        ),
                        holdout_baseline=(
                            invalid_baseline
                            if baseline == self.holdout_baseline
                            else self.holdout_baseline
                        ),
                    )
                self.assertFalse(output_dir.exists())

    def test_decision_directory_publish_failure_leaves_no_formal_output(self) -> None:
        self.write_fixture()

        with mock.patch(
            "scripts.select_joint_plugin_recipe._rename_directory_exclusive",
            create=True,
            side_effect=OSError("injected publish failure"),
        ):
            with self.assertRaisesRegex(SelectionError, "publish"):
                self.select()

        self.assertFalse(self.output_dir.exists())
        self.assertEqual(list(self.root.glob(".selection.*.tmp")), [])

    def test_decision_publish_does_not_replace_raced_empty_directory(self) -> None:
        self.write_fixture()
        real_lexists = os.path.lexists
        output_checks = 0

        def race_after_final_check(path: object) -> bool:
            nonlocal output_checks
            if Path(path) == self.output_dir:
                output_checks += 1
                if output_checks == 3:
                    self.output_dir.mkdir()
                    return False
            return real_lexists(path)

        with mock.patch(
            "scripts.select_joint_plugin_recipe.os.path.lexists",
            side_effect=race_after_final_check,
        ):
            with self.assertRaisesRegex(SelectionError, "already exists"):
                self.select()

        self.assertTrue(self.output_dir.is_dir())
        self.assertEqual(list(self.output_dir.iterdir()), [])
        self.assertEqual(list(self.root.glob(".selection.*.tmp")), [])

    def test_decision_directory_cannot_be_reused_across_decisions(self) -> None:
        self.write_fixture()
        shared_first = self.root / "shared-first"
        needs_adapter_first = self.root / "needs-adapter-first"

        self.select(output_dir=shared_first)
        with self.assertRaisesRegex(SelectionError, "already exists"):
            self.select(margin=0.01, output_dir=shared_first)
        self.assertFalse((shared_first / "joint_diagnostics.json").exists())

        self.select(margin=0.01, output_dir=needs_adapter_first)
        with self.assertRaisesRegex(SelectionError, "already exists"):
            self.select(output_dir=needs_adapter_first)
        for name in (
            "joint_selection.json",
            "standard_selection.json",
            "holdout_selection.json",
        ):
            self.assertFalse((needs_adapter_first / name).exists())

    def test_joint_sorting_uses_all_fixed_tiebreakers(self) -> None:
        low_min = {"aux_weight": 0.1, "variant": "low-min"}
        low_doa = {"aux_weight": 0.1, "variant": "low-doa"}
        winner = {"aux_weight": 0.1, "variant": "winner"}
        self.write_manifest(
            root=self.standard_root,
            manifest=self.standard_manifest,
            protocol=self.standard_protocol,
            rows=[
                {"model_name": "standard-low-min", "auc": 0.802, "recipe": low_min},
                {"model_name": "standard-low-doa", "auc": 0.803, "recipe": low_doa},
                {
                    "model_name": "standard-winner-late",
                    "epoch": 4,
                    "auc": 0.803,
                    "recipe": winner,
                },
                {
                    "model_name": "standard-winner-early",
                    "epoch": 1,
                    "auc": 0.803,
                    "recipe": winner,
                },
            ],
        )
        hashes = self.write_manifest(
            root=self.holdout_root,
            manifest=self.holdout_manifest,
            protocol=self.holdout_protocol,
            rows=[
                {"model_name": "holdout-low-min", "auc": 0.804, "recipe": low_min},
                {"model_name": "holdout-low-doa", "auc": 0.804, "recipe": low_doa},
                {
                    "model_name": "holdout-winner-late",
                    "epoch": 3,
                    "auc": 0.804,
                    "recipe": winner,
                },
                {
                    "model_name": "holdout-winner-early",
                    "epoch": 2,
                    "auc": 0.804,
                    "recipe": winner,
                },
            ],
        )
        self.write_holdout_doa(hashes, [
            {
                "model": "holdout-low-min",
                "holdout_doa": 0.59,
                "holdout_doa_weighted": 0.60,
            },
            {
                "model": "holdout-low-doa",
                "holdout_doa": 0.51,
                "holdout_doa_weighted": 0.60,
            },
            {
                "model": "holdout-winner-late",
                "holdout_doa": 0.52,
                "holdout_doa_weighted": 0.60,
            },
            {
                "model": "holdout-winner-early",
                "holdout_doa": 0.52,
                "holdout_doa_weighted": 0.60,
            },
        ])

        result = self.select()

        self.assertEqual(result["standard_model_name"], "standard-winner-early")
        self.assertEqual(result["holdout_model_name"], "holdout-winner-early")
        self.assertEqual(result["standard_epoch"], 1)
        self.assertEqual(result["holdout_epoch"], 2)

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
