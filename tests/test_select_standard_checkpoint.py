from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.plugin_campaign import compute_frozen_config_id, compute_recipe_id
from scripts.select_plugin_checkpoint import (
    SelectionError,
    load_candidate_manifest,
    validate_manifest_protocol,
)
from scripts.select_standard_checkpoint import select_standard_checkpoint


class SelectStandardCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "candidates").mkdir()
        self.manifest = self.root / "candidates" / "manifest.jsonl"
        self.output = self.root / "selection.json"
        self.backbone = {"latent_dim": 32, "gcn_layers": 2}
        self.protocol = {
            "split": "valid",
            "seed": 42,
            "doa_seed": 42,
            "min_responses": 3,
            "max_pairs_per_concept": 100_000,
            "split_seed": 2024,
            "q_matrix_sha256": hashlib.sha256(b"q-matrix").hexdigest(),
            "data_protocol": "standard",
            "dataset_name": "fixture",
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_candidates(self, rows: list[dict[str, object]]) -> None:
        manifest_rows: list[dict[str, object]] = []
        for index, row in enumerate(rows, start=1):
            epoch = int(row["epoch"])
            model_name = str(row.get("model_name", f"candidate-{index}"))
            candidate_dir = self.root / "candidates" / model_name
            candidate_dir.mkdir()
            checkpoint = candidate_dir / "checkpoint.pth"
            checkpoint.write_bytes(f"checkpoint-{model_name}".encode())
            mastery = candidate_dir / "mastery.npy"
            mastery.write_bytes(f"mastery-{model_name}".encode())
            id_maps = candidate_dir / "id_maps.json"
            id_maps.write_text(json.dumps({"model": model_name}), encoding="utf-8")
            plugin_config = dict(row["plugin_config"])
            recipe_config = dict(row.get("recipe_config", plugin_config))
            backbone = dict(row.get("backbone_config", self.backbone))
            protocol = dict(row.get("protocol", self.protocol))
            relative_dir = f"candidates/{model_name}"
            manifest_rows.append({
                "epoch": epoch,
                "model_name": model_name,
                "checkpoint_path": f"{relative_dir}/checkpoint.pth",
                "mastery_path": f"{relative_dir}/mastery.npy",
                "id_maps_path": f"{relative_dir}/id_maps.json",
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "mastery_sha256": hashlib.sha256(mastery.read_bytes()).hexdigest(),
                "id_maps_sha256": hashlib.sha256(id_maps.read_bytes()).hexdigest(),
                "validation": row["validation"],
                "plugin_config": plugin_config,
                "recipe_config": recipe_config,
                "recipe_id": compute_recipe_id(recipe_config, backbone),
                "backbone_config": backbone,
                "protocol": protocol,
                "test_metrics": {"auc": 1.0},
            })
        self.manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in manifest_rows),
            encoding="utf-8",
        )

    def test_standard_baseline_uses_auc_and_never_requires_doa(self) -> None:
        self.write_candidates([
            {
                "epoch": 2,
                "model_name": "winner-late",
                "validation": {"auc": 0.80, "acc": 0.7, "rmse": 0.4},
                "plugin_config": {"aux_weight": 0.0},
            },
            {
                "epoch": 1,
                "model_name": "winner-early",
                "validation": {"auc": 0.80, "acc": 0.6, "rmse": 0.5},
                "plugin_config": {"aux_weight": 0.0},
            },
            {
                "epoch": 3,
                "model_name": "plugin",
                "validation": {"auc": 0.99, "acc": 0.9, "rmse": 0.2},
                "plugin_config": {"aux_weight": 0.1},
            },
        ])

        selected = select_standard_checkpoint(
            mode="baseline",
            manifest_path=self.manifest,
            output_path=self.output,
        )

        self.assertEqual(selected["epoch"], 1)
        self.assertEqual(selected["protocol"]["data_protocol"], "standard")
        self.assertEqual(selected["validation"]["auc"], 0.80)
        self.assertNotIn("validation_doa", selected)
        self.assertNotIn("holdout_assignments_sha256", selected)
        self.assertNotIn("test_metrics", json.dumps(selected))
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
        self.assertEqual(json.loads(self.output.read_text()), selected)

    def test_standard_plugin_uses_zero_auc_tolerance(self) -> None:
        baseline = self.root / "baseline.json"
        baseline.write_text(json.dumps({
            "validation": {"auc": 0.80},
            "protocol": self.protocol,
            "dataset": "fixture",
        }), encoding="utf-8")
        self.write_candidates([{
            "epoch": 1,
            "model_name": "regressed",
            "validation": {"auc": 0.7999, "acc": 0.7, "rmse": 0.4},
            "plugin_config": {"aux_weight": 0.1},
        }])

        with self.assertRaisesRegex(SelectionError, "no standard plugin candidate"):
            select_standard_checkpoint(
                mode="plugin",
                manifest_path=self.manifest,
                baseline_selection_path=baseline,
                output_path=self.output,
            )
        self.assertFalse(self.output.exists())

    def test_standard_manifest_requires_one_dataset_standard_protocol_and_seed_42(self) -> None:
        cases = (
            (
                "standard",
                {**self.protocol, "data_protocol": "holdout"},
                {**self.protocol, "data_protocol": "holdout"},
            ),
            (
                "seed",
                {**self.protocol, "seed": 7},
                {**self.protocol, "seed": 7},
            ),
            (
                "mixed protocols",
                self.protocol,
                {**self.protocol, "dataset_name": "other-fixture"},
            ),
        )
        for case_number, (expected, first_protocol, second_protocol) in enumerate(
            cases,
            start=1,
        ):
            with self.subTest(expected=expected):
                self.write_candidates([
                    {
                        "epoch": 1,
                        "model_name": f"first-{case_number}",
                        "validation": {"auc": 0.80, "acc": 0.7, "rmse": 0.4},
                        "plugin_config": {"aux_weight": 0.0},
                        "protocol": first_protocol,
                    },
                    {
                        "epoch": 2,
                        "model_name": f"second-{case_number}",
                        "validation": {"auc": 0.81, "acc": 0.7, "rmse": 0.4},
                        "plugin_config": {"aux_weight": 0.0},
                        "protocol": second_protocol,
                    },
                ])
                with self.assertRaisesRegex(SelectionError, expected):
                    select_standard_checkpoint(
                        mode="baseline",
                        manifest_path=self.manifest,
                        output_path=self.output,
                    )

    def test_public_manifest_parser_accepts_legacy_holdout_protocol(self) -> None:
        legacy_protocol = {
            key: value
            for key, value in self.protocol.items()
            if key not in {"data_protocol", "dataset_name"}
        }
        self.write_candidates([{
            "epoch": 1,
            "model_name": "legacy",
            "validation": {"auc": 0.80, "acc": 0.7, "rmse": 0.4},
            "plugin_config": {"aux_weight": 0.0},
            "protocol": legacy_protocol,
        }])

        rows = load_candidate_manifest(self.manifest)
        validated = validate_manifest_protocol(rows)

        self.assertEqual(validated, legacy_protocol)


if __name__ == "__main__":
    unittest.main()
