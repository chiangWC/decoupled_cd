from __future__ import annotations

import json
import hashlib
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

    def test_finalization_replays_proofs_and_selects_xes_r0_over_failed_r1(self) -> None:
        from scripts.run_unified_validation import RECIPES
        from scripts.unified_a0_exploration import finalize_exploration

        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            proofs_dir = state_dir / "proofs"
            proofs_dir.mkdir()
            state = {
                "schema_version": 3,
                "mode": "a0_exploration",
                "complete": True,
                "active_pair": None,
                "dataset_ids": ["XES3G5M"],
                "issuance_counter": 2,
                "architecture_fingerprint": "a" * 64,
                "cohort_sha256": "b" * 64,
                "controller_id": "c" * 64,
                "route_commit": "d" * 40,
                "baseline_sha256": "e" * 64,
                "selected_recipes": {
                    "XES3G5M": {
                        "recipe_index": 1,
                        "numerical_recipe": RECIPES["XES3G5M"][1].__dict__,
                    }
                },
            }
            (state_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")

            for counter, recipe_index, aucs in (
                (1, 0, (0.78, 0.77, 0.76)),
                (2, 1, (0.58, 0.54, 0.50)),
            ):
                standard_auc, holdout_auc, zero_auc = aucs
                recipe = RECIPES["XES3G5M"][recipe_index].__dict__
                proof = {
                    "schema_version": 3,
                    "controller_id": state["controller_id"],
                    "route_commit": state["route_commit"],
                    "counter": counter,
                    "dataset_id": "XES3G5M",
                    "baseline_sha256": state["baseline_sha256"],
                    "deltas": {
                        "standard_overall_auc": standard_auc - 0.79,
                        "holdout_overall_auc": holdout_auc - 0.78,
                        "zero_auc": zero_auc - 0.77,
                    },
                    "split_proofs": {
                        "standard": {
                            "recipe_index": recipe_index,
                            "numerical_recipe": recipe,
                            "metrics": {
                                "overall_auc": standard_auc,
                                "ordinary_doa": 0.5,
                                "weighted_doa": 0.6,
                            },
                        },
                        "holdout": {
                            "recipe_index": recipe_index,
                            "numerical_recipe": recipe,
                            "metrics": {
                                "overall_auc": holdout_auc,
                                "zero_auc": zero_auc,
                            },
                        },
                    },
                }
                (proofs_dir / f"{counter:06d}-XES3G5M.json").write_text(
                    json.dumps(proof), encoding="utf-8"
                )

            rows = finalize_exploration(state_dir)
            finalized = json.loads((state_dir / "state.json").read_text())
            selected = finalized["selected_recipes"]["XES3G5M"]

            self.assertEqual(selected["recipe_index"], 0)
            self.assertEqual(rows[0]["recipe_index"], 0)
            self.assertEqual(selected["recipe_sha256"], rows[0]["recipe_sha256"])
            self.assertEqual(selected["proof_counter"], 1)
            proof_bytes = (proofs_dir / "000001-XES3G5M.json").read_bytes()
            self.assertEqual(selected["proof_sha256"], hashlib.sha256(proof_bytes).hexdigest())


if __name__ == "__main__":
    unittest.main()
