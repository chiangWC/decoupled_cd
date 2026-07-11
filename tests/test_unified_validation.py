from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_unified_validation import (
    DATASET_DIRECTORIES,
    ELIGIBLE_DATASET_IDS,
    GpuSnapshot,
    _split_paths,
    _write_json,
    architecture_fingerprint,
    assemble_candidate_rows,
    build_evaluation_commands,
    build_train_command,
    can_reach_primary_cohort,
    locked_gpu,
    parse_gpu_inventory,
    select_gpu_index,
    validate_smoke_summary,
    _write_synthetic_fixture,
    main as validation_main,
)


class UnifiedValidationRunnerTests(unittest.TestCase):
    def test_assemble_rejects_tampered_cohort_before_loading_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort_path = root / "cohort.json"
            cohort_path.write_text(
                json.dumps(
                    {
                        "dataset_ids": ["ASSIST09", "ASSIST17", "MOOCRadar"],
                        "cohort_sha256": "0" * 64,
                    }
                ),
                encoding="utf-8",
            )
            output_path = root / "rows.json"

            with self.assertRaisesRegex(ValueError, "canonical SHA-256 mismatch"):
                validation_main(
                    [
                        "assemble",
                        "--cohort",
                        str(cohort_path),
                        "--split-summary",
                        str(root / "must-not-be-read.json"),
                        "--output",
                        str(output_path),
                    ]
                )
            self.assertFalse(output_path.exists())

    def test_synthetic_validation_targets_are_disjoint_from_history(self):
        with tempfile.TemporaryDirectory() as directory:
            train_path, valid_path, _ = _write_synthetic_fixture(Path(directory))

            def interaction_pairs(path: Path) -> set[tuple[str, str]]:
                rows = path.read_text(encoding="utf-8").splitlines()[1:]
                return {
                    tuple(row.split(",")[:2])
                    for row in rows
                }

            self.assertFalse(
                interaction_pairs(train_path) & interaction_pairs(valid_path)
            )

    def test_script_entrypoint_imports_from_project_root(self):
        child_environment = os.environ.copy()
        child_environment.pop("MKL_THREADING_LAYER", None)
        completed = subprocess.run(
            [sys.executable, "scripts/run_unified_validation.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
            env=child_environment,
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
                        "architecture_manifest": {
                            "composer": "mask",
                            "decoder": "neuralcdm-monotonic",
                            "inference": "graph",
                            "mastery_output": "student-concept",
                            "modules": "m1-m2-m4-neuralcdm",
                            "version": 2,
                        },
                        "cohort_sha256": cohort_hash,
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
        mismatched = [dict(summary) for summary in summaries]
        mismatched[0]["cohort_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "summary cohort SHA-256 mismatch"):
            assemble_candidate_rows(
                mismatched,
                cohort_dataset_ids=["ASSIST09", "ASSIST17", "MOOCRadar"],
                cohort_sha256=cohort_hash,
            )
        with self.assertRaisesRegex(ValueError, "missing validation split"):
            assemble_candidate_rows(
                summaries[:-1],
                cohort_dataset_ids=["ASSIST09", "ASSIST17", "MOOCRadar"],
                cohort_sha256=cohort_hash,
            )

    def test_candidate_stops_when_three_successes_are_unreachable(self):
        self.assertTrue(can_reach_primary_cohort(successes=1, remaining=2))
        self.assertFalse(can_reach_primary_cohort(successes=1, remaining=1))

    def test_run_split_requires_capability_before_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_path = root / "attempt" / "validation-summary.json"
            with self.assertRaises(SystemExit):
                validation_main(
                    [
                        "run-split",
                        "--dataset-id",
                        "ASSIST09",
                        "--split-id",
                        "standard",
                        "--architecture",
                        "m2",
                        "--data-root",
                        str(root / "missing-data"),
                        "--device",
                        "cpu",
                        "--output",
                        str(output_path),
                    ]
                )
            self.assertFalse(output_path.parent.exists())

    def test_all_dataset_split_paths_map_to_expected_assets(self):
        root = Path("/datasets")
        self.assertEqual(set(DATASET_DIRECTORIES), set(ELIGIBLE_DATASET_IDS))
        for dataset_id, (standard_dir, holdout_dir) in DATASET_DIRECTORIES.items():
            for split_id, expected_dir in (
                ("standard", standard_dir),
                ("holdout", holdout_dir),
            ):
                train, valid, q_matrix, assignments = _split_paths(
                    dataset_id=dataset_id,
                    split_id=split_id,
                    data_root=root,
                )
                self.assertEqual(train, root / expected_dir / "train.csv")
                self.assertEqual(valid, root / expected_dir / "valid.csv")
                self.assertEqual(q_matrix, root / expected_dir / "Q_matrix.csv")
                if split_id == "holdout":
                    self.assertEqual(
                        assignments,
                        root / expected_dir / "student_concept_holdout_assignments.csv",
                    )
                else:
                    self.assertIsNone(assignments)

    def test_coverage_and_doa_commands_route_valid_without_test_leakage(self):
        coverage, doa = build_evaluation_commands(
            dataset_id="ASSIST17",
            split_id="holdout",
            architecture="m2",
            data_root=Path("/datasets"),
            train_summary_path=Path("/artifacts/train.json"),
            coverage_path=Path("/artifacts/coverage.json"),
            doa_path=Path("/artifacts/doa.json"),
            device="cuda:0",
        )
        for command in (coverage, doa):
            self.assertEqual(command[command.index("--split") + 1], "valid")
            self.assertEqual(
                command[command.index("--test-interactions") + 1],
                command[command.index("--valid-interactions") + 1],
            )
            self.assertNotIn("test.csv", command)
        self.assertIn("--holdout-assignments", doa)
        self.assertNotIn("--holdout-assignments", coverage)

    def test_json_outputs_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            _write_json(output, {"attempt": 1})
            with self.assertRaises(FileExistsError):
                _write_json(output, {"attempt": 2})
            self.assertEqual(json.loads(output.read_text())["attempt"], 1)

    def test_gpu_is_rechecked_after_lock_before_use(self):
        initial = [GpuSnapshot(0, 10, 1000, 0)]
        refreshed = [GpuSnapshot(0, 600, 1000, 0)]
        with patch(
            "scripts.run_unified_validation.query_gpu_inventory",
            side_effect=[("initial", initial), ("refreshed", refreshed)],
        ):
            with self.assertRaisesRegex(RuntimeError, "became ineligible"):
                with locked_gpu():
                    self.fail("ineligible GPU must not be yielded")


if __name__ == "__main__":
    unittest.main()
