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
    ARCHITECTURES,
    DATASET_DIRECTORIES,
    ELIGIBLE_DATASET_IDS,
    GpuSnapshot,
    NumericalRecipe,
    RECIPES,
    _split_paths,
    _write_json,
    architecture_fingerprint,
    architecture_spec,
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
            architecture="a0",
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
        self.assertEqual(
            command[command.index("--unified-completion") + 1], "prior"
        )
        self.assertEqual(
            command[command.index("--unified-completion-rank") + 1], "32"
        )
        self.assertEqual(
            command[command.index("--unified-evidence-loss-weight") + 1],
            "0.1",
        )
        self.assertEqual(
            command[command.index("--unified-completion-loss-weight") + 1],
            "0.0",
        )
        self.assertNotIn("--unified-mastery-loss-weight", command)
        self.assertNotIn("--unified-inference", command)
        self.assertNotIn("--unified-composer", command)

    def test_lowrank_training_command_registers_positive_completion_loss(self):
        command = build_train_command(
            dataset_id="ASSIST17",
            split_id="standard",
            architecture="a1",
            data_root=Path("/datasets"),
            output=Path("/artifacts/train-summary.json"),
            device="cpu",
        )

        self.assertEqual(
            command[command.index("--unified-completion") + 1], "lowrank"
        )
        self.assertEqual(
            command[command.index("--unified-completion-loss-weight") + 1],
            "0.1",
        )
        self.assertNotIn("--unified-mastery-loss-weight", command)

    def test_runner_recognizes_only_a0_and_a1_version_3_completions(self):
        self.assertEqual(ARCHITECTURES, {"a0": ("prior", 0.0), "a1": ("lowrank", 1.0)})
        for architecture, completion in (("a0", "prior"), ("a1", "lowrank")):
            command = build_train_command(
                dataset_id="ASSIST09",
                split_id="standard",
                architecture=architecture,
                data_root=Path("/datasets"),
                output=Path("/artifacts/train-summary.json"),
                device="cpu",
            )
            self.assertEqual(command[command.index("--unified-completion") + 1], completion)
            self.assertEqual(architecture_spec(architecture).manifest()["version"], 3)
        for removed in ("b0", "m2", "m2-m3"):
            with self.subTest(architecture=removed), self.assertRaisesRegex(
                ValueError, "unknown unified architecture"
            ):
                architecture_spec(removed)

    def test_registered_recipe_table_is_ordered_and_a1_can_inherit_a0_recipe(self):
        self.assertEqual(
            RECIPES["ASSIST17"],
            (
                NumericalRecipe("student_recompute_minibatch", 64, 40, 1e-3, 0.0, 5, 64),
                NumericalRecipe("student_recompute_minibatch", 64, 80, 2e-3, 0.0, 5, 128),
            ),
        )
        self.assertEqual(RECIPES["XES3G5M"][-1].epochs, 3000)
        selected_a0 = RECIPES["ASSIST17"][1]
        a0_command = build_train_command(
            dataset_id="ASSIST17",
            split_id="standard",
            architecture="a0",
            data_root=Path("/datasets"),
            output=Path("/artifacts/a0.json"),
            device="cpu",
            recipe=selected_a0,
        )
        a1_command = build_train_command(
            dataset_id="ASSIST17",
            split_id="standard",
            architecture="a1",
            data_root=Path("/datasets"),
            output=Path("/artifacts/a1.json"),
            device="cpu",
            recipe=selected_a0,
        )
        for flag in (
            "--concept-dim",
            "--epochs",
            "--learning-rate",
            "--weight-decay",
            "--early-stop-patience",
            "--training-mode",
            "--student-batch-size",
            "--unified-evidence-loss-weight",
        ):
            self.assertEqual(
                a0_command[a0_command.index(flag) + 1],
                a1_command[a1_command.index(flag) + 1],
            )

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

    def test_gpu_memory_eligibility_has_exact_half_open_boundary(self):
        self.assertEqual(select_gpu_index([GpuSnapshot(4, 49, 100, 99)]), 4)
        self.assertEqual(select_gpu_index([GpuSnapshot(5, 0, 100, 99)]), 5)
        with self.assertRaisesRegex(RuntimeError, "under half memory"):
            select_gpu_index([GpuSnapshot(6, 50, 100, 0)])

    def test_gpu_locks_are_nonblocking_per_physical_device(self):
        first_index = os.getpid() * 2 + 100_000
        second_index = first_index + 1
        snapshots = [
            GpuSnapshot(first_index, 1, 100, 0),
            GpuSnapshot(second_index, 2, 100, 0),
        ]
        lock_paths = [
            Path(f"/tmp/unified-mastery-gpu-{snapshot.index}.lock")
            for snapshot in snapshots
        ]
        try:
            with patch(
                "scripts.run_unified_validation.query_gpu_inventory",
                return_value=("inventory", snapshots),
            ):
                with locked_gpu() as first:
                    with locked_gpu() as second:
                        self.assertEqual(
                            (first[0], second[0]),
                            (first_index, second_index),
                        )
                    with patch(
                        "scripts.run_unified_validation.query_gpu_inventory",
                        return_value=("inventory", [snapshots[0]]),
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, "no unlocked eligible GPU"
                        ):
                            with locked_gpu():
                                self.fail("same physical GPU lock must not block")
        finally:
            for path in lock_paths:
                path.unlink(missing_ok=True)

    def test_smoke_summary_requires_mastery_loss_fingerprint_and_gpu_peak(self):
        fingerprint = architecture_fingerprint("a1")
        summary = {
            "architecture_manifest": architecture_spec("a1").manifest(),
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
        fingerprint = architecture_fingerprint("a1")
        manifest = architecture_spec("a1").manifest()
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
                        "architecture_manifest": manifest,
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

    def test_candidate_rows_reject_version_2_summary(self):
        old = {
            "inference": "graph",
            "composer": "mask",
            "decoder": "neuralcdm-monotonic",
            "mastery_output": "student-concept",
            "modules": "m1-m2-m4-neuralcdm",
            "version": 2,
        }
        summary = {
            "dataset_id": "ASSIST09",
            "split_id": "standard",
            "architecture_fingerprint": architecture_fingerprint("a1"),
            "architecture_manifest": old,
            "cohort_sha256": "a" * 64,
            "seed": 42,
            "overall_auc": 0.7,
            "zero_auc": 0.6,
            "ordinary_doa": 0.6,
            "weighted_doa": 0.6,
            "mastery_shape": [2, 3],
            "final_loss": 0.4,
        }
        with self.assertRaisesRegex(ValueError, "version 3"):
            assemble_candidate_rows(
                [summary, {**summary, "split_id": "holdout"}],
                cohort_dataset_ids=["ASSIST09"],
                cohort_sha256="a" * 64,
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
                        "a1",
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
            architecture="a1",
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
