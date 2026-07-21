from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts.run_response_credit_activation import (
    _atomic_csv,
    _evaluation_input_fingerprints,
    _git_snapshot,
    _live_origin_branch_head,
    _prediction_semantic_sha256,
    _runtime_environment,
    build_student_batch_plan,
    initialize_model,
    load_standard_stage1_barrier,
    parse_args,
    run_aggregate_phase,
    run_prediction_barrier_phase,
)
from utils.response_credit_evaluation import (
    BOOTSTRAP_REPLICATES,
    compute_stage1_gate,
)


def _passing_holdout_summary(dataset: str) -> dict:
    summary = {
        "schema_version": 1,
        "dataset": dataset,
        "split_kind": "holdout",
        "metrics": {
            scope: {
                variant: {"auc": 0.80, "brier": 0.15}
                for variant in ("full", "direct", "capacity")
            }
            for scope in ("overall", "C", "C_strict", "T")
        },
        "control_envelope_deltas": {
            scope: {
                "auc_vs_control_envelope": 0.003,
                "brier_vs_control_envelope": 0.0,
                "full_minus_direct_auc": 0.003,
                "full_minus_capacity_auc": 0.003,
            }
            for scope in ("overall", "C", "C_strict", "T")
        },
        "descriptive_higher_c_auc_control": "capacity",
        "slice_prevalence": {
            "C": {"rows": 600, "students": 120},
            "C_strict": {"rows": 200, "students": 60},
            "T": {"rows": 300, "students": 100},
        },
        "joint_c_bootstrap": {
            "requested_replicates": BOOTSTRAP_REPLICATES,
            "valid_replicates": BOOTSTRAP_REPLICATES,
            "invalid_replicates": 0,
            "invalid_indices": [],
            "sampling_manifest_sha256": f"samples-{dataset}",
            "confidence_interval_95": [0.001, 0.005],
            "minimum_valid_replicates": 1_800,
        },
    }
    return summary


def _write_official_stage1(
    root: Path,
    *,
    commit: str = "abc",
    topology: str = "topology",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    summaries = [
        _passing_holdout_summary(dataset)
        for dataset in ("MOOCRadar", "NIPS34")
    ]
    paths = []
    payloads = []
    for summary in summaries:
        dataset = summary["dataset"]
        path = root / f"{dataset}_evaluation.json"
        payload = {
            "schema_version": 1,
            "phase": "response_credit_evaluate",
            "formal": True,
            "dataset": dataset,
            "split_kind": "holdout",
            "git_commit": commit,
            "architecture_topology_sha256": topology,
            "comparison_signature": f"signature-{dataset}",
            "evaluation": summary,
            "prediction_manifests": {
                variant: {"path": variant, "sha256": variant}
                for variant in ("full", "direct", "capacity")
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
        payloads.append(payload)
    decision = compute_stage1_gate(summaries)
    decision.update(
        {
            "phase": "response_credit_aggregate",
            "aggregate_stage": "stage1",
            "git_commit": commit,
            "architecture_topology_sha256": topology,
            "evaluation_inputs": _evaluation_input_fingerprints(paths, payloads),
            "stage1_artifact": None,
            "multi_seed_used": False,
            "bootstrap_is_model_seed": False,
        }
    )
    path = root / "stage1.json"
    path.write_text(json.dumps(decision), encoding="utf-8")
    return path


def _write_prediction_matrix(root: Path) -> list[Path]:
    directories = []
    probability_columns = {
        "full": "prob_full",
        "direct": "prob_direct",
        "capacity": "prob_capacity",
    }
    for dataset in ("MOOCRadar", "NIPS34"):
        for variant, probability in probability_columns.items():
            directory = root / dataset / variant
            directory.mkdir(parents=True)
            prediction_path = directory / f"{variant}.csv"
            frame = pd.DataFrame(
                {
                    "source_row_id": [f"{dataset}-0", f"{dataset}-1"],
                    "stu_id": [1, 2],
                    "exer_id": [10, 11],
                    "target_coverage": [0.0, 0.5],
                    "in_c": [True, False],
                    "in_c_strict": [False, False],
                    "in_t": [True, False],
                    probability: [0.4, 0.6],
                }
            )
            _atomic_csv(prediction_path, frame)
            manifest = {
                "schema_version": 1,
                "phase": "response_credit_predict",
                "formal": True,
                "dataset": dataset,
                "split_kind": "holdout",
                "variant": variant,
                "git": {
                    "head": "commit",
                    "worktree_clean": True,
                    "formal_enforced": True,
                    "live_origin_branch_head": "commit",
                },
                "optimization": {"epochs": 20},
                "model": {
                    "architecture": {"topology_sha256": "topology"},
                },
                "artifacts": {
                    "predictions": {
                        "path": prediction_path.name,
                        "sha256": hashlib.sha256(
                            prediction_path.read_bytes()
                        ).hexdigest(),
                        "semantic_sha256": f"semantic-{dataset}-{variant}",
                        "row_order_sha256": f"order-{dataset}",
                        "rows": len(frame),
                    }
                },
                "leakage_audit": {
                    "validation_labels_loaded": False,
                    "prediction_artifact_contains_label": False,
                    "test_files_opened": False,
                },
            }
            (directory / "prediction_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            directories.append(directory)
    return directories


class TestResponseCreditRunner(unittest.TestCase):
    def test_global_prediction_barrier_accepts_complete_unlabeled_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            directories = _write_prediction_matrix(root / "predictions")
            result = run_prediction_barrier_phase(
                prediction_dirs=directories,
                split_kind="holdout",
                output_dir=root / "barrier",
            )
        self.assertTrue(result["checks"]["all_six_predictions_present"])
        self.assertTrue(result["checks"]["csv_headers_are_label_free"])
        self.assertFalse(result["leakage_audit"]["validation_labels_loaded"])
        self.assertFalse(result["leakage_audit"]["source_files_opened"])

    def test_global_prediction_barrier_rejects_incomplete_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            directories = _write_prediction_matrix(root / "predictions")
            with self.assertRaisesRegex(RuntimeError, "exactly 6"):
                run_prediction_barrier_phase(
                    prediction_dirs=directories[:-1],
                    split_kind="holdout",
                    output_dir=root / "barrier",
                )

    def test_global_prediction_barrier_rejects_label_column(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            directories = _write_prediction_matrix(root / "predictions")
            directory = directories[0]
            manifest_path = directory / "prediction_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            prediction_path = directory / manifest["artifacts"]["predictions"]["path"]
            frame = pd.read_csv(prediction_path)
            frame["label"] = [0, 1]
            _atomic_csv(prediction_path, frame)
            manifest["artifacts"]["predictions"]["sha256"] = hashlib.sha256(
                prediction_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "label column"):
                run_prediction_barrier_phase(
                    prediction_dirs=directories,
                    split_kind="holdout",
                    output_dir=root / "barrier",
                )

    def test_global_prediction_barrier_rejects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            directories = _write_prediction_matrix(root / "predictions")
            manifest_path = directories[0] / "prediction_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["predictions"]["sha256"] = "bad"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
                run_prediction_barrier_phase(
                    prediction_dirs=directories,
                    split_kind="holdout",
                    output_dir=root / "barrier",
                )

    def test_topology_hash_excludes_dataset_dimensions(self) -> None:
        first = SimpleNamespace(
            num_items=10,
            num_concepts=5,
            max_q_cardinality=2,
        )
        second = SimpleNamespace(
            num_items=100,
            num_concepts=30,
            max_q_cardinality=6,
        )
        _, first_audit = initialize_model(first, variant="full")
        _, second_audit = initialize_model(second, variant="capacity")
        self.assertEqual(
            first_audit["architecture"]["topology_sha256"],
            second_audit["architecture"]["topology_sha256"],
        )
        self.assertNotEqual(
            first_audit["architecture"]["instance_sha256"],
            second_audit["architecture"]["instance_sha256"],
        )
        self.assertNotEqual(
            first_audit["total_parameter_count"],
            second_audit["total_parameter_count"],
        )

    def test_prediction_semantic_hash_survives_csv_roundtrip(self) -> None:
        frame = pd.DataFrame(
            {
                "source_row_id": ["a", "b", "c"],
                "prob_full": np.asarray([0.1, 0.12345679, 0.9], dtype=np.float32),
            }
        )
        before = _prediction_semantic_sha256(frame, variant="full")
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "predictions.csv"
            _atomic_csv(path, frame)
            loaded = pd.read_csv(path)
        after = _prediction_semantic_sha256(loaded, variant="full")
        self.assertEqual(before, after)

    def test_formal_standard_requires_matching_passed_stage1(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = _write_official_stage1(Path(raw))
            payload = json.loads(path.read_text(encoding="utf-8"))
            loaded = load_standard_stage1_barrier(
                formal=True,
                split_kind="standard",
                stage1_json=path,
                git_head="abc",
            )
            self.assertEqual(loaded, payload)

            payload["stage1_passed"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "passed Stage 1"):
                load_standard_stage1_barrier(
                    formal=True,
                    split_kind="standard",
                    stage1_json=path,
                    git_head="abc",
                )
        with self.assertRaisesRegex(RuntimeError, "stage1-json"):
            load_standard_stage1_barrier(
                formal=True,
                split_kind="standard",
                stage1_json=None,
                git_head="abc",
            )

    def test_minimal_handwritten_stage1_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "stage1.json"
            path.write_text(
                json.dumps(
                    {
                        "gate": "response_credit_stage1_holdout",
                        "stage1_passed": True,
                        "git_commit": "abc",
                        "architecture_topology_sha256": "topology",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "official aggregate"):
                load_standard_stage1_barrier(
                    formal=True,
                    split_kind="standard",
                    stage1_json=path,
                    git_head="abc",
                )

    def test_stage1_rejects_changed_evaluation_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            stage1 = _write_official_stage1(root)
            evaluation = root / "MOOCRadar_evaluation.json"
            evaluation.write_text(
                evaluation.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                load_standard_stage1_barrier(
                    formal=True,
                    split_kind="standard",
                    stage1_json=stage1,
                    git_head="abc",
                )

    def test_stage2_rechecks_stage1_topology(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            stage1 = _write_official_stage1(root / "stage1_inputs")
            standard_paths = []
            for dataset in ("MOOCRadar", "NIPS34"):
                path = root / f"{dataset}_standard.json"
                payload = {
                    "schema_version": 1,
                    "phase": "response_credit_evaluate",
                    "formal": True,
                    "dataset": dataset,
                    "split_kind": "standard",
                    "git_commit": "abc",
                    "architecture_topology_sha256": "different-topology",
                    "comparison_signature": f"standard-{dataset}",
                    "evaluation": {
                        **_passing_holdout_summary(dataset),
                        "split_kind": "standard",
                    },
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
                standard_paths.append(path)
            with self.assertRaisesRegex(RuntimeError, "topology"):
                run_aggregate_phase(
                    stage="stage2",
                    evaluation_jsons=standard_paths,
                    output_dir=root / "aggregate",
                    stage1_json=stage1,
                )

    def test_student_batch_plan_is_shared_and_split_namespaced(self) -> None:
        first = build_student_batch_plan(
            50,
            dataset="MOOCRadar",
            split_kind="holdout",
            epochs=1,
        )
        second = build_student_batch_plan(
            50,
            dataset="MOOCRadar",
            split_kind="holdout",
            epochs=1,
        )
        standard = build_student_batch_plan(
            50,
            dataset="MOOCRadar",
            split_kind="standard",
            epochs=1,
        )
        np.testing.assert_array_equal(first.permutations, second.permutations)
        self.assertEqual(first.sha256, second.sha256)
        self.assertNotEqual(first.sha256, standard.sha256)

    def test_live_origin_check_requires_exact_branch_ref(self) -> None:
        advertised = SimpleNamespace(
            returncode=0,
            stdout="abc123\trefs/heads/codex/student-local-inductive\n",
            stderr="",
        )
        with patch(
            "scripts.run_response_credit_activation.subprocess.run",
            return_value=advertised,
        ) as mocked:
            self.assertEqual(
                _live_origin_branch_head("codex/student-local-inductive"),
                "abc123",
            )
        self.assertEqual(
            mocked.call_args.args[0],
            [
                "git",
                "ls-remote",
                "--heads",
                "origin",
                "refs/heads/codex/student-local-inductive",
            ],
        )

    def test_live_origin_retries_twice_then_succeeds(self) -> None:
        responses = [
            SimpleNamespace(returncode=128, stdout="", stderr="temporary one"),
            SimpleNamespace(returncode=128, stdout="", stderr="temporary two"),
            SimpleNamespace(
                returncode=0,
                stdout="abc123\trefs/heads/branch\n",
                stderr="",
            ),
        ]
        with (
            patch(
                "scripts.run_response_credit_activation.subprocess.run",
                side_effect=responses,
            ) as mocked,
            patch("scripts.run_response_credit_activation.time.sleep") as sleep,
        ):
            self.assertEqual(_live_origin_branch_head("branch"), "abc123")
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [0.5, 1.0],
        )

    def test_live_origin_final_failure_preserves_stderr(self) -> None:
        responses = [
            SimpleNamespace(returncode=128, stdout="", stderr="temporary one"),
            SimpleNamespace(returncode=128, stdout="", stderr="temporary two"),
            SimpleNamespace(returncode=128, stdout="", stderr="final network error"),
        ]
        with (
            patch(
                "scripts.run_response_credit_activation.subprocess.run",
                side_effect=responses,
            ) as mocked,
            patch("scripts.run_response_credit_activation.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "final network error"):
                _live_origin_branch_head("branch")
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_formal_git_snapshot_rejects_live_origin_mismatch(self) -> None:
        outputs = iter(
            [
                SimpleNamespace(stdout="local-head\n"),
                SimpleNamespace(stdout=""),
                SimpleNamespace(stdout="origin/branch\n"),
                SimpleNamespace(stdout="branch\n"),
                SimpleNamespace(stdout="local-head\n"),
                SimpleNamespace(
                    returncode=0,
                    stdout="remote-head\trefs/heads/branch\n",
                    stderr="",
                ),
            ]
        )
        with patch(
            "scripts.run_response_credit_activation.subprocess.run",
            side_effect=lambda *args, **kwargs: next(outputs),
        ):
            with self.assertRaisesRegex(RuntimeError, "live origin"):
                _git_snapshot(formal=True, expected_commit="local-head")

    def test_runtime_environment_records_reproducibility_fields(self) -> None:
        environment = _runtime_environment(torch.device("cpu"))
        for name in (
            "python_version",
            "python_executable",
            "torch_version",
            "torch_cuda_version",
            "cudnn_version",
            "conda_default_env",
            "requested_device",
            "deterministic_algorithms_enabled",
        ):
            self.assertIn(name, environment)
        self.assertEqual(environment["requested_device"], "cpu")
        self.assertIsNone(environment["resolved_cuda_index"])

    def test_cli_has_no_test_path_argument(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "predict",
                    "--dataset",
                    "MOOCRadar",
                    "--source-dir",
                    "/data",
                    "--split-kind",
                    "holdout",
                    "--variant",
                    "full",
                    "--output-dir",
                    "/out",
                    "--device",
                    "cpu",
                    "--test-path",
                    "/forbidden/test.csv",
                ]
            )

    def test_prediction_barrier_cli_has_no_source_argument(self) -> None:
        arguments = [
            "prediction-barrier",
            "--split-kind",
            "holdout",
            "--output-dir",
            "/out",
        ]
        for index in range(6):
            arguments.extend(["--prediction-dir", f"/prediction/{index}"])
        arguments.extend(["--source-dir", "/forbidden/valid"])
        with self.assertRaises(SystemExit):
            parse_args(arguments)


if __name__ == "__main__":
    unittest.main()
