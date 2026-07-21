from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pandas as pd

from data.pool_protocol import sha256_file
from data.target_local_pairing_features import build_feature_sets
from scripts.audit_target_local_pairing_protocol import validation_row_order_sha256
from scripts.target_local_pairing_runner import (
    ALL_VARIANTS,
    EXPECTED_DATASETS,
    PERMUTATION_REPLICATES,
    PROBABILITY_COLUMNS,
    _common_state_sha256,
    _hash_array,
    _load_locked_protocol,
    _load_prediction_artifacts,
    _parameter_schema,
    _prediction_semantic_sha256,
    _state_dict_sha256,
    build_batch_plan,
    compute_activation_gate,
    gradient_audit,
    initialize_models,
    parse_args,
    train_variant,
)
from tests.test_target_local_pairing_features import _profiles, _protocol


def _evaluation_payload(
    dataset: str,
    *,
    effect: float,
    ci_low: float = -0.001,
    brier: float = 0.0,
    identified: bool = True,
) -> dict:
    return {
        "schema_version": 1,
        "phase": "target_local_pairing_evaluate",
        "dataset": dataset,
        "git": {"head": "a" * 40},
        "architecture_family_sha256": "b" * 64,
        "feature_definition": {
            "item_numeric_names": ["x"],
            "support_statistic_names": ["s"],
            "response_feature_names": ["r"],
        },
        "protocol": {"identified": identified},
        "evaluation": {
            "dataset_effect": effect,
            "joint_min_delta_ci": [ci_low, ci_low + 0.01],
            "max_brier_regression": brier,
        },
    }


class TestTargetLocalPairingRunner(unittest.TestCase):
    def test_cli_has_strict_phases_and_no_test_argument(self) -> None:
        args = parse_args(
            [
                "predict",
                "--dataset",
                "ASSIST17",
                "--source-dir",
                "/source",
                "--protocol-json",
                "/protocol.json",
                "--expected-protocol-sha256",
                "a" * 64,
                "--expected-commit",
                "b" * 40,
                "--output-dir",
                "/output",
                "--device",
                "cpu",
            ]
        )
        self.assertEqual(args.command, "predict")
        self.assertFalse(any("test" in key for key in vars(args)))
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "predict",
                    "--dataset",
                    "ASSIST17",
                    "--source-dir",
                    "/source",
                    "--protocol-json",
                    "/protocol.json",
                    "--expected-protocol-sha256",
                    "a" * 64,
                    "--expected-commit",
                    "b" * 40,
                    "--output-dir",
                    "/output",
                    "--device",
                    "cpu",
                    "--test-path",
                    "/forbidden",
                ]
            )

    def test_batch_plan_is_frozen_and_complete(self) -> None:
        first = build_batch_plan(257, dataset="ASSIST17")
        second = build_batch_plan(257, dataset="ASSIST17")
        other = build_batch_plan(257, dataset="MOOCRadar")
        np.testing.assert_array_equal(first.permutations, second.permutations)
        self.assertEqual(first.sha256, second.sha256)
        self.assertNotEqual(first.sha256, other.sha256)
        self.assertEqual(first.permutations.shape, (20, 257))
        for permutation in first.permutations:
            np.testing.assert_array_equal(np.sort(permutation), np.arange(257))
        self.assertEqual([len(x) for x in first.epoch_batches(0)], [128, 128, 1])

    def test_pair_variants_have_exact_init_and_common_strong_init(self) -> None:
        features = SimpleNamespace(num_items=7, num_concepts=4)
        models, audit = initialize_models(features)
        pair_schemas = {_parameter_schema(models[name]) for name in (
            "real_pair",
            "late_fusion",
            "perm_pair",
        )}
        pair_hashes = {
            _state_dict_sha256(models[name].state_dict())
            for name in ("real_pair", "late_fusion", "perm_pair")
        }
        self.assertEqual(len(pair_schemas), 1)
        self.assertEqual(len(pair_hashes), 1)
        self.assertEqual(
            len({_common_state_sha256(models[name]) for name in ALL_VARIANTS}),
            1,
        )
        self.assertEqual(audit["pair_parameter_difference_percent"], 0.0)

        other_models, other_audit = initialize_models(
            SimpleNamespace(num_items=11, num_concepts=6)
        )
        self.assertEqual(
            audit["architecture"]["family_sha256"],
            other_audit["architecture"]["family_sha256"],
        )
        del other_models

    def test_fixed_twenty_epoch_training_smoke(self) -> None:
        profiles, q_lookup = _profiles()
        full = build_feature_sets(
            _protocol(profiles, q_lookup), optimizer_profiles=profiles
        ).optimizer
        # Two rows keep the smoke cheap; the production function still owns
        # the frozen 20-epoch loop and exposes no epoch override.
        features = __import__("dataclasses").replace(
            full, records=full.records[:2]
        )
        models, _ = initialize_models(features)
        audit = train_variant(
            models["real_pair"],
            variant="real_pair",
            features=features,
            batch_plan=build_batch_plan(2, dataset="ASSIST17"),
            device=__import__("torch").device("cpu"),
        )
        self.assertEqual(audit["epochs"], 20)
        self.assertEqual(len(audit["epoch_losses"]), 20)
        self.assertTrue(np.isfinite(audit["final_epoch_loss"]))

    def test_runner_gradient_audit_reaches_all_four_variants(self) -> None:
        profiles, q_lookup = _profiles()
        features = build_feature_sets(
            _protocol(profiles, q_lookup), optimizer_profiles=profiles
        ).optimizer
        models, _ = initialize_models(features)
        audit = gradient_audit(
            models,
            features=features,
            device=__import__("torch").device("cpu"),
        )
        self.assertEqual(set(audit), set(ALL_VARIANTS))
        self.assertTrue(
            all(value["minimum_gradient_norm"] > 0 for value in audit.values())
        )

    def test_locked_protocol_requires_exact_file_sha_and_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {
                                "dataset": "ASSIST17",
                                "audit": "target_local_pairing_protocol",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            expected = sha256_file(path)
            payload, actual = _load_locked_protocol(
                path, dataset="ASSIST17", expected_sha256=expected
            )
            self.assertEqual(payload["dataset"], "ASSIST17")
            self.assertEqual(actual, expected)
            with self.assertRaises(RuntimeError):
                _load_locked_protocol(
                    path, dataset="MOOCRadar", expected_sha256=expected
                )
            with self.assertRaises(RuntimeError):
                _load_locked_protocol(
                    path, dataset="ASSIST17", expected_sha256="0" * 64
                )

    def test_prediction_artifacts_require_unlabeled_aligned_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame(
                {
                    "source_row_id": ["r1", "r2", "r3"],
                    "stu_id": ["s1", "s1", "s2"],
                    "exer_id": ["i1", "i2", "i1"],
                    "coverage_bucket": ["exact_zero", "partial", "full"],
                    "optimizer_item_frequency": [10, 4, 10],
                    **{
                        column: np.linspace(0.2, 0.8, 3).astype(np.float32)
                        for column in PROBABILITY_COLUMNS.values()
                    },
                }
            )
            prediction_path = root / "predictions.csv"
            frame.to_csv(prediction_path, index=False)
            donor = np.tile(np.linspace(0.1, 0.9, 3), (PERMUTATION_REPLICATES, 1)).astype(
                np.float32
            )
            hashes = [f"{index:064x}" for index in range(PERMUTATION_REPLICATES)]
            donor_path = root / "donor.npz"
            np.savez_compressed(
                donor_path,
                probabilities=donor,
                source_row_ids=np.asarray(frame["source_row_id"], dtype="U64"),
                mapping_sha256=np.asarray(hashes, dtype="U64"),
            )
            manifest = {
                "phase": "target_local_pairing_predict",
                "artifacts": {
                    "predictions": {
                        "path": prediction_path.name,
                        "sha256": sha256_file(prediction_path),
                        "semantic_sha256": _prediction_semantic_sha256(frame),
                        "row_order_sha256": validation_row_order_sha256(frame),
                    },
                    "donor_sensitivity": {
                        "path": donor_path.name,
                        "sha256": sha256_file(donor_path),
                        "semantic_probability_sha256": _hash_array(donor),
                        "prediction_sha256": [
                            _hash_array(row) for row in donor
                        ],
                        "mapping_sha256": hashes,
                    },
                },
            }
            (root / "prediction_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            _, loaded, matrix, loaded_hashes = _load_prediction_artifacts(root)
            self.assertEqual(loaded["source_row_id"].tolist(), ["r1", "r2", "r3"])
            np.testing.assert_array_equal(matrix, donor)
            self.assertEqual(loaded_hashes, hashes)

            frame["label"] = [0, 1, 1]
            frame.to_csv(prediction_path, index=False)
            manifest["artifacts"]["predictions"]["sha256"] = sha256_file(prediction_path)
            (root / "prediction_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                _load_prediction_artifacts(root)

    def test_activation_gate_is_conjunctive_and_ignores_unidentified_regression(self) -> None:
        effects = {
            "ASSIST17": (0.0032, 0.0001, True),
            "MOOCRadar": (0.0021, 0.0002, True),
            "XES3G5M": (-0.0008, 0.0, True),
            "Junyi": (-0.5, 0.5, False),
        }
        payloads = [
            _evaluation_payload(
                dataset,
                effect=effect,
                ci_low=0.0001 if dataset == "ASSIST17" else -0.001,
                brier=brier,
                identified=identified,
            )
            for dataset, (effect, brier, identified) in effects.items()
        ]
        gate = compute_activation_gate(payloads)
        self.assertTrue(gate["activated"])
        self.assertEqual(
            gate["counted_effect_datasets"], ["ASSIST17", "MOOCRadar"]
        )
        self.assertFalse(gate["conditional_permutation_p_value_used"])

        regressed = [dict(payload) for payload in payloads]
        regressed[2] = _evaluation_payload("XES3G5M", effect=-0.00101)
        failed = compute_activation_gate(regressed)
        self.assertFalse(failed["activated"])
        self.assertFalse(
            failed["checks"]["no_identified_dataset_delta_below_minus_0.001"]
        )

    def test_aggregate_requires_exact_four_dataset_pool(self) -> None:
        payloads = [
            _evaluation_payload(dataset, effect=0.003, ci_low=0.001)
            for dataset in EXPECTED_DATASETS[:-1]
        ]
        with self.assertRaises(RuntimeError):
            compute_activation_gate(payloads)

    def test_missing_joint_ci_fails_cleanly_instead_of_crashing(self) -> None:
        payloads = [
            _evaluation_payload(dataset, effect=0.003, ci_low=0.001)
            for dataset in EXPECTED_DATASETS
        ]
        for payload in payloads:
            payload["evaluation"]["joint_min_delta_ci"] = None
        gate = compute_activation_gate(payloads)
        self.assertFalse(gate["activated"])
        self.assertFalse(
            gate["checks"]["at_least_one_joint_student_bootstrap_ci_low_gt_0"]
        )

    def test_shell_scheduler_is_bounded_and_transactional(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_target_local_pairing_audit.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("MAX_PARALLEL=3", script)
        self.assertNotIn("for ((slot =", script)
        self.assertIn("printf \"%s\\n\" \"${GPU_ROWS[@]}\"", script)
        self.assertIn("STAGING_ROOT=\"${OUTPUT_ROOT}.staging-", script)
        self.assertIn("Incomplete artifacts preserved", script)
        self.assertIn(
            "SCHEDULE_ORDER=(MOOCRadar ASSIST17 Junyi XES3G5M)",
            script,
        )
        self.assertIn("run_prediction_worker()", script)
        self.assertIn("WORKER_COUNT=${#GPU_SLOTS[@]}", script)
        self.assertNotIn("while (( next_dataset", script)
        self.assertLess(
            script.index("for index in \"${!PIDS[@]}\""),
            script.index("for dataset in \"${DATASETS[@]}\""),
        )
        self.assertLess(
            script.index("for dataset in \"${DATASETS[@]}\""),
            script.index("target_local_pairing_runner.py aggregate"),
        )


if __name__ == "__main__":
    unittest.main()
