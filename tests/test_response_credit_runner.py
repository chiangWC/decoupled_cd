from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.run_response_credit_activation import (
    _atomic_csv,
    _prediction_semantic_sha256,
    build_student_batch_plan,
    initialize_model,
    load_standard_stage1_barrier,
    parse_args,
)


class TestResponseCreditRunner(unittest.TestCase):
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
            path = Path(raw) / "stage1.json"
            payload = {
                "gate": "response_credit_stage1_holdout",
                "stage1_passed": True,
                "git_commit": "abc",
                "architecture_topology_sha256": "topology",
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
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


if __name__ == "__main__":
    unittest.main()
