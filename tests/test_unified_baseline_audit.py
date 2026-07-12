from __future__ import annotations

import hashlib
import math
from pathlib import Path
import tempfile
import unittest

from scripts.unified_baseline_audit import (
    BASELINE_SOURCE_PATHS,
    REQUIRED_BASELINE_FIELDS,
    audit_baseline_rows,
    inventory_baseline_sources,
)


class UnifiedBaselineAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {
            "dataset_id": "assist17", "model": "ORCDF", "seed": 42,
            "split_seed": 2024, "split": "holdout", "metric": "zero_auc",
            "value": 0.78, "data_sha256": "a" * 64, "q_sha256": "b" * 64,
            "prediction_sha256": "c" * 64,
            "prediction_order_sha256": "d" * 64,
            "config_sha256": "e" * 64, "checkpoint_sha256": "f" * 64,
            "source_path": "/abs/internal/result.json",
        }
        self.audits = {
            "assist17": {
                "eligible": True,
                "standard": {
                    "data_sha256": "1" * 64, "q_sha256": "b" * 64,
                    "prediction_order_sha256": "2" * 64,
                },
                "holdout": {
                    "data_sha256": "a" * 64, "q_sha256": "b" * 64,
                    "prediction_order_sha256": "d" * 64,
                },
            }
        }

    def test_required_fields_and_same_protocol_row_are_accepted(self) -> None:
        self.assertEqual(REQUIRED_BASELINE_FIELDS, (
            "dataset_id", "model", "seed", "split_seed", "split", "metric",
            "value", "data_sha256", "q_sha256", "prediction_sha256",
            "prediction_order_sha256", "config_sha256", "checkpoint_sha256",
            "source_path",
        ))
        result = audit_baseline_rows([self.row], self.audits)
        self.assertEqual(result["accepted_count"], 1)
        self.assertEqual(result["rejected_count"], 0)

    def test_each_protocol_or_provenance_mismatch_has_an_exact_reason(self) -> None:
        expected = {
            "seed": "seed mismatch: expected 42",
            "split_seed": "split_seed mismatch: expected 2024",
            "data_sha256": "data_sha256 mismatch for assist17/holdout",
            "q_sha256": "q_sha256 mismatch for assist17/holdout",
            "prediction_order_sha256": (
                "prediction_order_sha256 mismatch for assist17/holdout"
            ),
        }
        for field, reason in expected.items():
            with self.subTest(field=field):
                broken = dict(self.row)
                broken[field] = 7 if field.endswith("seed") else "0" * 64
                result = audit_baseline_rows([broken], self.audits)
                self.assertEqual(result["accepted_count"], 0)
                self.assertEqual(result["rejected_rows"][0]["reasons"], [reason])

    def test_missing_provenance_and_non_finite_metric_are_rejected(self) -> None:
        missing = dict(self.row)
        missing.pop("checkpoint_sha256")
        non_finite = dict(self.row, value=math.nan)
        result = audit_baseline_rows([missing, non_finite], self.audits)
        self.assertEqual(result["accepted_count"], 0)
        self.assertEqual(result["rejected_rows"][0]["reasons"],
                         ["missing required fields: checkpoint_sha256"])
        self.assertEqual(result["rejected_rows"][1]["reasons"],
                         ["value must be finite"])

    def test_strongest_same_protocol_auc_is_the_larger_accepted_value(self) -> None:
        weaker = dict(self.row, model="KaNCD", value=0.76)
        result = audit_baseline_rows([weaker, self.row], self.audits)
        strongest = result["strongest_comparators"]["assist17"]["holdout"]["zero_auc"]
        self.assertEqual(strongest["model"], "ORCDF")
        self.assertEqual(strongest["value"], 0.78)

    def test_ineligible_dataset_rows_cannot_enter_hard_gate_registry(self) -> None:
        audits = {"assist17": {**self.audits["assist17"], "eligible": False}}
        result = audit_baseline_rows([self.row], audits)
        self.assertEqual(result["accepted_count"], 0)
        self.assertEqual(result["rejected_rows"][0]["reasons"],
                         ["dataset is not eligible: assist17"])

    def test_source_inventory_hashes_files_and_records_missing_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            present = Path(directory) / "row.json"
            present.write_bytes(b"auditable source\n")
            missing = Path(directory) / "missing.json"
            inventory = inventory_baseline_sources([present, missing])
        self.assertEqual(inventory[0]["source_sha256"],
                         hashlib.sha256(b"auditable source\n").hexdigest())
        self.assertEqual(inventory[0]["reasons"], [])
        self.assertEqual(inventory[1]["reasons"], ["source path does not exist"])
        self.assertEqual(len(BASELINE_SOURCE_PATHS), 6)


if __name__ == "__main__":
    unittest.main()
