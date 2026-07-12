from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.unified_baseline_audit import audit_baseline_rows
from scripts.unified_cohort import (
    freeze_cohort,
    freeze_primary_cohort,
    load_verified_cohort,
)
from scripts.unified_dataset_audit import (
    DATASET_LAYOUTS,
    audit_pool,
    canonical_sha256,
)


class DatasetFixtureMixin:
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for dataset_id in [
            "ASSIST09",
            "ASSIST17",
            "NIPS34",
            "MOOCRadar",
            "XES3G5M",
        ]:
            self._write_pair(dataset_id, partial_only=dataset_id == "NIPS34")
        for dataset_id in ["Junyi", "EdNet-ICDM"]:
            standard_name, _ = DATASET_LAYOUTS[dataset_id]
            self._write_split(self.root / standard_name, include_q=False)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_pair(self, dataset_id: str, *, partial_only: bool = False) -> None:
        standard_name, holdout_name = DATASET_LAYOUTS[dataset_id]
        self._write_split(self.root / standard_name, partial_only=partial_only)
        self._write_split(
            self.root / holdout_name,
            include_summary=True,
            partial_only=partial_only,
        )

    @staticmethod
    def _write_split(
        path: Path,
        *,
        include_q: bool = True,
        include_summary: bool = False,
        partial_only: bool = False,
    ) -> None:
        path.mkdir(parents=True)
        (path / "train.csv").write_text(
            "stu_id,exer_id,cpt_seq,label\n0,10,1,1\n",
            encoding="utf-8",
        )
        valid_concepts = '"1,2"' if partial_only else "2"
        valid_rows = [
            f"0,11,{valid_concepts},{1 if index < 100 else 0}"
            for index in range(1000)
        ]
        (path / "valid.csv").write_text(
            "stu_id,exer_id,cpt_seq,label\n" + "\n".join(valid_rows) + "\n",
            encoding="utf-8",
        )
        (path / "test.csv").write_bytes(b"\xfftest-content-must-only-be-hashed\n")
        if include_q:
            (path / "Q_matrix.csv").write_text(
                'exer_id,cpt_seq\n10,1\n11,"1,2"\n',
                encoding="utf-8",
            )
        if include_summary:
            (path / "split_summary.json").write_text(
                '{"test_metrics":"must not influence eligibility"}\n',
                encoding="utf-8",
            )


class UnifiedDatasetAuditTests(DatasetFixtureMixin, unittest.TestCase):
    def test_ready_pool_and_canonical_exact_zero_are_reported_separately(self):
        audit = audit_pool(self.root)

        self.assertEqual(
            audit["asset_ready_dataset_ids"],
            ["ASSIST09", "ASSIST17", "NIPS34", "MOOCRadar", "XES3G5M"],
        )
        self.assertEqual(
            audit["eligible_dataset_ids"],
            ["ASSIST09", "ASSIST17", "MOOCRadar", "XES3G5M"],
        )
        nips = audit["datasets"]["NIPS34"]
        self.assertEqual(nips["standard"]["exact_zero_validation_rows"], 0)
        self.assertEqual(nips["holdout"]["exact_zero_validation_rows"], 0)
        self.assertEqual(nips["standard"]["rows_with_unseen_target_concepts"], 1000)
        self.assertEqual(nips["standard"]["partial_unseen_validation_rows"], 1000)
        assist = audit["datasets"]["ASSIST09"]["standard"]
        self.assertEqual(assist["exact_zero_validation_rows"], 1000)
        self.assertEqual(assist["zero_count"], 1000)
        self.assertEqual(assist["zero_positive_count"], 100)
        self.assertEqual(assist["zero_negative_count"], 900)
        self.assertEqual(assist["rows_with_unseen_target_concepts"], 1000)
        self.assertEqual(assist["partial_unseen_validation_rows"], 0)
        self.assertEqual(len(assist["data_sha256"]), 64)
        self.assertEqual(len(assist["q_sha256"]), 64)
        self.assertEqual(len(assist["prediction_order_sha256"]), 64)
        self.assertFalse(nips["eligible"])

    def test_zero_slice_thresholds_are_all_required_for_eligibility(self):
        for label in ("zero_count", "zero_positive_count", "zero_negative_count"):
            with self.subTest(label=label):
                standard_name, holdout_name = DATASET_LAYOUTS["ASSIST09"]
                for split_name in (standard_name, holdout_name):
                    split = self.root / split_name / "valid.csv"
                    if label == "zero_count":
                        positive, negative = 100, 899
                    elif label == "zero_positive_count":
                        positive, negative = 99, 901
                    else:
                        positive, negative = 901, 99
                    rows = ["0,11,2,1"] * positive + ["0,11,2,0"] * negative
                    split.write_text(
                        "stu_id,exer_id,cpt_seq,label\n" + "\n".join(rows) + "\n",
                        encoding="utf-8",
                    )
                record = audit_pool(self.root)["datasets"]["ASSIST09"]
                self.assertFalse(record["eligible"])
                self.assertIn(
                    f"standard {label} below eligibility threshold", record["reasons"]
                )

    def test_nips34_without_exact_zero_slice_remains_partial_only(self):
        nips = audit_pool(self.root)["datasets"]["NIPS34"]
        self.assertEqual(nips["status"], "partial-only")
        self.assertEqual(nips["standard"]["zero_count"], 0)
        self.assertFalse(nips["eligible"])

    def test_junyi_and_ednet_remain_provisional_without_q_and_holdout(self):
        audit = audit_pool(self.root)

        self.assertEqual(audit["provisional_dataset_ids"], ["Junyi", "EdNet-ICDM"])
        self.assertEqual(audit["datasets"]["Junyi"]["status"], "provisional")
        self.assertEqual(audit["datasets"]["EdNet-ICDM"]["status"], "provisional")

    def test_test_content_is_hashed_without_parsing_labels(self):
        audit = audit_pool(self.root)

        record = audit["datasets"]["ASSIST09"]
        self.assertTrue(audit["test_content_hash_only"])
        self.assertTrue(record["test_content_hash_only"])
        self.assertEqual(len(record["standard"]["test_sha256"]), 64)
        self.assertEqual(len(record["holdout"]["test_sha256"]), 64)

    def test_mismatched_q_id_domains_make_assets_ineligible(self):
        _, holdout_name = DATASET_LAYOUTS["ASSIST09"]
        (self.root / holdout_name / "Q_matrix.csv").write_text(
            'exer_id,cpt_seq\n10,1\n11,"1,2"\n12,3\n',
            encoding="utf-8",
        )

        record = audit_pool(self.root)["datasets"]["ASSIST09"]

        self.assertTrue(record["assets_present"])
        self.assertFalse(record["id_domains_match"])
        self.assertFalse(record["eligible"])


class UnifiedCohortFreezeTests(DatasetFixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.audit = audit_pool(self.root)
        self.cohort_path = self.root / "campaign" / "cohort.json"
        self.dataset_ids = ["ASSIST09", "ASSIST17", "MOOCRadar"]
        self.b0_references = {
            dataset_id: {
                "standard": f"b0/{dataset_id}/standard-validation.json",
                "holdout": f"b0/{dataset_id}/holdout-validation.json",
            }
            for dataset_id in self.dataset_ids
        }

    @staticmethod
    def _rehash_pool(audit: dict[str, object]) -> None:
        unhashed = dict(audit)
        unhashed.pop("audit_sha256")
        audit["audit_sha256"] = canonical_sha256(unhashed)

    def test_cohort_requires_exactly_three_datasets(self):
        with self.assertRaisesRegex(ValueError, "exactly three datasets"):
            freeze_cohort(
                self.cohort_path,
                self.dataset_ids[:2],
                audit=self.audit,
                b0_validation_references={
                    key: self.b0_references[key] for key in self.dataset_ids[:2]
                },
            )
        with self.assertRaisesRegex(ValueError, "exactly three datasets"):
            freeze_cohort(
                self.cohort_path,
                [*self.dataset_ids, "XES3G5M"],
                audit=self.audit,
                b0_validation_references={
                    **self.b0_references,
                    "XES3G5M": {"standard": "b0/xes/std", "holdout": "b0/xes/hold"},
                },
            )

    def test_freeze_includes_audit_hashes_b0_references_and_canonical_hash(self):
        frozen = freeze_cohort(
            self.cohort_path,
            self.dataset_ids,
            audit=self.audit,
            b0_validation_references=self.b0_references,
        )

        self.assertEqual(frozen["dataset_ids"], self.dataset_ids)
        self.assertEqual(frozen["audit_sha256"], self.audit["audit_sha256"])
        self.assertEqual(
            set(frozen["dataset_audit_sha256"]), set(self.dataset_ids)
        )
        self.assertEqual(frozen["b0_validation_references"], self.b0_references)
        unhashed = dict(frozen)
        actual_hash = unhashed.pop("cohort_sha256")
        canonical = json.dumps(
            unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        self.assertEqual(
            actual_hash, hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        )

    def test_public_loader_rejects_tampered_canonical_cohort(self):
        frozen = freeze_cohort(
            self.cohort_path,
            self.dataset_ids,
            audit=self.audit,
            b0_validation_references=self.b0_references,
        )
        self.assertEqual(load_verified_cohort(self.cohort_path), frozen)

        tampered = copy.deepcopy(frozen)
        tampered["dataset_ids"] = [*self.dataset_ids, "XES3G5M"]
        self.cohort_path.write_text(
            json.dumps(tampered),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "canonical SHA-256 mismatch"):
            load_verified_cohort(self.cohort_path)

    def test_same_freeze_is_idempotent_but_replacement_is_rejected(self):
        original = freeze_cohort(
            self.cohort_path,
            self.dataset_ids,
            audit=self.audit,
            b0_validation_references=self.b0_references,
        )
        repeated = freeze_cohort(
            self.cohort_path,
            self.dataset_ids,
            audit=self.audit,
            b0_validation_references=self.b0_references,
        )
        self.assertEqual(repeated, original)

        replacement = ["ASSIST09", "ASSIST17", "XES3G5M"]
        replacement_references = {
            dataset_id: {
                "standard": f"b0/{dataset_id}/standard-validation.json",
                "holdout": f"b0/{dataset_id}/holdout-validation.json",
            }
            for dataset_id in replacement
        }
        with self.assertRaisesRegex(ValueError, "dataset list mismatch"):
            freeze_cohort(
                self.cohort_path,
                replacement,
                audit=self.audit,
                b0_validation_references=replacement_references,
            )

    def test_freeze_rejects_stale_pool_audit_hash(self):
        tampered = copy.deepcopy(self.audit)
        tampered["root"] = "/tampered/root"

        with self.assertRaisesRegex(ValueError, "pool audit canonical SHA-256 mismatch"):
            freeze_cohort(
                self.cohort_path,
                self.dataset_ids,
                audit=tampered,
                b0_validation_references=self.b0_references,
            )

    def test_freeze_rejects_fabricated_selected_dataset_hash(self):
        tampered = copy.deepcopy(self.audit)
        tampered["datasets"]["ASSIST09"]["audit_sha256"] = "0" * 64
        self._rehash_pool(tampered)

        with self.assertRaisesRegex(
            ValueError, "dataset audit canonical SHA-256 mismatch: ASSIST09"
        ):
            freeze_cohort(
                self.cohort_path,
                self.dataset_ids,
                audit=tampered,
                b0_validation_references=self.b0_references,
            )

    def test_freeze_validates_record_hash_before_trusting_eligibility(self):
        tampered = copy.deepcopy(self.audit)
        tampered["datasets"]["ASSIST09"]["eligible"] = False
        self._rehash_pool(tampered)

        with self.assertRaisesRegex(
            ValueError, "dataset audit canonical SHA-256 mismatch: ASSIST09"
        ):
            freeze_cohort(
                self.cohort_path,
                self.dataset_ids,
                audit=tampered,
                b0_validation_references=self.b0_references,
            )

    def test_exact_replay_rejects_b0_metadata_drift(self):
        freeze_cohort(
            self.cohort_path,
            self.dataset_ids,
            audit=self.audit,
            b0_validation_references=self.b0_references,
        )
        changed_references = copy.deepcopy(self.b0_references)
        changed_references["ASSIST09"]["standard"] = "b0/replaced-validation.json"

        with self.assertRaisesRegex(ValueError, "frozen cohort metadata mismatch"):
            freeze_cohort(
                self.cohort_path,
                self.dataset_ids,
                audit=self.audit,
                b0_validation_references=changed_references,
            )

    def test_first_freeze_uses_exclusive_create_and_fsyncs_file_and_parent(self):
        with mock.patch(
            "scripts.unified_cohort.os.open", wraps=os.open
        ) as open_mock, mock.patch(
            "scripts.unified_cohort.os.fsync", wraps=os.fsync
        ) as fsync_mock:
            freeze_cohort(
                self.cohort_path,
                self.dataset_ids,
                audit=self.audit,
                b0_validation_references=self.b0_references,
            )

        first_open = open_mock.call_args_list[0]
        self.assertEqual(
            first_open.args[1], os.O_CREAT | os.O_EXCL | os.O_WRONLY
        )
        self.assertEqual(first_open.args[2], 0o644)
        self.assertEqual(fsync_mock.call_count, 2)


class PrimaryCohortRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.a0_rows = [
            {
                "dataset_id": dataset_id,
                "a0_zero_auc": zero_auc,
                "a0_standard_auc": standard_auc,
                "a0_holdout_auc": holdout_auc,
                "a0_fingerprint": character * 64,
                "failed_attempt_count": failures,
            }
            for dataset_id, zero_auc, standard_auc, holdout_auc, character, failures in (
                ("assist17", 0.80, 0.82, 0.81, "a", 1),
                ("moocradar", 0.82, 0.83, 0.82, "b", 2),
                ("xes3g5m", 0.79, 0.84, 0.83, "c", 0),
                ("assist09", 0.78, 0.80, 0.79, "d", 0),
            )
        ]
        audit_records = {}
        for dataset_id, count, character in (
            ("assist17", 3000, "1"),
            ("moocradar", 4000, "2"),
            ("xes3g5m", 2000, "3"),
            ("assist09", 5000, "4"),
        ):
            record = {
                "eligible": True,
                "zero_count": count,
                "standard": {
                    "data_sha256": character * 64,
                    "q_sha256": "a" * 64,
                    "prediction_order_sha256": "b" * 64,
                },
                "holdout": {
                    "data_sha256": character * 64,
                    "q_sha256": "a" * 64,
                    "prediction_order_sha256": "c" * 64,
                },
            }
            record["audit_sha256"] = canonical_sha256(record)
            audit_records[dataset_id] = record
        self.audit_rows = {"datasets": audit_records}
        self.audit_rows["audit_sha256"] = canonical_sha256(self.audit_rows)
        metric_values = {
            "assist17": (0.77, 0.80, 0.79),
            "moocradar": (0.80, 0.80, 0.80),
            "xes3g5m": (0.75, 0.82, 0.81),
            "assist09": (0.77, 0.78, 0.77),
        }
        baseline_rows = []
        for dataset_id, (zero_auc, standard_auc, holdout_auc) in metric_values.items():
            for split, metric, value in (
                ("holdout", "zero_auc", zero_auc),
                ("standard", "auc", standard_auc),
                ("holdout", "auc", holdout_auc),
            ):
                split_audit = audit_records[dataset_id][split]
                baseline_rows.append({
                    "dataset_id": dataset_id,
                    "model": "ORCDF",
                    "seed": 42,
                    "split_seed": 2024,
                    "split": split,
                    "metric": metric,
                    "value": value,
                    "data_sha256": split_audit["data_sha256"],
                    "q_sha256": split_audit["q_sha256"],
                    "prediction_sha256": "d" * 64,
                    "prediction_order_sha256": split_audit[
                        "prediction_order_sha256"
                    ],
                    "config_sha256": "e" * 64,
                    "checkpoint_sha256": "f" * 64,
                    "source_path": f"/baseline/{dataset_id}-{split}-{metric}.json",
                })
        self.baseline_audit = audit_baseline_rows(baseline_rows, self.audit_rows)

    @staticmethod
    def _trusted_expectations(cohort: dict[str, object]) -> dict[str, object]:
        return {
            "expected_dataset_ids": cohort["dataset_ids"],
            "expected_a0_fingerprints": cohort["a0_fingerprints"],
            "expected_comparator_audit_sha256": cohort[
                "comparator_audit_sha256"
            ],
            "expected_dataset_audit_sha256": cohort["dataset_audit_sha256"],
        }

    def test_exact_rank_key_selects_exactly_three_datasets(self) -> None:
        cohort = freeze_primary_cohort(
            self.a0_rows, self.baseline_audit, self.audit_rows
        )
        self.assertEqual(
            cohort["dataset_ids"], ["xes3g5m", "assist17", "moocradar"]
        )
        self.assertEqual(len(cohort["dataset_ids"]), 3)
        self.assertEqual(len(cohort["cohort_sha256"]), 64)

    def test_exclusive_freeze_is_idempotent_and_rejects_fingerprint_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cohort.json"
            first = freeze_primary_cohort(
                self.a0_rows, self.baseline_audit, self.audit_rows, path=path
            )
            repeated = freeze_primary_cohort(
                self.a0_rows, self.baseline_audit, self.audit_rows, path=path
            )
            self.assertEqual(repeated, first)
            changed = copy.deepcopy(self.a0_rows)
            changed[0]["a0_fingerprint"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "A0 fingerprint mismatch"):
                freeze_primary_cohort(
                    changed, self.baseline_audit, self.audit_rows, path=path
                )

    def test_loader_rejects_comparator_or_audit_hash_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cohort.json"
            cohort = freeze_primary_cohort(
                self.a0_rows, self.baseline_audit, self.audit_rows, path=path
            )
            for field in ("comparator_audit_sha256", "dataset_audit_sha256"):
                with self.subTest(field=field):
                    tampered = copy.deepcopy(cohort)
                    tampered[field] = "0" * 64
                    path.write_text(json.dumps(tampered), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "canonical SHA-256 mismatch"):
                        load_verified_cohort(path)
                    path.write_text(json.dumps(cohort), encoding="utf-8")

    def test_schema_v2_loader_requires_and_checks_trusted_expectations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cohort.json"
            cohort = freeze_primary_cohort(
                self.a0_rows, self.baseline_audit, self.audit_rows, path=path
            )
            with self.assertRaisesRegex(ValueError, "trusted schema-v2 expectations"):
                load_verified_cohort(path)

            tampered = copy.deepcopy(cohort)
            selected = tampered["dataset_ids"][0]
            tampered["a0_fingerprints"][selected] = "0" * 64
            unhashed = dict(tampered)
            unhashed.pop("cohort_sha256")
            tampered["cohort_sha256"] = canonical_sha256(unhashed)
            path.write_text(json.dumps(tampered), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "A0 fingerprint mismatch"):
                load_verified_cohort(path, **self._trusted_expectations(cohort))

    def test_cohort_boundary_rejects_rehashed_invalid_accepted_provenance(self) -> None:
        tampered = copy.deepcopy(self.baseline_audit)
        tampered["accepted_rows"][0]["seed"] = 7
        tampered.pop("audit_sha256")
        tampered["audit_sha256"] = canonical_sha256(tampered)

        with self.assertRaisesRegex(ValueError, "accepted baseline provenance"):
            freeze_primary_cohort(self.a0_rows, tampered, self.audit_rows)

    def test_cohort_boundary_rejects_rehashed_non_strongest_registry(self) -> None:
        tampered = copy.deepcopy(self.baseline_audit)
        strongest = tampered["strongest_comparators"]["assist17"]["holdout"][
            "zero_auc"
        ]
        tampered["strongest_comparators"]["assist17"]["holdout"]["zero_auc"] = {
            **strongest,
            "value": 0.99,
        }
        tampered.pop("audit_sha256")
        tampered["audit_sha256"] = canonical_sha256(tampered)

        with self.assertRaisesRegex(ValueError, "strongest comparator registry"):
            freeze_primary_cohort(self.a0_rows, tampered, self.audit_rows)

    def test_primary_freeze_removes_partial_file_after_fsync_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cohort.json"
            with mock.patch(
                "scripts.unified_cohort.os.fsync", side_effect=OSError("injected")
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    freeze_primary_cohort(
                        self.a0_rows, self.baseline_audit, self.audit_rows, path=path
                    )
            self.assertFalse(path.exists())

    def test_a0_fingerprint_must_be_lowercase_sha256(self) -> None:
        invalid = copy.deepcopy(self.a0_rows)
        invalid[0]["a0_fingerprint"] = "A" * 64
        with self.assertRaisesRegex(ValueError, "invalid A0 fingerprint"):
            freeze_primary_cohort(invalid, self.baseline_audit, self.audit_rows)
