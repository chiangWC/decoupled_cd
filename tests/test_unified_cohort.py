from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.unified_cohort import freeze_cohort
from scripts.unified_dataset_audit import DATASET_LAYOUTS, audit_pool


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
        (path / "valid.csv").write_text(
            f"stu_id,exer_id,cpt_seq,label\n0,11,{valid_concepts},0\n",
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
        self.assertEqual(nips["standard"]["rows_with_unseen_target_concepts"], 1)
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

    def test_cohort_requires_at_least_three_datasets(self):
        with self.assertRaisesRegex(ValueError, "at least three datasets"):
            freeze_cohort(
                self.cohort_path,
                self.dataset_ids[:2],
                audit=self.audit,
                b0_validation_references={
                    key: self.b0_references[key] for key in self.dataset_ids[:2]
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
