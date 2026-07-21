from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import pandas as pd

from scripts.audit_option_contrast_admission import (
    attach_options,
    option_integrity,
    split_partition_report,
    train_support_report,
    verify_train_option_boundary,
)
from scripts.prepare_option_contrast_pool import (
    canonicalize_option_rows,
    label_free_row_id,
    stable_key,
    verify_ednet_snapshot,
)


class OptionContrastProtocolTest(unittest.TestCase):
    def option_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "stu_id": range(10),
                "exer_id": [0] * 10,
                "cpt_seq": ["0"] * 10,
                "label": [1] + [0] * 9,
                "selected_option": [0] + [1] * 9,
                "correct_option": [0] * 10,
                "option_count": [4] * 10,
            }
        )

    def test_option_rows_canonicalize_with_unique_sidecar(self) -> None:
        core, options = canonicalize_option_rows(self.option_frame())
        self.assertEqual(len(core), 10)
        self.assertEqual(len(options), 10)
        self.assertFalse(options["source_row_id"].duplicated().any())
        attached, coverage = attach_options(core, options)
        self.assertEqual(coverage, 1.0)
        self.assertEqual(
            option_integrity(attached)["label_consistency_fraction"],
            1.0,
        )

    def test_option_rows_reject_duplicate_student_item(self) -> None:
        frame = self.option_frame()
        frame.loc[1, "stu_id"] = frame.loc[0, "stu_id"]
        with self.assertRaises(RuntimeError):
            canonicalize_option_rows(frame)

    def test_option_rows_reject_inconsistent_label(self) -> None:
        frame = self.option_frame()
        frame.loc[0, "label"] = 0
        with self.assertRaises(RuntimeError):
            canonicalize_option_rows(frame)

    def test_train_support_is_measured_across_students(self) -> None:
        core, options = canonicalize_option_rows(self.option_frame())
        attached, _ = attach_options(core, options)
        report = train_support_report(attached)
        self.assertEqual(report["item_option_cells"], 2)
        self.assertAlmostEqual(
            report["supported_train_row_fraction"],
            0.9,
        )
        self.assertEqual(report["items_with_two_supported_wrong_options"], 0)
        self.assertFalse(report["eligible"])

    def test_label_free_row_identity_does_not_change_with_outcome(self) -> None:
        original = self.option_frame()
        original_core, _ = canonicalize_option_rows(original)
        flipped = self.option_frame()
        flipped.loc[0, "label"] = 0
        flipped.loc[0, "selected_option"] = 1
        flipped_core, _ = canonicalize_option_rows(flipped)
        self.assertEqual(
            original_core.loc[0, "source_row_id"],
            flipped_core.loc[0, "source_row_id"],
        )
        self.assertEqual(
            label_free_row_id(0, 0),
            original_core.loc[0, "source_row_id"],
        )

    def test_source_row_join_rejects_student_item_identity_mismatch(self) -> None:
        core, options = canonicalize_option_rows(self.option_frame())
        altered = options.copy()
        altered.loc[0, "stu_id"] = "999"
        with self.assertRaises(RuntimeError):
            attach_options(core, altered)

    def test_integrity_agreement_is_conditional_on_available_rows(self) -> None:
        core, options = canonicalize_option_rows(self.option_frame())
        attached, _ = attach_options(core, options)
        attached.loc[0, "selected_option"] = float("nan")
        report = option_integrity(attached)
        self.assertAlmostEqual(report["available_fraction"], 0.9)
        self.assertEqual(report["valid_range_fraction"], 1.0)
        self.assertEqual(report["label_consistency_fraction"], 1.0)

    def test_train_option_boundary_rejects_extra_or_changed_rows(self) -> None:
        core, options = canonicalize_option_rows(self.option_frame())
        train = core.iloc[:8].copy()
        train_options = options.iloc[:8].copy()
        report = verify_train_option_boundary(
            train=train,
            train_options=train_options,
            audit_options=options,
        )
        self.assertTrue(report["exact_audit_value_match"])
        with self.assertRaises(RuntimeError):
            verify_train_option_boundary(
                train=train,
                train_options=options,
                audit_options=options,
            )
        altered = train_options.copy()
        altered.loc[altered.index[1], "selected_option"] = 2
        with self.assertRaises(RuntimeError):
            verify_train_option_boundary(
                train=train,
                train_options=altered,
                audit_options=options,
            )

    def test_split_partition_requires_disjoint_exhaustive_rows(self) -> None:
        core, _ = canonicalize_option_rows(self.option_frame())
        frames = {
            "data": core,
            "train": core.iloc[:6],
            "valid": core.iloc[6:8],
            "test": core.iloc[8:],
        }
        report = split_partition_report(frames)
        self.assertTrue(report["mutually_disjoint"])
        self.assertTrue(report["exhaustive_exact"])
        overlapping = {
            **frames,
            "valid": core.iloc[5:8],
        }
        report = split_partition_report(overlapping)
        self.assertFalse(report["mutually_disjoint"])
        self.assertFalse(report["exhaustive_exact"])
        incomplete = {
            **frames,
            "test": core.iloc[9:],
        }
        report = split_partition_report(incomplete)
        self.assertTrue(report["mutually_disjoint"])
        self.assertFalse(report["exhaustive_exact"])

    def test_stable_student_selection_key_is_reproducible(self) -> None:
        self.assertEqual(
            stable_key(2024, "ednet-student", 17),
            stable_key(2024, "ednet-student", 17),
        )
        self.assertNotEqual(
            stable_key(2024, "ednet-student", 17),
            stable_key(2024, "ednet-student", 18),
        )

    def test_ednet_snapshot_is_checked_against_official_rows(self) -> None:
        snapshot = pd.DataFrame(
            {
                "user_id": [10, 10],
                "timestamp": [2, 1],
                "question_id": ["q2", "q1"],
                "user_answer": ["B", "a"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "u10.csv"
            pd.DataFrame(
                {
                    "timestamp": [1, 2],
                    "solving_id": [1, 2],
                    "question_id": ["q1", "q2"],
                    "user_answer": ["a", "b"],
                    "elapsed_time": [100, 200],
                }
            ).to_csv(path, index=False)
            archive = Path(directory) / "ednet.zip"
            with ZipFile(archive, "w") as handle:
                handle.write(path, arcname="KT1/u10.csv")
            report = verify_ednet_snapshot(
                snapshot,
                students=[10],
                kt1_dir=directory,
                archive=archive,
            )
            self.assertEqual(report["rows"], 2)
            self.assertEqual(len(report["canonical_sha256"]), 64)
            altered = snapshot.copy()
            altered.loc[0, "user_answer"] = "c"
            with self.assertRaises(RuntimeError):
                verify_ednet_snapshot(
                    altered,
                    students=[10],
                    kt1_dir=directory,
                    archive=archive,
                )


if __name__ == "__main__":
    unittest.main()
