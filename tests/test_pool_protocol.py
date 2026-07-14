from __future__ import annotations

import unittest

import pandas as pd

from data.pool_protocol import (
    assign_atomic_holdout_split,
    assign_atomic_standard_split,
    canonicalize_interactions,
    derive_q_matrix,
    exact_row_overlap,
    group_overlap,
)


class PoolProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for student in range(8):
            for exercise in range(12):
                rows.append(
                    {
                        "stu_id": student,
                        "exer_id": exercise,
                        "cpt_seq": f"{exercise % 6}",
                        "label": (student + exercise) % 2,
                    }
                )
        rows.append(dict(rows[0]))
        rows.append({**rows[1], "label": 1 - rows[1]["label"]})
        self.raw = pd.DataFrame(rows)

    def test_canonicalization_deduplicates_only_exact_rows(self) -> None:
        clean, removed = canonicalize_interactions(self.raw)
        self.assertEqual(removed, 1)
        conflicting = clean[(clean["stu_id"] == "0") & (clean["exer_id"] == "1")]
        self.assertEqual(set(conflicting["label"]), {0, 1})
        self.assertFalse(clean["source_row_id"].duplicated().any())

    def test_standard_split_is_atomic_and_reproducible(self) -> None:
        clean, _ = canonicalize_interactions(self.raw)
        first = assign_atomic_standard_split(clean, seed=2024)
        second = assign_atomic_standard_split(clean, seed=2024)
        self.assertEqual(group_overlap(first), {"train_valid": 0, "train_test": 0, "valid_test": 0})
        self.assertEqual(exact_row_overlap(first), {"train_valid": 0, "train_test": 0, "valid_test": 0})
        for split in ("train", "valid", "test"):
            self.assertEqual(first[split]["source_row_id"].tolist(), second[split]["source_row_id"].tolist())

    def test_holdout_split_keeps_repeated_attempt_groups_atomic(self) -> None:
        clean, _ = canonicalize_interactions(self.raw)
        splits, assignments = assign_atomic_holdout_split(
            clean,
            seed=2024,
            min_student_interactions=5,
            min_student_concepts=2,
        )
        self.assertEqual(group_overlap(splits), {"train_valid": 0, "train_test": 0, "valid_test": 0})
        self.assertEqual(len(assignments), clean["stu_id"].nunique())

    def test_q_mapping_reports_conflicts(self) -> None:
        clean, _ = canonicalize_interactions(self.raw)
        q_matrix, conflicts = derive_q_matrix(clean)
        self.assertEqual(conflicts, 0)
        self.assertEqual(q_matrix["exer_id"].nunique(), 12)
        altered = clean.copy()
        altered.loc[0, "cpt_seq"] = "99"
        _, conflicts = derive_q_matrix(altered)
        self.assertEqual(conflicts, 1)


if __name__ == "__main__":
    unittest.main()
