import unittest

import pandas as pd

from scripts.evaluate_coverage_slice import (
    add_target_coverage,
    build_exercise_concepts,
    build_student_seen_concepts,
    coverage_bucket,
)


class CoverageSliceTest(unittest.TestCase):
    def test_build_student_seen_concepts_uses_train_history_only(self) -> None:
        train_frame = pd.DataFrame(
            [
                {"stu_id": 1, "exer_id": 11, "cpt_seq": "A,B", "label": 1},
                {"stu_id": 1, "exer_id": 12, "cpt_seq": "C", "label": 0},
                {"stu_id": 2, "exer_id": 13, "cpt_seq": "B", "label": 1},
            ]
        )

        seen = build_student_seen_concepts(train_frame)

        self.assertEqual(seen[1], {"A", "B", "C"})
        self.assertEqual(seen[2], {"B"})

    def test_build_exercise_concepts_merges_duplicate_q_rows(self) -> None:
        q_matrix = pd.DataFrame(
            [
                {"exer_id": 11, "cpt_seq": "A"},
                {"exer_id": 11, "cpt_seq": "B,C"},
            ]
        )

        exercise_concepts = build_exercise_concepts(q_matrix)

        self.assertEqual(exercise_concepts["11"], {"A", "B", "C"})

    def test_coverage_bucket_edges(self) -> None:
        self.assertEqual(coverage_bucket(None), "no_concepts")
        self.assertEqual(coverage_bucket(0.0), "zero")
        self.assertEqual(coverage_bucket(0.25), "low")
        self.assertEqual(coverage_bucket(0.5), "partial")
        self.assertEqual(coverage_bucket(0.75), "partial")
        self.assertEqual(coverage_bucket(1.0), "full")

    def test_add_target_coverage_uses_q_matrix_exercise_concepts(self) -> None:
        train_frame = pd.DataFrame(
            [
                {"stu_id": 1, "exer_id": 11, "cpt_seq": "A", "label": 1},
                {"stu_id": 2, "exer_id": 12, "cpt_seq": "A,B", "label": 0},
            ]
        )
        test_frame = pd.DataFrame(
            [
                {"stu_id": 1, "exer_id": 20, "cpt_seq": "A", "label": 1, "prob": 0.8},
                {"stu_id": 2, "exer_id": 21, "cpt_seq": "A", "label": 0, "prob": 0.2},
                {"stu_id": 3, "exer_id": 22, "cpt_seq": "C", "label": 1, "prob": 0.7},
            ]
        )
        q_matrix = pd.DataFrame(
            [
                {"exer_id": 20, "cpt_seq": "A,B"},
                {"exer_id": 21, "cpt_seq": "A,B"},
                {"exer_id": 22, "cpt_seq": "C"},
            ]
        )

        enriched = add_target_coverage(test_frame, train_frame=train_frame, q_matrix=q_matrix)

        self.assertEqual(enriched["target_concept_count"].tolist(), [2, 2, 1])
        self.assertEqual(enriched["target_seen_concept_count"].tolist(), [1, 2, 0])
        self.assertEqual(enriched["target_coverage"].tolist(), [0.5, 1.0, 0.0])
        self.assertEqual(enriched["coverage_bucket"].tolist(), ["partial", "full", "zero"])
        self.assertEqual(enriched["coverage_group"].tolist(), ["other", "full_coverage", "low_coverage"])


if __name__ == "__main__":
    unittest.main()
