from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.analyze_concept_depletion_gate import (
    load_aligned,
    summarize_dataset,
)
from scripts.prepare_concept_depletion_gate import (
    build_paired_arms,
    seen_concepts,
)


def _training_frame() -> pd.DataFrame:
    rows = []
    for student in ("1", "2"):
        for exercise, concept, label in (
            ("a1", "a", 1),
            ("a2", "a", 0),
            ("b1", "b", 1),
            ("b2", "b", 0),
            ("c1", "c", 1),
            ("c2", "c", 0),
            ("d1", "d", 1),
            ("d2", "d", 0),
        ):
            rows.append(
                {
                    "stu_id": student,
                    "exer_id": exercise,
                    "cpt_seq": concept,
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def _q_map() -> dict[str, frozenset[str]]:
    return {
        exercise: frozenset({exercise[0]})
        for exercise in ("a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2")
    }


class ConceptDepletionGateTest(unittest.TestCase):
    def test_paired_arms_match_quantity_and_target_coverage(self) -> None:
        train = _training_frame()
        audit = pd.DataFrame(
            [
                {
                    "audit_row_id": "row-1",
                    "stu_id": "1",
                    "exer_id": "a1",
                    "cpt_seq": "a",
                    "label": 1,
                },
                {
                    "audit_row_id": "row-2",
                    "stu_id": "2",
                    "exer_id": "a2",
                    "cpt_seq": "a",
                    "label": 0,
                },
            ]
        )
        concept, random, selected, summary = build_paired_arms(
            train,
            audit,
            _q_map(),
            dataset="toy",
            seed=2024,
            min_history=3,
        )

        self.assertEqual(len(concept), len(train) - 4)
        self.assertEqual(len(random), len(train) - 4)
        self.assertEqual(summary["selected_rows"], 2)
        self.assertTrue(summary["per_student_counts_matched"])
        self.assertEqual(selected["stu_id"].nunique(), 2)

        concept_seen = seen_concepts(concept, _q_map())
        random_seen = seen_concepts(random, _q_map())
        for student in ("1", "2"):
            self.assertNotIn("a", concept_seen[student])
            self.assertIn("a", random_seen[student])
            self.assertEqual(
                (concept["stu_id"] == student).sum(),
                (random["stu_id"] == student).sum(),
            )

        flipped = audit.copy()
        flipped["label"] = 1 - flipped["label"]
        _, _, selected_flipped, _ = build_paired_arms(
            train,
            flipped,
            _q_map(),
            dataset="toy",
            seed=2024,
            min_history=3,
        )
        self.assertTrue(
            selected[["audit_row_id", "stu_id", "exer_id"]].equals(
                selected_flipped[["audit_row_id", "stu_id", "exer_id"]]
            )
        )

    def test_analysis_uses_aligned_rows_and_positive_damage(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol"
            protocol.mkdir()
            selected = pd.DataFrame(
                [
                    {
                        "audit_row_id": "row-1",
                        "stu_id": "1",
                        "exer_id": "a1",
                        "label": 1,
                    },
                    {
                        "audit_row_id": "row-2",
                        "stu_id": "2",
                        "exer_id": "a2",
                        "label": 0,
                    },
                ]
            )
            selected.to_csv(protocol / "selected_targets.csv", index=False)
            (protocol / "manifest.json").write_text(
                json.dumps({"protocol": "toy"}), encoding="utf-8"
            )
            concept = selected.copy()
            concept["prob"] = [0.4, 0.6]
            random = selected.copy()
            random["prob"] = [0.8, 0.2]
            concept_path = root / "concept.csv"
            random_path = root / "random.csv"
            concept.to_csv(concept_path, index=False)
            random.to_csv(random_path, index=False)

            frame, hashes = load_aligned(
                protocol, concept_path, random_path
            )
            result = summarize_dataset(
                name="toy",
                frame=frame,
                hashes=hashes,
                bootstrap=200,
                seed=2024,
                min_selected=2,
            )

        self.assertTrue(result["qualified_for_gate"])
        self.assertTrue(result["supports_concept_specific_damage"])
        self.assertGreater(
            result["log_loss_damage_concept_minus_random"], 0
        )
        self.assertGreater(
            result["bootstrap"]["log_loss_damage"]["ci_low"], 0
        )
        self.assertGreater(result["auc_damage_random_minus_concept"], 0)


if __name__ == "__main__":
    unittest.main()
