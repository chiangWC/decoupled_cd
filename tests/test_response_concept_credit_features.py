from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from data.response_concept_credit_features import (
    _in_broad_c,
    _student_context,
    build_credit_feature_sets,
    build_item_reference_statistics,
    collate_credit_student_batch,
)
from scripts.audit_conditional_response_signature import StudentProfile
from scripts.audit_target_local_pairing_protocol import ValidationOnlyProtocol


def _frame(
    student: str,
    items: list[str],
    *,
    prefix: str,
    labels: bool = True,
) -> pd.DataFrame:
    rows = []
    for index, item in enumerate(items):
        row = {
            "source_row_id": f"{prefix}-{student}-{item}",
            "stu_id": student,
            "exer_id": item,
            "cpt_seq": "interaction-value-is-ignored",
        }
        if labels:
            row["label"] = index % 2
        rows.append(row)
    return pd.DataFrame(rows)


def _fixture() -> tuple[
    list[StudentProfile], ValidationOnlyProtocol, dict[str, tuple[str, ...]]
]:
    items = [f"i{index}" for index in range(18)]
    q_lookup = {
        item: (
            f"c{index % 4}",
            f"c{(index + 1) % 4}",
        )
        if index % 3
        else (f"c{index % 4}",)
        for index, item in enumerate(items)
    }
    profiles = []
    for index in range(20):
        query_items = [items[(index + offset) % len(items)] for offset in range(3)]
        support_items = [item for item in items if item not in query_items][:12]
        support = _frame(f"s{index}", support_items, prefix="support")
        query = _frame(f"s{index}", query_items, prefix="query")
        profiles.append(
            StudentProfile(
                student=f"s{index}",
                support=support,
                query=query,
                theta=0.0,
                raw_accuracy=float(support["label"].mean()),
                seen_concepts=frozenset(
                    concept
                    for item in support_items
                    for concept in q_lookup[item]
                ),
                fold=index % 5,
            )
        )
    validation_support = []
    validation_query = []
    for index in range(3):
        student = f"v{index}"
        validation_support.append(
            _frame(student, items[:12], prefix="valid-support")
        )
        validation_query.append(
            _frame(
                student,
                items[12:16],
                prefix="valid-query",
                labels=False,
            )
        )
    optimizer_train = pd.concat(
        [profile.support for profile in profiles]
        + [profile.query for profile in profiles],
        ignore_index=True,
    )
    protocol = ValidationOnlyProtocol(
        dataset="MOOCRadar",
        source_dir=Path("/synthetic"),
        optimizer_train=optimizer_train,
        validation_support=pd.concat(validation_support, ignore_index=True),
        validation_query=pd.concat(validation_query, ignore_index=True),
        q_lookup=q_lookup,
        audit={},
    )
    return profiles, protocol, q_lookup


class TestResponseConceptCreditFeatures(unittest.TestCase):
    def test_foldwise_item_statistics_exclude_held_query_labels(self) -> None:
        profiles, _, q_lookup = _fixture()
        items = sorted(q_lookup)
        item_index = {item: index + 1 for index, item in enumerate(items)}
        concept_index = {
            concept: index + 1
            for index, concept in enumerate(
                sorted({value for values in q_lookup.values() for value in values})
            )
        }
        held = [profile for profile in profiles if profile.fold == 0]
        reference = [profile for profile in profiles if profile.fold != 0]
        statistics = build_item_reference_statistics(
            reference, item_index=item_index
        )
        first = _student_context(
            held[0],
            statistics=statistics,
            item_index=item_index,
            q_lookup=q_lookup,
            concept_index=concept_index,
            max_q_cardinality=2,
        )
        flipped = replace(
            held[0],
            query=held[0].query.assign(label=1 - held[0].query["label"]),
        )
        second = _student_context(
            flipped,
            statistics=statistics,
            item_index=item_index,
            q_lookup=q_lookup,
            concept_index=concept_index,
            max_q_cardinality=2,
        )
        np.testing.assert_array_equal(
            first.support_item_ease, second.support_item_ease
        )
        np.testing.assert_array_equal(
            first.support_item_confidence,
            second.support_item_confidence,
        )
        self.assertNotIn(
            statistics.reference_students_sha256,
            {"", None},
        )

    def test_broad_c_counts_distinct_multi_q_support_items(self) -> None:
        q_lookup = {
            "a": ("c0", "c1"),
            "b": ("c0", "c2"),
            "c": ("c1", "c3"),
            "single": ("c0",),
        }
        support = pd.DataFrame(
            {
                "exer_id": ["a", "a", "b", "c", "single"],
                "label": [1, 0, 1, 0, 1],
            }
        )
        self.assertTrue(
            _in_broad_c(
                support,
                ("c0", "c1"),
                q_lookup=q_lookup,
            )
        )
        self.assertFalse(
            _in_broad_c(
                support.loc[support["exer_id"] != "c"],
                ("c0", "c1"),
                q_lookup=q_lookup,
            )
        )

    def test_builder_is_student_batched_and_validation_is_label_free(self) -> None:
        profiles, protocol, q_lookup = _fixture()
        result = build_credit_feature_sets(
            protocol,
            optimizer_profiles=profiles,
        )
        self.assertEqual(len(result.optimizer.contexts), len(profiles))
        self.assertIsNotNone(result.optimizer.label_sha256)
        self.assertIsNone(result.validation.label_sha256)
        self.assertTrue(
            all(record.label is None for record in result.validation.records)
        )
        self.assertEqual(
            result.audit["validation_item_statistics"]["validation_students_used"],
            False,
        )
        self.assertEqual(
            len(result.audit["fold_item_statistics"]),
            5,
        )
        self.assertTrue(
            all(
                not entry["held_query_rows_used_for_item_statistics"]
                for entry in result.audit["fold_item_statistics"]
            )
        )
        first = result.optimizer.contexts[0]
        for support_index, q_mask in zip(
            first.support_item_ids,
            first.support_q_mask,
            strict=True,
        ):
            item = next(
                name
                for name, index in result.optimizer.item_index.items()
                if index == int(support_index)
            )
            self.assertEqual(int(q_mask.sum()), len(q_lookup[item]))

        batch = collate_credit_student_batch(
            result.optimizer,
            [0, 1],
            device="cpu",
        )
        expected_query_rows = sum(
            len(result.optimizer.record_indices_by_context[index])
            for index in (0, 1)
        )
        self.assertEqual(batch.support_item_ids.shape[0], 2)
        self.assertEqual(batch.query_context_indices.shape[0], expected_query_rows)
        self.assertEqual(batch.labels.shape[0], expected_query_rows)
        self.assertEqual(
            set(batch.query_context_indices.tolist()),
            {0, 1},
        )

    def test_target_group_exclusion_is_enforced(self) -> None:
        profiles, protocol, _ = _fixture()
        bad = replace(
            profiles[0],
            support=pd.concat(
                [profiles[0].support, profiles[0].query.iloc[[0]]],
                ignore_index=True,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "target group"):
            build_credit_feature_sets(
                protocol,
                optimizer_profiles=[bad, *profiles[1:]],
            )


if __name__ == "__main__":
    unittest.main()
