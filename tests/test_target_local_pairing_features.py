from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import pandas as pd
import torch

from data.target_local_pairing_features import (
    ITEM_EASE_EPS,
    ITEM_NUMERIC_NAMES,
    RESPONSE_FEATURE_NAMES,
    SUPPORT_STATISTIC_NAMES,
    _feature_input_sha256,
    _item_numeric,
    _raw_feature_set,
    build_feature_sets,
    collate_pairing_batch,
    replace_validation_donors,
)
from scripts.audit_conditional_response_signature import (
    StudentProfile,
    build_reference_statistics,
)
from scripts.audit_target_local_pairing_protocol import (
    DonorMapping,
    ValidationOnlyProtocol,
    build_donor_mapping,
)


def _frame(
    student: str,
    items: list[str],
    *,
    prefix: str,
    label_offset: int = 0,
) -> pd.DataFrame:
    rows = []
    for index, item in enumerate(items):
        attempts = 2 if index == 0 else 1
        for attempt in range(attempts):
            rows.append(
                {
                    "source_row_id": f"{prefix}-{student}-{item}-{attempt}",
                    "stu_id": student,
                    "exer_id": item,
                    "cpt_seq": f"c{int(item[1:]) % 3}",
                    "label": (index + attempt + label_offset) % 2,
                }
            )
    return pd.DataFrame(rows)


def _profiles(count: int = 25) -> tuple[list[StudentProfile], dict[str, tuple[str, ...]]]:
    items = [f"i{index}" for index in range(18)]
    q_lookup = {
        item: (
            f"c{index % 3}",
            *(() if index % 4 else (f"c{(index + 1) % 3}",)),
        )
        for index, item in enumerate(items)
    }
    profiles = []
    for index in range(count):
        query_start = index % 6
        query_items = [
            items[(query_start + offset) % len(items)]
            for offset in range(4)
        ]
        support_items = [item for item in items if item not in query_items][:12]
        support = _frame(f"s{index}", support_items, prefix="support")
        query = _frame(f"s{index}", query_items, prefix="query")
        correct = float(support["label"].sum())
        attempts = float(len(support))
        accuracy = correct / attempts
        theta = float(
            np.clip(
                np.log(((correct + 1) / (attempts + 2)) / ((attempts - correct + 1) / (attempts + 2))),
                -4,
                4,
            )
        )
        profiles.append(
            StudentProfile(
                student=f"s{index}",
                support=support,
                query=query,
                theta=theta,
                raw_accuracy=accuracy,
                seen_concepts=frozenset(
                    concept
                    for item in support_items
                    for concept in q_lookup[item]
                ),
                fold=index % 5,
            )
        )
    return profiles, q_lookup


def _protocol(
    profiles: list[StudentProfile],
    q_lookup: dict[str, tuple[str, ...]],
) -> ValidationOnlyProtocol:
    validation_students = ["v0", "v1"]
    support_rows = []
    query_rows = []
    retained_items = sorted(
        {
            str(item)
            for profile in profiles
            for frame in (profile.support, profile.query)
            for item in frame["exer_id"].astype(str)
        }
    )
    for index, student in enumerate(validation_students):
        support_items = retained_items[:12]
        query_items = retained_items[12:16]
        support_rows.append(
            _frame(student, support_items, prefix="valid-support", label_offset=index)
        )
        feature_query = _frame(
            student, query_items, prefix="valid-query", label_offset=index
        ).drop(columns=["label"])
        query_rows.append(feature_query)
    optimizer_train = pd.concat(
        [profile.support for profile in profiles]
        + [profile.query for profile in profiles],
        ignore_index=True,
    )
    return ValidationOnlyProtocol(
        dataset="synthetic",
        source_dir=__import__("pathlib").Path("/synthetic"),
        optimizer_train=optimizer_train,
        validation_support=pd.concat(support_rows, ignore_index=True),
        validation_query=pd.concat(query_rows, ignore_index=True),
        q_lookup=q_lookup,
        audit={},
    )


class TestTargetLocalPairingFeatures(unittest.TestCase):
    def test_frozen_schema_and_transform_are_exact(self) -> None:
        self.assertEqual(len(ITEM_NUMERIC_NAMES), 16)
        self.assertEqual(len(SUPPORT_STATISTIC_NAMES), 6)
        self.assertEqual(
            RESPONSE_FEATURE_NAMES,
            (
                "response",
                "response_minus_foldwise_item_ease",
                "group_attempts_over_group_attempts_plus_one",
            ),
        )
        profiles, q_lookup = _profiles()
        items = sorted(q_lookup)
        item_index = {item: index for index, item in enumerate(items)}
        statistics, binner = build_reference_statistics(
            profiles[5:],
            items=items,
            item_index=item_index,
            q_lookup=q_lookup,
        )
        value = _item_numeric(
            item=items[0],
            student=profiles[0].student,
            theta=profiles[0].theta,
            statistics=statistics,
            binner=binner,
            item_index=item_index,
            q_lookup=q_lookup,
        )
        ease = float(
            np.clip(
                statistics.item_ease[item_index[items[0]]],
                ITEM_EASE_EPS,
                1 - ITEM_EASE_EPS,
            )
        )
        self.assertAlmostEqual(value[-3], np.log(ease / (1 - ease)))
        self.assertAlmostEqual(
            value[-2], np.log1p(statistics.item_count[item_index[items[0]]])
        )
        self.assertEqual(value[-1], len(q_lookup[items[0]]))

    def test_held_query_label_flip_cannot_change_held_fold_inputs(self) -> None:
        profiles, q_lookup = _profiles()
        held = [profile for profile in profiles if profile.fold == 0]
        reference = [profile for profile in profiles if profile.fold != 0]
        items = sorted(q_lookup)
        item_index = {item: index for index, item in enumerate(items)}
        statistics, binner = build_reference_statistics(
            reference,
            items=items,
            item_index=item_index,
            q_lookup=q_lookup,
        )
        model_item_index = {
            item: index + 1 for index, item in enumerate(items)
        }
        q_item_index = item_index
        donor = build_donor_mapping(
            held,
            statistics=statistics,
            item_index=item_index,
            q_lookup=q_lookup,
            split="optimizer_oof_query",
            fold=0,
            replicate=0,
        )
        first = _raw_feature_set(
            dataset="synthetic",
            split="optimizer_oof_query",
            profiles=held,
            statistics=statistics,
            binner=binner,
            donor_mapping=donor,
            item_index=item_index,
            q_item_index=q_item_index,
            q_lookup=q_lookup,
            model_item_index=model_item_index,
            labels_available=True,
        )
        flipped = [
            replace(profile, query=profile.query.assign(label=1 - profile.query["label"]))
            for profile in held
        ]
        second = _raw_feature_set(
            dataset="synthetic",
            split="optimizer_oof_query",
            profiles=flipped,
            statistics=statistics,
            binner=binner,
            donor_mapping=donor,
            item_index=item_index,
            q_item_index=q_item_index,
            q_lookup=q_lookup,
            model_item_index=model_item_index,
            labels_available=True,
        )
        for one, two in zip(first.contexts, second.contexts, strict=True):
            np.testing.assert_array_equal(
                one.support_item_numeric, two.support_item_numeric
            )
            np.testing.assert_array_equal(
                one.support_statistics, two.support_statistics
            )
        for one, two in zip(first.records, second.records, strict=True):
            np.testing.assert_array_equal(
                one.target_item_numeric, two.target_item_numeric
            )
            self.assertEqual(one.source_row_id, two.source_row_id)
            self.assertNotEqual(one.label, two.label)

    def test_builder_scales_from_oof_and_maps_zero_count_items_to_unk(self) -> None:
        profiles, q_lookup = _profiles()
        result = build_feature_sets(
            _protocol(profiles, q_lookup),
            optimizer_profiles=profiles,
        )
        self.assertEqual(
            result.optimizer.schema_sha256,
            result.validation.schema_sha256,
        )
        self.assertIsNotNone(result.optimizer.label_sha256)
        self.assertIsNone(result.validation.label_sha256)
        self.assertEqual(
            result.audit["item_numeric_names"], ITEM_NUMERIC_NAMES
        )
        self.assertEqual(
            result.audit["support_statistic_names"], SUPPORT_STATISTIC_NAMES
        )
        self.assertEqual(len(result.audit["folds"]), 5)
        self.assertEqual(
            len(result.audit["optimizer_mapping_replicate_zero_sha256"]),
            64,
        )
        self.assertTrue(
            all(
                len(fold["donor_mapping_sha256"]) == 64
                and len(fold["reference_students_sha256"]) == 64
                and len(fold["reference_rows_sha256"]) == 64
                for fold in result.audit["folds"]
            )
        )
        # At least one fold has an item absent from its reference query rows.
        self.assertTrue(
            any(
                bool((context.support_model_item_ids == 0).any())
                for context in result.optimizer.contexts
            )
        )

    def test_response_ease_confidence_and_q_union_reach_batch(self) -> None:
        profiles, q_lookup = _profiles()
        result = build_feature_sets(
            _protocol(profiles, q_lookup),
            optimizer_profiles=profiles,
        )
        batch = collate_pairing_batch(
            result.optimizer,
            [0, 1],
            device="cpu",
        )
        self.assertEqual(batch.support_responses.ndim, 2)
        self.assertTrue(
            torch.all(
                (batch.support_group_attempt_confidence > 0)
                | ~batch.support_mask
            )
        )
        first_context = result.optimizer.contexts[
            result.optimizer.records[0].context_index
        ]
        length = len(first_context.support_responses)
        np.testing.assert_allclose(
            (
                batch.support_responses[0, :length]
                - batch.support_item_ease[0, :length]
            ).numpy(),
            first_context.support_responses - first_context.support_item_ease,
        )
        self.assertGreaterEqual(
            int(batch.support_q_multi_hot[0, :length].sum().item()),
            length,
        )
        self.assertEqual(
            tuple(batch.support_statistics.shape),
            (2, len(SUPPORT_STATISTIC_NAMES)),
        )

    def test_donor_replacement_changes_only_donor_fields(self) -> None:
        profiles, q_lookup = _profiles()
        result = build_feature_sets(
            _protocol(profiles, q_lookup),
            optimizer_profiles=profiles,
        )
        mapping = build_donor_mapping(
            result.validation_profiles,
            statistics=result.validation_statistics,
            item_index=result.validation.q_item_index,
            q_lookup=q_lookup,
            split="validation_query",
            fold=None,
            replicate=7,
        )
        replaced = replace_validation_donors(result.validation, mapping)
        self.assertEqual(
            result.validation.row_order_sha256,
            replaced.row_order_sha256,
        )
        self.assertIs(
            result.validation.item_numeric_scaler,
            replaced.item_numeric_scaler,
        )
        self.assertIs(
            result.validation.support_statistics_scaler,
            replaced.support_statistics_scaler,
        )
        for before, after in zip(
            result.validation.contexts, replaced.contexts, strict=True
        ):
            np.testing.assert_array_equal(
                before.support_item_numeric, after.support_item_numeric
            )
            np.testing.assert_array_equal(
                before.support_responses, after.support_responses
            )
        for before, after in zip(
            result.validation.records, replaced.records, strict=True
        ):
            self.assertEqual(before.source_row_id, after.source_row_id)
            self.assertEqual(before.exercise, after.exercise)
            np.testing.assert_array_equal(
                before.target_item_numeric, after.target_item_numeric
            )

    def test_input_hash_includes_q_and_identity_mappings(self) -> None:
        profiles, q_lookup = _profiles()
        features = build_feature_sets(
            _protocol(profiles, q_lookup),
            optimizer_profiles=profiles,
        ).validation
        altered_q = list(features.q_concept_indices)
        row = next(
            index
            for index, concepts in enumerate(altered_q)
            if len(concepts) == 1
        )
        original = altered_q[row]
        altered_q[row] = ((original[0] + 1) % features.num_concepts,)
        altered = _feature_input_sha256(
            schema_sha256=features.schema_sha256,
            contexts=features.contexts,
            records=features.records,
            model_item_index=features.model_item_index,
            q_item_index=features.q_item_index,
            q_concept_indices=tuple(altered_q),
            concept_index=features.concept_index,
        )
        self.assertNotEqual(features.input_sha256, altered)


if __name__ == "__main__":
    unittest.main()
