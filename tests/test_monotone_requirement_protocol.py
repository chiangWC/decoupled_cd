from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from data.monotone_requirement_protocol import (
    GROUP_NAMESPACE,
    ITEM_OFFSET_STRENGTH,
    MIN_QUERY_GROUPS,
    MIN_SUPPORT_GROUPS,
    OPTIMIZER_FRACTION,
    ROLE_NAMESPACE,
    SPLIT_SEED,
    SUPPORT_FRACTION,
    RequirementFeatureSet,
    RequirementQueryRecord,
    RequirementStudentContext,
    build_train_only_requirement_protocol,
    collate_requirement_batch,
    load_audit_query_labels,
)
from data.pool_protocol import stable_fraction


def _is_optimizer(student: str) -> bool:
    return (
        stable_fraction(SPLIT_SEED, *ROLE_NAMESPACE, student)
        < OPTIMIZER_FRACTION
    )


def _is_support(student: str, item: str) -> bool:
    return (
        stable_fraction(SPLIT_SEED, *GROUP_NAMESPACE, student, item)
        < SUPPORT_FRACTION
    )


def _concepts(item_index: int) -> tuple[str, ...]:
    return (
        ("0", "1")
        if item_index % 4 == 0
        else ("1", "2")
        if item_index % 4 == 1
        else ("0", "2", "3")
        if item_index % 4 == 2
        else ("3",)
    )


def _write_source(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    q_rows: list[dict[str, object]] = []
    students = [f"s{index:03d}" for index in range(80)]
    for item_index in range(40):
        item = f"i{item_index:02d}"
        # Long-form and deliberately reversed entries exercise canonical Q union.
        for concept in reversed(_concepts(item_index)):
            q_rows.append({"exer_id": item, "cpt_seq": concept})
        for student_index, student in enumerate(students):
            label = (student_index * 3 + item_index) % 2
            rows.append(
                {
                    "stu_id": student,
                    "exer_id": item,
                    "cpt_seq": "999",
                    "label": label,
                }
            )
            if item_index in {0, 5}:
                rows.append(
                    {
                        "stu_id": student,
                        "exer_id": item,
                        "cpt_seq": "998",
                        "label": 1 - label,
                    }
                )

    # Audit-only items guarantee an unseen-by-optimizer offset fallback.
    for student_index, student in enumerate(students):
        if _is_optimizer(student):
            continue
        suffix = 0
        while True:
            item = f"audit_only_{student}_{suffix}"
            if not _is_support(student, item):
                break
            suffix += 1
        q_rows.extend(
            [
                {"exer_id": item, "cpt_seq": "3"},
                {"exer_id": item, "cpt_seq": "0"},
            ]
        )
        rows.append(
            {
                "stu_id": student,
                "exer_id": item,
                "cpt_seq": "999",
                "label": student_index % 2,
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(root / "train.csv", index=False)
    pd.DataFrame(q_rows).to_csv(root / "Q_matrix.csv", index=False)
    (root / "valid.csv").write_text("must,not,be,opened\n")
    (root / "test.csv").write_text("must,not,be,opened\n")
    return frame


def _build(root: Path):
    return build_train_only_requirement_protocol(
        dataset="synthetic",
        train_path=root / "train.csv",
        q_matrix_path=root / "Q_matrix.csv",
    )


class TestTrainOnlyRequirementProtocol(unittest.TestCase):
    def test_frozen_splits_q_union_atomic_groups_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source = _write_source(root)
            real_read_csv = pd.read_csv
            opened: list[str] = []

            def guarded(path: object, *args: object, **kwargs: object):
                name = Path(path).name
                opened.append(name)
                if name in {"valid.csv", "test.csv"}:
                    raise AssertionError(f"forbidden source opened: {name}")
                return real_read_csv(path, *args, **kwargs)

            with mock.patch(
                "data.monotone_requirement_protocol.pd.read_csv",
                side_effect=guarded,
            ):
                protocol = _build(root)

            self.assertEqual(set(opened), {"train.csv", "Q_matrix.csv"})
            self.assertEqual(protocol.q_lookup["i00"], ("0", "1"))
            self.assertEqual(protocol.q_lookup["i02"], ("0", "2", "3"))
            optimizer_students = {c.student_id for c in protocol.optimizer.contexts}
            audit_students = {c.student_id for c in protocol.audit.contexts}
            self.assertFalse(optimizer_students & audit_students)
            self.assertTrue(
                all(_is_optimizer(student) for student in optimizer_students)
            )
            self.assertTrue(
                all(not _is_optimizer(student) for student in audit_students)
            )

            for feature_set in (protocol.optimizer, protocol.audit):
                query_indices = {r.source_index for r in feature_set.records}
                for record in feature_set.records:
                    self.assertFalse(_is_support(record.student_id, record.item_id))
                    self.assertEqual(
                        record.q_pair_id,
                        "|".join(protocol.q_lookup[record.item_id]),
                    )
                    group_indices = set(
                        source.index[
                            (source["stu_id"] == record.student_id)
                            & (source["exer_id"] == record.item_id)
                        ]
                    )
                    self.assertTrue(group_indices <= query_indices)

            for student in optimizer_students | audit_students:
                groups = source.loc[source["stu_id"] == student, "exer_id"].unique()
                support_count = sum(_is_support(student, item) for item in groups)
                query_count = len(groups) - support_count
                self.assertGreaterEqual(support_count, MIN_SUPPORT_GROUPS)
                self.assertGreaterEqual(query_count, MIN_QUERY_GROUPS)

            self.assertEqual(
                list(protocol.audit.metadata_frame().columns),
                [
                    "row_id",
                    "student_id",
                    "item_id",
                    "q_pair_id",
                    "q_count",
                    "eligible",
                ],
            )
            self.assertTrue(all(r.label is None for r in protocol.audit.records))
            self.assertTrue(all(r.label is not None for r in protocol.optimizer.records))

    def test_readiness_and_item_offset_use_support_only(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source = _write_source(root)
            protocol = _build(root)
            retained = {
                context.student_id
                for features in (protocol.optimizer, protocol.audit)
                for context in features.contexts
            }

            context = protocol.audit.contexts[0]
            student_rows = source[source["stu_id"] == context.student_id]
            attempts = np.zeros(len(protocol.concept_index), dtype=np.int32)
            correct = np.zeros(len(protocol.concept_index), dtype=np.float64)
            for item, group in student_rows.groupby("exer_id", sort=False):
                if not _is_support(context.student_id, str(item)):
                    continue
                response = float(group["label"].mean())
                for concept in protocol.q_lookup[str(item)]:
                    index = protocol.concept_index[concept]
                    attempts[index] += 1
                    correct[index] += response
            np.testing.assert_array_equal(context.support_group_count, attempts)
            np.testing.assert_allclose(
                context.readiness,
                (correct + 1.0) / (attempts + 2.0),
                rtol=0,
                atol=1e-7,
            )

            support_groups: list[tuple[str, str, float]] = []
            eligible_source = source[
                source["stu_id"].isin(retained)
                & source["stu_id"].map(_is_optimizer)
            ]
            for (student, item), group in eligible_source.groupby(
                ["stu_id", "exer_id"], sort=False
            ):
                if _is_support(str(student), str(item)):
                    support_groups.append(
                        (str(student), str(item), float(group["label"].mean()))
                    )
            expected_global = (
                sum(value for _, _, value in support_groups) + 1.0
            ) / (len(support_groups) + 2.0)
            self.assertAlmostEqual(protocol.global_ease, expected_global)

            item_values = [
                value for _, item, value in support_groups if item == "i00"
            ]
            item_ease = (
                sum(item_values) + ITEM_OFFSET_STRENGTH * expected_global
            ) / (len(item_values) + ITEM_OFFSET_STRENGTH)
            expected_item_offset = np.log(item_ease / (1.0 - item_ease))
            i00_records = [
                record
                for features in (protocol.optimizer, protocol.audit)
                for record in features.records
                if record.item_id == "i00"
            ]
            self.assertTrue(i00_records)
            self.assertTrue(
                all(
                    np.isclose(record.item_offset, expected_item_offset)
                    for record in i00_records
                )
            )
            global_offset = np.log(expected_global / (1.0 - expected_global))
            unseen = [
                record
                for record in protocol.audit.records
                if record.item_id.startswith("audit_only_")
            ]
            self.assertTrue(unseen)
            self.assertTrue(
                all(np.isclose(record.item_offset, global_offset) for record in unseen)
            )

    def test_audit_query_labels_are_withheld_and_flip_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as first_raw, tempfile.TemporaryDirectory() as second_raw:
            first = Path(first_raw)
            second = Path(second_raw)
            source = _write_source(first)
            selected_calls: list[set[int]] = []
            from data import monotone_requirement_protocol as module

            original_loader = module._read_selected_labels

            def recording_loader(path: Path, indices: list[int]):
                selected_calls.append(set(map(int, indices)))
                return original_loader(path, indices)

            with mock.patch.object(
                module, "_read_selected_labels", side_effect=recording_loader
            ):
                original = _build(first)
            audit_indices = {r.source_index for r in original.audit.records}
            self.assertEqual(len(selected_calls), 1)
            self.assertFalse(selected_calls[0] & audit_indices)
            self.assertFalse(original.audit_summary["audit_query_labels_materialized"])

            flipped = source.copy()
            flipped.loc[sorted(audit_indices), "label"] = (
                1 - flipped.loc[sorted(audit_indices), "label"]
            )
            flipped.to_csv(second / "train.csv", index=False)
            pd.read_csv(first / "Q_matrix.csv").to_csv(
                second / "Q_matrix.csv", index=False
            )
            changed = _build(second)

            self.assertEqual(original.protocol_sha256, changed.protocol_sha256)
            self.assertEqual(
                original.optimizer.input_sha256, changed.optimizer.input_sha256
            )
            self.assertEqual(original.audit.input_sha256, changed.audit.input_sha256)
            self.assertEqual(
                original.audit.row_order_sha256, changed.audit.row_order_sha256
            )
            self.assertNotEqual(
                original.audit_summary["train_full_sha256"],
                changed.audit_summary["train_full_sha256"],
            )
            for left, right in zip(
                original.audit.contexts, changed.audit.contexts, strict=True
            ):
                np.testing.assert_array_equal(left.readiness, right.readiness)
                np.testing.assert_array_equal(
                    left.support_group_count, right.support_group_count
                )
            left_batch = collate_requirement_batch(
                original.audit, range(min(32, len(original.audit.records))), device="cpu"
            )
            right_batch = collate_requirement_batch(
                changed.audit, range(min(32, len(changed.audit.records))), device="cpu"
            )
            torch.testing.assert_close(left_batch.readiness, right_batch.readiness)
            torch.testing.assert_close(left_batch.q_mask, right_batch.q_mask)
            mutated = pd.read_csv(first / "train.csv")
            old_labels = load_audit_query_labels(original)
            mutated.loc[0, "label"] = 1 - int(mutated.loc[0, "label"])
            mutated.to_csv(first / "train.csv", index=False)
            with self.assertRaisesRegex(
                RuntimeError, "sealed pre-reveal source"
            ):
                load_audit_query_labels(original)

            with self.assertRaisesRegex(RuntimeError, "sealed pre-reveal source"):
                load_audit_query_labels(
                    changed,
                    expected_train_sha256=original.audit_summary[
                        "train_full_sha256"
                    ],
                )

            torch.testing.assert_close(left_batch.item_offset, right_batch.item_offset)
            self.assertEqual(left_batch.row_ids, right_batch.row_ids)
            self.assertIsNone(left_batch.labels)

            new_labels = load_audit_query_labels(changed)
            self.assertEqual(old_labels["row_id"].tolist(), new_labels["row_id"].tolist())
            np.testing.assert_array_equal(
                new_labels["label"].to_numpy(),
                1 - old_labels["label"].to_numpy(),
            )

    def test_global_concept_coordinates_and_lattice_permutation_semantics(self) -> None:
        context = RequirementStudentContext(
            "student",
            np.asarray([0.2, 0.2, 0.8, 0.9, 0.4], dtype=np.float32),
            np.asarray([4, 4, 4, 4, 4], dtype=np.int32),
        )

        def record(
            row_id: str,
            targets: tuple[int, ...],
            *,
            eligible: bool = True,
        ) -> RequirementQueryRecord:
            return RequirementQueryRecord(
                row_id=row_id,
                source_index=int(row_id[1:]),
                student_id="student",
                item_id=row_id,
                context_index=0,
                target_concept_indices=targets,
                q_pair_id="|".join(map(str, targets)),
                q_count=len(targets),
                eligible=eligible,
                item_offset=0.0,
                label=1,
            )

        features = RequirementFeatureSet(
            split="fixture",
            contexts=(context,),
            records=(
                record("r0", (0, 2)),
                record("r1", (1, 2)),
                record("r2", (2, 0)),
                record("r3", (0, 1, 2, 3, 4), eligible=False),
            ),
            input_sha256="input",
            row_order_sha256="rows",
            label_sha256="labels",
        )
        batch = collate_requirement_batch(features, range(4), device="cpu")
        self.assertEqual(tuple(batch.readiness.shape), (4, 5))
        self.assertEqual(tuple(batch.q_mask.shape), (4, 5))
        torch.testing.assert_close(batch.b[0], batch.b[1])
        torch.testing.assert_close(batch.g[0], batch.g[1])

        # A positive-coordinate NCD fixture can distinguish concept identity.
        positive_weights = torch.arange(1, 6, dtype=torch.float32)
        coordinate_score = (batch.readiness * batch.q_mask).matmul(positive_weights)
        self.assertNotEqual(coordinate_score[0].item(), coordinate_score[1].item())
        # The (b, g) lattice is intentionally invariant to the same multiset.
        self.assertEqual(
            (batch.b[0] + batch.g[0]).item(),
            (batch.b[1] + batch.g[1]).item(),
        )
        # Reordering one Q set changes neither global coordinates nor lattice inputs.
        torch.testing.assert_close(batch.readiness[0], batch.readiness[2])
        torch.testing.assert_close(batch.q_mask[0], batch.q_mask[2])
        torch.testing.assert_close(batch.b[0], batch.b[2])
        torch.testing.assert_close(batch.g[0], batch.g[2])
        # MAX_Q only governs eligibility; a Q5 descriptive row still collates.
        self.assertFalse(batch.eligible[3])
        self.assertEqual(int(batch.q_mask[3].sum()), 5)


if __name__ == "__main__":
    unittest.main()
