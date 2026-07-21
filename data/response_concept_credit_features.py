from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from scripts.audit_conditional_response_signature import StudentProfile
from scripts.audit_target_local_pairing_protocol import (
    NUM_FOLDS,
    ValidationOnlyProtocol,
    build_validation_profiles,
)


ITEM_STATIC_FEATURE_NAMES = (
    "foldwise_item_ease",
    "foldwise_item_confidence",
)
RESPONSE_FEATURE_NAMES = (
    "response",
    "response_minus_foldwise_item_ease",
    "group_attempts_over_group_attempts_plus_one",
)
TARGET_SCOPE_BY_DATASET = {
    "MOOCRadar": "exact_zero",
    "NIPS34": "low_coverage",
}


def _hash_values(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ItemReferenceStatistics:
    ease: np.ndarray
    confidence: np.ndarray
    count: np.ndarray
    global_rate: float
    reference_students_sha256: str
    reference_rows_sha256: str


@dataclass(frozen=True)
class CreditStudentContext:
    student: str
    fold: int
    support_item_ids: np.ndarray
    support_q_indices: np.ndarray
    support_q_mask: np.ndarray
    support_responses: np.ndarray
    support_item_ease: np.ndarray
    support_item_confidence: np.ndarray
    support_group_attempt_confidence: np.ndarray


@dataclass(frozen=True)
class CreditQueryRecord:
    source_row_id: str
    student: str
    exercise: str
    context_index: int
    target_q_indices: np.ndarray
    target_q_mask: np.ndarray
    target_coverage: float
    in_c: bool
    in_c_strict: bool
    in_t: bool
    label: int | None


@dataclass(frozen=True)
class CreditFeatureSet:
    dataset: str
    split: str
    contexts: tuple[CreditStudentContext, ...]
    records: tuple[CreditQueryRecord, ...]
    record_indices_by_context: tuple[tuple[int, ...], ...]
    item_index: dict[str, int]
    concept_index: dict[str, int]
    q_lookup: dict[str, tuple[str, ...]]
    max_q_cardinality: int
    input_sha256: str
    row_order_sha256: str
    label_sha256: str | None
    slice_hashes: dict[str, str]

    @property
    def num_items(self) -> int:
        return len(self.item_index)

    @property
    def num_concepts(self) -> int:
        return len(self.concept_index)


@dataclass(frozen=True)
class CreditFeatureBuildResult:
    optimizer: CreditFeatureSet
    validation: CreditFeatureSet
    validation_profiles: tuple[StudentProfile, ...]
    audit: dict[str, Any]


@dataclass(frozen=True)
class CreditStudentBatch:
    support_item_ids: torch.Tensor
    support_q_indices: torch.Tensor
    support_q_mask: torch.Tensor
    support_responses: torch.Tensor
    support_item_ease: torch.Tensor
    support_item_confidence: torch.Tensor
    support_group_attempt_confidence: torch.Tensor
    support_mask: torch.Tensor
    query_context_indices: torch.Tensor
    target_q_indices: torch.Tensor
    target_q_mask: torch.Tensor
    labels: torch.Tensor | None
    source_row_ids: tuple[str, ...]
    students: tuple[str, ...]
    exercises: tuple[str, ...]
    in_c: np.ndarray
    in_c_strict: np.ndarray
    in_t: np.ndarray
    target_coverage: np.ndarray


def build_item_reference_statistics(
    profiles: Sequence[StudentProfile],
    *,
    item_index: dict[str, int],
) -> ItemReferenceStatistics:
    """Build item outcomes from all rows of disjoint reference students."""
    correct = np.zeros(len(item_index), dtype=np.float64)
    count = np.zeros(len(item_index), dtype=np.float64)
    row_ids: list[str] = []
    for profile in profiles:
        for frame in (profile.support, profile.query):
            for row in frame.itertuples(index=False):
                item = str(row.exer_id)
                if item not in item_index:
                    continue
                index = item_index[item] - 1
                correct[index] += int(row.label)
                count[index] += 1.0
                row_ids.append(str(row.source_row_id))
    global_rate = (float(correct.sum()) + 1.0) / (float(count.sum()) + 2.0)
    ease = (correct + 2.0 * global_rate) / (count + 2.0)
    confidence = count / (count + 2.0)
    return ItemReferenceStatistics(
        ease=ease.astype(np.float32),
        confidence=confidence.astype(np.float32),
        count=count,
        global_rate=float(global_rate),
        reference_students_sha256=_hash_values(
            sorted(str(profile.student) for profile in profiles)
        ),
        reference_rows_sha256=_hash_values(sorted(row_ids)),
    )


def _padded_q(
    item: str,
    *,
    q_lookup: dict[str, tuple[str, ...]],
    concept_index: dict[str, int],
    max_q_cardinality: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.zeros(max_q_cardinality, dtype=np.int64)
    mask = np.zeros(max_q_cardinality, dtype=bool)
    values = [concept_index[concept] for concept in q_lookup[item]]
    indices[: len(values)] = values
    mask[: len(values)] = True
    return indices, mask


def _student_context(
    profile: StudentProfile,
    *,
    statistics: ItemReferenceStatistics,
    item_index: dict[str, int],
    q_lookup: dict[str, tuple[str, ...]],
    concept_index: dict[str, int],
    max_q_cardinality: int,
) -> CreditStudentContext:
    support = profile.support.copy()
    items = support["exer_id"].astype(str)
    item_ids = items.map(item_index).to_numpy(dtype=np.int64)
    if np.any(item_ids <= 0):
        raise RuntimeError("Every support item must have a positive model ID.")
    q_rows = [
        _padded_q(
            item,
            q_lookup=q_lookup,
            concept_index=concept_index,
            max_q_cardinality=max_q_cardinality,
        )
        for item in items
    ]
    q_indices = np.stack([row[0] for row in q_rows])
    q_mask = np.stack([row[1] for row in q_rows])
    zero_based_ids = item_ids - 1
    attempts = items.map(items.value_counts()).to_numpy(dtype=np.float32)
    return CreditStudentContext(
        student=str(profile.student),
        fold=int(profile.fold),
        support_item_ids=item_ids,
        support_q_indices=q_indices,
        support_q_mask=q_mask,
        support_responses=support["label"].to_numpy(dtype=np.float32),
        support_item_ease=statistics.ease[zero_based_ids],
        support_item_confidence=statistics.confidence[zero_based_ids],
        support_group_attempt_confidence=attempts / (attempts + 1.0),
    )


def _coverage(
    support: pd.DataFrame,
    target_concepts: tuple[str, ...],
    *,
    q_lookup: dict[str, tuple[str, ...]],
) -> float:
    seen = {
        concept
        for item in support["exer_id"].astype(str)
        for concept in q_lookup[item]
    }
    return len(seen.intersection(target_concepts)) / len(target_concepts)


def _in_broad_c(
    support: pd.DataFrame,
    target_concepts: tuple[str, ...],
    *,
    q_lookup: dict[str, tuple[str, ...]],
) -> bool:
    if len(target_concepts) < 2:
        return False
    target = set(target_concepts)
    eligible_items = {
        item
        for item in support["exer_id"].astype(str)
        if len(q_lookup[item]) >= 2 and target.intersection(q_lookup[item])
    }
    covered = {
        concept
        for item in eligible_items
        for concept in q_lookup[item]
        if concept in target
    }
    return len(eligible_items) >= 3 and len(covered) >= 2


def _in_strict_c(
    support: pd.DataFrame,
    target_concepts: tuple[str, ...],
    *,
    q_lookup: dict[str, tuple[str, ...]],
) -> bool:
    if len(target_concepts) < 2:
        return False
    correct = {concept: 0.0 for concept in target_concepts}
    attempts = {concept: 0.0 for concept in target_concepts}
    for row in support.itertuples(index=False):
        q_values = set(q_lookup[str(row.exer_id)])
        for concept in target_concepts:
            if concept in q_values:
                correct[concept] += int(row.label)
                attempts[concept] += 1.0
    if any(attempts[concept] < 3.0 for concept in target_concepts):
        return False
    rates = [
        (correct[concept] + 1.0) / (attempts[concept] + 2.0)
        for concept in target_concepts
    ]
    return max(rates) - min(rates) >= 0.10


def _query_records(
    profiles: Sequence[StudentProfile],
    *,
    context_offset: int,
    q_lookup: dict[str, tuple[str, ...]],
    concept_index: dict[str, int],
    max_q_cardinality: int,
    dataset: str,
    labels_available: bool,
) -> tuple[CreditQueryRecord, ...]:
    records: list[CreditQueryRecord] = []
    for local_index, profile in enumerate(profiles):
        support_groups = set(profile.support["exer_id"].astype(str))
        ordered_query = profile.query.sort_values("source_row_id", kind="stable")
        for row in ordered_query.itertuples(index=False):
            item = str(row.exer_id)
            if item in support_groups:
                raise RuntimeError("A target group survived in support history.")
            target_concepts = q_lookup[item]
            q_indices, q_mask = _padded_q(
                item,
                q_lookup=q_lookup,
                concept_index=concept_index,
                max_q_cardinality=max_q_cardinality,
            )
            coverage = _coverage(
                profile.support,
                target_concepts,
                q_lookup=q_lookup,
            )
            target_scope = TARGET_SCOPE_BY_DATASET[dataset]
            in_t = coverage == 0.0 if target_scope == "exact_zero" else coverage < 0.5
            records.append(
                CreditQueryRecord(
                    source_row_id=str(row.source_row_id),
                    student=str(profile.student),
                    exercise=item,
                    context_index=context_offset + local_index,
                    target_q_indices=q_indices,
                    target_q_mask=q_mask,
                    target_coverage=float(coverage),
                    in_c=_in_broad_c(
                        profile.support,
                        target_concepts,
                        q_lookup=q_lookup,
                    ),
                    in_c_strict=_in_strict_c(
                        profile.support,
                        target_concepts,
                        q_lookup=q_lookup,
                    ),
                    in_t=bool(in_t),
                    label=int(row.label) if labels_available else None,
                )
            )
    return tuple(records)


def _feature_set(
    *,
    dataset: str,
    split: str,
    contexts: Sequence[CreditStudentContext],
    records: Sequence[CreditQueryRecord],
    item_index: dict[str, int],
    concept_index: dict[str, int],
    q_lookup: dict[str, tuple[str, ...]],
    max_q_cardinality: int,
) -> CreditFeatureSet:
    contexts = tuple(contexts)
    records = tuple(records)
    row_ids = [record.source_row_id for record in records]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("Query source_row_id values must be unique.")
    by_context: list[list[int]] = [[] for _ in contexts]
    for record_index, record in enumerate(records):
        by_context[record.context_index].append(record_index)
    if any(not values for values in by_context):
        raise RuntimeError("Every retained student context must have query rows.")

    digest = hashlib.sha256()
    digest.update(dataset.encode("utf-8"))
    digest.update(split.encode("utf-8"))
    digest.update(_hash_values(f"{k}:{v}" for k, v in sorted(item_index.items())).encode())
    digest.update(_hash_values(f"{k}:{v}" for k, v in sorted(concept_index.items())).encode())
    for context in contexts:
        digest.update(context.student.encode("utf-8"))
        digest.update(str(context.fold).encode("ascii"))
        for array in (
            context.support_item_ids,
            context.support_q_indices,
            context.support_q_mask,
            context.support_responses,
            context.support_item_ease,
            context.support_item_confidence,
            context.support_group_attempt_confidence,
        ):
            digest.update(_hash_array(array).encode("ascii"))
    for record in records:
        digest.update(
            "\x1f".join(
                (
                    record.source_row_id,
                    record.student,
                    record.exercise,
                    str(record.context_index),
                    str(record.target_coverage),
                    str(int(record.in_c)),
                    str(int(record.in_c_strict)),
                    str(int(record.in_t)),
                )
            ).encode("utf-8")
        )
        digest.update(_hash_array(record.target_q_indices).encode("ascii"))
        digest.update(_hash_array(record.target_q_mask).encode("ascii"))
    labels = [record.label for record in records]
    label_sha256 = None if any(v is None for v in labels) else _hash_values(labels)
    slice_hashes = {
        name: _hash_values(
            record.source_row_id
            for record in records
            if getattr(record, field)
        )
        for name, field in (("C", "in_c"), ("C_strict", "in_c_strict"), ("T", "in_t"))
    }
    return CreditFeatureSet(
        dataset=dataset,
        split=split,
        contexts=contexts,
        records=records,
        record_indices_by_context=tuple(tuple(values) for values in by_context),
        item_index=dict(item_index),
        concept_index=dict(concept_index),
        q_lookup=dict(q_lookup),
        max_q_cardinality=int(max_q_cardinality),
        input_sha256=digest.hexdigest(),
        row_order_sha256=_hash_values(row_ids),
        label_sha256=label_sha256,
        slice_hashes=slice_hashes,
    )


def build_credit_feature_sets(
    protocol: ValidationOnlyProtocol,
    *,
    optimizer_profiles: Sequence[StudentProfile],
) -> CreditFeatureBuildResult:
    if protocol.dataset not in TARGET_SCOPE_BY_DATASET:
        raise ValueError(f"Unsupported dataset: {protocol.dataset}.")
    profiles = list(optimizer_profiles)
    retained_items = sorted(
        {
            str(item)
            for profile in profiles
            for frame in (profile.support, profile.query)
            for item in frame["exer_id"].astype(str)
        }
    )
    validation_items = set(protocol.validation_support["exer_id"].astype(str)) | set(
        protocol.validation_query["exer_id"].astype(str)
    )
    if not validation_items <= set(retained_items):
        raise RuntimeError("Protocol was not finalized against retained optimizer items.")
    item_index = {item: index + 1 for index, item in enumerate(retained_items)}
    concepts = sorted({c for values in protocol.q_lookup.values() for c in values})
    concept_index = {concept: index + 1 for index, concept in enumerate(concepts)}
    max_q_cardinality = max(len(values) for values in protocol.q_lookup.values())

    optimizer_contexts: list[CreditStudentContext] = []
    optimizer_records: list[CreditQueryRecord] = []
    fold_audits: list[dict[str, Any]] = []
    for fold in range(NUM_FOLDS):
        reference = [profile for profile in profiles if profile.fold != fold]
        held = sorted(
            (profile for profile in profiles if profile.fold == fold),
            key=lambda value: value.student,
        )
        statistics = build_item_reference_statistics(reference, item_index=item_index)
        offset = len(optimizer_contexts)
        optimizer_contexts.extend(
            _student_context(
                profile,
                statistics=statistics,
                item_index=item_index,
                q_lookup=protocol.q_lookup,
                concept_index=concept_index,
                max_q_cardinality=max_q_cardinality,
            )
            for profile in held
        )
        optimizer_records.extend(
            _query_records(
                held,
                context_offset=offset,
                q_lookup=protocol.q_lookup,
                concept_index=concept_index,
                max_q_cardinality=max_q_cardinality,
                dataset=protocol.dataset,
                labels_available=True,
            )
        )
        fold_audits.append(
            {
                "fold": fold,
                "held_students": len(held),
                "reference_students": len(reference),
                "reference_students_sha256": statistics.reference_students_sha256,
                "reference_rows_sha256": statistics.reference_rows_sha256,
                "held_query_rows_used_for_item_statistics": False,
            }
        )

    validation_profiles = sorted(
        build_validation_profiles(
            protocol.validation_support,
            protocol.validation_query,
            q_lookup=protocol.q_lookup,
        ),
        key=lambda value: value.student,
    )
    validation_statistics = build_item_reference_statistics(
        profiles, item_index=item_index
    )
    validation_contexts = [
        _student_context(
            profile,
            statistics=validation_statistics,
            item_index=item_index,
            q_lookup=protocol.q_lookup,
            concept_index=concept_index,
            max_q_cardinality=max_q_cardinality,
        )
        for profile in validation_profiles
    ]
    validation_records = _query_records(
        validation_profiles,
        context_offset=0,
        q_lookup=protocol.q_lookup,
        concept_index=concept_index,
        max_q_cardinality=max_q_cardinality,
        dataset=protocol.dataset,
        labels_available=False,
    )
    optimizer = _feature_set(
        dataset=protocol.dataset,
        split="optimizer_oof_query",
        contexts=optimizer_contexts,
        records=optimizer_records,
        item_index=item_index,
        concept_index=concept_index,
        q_lookup=protocol.q_lookup,
        max_q_cardinality=max_q_cardinality,
    )
    validation = _feature_set(
        dataset=protocol.dataset,
        split="validation_query",
        contexts=validation_contexts,
        records=validation_records,
        item_index=item_index,
        concept_index=concept_index,
        q_lookup=protocol.q_lookup,
        max_q_cardinality=max_q_cardinality,
    )
    if optimizer.label_sha256 is None or validation.label_sha256 is not None:
        raise RuntimeError("Optimizer/validation label separation failed.")
    c_records = [record for record in validation.records if record.in_c]
    strict_records = [record for record in validation.records if record.in_c_strict]
    return CreditFeatureBuildResult(
        optimizer=optimizer,
        validation=validation,
        validation_profiles=tuple(validation_profiles),
        audit={
            "schema_version": 1,
            "item_static_feature_names": ITEM_STATIC_FEATURE_NAMES,
            "response_feature_names": RESPONSE_FEATURE_NAMES,
            "target_scope": TARGET_SCOPE_BY_DATASET[protocol.dataset],
            "items": len(item_index),
            "concepts": len(concept_index),
            "max_q_cardinality": max_q_cardinality,
            "optimizer_contexts": len(optimizer.contexts),
            "optimizer_query_rows": len(optimizer.records),
            "validation_contexts": len(validation.contexts),
            "validation_query_rows": len(validation.records),
            "validation_c_rows": len(c_records),
            "validation_c_students": len({record.student for record in c_records}),
            "validation_c_strict_rows": len(strict_records),
            "validation_c_strict_students": len({record.student for record in strict_records}),
            "validation_t_rows": sum(record.in_t for record in validation.records),
            "fold_item_statistics": fold_audits,
            "validation_item_statistics": {
                "reference_students_sha256": validation_statistics.reference_students_sha256,
                "reference_rows_sha256": validation_statistics.reference_rows_sha256,
                "optimizer_students_only": True,
                "validation_students_used": False,
            },
            "optimizer_input_sha256": optimizer.input_sha256,
            "validation_input_sha256": validation.input_sha256,
            "optimizer_row_order_sha256": optimizer.row_order_sha256,
            "validation_row_order_sha256": validation.row_order_sha256,
            "validation_slice_sha256": validation.slice_hashes,
        },
    )


def collate_credit_student_batch(
    features: CreditFeatureSet,
    context_indices: Sequence[int],
    *,
    device: torch.device | str,
) -> CreditStudentBatch:
    selected = [int(index) for index in context_indices]
    if not selected:
        raise ValueError("A student batch cannot be empty.")
    if min(selected) < 0 or max(selected) >= len(features.contexts):
        raise IndexError("Student context index is out of range.")
    contexts = [features.contexts[index] for index in selected]
    max_support = max(len(context.support_item_ids) for context in contexts)
    batch_size = len(contexts)
    max_q = features.max_q_cardinality
    support_item_ids = np.zeros((batch_size, max_support), dtype=np.int64)
    support_q_indices = np.zeros((batch_size, max_support, max_q), dtype=np.int64)
    support_q_mask = np.zeros((batch_size, max_support, max_q), dtype=bool)
    support_responses = np.zeros((batch_size, max_support), dtype=np.float32)
    support_item_ease = np.zeros((batch_size, max_support), dtype=np.float32)
    support_item_confidence = np.zeros((batch_size, max_support), dtype=np.float32)
    support_group_confidence = np.zeros((batch_size, max_support), dtype=np.float32)
    support_mask = np.zeros((batch_size, max_support), dtype=bool)
    for row, context in enumerate(contexts):
        length = len(context.support_item_ids)
        support_item_ids[row, :length] = context.support_item_ids
        support_q_indices[row, :length] = context.support_q_indices
        support_q_mask[row, :length] = context.support_q_mask
        support_responses[row, :length] = context.support_responses
        support_item_ease[row, :length] = context.support_item_ease
        support_item_confidence[row, :length] = context.support_item_confidence
        support_group_confidence[row, :length] = context.support_group_attempt_confidence
        support_mask[row, :length] = True

    records: list[CreditQueryRecord] = []
    local_context: list[int] = []
    for local_index, global_index in enumerate(selected):
        indices = features.record_indices_by_context[global_index]
        records.extend(features.records[index] for index in indices)
        local_context.extend([local_index] * len(indices))
    target_q_indices = np.stack([record.target_q_indices for record in records])
    target_q_mask = np.stack([record.target_q_mask for record in records])
    labels = [record.label for record in records]
    tensor = lambda value, dtype=None: torch.as_tensor(value, dtype=dtype, device=device)
    label_tensor = None if any(v is None for v in labels) else tensor(labels, torch.float32)
    return CreditStudentBatch(
        support_item_ids=tensor(support_item_ids, torch.long),
        support_q_indices=tensor(support_q_indices, torch.long),
        support_q_mask=tensor(support_q_mask, torch.bool),
        support_responses=tensor(support_responses, torch.float32),
        support_item_ease=tensor(support_item_ease, torch.float32),
        support_item_confidence=tensor(support_item_confidence, torch.float32),
        support_group_attempt_confidence=tensor(support_group_confidence, torch.float32),
        support_mask=tensor(support_mask, torch.bool),
        query_context_indices=tensor(local_context, torch.long),
        target_q_indices=tensor(target_q_indices, torch.long),
        target_q_mask=tensor(target_q_mask, torch.bool),
        labels=label_tensor,
        source_row_ids=tuple(record.source_row_id for record in records),
        students=tuple(record.student for record in records),
        exercises=tuple(record.exercise for record in records),
        in_c=np.asarray([record.in_c for record in records], dtype=bool),
        in_c_strict=np.asarray([record.in_c_strict for record in records], dtype=bool),
        in_t=np.asarray([record.in_t for record in records], dtype=bool),
        target_coverage=np.asarray(
            [record.target_coverage for record in records], dtype=np.float32
        ),
    )
