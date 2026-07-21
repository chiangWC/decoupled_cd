from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from scripts.audit_conditional_response_signature import (
    SIGNATURE_NAMES,
    StudentProfile,
    build_reference_statistics,
    signature_at_bin,
)
from scripts.audit_target_local_pairing_protocol import (
    DonorMapping,
    ValidationOnlyProtocol,
    build_donor_mapping,
    build_validation_profiles,
)


ITEM_EASE_EPS = 0.01
ITEM_NUMERIC_NAMES = (
    *SIGNATURE_NAMES,
    "item_ease_logit_eps_0.01",
    "log1p_item_count",
    "q_cardinality",
)
SUPPORT_STATISTIC_NAMES = (
    "theta_logit",
    "raw_accuracy",
    "log1p_support_rows",
    "log1p_unique_support_items",
    "log1p_correct_rows",
    "log1p_incorrect_rows",
)
RESPONSE_FEATURE_NAMES = (
    "response",
    "response_minus_foldwise_item_ease",
    "group_attempts_over_group_attempts_plus_one",
)

if len(ITEM_NUMERIC_NAMES) != 16:
    raise RuntimeError("The preregistered item-numeric schema must have 16 fields.")
if len(SUPPORT_STATISTIC_NAMES) != 6:
    raise RuntimeError("The preregistered support-statistic schema must have 6 fields.")


def _hash_values(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def _hash_array(digest: Any, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())


def _feature_input_sha256(
    *,
    schema_sha256: str,
    contexts: Sequence[StudentFeatureContext],
    records: Sequence[QueryFeatureRecord],
    model_item_index: dict[str, int],
    q_item_index: dict[str, int],
    q_concept_indices: tuple[tuple[int, ...], ...],
    concept_index: dict[str, int],
) -> str:
    """Hash every semantic input, including the identities behind indices."""
    digest = hashlib.sha256()
    digest.update(schema_sha256.encode("ascii"))
    digest.update(
        _hash_values(
            f"{item}:{index}"
            for item, index in sorted(model_item_index.items())
        ).encode("ascii")
    )
    digest.update(
        _hash_values(
            f"{item}:{index}"
            for item, index in sorted(q_item_index.items())
        ).encode("ascii")
    )
    digest.update(
        _hash_values(
            f"{concept}:{index}"
            for concept, index in sorted(concept_index.items())
        ).encode("ascii")
    )
    digest.update(
        _hash_values(
            f"{row}:{','.join(map(str, concept_rows))}"
            for row, concept_rows in enumerate(q_concept_indices)
        ).encode("ascii")
    )
    for context in contexts:
        digest.update(context.student.encode("utf-8"))
        digest.update(str(context.fold).encode("ascii"))
        for array in (
            context.support_model_item_ids,
            context.support_q_item_rows,
            context.support_item_numeric,
            context.support_responses,
            context.support_item_ease,
            context.support_group_attempt_confidence,
            context.support_statistics,
        ):
            _hash_array(digest, array)
    for record in records:
        digest.update(
            "\x1f".join(
                (
                    record.source_row_id,
                    record.student,
                    record.exercise,
                    record.donor_exercise,
                    record.coverage_bucket,
                    record.q_cardinality_bucket,
                )
            ).encode("utf-8")
        )
        _hash_array(digest, record.target_item_numeric)
        _hash_array(digest, record.donor_item_numeric)
        digest.update(
            np.asarray(
                [
                    record.context_index,
                    record.target_model_item_id,
                    record.target_q_item_row,
                    record.donor_model_item_id,
                    record.donor_q_item_row,
                ],
                dtype="<i8",
            ).tobytes()
        )
    return digest.hexdigest()


def _clipped_logit(value: float) -> float:
    clipped = float(np.clip(value, ITEM_EASE_EPS, 1.0 - ITEM_EASE_EPS))
    return float(np.log(clipped / (1.0 - clipped)))


def _coverage_bucket(
    seen_concepts: frozenset[str],
    target_concepts: tuple[str, ...],
) -> str:
    overlap = len(seen_concepts.intersection(target_concepts))
    if overlap == 0:
        return "exact_zero"
    if overlap == len(target_concepts):
        return "full"
    return "partial"


@dataclass(frozen=True)
class FrozenScaler:
    names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    schema_sha256: str

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        *,
        names: Sequence[str],
    ) -> "FrozenScaler":
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(names):
            raise ValueError("Scaler values do not match the declared schema.")
        scaler = StandardScaler()
        scaler.fit(matrix)
        mean = np.asarray(scaler.mean_, dtype=np.float64)
        scale = np.asarray(scaler.scale_, dtype=np.float64)
        schema_sha256 = _hash_values(
            (
                *names,
                *mean.tolist(),
                *scale.tolist(),
            )
        )
        return cls(tuple(names), mean, scale, schema_sha256)

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.shape[-1] != len(self.names):
            raise ValueError("Values do not match the fitted scaler schema.")
        return ((matrix - self.mean) / self.scale).astype(np.float32)


@dataclass(frozen=True)
class StudentFeatureContext:
    student: str
    fold: int
    support_model_item_ids: np.ndarray
    support_q_item_rows: np.ndarray
    support_item_numeric: np.ndarray
    support_responses: np.ndarray
    support_item_ease: np.ndarray
    support_group_attempt_confidence: np.ndarray
    support_statistics: np.ndarray


@dataclass(frozen=True)
class QueryFeatureRecord:
    source_row_id: str
    student: str
    exercise: str
    context_index: int
    target_model_item_id: int
    target_q_item_row: int
    target_item_numeric: np.ndarray
    donor_exercise: str
    donor_model_item_id: int
    donor_q_item_row: int
    donor_item_numeric: np.ndarray
    coverage_bucket: str
    q_cardinality_bucket: str
    label: int | None


@dataclass(frozen=True)
class PairingFeatureSet:
    dataset: str
    split: str
    contexts: tuple[StudentFeatureContext, ...]
    records: tuple[QueryFeatureRecord, ...]
    model_item_index: dict[str, int]
    q_item_index: dict[str, int]
    q_concept_indices: tuple[tuple[int, ...], ...]
    concept_index: dict[str, int]
    item_numeric_scaler: FrozenScaler
    support_statistics_scaler: FrozenScaler
    optimizer_item_frequency: dict[str, int]
    input_sha256: str
    row_order_sha256: str
    label_sha256: str | None
    schema_sha256: str

    @property
    def num_items(self) -> int:
        return len(self.model_item_index)

    @property
    def num_concepts(self) -> int:
        return len(self.concept_index)


@dataclass(frozen=True)
class FeatureBuildResult:
    optimizer: PairingFeatureSet
    validation: PairingFeatureSet
    validation_profiles: tuple[StudentProfile, ...]
    validation_statistics: Any
    validation_binner: Any
    validation_donor_rep0: DonorMapping
    audit: dict[str, Any]


@dataclass(frozen=True)
class PairingBatch:
    support_item_ids: torch.Tensor
    support_q_multi_hot: torch.Tensor
    support_item_numeric: torch.Tensor
    support_responses: torch.Tensor
    support_item_ease: torch.Tensor
    support_group_attempt_confidence: torch.Tensor
    support_mask: torch.Tensor
    support_statistics: torch.Tensor
    target_item_ids: torch.Tensor
    target_q_multi_hot: torch.Tensor
    target_item_numeric: torch.Tensor
    donor_target_item_ids: torch.Tensor
    donor_target_q_multi_hot: torch.Tensor
    donor_target_item_numeric: torch.Tensor
    labels: torch.Tensor | None
    source_row_ids: tuple[str, ...]
    students: tuple[str, ...]
    exercises: tuple[str, ...]


@dataclass(frozen=True)
class _RawFeatureSet:
    dataset: str
    split: str
    contexts: tuple[StudentFeatureContext, ...]
    records: tuple[QueryFeatureRecord, ...]


def _item_numeric(
    *,
    item: str,
    student: str,
    theta: float,
    statistics: Any,
    binner: Any,
    item_index: dict[str, int],
    q_lookup: dict[str, tuple[str, ...]],
) -> np.ndarray:
    index = item_index[item]
    theta_bin = binner.assign(student=student, theta=theta)
    signature = signature_at_bin(statistics.signatures[index], theta_bin)
    values = np.concatenate(
        (
            signature,
            np.asarray(
                [
                    _clipped_logit(float(statistics.item_ease[index])),
                    np.log1p(float(statistics.item_count[index])),
                    float(len(q_lookup[item])),
                ],
                dtype=np.float64,
            ),
        )
    )
    if values.shape != (len(ITEM_NUMERIC_NAMES),):
        raise RuntimeError("Item numeric builder violated the frozen schema.")
    return values


def _support_statistics(profile: StudentProfile) -> np.ndarray:
    responses = profile.support["label"].to_numpy(dtype=np.float64)
    values = np.asarray(
        [
            profile.theta,
            profile.raw_accuracy,
            np.log1p(len(profile.support)),
            np.log1p(profile.support["exer_id"].astype(str).nunique()),
            np.log1p(float(responses.sum())),
            np.log1p(float((1.0 - responses).sum())),
        ],
        dtype=np.float64,
    )
    if values.shape != (len(SUPPORT_STATISTIC_NAMES),):
        raise RuntimeError("Support statistic builder violated the frozen schema.")
    return values


def _student_context(
    profile: StudentProfile,
    *,
    fold: int,
    statistics: Any,
    binner: Any,
    item_index: dict[str, int],
    q_item_index: dict[str, int],
    q_lookup: dict[str, tuple[str, ...]],
    model_item_index: dict[str, int],
) -> StudentFeatureContext:
    support = profile.support.copy()
    item_strings = support["exer_id"].astype(str)
    group_attempts = item_strings.map(item_strings.value_counts()).to_numpy(
        dtype=np.float64
    )
    responses = support["label"].to_numpy(dtype=np.float64)
    ease = np.asarray(
        [
            float(statistics.item_ease[item_index[item]])
            for item in item_strings
        ],
        dtype=np.float64,
    )
    model_ids = np.asarray(
        [
            (
                model_item_index[item]
                if statistics.item_count[item_index[item]] > 0
                else 0
            )
            for item in item_strings
        ],
        dtype=np.int64,
    )
    numeric = np.stack(
        [
            _item_numeric(
                item=item,
                student=profile.student,
                theta=profile.theta,
                statistics=statistics,
                binner=binner,
                item_index=item_index,
                q_lookup=q_lookup,
            )
            for item in item_strings
        ]
    )
    return StudentFeatureContext(
        student=profile.student,
        fold=fold,
        support_model_item_ids=model_ids,
        support_q_item_rows=np.asarray(
            [q_item_index[item] for item in item_strings],
            dtype=np.int64,
        ),
        support_item_numeric=numeric,
        support_responses=responses,
        support_item_ease=ease,
        support_group_attempt_confidence=(
            group_attempts / (group_attempts + 1.0)
        ),
        support_statistics=_support_statistics(profile),
    )


def _records_for_profiles(
    profiles: Sequence[StudentProfile],
    *,
    contexts: Sequence[StudentFeatureContext],
    donor_mapping: DonorMapping,
    statistics: Any,
    binner: Any,
    item_index: dict[str, int],
    q_item_index: dict[str, int],
    q_lookup: dict[str, tuple[str, ...]],
    model_item_index: dict[str, int],
    labels_available: bool,
) -> tuple[QueryFeatureRecord, ...]:
    context_index = {
        context.student: index for index, context in enumerate(contexts)
    }
    donor_by_row = donor_mapping.rows.set_index("source_row_id")[
        "donor_target_item"
    ].astype(str).to_dict()
    records: list[QueryFeatureRecord] = []
    for profile in profiles:
        target_cache: dict[str, tuple[int, int, np.ndarray]] = {}
        for item in sorted(set(profile.query["exer_id"].astype(str))):
            index = item_index[item]
            target_cache[item] = (
                model_item_index[item] if statistics.item_count[index] > 0 else 0,
                q_item_index[item],
                _item_numeric(
                    item=item,
                    student=profile.student,
                    theta=profile.theta,
                    statistics=statistics,
                    binner=binner,
                    item_index=item_index,
                    q_lookup=q_lookup,
                ),
            )
        for row in profile.query.itertuples(index=False):
            source_row_id = str(row.source_row_id)
            item = str(row.exer_id)
            donor = donor_by_row[source_row_id]
            target_id, target_q_row, target_numeric = target_cache[item]
            donor_id, donor_q_row, donor_numeric = target_cache[donor]
            records.append(
                QueryFeatureRecord(
                    source_row_id=source_row_id,
                    student=profile.student,
                    exercise=item,
                    context_index=context_index[profile.student],
                    target_model_item_id=target_id,
                    target_q_item_row=target_q_row,
                    target_item_numeric=target_numeric,
                    donor_exercise=donor,
                    donor_model_item_id=donor_id,
                    donor_q_item_row=donor_q_row,
                    donor_item_numeric=donor_numeric,
                    coverage_bucket=_coverage_bucket(
                        profile.seen_concepts, q_lookup[item]
                    ),
                    q_cardinality_bucket=(
                        str(len(q_lookup[item]))
                        if len(q_lookup[item]) <= 2
                        else "3_plus"
                    ),
                    label=int(row.label) if labels_available else None,
                )
            )
    records.sort(key=lambda value: (value.student, value.source_row_id))
    row_ids = [record.source_row_id for record in records]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("Query feature source_row_id values are not unique.")
    return tuple(records)


def _raw_feature_set(
    *,
    dataset: str,
    split: str,
    profiles: Sequence[StudentProfile],
    statistics: Any,
    binner: Any,
    donor_mapping: DonorMapping,
    item_index: dict[str, int],
    q_item_index: dict[str, int],
    q_lookup: dict[str, tuple[str, ...]],
    model_item_index: dict[str, int],
    labels_available: bool,
) -> _RawFeatureSet:
    ordered_profiles = sorted(profiles, key=lambda value: value.student)
    contexts = tuple(
        _student_context(
            profile,
            fold=profile.fold,
            statistics=statistics,
            binner=binner,
            item_index=item_index,
            q_item_index=q_item_index,
            q_lookup=q_lookup,
            model_item_index=model_item_index,
        )
        for profile in ordered_profiles
    )
    records = _records_for_profiles(
        ordered_profiles,
        contexts=contexts,
        donor_mapping=donor_mapping,
        statistics=statistics,
        binner=binner,
        item_index=item_index,
        q_item_index=q_item_index,
        q_lookup=q_lookup,
        model_item_index=model_item_index,
        labels_available=labels_available,
    )
    return _RawFeatureSet(dataset, split, contexts, records)


def _fit_scalers(raw: _RawFeatureSet) -> tuple[FrozenScaler, FrozenScaler]:
    item_rows = [context.support_item_numeric for context in raw.contexts]
    item_rows.extend(
        record.target_item_numeric.reshape(1, -1) for record in raw.records
    )
    item_scaler = FrozenScaler.fit(
        np.concatenate(item_rows, axis=0),
        names=ITEM_NUMERIC_NAMES,
    )
    support_scaler = FrozenScaler.fit(
        np.stack([context.support_statistics for context in raw.contexts]),
        names=SUPPORT_STATISTIC_NAMES,
    )
    return item_scaler, support_scaler


def _scaled_feature_set(
    raw: _RawFeatureSet,
    *,
    item_numeric_scaler: FrozenScaler,
    support_statistics_scaler: FrozenScaler,
    model_item_index: dict[str, int],
    q_item_index: dict[str, int],
    q_concept_indices: tuple[tuple[int, ...], ...],
    concept_index: dict[str, int],
    optimizer_item_frequency: dict[str, int],
) -> PairingFeatureSet:
    contexts = tuple(
        replace(
            context,
            support_item_numeric=item_numeric_scaler.transform(
                context.support_item_numeric
            ),
            support_statistics=support_statistics_scaler.transform(
                context.support_statistics
            ),
        )
        for context in raw.contexts
    )
    records = tuple(
        replace(
            record,
            target_item_numeric=item_numeric_scaler.transform(
                record.target_item_numeric
            ),
            donor_item_numeric=item_numeric_scaler.transform(
                record.donor_item_numeric
            ),
        )
        for record in raw.records
    )

    schema_sha256 = _hash_values(
        (
            *ITEM_NUMERIC_NAMES,
            *SUPPORT_STATISTIC_NAMES,
            *RESPONSE_FEATURE_NAMES,
            f"ease_eps={ITEM_EASE_EPS}",
            f"item_scaler={item_numeric_scaler.schema_sha256}",
            f"support_scaler={support_statistics_scaler.schema_sha256}",
        )
    )
    input_sha256 = _feature_input_sha256(
        schema_sha256=schema_sha256,
        contexts=contexts,
        records=records,
        model_item_index=model_item_index,
        q_item_index=q_item_index,
        q_concept_indices=q_concept_indices,
        concept_index=concept_index,
    )
    row_order_sha256 = _hash_values(record.source_row_id for record in records)
    labels = [record.label for record in records]
    label_sha256 = (
        None
        if any(label is None for label in labels)
        else _hash_values(int(label) for label in labels)
    )
    return PairingFeatureSet(
        dataset=raw.dataset,
        split=raw.split,
        contexts=contexts,
        records=records,
        model_item_index=dict(model_item_index),
        q_item_index=dict(q_item_index),
        q_concept_indices=q_concept_indices,
        concept_index=dict(concept_index),
        item_numeric_scaler=item_numeric_scaler,
        support_statistics_scaler=support_statistics_scaler,
        optimizer_item_frequency=dict(optimizer_item_frequency),
        input_sha256=input_sha256,
        row_order_sha256=row_order_sha256,
        label_sha256=label_sha256,
        schema_sha256=schema_sha256,
    )


def build_feature_sets(
    protocol: ValidationOnlyProtocol,
    *,
    optimizer_profiles: Sequence[StudentProfile],
    expected_validation_mapping_sha256: str | None = None,
) -> FeatureBuildResult:
    profiles = list(optimizer_profiles)
    retained_items = {
        str(item)
        for profile in profiles
        for frame in (profile.support, profile.query)
        for item in frame["exer_id"].astype(str)
    }
    validation_items = set(
        protocol.validation_support["exer_id"].astype(str)
    ) | set(protocol.validation_query["exer_id"].astype(str))
    if not validation_items <= retained_items:
        missing = sorted(validation_items - retained_items)
        raise RuntimeError(
            "Validation protocol was not finalized against retained optimizer "
            f"items; first missing items={missing[:10]}"
        )

    q_lookup = protocol.q_lookup
    q_items = sorted(q_lookup)
    q_item_index = {item: index for index, item in enumerate(q_items)}
    concepts = sorted(
        {concept for values in q_lookup.values() for concept in values}
    )
    concept_index = {
        concept: index for index, concept in enumerate(concepts)
    }
    q_concept_indices = tuple(
        tuple(concept_index[concept] for concept in q_lookup[item])
        for item in q_items
    )
    item_index = q_item_index
    model_item_index = {
        item: index + 1 for index, item in enumerate(sorted(retained_items))
    }

    optimizer_item_frequency: dict[str, int] = {
        item: 0 for item in sorted(retained_items)
    }
    for profile in profiles:
        for frame in (profile.support, profile.query):
            counts = frame["exer_id"].astype(str).value_counts()
            for item, count in counts.items():
                optimizer_item_frequency[str(item)] += int(count)

    raw_contexts: list[StudentFeatureContext] = []
    raw_records: list[QueryFeatureRecord] = []
    fold_audits = []
    context_offset = 0
    for fold in range(5):
        reference = [profile for profile in profiles if profile.fold != fold]
        held = [profile for profile in profiles if profile.fold == fold]
        statistics, binner = build_reference_statistics(
            reference,
            items=q_items,
            item_index=item_index,
            q_lookup=q_lookup,
        )
        donor = build_donor_mapping(
            held,
            statistics=statistics,
            item_index=item_index,
            q_lookup=q_lookup,
            split="optimizer_oof_query",
            fold=fold,
            replicate=0,
        )
        raw_fold = _raw_feature_set(
            dataset=protocol.dataset,
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
        raw_contexts.extend(raw_fold.contexts)
        raw_records.extend(
            replace(record, context_index=record.context_index + context_offset)
            for record in raw_fold.records
        )
        context_offset += len(raw_fold.contexts)
        fold_audits.append(
            {
                "fold": fold,
                "reference_students": len(reference),
                "held_students": len(held),
                "reference_students_sha256": statistics.reference_students_hash,
                "reference_rows_sha256": statistics.reference_rows_hash,
                "donor_mapping_sha256": donor.audit["mapping_sha256"],
                "zero_count_item_rows_map_to_unk": True,
            }
        )
    raw_records.sort(key=lambda value: (value.student, value.source_row_id))
    raw_optimizer = _RawFeatureSet(
        protocol.dataset,
        "optimizer_oof_query",
        tuple(raw_contexts),
        tuple(raw_records),
    )
    item_scaler, support_scaler = _fit_scalers(raw_optimizer)

    validation_profiles = build_validation_profiles(
        protocol.validation_support,
        protocol.validation_query,
        q_lookup=q_lookup,
    )
    validation_statistics, validation_binner = build_reference_statistics(
        profiles,
        items=q_items,
        item_index=item_index,
        q_lookup=q_lookup,
    )
    validation_donor = build_donor_mapping(
        validation_profiles,
        statistics=validation_statistics,
        item_index=item_index,
        q_lookup=q_lookup,
        split="validation_query",
        fold=None,
        replicate=0,
    )
    if (
        expected_validation_mapping_sha256 is not None
        and validation_donor.audit["mapping_sha256"]
        != expected_validation_mapping_sha256
    ):
        raise RuntimeError("Validation donor mapping does not match locked audit.")
    raw_validation = _raw_feature_set(
        dataset=protocol.dataset,
        split="validation_query",
        profiles=validation_profiles,
        statistics=validation_statistics,
        binner=validation_binner,
        donor_mapping=validation_donor,
        item_index=item_index,
        q_item_index=q_item_index,
        q_lookup=q_lookup,
        model_item_index=model_item_index,
        labels_available=False,
    )

    optimizer = _scaled_feature_set(
        raw_optimizer,
        item_numeric_scaler=item_scaler,
        support_statistics_scaler=support_scaler,
        model_item_index=model_item_index,
        q_item_index=q_item_index,
        q_concept_indices=q_concept_indices,
        concept_index=concept_index,
        optimizer_item_frequency=optimizer_item_frequency,
    )
    validation = _scaled_feature_set(
        raw_validation,
        item_numeric_scaler=item_scaler,
        support_statistics_scaler=support_scaler,
        model_item_index=model_item_index,
        q_item_index=q_item_index,
        q_concept_indices=q_concept_indices,
        concept_index=concept_index,
        optimizer_item_frequency=optimizer_item_frequency,
    )
    return FeatureBuildResult(
        optimizer=optimizer,
        validation=validation,
        validation_profiles=tuple(validation_profiles),
        validation_statistics=validation_statistics,
        validation_binner=validation_binner,
        validation_donor_rep0=validation_donor,
        audit={
            "schema_sha256": optimizer.schema_sha256,
            "item_numeric_names": ITEM_NUMERIC_NAMES,
            "support_statistic_names": SUPPORT_STATISTIC_NAMES,
            "response_feature_names": RESPONSE_FEATURE_NAMES,
            "item_ease_eps": ITEM_EASE_EPS,
            "item_numeric_scaler_sha256": item_scaler.schema_sha256,
            "support_statistics_scaler_sha256": support_scaler.schema_sha256,
            "retained_model_items": len(model_item_index),
            "concepts": len(concept_index),
            "optimizer_input_sha256": optimizer.input_sha256,
            "validation_input_sha256": validation.input_sha256,
            "optimizer_row_order_sha256": optimizer.row_order_sha256,
            "validation_row_order_sha256": validation.row_order_sha256,
            "folds": fold_audits,
            "validation_donor_mapping_sha256": validation_donor.audit[
                "mapping_sha256"
            ],
        },
    )


def replace_validation_donors(
    features: PairingFeatureSet,
    mapping: DonorMapping,
) -> PairingFeatureSet:
    if features.split != "validation_query":
        raise ValueError("Donor replacement is validation-only.")
    donor_by_row = mapping.rows.set_index("source_row_id")[
        "donor_target_item"
    ].astype(str).to_dict()
    target_by_student_item = {
        (record.student, record.exercise): record
        for record in features.records
    }
    if set(donor_by_row) != {
        record.source_row_id for record in features.records
    }:
        raise RuntimeError("Donor mapping and validation features are misaligned.")
    records = []
    for record in features.records:
        donor_item = donor_by_row[record.source_row_id]
        donor_record = target_by_student_item.get((record.student, donor_item))
        if donor_record is None:
            raise RuntimeError("Donor item is absent from the same student's query.")
        records.append(
            replace(
                record,
                donor_exercise=donor_item,
                donor_model_item_id=donor_record.target_model_item_id,
                donor_q_item_row=donor_record.target_q_item_row,
                donor_item_numeric=donor_record.target_item_numeric,
            )
        )
    updated_records = tuple(records)
    input_sha256 = _feature_input_sha256(
        schema_sha256=features.schema_sha256,
        contexts=features.contexts,
        records=updated_records,
        model_item_index=features.model_item_index,
        q_item_index=features.q_item_index,
        q_concept_indices=features.q_concept_indices,
        concept_index=features.concept_index,
    )
    return replace(
        features,
        records=updated_records,
        input_sha256=input_sha256,
    )


def collate_pairing_batch(
    features: PairingFeatureSet,
    record_indices: Sequence[int],
    *,
    device: torch.device | str,
) -> PairingBatch:
    indices = [int(index) for index in record_indices]
    if not indices:
        raise ValueError("A pairing batch cannot be empty.")
    records = [features.records[index] for index in indices]
    contexts = [features.contexts[record.context_index] for record in records]
    batch = len(records)
    max_support = max(len(context.support_model_item_ids) for context in contexts)
    concepts = features.num_concepts

    support_item_ids = np.zeros((batch, max_support), dtype=np.int64)
    support_q = np.zeros((batch, max_support, concepts), dtype=np.float32)
    support_numeric = np.zeros(
        (batch, max_support, len(ITEM_NUMERIC_NAMES)), dtype=np.float32
    )
    support_responses = np.zeros((batch, max_support), dtype=np.float32)
    support_ease = np.zeros((batch, max_support), dtype=np.float32)
    support_confidence = np.zeros((batch, max_support), dtype=np.float32)
    support_mask = np.zeros((batch, max_support), dtype=bool)
    support_statistics = np.zeros(
        (batch, len(SUPPORT_STATISTIC_NAMES)), dtype=np.float32
    )
    target_item_ids = np.zeros(batch, dtype=np.int64)
    target_q = np.zeros((batch, concepts), dtype=np.float32)
    target_numeric = np.zeros(
        (batch, len(ITEM_NUMERIC_NAMES)), dtype=np.float32
    )
    donor_item_ids = np.zeros(batch, dtype=np.int64)
    donor_q = np.zeros((batch, concepts), dtype=np.float32)
    donor_numeric = np.zeros(
        (batch, len(ITEM_NUMERIC_NAMES)), dtype=np.float32
    )

    for row, (record, context) in enumerate(zip(records, contexts, strict=True)):
        length = len(context.support_model_item_ids)
        support_item_ids[row, :length] = context.support_model_item_ids
        support_numeric[row, :length] = context.support_item_numeric
        support_responses[row, :length] = context.support_responses
        support_ease[row, :length] = context.support_item_ease
        support_confidence[row, :length] = (
            context.support_group_attempt_confidence
        )
        support_mask[row, :length] = True
        support_statistics[row] = context.support_statistics
        for position, q_row in enumerate(context.support_q_item_rows):
            support_q[
                row,
                position,
                list(features.q_concept_indices[int(q_row)]),
            ] = 1.0
        target_item_ids[row] = record.target_model_item_id
        target_q[row, list(features.q_concept_indices[record.target_q_item_row])] = 1.0
        target_numeric[row] = record.target_item_numeric
        donor_item_ids[row] = record.donor_model_item_id
        donor_q[row, list(features.q_concept_indices[record.donor_q_item_row])] = 1.0
        donor_numeric[row] = record.donor_item_numeric

    labels = [record.label for record in records]
    label_tensor = (
        None
        if any(label is None for label in labels)
        else torch.as_tensor(labels, dtype=torch.float32, device=device)
    )
    tensor = lambda value, dtype=None: torch.as_tensor(
        value, dtype=dtype, device=device
    )
    return PairingBatch(
        support_item_ids=tensor(support_item_ids, torch.long),
        support_q_multi_hot=tensor(support_q, torch.float32),
        support_item_numeric=tensor(support_numeric, torch.float32),
        support_responses=tensor(support_responses, torch.float32),
        support_item_ease=tensor(support_ease, torch.float32),
        support_group_attempt_confidence=tensor(
            support_confidence, torch.float32
        ),
        support_mask=tensor(support_mask, torch.bool),
        support_statistics=tensor(support_statistics, torch.float32),
        target_item_ids=tensor(target_item_ids, torch.long),
        target_q_multi_hot=tensor(target_q, torch.float32),
        target_item_numeric=tensor(target_numeric, torch.float32),
        donor_target_item_ids=tensor(donor_item_ids, torch.long),
        donor_target_q_multi_hot=tensor(donor_q, torch.float32),
        donor_target_item_numeric=tensor(donor_numeric, torch.float32),
        labels=label_tensor,
        source_row_ids=tuple(record.source_row_id for record in records),
        students=tuple(record.student for record in records),
        exercises=tuple(record.exercise for record in records),
    )
