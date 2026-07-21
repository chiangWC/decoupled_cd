from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from data.pool_protocol import derive_q_matrix, sha256_file, stable_fraction


SPLIT_SEED = 2024
ROLE_NAMESPACE = ("requirement_surface", "student")
GROUP_NAMESPACE = ("requirement_surface", "group")
OPTIMIZER_FRACTION = 0.8
SUPPORT_FRACTION = 0.7
MIN_SUPPORT_GROUPS = 10
MIN_QUERY_GROUPS = 3
MIN_TARGET_CONCEPT_GROUPS = 3
ITEM_OFFSET_STRENGTH = 10.0
MAX_Q_CARDINALITY = 4


def _canonical_id(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


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


def _row_id(source_index: int, student: str, item: str) -> str:
    value = "\x1f".join(
        ("requirement-surface-train-row", str(source_index), student, item)
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class RequirementStudentContext:
    student_id: str
    readiness: np.ndarray
    support_group_count: np.ndarray


@dataclass(frozen=True)
class RequirementQueryRecord:
    row_id: str
    source_index: int
    student_id: str
    item_id: str
    context_index: int
    target_concept_indices: tuple[int, ...]
    q_pair_id: str
    q_count: int
    eligible: bool
    item_offset: float
    label: int | None


@dataclass(frozen=True)
class RequirementFeatureSet:
    split: str
    contexts: tuple[RequirementStudentContext, ...]
    records: tuple[RequirementQueryRecord, ...]
    input_sha256: str
    row_order_sha256: str
    label_sha256: str | None

    def metadata_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "row_id": [r.row_id for r in self.records],
                "student_id": [r.student_id for r in self.records],
                "item_id": [r.item_id for r in self.records],
                "q_pair_id": [r.q_pair_id for r in self.records],
                "q_count": [r.q_count for r in self.records],
                "eligible": [r.eligible for r in self.records],
            }
        )


@dataclass(frozen=True)
class TrainOnlyRequirementProtocol:
    dataset: str
    train_path: Path
    q_matrix_path: Path
    q_lookup: dict[str, tuple[str, ...]]
    concept_index: dict[str, int]
    global_ease: float
    optimizer: RequirementFeatureSet
    audit: RequirementFeatureSet
    protocol_sha256: str
    audit_summary: dict[str, Any]


@dataclass(frozen=True)
class RequirementBatch:
    readiness: torch.Tensor
    q_mask: torch.Tensor
    b: torch.Tensor
    g: torch.Tensor
    item_offset: torch.Tensor
    labels: torch.Tensor | None
    row_ids: tuple[str, ...]
    student_ids: tuple[str, ...]
    item_ids: tuple[str, ...]
    q_pair_ids: tuple[str, ...]
    q_count: np.ndarray
    eligible: np.ndarray


def _read_identity_projection(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    missing = {"stu_id", "exer_id", "label"} - set(header.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    raw = pd.read_csv(path, usecols=lambda column: column != "label")
    rows = raw.loc[:, ["stu_id", "exer_id"]].copy().reset_index(drop=True)
    rows.insert(0, "source_index", np.arange(len(rows), dtype=np.int64))
    rows["stu_id"] = rows["stu_id"].map(_canonical_id)
    rows["exer_id"] = rows["exer_id"].map(_canonical_id)
    rows.insert(
        1,
        "row_id",
        [
            _row_id(int(index), student, item)
            for index, student, item in rows[
                ["source_index", "stu_id", "exer_id"]
            ].itertuples(index=False, name=None)
        ],
    )
    if rows["row_id"].duplicated().any():
        raise RuntimeError("Outcome-free row ID collision.")
    return rows


def _read_selected_labels(path: Path, source_indices: Sequence[int]) -> dict[int, int]:
    """Read no audit-query labels during protocol construction."""
    selected = sorted({int(value) for value in source_indices})
    if not selected:
        return {}
    allowed = set(selected)
    frame = pd.read_csv(
        path,
        usecols=["label"],
        skiprows=lambda line: line > 0 and (line - 1) not in allowed,
    )
    labels = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if len(labels) != len(selected) or not set(labels.unique()).issubset({0, 1}):
        raise RuntimeError("Selective label loading failed.")
    return dict(zip(selected, labels.tolist(), strict=True))


def _group_outcomes(
    rows: pd.DataFrame, labels: dict[int, int]
) -> Iterable[tuple[str, str, float]]:
    for (student, item), group in rows.groupby(["stu_id", "exer_id"], sort=False):
        values = [labels[int(index)] for index in group["source_index"]]
        yield str(student), str(item), float(np.mean(values))


def _contexts(
    support: pd.DataFrame,
    labels: dict[int, int],
    q_lookup: dict[str, tuple[str, ...]],
    concept_index: dict[str, int],
) -> tuple[RequirementStudentContext, ...]:
    output = []
    for student, rows in support.groupby("stu_id", sort=True):
        attempts = np.zeros(len(concept_index), dtype=np.int32)
        correct = np.zeros(len(concept_index), dtype=np.float64)
        for _, item, response in _group_outcomes(rows, labels):
            for concept in q_lookup[item]:
                index = concept_index[concept]
                attempts[index] += 1
                correct[index] += response
        output.append(
            RequirementStudentContext(
                str(student),
                ((correct + 1.0) / (attempts + 2.0)).astype(np.float32),
                attempts,
            )
        )
    return tuple(output)


def _offsets(
    optimizer_support: pd.DataFrame, labels: dict[int, int]
) -> tuple[float, float, dict[str, float]]:
    groups = list(_group_outcomes(optimizer_support, labels))
    global_ease = (sum(value for _, _, value in groups) + 1.0) / (
        len(groups) + 2.0
    )
    per_item: dict[str, list[float]] = {}
    for _, item, value in groups:
        per_item.setdefault(item, []).append(value)
    offsets = {}
    for item, values in per_item.items():
        ease = (sum(values) + ITEM_OFFSET_STRENGTH * global_ease) / (
            len(values) + ITEM_OFFSET_STRENGTH
        )
        ease = float(np.clip(ease, 1e-4, 1.0 - 1e-4))
        offsets[item] = float(np.log(ease / (1.0 - ease)))
    clipped = float(np.clip(global_ease, 1e-4, 1.0 - 1e-4))
    return float(global_ease), float(np.log(clipped / (1.0 - clipped))), offsets


def _features(
    split: str,
    contexts: tuple[RequirementStudentContext, ...],
    query: pd.DataFrame,
    labels: dict[int, int],
    labels_available: bool,
    q_lookup: dict[str, tuple[str, ...]],
    concept_index: dict[str, int],
    global_offset: float,
    offsets: dict[str, float],
) -> RequirementFeatureSet:
    context_lookup = {value.student_id: i for i, value in enumerate(contexts)}
    records = []
    for row in query.sort_values("source_index", kind="stable").itertuples(index=False):
        concepts = q_lookup[str(row.exer_id)]
        indices = tuple(concept_index[value] for value in concepts)
        context_index = context_lookup[str(row.stu_id)]
        context = contexts[context_index]
        eligible = 2 <= len(indices) <= 4 and all(
            context.support_group_count[index] >= MIN_TARGET_CONCEPT_GROUPS
            for index in indices
        )
        records.append(
            RequirementQueryRecord(
                str(row.row_id),
                int(row.source_index),
                str(row.stu_id),
                str(row.exer_id),
                context_index,
                indices,
                "|".join(concepts),
                len(concepts),
                bool(eligible),
                float(offsets.get(str(row.exer_id), global_offset)),
                labels[int(row.source_index)] if labels_available else None,
            )
        )
    digest = hashlib.sha256()
    digest.update(split.encode())
    for context in contexts:
        digest.update(context.student_id.encode())
        digest.update(_hash_array(context.readiness).encode())
        digest.update(_hash_array(context.support_group_count).encode())
    for record in records:
        digest.update(
            "\x1f".join(
                (
                    record.row_id,
                    record.student_id,
                    record.item_id,
                    record.q_pair_id,
                    str(record.q_count),
                    str(int(record.eligible)),
                    f"{record.item_offset:.17g}",
                    ",".join(map(str, record.target_concept_indices)),
                )
            ).encode()
        )
    return RequirementFeatureSet(
        split,
        tuple(contexts),
        tuple(records),
        digest.hexdigest(),
        _hash_values(record.row_id for record in records),
        (
            _hash_values(int(record.label) for record in records)
            if labels_available
            else None
        ),
    )


def build_train_only_requirement_protocol(
    *,
    dataset: str,
    train_path: str | Path,
    q_matrix_path: str | Path,
) -> TrainOnlyRequirementProtocol:
    train_path = Path(train_path)
    q_matrix_path = Path(q_matrix_path)
    rows = _read_identity_projection(train_path)
    q_matrix, q_conflicts = derive_q_matrix(pd.read_csv(q_matrix_path))
    q_lookup = {
        str(row.exer_id): tuple(str(row.cpt_seq).split(","))
        for row in q_matrix.itertuples(index=False)
    }
    missing = sorted(set(rows["exer_id"]) - set(q_lookup))
    if missing:
        raise RuntimeError(f"Train exercises absent from Q matrix: {missing[:10]}")
    concepts = sorted({c for values in q_lookup.values() for c in values})
    concept_index = {concept: i for i, concept in enumerate(concepts)}

    optimizer_students = {
        student
        for student in set(rows["stu_id"])
        if stable_fraction(SPLIT_SEED, *ROLE_NAMESPACE, student)
        < OPTIMIZER_FRACTION
    }
    rows["role"] = np.where(
        rows["stu_id"].isin(optimizer_students), "optimizer", "audit"
    )
    groups = rows[["stu_id", "exer_id"]].drop_duplicates()
    groups["is_support"] = [
        stable_fraction(SPLIT_SEED, *GROUP_NAMESPACE, student, item)
        < SUPPORT_FRACTION
        for student, item in groups.itertuples(index=False, name=None)
    ]
    rows = rows.merge(
        groups, on=["stu_id", "exer_id"], how="left", validate="many_to_one"
    )
    counts = rows.groupby(["stu_id", "is_support"])["exer_id"].nunique().unstack(
        fill_value=0
    )
    for column in (False, True):
        if column not in counts:
            counts[column] = 0
    retained = set(
        counts.index[
            (counts[True] >= MIN_SUPPORT_GROUPS)
            & (counts[False] >= MIN_QUERY_GROUPS)
        ]
    )
    rows = rows[rows["stu_id"].isin(retained)].copy()
    optimizer = rows[rows["role"] == "optimizer"]
    audit = rows[rows["role"] == "audit"]
    optimizer_support = optimizer[optimizer["is_support"]]
    optimizer_query = optimizer[~optimizer["is_support"]]
    audit_support = audit[audit["is_support"]]
    audit_query = audit[~audit["is_support"]]
    if set(optimizer["stu_id"]) & set(audit["stu_id"]):
        raise RuntimeError("Optimizer and audit students overlap.")

    selected = sorted(
        set(optimizer["source_index"]) | set(audit_support["source_index"])
    )
    labels = _read_selected_labels(train_path, selected)
    optimizer_contexts = _contexts(
        optimizer_support, labels, q_lookup, concept_index
    )
    audit_contexts = _contexts(audit_support, labels, q_lookup, concept_index)
    global_ease, global_offset, item_offsets = _offsets(
        optimizer_support, labels
    )
    optimizer_features = _features(
        "optimizer_query",
        optimizer_contexts,
        optimizer_query,
        labels,
        True,
        q_lookup,
        concept_index,
        global_offset,
        item_offsets,
    )
    audit_features = _features(
        "audit_query",
        audit_contexts,
        audit_query,
        labels,
        False,
        q_lookup,
        concept_index,
        global_offset,
        item_offsets,
    )
    identity_hash = _hash_values(
        f"{r.source_index}:{r.row_id}:{r.stu_id}:{r.exer_id}:{r.role}:{int(r.is_support)}"
        for r in rows.sort_values("source_index").itertuples(index=False)
    )
    q_hash = _hash_values(
        f"{item}:{','.join(values)}" for item, values in sorted(q_lookup.items())
    )
    protocol_hash = _hash_values(
        (
            dataset,
            identity_hash,
            q_hash,
            optimizer_features.input_sha256,
            audit_features.input_sha256,
        )
    )
    summary = {
        "split_seed": SPLIT_SEED,
        "role_key": [SPLIT_SEED, *ROLE_NAMESPACE, "<student>"],
        "group_key": [SPLIT_SEED, *GROUP_NAMESPACE, "<student>", "<item>"],
        "optimizer_students": int(optimizer["stu_id"].nunique()),
        "audit_students": int(audit["stu_id"].nunique()),
        "optimizer_support_rows": len(optimizer_support),
        "optimizer_query_rows": len(optimizer_query),
        "audit_support_rows": len(audit_support),
        "audit_query_rows": len(audit_query),
        "audit_eligible_rows": sum(r.eligible for r in audit_features.records),
        "q_conflicts_unioned": int(q_conflicts),
        "labels_materialized_rows": len(selected),
        "audit_query_labels_materialized": False,
        "identity_projection_sha256": identity_hash,
        "q_union_sha256": q_hash,
        "train_full_sha256": sha256_file(train_path),
        "q_matrix_sha256": sha256_file(q_matrix_path),
    }
    return TrainOnlyRequirementProtocol(
        dataset,
        train_path.resolve(),
        q_matrix_path.resolve(),
        q_lookup,
        concept_index,
        global_ease,
        optimizer_features,
        audit_features,
        protocol_hash,
        summary,
    )


def load_audit_query_labels(
    protocol: TrainOnlyRequirementProtocol,
    *,
    train_path: str | Path | None = None,
) -> pd.DataFrame:
    path = Path(train_path) if train_path is not None else protocol.train_path
    if path.resolve() != protocol.train_path:
        raise ValueError("Evaluation path differs from frozen train path.")
    labels = _read_selected_labels(
        path, [record.source_index for record in protocol.audit.records]
    )
    return pd.DataFrame(
        {
            "row_id": [record.row_id for record in protocol.audit.records],
            "label": [labels[record.source_index] for record in protocol.audit.records],
        }
    )


def collate_requirement_batch(
    features: RequirementFeatureSet,
    indices: Sequence[int],
    *,
    device: torch.device | str,
) -> RequirementBatch:
    records = [features.records[int(index)] for index in indices]
    if not records:
        raise ValueError("Empty requirement batch.")
    concept_count = len(features.contexts[0].readiness)
    if any(len(context.readiness) != concept_count for context in features.contexts):
        raise RuntimeError("Student contexts disagree on the concept universe.")
    readiness = np.ones((len(records), concept_count), dtype=np.float32)
    mask = np.zeros_like(readiness, dtype=bool)
    for row, record in enumerate(records):
        target = list(record.target_concept_indices)
        values = features.contexts[record.context_index].readiness[target]
        readiness[row, target] = values
        mask[row, target] = True
    counts = mask.sum(axis=1)
    if np.any(counts == 0):
        raise RuntimeError("Q=0 is invalid.")
    b = np.where(mask, readiness, 1.0).min(axis=1)
    g = np.where(mask, readiness, 0.0).sum(axis=1) / counts
    label_values = [record.label for record in records]
    tensor = lambda x, dtype: torch.as_tensor(x, dtype=dtype, device=device)
    return RequirementBatch(
        tensor(readiness, torch.float32),
        tensor(mask, torch.bool),
        tensor(b, torch.float32),
        tensor(g, torch.float32),
        tensor([record.item_offset for record in records], torch.float32),
        (
            None
            if any(value is None for value in label_values)
            else tensor(label_values, torch.float32)
        ),
        tuple(record.row_id for record in records),
        tuple(record.student_id for record in records),
        tuple(record.item_id for record in records),
        tuple(record.q_pair_id for record in records),
        np.asarray([record.q_count for record in records], dtype=np.int64),
        np.asarray([record.eligible for record in records], dtype=bool),
    )
