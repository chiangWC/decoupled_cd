from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pool_protocol import (
    canonical_concepts,
    canonicalize_interactions,
    derive_q_matrix,
    sha256_file,
    stable_fraction,
)
from scripts.audit_conditional_response_signature import (
    ReferenceStatistics,
    StudentProfile,
    build_reference_statistics,
    support_theta,
)


MODEL_SEED = 42
SPLIT_SEED = 2024
OPTIMIZER_FRACTION = 0.80
INNER_SUPPORT_FRACTION = 0.70
MIN_SUPPORT_GROUPS = 10
MIN_QUERY_GROUPS = 3
NUM_FOLDS = 5
PERMUTATION_REPLICATES = 200
MIN_CHANGED_ROW_FRACTION = 0.20
MIN_MOVABLE_STUDENTS = 100

ROLE_NAMESPACE = "target-local-validation-role"
INNER_QUERY_NAMESPACE = "target-local-optimizer-inner-query"
FOLD_NAMESPACE = "target-local-optimizer-fold"
DONOR_ORDER_NAMESPACE = "target-local-donor-order"
DONOR_SHIFT_NAMESPACE = "target-local-donor-shift"

ALLOWED_SOURCE_FILES = ("train.csv", "valid.csv", "Q_matrix.csv")


def _hash_values(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def _concepts(value: object) -> tuple[str, ...]:
    return tuple(token for token in canonical_concepts(value).split(",") if token)


def _group_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        zip(
            frame["stu_id"].astype(str),
            frame["exer_id"].astype(str),
            strict=True,
        )
    )


def _select_students(frame: pd.DataFrame, students: set[str]) -> pd.DataFrame:
    return frame.loc[frame["stu_id"].astype(str).isin(students)].copy()


def _canonical_id(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def _validation_identity_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive outcome-free validation row IDs from source identity columns."""
    missing = {"stu_id", "exer_id"} - set(frame.columns)
    if missing:
        raise ValueError(
            f"Validation data is missing identity columns: {sorted(missing)}"
        )
    output = frame.loc[:, ["stu_id", "exer_id"]].copy()
    output["stu_id"] = output["stu_id"].map(_canonical_id)
    output["exer_id"] = output["exer_id"].map(_canonical_id)
    if "split_row_index" in frame:
        stable_index = pd.to_numeric(
            frame["split_row_index"], errors="raise"
        ).astype(int)
    else:
        # Source row order is part of the immutable data version. Unlike the
        # legacy source_row_id, this position never includes the outcome.
        stable_index = pd.Series(np.arange(len(frame)), index=frame.index)
    output.insert(0, "source_row_index", stable_index.to_numpy())
    output.insert(
        0,
        "source_row_id",
        [
            hashlib.sha256(
                "\x1f".join(
                    ("target-local-valid-row", str(index), student, item)
                ).encode("utf-8")
            ).hexdigest()[:32]
            for index, student, item in zip(
                stable_index,
                output["stu_id"],
                output["exer_id"],
                strict=True,
            )
        ],
    )
    if output["source_row_id"].duplicated().any():
        raise RuntimeError("Validation label-free row-ID collision detected.")
    return output


def validation_row_order_sha256(frame: pd.DataFrame) -> str:
    """Hash validation row IDs in their current prediction/evaluation order."""
    if "source_row_id" not in frame:
        raise ValueError("Validation rows are missing source_row_id.")
    return _hash_values(frame["source_row_id"].astype(str))


def _unique_validation_ids(frame: pd.DataFrame, *, name: str) -> pd.Series:
    if "source_row_id" not in frame:
        raise ValueError(f"{name} is missing source_row_id.")
    identifiers = frame["source_row_id"].astype(str)
    duplicated = identifiers[identifiers.duplicated(keep=False)]
    if not duplicated.empty:
        examples = sorted(set(duplicated))[:5]
        raise RuntimeError(
            f"{name} contains duplicate source_row_id values: {examples}"
        )
    return identifiers


def load_validation_feature_rows(path: Path) -> pd.DataFrame:
    """Read validation covariates while never loading or hashing its label."""
    frame = pd.read_csv(path, usecols=lambda column: column != "label")
    missing = {"stu_id", "exer_id", "cpt_seq"} - set(frame.columns)
    if missing:
        raise ValueError(
            f"Validation data is missing feature columns: {sorted(missing)}"
        )
    identity = _validation_identity_rows(frame)
    output = identity.copy()
    output["cpt_seq"] = frame["cpt_seq"].map(canonical_concepts)
    if (output["cpt_seq"] == "").any():
        raise ValueError("Every validation interaction must have a concept.")
    return output.reset_index(drop=True)


def validation_feature_sha256(frame: pd.DataFrame) -> str:
    columns = (
        "source_row_id",
        "source_row_index",
        "stu_id",
        "exer_id",
        "cpt_seq",
    )
    return _hash_values(
        "\x1e".join(map(str, row))
        for row in frame.loc[:, columns].itertuples(index=False, name=None)
    )


def load_validation_labels_for_evaluation(
    path: Path,
    *,
    feature_rows: pd.DataFrame,
    expected_valid_sha256: str,
) -> pd.DataFrame:
    """Reload outcomes after prediction and select them by label-free row ID."""
    feature_ids = _unique_validation_ids(feature_rows, name="Validation features")
    feature_order_before = validation_row_order_sha256(feature_rows)
    valid_sha256_before = sha256_file(path)
    if valid_sha256_before != expected_valid_sha256:
        raise RuntimeError(
            "Validation full-file SHA-256 differs from the frozen protocol audit."
        )

    frame = pd.read_csv(
        path,
        usecols=lambda column: column
        in {"stu_id", "exer_id", "split_row_index", "label"},
    )
    if "label" not in frame:
        raise ValueError("Validation data is missing label.")
    identity = _validation_identity_rows(frame)
    all_label_ids = _unique_validation_ids(identity, name="Validation labels")
    labels = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError("Validation labels must be binary.")
    label_rows = identity.assign(label=labels.to_numpy())

    feature_id_set = set(feature_ids)
    label_id_set = set(all_label_ids)
    missing = sorted(feature_id_set - label_id_set)
    if missing:
        raise RuntimeError(
            "Validation feature/label ID sets differ; labels are missing IDs: "
            f"{missing[:5]}"
        )
    selected = label_rows.loc[all_label_ids.isin(feature_id_set)].copy()
    selected_ids = _unique_validation_ids(selected, name="Selected validation labels")
    if len(selected) != len(feature_rows) or set(selected_ids) != feature_id_set:
        raise RuntimeError("Selected validation label IDs do not match feature IDs.")

    valid_sha256_after = sha256_file(path)
    if valid_sha256_after != valid_sha256_before:
        raise RuntimeError("Validation file changed while labels were being loaded.")
    feature_order_after = validation_row_order_sha256(feature_rows)
    if feature_order_after != feature_order_before:
        raise RuntimeError("Validation feature row order changed during label loading.")
    selected.attrs.update(
        {
            "valid_full_sha256_before": valid_sha256_before,
            "valid_full_sha256_after": valid_sha256_after,
            "feature_row_order_sha256_before": feature_order_before,
            "feature_row_order_sha256_after": feature_order_after,
        }
    )
    return selected.reset_index(drop=True)


def join_validation_predictions_with_labels(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    expected_prediction_order_sha256: str | None = None,
) -> pd.DataFrame:
    """Join one prediction and one label per row ID without changing order."""
    if "label" in predictions:
        raise ValueError("Predictions must not already contain label.")
    prediction_ids = _unique_validation_ids(
        predictions, name="Validation predictions"
    )
    label_ids = _unique_validation_ids(labels, name="Validation labels")
    prediction_id_set = set(prediction_ids)
    label_id_set = set(label_ids)
    if prediction_id_set != label_id_set:
        missing = sorted(prediction_id_set - label_id_set)
        unexpected = sorted(label_id_set - prediction_id_set)
        raise RuntimeError(
            "Validation prediction/label ID sets differ: "
            f"missing_labels={missing[:5]}, unexpected_labels={unexpected[:5]}"
        )

    order_before = validation_row_order_sha256(predictions)
    if (
        expected_prediction_order_sha256 is not None
        and order_before != expected_prediction_order_sha256
    ):
        raise RuntimeError("Validation prediction row-order SHA-256 is unexpected.")
    joined = predictions.merge(
        labels.loc[:, ["source_row_id", "label"]],
        on="source_row_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if joined["label"].isna().any() or len(joined) != len(predictions):
        raise RuntimeError("Validation prediction/label join is incomplete.")
    order_after = validation_row_order_sha256(joined)
    if order_after != order_before:
        raise RuntimeError("Validation prediction order changed during label join.")
    joined.attrs.update(labels.attrs)
    joined.attrs.update(
        {
            "prediction_row_order_sha256_before": order_before,
            "prediction_row_order_sha256_after": order_after,
        }
    )
    return joined


def _replace_q(
    frame: pd.DataFrame,
    *,
    q_lookup: dict[str, tuple[str, ...]],
    split: str,
) -> pd.DataFrame:
    items = frame["exer_id"].astype(str)
    missing = sorted(set(items) - set(q_lookup))
    if missing:
        raise RuntimeError(
            f"{split} contains exercises absent from Q_matrix.csv: {missing[:10]}"
        )
    output = frame.copy()
    output["cpt_seq"] = items.map(
        {item: ",".join(concepts) for item, concepts in q_lookup.items()}
    )
    return output


@dataclass(frozen=True)
class ValidationOnlyProtocol:
    dataset: str
    source_dir: Path
    optimizer_train: pd.DataFrame
    validation_support: pd.DataFrame
    validation_query: pd.DataFrame
    q_lookup: dict[str, tuple[str, ...]]
    audit: dict[str, Any]


@dataclass(frozen=True)
class FoldReference:
    fold: int
    statistics: ReferenceStatistics
    ease_tercile_edges: np.ndarray
    reference_students: tuple[str, ...]
    reference_rows_sha256: str


@dataclass(frozen=True)
class DonorMapping:
    rows: pd.DataFrame
    audit: dict[str, Any]


def load_validation_only_protocol(
    dataset: str,
    source_dir: Path,
) -> ValidationOnlyProtocol:
    """Build a student-disjoint protocol without opening test or a manifest."""
    source_dir = Path(source_dir)
    required = {name: source_dir / name for name in ALLOWED_SOURCE_FILES}
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing validation-only source files: {missing}")

    raw_train = pd.read_csv(required["train.csv"])
    valid = load_validation_feature_rows(required["valid.csv"])
    raw_q = pd.read_csv(required["Q_matrix.csv"])
    train, train_duplicates = canonicalize_interactions(raw_train)
    q_matrix, q_conflicts = derive_q_matrix(raw_q)
    q_lookup = {
        str(row.exer_id): _concepts(row.cpt_seq)
        for row in q_matrix.itertuples(index=False)
    }
    if any(not concepts for concepts in q_lookup.values()):
        raise RuntimeError("Every Q_matrix exercise must have at least one concept.")
    train = _replace_q(train, q_lookup=q_lookup, split="train.csv")
    valid = _replace_q(valid, q_lookup=q_lookup, split="valid.csv")

    train_students = set(train["stu_id"].astype(str))
    valid_students = set(valid["stu_id"].astype(str))
    common_students = train_students & valid_students
    optimizer_students: set[str] = set()
    validation_students: set[str] = set()
    for student in common_students:
        fraction = stable_fraction(SPLIT_SEED, ROLE_NAMESPACE, student)
        destination = (
            optimizer_students
            if fraction < OPTIMIZER_FRACTION
            else validation_students
        )
        destination.add(student)
    if not optimizer_students or not validation_students:
        raise RuntimeError("Stable role assignment produced an empty role.")

    optimizer_train = _select_students(train, optimizer_students)
    known_items = set(optimizer_train["exer_id"].astype(str))
    raw_support = _select_students(train, validation_students)
    raw_query = _select_students(valid, validation_students)

    support_known_mask = raw_support["exer_id"].astype(str).isin(known_items)
    query_known_mask = raw_query["exer_id"].astype(str).isin(known_items)
    support_unknown = raw_support.loc[~support_known_mask]
    query_unknown = raw_query.loc[~query_known_mask]
    support = raw_support.loc[support_known_mask].copy()
    query = raw_query.loc[query_known_mask].copy()

    support_groups = _group_set(support)
    query_overlap_mask = np.asarray(
        [
            (str(student), str(item)) in support_groups
            for student, item in zip(
                query["stu_id"], query["exer_id"], strict=True
            )
        ],
        dtype=bool,
    )
    overlap_query = query.loc[query_overlap_mask]
    query = query.loc[~query_overlap_mask].copy()

    support_counts = (
        support.assign(_student=support["stu_id"].astype(str))
        .groupby("_student")["exer_id"]
        .nunique()
    )
    query_counts = (
        query.assign(_student=query["stu_id"].astype(str))
        .groupby("_student")["exer_id"]
        .nunique()
    )
    retained_students = {
        student
        for student in validation_students
        if int(support_counts.get(student, 0)) >= MIN_SUPPORT_GROUPS
        and int(query_counts.get(student, 0)) >= MIN_QUERY_GROUPS
    }
    validation_support = (
        _select_students(support, retained_students)
        .sort_values("source_row_id", kind="stable")
        .reset_index(drop=True)
    )
    validation_query = (
        _select_students(query, retained_students)
        .sort_values("source_row_id", kind="stable")
        .reset_index(drop=True)
    )
    optimizer_train = (
        optimizer_train.sort_values("source_row_id", kind="stable")
        .reset_index(drop=True)
    )

    optimizer_output_students = set(optimizer_train["stu_id"].astype(str))
    validation_output_students = set(validation_query["stu_id"].astype(str))
    if optimizer_output_students & validation_output_students:
        raise RuntimeError("Optimizer and validation students overlap.")
    if _group_set(validation_support) & _group_set(validation_query):
        raise RuntimeError("Validation support/query groups overlap after filtering.")

    source_hashes = {
        "train.csv": {
            "sha256": sha256_file(required["train.csv"]),
            "rows": int(len(raw_train)),
        },
        "valid.csv": {
            "sha256": sha256_file(required["valid.csv"]),
            "feature_projection_sha256": validation_feature_sha256(valid),
            "rows": int(len(valid)),
            "label_loaded": False,
            "full_file_hash_used_for_split_or_mapping": False,
        },
        "Q_matrix.csv": {
            "sha256": sha256_file(required["Q_matrix.csv"]),
            "rows": int(len(raw_q)),
        },
    }
    audit = {
        "schema_version": 1,
        "protocol": "student_disjoint_validation_only",
        "dataset": dataset,
        "source_directory": str(source_dir.resolve()),
        "opened_source_files": list(ALLOWED_SOURCE_FILES),
        "source_files": source_hashes,
        "split_seed": SPLIT_SEED,
        "role_namespace": ROLE_NAMESPACE,
        "optimizer_fraction": OPTIMIZER_FRACTION,
        "source_students": {
            "train": len(train_students),
            "valid": len(valid_students),
            "common": len(common_students),
        },
        "assigned_students": {
            "optimizer": len(optimizer_students),
            "validation": len(validation_students),
            "validation_retained": len(retained_students),
        },
        "rows": {
            "optimizer_train": len(optimizer_train),
            "validation_support": len(validation_support),
            "validation_query": len(validation_query),
            "validation_support_unknown_item_removed": len(support_unknown),
            "validation_query_unknown_item_removed": len(query_unknown),
            "validation_query_support_group_overlap_removed": len(overlap_query),
        },
        "duplicates_removed": {
            "train": int(train_duplicates),
            "valid_features": 0,
        },
        "q_union": {
            "exercises": len(q_lookup),
            "concepts": len(
                {concept for concepts in q_lookup.values() for concept in concepts}
            ),
            "conflicts_reported_by_union": int(q_conflicts),
        },
        "overlap": {
            "optimizer_validation_students": 0,
            "validation_support_query_groups": 0,
        },
        "known_items": {
            "provisional_optimizer_role_union": len(known_items),
            "provisional_optimizer_role_union_sha256": _hash_values(
                sorted(known_items)
            ),
            "finalized_after_optimizer_retention": False,
        },
        "hashes": {
            "optimizer_students": _hash_values(sorted(optimizer_output_students)),
            "validation_students": _hash_values(sorted(validation_output_students)),
            "optimizer_rows": _hash_values(
                optimizer_train["source_row_id"].astype(str)
            ),
            "validation_support_rows": _hash_values(
                validation_support["source_row_id"].astype(str)
            ),
            "validation_query_rows": _hash_values(
                validation_query["source_row_id"].astype(str)
            ),
        },
        "validation_query_labels_used_for_protocol": False,
        "test_files_opened": False,
        "legacy_manifest_opened": False,
    }
    return ValidationOnlyProtocol(
        dataset=dataset,
        source_dir=source_dir.resolve(),
        optimizer_train=optimizer_train,
        validation_support=validation_support,
        validation_query=validation_query,
        q_lookup=q_lookup,
        audit=audit,
    )


def build_optimizer_profiles(
    frame: pd.DataFrame,
    *,
    q_lookup: dict[str, tuple[str, ...]],
) -> tuple[list[StudentProfile], dict[str, Any]]:
    profiles: list[StudentProfile] = []
    skipped = 0
    for student, student_frame in frame.groupby(
        frame["stu_id"].astype(str), sort=False
    ):
        student = str(student)
        items = sorted(set(student_frame["exer_id"].astype(str)))
        query_items = {
            item
            for item in items
            if stable_fraction(
                SPLIT_SEED, INNER_QUERY_NAMESPACE, student, item
            )
            >= INNER_SUPPORT_FRACTION
        }
        query_mask = student_frame["exer_id"].astype(str).isin(query_items)
        support = student_frame.loc[~query_mask].copy().reset_index(drop=True)
        query = student_frame.loc[query_mask].copy().reset_index(drop=True)
        if (
            support["exer_id"].astype(str).nunique() < MIN_SUPPORT_GROUPS
            or query["exer_id"].astype(str).nunique() < MIN_QUERY_GROUPS
        ):
            skipped += 1
            continue
        if _group_set(support) & _group_set(query):
            raise RuntimeError("Optimizer support/query groups are not atomic.")
        theta, raw_accuracy = support_theta(support)
        seen = {
            concept
            for item in support["exer_id"].astype(str)
            for concept in q_lookup[item]
        }
        fold = min(
            int(
                NUM_FOLDS
                * stable_fraction(SPLIT_SEED, FOLD_NAMESPACE, student)
            ),
            NUM_FOLDS - 1,
        )
        profiles.append(
            StudentProfile(
                student=student,
                support=support,
                query=query,
                theta=theta,
                raw_accuracy=raw_accuracy,
                seen_concepts=frozenset(seen),
                fold=fold,
            )
        )
    fold_counts = {
        str(fold): sum(profile.fold == fold for profile in profiles)
        for fold in range(NUM_FOLDS)
    }
    if any(count == 0 for count in fold_counts.values()):
        raise RuntimeError(f"An optimizer OOF fold is empty: {fold_counts}")
    return profiles, {
        "inner_query_namespace": INNER_QUERY_NAMESPACE,
        "support_fraction": INNER_SUPPORT_FRACTION,
        "minimum_support_groups": MIN_SUPPORT_GROUPS,
        "minimum_query_groups": MIN_QUERY_GROUPS,
        "retained_students": len(profiles),
        "skipped_students": skipped,
        "fold_namespace": FOLD_NAMESPACE,
        "fold_counts": fold_counts,
        "fold_assignment_sha256": _hash_values(
            f"{profile.student}:{profile.fold}"
            for profile in sorted(profiles, key=lambda value: value.student)
        ),
        "support_rows": int(sum(len(profile.support) for profile in profiles)),
        "query_rows": int(sum(len(profile.query) for profile in profiles)),
        "support_query_group_overlap": 0,
    }


def retained_optimizer_items(profiles: Iterable[StudentProfile]) -> set[str]:
    """Return the item union actually represented by retained optimizer rows."""
    items: set[str] = set()
    for profile in profiles:
        items.update(profile.support["exer_id"].astype(str))
        items.update(profile.query["exer_id"].astype(str))
    if not items:
        raise RuntimeError("Retained optimizer profiles contain no items.")
    return items


def finalize_protocol_for_retained_optimizer(
    protocol: ValidationOnlyProtocol,
    profiles: Iterable[StudentProfile],
) -> ValidationOnlyProtocol:
    """Finalize validation eligibility using only retained optimizer items."""
    retained_profiles = list(profiles)
    known_items = retained_optimizer_items(retained_profiles)
    retained_optimizer_students = {
        str(profile.student) for profile in retained_profiles
    }
    assigned_optimizer_students = set(
        protocol.optimizer_train["stu_id"].astype(str)
    )
    if not retained_optimizer_students <= assigned_optimizer_students:
        raise RuntimeError("Retained optimizer profiles contain an unassigned student.")

    support_before = protocol.validation_support
    query_before = protocol.validation_query
    support_known_mask = support_before["exer_id"].astype(str).isin(known_items)
    query_known_mask = query_before["exer_id"].astype(str).isin(known_items)
    newly_unknown_support = support_before.loc[~support_known_mask]
    newly_unknown_query = query_before.loc[~query_known_mask]
    support = support_before.loc[support_known_mask].copy()
    query = query_before.loc[query_known_mask].copy()

    support_groups = _group_set(support)
    query_overlap_mask = np.asarray(
        [
            (str(student), str(item)) in support_groups
            for student, item in zip(
                query["stu_id"], query["exer_id"], strict=True
            )
        ],
        dtype=bool,
    )
    newly_overlapping_query = query.loc[query_overlap_mask]
    query = query.loc[~query_overlap_mask].copy()

    support_counts = (
        support.assign(_student=support["stu_id"].astype(str))
        .groupby("_student")["exer_id"]
        .nunique()
    )
    query_counts = (
        query.assign(_student=query["stu_id"].astype(str))
        .groupby("_student")["exer_id"]
        .nunique()
    )
    candidate_students = set(support["stu_id"].astype(str)) | set(
        query["stu_id"].astype(str)
    )
    validation_students = {
        student
        for student in candidate_students
        if int(support_counts.get(student, 0)) >= MIN_SUPPORT_GROUPS
        and int(query_counts.get(student, 0)) >= MIN_QUERY_GROUPS
    }
    validation_support = (
        _select_students(support, validation_students)
        .sort_values("source_row_id", kind="stable")
        .reset_index(drop=True)
    )
    validation_query = (
        _select_students(query, validation_students)
        .sort_values("source_row_id", kind="stable")
        .reset_index(drop=True)
    )
    validation_output_students = set(validation_query["stu_id"].astype(str))
    if retained_optimizer_students & validation_output_students:
        raise RuntimeError("Retained optimizer and validation students overlap.")
    if _group_set(validation_support) & _group_set(validation_query):
        raise RuntimeError("Final validation support/query groups overlap.")
    if not set(validation_support["exer_id"].astype(str)) <= known_items:
        raise RuntimeError("Final validation support contains an unknown item.")
    if not set(validation_query["exer_id"].astype(str)) <= known_items:
        raise RuntimeError("Final validation query contains an unknown item.")

    prior_validation_students = set(
        protocol.validation_query["stu_id"].astype(str)
    )
    audit = dict(protocol.audit)
    audit["assigned_students"] = dict(audit["assigned_students"])
    audit["assigned_students"]["validation_retained"] = len(
        validation_output_students
    )
    audit["rows"] = dict(audit["rows"])
    audit["rows"].update(
        {
            "validation_support": len(validation_support),
            "validation_query": len(validation_query),
            "validation_support_unknown_item_removed": int(
                audit["rows"]["validation_support_unknown_item_removed"]
                + len(newly_unknown_support)
            ),
            "validation_query_unknown_item_removed": int(
                audit["rows"]["validation_query_unknown_item_removed"]
                + len(newly_unknown_query)
            ),
            "validation_query_support_group_overlap_removed": int(
                audit["rows"]["validation_query_support_group_overlap_removed"]
                + len(newly_overlapping_query)
            ),
        }
    )
    audit["known_items"] = {
        **audit.get("known_items", {}),
        "retained_optimizer_profile_union": len(known_items),
        "retained_optimizer_profile_union_sha256": _hash_values(
            sorted(known_items)
        ),
        "new_validation_support_rows_removed": len(newly_unknown_support),
        "new_validation_query_rows_removed": len(newly_unknown_query),
        "new_validation_students_removed": len(
            prior_validation_students - validation_output_students
        ),
        "finalized_after_optimizer_retention": True,
    }
    audit["hashes"] = dict(audit["hashes"])
    audit["hashes"].update(
        {
            "retained_optimizer_students": _hash_values(
                sorted(retained_optimizer_students)
            ),
            "retained_optimizer_items": _hash_values(sorted(known_items)),
            "validation_students": _hash_values(
                sorted(validation_output_students)
            ),
            "validation_support_rows": _hash_values(
                validation_support["source_row_id"].astype(str)
            ),
            "validation_query_rows": _hash_values(
                validation_query["source_row_id"].astype(str)
            ),
        }
    )
    return ValidationOnlyProtocol(
        dataset=protocol.dataset,
        source_dir=protocol.source_dir,
        optimizer_train=protocol.optimizer_train,
        validation_support=validation_support,
        validation_query=validation_query,
        q_lookup=protocol.q_lookup,
        audit=audit,
    )


def build_validation_profiles(
    support: pd.DataFrame,
    query: pd.DataFrame,
    *,
    q_lookup: dict[str, tuple[str, ...]],
) -> list[StudentProfile]:
    query_by_student = {
        str(student): frame.copy().reset_index(drop=True)
        for student, frame in query.groupby(query["stu_id"].astype(str), sort=False)
    }
    profiles: list[StudentProfile] = []
    for student, student_support in support.groupby(
        support["stu_id"].astype(str), sort=False
    ):
        student = str(student)
        student_query = query_by_student.get(student)
        if student_query is None or student_query.empty:
            continue
        theta, raw_accuracy = support_theta(student_support)
        seen = {
            concept
            for item in student_support["exer_id"].astype(str)
            for concept in q_lookup[item]
        }
        profiles.append(
            StudentProfile(
                student=student,
                support=student_support.copy().reset_index(drop=True),
                query=student_query,
                theta=theta,
                raw_accuracy=raw_accuracy,
                seen_concepts=frozenset(seen),
                fold=-1,
            )
        )
    return profiles


def ease_tercile_edges(statistics: ReferenceStatistics) -> np.ndarray:
    observed = statistics.item_ease[statistics.item_count > 0]
    values = observed if len(observed) >= 3 else statistics.item_ease
    if len(values) == 0:
        raise ValueError("Cannot define ease terciles without items.")
    return np.asarray(np.quantile(values, [1.0 / 3.0, 2.0 / 3.0]), dtype=float)


def q_cardinality_bucket(concepts: tuple[str, ...]) -> str:
    count = len(concepts)
    if count < 1:
        raise ValueError("Target Q must contain at least one concept.")
    return str(count) if count <= 2 else "3_plus"


def _ease_tercile(value: float, edges: np.ndarray) -> int:
    return int(np.searchsorted(edges, value, side="right"))


def build_donor_mapping(
    profiles: Iterable[StudentProfile],
    *,
    statistics: ReferenceStatistics,
    item_index: dict[str, int],
    q_lookup: dict[str, tuple[str, ...]],
    split: str,
    fold: int | None,
    replicate: int,
) -> DonorMapping:
    if replicate < 0:
        raise ValueError("Permutation replicate must be non-negative.")
    edges = ease_tercile_edges(statistics)
    assignments: list[dict[str, Any]] = []
    cell_counts: list[dict[str, Any]] = []
    movable_students: set[str] = set()
    changed_students: set[str] = set()

    for profile in sorted(profiles, key=lambda value: value.student):
        query = profile.query.copy()
        query["_item"] = query["exer_id"].astype(str)
        rows_by_item = {
            item: tuple(sorted(frame["source_row_id"].astype(str)))
            for item, frame in query.groupby("_item", sort=False)
        }
        cells: dict[tuple[str, int], list[str]] = {}
        for item in sorted(rows_by_item):
            if item not in item_index:
                raise RuntimeError(f"Donor target {item!r} is absent from item index.")
            stratum = (
                q_cardinality_bucket(q_lookup[item]),
                _ease_tercile(
                    float(statistics.item_ease[item_index[item]]), edges
                ),
            )
            cells.setdefault(stratum, []).append(item)

        item_to_donor: dict[str, str | None] = {}
        item_to_stratum: dict[str, tuple[str, int]] = {}
        for stratum, members in sorted(cells.items()):
            members = sorted(set(members))
            item_to_stratum.update({item: stratum for item in members})
            cell_counts.append(
                {
                    "student": profile.student,
                    "q_cardinality_bucket": stratum[0],
                    "ease_tercile": stratum[1],
                    "distinct_items": len(members),
                    "query_rows": int(sum(len(rows_by_item[item]) for item in members)),
                }
            )
            if len(members) < 2:
                item_to_donor[members[0]] = None
                continue
            movable_students.add(profile.student)
            ordered = sorted(
                members,
                key=lambda item: (
                    stable_fraction(
                        SPLIT_SEED,
                        DONOR_ORDER_NAMESPACE,
                        split,
                        "none" if fold is None else fold,
                        replicate,
                        profile.student,
                        stratum[0],
                        stratum[1],
                        item,
                    ),
                    item,
                ),
            )
            shift_fraction = stable_fraction(
                SPLIT_SEED,
                DONOR_SHIFT_NAMESPACE,
                split,
                "none" if fold is None else fold,
                replicate,
                profile.student,
                stratum[0],
                stratum[1],
            )
            shift = 1 + min(
                int(shift_fraction * (len(ordered) - 1)),
                len(ordered) - 2,
            )
            for position, item in enumerate(ordered):
                donor = ordered[(position + shift) % len(ordered)]
                if donor == item:
                    raise RuntimeError("Eligible donor mapping contains a fixed point.")
                item_to_donor[item] = donor

        for row in query.sort_values("source_row_id", kind="stable").itertuples(
            index=False
        ):
            item = str(row.exer_id)
            donor = item_to_donor[item]
            movable = donor is not None
            effective_donor = item if donor is None else donor
            if movable:
                changed_students.add(profile.student)
            stratum = item_to_stratum[item]
            assignments.append(
                {
                    "split": split,
                    "fold": -1 if fold is None else fold,
                    "replicate": replicate,
                    "source_row_id": str(row.source_row_id),
                    "stu_id": profile.student,
                    "target_item": item,
                    "donor_target_item": effective_donor,
                    "donor_source_row_id": rows_by_item[effective_donor][0],
                    "q_cardinality_bucket": stratum[0],
                    "ease_tercile": stratum[1],
                    "movable": movable,
                    "changed": movable and effective_donor != item,
                }
            )

    mapping = pd.DataFrame(assignments)
    if mapping.empty:
        raise RuntimeError("Donor mapping contains no query rows.")
    eligible = mapping["movable"].to_numpy(dtype=bool)
    changed = mapping["changed"].to_numpy(dtype=bool)
    eligible_fixed_points = int(
        (
            eligible
            & (
                mapping["target_item"].astype(str).to_numpy()
                == mapping["donor_target_item"].astype(str).to_numpy()
            )
        ).sum()
    )
    if eligible_fixed_points:
        raise RuntimeError("Eligible donor mapping contains fixed points.")
    changed_rows = int(changed.sum())
    mapping_hash = _hash_values(
        ":".join(
            (
                str(row.split),
                str(row.fold),
                str(row.replicate),
                str(row.source_row_id),
                str(row.stu_id),
                str(row.target_item),
                str(row.donor_target_item),
                str(row.q_cardinality_bucket),
                str(row.ease_tercile),
            )
        )
        for row in mapping.itertuples(index=False)
    )
    cell_hash = _hash_values(
        ":".join(map(str, row.values()))
        for row in sorted(
            cell_counts,
            key=lambda value: (
                value["student"],
                value["q_cardinality_bucket"],
                value["ease_tercile"],
            ),
        )
    )
    audit = {
        "split": split,
        "fold": fold,
        "replicate": replicate,
        "order_namespace": DONOR_ORDER_NAMESPACE,
        "shift_namespace": DONOR_SHIFT_NAMESPACE,
        "strata": "student x target-Q-cardinality(1/2/3+) x train-only ease tercile",
        "ease_tercile_edges": edges.tolist(),
        "collapsed_ease_edges": int(np.sum(np.diff(edges) <= 0.0)),
        "query_rows": len(mapping),
        "students": int(mapping["stu_id"].astype(str).nunique()),
        "cells": cell_counts,
        "cell_count": len(cell_counts),
        "cell_item_counts_sha256": cell_hash,
        "movable_rows": int(eligible.sum()),
        "changed_rows": changed_rows,
        "changed_row_fraction": changed_rows / len(mapping),
        "movable_students": len(movable_students),
        "changed_students": len(changed_students),
        "eligible_fixed_points": eligible_fixed_points,
        "cross_stratum_fallbacks": 0,
        "mapping_sha256": mapping_hash,
        "label_columns_read_for_mapping": [],
        "identified": (
            changed_rows / len(mapping) >= MIN_CHANGED_ROW_FRACTION
            and len(changed_students) >= MIN_MOVABLE_STUDENTS
        ),
        "identification_rule": (
            "changed_row_fraction>=0.20 and changed_students>=100"
        ),
    }
    return DonorMapping(rows=mapping, audit=audit)


def build_fold_references(
    profiles: list[StudentProfile],
    *,
    items: list[str],
    item_index: dict[str, int],
    q_lookup: dict[str, tuple[str, ...]],
) -> list[FoldReference]:
    output: list[FoldReference] = []
    for fold in range(NUM_FOLDS):
        reference = [profile for profile in profiles if profile.fold != fold]
        held = [profile for profile in profiles if profile.fold == fold]
        if {profile.student for profile in reference} & {
            profile.student for profile in held
        }:
            raise RuntimeError("OOF reference and held students overlap.")
        statistics, _ = build_reference_statistics(
            reference,
            items=items,
            item_index=item_index,
            q_lookup=q_lookup,
        )
        output.append(
            FoldReference(
                fold=fold,
                statistics=statistics,
                ease_tercile_edges=ease_tercile_edges(statistics),
                reference_students=tuple(
                    sorted(profile.student for profile in reference)
                ),
                reference_rows_sha256=statistics.reference_rows_hash,
            )
        )
    return output


def audit_dataset_protocol(
    dataset: str,
    source_dir: Path,
    *,
    permutation_replicates: int = PERMUTATION_REPLICATES,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if permutation_replicates != PERMUTATION_REPLICATES:
        raise ValueError("The preregistration fixes 200 permutation replicates.")
    protocol = load_validation_only_protocol(dataset, source_dir)
    profiles, optimizer_audit = build_optimizer_profiles(
        protocol.optimizer_train, q_lookup=protocol.q_lookup
    )
    protocol = finalize_protocol_for_retained_optimizer(protocol, profiles)
    items = sorted(retained_optimizer_items(profiles))
    item_index = {item: index for index, item in enumerate(items)}
    fold_references = build_fold_references(
        profiles,
        items=items,
        item_index=item_index,
        q_lookup=protocol.q_lookup,
    )

    optimizer_mappings = []
    fold_audits = []
    for reference in fold_references:
        held = [profile for profile in profiles if profile.fold == reference.fold]
        donor = build_donor_mapping(
            held,
            statistics=reference.statistics,
            item_index=item_index,
            q_lookup=protocol.q_lookup,
            split="optimizer_oof_query",
            fold=reference.fold,
            replicate=0,
        )
        optimizer_mappings.append(donor.rows)
        fold_audits.append(
            {
                "fold": reference.fold,
                "reference_students": len(reference.reference_students),
                "held_students": len(held),
                "reference_students_sha256": _hash_values(
                    reference.reference_students
                ),
                "reference_rows_sha256": reference.reference_rows_sha256,
                "ease_tercile_edges": reference.ease_tercile_edges.tolist(),
                "donor": donor.audit,
            }
        )
    optimizer_mapping = pd.concat(optimizer_mappings, ignore_index=True)

    validation_profiles = build_validation_profiles(
        protocol.validation_support,
        protocol.validation_query,
        q_lookup=protocol.q_lookup,
    )
    validation_statistics, _ = build_reference_statistics(
        profiles,
        items=items,
        item_index=item_index,
        q_lookup=protocol.q_lookup,
    )
    validation_mapping_zero: pd.DataFrame | None = None
    validation_zero_full_audit: dict[str, Any] | None = None
    permutation_audits = []
    for replicate in range(permutation_replicates):
        donor = build_donor_mapping(
            validation_profiles,
            statistics=validation_statistics,
            item_index=item_index,
            q_lookup=protocol.q_lookup,
            split="validation_query",
            fold=None,
            replicate=replicate,
        )
        if replicate == 0:
            validation_mapping_zero = donor.rows
            validation_zero_full_audit = donor.audit
        permutation_audits.append(
            {
                key: value
                for key, value in donor.audit.items()
                if key != "cells"
            }
        )
    assert validation_mapping_zero is not None
    assert validation_zero_full_audit is not None
    validation_zero_audit = permutation_audits[0]

    summary = {
        "schema_version": 1,
        "audit": "target_local_pairing_protocol",
        "dataset": dataset,
        "fixed_protocol": {
            "model_seed": MODEL_SEED,
            "split_seed": SPLIT_SEED,
            "optimizer_fraction": OPTIMIZER_FRACTION,
            "inner_support_fraction": INNER_SUPPORT_FRACTION,
            "optimizer_folds": NUM_FOLDS,
            "permutation_replicates": PERMUTATION_REPLICATES,
            "minimum_changed_row_fraction": MIN_CHANGED_ROW_FRACTION,
            "minimum_movable_students": MIN_MOVABLE_STUDENTS,
        },
        "source": protocol.audit,
        "optimizer": optimizer_audit,
        "folds": fold_audits,
        "validation_reference": {
            "optimizer_students": len(profiles),
            "reference_students_sha256": validation_statistics.reference_students_hash,
            "reference_rows_sha256": validation_statistics.reference_rows_hash,
            "ease_tercile_edges": ease_tercile_edges(
                validation_statistics
            ).tolist(),
        },
        "donor_replica_zero": validation_zero_full_audit,
        "validation_permutation_audits": permutation_audits,
        "leakage_audit": {
            "opened_files": list(ALLOWED_SOURCE_FILES),
            "test_files_opened": False,
            "legacy_manifest_opened": False,
            "validation_query_labels_used_for_split": False,
            "validation_query_labels_used_for_item_statistics": False,
            "validation_query_labels_used_for_donor_mapping": False,
            "optimizer_validation_student_overlap": 0,
            "validation_support_query_group_overlap": 0,
            "oof_reference_held_student_overlap": 0,
        },
        "hashes": {
            "optimizer_mapping_replicate_zero": _hash_values(
                optimizer_mapping["source_row_id"].astype(str)
                + ":"
                + optimizer_mapping["donor_target_item"].astype(str)
            ),
            "validation_mapping_replicate_zero": validation_zero_audit[
                "mapping_sha256"
            ],
            "validation_permutation_hashes": _hash_values(
                audit["mapping_sha256"] for audit in permutation_audits
            ),
        },
    }
    return summary, optimizer_mapping, validation_mapping_zero


def _parse_specs(values: list[str]) -> list[tuple[str, Path]]:
    output = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=SOURCE_DIR, got {value!r}.")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not name or not path.is_dir():
            raise ValueError(f"Invalid dataset specification: {value!r}.")
        output.append((name, path))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the preregistered target-local validation-only protocol and "
            "label-blind donor mappings. The CLI has no test argument."
        )
    )
    parser.add_argument(
        "--dataset", action="append", required=True, metavar="NAME=SOURCE_DIR"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, default=MODEL_SEED)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--permutation-replicates",
        type=int,
        default=PERMUTATION_REPLICATES,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_seed != MODEL_SEED:
        raise ValueError("This audit fixes model seed=42.")
    if args.split_seed != SPLIT_SEED:
        raise ValueError("This audit fixes split seed=2024.")
    if args.permutation_replicates != PERMUTATION_REPLICATES:
        raise ValueError("This audit fixes 200 conditional permutations.")

    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} is non-empty; refusing to overwrite protocol artifacts."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = []
    for name, source_dir in _parse_specs(args.dataset):
        summary, optimizer_mapping, validation_mapping = audit_dataset_protocol(
            name,
            source_dir,
            permutation_replicates=args.permutation_replicates,
        )
        optimizer_path = output_dir / f"{name}_optimizer_donor_rep0.csv"
        validation_path = output_dir / f"{name}_validation_donor_rep0.csv"
        optimizer_mapping.to_csv(optimizer_path, index=False)
        validation_mapping.to_csv(validation_path, index=False)
        summary["mapping_files"] = {
            "optimizer_rep0": {
                "path": str(optimizer_path.resolve()),
                "sha256": sha256_file(optimizer_path),
            },
            "validation_rep0": {
                "path": str(validation_path.resolve()),
                "sha256": sha256_file(validation_path),
            },
        }
        (output_dir / f"{name}_protocol.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        datasets.append(summary)

    aggregate = {
        "schema_version": 1,
        "audit": "target_local_pairing_protocol",
        "datasets": datasets,
        "formal_training_started": False,
        "test_accessed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                summary["dataset"]: summary["donor_replica_zero"]["identified"]
                for summary in datasets
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
