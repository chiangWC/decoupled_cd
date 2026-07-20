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
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

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
from scripts.paired_student_bootstrap import paired_student_cluster_bootstrap


MODEL_SEED = 42
SPLIT_SEED = 2024
NUM_FOLDS = 5
SUPPORT_FRACTION = 0.70
MIN_SUPPORT_GROUPS = 10
MIN_QUERY_GROUPS = 3
NUM_BINS = 5
BETA_STRENGTH = 20.0
BOOTSTRAP_REPLICATES = 2000

COMMON_NUMERIC_NAMES = (
    "theta_logit",
    "support_raw_accuracy",
    "log_support_rows",
    "target_q_cardinality",
    "train_only_item_ease",
    "log_train_only_item_count",
)
SIGNATURE_NAMES = (
    *tuple(f"residual_bin_{index}" for index in range(NUM_BINS)),
    *tuple(f"confidence_bin_{index}" for index in range(NUM_BINS)),
    "residual_at_theta",
    "confidence_at_theta",
    "local_slope",
)
VARIANTS = ("direct", "strong_2pl", "full", "capacity")


def _concepts(value: object) -> tuple[str, ...]:
    return tuple(token for token in canonical_concepts(value).split(",") if token)


def _hash_values(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def _logit(value: float) -> float:
    clipped = float(np.clip(value, 1e-8, 1.0 - 1e-8))
    return float(np.log(clipped / (1.0 - clipped)))


def support_theta(support: pd.DataFrame) -> tuple[float, float]:
    correct = float(support["label"].sum())
    attempts = float(len(support))
    posterior_accuracy = (correct + 1.0) / (attempts + 2.0)
    theta = float(np.clip(_logit(posterior_accuracy), -4.0, 4.0))
    return theta, correct / attempts


@dataclass(frozen=True)
class StudentProfile:
    student: str
    support: pd.DataFrame
    query: pd.DataFrame
    theta: float
    raw_accuracy: float
    seen_concepts: frozenset[str]
    fold: int


@dataclass(frozen=True)
class RankQuintileBinner:
    students: tuple[str, ...]
    theta: np.ndarray
    bins: np.ndarray
    displayed_edges: np.ndarray
    collapsed_edges: int
    seed: int

    @classmethod
    def fit(
        cls,
        profiles: Iterable[StudentProfile],
        *,
        seed: int,
    ) -> "RankQuintileBinner":
        values = [(profile.student, profile.theta) for profile in profiles]
        if len(values) < NUM_BINS:
            raise ValueError("At least five reference students are required.")
        ordered = sorted(
            values,
            key=lambda value: (
                value[1],
                stable_fraction(seed, "response-signature-theta-tie", value[0]),
            ),
        )
        students = tuple(value[0] for value in ordered)
        theta = np.asarray([value[1] for value in ordered], dtype=np.float64)
        bins = np.minimum(
            (np.arange(len(theta), dtype=np.int64) * NUM_BINS) // len(theta),
            NUM_BINS - 1,
        )
        edges = np.quantile(theta, [0.2, 0.4, 0.6, 0.8])
        return cls(
            students=students,
            theta=theta,
            bins=bins,
            displayed_edges=np.asarray(edges, dtype=np.float64),
            collapsed_edges=int(np.sum(np.diff(edges) <= 0.0)),
            seed=seed,
        )

    def assign(self, *, student: str, theta: float) -> int:
        lower = int(np.searchsorted(self.theta, theta, side="left"))
        upper = int(np.searchsorted(self.theta, theta, side="right"))
        if upper > lower:
            tie = stable_fraction(
                self.seed, "response-signature-theta-tie", student
            )
            rank = lower + min(int(tie * (upper - lower)), upper - lower - 1)
        else:
            rank = lower
        rank = min(max(rank, 0), len(self.theta) - 1)
        return min(int(rank * NUM_BINS // len(self.theta)), NUM_BINS - 1)


@dataclass(frozen=True)
class ReferenceStatistics:
    signatures: np.ndarray
    item_ease: np.ndarray
    item_count: np.ndarray
    bin_edges: np.ndarray
    collapsed_bin_edges: int
    reference_students_hash: str
    reference_rows_hash: str


@dataclass
class ExampleSet:
    common_numeric: list[np.ndarray]
    q_indices: list[tuple[int, ...]]
    item_indices: list[int]
    theta: list[float]
    full_signature: list[np.ndarray]
    capacity_signature: list[np.ndarray]
    labels: list[int]
    rows: list[dict[str, Any]]

    @classmethod
    def empty(cls) -> "ExampleSet":
        return cls([], [], [], [], [], [], [], [])


def load_protocol(
    directory: Path,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, tuple[str, ...]],
    dict[str, Any],
]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "student_disjoint_support_query":
        raise ValueError(f"Not a student-disjoint manifest: {manifest_path}")
    if int(manifest.get("audit", {}).get("seed", -1)) != SPLIT_SEED:
        raise ValueError("The manifest must use split_seed=2024.")
    required = ("train.csv", "valid_support.csv", "valid_query.csv", "Q_matrix.csv")
    verified_hashes = {}
    for filename in required:
        expected = manifest["files"][filename]["sha256"]
        actual = sha256_file(directory / filename)
        if actual != expected:
            raise RuntimeError(f"Manifest hash mismatch for {filename}.")
        verified_hashes[filename] = actual

    q_matrix, conflicts = derive_q_matrix(pd.read_csv(directory / "Q_matrix.csv"))
    if conflicts:
        # Long-form Q rows are expected; derive_q_matrix unions them by exercise.
        pass
    q_lookup = {
        str(row.exer_id): _concepts(row.cpt_seq)
        for row in q_matrix.itertuples(index=False)
    }
    splits = {}
    for name, filename in (
        ("train", "train.csv"),
        ("valid_support", "valid_support.csv"),
        ("valid_query", "valid_query.csv"),
    ):
        frame, _ = canonicalize_interactions(pd.read_csv(directory / filename))
        mapped = frame["exer_id"].astype(str).map(
            {item: ",".join(concepts) for item, concepts in q_lookup.items()}
        )
        if mapped.isna().any():
            raise RuntimeError(f"{filename} contains an exercise absent from Q.")
        frame["cpt_seq"] = mapped
        splits[name] = frame

    train_students = set(splits["train"]["stu_id"].astype(str))
    valid_support_students = set(splits["valid_support"]["stu_id"].astype(str))
    valid_query_students = set(splits["valid_query"]["stu_id"].astype(str))
    if train_students & valid_query_students:
        raise RuntimeError("Optimizer and validation students overlap.")
    if valid_support_students != valid_query_students:
        raise RuntimeError("Validation support/query student sets differ.")
    support_groups = set(
        zip(
            splits["valid_support"]["stu_id"].astype(str),
            splits["valid_support"]["exer_id"].astype(str),
            strict=True,
        )
    )
    query_groups = set(
        zip(
            splits["valid_query"]["stu_id"].astype(str),
            splits["valid_query"]["exer_id"].astype(str),
            strict=True,
        )
    )
    if support_groups & query_groups:
        raise RuntimeError("Validation support/query groups overlap.")
    return splits, q_lookup, {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "verified_file_hashes": verified_hashes,
        "optimizer_validation_student_overlap": 0,
        "validation_support_query_group_overlap": 0,
    }


def optimizer_profiles(
    frame: pd.DataFrame,
    *,
    q_lookup: dict[str, tuple[str, ...]],
) -> tuple[list[StudentProfile], dict[str, Any]]:
    profiles = []
    skipped = 0
    all_support_groups: set[tuple[str, str]] = set()
    all_query_groups: set[tuple[str, str]] = set()
    for student, student_frame in frame.groupby(
        frame["stu_id"].astype(str), sort=False
    ):
        student = str(student)
        query_items = {
            str(item)
            for item in student_frame["exer_id"].astype(str).unique()
            if stable_fraction(
                SPLIT_SEED, "response-signature-inner-query", student, item
            )
            >= SUPPORT_FRACTION
        }
        query_mask = student_frame["exer_id"].astype(str).isin(query_items)
        support = student_frame.loc[~query_mask].copy().reset_index(drop=True)
        query = student_frame.loc[query_mask].copy().reset_index(drop=True)
        support_group_count = support["exer_id"].astype(str).nunique()
        query_group_count = query["exer_id"].astype(str).nunique()
        if (
            support_group_count < MIN_SUPPORT_GROUPS
            or query_group_count < MIN_QUERY_GROUPS
        ):
            skipped += 1
            continue
        support_groups = {(student, str(item)) for item in support["exer_id"]}
        query_groups = {(student, str(item)) for item in query["exer_id"]}
        if support_groups & query_groups:
            raise RuntimeError("Optimizer support/query split is not atomic.")
        all_support_groups.update(support_groups)
        all_query_groups.update(query_groups)
        theta, raw_accuracy = support_theta(support)
        seen = {
            concept
            for item in support["exer_id"].astype(str)
            for concept in q_lookup[item]
        }
        fold = min(
            int(
                NUM_FOLDS
                * stable_fraction(
                    SPLIT_SEED, "response-signature-fold", student
                )
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
    if all_support_groups & all_query_groups:
        raise RuntimeError("Optimizer support/query groups overlap globally.")
    fold_counts = {
        str(fold): sum(profile.fold == fold for profile in profiles)
        for fold in range(NUM_FOLDS)
    }
    if any(count == 0 for count in fold_counts.values()):
        raise RuntimeError(f"An optimizer OOF fold is empty: {fold_counts}")
    assignment_hash = _hash_values(
        f"{profile.student}:{profile.fold}"
        for profile in sorted(profiles, key=lambda value: value.student)
    )
    return profiles, {
        "retained_students": len(profiles),
        "skipped_students": skipped,
        "support_rows": int(sum(len(profile.support) for profile in profiles)),
        "query_rows": int(sum(len(profile.query) for profile in profiles)),
        "support_query_group_overlap": 0,
        "fold_counts": fold_counts,
        "fold_assignment_sha256": assignment_hash,
    }


def validation_profiles(
    support: pd.DataFrame,
    query: pd.DataFrame,
    *,
    q_lookup: dict[str, tuple[str, ...]],
) -> list[StudentProfile]:
    output = []
    for student, student_support in support.groupby(
        support["stu_id"].astype(str), sort=False
    ):
        student = str(student)
        student_query = query.loc[query["stu_id"].astype(str) == student]
        if student_query.empty:
            continue
        theta, raw_accuracy = support_theta(student_support)
        seen = {
            concept
            for item in student_support["exer_id"].astype(str)
            for concept in q_lookup[item]
        }
        output.append(
            StudentProfile(
                student=student,
                support=student_support.copy().reset_index(drop=True),
                query=student_query.copy().reset_index(drop=True),
                theta=theta,
                raw_accuracy=raw_accuracy,
                seen_concepts=frozenset(seen),
                fold=-1,
            )
        )
    return output


def make_derangement(
    *,
    items: list[str],
    q_lookup: dict[str, tuple[str, ...]],
) -> tuple[dict[str, str | None], dict[str, Any]]:
    groups: dict[tuple[str, ...], list[str]] = {}
    for item in items:
        groups.setdefault(q_lookup[item], []).append(item)
    mapping: dict[str, str | None] = {}
    eligible_fixed_points = 0
    singleton_items = 0
    for q_key, members in groups.items():
        ordered = sorted(
            members,
            key=lambda item: (
                stable_fraction(
                    SPLIT_SEED,
                    "response-signature-derangement",
                    ",".join(q_key),
                    item,
                ),
                item,
            ),
        )
        if len(ordered) == 1:
            mapping[ordered[0]] = None
            singleton_items += 1
            continue
        for index, item in enumerate(ordered):
            mapping[item] = ordered[(index + 1) % len(ordered)]
            eligible_fixed_points += int(mapping[item] == item)
    if eligible_fixed_points:
        raise RuntimeError("Capacity derangement contains a fixed point.")
    return mapping, {
        "groups": len(groups),
        "singleton_items": singleton_items,
        "eligible_items": len(items) - singleton_items,
        "eligible_fixed_points": eligible_fixed_points,
        "mapping_sha256": _hash_values(
            f"{item}:{mapping[item]}" for item in sorted(mapping)
        ),
    }


def build_reference_statistics(
    profiles: list[StudentProfile],
    *,
    items: list[str],
    item_index: dict[str, int],
    q_lookup: dict[str, tuple[str, ...]],
) -> tuple[ReferenceStatistics, RankQuintileBinner]:
    binner = RankQuintileBinner.fit(profiles, seed=SPLIT_SEED)
    correct = np.zeros((len(items), NUM_BINS), dtype=np.float64)
    attempts = np.zeros((len(items), NUM_BINS), dtype=np.float64)
    reference_rows = []
    for profile in profiles:
        bin_index = binner.assign(student=profile.student, theta=profile.theta)
        for row in profile.query.itertuples(index=False):
            index = item_index[str(row.exer_id)]
            correct[index, bin_index] += int(row.label)
            attempts[index, bin_index] += 1.0
            reference_rows.append(str(row.source_row_id))

    global_correct = correct.sum(axis=0)
    global_attempts = attempts.sum(axis=0)
    global_probability = (global_correct + 1.0) / (global_attempts + 2.0)
    q_groups: dict[tuple[str, ...], list[int]] = {}
    for item, index in item_index.items():
        q_groups.setdefault(q_lookup[item], []).append(index)
    signatures = np.zeros((len(items), len(SIGNATURE_NAMES)), dtype=np.float64)
    for members in q_groups.values():
        if len(members) == 1:
            # Leave-item-out exact-Q information does not exist. Full and the
            # deranged control both receive an all-zero 13-dimensional block.
            continue
        group_correct = correct[members].sum(axis=0)
        group_attempts = attempts[members].sum(axis=0)
        for index in members:
            prototype = (
                group_correct
                - correct[index]
                + BETA_STRENGTH * global_probability
            ) / (
                group_attempts
                - attempts[index]
                + BETA_STRENGTH
            )
            item_probability = (
                correct[index] + BETA_STRENGTH * prototype
            ) / (attempts[index] + BETA_STRENGTH)
            confidence = attempts[index] / (attempts[index] + BETA_STRENGTH)
            residual = confidence * (
                np.log(np.clip(item_probability, 0.01, 0.99) / np.clip(1.0 - item_probability, 0.01, 0.99))
                - np.log(np.clip(prototype, 0.01, 0.99) / np.clip(1.0 - prototype, 0.01, 0.99))
            )
            signatures[index, :NUM_BINS] = residual
            signatures[index, NUM_BINS : 2 * NUM_BINS] = confidence

    item_correct = correct.sum(axis=1)
    item_attempts = attempts.sum(axis=1)
    global_rate = (float(item_correct.sum()) + 1.0) / (
        float(item_attempts.sum()) + 2.0
    )
    item_ease = (item_correct + 2.0 * global_rate) / (item_attempts + 2.0)
    return ReferenceStatistics(
        signatures=signatures,
        item_ease=item_ease,
        item_count=item_attempts,
        bin_edges=binner.displayed_edges,
        collapsed_bin_edges=binner.collapsed_edges,
        reference_students_hash=_hash_values(
            sorted(profile.student for profile in profiles)
        ),
        reference_rows_hash=_hash_values(sorted(reference_rows)),
    ), binner


def signature_at_bin(curve: np.ndarray, bin_index: int) -> np.ndarray:
    residual = curve[:NUM_BINS]
    confidence = curve[NUM_BINS : 2 * NUM_BINS]
    if bin_index == 0:
        slope = residual[1] - residual[0]
    elif bin_index == NUM_BINS - 1:
        slope = residual[-1] - residual[-2]
    else:
        slope = (residual[bin_index + 1] - residual[bin_index - 1]) / 2.0
    return np.concatenate(
        (
            residual,
            confidence,
            np.asarray(
                [residual[bin_index], confidence[bin_index], slope],
                dtype=np.float64,
            ),
        )
    )


def append_examples(
    output: ExampleSet,
    profiles: Iterable[StudentProfile],
    *,
    statistics: ReferenceStatistics,
    binner: RankQuintileBinner,
    items: list[str],
    item_index: dict[str, int],
    concepts: list[str],
    concept_index: dict[str, int],
    q_lookup: dict[str, tuple[str, ...]],
    derangement: dict[str, str | None],
    dataset: str,
    split: str,
) -> None:
    del items, concepts
    for profile in profiles:
        theta_bin = binner.assign(student=profile.student, theta=profile.theta)
        for row in profile.query.itertuples(index=False):
            item = str(row.exer_id)
            index = item_index[item]
            q_values = q_lookup[item]
            q_indices = tuple(concept_index[value] for value in q_values)
            overlap = len(profile.seen_concepts.intersection(q_values))
            if overlap == 0:
                coverage = "exact_zero"
            elif overlap == len(q_values):
                coverage = "full"
            else:
                coverage = "partial"
            q_cardinality = len(q_values)
            full_curve = statistics.signatures[index]
            capacity_item = derangement[item]
            if capacity_item is None:
                capacity_curve = np.zeros(len(SIGNATURE_NAMES), dtype=np.float64)
            else:
                capacity_curve = statistics.signatures[item_index[capacity_item]]
            output.common_numeric.append(
                np.asarray(
                    [
                        profile.theta,
                        profile.raw_accuracy,
                        np.log1p(len(profile.support)),
                        q_cardinality,
                        statistics.item_ease[index],
                        np.log1p(statistics.item_count[index]),
                    ],
                    dtype=np.float64,
                )
            )
            output.q_indices.append(q_indices)
            output.item_indices.append(index)
            output.theta.append(profile.theta)
            output.full_signature.append(signature_at_bin(full_curve, theta_bin))
            output.capacity_signature.append(
                signature_at_bin(capacity_curve, theta_bin)
            )
            output.labels.append(int(row.label))
            output.rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "source_row_id": str(row.source_row_id),
                    "stu_id": profile.student,
                    "exer_id": item,
                    "label": int(row.label),
                    "theta": profile.theta,
                    "theta_bin": theta_bin,
                    "coverage_bucket": coverage,
                    "q_cardinality": q_cardinality,
                    "q_cardinality_bucket": (
                        str(q_cardinality) if q_cardinality <= 2 else "3_plus"
                    ),
                    "support_rows": len(profile.support),
                    "optimizer_fold": profile.fold,
                }
            )


def make_design_matrix(
    examples: ExampleSet,
    *,
    variant: str,
    num_items: int,
    num_concepts: int,
) -> sparse.csr_matrix:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    common = sparse.csr_matrix(np.stack(examples.common_numeric))
    row_indices = np.arange(len(examples.labels), dtype=np.int64)
    q_rows: list[int] = []
    q_columns: list[int] = []
    q_values: list[float] = []
    for row, indices in enumerate(examples.q_indices):
        for index in indices:
            q_rows.append(row)
            q_columns.append(index)
            q_values.append(1.0)
    q_matrix = sparse.csr_matrix(
        (q_values, (q_rows, q_columns)),
        shape=(len(row_indices), num_concepts),
    )
    direct = sparse.hstack((common, q_matrix), format="csr")
    if variant == "direct":
        return direct
    item_indices = np.asarray(examples.item_indices, dtype=np.int64)
    item = sparse.csr_matrix(
        (np.ones(len(row_indices)), (row_indices, item_indices)),
        shape=(len(row_indices), num_items),
    )
    theta_item = sparse.csr_matrix(
        (np.asarray(examples.theta), (row_indices, item_indices)),
        shape=(len(row_indices), num_items),
    )
    strong = sparse.hstack((direct, item, theta_item), format="csr")
    if variant == "strong_2pl":
        return strong
    signature = np.stack(
        examples.full_signature if variant == "full" else examples.capacity_signature
    )
    return sparse.hstack((strong, sparse.csr_matrix(signature)), format="csr")


def make_preprocessed_designs(
    fit: ExampleSet,
    validation: ExampleSet,
    *,
    num_items: int,
    num_concepts: int,
) -> tuple[dict[str, sparse.csr_matrix], dict[str, sparse.csr_matrix], dict[str, Any]]:
    """Scale dense numeric blocks while preserving sparse indicators verbatim."""
    fit_common = np.stack(fit.common_numeric)
    validation_common = np.stack(validation.common_numeric)
    common_scaler = StandardScaler()
    scaled_fit_common = common_scaler.fit_transform(fit_common)
    scaled_validation_common = common_scaler.transform(validation_common)

    fit_full_signature = np.stack(fit.full_signature)
    fit_capacity_signature = np.stack(fit.capacity_signature)
    validation_full_signature = np.stack(validation.full_signature)
    validation_capacity_signature = np.stack(validation.capacity_signature)
    # A single, label-free scaler is fit symmetrically to both same-width
    # signature blocks. Full and Capacity therefore share exact preprocessing.
    signature_scaler = StandardScaler()
    signature_scaler.fit(
        np.vstack((fit_full_signature, fit_capacity_signature))
    )
    scaled_fit_signature = {
        "full": signature_scaler.transform(fit_full_signature),
        "capacity": signature_scaler.transform(fit_capacity_signature),
    }
    scaled_validation_signature = {
        "full": signature_scaler.transform(validation_full_signature),
        "capacity": signature_scaler.transform(validation_capacity_signature),
    }

    raw_fit_direct = make_design_matrix(
        fit, variant="direct", num_items=num_items, num_concepts=num_concepts
    )
    raw_validation_direct = make_design_matrix(
        validation, variant="direct", num_items=num_items, num_concepts=num_concepts
    )
    raw_fit_strong = make_design_matrix(
        fit, variant="strong_2pl", num_items=num_items, num_concepts=num_concepts
    )
    raw_validation_strong = make_design_matrix(
        validation, variant="strong_2pl", num_items=num_items, num_concepts=num_concepts
    )
    common_width = len(COMMON_NUMERIC_NAMES)
    fit_direct = sparse.hstack(
        (sparse.csr_matrix(scaled_fit_common), raw_fit_direct[:, common_width:]),
        format="csr",
    )
    validation_direct = sparse.hstack(
        (
            sparse.csr_matrix(scaled_validation_common),
            raw_validation_direct[:, common_width:],
        ),
        format="csr",
    )
    fit_strong = sparse.hstack(
        (sparse.csr_matrix(scaled_fit_common), raw_fit_strong[:, common_width:]),
        format="csr",
    )
    validation_strong = sparse.hstack(
        (
            sparse.csr_matrix(scaled_validation_common),
            raw_validation_strong[:, common_width:],
        ),
        format="csr",
    )
    fit_matrices = {"direct": fit_direct, "strong_2pl": fit_strong}
    validation_matrices = {
        "direct": validation_direct,
        "strong_2pl": validation_strong,
    }
    for variant in ("full", "capacity"):
        fit_matrices[variant] = sparse.hstack(
            (fit_strong, sparse.csr_matrix(scaled_fit_signature[variant])),
            format="csr",
        )
        validation_matrices[variant] = sparse.hstack(
            (
                validation_strong,
                sparse.csr_matrix(scaled_validation_signature[variant]),
            ),
            format="csr",
        )
    return fit_matrices, validation_matrices, {
        "common_dense_scaler": "StandardScaler fit on optimizer OOF common numeric",
        "signature_dense_scaler": (
            "one shared StandardScaler fit symmetrically on pooled Full/Capacity "
            "optimizer OOF signatures"
        ),
        "q_and_item_one_hot_scaled": False,
        "theta_item_interaction_scaled": False,
        "common_scaler_sha256": _hash_values(
            np.concatenate((common_scaler.mean_, common_scaler.scale_))
        ),
        "signature_scaler_sha256": _hash_values(
            np.concatenate((signature_scaler.mean_, signature_scaler.scale_))
        ),
    }


def fit_predictors(
    fit: ExampleSet,
    validation: ExampleSet,
    *,
    num_items: int,
    num_concepts: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    labels = np.asarray(fit.labels, dtype=np.int64)
    predictions = {}
    dimensions = {}
    nonzero = {}
    fit_matrices, validation_matrices, preprocessing_audit = (
        make_preprocessed_designs(
            fit,
            validation,
            num_items=num_items,
            num_concepts=num_concepts,
        )
    )
    for variant in VARIANTS:
        x_fit = fit_matrices[variant]
        x_validation = validation_matrices[variant]
        estimator = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            max_iter=1000,
            random_state=MODEL_SEED,
        )
        estimator.fit(x_fit, labels)
        predictions[variant] = estimator.predict_proba(x_validation)[:, 1]
        dimensions[variant] = int(x_fit.shape[1])
        nonzero[variant] = int(x_fit.nnz)
    if dimensions["full"] != dimensions["capacity"]:
        raise RuntimeError("Full and capacity feature dimensions differ.")
    return predictions, {
        "fit_rows": len(fit.labels),
        "feature_dimensions": dimensions,
        "fit_nonzero_entries": nonzero,
        "preprocessing": preprocessing_audit,
        "logistic": "L2 LogisticRegression(C=1, solver=liblinear)",
        "full_signature_dimensions": len(SIGNATURE_NAMES),
        "all_variants_share_common_preprocessing": True,
    }


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return {
        "auc": (
            float(roc_auc_score(labels, probabilities))
            if len(np.unique(labels)) == 2
            else None
        ),
        "brier": float(brier_score_loss(labels, probabilities)),
        "acc": float(accuracy_score(labels, probabilities >= 0.5)),
        "rmse": float(np.sqrt(mean_squared_error(labels, probabilities))),
        "rows": int(len(labels)),
        "label_0": int((labels == 0).sum()),
        "label_1": int((labels == 1).sum()),
    }


def slice_metrics(rows: pd.DataFrame, probability: np.ndarray) -> dict[str, Any]:
    labels = rows["label"].to_numpy(dtype=np.int64)
    output = {"overall": _metrics(labels, probability), "coverage": {}, "q_cardinality": {}}
    for value in ("exact_zero", "partial", "full"):
        mask = rows["coverage_bucket"].astype(str).to_numpy() == value
        if mask.any():
            output["coverage"][value] = _metrics(labels[mask], probability[mask])
    for value in ("1", "2", "3_plus"):
        mask = rows["q_cardinality_bucket"].astype(str).to_numpy() == value
        if mask.any():
            output["q_cardinality"][value] = _metrics(labels[mask], probability[mask])
    return output


def bootstrap_comparison(
    rows: pd.DataFrame,
    *,
    full_probability: np.ndarray,
    control_probability: np.ndarray,
    scope: str,
) -> dict[str, Any]:
    full = rows.copy()
    control = rows.copy()
    full["prob"] = full_probability
    control["prob"] = control_probability
    # The shared bootstrap utility calls the exact-zero bucket ``zero`` while
    # this audit uses the explicit ``exact_zero`` label in its reports.
    full["coverage_bucket"] = full["coverage_bucket"].replace(
        {"exact_zero": "zero"}
    )
    control["coverage_bucket"] = control["coverage_bucket"].replace(
        {"exact_zero": "zero"}
    )
    translated_scope = "bucket:zero" if scope == "exact_zero" else scope
    return paired_student_cluster_bootstrap(
        full=full,
        control=control,
        scope=translated_scope,
        replicates=BOOTSTRAP_REPLICATES,
        seed=SPLIT_SEED,
    )


def audit_dataset(name: str, directory: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    splits, q_lookup, protocol_audit = load_protocol(directory)
    items = sorted(q_lookup)
    item_index = {item: index for index, item in enumerate(items)}
    concepts = sorted({concept for values in q_lookup.values() for concept in values})
    concept_index = {concept: index for index, concept in enumerate(concepts)}
    derangement, derangement_audit = make_derangement(
        items=items, q_lookup=q_lookup
    )
    profiles, optimizer_audit = optimizer_profiles(
        splits["train"], q_lookup=q_lookup
    )
    fit = ExampleSet.empty()
    fold_audits = []
    for fold in range(NUM_FOLDS):
        reference = [profile for profile in profiles if profile.fold != fold]
        held = [profile for profile in profiles if profile.fold == fold]
        if {profile.student for profile in reference} & {
            profile.student for profile in held
        }:
            raise RuntimeError("OOF reference and held students overlap.")
        statistics, binner = build_reference_statistics(
            reference,
            items=items,
            item_index=item_index,
            q_lookup=q_lookup,
        )
        before = len(fit.labels)
        append_examples(
            fit,
            held,
            statistics=statistics,
            binner=binner,
            items=items,
            item_index=item_index,
            concepts=concepts,
            concept_index=concept_index,
            q_lookup=q_lookup,
            derangement=derangement,
            dataset=name,
            split="optimizer_oof_query",
        )
        fold_audits.append(
            {
                "fold": fold,
                "reference_students": len(reference),
                "held_students": len(held),
                "fit_rows": len(fit.labels) - before,
                "student_overlap": 0,
                "reference_students_sha256": statistics.reference_students_hash,
                "reference_rows_sha256": statistics.reference_rows_hash,
                "theta_quintile_edges": statistics.bin_edges.tolist(),
                "collapsed_theta_edges": statistics.collapsed_bin_edges,
            }
        )

    validation = validation_profiles(
        splits["valid_support"],
        splits["valid_query"],
        q_lookup=q_lookup,
    )
    validation_statistics, validation_binner = build_reference_statistics(
        profiles,
        items=items,
        item_index=item_index,
        q_lookup=q_lookup,
    )
    valid_examples = ExampleSet.empty()
    append_examples(
        valid_examples,
        validation,
        statistics=validation_statistics,
        binner=validation_binner,
        items=items,
        item_index=item_index,
        concepts=concepts,
        concept_index=concept_index,
        q_lookup=q_lookup,
        derangement=derangement,
        dataset=name,
        split="valid_query",
    )
    fit_row_ids = {str(row["source_row_id"]) for row in fit.rows}
    valid_row_ids = {str(row["source_row_id"]) for row in valid_examples.rows}
    if fit_row_ids & valid_row_ids:
        raise RuntimeError("Optimizer and validation query rows overlap.")
    predictions, model_audit = fit_predictors(
        fit,
        valid_examples,
        num_items=len(items),
        num_concepts=len(concepts),
    )
    prediction_rows = pd.DataFrame(valid_examples.rows)
    metrics = {}
    for variant, probability in predictions.items():
        prediction_rows[f"prob_{variant}"] = probability
        metrics[variant] = slice_metrics(prediction_rows, probability)

    control_name = max(
        ("strong_2pl", "capacity"),
        key=lambda value: metrics[value]["overall"]["auc"],
    )
    bootstraps = {}
    for control in ("strong_2pl", "capacity"):
        bootstraps[control] = {
            "overall": bootstrap_comparison(
                prediction_rows,
                full_probability=predictions["full"],
                control_probability=predictions[control],
                scope="overall",
            )
        }
        zero_metrics = metrics["full"]["coverage"].get("exact_zero")
        if zero_metrics is not None and zero_metrics["auc"] is not None:
            bootstraps[control]["exact_zero"] = bootstrap_comparison(
                prediction_rows,
                full_probability=predictions["full"],
                control_probability=predictions[control],
                scope="exact_zero",
            )

    overall_full = metrics["full"]["overall"]
    overall_control = metrics[control_name]["overall"]
    zero_full = metrics["full"]["coverage"].get("exact_zero")
    zero_control = metrics[control_name]["coverage"].get("exact_zero")
    prediction_hashes = {
        variant: hashlib.sha256(
            np.asarray(probability, dtype="<f8").tobytes()
        ).hexdigest()
        for variant, probability in predictions.items()
    }
    summary = {
        "schema_version": 1,
        "audit": "cross_fitted_conditional_response_signature",
        "dataset": name,
        "source": protocol_audit,
        "fixed_protocol": {
            "model_seed": MODEL_SEED,
            "split_seed": SPLIT_SEED,
            "optimizer_folds": NUM_FOLDS,
            "student_inner_support_fraction": SUPPORT_FRACTION,
            "minimum_support_groups": MIN_SUPPORT_GROUPS,
            "minimum_query_groups": MIN_QUERY_GROUPS,
            "theta": "clip(logit((correct+1)/(attempts+2)),-4,4)",
            "theta_bins": "five stable-rank student quintiles",
            "beta_strength": BETA_STRENGTH,
            "logistic_c": 1.0,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "optimizer": optimizer_audit,
        "folds": fold_audits,
        "validation_reference": {
            "students": len(profiles),
            "reference_students_sha256": validation_statistics.reference_students_hash,
            "reference_rows_sha256": validation_statistics.reference_rows_hash,
            "theta_quintile_edges": validation_statistics.bin_edges.tolist(),
            "collapsed_theta_edges": validation_statistics.collapsed_bin_edges,
        },
        "derangement": derangement_audit,
        "leakage_audit": {
            "optimizer_validation_student_overlap": 0,
            "optimizer_validation_query_row_overlap": 0,
            "validation_support_query_group_overlap": 0,
            "oof_reference_held_student_overlap": 0,
            "validation_query_labels_used_for_theta": False,
            "validation_query_labels_used_for_signature": False,
            "validation_query_labels_used_for_scaler_or_fit": False,
        },
        "rows": {
            "optimizer_oof": len(fit.labels),
            "validation": len(valid_examples.labels),
            "optimizer_oof_row_sha256": _hash_values(
                row["source_row_id"] for row in fit.rows
            ),
            "validation_row_sha256": _hash_values(
                row["source_row_id"] for row in valid_examples.rows
            ),
        },
        "model_audit": model_audit,
        "feature_contract": {
            "direct": "six common numeric fields plus target-Q multi-hot",
            "strong_2pl": (
                "Direct plus item-ID intercept and theta multiplied by item-ID; "
                "slopes are unconstrained"
            ),
            "full": "Strong2PL-like plus the declared 13-dimensional signature",
            "capacity": (
                "Strong2PL-like plus the same 13 dimensions from a stable, "
                "no-fixed-point exact-Q item derangement"
            ),
            "common_numeric_names": COMMON_NUMERIC_NAMES,
            "signature_names": SIGNATURE_NAMES,
        },
        "metrics": metrics,
        "stronger_control": control_name,
        "delta_vs_stronger_control": {
            "overall_auc": overall_full["auc"] - overall_control["auc"],
            "overall_brier": overall_full["brier"] - overall_control["brier"],
            "exact_zero_auc": (
                None
                if zero_full is None
                or zero_control is None
                or zero_full["auc"] is None
                or zero_control["auc"] is None
                else zero_full["auc"] - zero_control["auc"]
            ),
            "exact_zero_brier": (
                None
                if zero_full is None or zero_control is None
                else zero_full["brier"] - zero_control["brier"]
            ),
        },
        "bootstrap": bootstraps,
        "prediction_sha256": prediction_hashes,
    }
    return summary, prediction_rows


def evaluate_gate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if len(summaries) != 2:
        raise ValueError("The fixed gate requires exactly ASSIST17 and XES3G5M.")
    by_name = {summary["dataset"].lower(): summary for summary in summaries}
    assist = next(
        (value for key, value in by_name.items() if "assist17" in key), None
    )
    xes = next((value for key, value in by_name.items() if "xes" in key), None)
    if assist is None or xes is None:
        raise ValueError("Dataset names must identify ASSIST17 and XES3G5M.")
    overall_auc = [
        value["delta_vs_stronger_control"]["overall_auc"]
        for value in (assist, xes)
    ]
    overall_brier = [
        value["delta_vs_stronger_control"]["overall_brier"]
        for value in (assist, xes)
    ]
    ci_lows = [
        value["bootstrap"][value["stronger_control"]]["overall"]["ci_low"]
        for value in (assist, xes)
    ]
    xes_zero_auc = xes["delta_vs_stronger_control"]["exact_zero_auc"]
    xes_zero_brier = xes["delta_vs_stronger_control"]["exact_zero_brier"]
    checks = {
        "both_overall_auc_at_least_0.003": all(
            value >= 0.003 for value in overall_auc
        ),
        "one_overall_auc_at_least_0.005": max(overall_auc) >= 0.005,
        "both_overall_brier_regression_at_most_0.0001": all(
            value <= 0.0001 for value in overall_brier
        ),
        "one_overall_bootstrap_ci_low_above_zero": max(ci_lows) > 0.0,
        "xes_exact_zero_auc_at_least_0.005": (
            xes_zero_auc is not None and xes_zero_auc >= 0.005
        ),
        "xes_exact_zero_brier_regression_at_most_0.0002": (
            xes_zero_brier is not None and xes_zero_brier <= 0.0002
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "dataset_stronger_controls": {
            value["dataset"]: value["stronger_control"]
            for value in (assist, xes)
        },
        "dataset_overall_auc_deltas": {
            value["dataset"]: value["delta_vs_stronger_control"]["overall_auc"]
            for value in (assist, xes)
        },
        "dataset_overall_brier_deltas": {
            value["dataset"]: value["delta_vs_stronger_control"]["overall_brier"]
            for value in (assist, xes)
        },
        "dataset_overall_ci_lows": {
            value["dataset"]: value["bootstrap"][value["stronger_control"]]["overall"]["ci_low"]
            for value in (assist, xes)
        },
        "xes_exact_zero_auc_delta": xes_zero_auc,
        "xes_exact_zero_brier_delta": xes_zero_brier,
        "rule": (
            "Full vs the single stronger Strong2PL-like/Capacity control per "
            "dataset: both overall dAUC>=.003, one>=.005, both dBrier<=.0001, "
            "one student-bootstrap 95% CI low>0, and XES exact-zero dAUC>=.005 "
            "with dBrier<=.0002. Any failed check rejects the mechanism."
        ),
    }


def _parse_specs(values: list[str]) -> list[tuple[str, Path]]:
    output = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=DIR, got {value!r}.")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not name or not path.is_dir():
            raise ValueError(f"Invalid dataset specification: {value!r}.")
        output.append((name, path))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-fitted conditional response signature audit."
    )
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-seed", type=int, default=MODEL_SEED)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_seed != MODEL_SEED:
        raise ValueError("This audit fixes model seed=42.")
    if args.split_seed != SPLIT_SEED:
        raise ValueError("This audit fixes split seed=2024.")
    if args.bootstrap_replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("This audit fixes 2000 bootstrap replicates.")
    specs = _parse_specs(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    metric_rows = []
    for name, directory in specs:
        summary, rows = audit_dataset(name, directory)
        summaries.append(summary)
        prediction_path = output_dir / f"{name}_valid_predictions.csv"
        rows.to_csv(prediction_path, index=False, float_format="%.17g")
        summary["prediction_file_sha256"] = sha256_file(prediction_path)
        summary["prediction_value_hash_contract"] = (
            "little-endian float64 bytes; use pandas read_csv "
            "float_precision=round_trip for reconstruction"
        )
        (output_dir / f"{name}_audit.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        for variant, groups in summary["metrics"].items():
            for scope_type, scopes in groups.items():
                if scope_type == "overall":
                    metric_rows.append(
                        {"dataset": name, "variant": variant, "scope": "overall", **scopes}
                    )
                else:
                    for scope, values in scopes.items():
                        metric_rows.append(
                            {
                                "dataset": name,
                                "variant": variant,
                                "scope": f"{scope_type}:{scope}",
                                **values,
                            }
                        )
    gate = evaluate_gate(summaries)
    aggregate = {
        "schema_version": 1,
        "audit": "cross_fitted_conditional_response_signature",
        "datasets": summaries,
        "gate": gate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(metric_rows).to_csv(output_dir / "metrics.csv", index=False)
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
