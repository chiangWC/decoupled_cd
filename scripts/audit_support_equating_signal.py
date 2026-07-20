from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pool_protocol import (
    assign_student_disjoint_support_query,
    canonical_concepts,
    canonicalize_interactions,
    derive_q_matrix,
    sha256_file,
    stable_fraction,
)


BASE_NUMERIC_NAMES = (
    "support_raw_accuracy",
    "log_support_rows",
    "support_concept_coverage",
    "target_item_ease",
    "log_target_item_population_rows",
    "target_q_cardinality",
)
EQUATING_NUMERIC_NAMES = (
    "support_item_ease_mean",
    "support_item_ease_std",
    "support_item_ease_q10",
    "support_item_ease_q25",
    "support_item_ease_q50",
    "support_item_ease_q75",
    "support_item_ease_q90",
    "support_item_ease_min",
    "support_item_ease_max",
    "response_minus_ease_mean",
    "response_minus_ease_std",
    "response_minus_ease_q25",
    "response_minus_ease_q50",
    "response_minus_ease_q75",
    *tuple(
        value
        for bin_index in range(5)
        for value in (
            f"difficulty_bin_{bin_index}_accuracy",
            f"difficulty_bin_{bin_index}_log_rows",
            f"difficulty_bin_{bin_index}_confidence",
        )
    ),
    "target_minus_support_ease",
    "support_fraction_harder_than_target",
    "support_fraction_easier_than_target",
    "target_local_residual",
    "target_local_confidence",
)


@dataclass(frozen=True)
class ItemCountStore:
    item_index: dict[str, int]
    total_correct: np.ndarray
    total_attempts: np.ndarray
    student_correct: dict[str, np.ndarray]
    student_attempts: dict[str, np.ndarray]


@dataclass(frozen=True)
class ItemStatistics:
    ease: np.ndarray
    attempts: np.ndarray
    bin_boundaries: np.ndarray
    global_rate: float


@dataclass
class ExampleSet:
    base_numeric: list[np.ndarray]
    full_numeric: list[np.ndarray]
    target_items: list[int]
    target_concepts: list[tuple[int, ...]]
    labels: list[int]
    rows: list[dict[str, Any]]

    @classmethod
    def empty(cls) -> "ExampleSet":
        return cls([], [], [], [], [], [])


@dataclass(frozen=True)
class SupportProfile:
    base_student: np.ndarray
    equating_static: np.ndarray
    item_ease: np.ndarray
    residual: np.ndarray
    seen_concepts: frozenset[str]


def _concepts(value: object) -> tuple[str, ...]:
    return tuple(token for token in canonical_concepts(value).split(",") if token)


def _load_q_union(path: Path) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    q_matrix, _ = derive_q_matrix(pd.read_csv(path))
    lookup = {
        str(row.exer_id): _concepts(row.cpt_seq)
        for row in q_matrix.itertuples(index=False)
    }
    return q_matrix, lookup


def _attach_q_union(
    frame: pd.DataFrame,
    q_lookup: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    output = frame.copy()
    mapped = output["exer_id"].astype(str).map(
        {exercise: ",".join(concepts) for exercise, concepts in q_lookup.items()}
    )
    if mapped.isna().any():
        missing = output.loc[mapped.isna(), "exer_id"].astype(str).unique()[:10]
        raise ValueError(f"Exercises absent from source Q-matrix: {missing.tolist()}")
    output["cpt_seq"] = mapped
    return output


def load_protocol(
    dataset_dir: Path,
    *,
    split_seed: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, tuple[str, ...]], dict[str, Any]]:
    q_matrix, q_lookup = _load_q_union(dataset_dir / "Q_matrix.csv")
    manifest_path = dataset_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    generated = (dataset_dir / "valid_support.csv").exists() or (
        manifest is not None
        and manifest.get("protocol") == "student_disjoint_support_query"
    )
    if generated:
        output = {}
        for name, filename in (
            ("train", "train.csv"),
            ("valid_support", "valid_support.csv"),
            ("valid_query", "valid_query.csv"),
        ):
            output[name], _ = canonicalize_interactions(
                pd.read_csv(dataset_dir / filename)
            )
            output[name] = _attach_q_union(output[name], q_lookup)
        source = {
            "mode": "generated_student_disjoint",
            "directory": str(dataset_dir.resolve()),
            "files": {
                filename: sha256_file(dataset_dir / filename)
                for filename in (
                    "train.csv",
                    "valid_support.csv",
                    "valid_query.csv",
                    "Q_matrix.csv",
                )
            },
            "manifest": manifest,
        }
        return output, q_matrix, q_lookup, source

    standard = {}
    hashes = {}
    for split in ("train", "valid", "test"):
        standard[split], _ = canonicalize_interactions(
            pd.read_csv(dataset_dir / f"{split}.csv")
        )
        standard[split] = _attach_q_union(standard[split], q_lookup)
        hashes[f"{split}.csv"] = sha256_file(dataset_dir / f"{split}.csv")
    splits, audit = assign_student_disjoint_support_query(
        train_frame=standard["train"],
        valid_frame=standard["valid"],
        test_frame=standard["test"],
        q_matrix=q_matrix,
        seed=split_seed,
    )
    hashes["Q_matrix.csv"] = sha256_file(dataset_dir / "Q_matrix.csv")
    return {
        "train": splits["train"],
        "valid_support": splits["valid_support"],
        "valid_query": splits["valid_query"],
    }, q_matrix, q_lookup, {
        "mode": "derived_in_memory_from_standard",
        "directory": str(dataset_dir.resolve()),
        "files": hashes,
        "student_disjoint_audit": audit,
    }


def split_optimizer_support_query(
    frame: pd.DataFrame,
    *,
    seed: int,
    query_fraction: float,
    min_support_rows: int,
    min_query_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    support_parts = []
    query_parts = []
    retained = 0
    skipped = 0
    for student, student_frame in frame.groupby(frame["stu_id"].astype(str), sort=False):
        student = str(student)
        query_exercises = {
            str(exercise)
            for exercise in student_frame["exer_id"].astype(str).unique()
            if stable_fraction(seed, "equating-fit-query", student, exercise)
            < query_fraction
        }
        query_mask = student_frame["exer_id"].astype(str).isin(query_exercises)
        support = student_frame.loc[~query_mask]
        query = student_frame.loc[query_mask]
        if len(support) < min_support_rows or len(query) < min_query_rows:
            skipped += 1
            continue
        retained += 1
        support_parts.append(support)
        query_parts.append(query)
    if not support_parts or not query_parts:
        raise RuntimeError("Optimizer support/query partition retained no students.")
    support = pd.concat(support_parts, ignore_index=True)
    query = pd.concat(query_parts, ignore_index=True)
    support_groups = set(
        zip(support["stu_id"].astype(str), support["exer_id"].astype(str), strict=True)
    )
    query_groups = set(
        zip(query["stu_id"].astype(str), query["exer_id"].astype(str), strict=True)
    )
    overlap = support_groups & query_groups
    if overlap:
        raise RuntimeError(f"Optimizer support/query group leakage: {len(overlap)}")
    return support, query, {
        "retained_students": retained,
        "skipped_students": skipped,
        "support_rows": len(support),
        "query_rows": len(query),
        "student_exercise_group_overlap": 0,
    }


def build_item_count_store(frame: pd.DataFrame) -> ItemCountStore:
    items = sorted(frame["exer_id"].astype(str).unique())
    item_index = {item: index for index, item in enumerate(items)}
    total_correct = np.zeros(len(items), dtype=np.float64)
    total_attempts = np.zeros(len(items), dtype=np.float64)
    student_correct: dict[str, np.ndarray] = {}
    student_attempts: dict[str, np.ndarray] = {}
    for row in frame.itertuples(index=False):
        student = str(row.stu_id)
        item = item_index[str(row.exer_id)]
        if student not in student_correct:
            student_correct[student] = np.zeros(len(items), dtype=np.float64)
            student_attempts[student] = np.zeros(len(items), dtype=np.float64)
        total_correct[item] += int(row.label)
        total_attempts[item] += 1
        student_correct[student][item] += int(row.label)
        student_attempts[student][item] += 1
    return ItemCountStore(
        item_index=item_index,
        total_correct=total_correct,
        total_attempts=total_attempts,
        student_correct=student_correct,
        student_attempts=student_attempts,
    )


def item_statistics(
    store: ItemCountStore,
    *,
    leave_student_out: str | None,
    prior_strength: float,
) -> ItemStatistics:
    correct = store.total_correct.copy()
    attempts = store.total_attempts.copy()
    if leave_student_out is not None:
        correct -= store.student_correct[leave_student_out]
        attempts -= store.student_attempts[leave_student_out]
    if np.any(correct < -1e-8) or np.any(attempts < -1e-8):
        raise RuntimeError("Negative leave-student-out item counts.")
    total_attempts = float(attempts.sum())
    if total_attempts <= 0:
        raise RuntimeError("No population rows remain after leave-student-out.")
    global_rate = float(correct.sum() / total_attempts)
    ease = (correct + prior_strength * global_rate) / (attempts + prior_strength)
    eligible = ease[attempts > 0]
    boundaries = np.quantile(eligible, [0.2, 0.4, 0.6, 0.8])
    return ItemStatistics(
        ease=ease,
        attempts=attempts,
        bin_boundaries=np.asarray(boundaries, dtype=np.float64),
        global_rate=global_rate,
    )


def _quantiles(values: np.ndarray, probabilities: tuple[float, ...]) -> list[float]:
    return [float(value) for value in np.quantile(values, probabilities)]


def build_support_profile(
    support: pd.DataFrame,
    *,
    statistics: ItemStatistics,
    item_index: dict[str, int],
    q_lookup: dict[str, tuple[str, ...]],
    num_concepts: int,
    bin_confidence_strength: float,
) -> SupportProfile:
    labels = support["label"].to_numpy(dtype=np.float64)
    indices = np.asarray(
        [item_index[str(item)] for item in support["exer_id"]], dtype=np.int64
    )
    ease = statistics.ease[indices]
    residual = labels - ease
    seen = {
        concept
        for exercise in support["exer_id"].astype(str)
        for concept in q_lookup[exercise]
    }
    base_student = np.asarray(
        [
            float(labels.mean()),
            float(np.log1p(len(labels))),
            float(len(seen) / num_concepts),
        ],
        dtype=np.float64,
    )
    ease_summary = [
        float(ease.mean()),
        float(ease.std()),
        *_quantiles(ease, (0.10, 0.25, 0.50, 0.75, 0.90)),
        float(ease.min()),
        float(ease.max()),
    ]
    residual_summary = [
        float(residual.mean()),
        float(residual.std()),
        *_quantiles(residual, (0.25, 0.50, 0.75)),
    ]
    bins = np.digitize(ease, statistics.bin_boundaries, right=True)
    bin_summary = []
    for bin_index in range(5):
        mask = bins == bin_index
        count = int(mask.sum())
        accuracy = float(labels[mask].mean()) if count else float(labels.mean())
        bin_summary.extend(
            (
                accuracy,
                float(np.log1p(count)),
                float(count / (count + bin_confidence_strength)),
            )
        )
    return SupportProfile(
        base_student=base_student,
        equating_static=np.asarray(
            ease_summary + residual_summary + bin_summary, dtype=np.float64
        ),
        item_ease=ease,
        residual=residual,
        seen_concepts=frozenset(seen),
    )


def build_row_features(
    profile: SupportProfile,
    *,
    target_item: str,
    target_concepts: tuple[str, ...],
    statistics: ItemStatistics,
    item_index: dict[str, int],
    local_bandwidth: float,
    local_confidence_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    target_index = item_index[target_item]
    target_ease = float(statistics.ease[target_index])
    base = np.concatenate(
        (
            profile.base_student,
            np.asarray(
                [
                    target_ease,
                    np.log1p(statistics.attempts[target_index]),
                    np.log1p(len(target_concepts)),
                ],
                dtype=np.float64,
            ),
        )
    )
    distance = np.abs(profile.item_ease - target_ease)
    weights = np.exp(-distance / local_bandwidth)
    weight_sum = float(weights.sum())
    local_residual = float(np.dot(weights, profile.residual) / weight_sum)
    target_relative = np.asarray(
        [
            target_ease - float(profile.item_ease.mean()),
            float(np.mean(profile.item_ease < target_ease)),
            float(np.mean(profile.item_ease > target_ease)),
            local_residual,
            weight_sum / (weight_sum + local_confidence_strength),
        ],
        dtype=np.float64,
    )
    full = np.concatenate((base, profile.equating_static, target_relative))
    if len(base) != len(BASE_NUMERIC_NAMES):
        raise RuntimeError("Base feature shape does not match its declaration.")
    if len(full) != len(BASE_NUMERIC_NAMES) + len(EQUATING_NUMERIC_NAMES):
        raise RuntimeError("Equating feature shape does not match its declaration.")
    return base, full


def append_student_examples(
    examples: ExampleSet,
    *,
    student: str,
    support: pd.DataFrame,
    query: pd.DataFrame,
    statistics: ItemStatistics,
    item_store: ItemCountStore,
    q_lookup: dict[str, tuple[str, ...]],
    concept_index: dict[str, int],
    dataset_name: str,
    split_name: str,
    bin_confidence_strength: float,
    local_bandwidth: float,
    local_confidence_strength: float,
) -> int:
    profile = build_support_profile(
        support,
        statistics=statistics,
        item_index=item_store.item_index,
        q_lookup=q_lookup,
        num_concepts=len(concept_index),
        bin_confidence_strength=bin_confidence_strength,
    )
    exact_zero_rows = 0
    for row in query.itertuples(index=False):
        target_item = str(row.exer_id)
        if target_item not in item_store.item_index:
            raise ValueError(f"Unknown target item {target_item}.")
        target_concepts = q_lookup[target_item]
        base, full = build_row_features(
            profile,
            target_item=target_item,
            target_concepts=target_concepts,
            statistics=statistics,
            item_index=item_store.item_index,
            local_bandwidth=local_bandwidth,
            local_confidence_strength=local_confidence_strength,
        )
        target_indices = tuple(concept_index[value] for value in target_concepts)
        exact_zero = not bool(profile.seen_concepts.intersection(target_concepts))
        exact_zero_rows += int(exact_zero)
        examples.base_numeric.append(base)
        examples.full_numeric.append(full)
        examples.target_items.append(item_store.item_index[target_item])
        examples.target_concepts.append(target_indices)
        examples.labels.append(int(row.label))
        examples.rows.append(
            {
                "dataset": dataset_name,
                "split": split_name,
                "source_row_id": str(row.source_row_id),
                "student": student,
                "exercise": target_item,
                "label": int(row.label),
                "exact_zero": int(exact_zero),
                "support_rows": len(support),
            }
        )
    return exact_zero_rows


def _capacity_numeric(base_numeric: np.ndarray) -> np.ndarray:
    extra_dimensions = len(EQUATING_NUMERIC_NAMES)
    repeats = int(np.ceil(extra_dimensions / base_numeric.shape[1]))
    duplicated = np.tile(base_numeric, (1, repeats))[:, :extra_dimensions]
    return np.concatenate((base_numeric, duplicated), axis=1)


def make_design_matrix(
    examples: ExampleSet,
    *,
    full: bool,
    num_items: int,
    num_concepts: int,
    capacity_control: bool = False,
) -> sparse.csr_matrix:
    if full and capacity_control:
        raise ValueError("full and capacity_control are mutually exclusive.")
    base_numeric = np.stack(examples.base_numeric)
    if full:
        numeric_values = np.stack(examples.full_numeric)
    elif capacity_control:
        numeric_values = _capacity_numeric(base_numeric)
    else:
        numeric_values = base_numeric
    numeric = sparse.csr_matrix(numeric_values, dtype=np.float64)
    row_index = np.arange(len(examples.labels), dtype=np.int64)
    item = sparse.csr_matrix(
        (
            np.ones(len(row_index), dtype=np.float64),
            (row_index, np.asarray(examples.target_items)),
        ),
        shape=(len(row_index), num_items),
    )
    concept_rows = []
    concept_columns = []
    concept_values = []
    for row, concepts in enumerate(examples.target_concepts):
        weight = 1.0 / len(concepts)
        for concept in concepts:
            concept_rows.append(row)
            concept_columns.append(concept)
            concept_values.append(weight)
    q_features = sparse.csr_matrix(
        (concept_values, (concept_rows, concept_columns)),
        shape=(len(row_index), num_concepts),
    )
    return sparse.hstack((numeric, item, q_features), format="csr")


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
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
        "positive_rate": float(labels.mean()) if len(labels) else None,
        "label_counts": {
            "0": int((labels == 0).sum()),
            "1": int((labels == 1).sum()),
        },
    }


def evaluate_slices(
    labels: np.ndarray,
    probabilities: np.ndarray,
    exact_zero: np.ndarray,
) -> dict[str, dict[str, Any]]:
    return {
        "overall": _metrics(labels, probabilities),
        "exact_zero": _metrics(labels[exact_zero], probabilities[exact_zero]),
        "nonzero": _metrics(labels[~exact_zero], probabilities[~exact_zero]),
    }


def fit_predictors(
    fit_examples: ExampleSet,
    valid_examples: ExampleSet,
    *,
    num_items: int,
    num_concepts: int,
    seed: int,
    extra_trees: int,
    extra_trees_jobs: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    y_fit = np.asarray(fit_examples.labels, dtype=np.int64)
    x_fit_raw = make_design_matrix(
        fit_examples, full=False, num_items=num_items, num_concepts=num_concepts
    )
    x_valid_raw = make_design_matrix(
        valid_examples, full=False, num_items=num_items, num_concepts=num_concepts
    )
    x_fit_capacity = make_design_matrix(
        fit_examples,
        full=False,
        capacity_control=True,
        num_items=num_items,
        num_concepts=num_concepts,
    )
    x_valid_capacity = make_design_matrix(
        valid_examples,
        full=False,
        capacity_control=True,
        num_items=num_items,
        num_concepts=num_concepts,
    )
    x_fit_full = make_design_matrix(
        fit_examples, full=True, num_items=num_items, num_concepts=num_concepts
    )
    x_valid_full = make_design_matrix(
        valid_examples, full=True, num_items=num_items, num_concepts=num_concepts
    )
    if x_fit_capacity.shape[1] != x_fit_full.shape[1]:
        raise RuntimeError("Capacity and equated feature counts differ.")

    def logistic() -> Any:
        return make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(
                C=1.0,
                max_iter=500,
                solver="liblinear",
                random_state=seed,
            ),
        )

    raw_logistic = logistic()
    capacity_logistic = logistic()
    equated_logistic = logistic()
    raw_logistic.fit(x_fit_raw, y_fit)
    capacity_logistic.fit(x_fit_capacity, y_fit)
    equated_logistic.fit(x_fit_full, y_fit)
    predictions = {
        "raw_logistic": raw_logistic.predict_proba(x_valid_raw)[:, 1],
        "capacity_logistic": capacity_logistic.predict_proba(
            x_valid_capacity
        )[:, 1],
        "equated_logistic": equated_logistic.predict_proba(x_valid_full)[:, 1],
    }
    model_audit: dict[str, Any] = {
        "raw_features": x_fit_raw.shape[1],
        "capacity_features": x_fit_capacity.shape[1],
        "equated_features": x_fit_full.shape[1],
        "fit_rows": x_fit_raw.shape[0],
        "capacity_control": (
            "Repeat shared raw numeric features to match the equated numeric width."
        ),
    }
    if extra_trees > 0:
        def forest() -> ExtraTreesClassifier:
            return ExtraTreesClassifier(
                n_estimators=extra_trees,
                max_depth=18,
                min_samples_leaf=20,
                max_features="sqrt",
                n_jobs=extra_trees_jobs,
                random_state=seed,
            )

        raw_forest = forest()
        capacity_forest = forest()
        equated_forest = forest()
        raw_forest.fit(x_fit_raw, y_fit)
        capacity_forest.fit(x_fit_capacity, y_fit)
        equated_forest.fit(x_fit_full, y_fit)
        predictions["raw_extra_trees"] = raw_forest.predict_proba(
            x_valid_raw
        )[:, 1]
        predictions["capacity_extra_trees"] = capacity_forest.predict_proba(
            x_valid_capacity
        )[:, 1]
        predictions["equated_extra_trees"] = equated_forest.predict_proba(
            x_valid_full
        )[:, 1]
    return predictions, model_audit


def passes_gate(
    auc_deltas: list[float],
    brier_deltas: list[float],
) -> tuple[bool, bool, bool]:
    auc_passed = (
        len(auc_deltas) >= 2
        and all(delta >= 0.003 for delta in auc_deltas)
        and max(auc_deltas) >= 0.005
    )
    brier_passed = (
        len(brier_deltas) >= 2
        and all(delta <= 1e-4 for delta in brier_deltas)
    )
    return auc_passed and brier_passed, auc_passed, brier_passed


def audit_dataset(
    name: str,
    dataset_dir: Path,
    *,
    split_seed: int,
    fit_query_fraction: float,
    min_support_rows: int,
    min_query_rows: int,
    prior_strength: float,
    bin_confidence_strength: float,
    local_bandwidth: float,
    local_confidence_strength: float,
    extra_trees: int,
    extra_trees_jobs: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    splits, q_matrix, q_lookup, source = load_protocol(
        dataset_dir, split_seed=split_seed
    )
    optimizer = splits["train"]
    valid_support = splits["valid_support"]
    valid_query = splits["valid_query"]
    fit_support, fit_query, fit_partition = split_optimizer_support_query(
        optimizer,
        seed=split_seed,
        query_fraction=fit_query_fraction,
        min_support_rows=min_support_rows,
        min_query_rows=min_query_rows,
    )
    optimizer_students = set(optimizer["stu_id"].astype(str))
    valid_students = set(valid_query["stu_id"].astype(str))
    student_overlap = optimizer_students & valid_students
    if student_overlap:
        raise RuntimeError(f"Optimizer/validation student leakage: {len(student_overlap)}")
    valid_support_groups = set(
        zip(
            valid_support["stu_id"].astype(str),
            valid_support["exer_id"].astype(str),
            strict=True,
        )
    )
    valid_query_groups = set(
        zip(
            valid_query["stu_id"].astype(str),
            valid_query["exer_id"].astype(str),
            strict=True,
        )
    )
    if valid_support_groups & valid_query_groups:
        raise RuntimeError("Validation support/query student-exercise leakage.")

    concepts = sorted({value for values in q_lookup.values() for value in values})
    concept_index = {value: index for index, value in enumerate(concepts)}
    item_store = build_item_count_store(optimizer)
    fit_examples = ExampleSet.empty()
    for student, support in fit_support.groupby(
        fit_support["stu_id"].astype(str), sort=False
    ):
        student = str(student)
        query = fit_query.loc[fit_query["stu_id"].astype(str) == student]
        statistics = item_statistics(
            item_store,
            leave_student_out=student,
            prior_strength=prior_strength,
        )
        append_student_examples(
            fit_examples,
            student=student,
            support=support,
            query=query,
            statistics=statistics,
            item_store=item_store,
            q_lookup=q_lookup,
            concept_index=concept_index,
            dataset_name=name,
            split_name="optimizer_fit_query",
            bin_confidence_strength=bin_confidence_strength,
            local_bandwidth=local_bandwidth,
            local_confidence_strength=local_confidence_strength,
        )

    validation_statistics = item_statistics(
        item_store,
        leave_student_out=None,
        prior_strength=prior_strength,
    )
    valid_examples = ExampleSet.empty()
    for student, support in valid_support.groupby(
        valid_support["stu_id"].astype(str), sort=False
    ):
        student = str(student)
        query = valid_query.loc[valid_query["stu_id"].astype(str) == student]
        if query.empty:
            continue
        append_student_examples(
            valid_examples,
            student=student,
            support=support,
            query=query,
            statistics=validation_statistics,
            item_store=item_store,
            q_lookup=q_lookup,
            concept_index=concept_index,
            dataset_name=name,
            split_name="valid_query",
            bin_confidence_strength=bin_confidence_strength,
            local_bandwidth=local_bandwidth,
            local_confidence_strength=local_confidence_strength,
        )
    if not fit_examples.labels or not valid_examples.labels:
        raise RuntimeError("Audit produced no fitting or validation examples.")

    predictions, model_audit = fit_predictors(
        fit_examples,
        valid_examples,
        num_items=len(item_store.item_index),
        num_concepts=len(concept_index),
        seed=split_seed,
        extra_trees=extra_trees,
        extra_trees_jobs=extra_trees_jobs,
    )
    labels = np.asarray(valid_examples.labels, dtype=np.int64)
    exact_zero = np.asarray(
        [bool(row["exact_zero"]) for row in valid_examples.rows], dtype=bool
    )
    model_metrics = {
        model: evaluate_slices(labels, values, exact_zero)
        for model, values in predictions.items()
    }
    control_name = max(
        (
            model
            for model in model_metrics
            if model.startswith(("raw_", "capacity_"))
        ),
        key=lambda model: model_metrics[model]["overall"]["auc"],
    )
    candidate_name = max(
        (model for model in model_metrics if model.startswith("equated_")),
        key=lambda model: model_metrics[model]["overall"]["auc"],
    )
    control = model_metrics[control_name]["overall"]
    candidate = model_metrics[candidate_name]["overall"]
    for model, slices in model_metrics.items():
        slices["overall"]["delta_auc_vs_stronger_control"] = (
            slices["overall"]["auc"] - control["auc"]
        )
        slices["overall"]["delta_brier_vs_stronger_control"] = (
            slices["overall"]["brier"] - control["brier"]
        )
    summary = {
        "dataset": name,
        "source": source,
        "protocol": {
            "split_seed": split_seed,
            "fit_query_fraction": fit_query_fraction,
            "optimizer_validation_student_overlap": 0,
            "validation_support_query_group_overlap": 0,
            "fit_partition": fit_partition,
            "q_semantics": "per-exercise union of every long-form Q row",
            "multi_concept_exercises": int(
                q_matrix["cpt_seq"].astype(str).str.contains(",").sum()
            ),
            "leave_student_out_item_statistics": True,
            "validation_statistics_source": "optimizer train only",
        },
        "features": {
            "common_numeric": list(BASE_NUMERIC_NAMES),
            "equating_numeric": list(EQUATING_NUMERIC_NAMES),
            "capacity_control": "deterministic repetition of shared numeric features",
            "shared_target_item_identity": True,
            "shared_target_q_multihot": True,
        },
        "model_audit": model_audit,
        "validation": {
            "students": int(valid_query["stu_id"].nunique()),
            "rows": len(valid_examples.labels),
            "exact_zero_rows": int(exact_zero.sum()),
            "exact_zero_label_counts": {
                "0": int(((labels == 0) & exact_zero).sum()),
                "1": int(((labels == 1) & exact_zero).sum()),
            },
        },
        "models": model_metrics,
        "stronger_control_name": control_name,
        "best_candidate_name": candidate_name,
        "best_candidate_delta_auc": candidate["auc"] - control["auc"],
        "best_candidate_delta_brier": candidate["brier"] - control["brier"],
    }
    rows = pd.DataFrame(valid_examples.rows)
    for model, values in predictions.items():
        rows[f"prediction_{model}"] = values
    return summary, rows


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
        description="Audit support-composition difficulty-equating signal."
    )
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-seed", type=int, default=2024)
    parser.add_argument("--fit-query-fraction", type=float, default=0.25)
    parser.add_argument("--min-support-rows", type=int, default=10)
    parser.add_argument("--min-query-rows", type=int, default=3)
    parser.add_argument("--prior-strength", type=float, default=10.0)
    parser.add_argument("--bin-confidence-strength", type=float, default=5.0)
    parser.add_argument("--local-bandwidth", type=float, default=0.10)
    parser.add_argument("--local-confidence-strength", type=float, default=5.0)
    parser.add_argument("--extra-trees", type=int, default=160)
    parser.add_argument("--extra-trees-jobs", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split_seed != 2024:
        raise ValueError("This audit fixes split_seed=2024.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    table_rows = []
    for name, path in _parse_specs(args.dataset):
        summary, rows = audit_dataset(
            name,
            path,
            split_seed=args.split_seed,
            fit_query_fraction=args.fit_query_fraction,
            min_support_rows=args.min_support_rows,
            min_query_rows=args.min_query_rows,
            prior_strength=args.prior_strength,
            bin_confidence_strength=args.bin_confidence_strength,
            local_bandwidth=args.local_bandwidth,
            local_confidence_strength=args.local_confidence_strength,
            extra_trees=args.extra_trees,
            extra_trees_jobs=args.extra_trees_jobs,
        )
        summaries.append(summary)
        rows.to_csv(output_dir / f"{name}_valid_predictions.csv", index=False)
        (output_dir / f"{name}_audit.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        for model, slices in summary["models"].items():
            for slice_name, metrics in slices.items():
                table_rows.append(
                    {
                        "dataset": name,
                        "model": model,
                        "slice": slice_name,
                        **metrics,
                    }
                )
    auc_deltas = [float(value["best_candidate_delta_auc"]) for value in summaries]
    brier_deltas = [float(value["best_candidate_delta_brier"]) for value in summaries]
    passed, auc_passed, brier_passed = passes_gate(auc_deltas, brier_deltas)
    aggregate = {
        "schema_version": 1,
        "audit": "support_composition_difficulty_equating",
        "datasets": summaries,
        "gate": {
            "rule": (
                "relative to the stronger raw/capacity control: delta AUC >= .003 "
                "on both datasets, one >= .005, and every delta Brier <= 1e-4"
            ),
            "passed": passed,
            "auc_passed": auc_passed,
            "brier_passed": brier_passed,
            "dataset_auc_deltas": {
                value["dataset"]: value["best_candidate_delta_auc"]
                for value in summaries
            },
            "dataset_brier_deltas": {
                value["dataset"]: value["best_candidate_delta_brier"]
                for value in summaries
            },
            "stronger_controls": {
                value["dataset"]: value["stronger_control_name"]
                for value in summaries
            },
            "best_candidates": {
                value["dataset"]: value["best_candidate_name"]
                for value in summaries
            },
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(table_rows).to_csv(output_dir / "metrics.csv", index=False)
    print(json.dumps(aggregate["gate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
