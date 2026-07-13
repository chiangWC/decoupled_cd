from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import prepare_experiment_split_bundles
from utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activation audit for MNAR/exposure-aware state completion.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--registry", default="configs/r28_registry.json")
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="dataset=/absolute/path/to/holdout_validation_predictions.csv",
    )
    parser.add_argument("--maximum-cells", type=int, default=200000)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--output", default="results/r28/mnar_activation.json")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_prediction_map(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        dataset, path = value.split("=", 1)
        result[dataset] = Path(path)
    return result


def cell_features(evidence: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    attempts = evidence[..., 0].astype(np.float64)
    correct = evidence[..., 1].astype(np.float64)
    observed = attempts > 0
    student_attempts = attempts.sum(axis=1)
    concept_attempts = attempts.sum(axis=0)
    student_success = correct.sum(axis=1) / np.maximum(student_attempts, 1.0)
    concept_success = correct.sum(axis=0) / np.maximum(concept_attempts, 1.0)
    student_coverage = observed.mean(axis=1)
    concept_coverage = observed.mean(axis=0)
    student = np.stack(
        [student_coverage, student_success, np.log1p(student_attempts)],
        axis=1,
    )
    concept = np.stack(
        [concept_coverage, concept_success, np.log1p(concept_attempts)],
        axis=1,
    )
    return observed, student, concept, attempts


def sampled_cells(
    *,
    observed: np.ndarray,
    student: np.ndarray,
    concept: np.ndarray,
    maximum: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(observed.ravel())
    negative = np.flatnonzero(~observed.ravel())
    per_class = min(maximum // 2, len(positive), len(negative))
    selected = np.concatenate(
        [
            rng.choice(positive, per_class, replace=False),
            rng.choice(negative, per_class, replace=False),
        ]
    )
    rng.shuffle(selected)
    student_ids, concept_ids = np.unravel_index(selected, observed.shape)
    coverage = np.stack(
        [student[student_ids, 0], concept[concept_ids, 0]],
        axis=1,
    )
    rich = np.concatenate([student[student_ids], concept[concept_ids]], axis=1)
    labels = observed[student_ids, concept_ids].astype(np.int64)
    return coverage, rich, labels, student_ids


def crossfit_auc(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, np.ndarray]:
    predictions = np.empty(labels.size, dtype=np.float64)
    splitter = GroupKFold(n_splits=5)
    for train_indices, test_indices in splitter.split(features, labels, groups):
        model = LogisticRegression(max_iter=300, class_weight="balanced")
        model.fit(features[train_indices], labels[train_indices])
        predictions[test_indices] = model.predict_proba(features[test_indices])[:, 1]
    return float(roc_auc_score(labels, predictions)), predictions


def residual_association(
    *,
    predictions: pd.DataFrame,
    propensity: np.ndarray,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float | int]:
    error = np.abs(predictions["label"].to_numpy(dtype=np.float64) - predictions["prob"].to_numpy(dtype=np.float64))
    coverage = predictions["target_coverage"].fillna(0.0).to_numpy(dtype=np.float64)
    design = np.stack([np.ones_like(coverage), coverage, coverage**2], axis=1)
    error_residual = error - design @ np.linalg.lstsq(design, error, rcond=None)[0]
    propensity_residual = propensity - design @ np.linalg.lstsq(design, propensity, rcond=None)[0]
    observed = float(np.corrcoef(error_residual, propensity_residual)[0, 1])

    students = predictions["stu_id"].astype(str).to_numpy()
    unique_students = np.unique(students)
    groups = {student: np.flatnonzero(students == student) for student in unique_students}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(bootstrap_samples):
        sampled = rng.choice(unique_students, size=len(unique_students), replace=True)
        indices = np.concatenate([groups[student] for student in sampled])
        value = np.corrcoef(error_residual[indices], propensity_residual[indices])[0, 1]
        if np.isfinite(value):
            values.append(float(value))
    low, high = np.percentile(np.asarray(values), [2.5, 97.5])
    return {
        "correlation": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_samples": len(values),
    }


def target_propensity(
    *,
    predictions: pd.DataFrame,
    bundle: Any,
    student_features: np.ndarray,
    concept_features: np.ndarray,
    model: LogisticRegression,
) -> np.ndarray:
    values = np.empty(len(predictions), dtype=np.float64)
    q_matrix = bundle.q_matrix_tensor.numpy()
    for row_index, row in enumerate(predictions.itertuples(index=False)):
        student_id = int(row.mapped_student_id)
        exercise_id = int(row.mapped_exercise_id)
        concepts = np.flatnonzero(q_matrix[exercise_id] > 0)
        if concepts.size == 0:
            values[row_index] = 0.5
            continue
        rich = np.concatenate(
            [
                np.repeat(student_features[student_id][None, :], concepts.size, axis=0),
                concept_features[concepts],
            ],
            axis=1,
        )
        values[row_index] = float(model.predict_proba(rich)[:, 1].mean())
    return values


def audit_dataset(
    *,
    dataset: str,
    prediction_path: Path,
    data_root: Path,
    registry: dict[str, Any],
    maximum_cells: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    spec = registry["datasets"][dataset]
    directory = data_root / spec["holdout_dir"]
    bundles = prepare_experiment_split_bundles(
        train_interactions_path=directory / "train.csv",
        valid_interactions_path=directory / "valid.csv",
        test_interactions_path=directory / "test.csv",
        q_matrix_path=directory / "Q_matrix.csv",
    )
    evidence = bundles["train"].student_concept_evidence_tensor.numpy()
    observed, student, concept, _ = cell_features(evidence)
    coverage, rich, labels, groups = sampled_cells(
        observed=observed,
        student=student,
        concept=concept,
        maximum=maximum_cells,
        seed=42,
    )
    coverage_auc, _ = crossfit_auc(features=coverage, labels=labels, groups=groups)
    rich_auc, _ = crossfit_auc(features=rich, labels=labels, groups=groups)
    propensity_model = LogisticRegression(max_iter=300, class_weight="balanced").fit(rich, labels)

    predictions = pd.read_csv(prediction_path)
    required = {"stu_id", "mapped_student_id", "mapped_exercise_id", "label", "prob", "target_coverage"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"{prediction_path} is missing {sorted(required - set(predictions.columns))}")
    propensity = target_propensity(
        predictions=predictions,
        bundle=bundles["valid"],
        student_features=student,
        concept_features=concept,
        model=propensity_model,
    )
    association = residual_association(
        predictions=predictions,
        propensity=propensity,
        bootstrap_samples=bootstrap_samples,
        seed=42,
    )
    return {
        "dataset": dataset,
        "sampled_cells": int(labels.size),
        "coverage_only_mask_auc": coverage_auc,
        "richer_mask_auc": rich_auc,
        "mask_auc_gain": rich_auc - coverage_auc,
        "propensity_error_association_after_coverage_control": association,
    }


def main() -> None:
    args = parse_args()
    if args.maximum_cells < 1000 or args.bootstrap_samples < 100:
        raise ValueError("Use at least 1000 cells and 100 bootstrap samples.")
    registry = load_json(args.registry)
    prediction_map = parse_prediction_map(args.prediction)
    reports = [
        audit_dataset(
            dataset=dataset,
            prediction_path=path,
            data_root=Path(args.data_root),
            registry=registry,
            maximum_cells=args.maximum_cells,
            bootstrap_samples=args.bootstrap_samples,
        )
        for dataset, path in sorted(prediction_map.items())
    ]
    gains = [float(report["mask_auc_gain"]) for report in reports]
    association_significant = any(
        report["propensity_error_association_after_coverage_control"]["ci_low"] > 0.0
        or report["propensity_error_association_after_coverage_control"]["ci_high"] < 0.0
        for report in reports
    )
    activation = (
        (len(gains) >= 2 and all(gain >= 0.02 for gain in gains[:2]))
        or any(gain >= 0.03 for gain in gains)
    ) and association_significant
    payload = {
        "schema_version": 1,
        "seed": 42,
        "reports": reports,
        "activation_threshold": "both mask AUROC gains >=0.02 or one >=0.03, plus significant residual error association",
        "exposure_aware_completion_activated": activation,
        "next_action": "implement_exposure_dr" if activation else "skip_to_partial_vae",
    }
    write_json(payload, args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
