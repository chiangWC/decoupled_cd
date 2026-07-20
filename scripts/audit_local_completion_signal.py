from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
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


@dataclass(frozen=True)
class PseudoHoldout:
    student: str
    hidden_concepts: tuple[str, ...]
    visible_rows: pd.DataFrame
    target_correct: dict[str, int]
    target_attempts: dict[str, int]
    original_rows: int
    removed_rows: int


def _concepts(value: object) -> tuple[str, ...]:
    return tuple(token for token in canonical_concepts(value).split(",") if token)


def build_pseudo_holdout(
    student_frame: pd.DataFrame,
    *,
    student: str,
    seed: int,
    hide_fraction: float,
    min_history_rows: int,
    min_visible_concepts: int,
    min_hidden_concepts: int,
) -> PseudoHoldout | None:
    """Hide concepts, deleting every interaction touching a hidden concept."""
    row_concepts = [_concepts(value) for value in student_frame["cpt_seq"]]
    all_concepts = sorted({concept for values in row_concepts for concept in values})
    if len(student_frame) < min_history_rows or len(all_concepts) <= min_visible_concepts:
        return None
    desired = min(
        max(min_hidden_concepts, round(hide_fraction * len(all_concepts))),
        len(all_concepts) - min_visible_concepts,
    )
    ordered = sorted(
        all_concepts,
        key=lambda concept: stable_fraction(
            seed, "local-completion-concept", student, concept
        ),
    )
    hidden: set[str] = set()
    for concept in ordered:
        candidate = hidden | {concept}
        visible_mask = [not (candidate & set(values)) for values in row_concepts]
        visible = student_frame.loc[visible_mask]
        visible_concepts = {
            token for value in visible["cpt_seq"] for token in _concepts(value)
        }
        if (
            len(visible) >= min_history_rows
            and len(visible_concepts) >= min_visible_concepts
        ):
            hidden = candidate
        if len(hidden) >= desired:
            break
    if len(hidden) < min_hidden_concepts:
        return None
    visible_mask = [not (hidden & set(values)) for values in row_concepts]
    visible = student_frame.loc[visible_mask].copy().reset_index(drop=True)
    if any(hidden & set(_concepts(value)) for value in visible["cpt_seq"]):
        raise RuntimeError(f"Pseudo-holdout leakage for student {student}.")
    correct = {concept: 0 for concept in hidden}
    attempts = {concept: 0 for concept in hidden}
    for label, values in zip(student_frame["label"], row_concepts, strict=True):
        for concept in hidden.intersection(values):
            correct[concept] += int(label)
            attempts[concept] += 1
    if any(value == 0 for value in attempts.values()):
        raise RuntimeError(f"Hidden concept without a target for student {student}.")
    return PseudoHoldout(
        student=str(student),
        hidden_concepts=tuple(sorted(hidden)),
        visible_rows=visible,
        target_correct=correct,
        target_attempts=attempts,
        original_rows=len(student_frame),
        removed_rows=len(student_frame) - len(visible),
    )


def _parse_specs(values: list[str]) -> list[tuple[str, Path]]:
    output: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=DIR, got {value!r}.")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not name or not path.is_dir():
            raise ValueError(f"Invalid dataset specification: {value!r}.")
        output.append((name, path))
    return output

def _load_q_union(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Union long-form and comma-form Q rows into one row per exercise."""
    q_matrix, _ = derive_q_matrix(pd.read_csv(path))
    q_lookup = dict(
        q_matrix[["exer_id", "cpt_seq"]].itertuples(index=False, name=None)
    )
    return q_matrix, q_lookup


def _attach_q_union(
    frame: pd.DataFrame,
    q_lookup: dict[str, str],
) -> pd.DataFrame:
    output = frame.copy()
    mapped = output["exer_id"].astype(str).map(q_lookup)
    if mapped.isna().any():
        missing = output.loc[mapped.isna(), "exer_id"].astype(str).unique()[:10]
        raise ValueError(f"Exercises absent from source Q-matrix: {missing.tolist()}")
    output["cpt_seq"] = mapped
    return output



def _load_training_pool(
    dataset_dir: Path, *, split_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    q_matrix, q_lookup = _load_q_union(dataset_dir / "Q_matrix.csv")
    manifest_path = dataset_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    is_protocol = (dataset_dir / "valid_support.csv").exists() or (
        manifest is not None
        and manifest.get("protocol") == "student_disjoint_support_query"
    )
    if is_protocol:
        train, duplicates = canonicalize_interactions(
            pd.read_csv(dataset_dir / "train.csv")
        )
        train = _attach_q_union(train, q_lookup)
        return train, q_matrix, {
            "mode": "student_disjoint_train",
            "directory": str(dataset_dir.resolve()),
            "train_sha256": sha256_file(dataset_dir / "train.csv"),
            "q_sha256": sha256_file(dataset_dir / "Q_matrix.csv"),
            "duplicates_removed": duplicates,
        }
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        frames[split], _ = canonicalize_interactions(
            pd.read_csv(dataset_dir / f"{split}.csv")
        )
        frames[split] = _attach_q_union(frames[split], q_lookup)
        hashes[f"{split}.csv"] = sha256_file(dataset_dir / f"{split}.csv")
    splits, split_audit = assign_student_disjoint_support_query(
        train_frame=frames["train"],
        valid_frame=frames["valid"],
        test_frame=frames["test"],
        q_matrix=q_matrix,
        seed=split_seed,
    )
    hashes["Q_matrix.csv"] = sha256_file(dataset_dir / "Q_matrix.csv")
    return splits["train"], q_matrix, {
        "mode": "derived_in_memory_from_standard",
        "directory": str(dataset_dir.resolve()),
        "files": hashes,
        "student_disjoint_audit": split_audit,
    }


def _aggregate(
    frame: pd.DataFrame, concept_index: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    correct = np.zeros(len(concept_index), dtype=np.float64)
    attempts = np.zeros(len(concept_index), dtype=np.float64)
    for row in frame.itertuples(index=False):
        for concept in _concepts(row.cpt_seq):
            index = concept_index.get(concept)
            if index is not None:
                correct[index] += int(row.label)
                attempts[index] += 1
    return correct, attempts


def _logit(value: float) -> float:
    value = float(np.clip(value, 1e-5, 1.0 - 1e-5))
    return float(np.log(value / (1.0 - value)))


def _features(
    holdout: PseudoHoldout,
    *,
    target_index: int,
    concept_index: dict[str, int],
    population_correct: np.ndarray,
    population_attempts: np.ndarray,
    global_rate: float,
    prior_strength: float,
    subtract_correct: np.ndarray | None,
    subtract_attempts: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    prior_correct = population_correct.copy()
    prior_attempts = population_attempts.copy()
    if subtract_correct is not None and subtract_attempts is not None:
        prior_correct -= subtract_correct
        prior_attempts -= subtract_attempts
    priors = (prior_correct + prior_strength * global_rate) / (
        prior_attempts + prior_strength
    )
    visible_correct, visible_attempts = _aggregate(
        holdout.visible_rows, concept_index
    )
    observed = (visible_attempts > 0).astype(np.float64)
    smoothed = (visible_correct + prior_strength * priors) / (
        visible_attempts + prior_strength
    )
    confidence = np.sqrt(visible_attempts / (visible_attempts + prior_strength))
    residual = (smoothed - priors) * confidence
    labels = holdout.visible_rows["label"].to_numpy(dtype=np.float64)
    ability = (labels.sum() + prior_strength * global_rate) / (
        len(labels) + prior_strength
    )
    target_prior = float(priors[target_index])
    common = np.asarray(
        [_logit(ability), _logit(target_prior), np.log1p(len(labels)), observed.mean()]
    )
    target_one_hot = np.zeros(len(concept_index), dtype=np.float64)
    target_one_hot[target_index] = 1.0
    ability_feature = np.concatenate((common, target_one_hot))
    cross_feature = np.concatenate((common, residual, observed, target_one_hot))
    return ability_feature, cross_feature, target_prior


def _metrics(
    prediction: np.ndarray, correct: np.ndarray, attempts: np.ndarray
) -> dict[str, float | int | None]:
    prediction = np.clip(np.asarray(prediction, dtype=np.float64), 0.0, 1.0)
    correct = np.asarray(correct, dtype=np.float64)
    attempts = np.asarray(attempts, dtype=np.float64)
    mean = correct / attempts
    labels = np.concatenate((np.ones(len(prediction)), np.zeros(len(prediction))))
    scores = np.concatenate((prediction, prediction))
    weights = np.concatenate((correct, attempts - correct))
    active = weights > 0
    auc = (
        float(roc_auc_score(labels[active], scores[active], sample_weight=weights[active]))
        if correct.sum() > 0 and (attempts - correct).sum() > 0
        else None
    )
    hard = (mean >= 0.5).astype(int)
    return {
        "auc": auc,
        "response_weighted_brier": float(
            np.sum(correct * (1 - prediction) ** 2 + (attempts - correct) * prediction**2)
            / attempts.sum()
        ),
        "cell_brier": float(np.mean((prediction - mean) ** 2)),
        "hard_cell_auc": (
            float(roc_auc_score(hard, prediction))
            if len(np.unique(hard)) == 2
            else None
        ),
        "cells": len(prediction),
        "response_weight": int(attempts.sum()),
    }

def _targetwise_ridge_predictions(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    fit_target: np.ndarray,
    fit_weight: np.ndarray,
    x_eval: np.ndarray,
    eval_target: np.ndarray,
    fallback: np.ndarray,
    *,
    alpha: float,
    seed: int,
    min_samples: int = 20,
) -> tuple[np.ndarray, int]:
    """Fit one cross-concept ridge per target concept."""
    prediction = np.asarray(fallback, dtype=np.float64).copy()
    trained = 0
    for target in np.unique(eval_target):
        fit_mask = fit_target == target
        eval_mask = eval_target == target
        if fit_mask.sum() < min_samples:
            continue
        if np.ptp(y_fit[fit_mask]) <= 1e-8:
            continue
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=alpha, solver="lsqr", random_state=seed),
        )
        model.fit(
            x_fit[fit_mask],
            y_fit[fit_mask],
            ridge__sample_weight=fit_weight[fit_mask],
        )
        prediction[eval_mask] = model.predict(x_eval[eval_mask])
        trained += 1
    return prediction, trained



def audit_dataset(
    name: str,
    dataset_dir: Path,
    *,
    split_seed: int,
    eval_fraction: float,
    hide_fraction: float,
    min_history_rows: int,
    min_visible_concepts: int,
    min_hidden_concepts: int,
    prior_strength: float,
    ridge_alpha: float,
    extra_trees: int,
    extra_trees_jobs: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train, q_matrix, source = _load_training_pool(dataset_dir, split_seed=split_seed)
    concepts = sorted(
        {
            concept
            for value in q_matrix["cpt_seq"]
            for concept in _concepts(value)
        }
    )
    concept_index = {concept: index for index, concept in enumerate(concepts)}
    grouped = {
        str(student): frame.copy()
        for student, frame in train.groupby(train["stu_id"].astype(str), sort=False)
    }
    fit_students = {
        student
        for student in grouped
        if stable_fraction(split_seed, "local-completion-student", student)
        >= eval_fraction
    }
    eval_students = set(grouped) - fit_students
    if fit_students & eval_students:
        raise RuntimeError("Fit/eval student leakage.")
    population_correct = np.zeros(len(concepts))
    population_attempts = np.zeros(len(concepts))
    fit_totals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for student in fit_students:
        fit_totals[student] = _aggregate(grouped[student], concept_index)
        population_correct += fit_totals[student][0]
        population_attempts += fit_totals[student][1]
    fit_rows = train.loc[train["stu_id"].astype(str).isin(fit_students)]
    global_rate = float(fit_rows["label"].mean())

    features = {role: {"ability": [], "cross": []} for role in ("fit", "eval")}
    targets = {
        role: {"mean": [], "correct": [], "attempts": [], "prior": [], "index": []}
        for role in ("fit", "eval")
    }
    records: list[dict[str, Any]] = []
    retained = {"fit": 0, "eval": 0}
    skipped = {"fit": 0, "eval": 0}
    removed = {"fit": 0, "eval": 0}
    original = {"fit": 0, "eval": 0}
    residual_rows = 0
    for student, student_frame in grouped.items():
        role = "fit" if student in fit_students else "eval"
        holdout = build_pseudo_holdout(
            student_frame,
            student=student,
            seed=split_seed,
            hide_fraction=hide_fraction,
            min_history_rows=min_history_rows,
            min_visible_concepts=min_visible_concepts,
            min_hidden_concepts=min_hidden_concepts,
        )
        if holdout is None:
            skipped[role] += 1
            continue
        retained[role] += 1
        removed[role] += holdout.removed_rows
        original[role] += holdout.original_rows
        hidden = set(holdout.hidden_concepts)
        residual_rows += sum(
            bool(hidden & set(_concepts(value)))
            for value in holdout.visible_rows["cpt_seq"]
        )
        subtract = fit_totals.get(student)
        for concept in holdout.hidden_concepts:
            index = concept_index[concept]
            ability, cross, target_prior = _features(
                holdout,
                target_index=index,
                concept_index=concept_index,
                population_correct=population_correct,
                population_attempts=population_attempts,
                global_rate=global_rate,
                prior_strength=prior_strength,
                subtract_correct=subtract[0] if subtract else None,
                subtract_attempts=subtract[1] if subtract else None,
            )
            correct = holdout.target_correct[concept]
            attempts = holdout.target_attempts[concept]
            features[role]["ability"].append(ability)
            features[role]["cross"].append(cross)
            targets[role]["mean"].append(correct / attempts)
            targets[role]["correct"].append(correct)
            targets[role]["attempts"].append(attempts)
            targets[role]["prior"].append(target_prior)
            targets[role]["index"].append(index)
            records.append(
                {
                    "dataset": name,
                    "role": role,
                    "student": student,
                    "concept": concept,
                    "correct": correct,
                    "attempts": attempts,
                    "target_mean": correct / attempts,
                    "visible_rows": len(holdout.visible_rows),
                    "removed_rows": holdout.removed_rows,
                    "population_prior": target_prior,
                }
            )
    if residual_rows:
        raise RuntimeError(f"{residual_rows} visible rows retain hidden concepts.")
    if not features["fit"]["ability"] or not features["eval"]["ability"]:
        raise RuntimeError("No fit/eval pseudo-held cells.")

    x_fit_a = np.stack(features["fit"]["ability"])
    x_eval_a = np.stack(features["eval"]["ability"])
    x_fit_c = np.stack(features["fit"]["cross"])
    x_eval_c = np.stack(features["eval"]["cross"])
    y_fit = np.asarray(targets["fit"]["mean"])
    weight_fit = np.asarray(targets["fit"]["attempts"])
    ability_model = make_pipeline(
        StandardScaler(), Ridge(alpha=ridge_alpha, random_state=split_seed)
    )
    ability_model.fit(x_fit_a, y_fit, ridge__sample_weight=weight_fit)
    cross_ridge = make_pipeline(
        StandardScaler(), Ridge(alpha=ridge_alpha, random_state=split_seed)
    )
    cross_ridge.fit(x_fit_c, y_fit, ridge__sample_weight=weight_fit)
    predictions = {
        "population_concept_prior": np.asarray(targets["eval"]["prior"]),
        "ability_concept_ridge": ability_model.predict(x_eval_a),
        "cross_concept_ridge": cross_ridge.predict(x_eval_c),
    }
    if extra_trees > 0:
        forest = ExtraTreesRegressor(
            n_estimators=extra_trees,
            max_depth=18,
            min_samples_leaf=10,
            max_features="sqrt",
            n_jobs=extra_trees_jobs,
            random_state=split_seed,
        )
        forest.fit(x_fit_c, y_fit, sample_weight=weight_fit)
        predictions["cross_concept_extra_trees"] = forest.predict(x_eval_c)
    fit_target = np.asarray(targets["fit"]["index"], dtype=np.int64)
    eval_target = np.asarray(targets["eval"]["index"], dtype=np.int64)
    targetwise_cross, targetwise_models = _targetwise_ridge_predictions(
        x_fit_c[:, : 4 + 2 * len(concepts)],
        y_fit,
        fit_target,
        weight_fit,
        x_eval_c[:, : 4 + 2 * len(concepts)],
        eval_target,
        predictions["ability_concept_ridge"],
        alpha=ridge_alpha,
        seed=split_seed,
    )
    predictions["cross_concept_targetwise_ridge"] = targetwise_cross
    correct_eval = np.asarray(targets["eval"]["correct"])
    attempts_eval = np.asarray(targets["eval"]["attempts"])
    metrics = {
        model: _metrics(values, correct_eval, attempts_eval)
        for model, values in predictions.items()
    }
    base = metrics["population_concept_prior"]
    control_names = ("population_concept_prior", "ability_concept_ridge")
    stronger_control_name = max(
        control_names, key=lambda model: metrics[model]["auc"]
    )
    stronger_control = metrics[stronger_control_name]
    for values in metrics.values():
        values["delta_auc_vs_population"] = (
            values["auc"] - base["auc"]
            if values["auc"] is not None and base["auc"] is not None
            else None
        )
        values["delta_brier_vs_population"] = (
            values["response_weighted_brier"] - base["response_weighted_brier"]
        )
        values["delta_auc_vs_stronger_control"] = (
            values["auc"] - stronger_control["auc"]
            if values["auc"] is not None and stronger_control["auc"] is not None
            else None
        )
        values["delta_brier_vs_stronger_control"] = (
            values["response_weighted_brier"]
            - stronger_control["response_weighted_brier"]
        )
    best_name = max(
        (model for model in metrics if model.startswith("cross_concept_")),
        key=lambda model: metrics[model]["auc"],
    )
    summary = {
        "dataset": name,
        "source": source,
        "protocol": {
            "split_seed": split_seed,
            "fit_eval_assignment": "stable SHA-256 hash by student",
            "eval_fraction": eval_fraction,
            "hide_fraction": hide_fraction,
            "fit_eval_student_overlap": 0,
            "hidden_concept_residual_rows": residual_rows,
            "multi_concept_deletion": (
                "Concept membership is the per-exercise union of all source "
                "Q-matrix rows. "
                "Delete every interaction touching any hidden concept, then "
                "recompute all visible features."
            ),
        },
        "population": {
            "students": len(grouped),
            "fit_students_assigned": len(fit_students),
            "eval_students_assigned": len(eval_students),
            "fit_students_retained": retained["fit"],
            "eval_students_retained": retained["eval"],
            "skipped_students": skipped,
            "concepts": len(concepts),
            "targetwise_cross_models_trained": targetwise_models,
            "multi_concept_exercises": int(
                q_matrix["cpt_seq"].astype(str).str.contains(",").sum()
            ),
            "fit_cells": len(targets["fit"]["mean"]),
            "eval_cells": len(targets["eval"]["mean"]),
            "fit_removed_row_fraction": removed["fit"] / original["fit"],
            "eval_removed_row_fraction": removed["eval"] / original["eval"],
        },
        "models": metrics,
        "stronger_control_name": stronger_control_name,
        "best_completion_model": best_name,
        "best_completion_delta_auc": metrics[best_name][
            "delta_auc_vs_stronger_control"
        ],
        "best_completion_delta_brier": metrics[best_name][
            "delta_brier_vs_stronger_control"
        ],
    }
    cells = pd.DataFrame(records)
    eval_index = cells.index[cells["role"] == "eval"]
    for model, values in predictions.items():
        cells.loc[eval_index, f"prediction_{model}"] = np.clip(values, 0, 1)
    return summary, cells
def passes_opportunity_gate(
    auc_deltas: list[float],
    brier_deltas: list[float],
) -> tuple[bool, bool, bool]:
    auc_passed = len(auc_deltas) >= 2 and (
        all(delta >= 0.005 for delta in auc_deltas)
        or (max(auc_deltas) >= 0.010 and min(auc_deltas) >= 0.0)
    )
    brier_passed = len(brier_deltas) >= 2 and all(
        delta <= 1e-4 for delta in brier_deltas
    )
    return auc_passed and brier_passed, auc_passed, brier_passed




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit train-only student-local completion signal on pseudo-exact-zero "
            "concept cells."
        )
    )
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-seed", type=int, default=2024)
    parser.add_argument("--eval-fraction", type=float, default=0.25)
    parser.add_argument("--hide-fraction", type=float, default=0.30)
    parser.add_argument("--min-history-rows", type=int, default=10)
    parser.add_argument("--min-visible-concepts", type=int, default=2)
    parser.add_argument("--min-hidden-concepts", type=int, default=1)
    parser.add_argument("--prior-strength", type=float, default=5.0)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--extra-trees", type=int, default=160)
    parser.add_argument("--extra-trees-jobs", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split_seed != 2024:
        raise ValueError("This audit fixes split_seed=2024.")
    if not 0.0 < args.eval_fraction < 1.0:
        raise ValueError("--eval-fraction must be in (0, 1).")
    if not 0.0 < args.hide_fraction < 1.0:
        raise ValueError("--hide-fraction must be in (0, 1).")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    table_rows = []
    for name, path in _parse_specs(args.dataset):
        summary, cells = audit_dataset(
            name,
            path,
            split_seed=args.split_seed,
            eval_fraction=args.eval_fraction,
            hide_fraction=args.hide_fraction,
            min_history_rows=args.min_history_rows,
            min_visible_concepts=args.min_visible_concepts,
            min_hidden_concepts=args.min_hidden_concepts,
            prior_strength=args.prior_strength,
            ridge_alpha=args.ridge_alpha,
            extra_trees=args.extra_trees,
            extra_trees_jobs=args.extra_trees_jobs,
        )
        summaries.append(summary)
        cells.to_csv(output_dir / f"{name}_masked_cells.csv", index=False)
        (output_dir / f"{name}_audit.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        for model, metrics in summary["models"].items():
            table_rows.append({"dataset": name, "model": model, **metrics})
    deltas = [float(summary["best_completion_delta_auc"]) for summary in summaries]
    brier_deltas = [
        float(summary["best_completion_delta_brier"]) for summary in summaries
    ]
    passed, auc_passed, brier_passed = passes_opportunity_gate(
        deltas, brier_deltas
    )
    aggregate = {
        "schema_version": 1,
        "audit": "student_local_completion_opportunity",
        "datasets": summaries,
        "gate": {
            "rule": (
                "Relative to the stronger of population prior and ability+concept "
                "Ridge: cross-concept delta AUC >= .005 on both datasets, or "
                ">= .010 on one and non-negative on the other; response-weighted "
                "Brier may regress by at most 1e-4 on every dataset."
            ),
            "passed": passed,
            "auc_passed": auc_passed,
            "brier_passed": brier_passed,
            "dataset_deltas": {
                summary["dataset"]: summary["best_completion_delta_auc"]
                for summary in summaries
            },
            "dataset_brier_deltas": {
                summary["dataset"]: summary["best_completion_delta_brier"]
                for summary in summaries
            },
            "stronger_controls": {
                summary["dataset"]: summary["stronger_control_name"]
                for summary in summaries
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
