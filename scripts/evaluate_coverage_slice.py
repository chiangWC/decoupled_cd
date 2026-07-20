from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import prepare_experiment_split_bundles
from data.q_matrix import normalize_concept_sequence
from scripts.analyze_prediction_slices import derive_q_matrix_from_splits_if_needed, load_model
from scripts.evaluate_history_hiding_stress import (
    infer_model_name,
    load_summary,
    normalize_summary_for_current_loader,
    predict_bundle,
)
from utils import compute_metrics, resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint metrics by target train-history concept coverage slices."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--summary", action="append", required=True, help="Training summary JSON. Repeat per model.")
    parser.add_argument(
        "--model-name",
        action="append",
        default=None,
        help="Display name for the matching --summary. Repeat the same number of times as --summary.",
    )
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--train-interactions", default=None, help="Override train split path for every summary.")
    parser.add_argument("--valid-interactions", default=None, help="Override valid split path for every summary.")
    parser.add_argument("--test-interactions", default=None, help="Override test split path for every summary.")
    parser.add_argument(
        "--valid-history-interactions",
        default=None,
        help="Override validation-student support history.",
    )
    parser.add_argument(
        "--test-history-interactions",
        default=None,
        help="Override test-student support history.",
    )
    parser.add_argument("--q-matrix", default=None, help="Override Q-matrix path for every summary.")
    parser.add_argument("--concept-graph", default=None, help="Override concept graph path for every summary.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--slice-csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    args = parser.parse_args()
    if args.model_name is not None and len(args.model_name) != len(args.summary):
        raise ValueError("--model-name must be repeated the same number of times as --summary.")
    return args


def resolve_split_paths(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, str | None]:
    train_path = args.train_interactions or summary.get("train_interactions")
    valid_path = args.valid_interactions or summary.get("valid_interactions")
    test_path = args.test_interactions or summary.get("test_interactions")
    valid_history_path = (
        args.valid_history_interactions
        or summary.get("valid_history_interactions")
    )
    test_history_path = (
        args.test_history_interactions
        or summary.get("test_history_interactions")
    )
    if not all([train_path, valid_path, test_path]):
        raise ValueError("Coverage slice evaluation requires train/valid/test interaction paths.")

    q_matrix_path = args.q_matrix or summary.get("q_matrix")
    if q_matrix_path is not None and not Path(str(q_matrix_path)).exists():
        q_matrix_path = None
    q_matrix_path = derive_q_matrix_from_splits_if_needed(
        str(train_path),
        str(valid_path),
        str(test_path),
        None if q_matrix_path is None else str(q_matrix_path),
        valid_history_path=(
            None if valid_history_path is None else str(valid_history_path)
        ),
        test_history_path=(
            None if test_history_path is None else str(test_history_path)
        ),
    )

    return {
        "train_interactions": str(train_path),
        "valid_interactions": str(valid_path),
        "test_interactions": str(test_path),
        "valid_history_interactions": (
            None if valid_history_path is None else str(valid_history_path)
        ),
        "test_history_interactions": (
            None if test_history_path is None else str(test_history_path)
        ),
        "q_matrix": str(q_matrix_path),
        "concept_graph": args.concept_graph or summary.get("concept_graph"),
        "prerequisite_graph": summary.get("prerequisite_graph"),
        "similarity_graph": summary.get("similarity_graph"),
    }


def prepare_bundles_from_paths(paths: dict[str, str | None]) -> dict[str, Any]:
    return prepare_experiment_split_bundles(
        train_interactions_path=str(paths["train_interactions"]),
        valid_interactions_path=str(paths["valid_interactions"]),
        test_interactions_path=str(paths["test_interactions"]),
        valid_history_interactions_path=paths.get("valid_history_interactions"),
        test_history_interactions_path=paths.get("test_history_interactions"),
        q_matrix_path=str(paths["q_matrix"]),
        concept_graph_path=paths.get("concept_graph"),
        prerequisite_graph_path=paths.get("prerequisite_graph"),
        similarity_graph_path=paths.get("similarity_graph"),
    )


def build_student_seen_concepts(
    train_frame: pd.DataFrame,
    *,
    exercise_concepts: dict[str, set[str]] | None = None,
) -> dict[Any, set[str]]:
    seen: dict[Any, set[str]] = {}
    for row in train_frame.itertuples(index=False):
        concepts = (
            exercise_concepts.get(str(row.exer_id), set())
            if exercise_concepts is not None
            else set(normalize_concept_sequence(row.cpt_seq))
        )
        seen.setdefault(row.stu_id, set()).update(concepts)
    return seen


def build_exercise_concepts(q_matrix: pd.DataFrame) -> dict[str, set[str]]:
    exercise_concepts: dict[str, set[str]] = {}
    for row in q_matrix.itertuples(index=False):
        concepts = set(normalize_concept_sequence(row.cpt_seq))
        exercise_concepts.setdefault(str(row.exer_id), set()).update(concepts)
    return exercise_concepts


def coverage_bucket(coverage: float | None) -> str:
    if coverage is None:
        return "no_concepts"
    if coverage == 0.0:
        return "zero"
    if coverage < 0.5:
        return "low"
    if coverage < 1.0:
        return "partial"
    return "full"


def add_target_coverage(
    frame: pd.DataFrame,
    *,
    q_matrix: pd.DataFrame,
    train_frame: pd.DataFrame | None = None,
    student_history_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if student_history_frame is None:
        if train_frame is None:
            raise ValueError(
                "Provide student_history_frame (or legacy train_frame)."
            )
        student_history_frame = train_frame
    enriched = frame.copy()
    exercise_concepts = build_exercise_concepts(q_matrix)
    student_seen = build_student_seen_concepts(
        student_history_frame,
        exercise_concepts=exercise_concepts,
    )
    coverages: list[float | None] = []
    concept_counts: list[int] = []
    seen_counts: list[int] = []

    for row in enriched.itertuples(index=False):
        concepts = exercise_concepts.get(str(row.exer_id), set(normalize_concept_sequence(row.cpt_seq)))
        concept_count = len(concepts)
        seen_count = len(concepts & student_seen.get(row.stu_id, set()))
        concept_counts.append(concept_count)
        seen_counts.append(seen_count)
        coverages.append(None if concept_count == 0 else seen_count / concept_count)

    enriched["target_concept_count"] = concept_counts
    enriched["target_seen_concept_count"] = seen_counts
    enriched["target_coverage"] = coverages
    enriched["coverage_bucket"] = [coverage_bucket(value) for value in coverages]
    enriched["coverage_group"] = [
        "low_coverage" if value is not None and value < 0.5 else "full_coverage" if value == 1.0 else "other"
        for value in coverages
    ]
    return enriched


def metric_row(frame: pd.DataFrame, *, metric_scope: str, model_name: str, dataset_name: str) -> dict[str, Any]:
    metrics = compute_metrics(frame["label"].to_numpy(), frame["prob"].to_numpy())
    return {
        "dataset": dataset_name,
        "model": model_name,
        "scope": metric_scope,
        "count": int(len(frame)),
        "label_rate": float(frame["label"].mean()) if len(frame) else float("nan"),
        "mean_prob": float(frame["prob"].mean()) if len(frame) else float("nan"),
        "auc": float(metrics["auc"]),
        "acc": float(metrics["acc"]),
        "rmse": float(metrics["rmse"]),
        "brier": float(metrics["brier"]),
        "ece": float(metrics["ece"]),
    }


def compute_slice_rows(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    model_name: str,
) -> list[dict[str, Any]]:
    rows = [metric_row(frame, metric_scope="overall", model_name=model_name, dataset_name=dataset_name)]
    for bucket in ["zero", "low", "partial", "full", "no_concepts"]:
        group = frame[frame["coverage_bucket"] == bucket]
        if len(group):
            rows.append(metric_row(group, metric_scope=f"bucket:{bucket}", model_name=model_name, dataset_name=dataset_name))
    for group_name in ["low_coverage", "full_coverage"]:
        group = frame[frame["coverage_group"] == group_name]
        if len(group):
            rows.append(metric_row(group, metric_scope=group_name, model_name=model_name, dataset_name=dataset_name))
    return rows


def build_summary_row(slice_rows: list[dict[str, Any]], *, dataset_name: str, model_name: str) -> dict[str, Any]:
    by_scope = {row["scope"]: row for row in slice_rows}
    overall = by_scope.get("overall", {})
    low = by_scope.get("low_coverage", {})
    full = by_scope.get("full_coverage", {})
    return {
        "dataset": dataset_name,
        "model": model_name,
        "overall_auc": overall.get("auc"),
        "overall_brier": overall.get("brier"),
        "overall_ece": overall.get("ece"),
        "low_coverage_count": low.get("count", 0),
        "low_coverage_auc": low.get("auc"),
        "full_coverage_count": full.get("count", 0),
        "full_coverage_auc": full.get("auc"),
        "coverage_gap": (
            float(full["auc"]) - float(low["auc"]) if "auc" in full and "auc" in low else None
        ),
        "low_coverage_brier": low.get("brier"),
        "low_coverage_ece": low.get("ece"),
    }


def main() -> None:
    args = parse_args()
    device = str(resolve_device(args.device, args.gpus))
    model_names = args.model_name or [infer_model_name(path) for path in args.summary]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_path.with_name(f"{output_path.stem}_predictions")
    prediction_dir.mkdir(parents=True, exist_ok=True)

    all_slice_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    run_metadata: list[dict[str, Any]] = []
    for summary_path, model_name in zip(args.summary, model_names, strict=True):
        summary = normalize_summary_for_current_loader(load_summary(summary_path))
        paths = resolve_split_paths(summary, args)
        bundles = prepare_bundles_from_paths(paths)
        target_bundle = bundles[args.split]
        model = load_model(
            summary=summary,
            checkpoint_path=summary["best_checkpoint_path"],
            bundles=bundles,
            concept_dim=int(summary["concept_dim"]),
            device=device,
        )
        labels, probs, loss = predict_bundle(bundle=target_bundle, model=model, device=device)
        prediction_frame = target_bundle.interactions.reset_index(drop=True).copy()
        prediction_frame["label"] = labels.numpy()
        prediction_frame["prob"] = probs.numpy()
        enriched = add_target_coverage(
            prediction_frame,
            student_history_frame=target_bundle.history_interactions,
            q_matrix=bundles["shared"]["q_matrix"],
        )
        safe_model_name = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in model_name
        )
        prediction_path = prediction_dir / f"{safe_model_name}_{args.split}_predictions.csv"
        enriched.to_csv(prediction_path, index=False)
        slice_rows = compute_slice_rows(enriched, dataset_name=args.dataset_name, model_name=model_name)
        all_slice_rows.extend(slice_rows)
        summary_rows.append(build_summary_row(slice_rows, dataset_name=args.dataset_name, model_name=model_name))
        original_metrics = compute_metrics(labels.numpy(), probs.numpy())
        run_metadata.append(
            {
                "summary_path": summary_path,
                "model": model_name,
                "split_paths": paths,
                "loss": loss,
                "original_metrics": original_metrics,
                "prediction_path": str(prediction_path.resolve()),
                "coverage_bucket_counts": {
                    str(key): int(value) for key, value in enriched["coverage_bucket"].value_counts().items()
                },
            }
        )

    payload = {
        "dataset": args.dataset_name,
        "split": args.split,
        "device": device,
        "low_coverage_definition": "target_coverage < 0.5, including coverage == 0",
        "runs": run_metadata,
        "slices": all_slice_rows,
        "summary": summary_rows,
    }
    write_json(payload, output_path)

    slice_csv = Path(args.slice_csv) if args.slice_csv else output_path.with_name(f"{output_path.stem}_slices.csv")
    summary_csv = (
        Path(args.summary_csv) if args.summary_csv else output_path.with_name(f"{output_path.stem}_summary.csv")
    )
    pd.DataFrame(all_slice_rows).to_csv(slice_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
