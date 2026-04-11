from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import apply_dataset_defaults
from data import prepare_experiment_split_bundles
from data.q_matrix import normalize_concept_sequence
from models import DecoupledCDM
from trainers.engine import _bundle_tensors, _validate_history_visibility
from utils import compute_metrics, resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze prediction metrics by interpretable test slices.")
    parser.add_argument("--summary", required=True, help="Training summary JSON containing model config and best checkpoint path.")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path. Defaults to summary['best_checkpoint_path'].")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--train-interactions", default=None)
    parser.add_argument("--valid-interactions", default=None)
    parser.add_argument("--test-interactions", default=None)
    parser.add_argument("--q-matrix", default=None)
    parser.add_argument("--concept-graph", default=None)
    parser.add_argument("--graph-mode", choices=["single", "dual"], default="single")
    parser.add_argument("--concept-dim", type=int, default=None)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--min-count", type=int, default=100)
    parser.add_argument("--output", default="results/prediction_slice_report.json")
    parser.add_argument("--csv-output", default=None)
    args = parser.parse_args()
    return apply_dataset_defaults(args, parser)


def derive_q_matrix_from_splits_if_needed(
    train_path: str,
    valid_path: str,
    test_path: str,
    q_matrix_path: str | None,
) -> str:
    if q_matrix_path is not None:
        return q_matrix_path
    frames = [
        pd.read_csv(train_path, usecols=["exer_id", "cpt_seq"]),
        pd.read_csv(valid_path, usecols=["exer_id", "cpt_seq"]),
        pd.read_csv(test_path, usecols=["exer_id", "cpt_seq"]),
    ]
    q_df = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    output_path = Path("results") / "derived_q_matrix_slice_eval.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    q_df.to_csv(output_path, index=False)
    return str(output_path)


def load_model(
    *,
    summary: dict[str, Any],
    checkpoint_path: str,
    bundles: dict[str, Any],
    concept_dim: int,
    device: str,
) -> DecoupledCDM:
    train_bundle = bundles["train"]
    model = DecoupledCDM(
        num_students=train_bundle.num_students,
        num_exercises=train_bundle.num_exercises,
        num_concepts=train_bundle.num_concepts,
        concept_dim=concept_dim,
        graph_mode=str(summary.get("graph_mode", "single")),
        student_gate_prior_alpha=float(summary.get("student_gate_prior_alpha", 1.0)),
        student_gate_prior_beta=float(summary.get("student_gate_prior_beta", 1.0)),
        gs_mode=str(summary.get("gs_mode", "conditional")),
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(torch.device(device))
    model.eval()
    return model


def predict_bundle(*, bundle: Any, model: DecoupledCDM, device: str) -> pd.DataFrame:
    _validate_history_visibility(bundle)
    torch_device = torch.device(device)
    tensors = _bundle_tensors(bundle, torch_device)
    with torch.no_grad():
        output = model(
            q_matrix=tensors["q_matrix"],
            concept_graph=tensors["concept_graph"],
            prerequisite_graph=tensors["prerequisite_graph"],
            similarity_graph=tensors["similarity_graph"],
            student_exercise_mask=tensors["student_exercise_mask"],
            response_matrix=tensors["response_matrix"],
            student_tkc_mask=tensors["student_tkc_mask"],
            student_ukc_mask=tensors["student_ukc_mask"],
            target_student_ids=tensors["interaction_student_ids"],
            target_exercise_ids=tensors["interaction_exercise_ids"],
        )

    frame = bundle.interactions.reset_index(drop=True).copy()
    frame["label"] = tensors["interaction_labels"].detach().cpu().numpy()
    frame["prob"] = output.probs.detach().cpu().numpy()
    frame["pred"] = (frame["prob"] >= 0.5).astype(int)
    frame["abs_error"] = (frame["label"] - frame["prob"]).abs()
    frame["squared_error"] = (frame["label"] - frame["prob"]) ** 2
    frame["confidence"] = frame["prob"].where(frame["prob"] >= 0.5, 1.0 - frame["prob"])
    return frame


def _format_count_bin(value: float) -> str:
    if value <= 0:
        return "0"
    if value <= 5:
        return "1-5"
    if value <= 20:
        return "6-20"
    if value <= 50:
        return "21-50"
    if value <= 100:
        return "51-100"
    return "101+"


def _format_rate_bin(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "missing"
    if value < 0.4:
        return "<0.4"
    if value < 0.6:
        return "0.4-0.6"
    if value < 0.8:
        return "0.6-0.8"
    return "0.8-1.0"


def _format_concept_count(value: int) -> str:
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    if value == 3:
        return "3"
    return "4+"


def _format_overlap(row: pd.Series) -> str:
    if row["concept_count"] == 0:
        return "no_concepts"
    if row["seen_concept_count"] == 0:
        return "none_seen"
    if row["seen_concept_count"] == row["concept_count"]:
        return "all_seen"
    return "partial_seen"


def add_slice_features(frame: pd.DataFrame, train_frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()

    student_count = train_frame.groupby("stu_id").size()
    student_acc = train_frame.groupby("stu_id")["label"].mean()
    exercise_count = train_frame.groupby("exer_id").size()
    exercise_acc = train_frame.groupby("exer_id")["label"].mean()

    student_concepts: dict[Any, set[str]] = {}
    for row in train_frame.itertuples(index=False):
        student_concepts.setdefault(row.stu_id, set()).update(normalize_concept_sequence(row.cpt_seq))

    enriched["concepts"] = enriched["cpt_seq"].map(normalize_concept_sequence)
    enriched["concept_count"] = enriched["concepts"].map(len)
    enriched["student_history_count"] = enriched["stu_id"].map(student_count).fillna(0).astype(int)
    enriched["student_history_acc"] = enriched["stu_id"].map(student_acc)
    enriched["exercise_train_count"] = enriched["exer_id"].map(exercise_count).fillna(0).astype(int)
    enriched["exercise_train_acc"] = enriched["exer_id"].map(exercise_acc)
    enriched["seen_concept_count"] = [
        len(set(concepts) & student_concepts.get(student_id, set()))
        for student_id, concepts in zip(enriched["stu_id"], enriched["concepts"], strict=True)
    ]

    enriched["student_history_count_bin"] = enriched["student_history_count"].map(_format_count_bin)
    enriched["student_history_acc_bin"] = enriched["student_history_acc"].map(_format_rate_bin)
    enriched["exercise_train_count_bin"] = enriched["exercise_train_count"].map(_format_count_bin)
    enriched["exercise_train_acc_bin"] = enriched["exercise_train_acc"].map(_format_rate_bin)
    enriched["concept_count_bin"] = enriched["concept_count"].map(_format_concept_count)
    enriched["student_item_concept_overlap"] = enriched.apply(_format_overlap, axis=1)
    enriched["prob_bin"] = pd.cut(
        enriched["prob"],
        bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        include_lowest=True,
    ).astype(str)
    enriched["confidence_bin"] = pd.cut(
        enriched["confidence"],
        bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        include_lowest=True,
    ).astype(str)
    return enriched.drop(columns=["concepts"])


def slice_metrics(frame: pd.DataFrame, *, group_col: str, min_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(group_col, dropna=False):
        count = len(group)
        if count < min_count:
            continue
        metrics = compute_metrics(group["label"].to_numpy(), group["prob"].to_numpy())
        rows.append(
            {
                "slice": group_col,
                "value": str(value),
                "count": count,
                "label_rate": float(group["label"].mean()),
                "mean_prob": float(group["prob"].mean()),
                "mean_abs_error": float(group["abs_error"].mean()),
                "auc": metrics["auc"],
                "acc": metrics["acc"],
                "rmse": metrics["rmse"],
                "brier": metrics["brier"],
                "ece": metrics["ece"],
            }
        )
    return sorted(rows, key=lambda row: (row["slice"], row["value"]))


def main() -> None:
    args = parse_args()
    with Path(args.summary).open() as handle:
        summary = json.load(handle)
    if args.concept_dim is None:
        args.concept_dim = summary.get("concept_dim")
    if args.concept_dim is None:
        raise ValueError("Provide --concept-dim when the summary JSON does not include concept_dim.")

    if not all([args.train_interactions, args.valid_interactions, args.test_interactions]):
        raise ValueError("Slice analysis requires train/valid/test interaction paths.")

    q_matrix_path = derive_q_matrix_from_splits_if_needed(
        args.train_interactions,
        args.valid_interactions,
        args.test_interactions,
        args.q_matrix,
    )
    bundles = prepare_experiment_split_bundles(
        train_interactions_path=args.train_interactions,
        valid_interactions_path=args.valid_interactions,
        test_interactions_path=args.test_interactions,
        q_matrix_path=q_matrix_path,
        concept_graph_path=args.concept_graph,
    )
    checkpoint_path = args.checkpoint or summary["best_checkpoint_path"]
    device = str(resolve_device(args.device, args.gpus))
    model = load_model(
        summary=summary,
        checkpoint_path=checkpoint_path,
        bundles=bundles,
        concept_dim=args.concept_dim,
        device=device,
    )
    prediction_frame = predict_bundle(bundle=bundles[args.split], model=model, device=device)
    enriched = add_slice_features(prediction_frame, bundles["train"].interactions)

    slice_columns = [
        "student_history_count_bin",
        "student_history_acc_bin",
        "exercise_train_count_bin",
        "exercise_train_acc_bin",
        "concept_count_bin",
        "student_item_concept_overlap",
        "prob_bin",
        "confidence_bin",
    ]
    slices = [
        row
        for column in slice_columns
        for row in slice_metrics(enriched, group_col=column, min_count=args.min_count)
    ]
    overall = compute_metrics(enriched["label"].to_numpy(), enriched["prob"].to_numpy())
    payload = {
        "summary": str(Path(args.summary)),
        "checkpoint": checkpoint_path,
        "split": args.split,
        "device": device,
        "count": int(len(enriched)),
        "overall": overall,
        "slices": slices,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(payload, output_path)

    csv_output = Path(args.csv_output) if args.csv_output else output_path.with_suffix(".csv")
    pd.DataFrame(slices).to_csv(csv_output, index=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
