from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import StepDataBundle, prepare_experiment_split_bundles
from data.pipeline import build_history_tensors
from models import DecoupledCDM, DecoupledCDMEnsemble
from scripts.analyze_prediction_slices import load_model
from trainers.engine import _bundle_tensors, _validate_history_visibility
from utils import compute_metrics, resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint robustness after randomly hiding student train-history interactions."
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
    parser.add_argument("--hide-ratios", default="0.2,0.4,0.6,0.8")
    parser.add_argument("--mask-seeds", default="11,13,17")
    parser.add_argument(
        "--history-mask-mode",
        choices=["interaction"],
        default="interaction",
        help="Currently supports student-side random interaction hiding.",
    )
    parser.add_argument(
        "--keep-exercise-evidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep exercise-level evidence built from the original train split while masking student-side history.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-run-csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    args = parser.parse_args()
    args.hide_ratios = _parse_float_list(args.hide_ratios, name="--hide-ratios")
    args.mask_seeds = _parse_int_list(args.mask_seeds, name="--mask-seeds")
    if args.model_name is not None and len(args.model_name) != len(args.summary):
        raise ValueError("--model-name must be repeated the same number of times as --summary.")
    for ratio in args.hide_ratios:
        if ratio <= 0.0 or ratio >= 1.0:
            raise ValueError("--hide-ratios values must be in (0, 1).")
    return args


def _parse_float_list(value: str, *, name: str) -> list[float]:
    try:
        parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of floats.") from exc
    if not parsed:
        raise ValueError(f"{name} must not be empty.")
    return parsed


def _parse_int_list(value: str, *, name: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of integers.") from exc
    if not parsed:
        raise ValueError(f"{name} must not be empty.")
    return parsed


def infer_model_name(summary_path: str) -> str:
    stem = Path(summary_path).stem
    replacements = {
        "baseline_reproduce": "Exp110 full",
        "exp110_full": "Exp110 full",
        "exp81_baseline": "Exp81 baseline",
        "a_no_cog_align": "Exp110 w/o cognitive alignment",
        "exp110_no_cog_align": "Exp110 w/o cognitive alignment",
        "a_no_dual_tower": "Exp110 w/o dual tower",
        "exp110_no_dual_tower": "Exp110 w/o dual tower",
    }
    for token, label in replacements.items():
        if token in stem:
            return label
    return stem


def load_summary(path: str) -> dict[str, Any]:
    with Path(path).open() as handle:
        return json.load(handle)


def prepare_bundles(summary: dict[str, Any]) -> dict[str, Any]:
    return prepare_experiment_split_bundles(
        train_interactions_path=summary["train_interactions"],
        valid_interactions_path=summary["valid_interactions"],
        test_interactions_path=summary["test_interactions"],
        q_matrix_path=summary["q_matrix"],
        concept_graph_path=summary.get("concept_graph"),
        prerequisite_graph_path=summary.get("prerequisite_graph"),
        similarity_graph_path=summary.get("similarity_graph"),
    )


def mask_train_history_interactions(train_frame: pd.DataFrame, *, hide_ratio: float, seed: int) -> pd.DataFrame:
    rng = torch.Generator()
    rng.manual_seed(int(seed))
    keep_chunks: list[pd.DataFrame] = []
    for _, group in train_frame.groupby("stu_id", sort=False):
        random_values = torch.rand(len(group), generator=rng).numpy()
        keep_mask = random_values >= hide_ratio
        if keep_mask.any():
            keep_chunks.append(group.loc[keep_mask])
    if not keep_chunks:
        return train_frame.iloc[0:0].copy()
    return pd.concat(keep_chunks, ignore_index=True)


def build_hidden_bundle(
    *,
    bundle: StepDataBundle,
    masked_history: pd.DataFrame,
    keep_exercise_evidence: bool,
) -> StepDataBundle:
    history_tensors = build_history_tensors(
        history_interactions=masked_history,
        student_id_map=bundle.student_id_map,
        exercise_id_map=bundle.exercise_id_map,
        concept_id_map=bundle.concept_id_map,
    )
    exercise_evidence = (
        bundle.exercise_evidence_tensor if keep_exercise_evidence else history_tensors["exercise_evidence_tensor"]
    )
    return replace(
        bundle,
        history_interactions=masked_history,
        student_exercise_mask=history_tensors["student_exercise_mask"],
        student_tkc_mask=history_tensors["student_tkc_mask"],
        student_ukc_mask=history_tensors["student_ukc_mask"],
        response_matrix_tensor=history_tensors["response_matrix_tensor"],
        student_concept_evidence_tensor=history_tensors["student_concept_evidence_tensor"],
        exercise_evidence_tensor=exercise_evidence,
    )


def predict_bundle(
    *,
    bundle: StepDataBundle,
    model: DecoupledCDM | DecoupledCDMEnsemble,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, float]:
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
            student_concept_evidence=tensors["student_concept_evidence"],
            exercise_evidence=tensors["exercise_evidence"],
            target_student_ids=tensors["interaction_student_ids"],
            target_exercise_ids=tensors["interaction_exercise_ids"],
            use_student_subset=True,
        )
        labels = tensors["interaction_labels"]
        loss = torch.nn.functional.binary_cross_entropy(output.probs, labels)
    return labels.detach().cpu(), output.probs.detach().cpu(), float(loss.item())


def evaluate_bundle(
    *,
    bundle: StepDataBundle,
    model: DecoupledCDM | DecoupledCDMEnsemble,
    device: str,
) -> dict[str, float]:
    labels, probs, loss = predict_bundle(bundle=bundle, model=model, device=device)
    metrics = compute_metrics(labels=labels.numpy(), probs=probs.numpy())
    metrics["loss"] = loss
    return {key: float(value) for key, value in metrics.items() if key != "calibration_bins"}


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame([row for row in rows if row["hide_ratio"] > 0.0])
    if frame.empty:
        return []
    metric_cols = ["hidden_auc", "delta_auc", "hidden_acc", "hidden_brier", "hidden_ece", "hidden_history_rows"]
    summary_rows: list[dict[str, Any]] = []
    for (dataset, model, hide_ratio), group in frame.groupby(["dataset", "model", "hide_ratio"], sort=False):
        row: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "hide_ratio": float(hide_ratio),
            "n": int(len(group)),
            "original_auc": float(group["original_auc"].iloc[0]),
        }
        for col in metric_cols:
            values = group[col].astype(float)
            row[f"mean_{col}"] = float(values.mean())
            row[f"std_{col}"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
        summary_rows.append(row)
    return summary_rows


def main() -> None:
    args = parse_args()
    device = str(resolve_device(args.device, args.gpus))
    summary_paths = list(args.summary)
    model_names = args.model_name or [infer_model_name(path) for path in summary_paths]

    rows: list[dict[str, Any]] = []
    for summary_path, model_name in zip(summary_paths, model_names, strict=True):
        summary = load_summary(summary_path)
        bundles = prepare_bundles(summary)
        target_bundle = bundles[args.split]
        model = load_model(
            summary=summary,
            checkpoint_path=summary["best_checkpoint_path"],
            bundles=bundles,
            concept_dim=int(summary["concept_dim"]),
            device=device,
        )
        original_metrics = evaluate_bundle(bundle=target_bundle, model=model, device=device)
        original_history_rows = int(len(bundles["train"].interactions))
        rows.append(
            {
                "dataset": args.dataset_name,
                "model": model_name,
                "summary_path": summary_path,
                "split": args.split,
                "hide_ratio": 0.0,
                "mask_seed": None,
                "original_auc": original_metrics["auc"],
                "hidden_auc": original_metrics["auc"],
                "delta_auc": 0.0,
                "hidden_acc": original_metrics["acc"],
                "hidden_brier": original_metrics["brier"],
                "hidden_ece": original_metrics["ece"],
                "hidden_loss": original_metrics["loss"],
                "original_history_rows": original_history_rows,
                "hidden_history_rows": original_history_rows,
                "hidden_history_ratio": 1.0,
            }
        )
        for hide_ratio in args.hide_ratios:
            for mask_seed in args.mask_seeds:
                masked_history = mask_train_history_interactions(
                    bundles["train"].interactions,
                    hide_ratio=hide_ratio,
                    seed=mask_seed,
                )
                hidden_bundle = build_hidden_bundle(
                    bundle=target_bundle,
                    masked_history=masked_history,
                    keep_exercise_evidence=args.keep_exercise_evidence,
                )
                hidden_metrics = evaluate_bundle(bundle=hidden_bundle, model=model, device=device)
                hidden_history_rows = int(len(masked_history))
                rows.append(
                    {
                        "dataset": args.dataset_name,
                        "model": model_name,
                        "summary_path": summary_path,
                        "split": args.split,
                        "hide_ratio": float(hide_ratio),
                        "mask_seed": int(mask_seed),
                        "original_auc": original_metrics["auc"],
                        "hidden_auc": hidden_metrics["auc"],
                        "delta_auc": original_metrics["auc"] - hidden_metrics["auc"],
                        "hidden_acc": hidden_metrics["acc"],
                        "hidden_brier": hidden_metrics["brier"],
                        "hidden_ece": hidden_metrics["ece"],
                        "hidden_loss": hidden_metrics["loss"],
                        "original_history_rows": original_history_rows,
                        "hidden_history_rows": hidden_history_rows,
                        "hidden_history_ratio": hidden_history_rows / original_history_rows
                        if original_history_rows
                        else math.nan,
                    }
                )

    summary_rows = summarize_rows(rows)
    payload = {
        "dataset": args.dataset_name,
        "split": args.split,
        "hide_ratios": args.hide_ratios,
        "mask_seeds": args.mask_seeds,
        "history_mask_mode": args.history_mask_mode,
        "keep_exercise_evidence": args.keep_exercise_evidence,
        "device": device,
        "rows": rows,
        "summary": summary_rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(payload, output_path)

    per_run_csv = (
        Path(args.per_run_csv) if args.per_run_csv else output_path.with_name(f"{output_path.stem}_per_run.csv")
    )
    summary_csv = (
        Path(args.summary_csv) if args.summary_csv else output_path.with_name(f"{output_path.stem}_summary.csv")
    )
    pd.DataFrame(rows).to_csv(per_run_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
