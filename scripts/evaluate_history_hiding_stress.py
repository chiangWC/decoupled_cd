from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import StepDataBundle, prepare_experiment_split_bundles
from data.pipeline import build_history_tensors
from models import DecoupledCDM, DecoupledCDMEnsemble
from scripts.analyze_prediction_slices import derive_q_matrix_from_splits_if_needed, load_model
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
    parser.add_argument("--train-interactions", default=None, help="Override train split path for every summary.")
    parser.add_argument("--valid-interactions", default=None, help="Override valid split path for every summary.")
    parser.add_argument("--test-interactions", default=None, help="Override test split path for every summary.")
    parser.add_argument(
        "--valid-history-interactions",
        default=None,
        help="Override validation support-history path for every summary.",
    )
    parser.add_argument(
        "--test-history-interactions",
        default=None,
        help="Override test support-history path for every summary.",
    )
    parser.add_argument("--q-matrix", default=None, help="Override Q-matrix path for every summary.")
    parser.add_argument("--concept-graph", default=None, help="Override concept graph path for every summary.")
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
    parser.add_argument(
        "--curve-svg",
        default=None,
        help="Optional SVG path for the validation AUC stress curve.",
    )
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


def normalize_summary_for_current_loader(summary: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(summary)
    apply_mode = str(normalized.get("concept_evidence_prior_apply_mode", "all"))
    if apply_mode not in {"all", "eval_only", "train_only"}:
        if bool(normalized.get("concept_evidence_prior_residual", False)):
            raise ValueError(
                "Unsupported concept_evidence_prior_apply_mode in an enabled concept-prior summary: "
                f"{apply_mode}"
            )
        normalized["concept_evidence_prior_apply_mode"] = "all"
    return normalized


def resolve_split_paths(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, str | None]:
    train_path = args.train_interactions or summary.get("train_interactions")
    valid_path = args.valid_interactions or summary.get("valid_interactions")
    test_path = args.test_interactions or summary.get("test_interactions")
    if not all([train_path, valid_path, test_path]):
        raise ValueError("History hiding stress evaluation requires train/valid/test interaction paths.")
    valid_history_path = (
        args.valid_history_interactions
        or summary.get("valid_history_interactions")
    )
    test_history_path = (
        args.test_history_interactions
        or summary.get("test_history_interactions")
    )
    if (valid_history_path is None) != (test_history_path is None):
        raise ValueError(
            "Validation and test support-history paths must be provided together."
        )

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


def prepare_bundles(summary: dict[str, Any], args: argparse.Namespace | None = None) -> dict[str, Any]:
    if args is not None:
        paths = resolve_split_paths(summary, args)
        return prepare_experiment_split_bundles(
            train_interactions_path=str(paths["train_interactions"]),
            valid_interactions_path=str(paths["valid_interactions"]),
            test_interactions_path=str(paths["test_interactions"]),
            q_matrix_path=str(paths["q_matrix"]),
            valid_history_interactions_path=paths.get(
                "valid_history_interactions"
            ),
            test_history_interactions_path=paths.get(
                "test_history_interactions"
            ),
            concept_graph_path=paths.get("concept_graph"),
            prerequisite_graph_path=paths.get("prerequisite_graph"),
            similarity_graph_path=paths.get("similarity_graph"),
        )

    return prepare_experiment_split_bundles(
        train_interactions_path=summary["train_interactions"],
        valid_interactions_path=summary["valid_interactions"],
        test_interactions_path=summary["test_interactions"],
        q_matrix_path=summary["q_matrix"],
        valid_history_interactions_path=summary.get(
            "valid_history_interactions"
        ),
        test_history_interactions_path=summary.get(
            "test_history_interactions"
        ),
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


def history_mask_hash(frame: pd.DataFrame) -> str:
    normalized = frame.fillna("<NA>").astype(str)
    values = pd.util.hash_pandas_object(normalized, index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


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
        q_matrix=bundle.q_matrix,
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
    model: torch.nn.Module,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    _validate_history_visibility(bundle)
    torch_device = torch.device(device)
    tensors = _bundle_tensors(bundle, torch_device)
    labels = tensors["interaction_labels"]
    student_batch_size = getattr(model, "evaluation_student_batch_size", None)
    with torch.no_grad():
        if student_batch_size is None:
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
            probs = output.probs
        else:
            probs = torch.empty_like(labels)
            unique_students = torch.unique(
                tensors["interaction_student_ids"],
                sorted=True,
            )
            for start in range(0, unique_students.numel(), int(student_batch_size)):
                student_ids = unique_students[start : start + int(student_batch_size)]
                selected = torch.zeros(
                    tensors["student_exercise_mask"].size(0),
                    dtype=torch.bool,
                    device=torch_device,
                )
                selected[student_ids] = True
                row_indices = torch.nonzero(
                    selected[tensors["interaction_student_ids"]],
                    as_tuple=False,
                ).squeeze(-1)
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
                    target_student_ids=tensors["interaction_student_ids"][row_indices],
                    target_exercise_ids=tensors["interaction_exercise_ids"][row_indices],
                    use_student_subset=True,
                )
                probs[row_indices] = output.probs
        loss = torch.nn.functional.binary_cross_entropy(probs, labels)
    return labels.detach().cpu(), probs.detach().cpu(), float(loss.item())


def evaluate_bundle(
    *,
    bundle: StepDataBundle,
    model: DecoupledCDM | DecoupledCDMEnsemble,
    device: str,
    coverage_masks: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    labels, probs, loss = predict_bundle(bundle=bundle, model=model, device=device)
    label_values = labels.numpy()
    probability_values = probs.numpy()
    metrics = compute_metrics(labels=label_values, probs=probability_values)
    metrics["loss"] = loss
    masks = coverage_masks or fixed_coverage_masks(bundle)
    for bucket, mask in masks.items():
        selected = mask.numpy()
        metrics[f"{bucket}_rows"] = int(selected.sum())
        if selected.sum() > 0 and len(set(label_values[selected].tolist())) == 2:
            slice_metrics = compute_metrics(
                labels=label_values[selected],
                probs=probability_values[selected],
            )
            metrics[f"{bucket}_auc"] = float(slice_metrics["auc"])
            metrics[f"{bucket}_brier"] = float(slice_metrics["brier"])
        else:
            metrics[f"{bucket}_auc"] = math.nan
            metrics[f"{bucket}_brier"] = math.nan
    return {key: float(value) for key, value in metrics.items() if key != "calibration_bins"}


def fixed_coverage_masks(bundle: StepDataBundle) -> dict[str, torch.Tensor]:
    required = bundle.q_matrix_tensor[bundle.interaction_exercise_ids] > 0
    observed = bundle.student_tkc_mask[bundle.interaction_student_ids] > 0
    overlap = (required & observed).sum(dim=1)
    required_count = required.sum(dim=1)
    return {
        "zero": overlap.eq(0),
        "partial": overlap.gt(0) & overlap.lt(required_count),
        "full": overlap.eq(required_count),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    metric_cols = [
        "hidden_auc",
        "delta_auc",
        "hidden_acc",
        "hidden_brier",
        "hidden_ece",
        "hidden_history_rows",
        "hidden_zero_auc",
        "delta_zero_auc",
        "hidden_full_auc",
        "delta_full_auc",
    ]
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


def plot_curve_svg(summary_rows: list[dict[str, Any]], output: Path) -> None:
    frame = pd.DataFrame(summary_rows)
    if frame.empty:
        return
    width, height = 920, 560
    left, right, top, bottom = 90, 35, 55, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    minimum_auc = float(frame["mean_hidden_auc"].min())
    maximum_auc = float(frame["mean_hidden_auc"].max())
    padding = max((maximum_auc - minimum_auc) * 0.15, 0.005)
    minimum_auc -= padding
    maximum_auc += padding

    def x_position(retained: float) -> float:
        return left + retained * plot_width

    def y_position(auc: float) -> float:
        return top + (maximum_auc - auc) / (maximum_auc - minimum_auc) * plot_height

    colors = ["#1769aa", "#d1495b", "#2a9d8f", "#7b2cbf"]
    dataset_name = str(frame["dataset"].iloc[0])
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#777;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.tick{font-size:12px}.label{font-size:15px}.title{font-size:18px;font-weight:bold}</style>',
        f'<text class="title" x="{width / 2}" y="28" text-anchor="middle">{html.escape(dataset_name)} History Hiding Stress Curve</text>',
    ]
    for retained in (0.2, 0.4, 0.6, 0.8, 1.0):
        x = x_position(retained)
        elements.extend([
            f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>',
            f'<text class="tick" x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle">{retained * 100:.0f}%</text>',
        ])
    for auc in np.linspace(minimum_auc, maximum_auc, 6):
        y = y_position(float(auc))
        elements.extend([
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>',
            f'<text class="tick" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">{auc:.3f}</text>',
        ])
    elements.extend([
        f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
        f'<text class="label" x="{left + plot_width / 2}" y="{height - 18}" text-anchor="middle">Student history retained</text>',
        f'<text class="label" x="22" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 22 {top + plot_height / 2})">Validation AUC</text>',
    ])
    legend_x = left + 15
    for model_index, (model, group) in enumerate(frame.groupby("model", sort=False)):
        color = colors[model_index % len(colors)]
        ordered = group.assign(retained=1.0 - group["hide_ratio"]).sort_values("retained")
        points = [
            (x_position(float(row.retained)), y_position(float(row.mean_hidden_auc)))
            for row in ordered.itertuples(index=False)
        ]
        path = " ".join(
            [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
            + [f"L {x:.2f} {y:.2f}" for x, y in points[1:]]
        )
        elements.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3"/>')
        for row, (x, y) in zip(ordered.itertuples(index=False), points):
            standard_deviation = float(row.std_hidden_auc)
            low_y = y_position(float(row.mean_hidden_auc) - standard_deviation)
            high_y = y_position(float(row.mean_hidden_auc) + standard_deviation)
            elements.extend([
                f'<line x1="{x:.2f}" y1="{low_y:.2f}" x2="{x:.2f}" y2="{high_y:.2f}" stroke="{color}" stroke-width="1.5"/>',
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}"/>',
            ])
        legend_y = top + 12 + model_index * 24
        elements.extend([
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
            f'<text class="tick" x="{legend_x + 36}" y="{legend_y + 4}">{html.escape(str(model))}</text>',
        ])
    elements.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = str(resolve_device(args.device, args.gpus))
    summary_paths = list(args.summary)
    model_names = args.model_name or [infer_model_name(path) for path in summary_paths]

    rows: list[dict[str, Any]] = []
    for summary_path, model_name in zip(summary_paths, model_names, strict=True):
        summary = normalize_summary_for_current_loader(load_summary(summary_path))
        paths = resolve_split_paths(summary, args)
        separate_history = paths["valid_history_interactions"] is not None
        if separate_history and not args.keep_exercise_evidence:
            raise ValueError(
                "--no-keep-exercise-evidence is invalid for a student-disjoint "
                "support/query protocol because exercise evidence must remain "
                "optimizer-train-only."
            )
        bundles = prepare_bundles(summary, args)
        target_bundle = bundles[args.split]
        base_history = target_bundle.history_interactions
        model = load_model(
            summary=summary,
            checkpoint_path=summary["best_checkpoint_path"],
            bundles=bundles,
            concept_dim=int(summary["concept_dim"]),
            device=device,
        )
        coverage_masks = fixed_coverage_masks(target_bundle)
        original_metrics = evaluate_bundle(
            bundle=target_bundle,
            model=model,
            device=device,
            coverage_masks=coverage_masks,
        )
        original_history_rows = int(len(base_history))
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
                "mask_hash": history_mask_hash(base_history),
                "original_zero_auc": original_metrics["zero_auc"],
                "hidden_zero_auc": original_metrics["zero_auc"],
                "delta_zero_auc": 0.0,
                "original_full_auc": original_metrics["full_auc"],
                "hidden_full_auc": original_metrics["full_auc"],
                "delta_full_auc": 0.0,
            }
        )
        for hide_ratio in args.hide_ratios:
            for mask_seed in args.mask_seeds:
                masked_history = mask_train_history_interactions(
                    base_history,
                    hide_ratio=hide_ratio,
                    seed=mask_seed,
                )
                hidden_bundle = build_hidden_bundle(
                    bundle=target_bundle,
                    masked_history=masked_history,
                    keep_exercise_evidence=args.keep_exercise_evidence,
                )
                hidden_metrics = evaluate_bundle(
                    bundle=hidden_bundle,
                    model=model,
                    device=device,
                    coverage_masks=coverage_masks,
                )
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
                        "mask_hash": history_mask_hash(masked_history),
                        "original_zero_auc": original_metrics["zero_auc"],
                        "hidden_zero_auc": hidden_metrics["zero_auc"],
                        "delta_zero_auc": original_metrics["zero_auc"] - hidden_metrics["zero_auc"],
                        "original_full_auc": original_metrics["full_auc"],
                        "hidden_full_auc": hidden_metrics["full_auc"],
                        "delta_full_auc": original_metrics["full_auc"] - hidden_metrics["full_auc"],
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
    if args.curve_svg:
        plot_curve_svg(summary_rows, Path(args.curve_svg))
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
