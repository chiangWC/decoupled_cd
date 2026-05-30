from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import DecoupledCDM, DecoupledCDMEnsemble
from scripts.analyze_prediction_slices import load_model
from scripts.evaluate_coverage_slice import (
    add_target_coverage,
    prepare_bundles_from_paths,
    resolve_split_paths,
)
from scripts.evaluate_history_hiding_stress import (
    infer_model_name,
    load_summary,
    normalize_summary_for_current_loader,
)
from trainers.engine import _bundle_tensors, _validate_history_visibility
from utils import resolve_device, write_json


DEFAULT_STUDENT_COVERAGE_BINS = "0,0.05,0.1,0.2,0.4,0.6,0.8,1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate learned TKC/UKC student fusion gate behavior by coverage slices."
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
    parser.add_argument("--q-matrix", default=None, help="Override Q-matrix path for every summary.")
    parser.add_argument("--concept-graph", default=None, help="Override concept graph path for every summary.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--student-coverage-bins", default=DEFAULT_STUDENT_COVERAGE_BINS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bin-csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    args = parser.parse_args()
    if args.model_name is not None and len(args.model_name) != len(args.summary):
        raise ValueError("--model-name must be repeated the same number of times as --summary.")
    args.student_coverage_bins = parse_bin_edges(args.student_coverage_bins)
    return args


def parse_bin_edges(value: str) -> list[float]:
    try:
        edges = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--student-coverage-bins must be a comma-separated list of floats.") from exc
    if len(edges) < 2:
        raise ValueError("--student-coverage-bins must include at least two edges.")
    if edges[0] < 0.0 or edges[-1] > 1.0:
        raise ValueError("--student-coverage-bins edges must stay within [0, 1].")
    for left, right in zip(edges, edges[1:], strict=True):
        if right <= left:
            raise ValueError("--student-coverage-bins edges must be strictly increasing.")
    return edges


def format_coverage_bin(value: float, edges: list[float]) -> str:
    if math.isnan(value):
        return "missing"
    if value < edges[0] or value > edges[-1]:
        return "out_of_range"
    for index, (left, right) in enumerate(zip(edges, edges[1:], strict=True)):
        if value < right or index == len(edges) - 2:
            close = "]" if index == len(edges) - 2 else ")"
            return f"[{left:.2f},{right:.2f}{close}"
    return "out_of_range"


def _safe_corr(left: pd.Series, right: pd.Series) -> float | None:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 2 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return None
    return float(frame["left"].corr(frame["right"]))


def _tower_items(model: DecoupledCDM | DecoupledCDMEnsemble) -> list[tuple[str, DecoupledCDM]]:
    if isinstance(model, DecoupledCDMEnsemble):
        return [("primary", model.primary), ("secondary", model.secondary)]
    return [("single", model)]


def _interaction_gate_frame(
    *,
    tower: DecoupledCDM,
    bundle: Any,
    device: str,
) -> pd.DataFrame:
    _validate_history_visibility(bundle)
    torch_device = torch.device(device)
    tensors = _bundle_tensors(bundle, torch_device)
    target_student_ids = tensors["interaction_student_ids"]
    unique_students, inverse = torch.unique(target_student_ids, sorted=True, return_inverse=True)
    with torch.no_grad():
        propagated = tower.propagation(
            concept_embeddings=tower.concept_embedding.weight,
            exercise_embeddings=tower.exercise_embedding.weight,
            q_matrix=tensors["q_matrix"],
            concept_graph=tensors["concept_graph"],
            prerequisite_graph=tensors["prerequisite_graph"],
            similarity_graph=tensors["similarity_graph"],
            student_exercise_mask=tensors["student_exercise_mask"],
            response_matrix=tensors["response_matrix"],
            student_tkc_mask=tensors["student_tkc_mask"],
            student_ukc_mask=tensors["student_ukc_mask"],
            student_concept_evidence=tensors["student_concept_evidence"],
            student_indices=unique_students,
        )
        student_global_coverage = tensors["student_tkc_mask"].index_select(0, unique_students).mean(dim=1)

    frame = bundle.interactions.reset_index(drop=True).copy()
    frame["student_global_coverage"] = student_global_coverage.detach().cpu()[inverse.detach().cpu()].numpy()
    frame["tkc_weight"] = propagated.tkc_weight.squeeze(-1).detach().cpu()[inverse.detach().cpu()].numpy()
    return frame


def _add_mean_tower_frame(frames: list[pd.DataFrame]) -> pd.DataFrame | None:
    if len(frames) != 2:
        return None
    mean_frame = frames[0].copy()
    mean_frame["tkc_weight"] = (frames[0]["tkc_weight"].to_numpy() + frames[1]["tkc_weight"].to_numpy()) * 0.5
    return mean_frame


def aggregate_bin_rows(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    model_name: str,
    tower: str,
    student_coverage_bins: list[float],
) -> list[dict[str, Any]]:
    enriched = frame.copy()
    enriched["student_coverage_bin"] = [
        format_coverage_bin(float(value), student_coverage_bins) for value in enriched["student_global_coverage"]
    ]
    rows: list[dict[str, Any]] = []
    rows.extend(
        _aggregate_grouped_rows(
            enriched,
            group_column="student_coverage_bin",
            axis="student_global_coverage",
            dataset_name=dataset_name,
            model_name=model_name,
            tower=tower,
        )
    )
    rows.extend(
        _aggregate_grouped_rows(
            enriched,
            group_column="coverage_bucket",
            axis="target_coverage",
            dataset_name=dataset_name,
            model_name=model_name,
            tower=tower,
        )
    )
    return rows


def _aggregate_grouped_rows(
    frame: pd.DataFrame,
    *,
    group_column: str,
    axis: str,
    dataset_name: str,
    model_name: str,
    tower: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_name, group in frame.groupby(group_column, sort=False, dropna=False):
        tkc_weight = group["tkc_weight"]
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "tower": tower,
                "axis": axis,
                "bin": str(group_name),
                "count": int(len(group)),
                "mean_tkc_weight": float(tkc_weight.mean()),
                "std_tkc_weight": float(tkc_weight.std(ddof=0)) if len(group) else None,
                "min_tkc_weight": float(tkc_weight.min()),
                "max_tkc_weight": float(tkc_weight.max()),
                "mean_student_global_coverage": float(group["student_global_coverage"].mean()),
                "mean_target_coverage": (
                    float(group["target_coverage"].mean()) if group["target_coverage"].notna().any() else None
                ),
            }
        )
    return rows


def summary_row(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    model_name: str,
    tower: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "model": model_name,
        "tower": tower,
        "count": int(len(frame)),
        "mean_tkc_weight": float(frame["tkc_weight"].mean()),
        "std_tkc_weight": float(frame["tkc_weight"].std(ddof=0)),
        "mean_student_global_coverage": float(frame["student_global_coverage"].mean()),
        "mean_target_coverage": (
            float(frame["target_coverage"].mean()) if frame["target_coverage"].notna().any() else None
        ),
        "corr_student_global_coverage_tkc_weight": _safe_corr(
            frame["student_global_coverage"], frame["tkc_weight"]
        ),
        "corr_target_coverage_tkc_weight": _safe_corr(frame["target_coverage"], frame["tkc_weight"]),
    }


def main() -> None:
    args = parse_args()
    device = str(resolve_device(args.device, args.gpus))
    model_names = args.model_name or [infer_model_name(path) for path in args.summary]

    all_bin_rows: list[dict[str, Any]] = []
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

        tower_frames: list[pd.DataFrame] = []
        for tower_name, tower in _tower_items(model):
            gate_frame = _interaction_gate_frame(tower=tower, bundle=target_bundle, device=device)
            enriched = add_target_coverage(
                gate_frame,
                train_frame=bundles["train"].interactions,
                q_matrix=bundles["shared"]["q_matrix"],
            )
            tower_frames.append(enriched)
            all_bin_rows.extend(
                aggregate_bin_rows(
                    enriched,
                    dataset_name=args.dataset_name,
                    model_name=model_name,
                    tower=tower_name,
                    student_coverage_bins=args.student_coverage_bins,
                )
            )
            summary_rows.append(
                summary_row(
                    enriched,
                    dataset_name=args.dataset_name,
                    model_name=model_name,
                    tower=tower_name,
                )
            )

        mean_frame = _add_mean_tower_frame(tower_frames)
        if mean_frame is not None:
            all_bin_rows.extend(
                aggregate_bin_rows(
                    mean_frame,
                    dataset_name=args.dataset_name,
                    model_name=model_name,
                    tower="mean",
                    student_coverage_bins=args.student_coverage_bins,
                )
            )
            summary_rows.append(
                summary_row(
                    mean_frame,
                    dataset_name=args.dataset_name,
                    model_name=model_name,
                    tower="mean",
                )
            )

        run_metadata.append(
            {
                "summary_path": summary_path,
                "model": model_name,
                "split_paths": paths,
                "is_dual_tower": isinstance(model, DecoupledCDMEnsemble),
                "student_coverage_bins": args.student_coverage_bins,
            }
        )

    payload = {
        "dataset": args.dataset_name,
        "split": args.split,
        "device": device,
        "gate_definition": (
            "student-level tkc_weight = sigmoid(student_fusion_gate([student_global_coverage, tkc_mean, ukc_mean]))"
        ),
        "target_coverage_note": "target_coverage is an interaction-level association, not a direct gate input.",
        "runs": run_metadata,
        "bins": all_bin_rows,
        "summary": summary_rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(payload, output_path)

    bin_csv = Path(args.bin_csv) if args.bin_csv else output_path.with_name(f"{output_path.stem}_bins.csv")
    summary_csv = (
        Path(args.summary_csv) if args.summary_csv else output_path.with_name(f"{output_path.stem}_summary.csv")
    )
    pd.DataFrame(all_bin_rows).to_csv(bin_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
