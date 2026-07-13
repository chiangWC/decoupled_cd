from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import write_json


WITHOUT_MODE = "direct_prior"
AXES = ("S", "H", "T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate r28 Pareto and clean-ablation gates.")
    parser.add_argument("--campaign-root", default="results/r28/campaign")
    parser.add_argument("--output", default="results/r28/campaign_evaluation.json")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--full-mode",
        choices=[
            "relational",
            "partial_vae",
            "difficulty_set",
            "poe_ability",
            "hierarchical_bayes",
            "cohort_conditioned",
            "bipolar_prototype",
        ],
        default="relational",
    )
    parser.add_argument(
        "--capacity-mode",
        choices=[
            "capacity_mlp",
            "difficulty_capacity",
            "poe_capacity",
            "hierarchical_capacity",
            "cohort_capacity",
            "bipolar_capacity",
        ],
        default="capacity_mlp",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover(root: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    summaries: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sorted(root.glob("*/summary.json")):
        summary = load_json(path)
        if summary.get("evaluation_stage") != "validation":
            continue
        key = (summary["dataset"], summary["dataset_variant"], summary["state_completer"])
        summary["_summary_path"] = str(path.resolve())
        summaries[key] = summary
    return summaries


def build_rows(
    summaries: dict[tuple[str, str, str], dict[str, Any]],
    *,
    full_mode: str,
    capacity_mode: str,
) -> list[dict[str, Any]]:
    datasets = sorted({key[0] for key in summaries})
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for mode in (full_mode, WITHOUT_MODE, capacity_mode):
            standard = summaries.get((dataset, "standard", mode))
            holdout = summaries.get((dataset, "holdout", mode))
            if standard is None or holdout is None or holdout.get("target_metrics") is None:
                continue
            row = {
                "dataset": dataset,
                "mode": mode,
                "S": float(standard["metrics"]["auc"]),
                "H": float(holdout["metrics"]["auc"]),
                "T": float(holdout["target_metrics"]["auc"]),
                "S_margin": float(standard["overall_external_margin"]),
                "H_margin": float(holdout["overall_external_margin"]),
                "T_margin": float(holdout["target_external_margin"]),
                "common_initialization_hash_standard": standard["common_initialization_hash"],
                "common_initialization_hash_holdout": holdout["common_initialization_hash"],
                "parameter_count": int(standard["active_parameter_count"]),
            }
            row["ordinary_win"] = (
                row["S_margin"] >= -0.002
                and row["H_margin"] >= -0.002
                and row["T_margin"] > 0.0
            )
            row["strict_win"] = (
                row["S_margin"] >= 0.0
                and row["H_margin"] >= 0.0
                and row["T_margin"] > 0.0
            )
            rows.append(row)
    return rows


def stronger_control(
    rows: list[dict[str, Any]],
    dataset: str,
    axis: str,
    *,
    capacity_mode: str,
) -> dict[str, Any]:
    controls = [
        row for row in rows
        if row["dataset"] == dataset and row["mode"] in {WITHOUT_MODE, capacity_mode}
    ]
    if len(controls) != 2:
        raise ValueError(f"Both clean controls are required for {dataset}.")
    return max(controls, key=lambda row: float(row[axis]))


def _aligned_target_frames(
    *,
    full_summary: dict[str, Any],
    control_summary: dict[str, Any],
    target_scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.read_csv(full_summary["prediction_path"])
    control = pd.read_csv(control_summary["prediction_path"])
    keys = ["row_index", "stu_id", "exer_id", "label"]
    if not full[keys].equals(control[keys]):
        raise RuntimeError("Full/control prediction rows are not aligned.")
    if target_scope == "bucket:zero":
        mask = full["coverage_bucket"] == "zero"
    elif target_scope == "low_coverage":
        mask = full["coverage_group"] == "low_coverage"
    else:
        raise ValueError(f"Unknown target scope {target_scope!r}.")
    return full.loc[mask].reset_index(drop=True), control.loc[mask].reset_index(drop=True)


def paired_student_bootstrap(
    *,
    full: pd.DataFrame,
    control: pd.DataFrame,
    samples: int,
    seed: int = 42,
) -> dict[str, float | int]:
    student_column = "stu_id"
    students = full[student_column].astype(str).unique()
    grouped = {
        student: np.flatnonzero(full[student_column].astype(str).to_numpy() == student)
        for student in students
    }
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(samples):
        sampled = rng.choice(students, size=len(students), replace=True)
        indices = np.concatenate([grouped[student] for student in sampled])
        labels = full["label"].to_numpy()[indices]
        if np.unique(labels).size < 2:
            continue
        full_auc = roc_auc_score(labels, full["prob"].to_numpy()[indices])
        control_auc = roc_auc_score(labels, control["prob"].to_numpy()[indices])
        deltas.append(float(full_auc - control_auc))
    if not deltas:
        raise RuntimeError("No valid paired bootstrap resamples contained both labels.")
    values = np.asarray(deltas)
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "samples": int(values.size),
        "mean_delta": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def evaluate(
    summaries: dict[tuple[str, str, str], dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    full_mode: str,
    capacity_mode: str,
) -> dict[str, Any]:
    by_dataset_mode = {(row["dataset"], row["mode"]): row for row in rows}
    datasets = sorted({row["dataset"] for row in rows if row["mode"] == full_mode})
    deltas: dict[str, dict[str, Any]] = {}
    bootstrap: dict[str, Any] = {}
    for dataset in datasets:
        full = by_dataset_mode[(dataset, full_mode)]
        dataset_delta: dict[str, Any] = {}
        for axis in AXES:
            control = stronger_control(
                rows,
                dataset,
                axis,
                capacity_mode=capacity_mode,
            )
            dataset_delta[axis] = float(full[axis]) - float(control[axis])
            dataset_delta[f"{axis}_control"] = control["mode"]
        deltas[dataset] = dataset_delta

        target_control_mode = dataset_delta["T_control"]
        full_summary = summaries[(dataset, "holdout", full_mode)]
        control_summary = summaries[(dataset, "holdout", target_control_mode)]
        full_frame, control_frame = _aligned_target_frames(
            full_summary=full_summary,
            control_summary=control_summary,
            target_scope=(
                "low_coverage" if dataset == "nips34" else "bucket:zero"
            ),
        )
        bootstrap[dataset] = paired_student_bootstrap(
            full=full_frame,
            control=control_frame,
            samples=bootstrap_samples,
        )

    full_rows = [row for row in rows if row["mode"] == full_mode]
    winning = [row["dataset"] for row in full_rows if row["ordinary_win"]]
    strict = [row["dataset"] for row in full_rows if row["strict_win"]]
    screen_datasets = [dataset for dataset in ("moocradar", "xes3g5m") if dataset in deltas]
    screen_positive = any(deltas[dataset]["T"] >= 0.002 for dataset in screen_datasets)
    screen_no_regression = any(
        all(
            all(deltas[other][axis] >= -0.001 for axis in AXES)
            for other in screen_datasets
            if other != primary
        )
        for primary in screen_datasets
        if deltas[primary]["T"] >= 0.002
    )
    screen_preserves_wins = all(
        by_dataset_mode[(dataset, full_mode)]["ordinary_win"]
        for dataset in screen_datasets
    )
    preliminary_pass = (
        len(screen_datasets) == 2
        and screen_positive
        and screen_no_regression
        and screen_preserves_wins
    )

    winning_deltas = {dataset: deltas[dataset] for dataset in winning if dataset in deltas}
    target_qualified = [dataset for dataset, value in winning_deltas.items() if value["T"] >= 0.002]
    all_axes_safe = all(
        value[axis] >= -0.001
        for value in winning_deltas.values()
        for axis in AXES
    )
    significant = any(bootstrap[dataset]["ci_low"] > 0.0 for dataset in winning if dataset in bootstrap)
    full_worst_target_margin = min(
        (by_dataset_mode[(dataset, full_mode)]["T_margin"] for dataset in winning),
        default=float("-inf"),
    )
    control_worst_target_margin = min(
        (
            stronger_control(
                rows,
                dataset,
                "T",
                capacity_mode=capacity_mode,
            )["T_margin"]
            for dataset in winning
        ),
        default=float("-inf"),
    )
    added_win = any(
        by_dataset_mode[(dataset, full_mode)]["ordinary_win"]
        and all(
            not by_dataset_mode[(dataset, mode)]["ordinary_win"]
            for mode in (WITHOUT_MODE, capacity_mode)
        )
        for dataset in winning
    )
    worst_target_margin_improvement = (
        full_worst_target_margin - control_worst_target_margin
        if winning
        else None
    )
    contribution_gate = added_win or (
        worst_target_margin_improvement is not None
        and worst_target_margin_improvement >= 0.001
    )
    module_qualified = (
        len(winning) >= 3
        and len(strict) >= 2
        and len(target_qualified) >= 2
        and any(winning_deltas[dataset]["T"] >= 0.003 for dataset in target_qualified)
        and all_axes_safe
        and significant
        and contribution_gate
    )

    hashes = {
        dataset: {
            mode: {
                "standard": by_dataset_mode[(dataset, mode)]["common_initialization_hash_standard"],
                "holdout": by_dataset_mode[(dataset, mode)]["common_initialization_hash_holdout"],
            }
            for mode in (full_mode, WITHOUT_MODE, capacity_mode)
            if (dataset, mode) in by_dataset_mode
        }
        for dataset in datasets
    }
    hash_audit = all(
        len({mode_hashes[mode][variant] for mode in mode_hashes}) == 1
        for mode_hashes in hashes.values()
        for variant in ("standard", "holdout")
    )
    capacity_audit = all(
        abs(
            by_dataset_mode[(dataset, full_mode)]["parameter_count"]
            - by_dataset_mode[(dataset, capacity_mode)]["parameter_count"]
        )
        / by_dataset_mode[(dataset, full_mode)]["parameter_count"]
        <= 0.10
        for dataset in datasets
    )
    return {
        "rows": rows,
        "full_mode": full_mode,
        "capacity_mode": capacity_mode,
        "deltas_vs_stronger_control": deltas,
        "paired_student_bootstrap_target_auc": bootstrap,
        "winning_datasets": winning,
        "strict_winning_datasets": strict,
        "ordinary_win_count": len(winning),
        "strict_win_count": len(strict),
        "preliminary_screen_pass": preliminary_pass,
        "module_qualified": module_qualified,
        "qualification_details": {
            "target_qualified_datasets": target_qualified,
            "all_winning_axes_safe": all_axes_safe,
            "significant_target_delta": significant,
            "added_external_win": added_win,
            "worst_target_margin_improvement": worst_target_margin_improvement,
        },
        "common_initialization_hash_audit": hash_audit,
        "capacity_parameter_audit": capacity_audit,
        "next_action": (
            "freeze_and_test_confirmation"
            if module_qualified
            else "expand_to_assist17"
            if preliminary_pass and "assist_17" not in datasets
            else "reject_candidate_and_follow_registered_replacement_order"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples.")
    summaries = discover(Path(args.campaign_root))
    rows = build_rows(
        summaries,
        full_mode=args.full_mode,
        capacity_mode=args.capacity_mode,
    )
    payload = evaluate(
        summaries,
        rows,
        bootstrap_samples=args.bootstrap_samples,
        full_mode=args.full_mode,
        capacity_mode=args.capacity_mode,
    )
    write_json(payload, args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
