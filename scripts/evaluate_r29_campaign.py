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

from scripts.train_r28 import load_json
from utils import write_json
from utils.r29_evaluation import classify_external_win


MODES = ("meta_implicit", "direct_prior", "capacity_control")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate r29 clean controls and module gates.")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser.parse_args()


def summary_path(root: Path, dataset: str, variant: str, mode: str) -> Path:
    return root / f"{dataset}__{variant}__{mode}__none" / "summary.json"


def load_result(root: Path, dataset: str, variant: str, mode: str) -> dict[str, Any]:
    path = summary_path(root, dataset, variant, mode)
    if not path.exists():
        raise FileNotFoundError(path)
    return load_json(path)


def target_predictions(summary: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(summary["prediction_path"])
    target = summary["target_metrics"]
    scope = target["scope"]
    if scope.startswith("bucket:"):
        frame = frame[frame["coverage_bucket"] == scope.split(":", 1)[1]]
    elif scope == "low_coverage":
        frame = frame[frame["coverage_group"] == "low_coverage"]
    else:
        raise ValueError(f"Unsupported target scope: {scope}")
    return frame.reset_index(drop=True)


def clustered_bootstrap_delta(
    full: pd.DataFrame,
    control: pd.DataFrame,
    *,
    replicates: int,
    seed: int = 42,
) -> dict[str, float | int]:
    row_key = "source_row_id" if "source_row_id" in full.columns else "row_index"
    if row_key not in control or full[row_key].astype(str).tolist() != control[row_key].astype(str).tolist():
        raise RuntimeError("Full/control target rows are not aligned.")
    if not np.array_equal(full["label"].to_numpy(), control["label"].to_numpy()):
        raise RuntimeError("Full/control target labels differ.")
    cluster_key = "stu_id" if "stu_id" in full.columns else "mapped_student_id"
    clusters = full[cluster_key].astype(str).to_numpy()
    unique = np.unique(clusters)
    row_indices = {student: np.flatnonzero(clusters == student) for student in unique}
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    labels = full["label"].to_numpy()
    full_probs = full["prob"].to_numpy()
    control_probs = control["prob"].to_numpy()
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([row_indices[student] for student in sampled])
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        deltas.append(
            float(
                roc_auc_score(sampled_labels, full_probs[indices])
                - roc_auc_score(sampled_labels, control_probs[indices])
            )
        )
    if not deltas:
        raise RuntimeError("No valid clustered bootstrap replicates.")
    return {
        "replicates": len(deltas),
        "mean": float(np.mean(deltas)),
        "lower_95": float(np.quantile(deltas, 0.025)),
        "upper_95": float(np.quantile(deltas, 0.975)),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    rows: list[dict[str, Any]] = []
    for dataset in args.dataset:
        results = {
            mode: {
                variant: load_result(root, dataset, variant, mode)
                for variant in ("standard", "holdout")
            }
            for mode in MODES
        }
        control = max(
            ("direct_prior", "capacity_control"),
            key=lambda mode: float(results[mode]["holdout"]["target_metrics"]["auc"]),
        )
        full = results["meta_implicit"]
        baseline = results[control]
        full_values = {
            "S": float(full["standard"]["metrics"]["auc"]),
            "H": float(full["holdout"]["metrics"]["auc"]),
            "T": float(full["holdout"]["target_metrics"]["auc"]),
        }
        control_values = {
            "S": float(baseline["standard"]["metrics"]["auc"]),
            "H": float(baseline["holdout"]["metrics"]["auc"]),
            "T": float(baseline["holdout"]["target_metrics"]["auc"]),
        }
        deltas = {axis: full_values[axis] - control_values[axis] for axis in ("S", "H", "T")}
        margins = {
            "S": float(full["standard"]["overall_external_margin"]),
            "H": float(full["holdout"]["overall_external_margin"]),
            "T": float(full["holdout"]["target_external_margin"]),
        }
        ordinary, strict = classify_external_win(margins)
        bootstrap = clustered_bootstrap_delta(
            target_predictions(full["holdout"]),
            target_predictions(baseline["holdout"]),
            replicates=args.bootstrap_replicates,
        )
        rows.append(
            {
                "dataset": dataset,
                "stronger_control": control,
                "full": full_values,
                "control": control_values,
                "delta": deltas,
                "external_margin": margins,
                "ordinary_win": ordinary,
                "strict_win": strict,
                "no_axis_regression_beyond_0.001": min(deltas.values()) >= -0.001,
                "target_bootstrap": bootstrap,
                "full_parameter_count": full["standard"]["active_parameter_count"],
                "capacity_parameter_count": results["capacity_control"]["standard"][
                    "active_parameter_count"
                ],
                "parameter_difference_ratio": abs(
                    full["standard"]["active_parameter_count"]
                    - results["capacity_control"]["standard"]["active_parameter_count"]
                )
                / max(full["standard"]["active_parameter_count"], 1),
                "capacity_common_initialization_equal": (
                    full["standard"]["common_initialization_hash"]
                    == results["capacity_control"]["standard"]["common_initialization_hash"]
                ),
            }
        )
    winning = [row for row in rows if row["ordinary_win"]]
    target_two = [row for row in winning if row["delta"]["T"] >= 0.002]
    payload = {
        "schema_version": 1,
        "candidate": "meta_implicit",
        "stage": "validation",
        "rows": rows,
        "screen": {
            "ordinary_wins": len(winning),
            "strict_wins": sum(row["strict_win"] for row in rows),
            "datasets_with_target_delta_at_least_0.002": len(target_two),
            "one_target_delta_at_least_0.003": any(
                row["delta"]["T"] >= 0.003 for row in winning
            ),
            "one_positive_bootstrap_lower_bound": any(
                row["target_bootstrap"]["lower_95"] > 0 for row in winning
            ),
            "all_winning_axes_preserved": all(
                row["no_axis_regression_beyond_0.001"] for row in winning
            ),
            "preliminary_module_gate": (
                len(target_two) >= 2
                and any(row["delta"]["T"] >= 0.003 for row in winning)
                and any(row["target_bootstrap"]["lower_95"] > 0 for row in winning)
                and all(row["no_axis_regression_beyond_0.001"] for row in winning)
            ),
        },
    }
    write_json(payload, args.output)
    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "dataset": row["dataset"],
                "control": row["stronger_control"],
                **{f"full_{axis}": row["full"][axis] for axis in ("S", "H", "T")},
                **{f"delta_{axis}": row["delta"][axis] for axis in ("S", "H", "T")},
                **{f"margin_{axis}": row["external_margin"][axis] for axis in ("S", "H", "T")},
                "ordinary_win": row["ordinary_win"],
                "strict_win": row["strict_win"],
                "target_ci_lower": row["target_bootstrap"]["lower_95"],
                "target_ci_upper": row["target_bootstrap"]["upper_95"],
            }
        )
    pd.DataFrame(table_rows).to_csv(Path(args.output).with_suffix(".csv"), index=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
