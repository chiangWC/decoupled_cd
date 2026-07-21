from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Student-clustered paired bootstrap for aligned prediction AUC differences."
    )
    parser.add_argument("--full", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument(
        "--scope",
        choices=["overall", "bucket:zero", "low_coverage"],
        default="bucket:zero",
    )
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.replicates < 100:
        raise ValueError("--replicates must be at least 100.")
    return args


def _target_scope_values(frame: pd.DataFrame) -> np.ndarray:
    if "coverage_bucket" in frame:
        return frame["coverage_bucket"].astype(str).to_numpy()
    if "student_item_concept_overlap" in frame:
        overlap = frame["student_item_concept_overlap"].astype(str)
        return overlap.map(
            {
                "none_seen": "zero",
                "partial_seen": "low_coverage",
                "all_seen": "observed",
                "no_concepts": "no_concepts",
            }
        ).fillna(overlap).to_numpy()
    raise ValueError("Prediction file has no supported target-scope column.")


def _scope_mask(frame: pd.DataFrame, scope: str) -> np.ndarray:
    if scope == "overall":
        return np.ones(len(frame), dtype=bool)
    target_scope = _target_scope_values(frame)
    if scope == "bucket:zero":
        return target_scope == "zero"
    if scope == "low_coverage":
        if "coverage_group" in frame:
            return (
                frame["coverage_group"].astype(str).to_numpy()
                == "low_coverage"
            )
        return target_scope == "low_coverage"
    raise ValueError(f"Unsupported scope: {scope}")


def _validate_alignment(
    full: pd.DataFrame,
    control: pd.DataFrame,
    *,
    require_target_scope: bool,
) -> None:
    if len(full) != len(control):
        raise ValueError("Prediction files have different row counts.")
    required = ["stu_id", "exer_id", "label", "prob"]
    for column in required:
        if column not in full or column not in control:
            raise ValueError(f"Missing required prediction column: {column}")
    for column in ["stu_id", "exer_id"]:
        left = full[column].astype(str).to_numpy()
        right = control[column].astype(str).to_numpy()
        if not np.array_equal(left, right):
            mismatch = int(np.flatnonzero(left != right)[0])
            raise ValueError(f"Prediction rows are misaligned at row {mismatch} ({column}).")
    left_labels = pd.to_numeric(full["label"], errors="raise").to_numpy(
        dtype=float
    )
    right_labels = pd.to_numeric(control["label"], errors="raise").to_numpy(
        dtype=float
    )
    if not np.array_equal(left_labels, right_labels):
        mismatch = int(np.flatnonzero(left_labels != right_labels)[0])
        raise ValueError(
            f"Prediction rows are misaligned at row {mismatch} (label)."
        )
    if require_target_scope and not np.array_equal(
        _target_scope_values(full),
        _target_scope_values(control),
    ):
        raise ValueError("Prediction files disagree on target-scope membership.")


def paired_student_cluster_bootstrap(
    *,
    full: pd.DataFrame,
    control: pd.DataFrame,
    scope: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    _validate_alignment(
        full,
        control,
        require_target_scope=scope != "overall",
    )
    mask = _scope_mask(full, scope)
    scoped = full.loc[mask].reset_index(drop=True)
    scoped_control = control.loc[mask].reset_index(drop=True)
    if len(scoped) == 0:
        raise ValueError(f"Scope {scope} contains no rows.")
    labels = scoped["label"].to_numpy(dtype=float)
    if np.unique(labels).size < 2:
        raise ValueError(f"Scope {scope} does not contain both labels.")
    full_probs = scoped["prob"].to_numpy(dtype=float)
    control_probs = scoped_control["prob"].to_numpy(dtype=float)

    students, student_inverse = np.unique(
        scoped["stu_id"].astype(str).to_numpy(),
        return_inverse=True,
    )
    full_auc = float(roc_auc_score(labels, full_probs))
    control_auc = float(roc_auc_score(labels, control_probs))
    observed_delta = full_auc - control_auc

    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(replicates):
        sampled = rng.integers(0, len(students), size=len(students))
        cluster_counts = np.bincount(sampled, minlength=len(students))
        row_weights = cluster_counts[student_inverse]
        positive_weight = float(row_weights[labels > 0.5].sum())
        negative_weight = float(row_weights[labels <= 0.5].sum())
        if positive_weight == 0.0 or negative_weight == 0.0:
            continue
        full_sample_auc = roc_auc_score(labels, full_probs, sample_weight=row_weights)
        control_sample_auc = roc_auc_score(
            labels,
            control_probs,
            sample_weight=row_weights,
        )
        deltas.append(float(full_sample_auc - control_sample_auc))
    if len(deltas) < max(100, int(replicates * 0.9)):
        raise RuntimeError("Too many bootstrap replicates lacked both labels.")

    delta_array = np.asarray(deltas)
    return {
        "scope": scope,
        "row_count": int(len(scoped)),
        "student_count": int(len(students)),
        "full_auc": full_auc,
        "control_auc": control_auc,
        "delta_auc": observed_delta,
        "ci_low": float(np.quantile(delta_array, 0.025)),
        "ci_high": float(np.quantile(delta_array, 0.975)),
        "probability_delta_positive": float(np.mean(delta_array > 0.0)),
        "replicates_requested": int(replicates),
        "replicates_used": int(len(deltas)),
        "bootstrap_seed": int(seed),
    }


def main() -> None:
    args = parse_args()
    full = pd.read_csv(args.full)
    control = pd.read_csv(args.control)
    result = paired_student_cluster_bootstrap(
        full=full,
        control=control,
        scope=args.scope,
        replicates=args.replicates,
        seed=args.seed,
    )
    result["full_path"] = str(Path(args.full).resolve())
    result["control_path"] = str(Path(args.control).resolve())
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
