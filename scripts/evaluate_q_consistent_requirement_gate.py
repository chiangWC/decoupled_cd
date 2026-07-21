from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_target_coverage_gate import (
    ALIGNMENT_COLUMNS,
    load_train_valid,
    normalize_interactions,
    probabilities,
    sha256_file,
)
from scripts.paired_student_bootstrap import paired_student_cluster_bootstrap


TARGET_SCOPES = ("bucket:zero", "low_coverage")
OVERALL_TOLERANCE = 0.002
TARGET_GAIN = 0.002
LARGE_TARGET_GAIN = 0.003
MAX_REGRESSION = 0.001
OPTIONAL_ROW_IDS = ("source_row_id", "split_row_index")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    standard_dir: Path
    holdout_dir: Path
    full_standard: Path
    control_standard: Path
    external_standard: Path
    full_holdout: Path
    control_holdout: Path
    external_holdout: Path
    external_target: Path
    target_scope: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only Requirement gate with a train/valid/Q-derived target "
            "mask shared by Full, control, and external predictions. test.csv is "
            "never opened."
        )
    )
    parser.add_argument(
        "--dataset-spec",
        action="append",
        nargs=11,
        required=True,
        metavar=(
            "NAME", "STANDARD_DIR", "HOLDOUT_DIR", "FULL_STANDARD",
            "CONTROL_STANDARD", "EXTERNAL_STANDARD", "FULL_HOLDOUT",
            "CONTROL_HOLDOUT", "EXTERNAL_HOLDOUT", "EXTERNAL_TARGET",
            "TARGET_SCOPE",
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap < 100:
        raise ValueError("--bootstrap must be at least 100.")
    return args


def parse_specs(raw_specs: list[list[str]]) -> list[DatasetSpec]:
    specs = []
    for raw in raw_specs:
        if raw[10] not in TARGET_SCOPES:
            raise ValueError(f"{raw[0]}: unsupported target scope {raw[10]}.")
        specs.append(
            DatasetSpec(
                raw[0], *(Path(value) for value in raw[1:10]), raw[10]
            )
        )
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("Dataset names must be unique.")
    return specs


def row_identity_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame[list(ALIGNMENT_COLUMNS)].itertuples(index=False, name=None):
        digest.update("\x1f".join(map(str, row)).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def load_aligned_probabilities(
    valid: pd.DataFrame, path: Path, *, role: str
) -> tuple[np.ndarray, dict[str, Any]]:
    prediction = normalize_interactions(pd.read_csv(path), path)
    if len(prediction) != len(valid):
        raise ValueError(
            f"{role}: row mismatch valid={len(valid)}, prediction={len(prediction)}."
        )
    checked = list(ALIGNMENT_COLUMNS)
    for column in ALIGNMENT_COLUMNS:
        left, right = prediction[column].to_numpy(), valid[column].to_numpy()
        if not np.array_equal(left, right):
            row = int(np.flatnonzero(left != right)[0])
            raise ValueError(f"{role}: {column} mismatch at row {row}.")
    for column in OPTIONAL_ROW_IDS:
        if column not in valid:
            continue
        if column not in prediction:
            raise ValueError(f"{role}: missing row identity column {column}.")
        left = prediction[column].astype(str).to_numpy()
        right = valid[column].astype(str).to_numpy()
        if not np.array_equal(left, right):
            row = int(np.flatnonzero(left != right)[0])
            raise ValueError(f"{role}: {column} mismatch at row {row}.")
        checked.append(column)
    return probabilities(prediction, path), {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(prediction),
        "checked_identity_columns": checked,
    }


def build_q_consistent_coverage(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    q_map: dict[str, frozenset[str]],
) -> pd.DataFrame:
    seen: dict[str, set[str]] = defaultdict(set)
    for row in train.itertuples(index=False):
        seen[str(row.stu_id)].update(q_map[str(row.exer_id)])
    unknown = sorted(set(valid["stu_id"]).difference(seen))
    if unknown:
        raise ValueError(f"Validation students lack train history: {unknown[:5]}.")
    counts, overlaps, coverage = [], [], []
    for row in valid.itertuples(index=False):
        required = q_map[str(row.exer_id)]
        overlap = len(required.intersection(seen[str(row.stu_id)]))
        counts.append(len(required))
        overlaps.append(overlap)
        coverage.append(overlap / len(required))
    result = valid[list(ALIGNMENT_COLUMNS)].copy()
    result["target_concept_count"] = counts
    result["target_seen_concept_count"] = overlaps
    result["target_coverage"] = coverage
    result["coverage_bucket"] = np.select(
        [result.target_coverage.eq(0), result.target_coverage.lt(0.5),
         result.target_coverage.lt(1)],
        ["zero", "low", "partial"], default="full",
    )
    result["coverage_group"] = np.where(
        result.target_coverage.lt(0.5), "low_coverage",
        np.where(result.target_coverage.eq(1), "full_coverage", "other"),
    )
    return result


def target_mask(frame: pd.DataFrame, scope: str) -> np.ndarray:
    mask = (
        frame.coverage_bucket.eq("zero").to_numpy()
        if scope == "bucket:zero"
        else frame.coverage_group.eq("low_coverage").to_numpy()
    )
    labels = frame.loc[mask, "label"]
    if len(labels) == 0 or labels.nunique() != 2:
        raise ValueError(f"Target scope {scope} is empty or lacks both labels.")
    return mask


def auc(labels: pd.Series, prediction: np.ndarray, *, context: str) -> float:
    if labels.nunique() != 2:
        raise ValueError(f"{context}: AUC requires both labels.")
    return float(roc_auc_score(labels.to_numpy(), prediction))


def external_win_flags(
    full: dict[str, float], external: dict[str, float]
) -> dict[str, bool]:
    target_wins = full["T"] > external["T"]
    return {
        "ordinary": bool(
            full["S"] >= external["S"] - OVERALL_TOLERANCE
            and full["H"] >= external["H"] - OVERALL_TOLERANCE
            and target_wins
        ),
        "strict": bool(
            full["S"] >= external["S"]
            and full["H"] >= external["H"]
            and target_wins
        ),
    }


def evaluate_dataset(
    spec: DatasetSpec, *, bootstrap: int, seed: int
) -> dict[str, Any]:
    _, standard_valid, _, standard_hashes = load_train_valid(spec.standard_dir)
    holdout_train, holdout_valid, q_map, holdout_hashes = load_train_valid(
        spec.holdout_dir
    )
    coverage = build_q_consistent_coverage(holdout_train, holdout_valid, q_map)
    mask = target_mask(coverage, spec.target_scope)
    inputs = {
        "full_standard": (standard_valid, spec.full_standard),
        "control_standard": (standard_valid, spec.control_standard),
        "external_standard": (standard_valid, spec.external_standard),
        "full_holdout": (holdout_valid, spec.full_holdout),
        "control_holdout": (holdout_valid, spec.control_holdout),
        "external_holdout": (holdout_valid, spec.external_holdout),
        "external_target": (holdout_valid, spec.external_target),
    }
    prediction, artifacts = {}, {}
    for role, (valid, path) in inputs.items():
        prediction[role], artifacts[role] = load_aligned_probabilities(
            valid, path, role=f"{spec.name}/{role}"
        )
    standard_labels, holdout_labels = standard_valid.label, holdout_valid.label
    target_labels = holdout_valid.loc[mask, "label"]

    model_columns = {
        "full": ("full_standard", "full_holdout", "full_holdout"),
        "control": ("control_standard", "control_holdout", "control_holdout"),
        "external": ("external_standard", "external_holdout", "external_target"),
    }
    aucs = {}
    for model, (standard_key, holdout_key, target_key) in model_columns.items():
        aucs[model] = {
            "S": auc(standard_labels, prediction[standard_key], context=f"{spec.name}/{model}/S"),
            "H": auc(holdout_labels, prediction[holdout_key], context=f"{spec.name}/{model}/H"),
            "T": auc(target_labels, prediction[target_key][mask], context=f"{spec.name}/{model}/T"),
        }

    bootstrap_columns = [
        "stu_id", "exer_id", "label", "coverage_bucket", "coverage_group"
    ]
    bootstrap_full = coverage[bootstrap_columns].copy()
    bootstrap_control = coverage[bootstrap_columns].copy()
    bootstrap_full["prob"] = prediction["full_holdout"]
    bootstrap_control["prob"] = prediction["control_holdout"]
    bootstrap_result = paired_student_cluster_bootstrap(
        full=bootstrap_full, control=bootstrap_control, scope=spec.target_scope,
        replicates=bootstrap, seed=seed,
    )
    delta = {axis: aucs["full"][axis] - aucs["control"][axis]
             for axis in ("S", "H", "T")}
    margin = {axis: aucs["full"][axis] - aucs["external"][axis]
              for axis in ("S", "H", "T")}
    return {
        "dataset": spec.name,
        "target_scope": spec.target_scope,
        "target_rows": int(mask.sum()),
        "target_students": int(coverage.loc[mask, "stu_id"].nunique()),
        "target_label_counts": {
            str(k): int(v) for k, v in target_labels.value_counts().sort_index().items()
        },
        "auc": aucs,
        "delta_auc_full_minus_control": delta,
        "margin_auc_full_minus_external": margin,
        "external_win": external_win_flags(aucs["full"], aucs["external"]),
        "target_bootstrap": bootstrap_result,
        "row_identity_sha256": {
            "standard": row_identity_sha256(standard_valid),
            "holdout": row_identity_sha256(holdout_valid),
        },
        "data_hashes": {"standard": standard_hashes, "holdout": holdout_hashes},
        "prediction_artifacts": artifacts,
    }


def evaluate_gate(results: list[dict[str, Any]]) -> dict[str, Any]:
    winning = [result for result in results if result["external_win"]["ordinary"]]
    gains = [result["delta_auc_full_minus_control"]["T"] for result in winning]
    checks = {
        "at_least_two_wins_with_delta_t_ge_0.002": sum(
            gain >= TARGET_GAIN for gain in gains
        ) >= 2,
        "at_least_one_win_with_delta_t_ge_0.003": any(
            gain >= LARGE_TARGET_GAIN for gain in gains
        ),
        "at_least_one_winning_ci_low_gt_0": any(
            result["target_bootstrap"]["ci_low"] > 0 for result in winning
        ),
        "no_winning_axis_regression_gt_0.001": all(
            all(delta >= -MAX_REGRESSION
                for delta in result["delta_auc_full_minus_control"].values())
            for result in winning
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "ordinary_external_win_count": len(winning),
        "strict_external_win_count": sum(
            result["external_win"]["strict"] for result in results
        ),
        "winning_datasets": [result["dataset"] for result in winning],
        "nonwinning_datasets": [result["dataset"] for result in results
                                if not result["external_win"]["ordinary"]],
        "all_supplied_datasets_no_axis_regression_gt_0.001": all(
            all(delta >= -MAX_REGRESSION
                for delta in result["delta_auc_full_minus_control"].values())
            for result in results
        ),
        "thresholds": {
            "ordinary_overall_tolerance": OVERALL_TOLERANCE,
            "target_gain_two_datasets": TARGET_GAIN,
            "target_gain_one_dataset": LARGE_TARGET_GAIN,
            "max_axis_regression": MAX_REGRESSION,
            "target_external_comparison": "strict",
            "bootstrap_ci_comparison": "ci_low > 0",
        },
    }


def main() -> None:
    args = parse_args()
    results = [evaluate_dataset(spec, bootstrap=args.bootstrap, seed=args.seed)
               for spec in parse_specs(args.dataset_spec)]
    payload = {
        "schema_version": 1,
        "protocol": "validation_only_q_consistent_requirement_gate",
        "test_artifacts_accessed": False,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "datasets": results,
        "gate": evaluate_gate(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
