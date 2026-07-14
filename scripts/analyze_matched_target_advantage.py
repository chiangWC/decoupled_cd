from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ALIGNMENT_COLUMNS = ("stu_id", "exer_id", "label")
MATCH_UNITS = {"item": "exer_id", "student": "stu_id"}
EPSILON = 1.0e-7


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    ours_path: Path
    external_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether the row-aligned advantage over an external model "
            "is larger on a target coverage slice than on matched full-coverage rows."
        )
    )
    parser.add_argument(
        "--dataset-spec",
        action="append",
        nargs=3,
        metavar=("NAME", "OURS_CSV", "EXTERNAL_CSV"),
        required=True,
        help="Repeat once per dataset.",
    )
    parser.add_argument("--target-bucket", default="zero")
    parser.add_argument("--reference-bucket", default="full")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _validate_probability(values: pd.Series, *, label: str) -> np.ndarray:
    probabilities = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{label} probabilities contain non-finite values.")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError(f"{label} probabilities must be in [0, 1].")
    return probabilities


def load_aligned_predictions(spec: DatasetSpec) -> pd.DataFrame:
    ours = pd.read_csv(spec.ours_path)
    external = pd.read_csv(spec.external_path)
    required_ours = {*ALIGNMENT_COLUMNS, "prob"}
    required_external = {*ALIGNMENT_COLUMNS, "prob", "coverage_bucket"}
    missing_ours = sorted(required_ours.difference(ours.columns))
    missing_external = sorted(required_external.difference(external.columns))
    if missing_ours:
        raise ValueError(f"{spec.name}: ours is missing columns {missing_ours}.")
    if missing_external:
        raise ValueError(
            f"{spec.name}: external is missing columns {missing_external}."
        )
    if len(ours) != len(external):
        raise ValueError(
            f"{spec.name}: row count mismatch: ours={len(ours)}, "
            f"external={len(external)}."
        )
    for column in ALIGNMENT_COLUMNS:
        ours_values = pd.to_numeric(ours[column], errors="raise").to_numpy()
        external_values = pd.to_numeric(
            external[column], errors="raise"
        ).to_numpy()
        if not np.array_equal(ours_values, external_values):
            mismatch = int(np.flatnonzero(ours_values != external_values)[0])
            raise ValueError(
                f"{spec.name}: {column} is not row-aligned at row {mismatch}."
            )

    labels = pd.to_numeric(ours["label"], errors="raise").to_numpy(dtype=np.int64)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{spec.name}: labels must be binary.")
    ours_prob = _validate_probability(ours["prob"], label=f"{spec.name} ours")
    external_prob = _validate_probability(
        external["prob"], label=f"{spec.name} external"
    )
    clipped_ours = np.clip(ours_prob, EPSILON, 1.0 - EPSILON)
    clipped_external = np.clip(external_prob, EPSILON, 1.0 - EPSILON)
    labels_float = labels.astype(np.float64)
    frame = pd.DataFrame(
        {
            "stu_id": pd.to_numeric(ours["stu_id"], errors="raise").to_numpy(),
            "exer_id": pd.to_numeric(ours["exer_id"], errors="raise").to_numpy(),
            "label": labels,
            "ours_prob": ours_prob,
            "external_prob": external_prob,
            "coverage_bucket": external["coverage_bucket"].astype(str).to_numpy(),
            "brier_advantage": (labels_float - external_prob) ** 2
            - (labels_float - ours_prob) ** 2,
            "log_loss_advantage": -(
                labels_float * np.log(clipped_external)
                + (1.0 - labels_float) * np.log(1.0 - clipped_external)
            )
            + (
                labels_float * np.log(clipped_ours)
                + (1.0 - labels_float) * np.log(1.0 - clipped_ours)
            ),
        }
    )
    return frame


def _weighted_auc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> float:
    if np.unique(labels[weights > 0.0]).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities, sample_weight=weights))


def _balanced_group_weights(
    frame: pd.DataFrame,
    *,
    unit_column: str,
    group_column: str,
) -> np.ndarray:
    counts = frame.groupby([unit_column, group_column])[unit_column].transform("size")
    return 1.0 / counts.to_numpy(dtype=np.float64)


def matched_effect(
    frame: pd.DataFrame,
    *,
    match_name: str,
    target_bucket: str,
    reference_bucket: str,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    if match_name not in MATCH_UNITS:
        raise ValueError(f"Unknown matching strategy: {match_name}.")
    unit_column = MATCH_UNITS[match_name]
    selected = frame.loc[
        frame["coverage_bucket"].isin([target_bucket, reference_bucket])
    ].copy()
    selected["group"] = np.where(
        selected["coverage_bucket"].eq(target_bucket), "target", "reference"
    )
    availability = selected.groupby(unit_column)["group"].nunique()
    eligible_units = availability.index[availability.eq(2)]
    matched = selected.loc[selected[unit_column].isin(eligible_units)].copy()
    if eligible_units.empty:
        raise ValueError(f"No eligible {match_name} units contain both groups.")

    unit_group = (
        matched.groupby([unit_column, "group"], sort=True)
        .agg(
            rows=("label", "size"),
            brier_advantage=("brier_advantage", "mean"),
            log_loss_advantage=("log_loss_advantage", "mean"),
        )
        .reset_index()
    )
    wide = unit_group.pivot(index=unit_column, columns="group")
    brier_unit_did = (
        wide["brier_advantage"]["target"]
        - wide["brier_advantage"]["reference"]
    ).to_numpy(dtype=np.float64)
    log_loss_unit_did = (
        wide["log_loss_advantage"]["target"]
        - wide["log_loss_advantage"]["reference"]
    ).to_numpy(dtype=np.float64)

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0, len(eligible_units), size=(bootstrap, len(eligible_units))
    )
    brier_bootstrap = brier_unit_did[sample_indices].mean(axis=1)
    log_loss_bootstrap = log_loss_unit_did[sample_indices].mean(axis=1)

    base_weights = _balanced_group_weights(
        matched, unit_column=unit_column, group_column="group"
    )
    group_metrics: dict[str, dict[str, float | int]] = {}
    for group in ("target", "reference"):
        mask = matched["group"].eq(group).to_numpy()
        labels = matched.loc[mask, "label"].to_numpy(dtype=np.int64)
        weights = base_weights[mask]
        ours_auc = _weighted_auc(
            labels,
            matched.loc[mask, "ours_prob"].to_numpy(dtype=np.float64),
            weights,
        )
        external_auc = _weighted_auc(
            labels,
            matched.loc[mask, "external_prob"].to_numpy(dtype=np.float64),
            weights,
        )
        group_metrics[group] = {
            "rows": int(mask.sum()),
            "positive_rate": float(np.average(labels, weights=weights)),
            "ours_auc": ours_auc,
            "external_auc": external_auc,
            "auc_margin": ours_auc - external_auc,
            "brier_advantage": float(
                wide["brier_advantage"][group].mean()
            ),
            "log_loss_advantage": float(
                wide["log_loss_advantage"][group].mean()
            ),
        }

    brier_did = float(brier_unit_did.mean())
    log_loss_did = float(log_loss_unit_did.mean())
    auc_did = float(
        group_metrics["target"]["auc_margin"]
        - group_metrics["reference"]["auc_margin"]
    )
    return {
        "match": match_name,
        "unit_column": unit_column,
        "eligible_units": int(len(eligible_units)),
        "target_bucket": target_bucket,
        "reference_bucket": reference_bucket,
        "target": group_metrics["target"],
        "reference": group_metrics["reference"],
        "brier_advantage_did": brier_did,
        "brier_advantage_did_ci_low": float(
            np.quantile(brier_bootstrap, 0.025)
        ),
        "brier_advantage_did_ci_high": float(
            np.quantile(brier_bootstrap, 0.975)
        ),
        "log_loss_advantage_did": log_loss_did,
        "log_loss_advantage_did_ci_low": float(
            np.quantile(log_loss_bootstrap, 0.025)
        ),
        "log_loss_advantage_did_ci_high": float(
            np.quantile(log_loss_bootstrap, 0.975)
        ),
        "auc_margin_did": auc_did,
        "bootstrap_units": match_name,
        "bootstrap_replicates": int(bootstrap),
    }


def _flat_row(dataset: str, result: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "match": result["match"],
        "eligible_units": result["eligible_units"],
        "target_bucket": result["target_bucket"],
        "reference_bucket": result["reference_bucket"],
    }
    for group in ("target", "reference"):
        for key, value in result[group].items():
            row[f"{group}_{key}"] = value
    for key, value in result.items():
        if key not in {
            "match",
            "unit_column",
            "eligible_units",
            "target_bucket",
            "reference_bucket",
            "target",
            "reference",
        }:
            row[key] = value
    return row


def plot_forest(summary: pd.DataFrame, *, metric: str, output: Path) -> None:
    point_column = metric
    low_column = f"{metric}_ci_low"
    high_column = f"{metric}_ci_high"
    metric_label = {
        "brier_advantage_did": "Brier-score reduction advantage",
        "log_loss_advantage_did": "Log-loss reduction advantage",
    }.get(metric, metric)
    labels = [
        f"{dataset} / {match}"
        for dataset, match in zip(summary["dataset"], summary["match"])
    ]
    points = summary[point_column].to_numpy(dtype=np.float64)
    lows = summary[low_column].to_numpy(dtype=np.float64)
    highs = summary[high_column].to_numpy(dtype=np.float64)
    width = 1000
    left = 260
    right = 40
    top = 42
    row_height = 44
    bottom = 72
    height = top + row_height * len(summary) + bottom
    minimum = float(min(0.0, lows.min()))
    maximum = float(max(0.0, highs.max()))
    span = max(maximum - minimum, 1.0e-9)
    minimum -= span * 0.08
    maximum += span * 0.08

    def x_position(value: float) -> float:
        fraction = (value - minimum) / (maximum - minimum)
        return left + fraction * (width - left - right)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}'
        '.label{font-size:15px}.tick{font-size:12px;fill:#555}'
        '.axis{stroke:#888;stroke-width:1}.ci{stroke:#1769aa;stroke-width:3}'
        '.point{fill:#1769aa}</style>',
        f'<text class="label" x="{width / 2:.2f}" y="22" '
        f'text-anchor="middle">{html.escape(metric_label)}</text>',
    ]
    zero_x = x_position(0.0)
    elements.append(
        f'<line x1="{zero_x:.2f}" y1="{top - 18}" x2="{zero_x:.2f}" '
        f'y2="{height - bottom + 10}" stroke="#555" stroke-width="1" '
        'stroke-dasharray="5,4"/>'
    )
    for index, (label, point, low, high) in enumerate(
        zip(labels, points, lows, highs)
    ):
        y = top + index * row_height
        low_x = x_position(float(low))
        high_x = x_position(float(high))
        point_x = x_position(float(point))
        elements.extend(
            [
                f'<text class="label" x="{left - 14}" y="{y + 5}" '
                f'text-anchor="end">{html.escape(label)}</text>',
                f'<line class="ci" x1="{low_x:.2f}" y1="{y}" '
                f'x2="{high_x:.2f}" y2="{y}"/>',
                f'<line class="ci" x1="{low_x:.2f}" y1="{y - 5}" '
                f'x2="{low_x:.2f}" y2="{y + 5}"/>',
                f'<line class="ci" x1="{high_x:.2f}" y1="{y - 5}" '
                f'x2="{high_x:.2f}" y2="{y + 5}"/>',
                f'<circle class="point" cx="{point_x:.2f}" cy="{y}" r="5"/>',
            ]
        )
    axis_y = height - bottom + 18
    elements.append(
        f'<line class="axis" x1="{left}" y1="{axis_y}" '
        f'x2="{width - right}" y2="{axis_y}"/>'
    )
    for value in np.linspace(minimum, maximum, num=6):
        x = x_position(float(value))
        elements.extend(
            [
                f'<line class="axis" x1="{x:.2f}" y1="{axis_y}" '
                f'x2="{x:.2f}" y2="{axis_y + 5}"/>',
                f'<text class="tick" x="{x:.2f}" y="{axis_y + 21}" '
                f'text-anchor="middle">{value:.4f}</text>',
            ]
        )
    elements.append(
        f'<text class="label" x="{(left + width - right) / 2:.2f}" '
        f'y="{height - 12}" text-anchor="middle">'
        f'Target-minus-matched-seen {html.escape(metric_label.lower())} '
        '(positive favors target-specific gain)'
        '</text>'
    )
    elements.append("</svg>")
    output.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive.")
    specs = [
        DatasetSpec(name, Path(ours), Path(external))
        for name, ours, external in args.dataset_spec
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nested: dict[str, Any] = {
        "schema_version": 1,
        "estimand": (
            "(ours minus external advantage on target) minus "
            "(ours minus external advantage on matched full coverage)"
        ),
        "positive_direction": "larger relative advantage on target rows",
        "bootstrap": int(args.bootstrap),
        "seed": int(args.seed),
        "datasets": {},
    }
    flat_rows: list[dict[str, Any]] = []
    for dataset_index, spec in enumerate(specs):
        frame = load_aligned_predictions(spec)
        dataset_results: dict[str, Any] = {}
        for match_index, match_name in enumerate(MATCH_UNITS):
            result = matched_effect(
                frame,
                match_name=match_name,
                target_bucket=args.target_bucket,
                reference_bucket=args.reference_bucket,
                bootstrap=args.bootstrap,
                seed=args.seed + 1009 * dataset_index + 53 * match_index,
            )
            dataset_results[match_name] = result
            flat_rows.append(_flat_row(spec.name, result))
        nested["datasets"][spec.name] = dataset_results

    summary = pd.DataFrame(flat_rows)
    summary.to_csv(args.output_dir / "matched_target_summary.csv", index=False)
    with (args.output_dir / "matched_target_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(nested, handle, indent=2, sort_keys=True)
        handle.write("\n")
    plot_forest(
        summary,
        metric="brier_advantage_did",
        output=args.output_dir / "matched_brier_advantage_forest.svg",
    )
    plot_forest(
        summary,
        metric="log_loss_advantage_did",
        output=args.output_dir / "matched_log_loss_advantage_forest.svg",
    )


if __name__ == "__main__":
    main()
