from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


REQUIRED_METRICS = (
    "standard_overall_auc",
    "holdout_overall_auc",
    "zero_auc",
    "ordinary_doa",
    "weighted_doa",
)
IDENTITY_FIELDS = (
    "dataset_id",
    "cohort_sha256",
    "architecture_fingerprint",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _has_test_token(value: str) -> bool:
    expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    expanded = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", expanded)
    tokens = re.findall(r"[A-Za-z0-9]+", expanded)
    return any(token.lower() == "test" for token in tokens)


def _reject_test_references(value: object, *, location: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _has_test_token(str(key)):
                raise ValueError(f"test metric/path is forbidden at {location}.{key}")
            _reject_test_references(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_test_references(item, location=f"{location}[{index}]")
    elif isinstance(value, str) and _has_test_token(value):
        raise ValueError(f"test metric/path is forbidden at {location}")


def _validated_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    label: str,
) -> tuple[dict[str, dict[str, object]], str, str, tuple[str, ...]]:
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError(f"{label} rows must be a nonempty sequence")
    _reject_test_references(rows, location=label)

    indexed: dict[str, dict[str, object]] = {}
    fingerprints: set[str] = set()
    cohort_hashes: set[str] = set()
    metric_fields: tuple[str, ...] | None = None
    for index, source_row in enumerate(rows):
        if not isinstance(source_row, Mapping):
            raise ValueError(f"{label} row {index} must be a JSON object")
        row = dict(source_row)
        missing = [
            field
            for field in (*IDENTITY_FIELDS, *REQUIRED_METRICS, "parameter_count")
            if field not in row
        ]
        if missing:
            raise ValueError(f"{label} row {index} is missing fields: {missing}")

        dataset_id = row["dataset_id"]
        cohort_sha256 = row["cohort_sha256"]
        architecture_fingerprint = row["architecture_fingerprint"]
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError(f"{label} row {index} has invalid dataset_id")
        if dataset_id in indexed:
            raise ValueError(f"{label} rows contain duplicate dataset_id: {dataset_id}")
        if not isinstance(cohort_sha256, str) or not SHA256_PATTERN.fullmatch(
            cohort_sha256
        ):
            raise ValueError(
                f"{label} row {index} cohort_sha256 must be lowercase 64-hex SHA-256"
            )
        if not isinstance(
            architecture_fingerprint, str
        ) or not SHA256_PATTERN.fullmatch(architecture_fingerprint):
            raise ValueError(
                f"{label} row {index} architecture_fingerprint must be "
                "lowercase 64-hex SHA-256"
            )
        parameter_count = row["parameter_count"]
        if type(parameter_count) is not int or parameter_count < 0:
            raise ValueError(
                f"{label} row {dataset_id} parameter_count must be a "
                "nonnegative integer"
            )

        current_metric_fields = tuple(
            sorted(
                key
                for key in row
                if key not in IDENTITY_FIELDS
                and (key.endswith("_auc") or key.endswith("_doa"))
            )
        )
        if metric_fields is None:
            metric_fields = current_metric_fields
        elif current_metric_fields != metric_fields:
            raise ValueError(f"{label} rows have mismatched validation metric sets")
        for field in current_metric_fields:
            value = row[field]
            if (
                type(value) is not float
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(
                    f"{label} row {dataset_id} metric {field} must be a "
                    "finite float in [0, 1]"
                )

        indexed[dataset_id] = row
        fingerprints.add(architecture_fingerprint)
        cohort_hashes.add(cohort_sha256)

    if len(fingerprints) != 1:
        raise ValueError(f"{label} rows contain mixed architecture fingerprints")
    if len(cohort_hashes) != 1:
        raise ValueError(f"{label} rows contain mixed cohort hashes")
    assert metric_fields is not None
    return indexed, fingerprints.pop(), cohort_hashes.pop(), metric_fields


def evaluate_candidate(
    baseline_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    baseline, baseline_fingerprint, baseline_cohort, baseline_metrics = _validated_rows(
        baseline_rows,
        label="baseline",
    )
    candidate, candidate_fingerprint, candidate_cohort, candidate_metrics = _validated_rows(
        candidate_rows,
        label="candidate",
    )
    if set(baseline) != set(candidate):
        raise ValueError(
            "baseline and candidate dataset sets differ: "
            f"baseline={sorted(baseline)}, candidate={sorted(candidate)}"
        )
    if baseline_metrics != candidate_metrics:
        raise ValueError("baseline and candidate validation metric sets differ")

    dataset_ids = sorted(baseline)
    deltas: dict[str, dict[str, float]] = {}
    for dataset_id in dataset_ids:
        dataset_deltas: dict[str, float] = {}
        for metric in baseline_metrics:
            baseline_value = baseline[dataset_id][metric]
            candidate_value = candidate[dataset_id][metric]
            assert type(baseline_value) is float
            assert type(candidate_value) is float
            delta = candidate_value - baseline_value
            if not math.isfinite(delta):
                raise ValueError(
                    f"computed delta is not finite for {dataset_id}.{metric}"
                )
            dataset_deltas[metric] = delta
        deltas[dataset_id] = dataset_deltas
    required_improvements = math.ceil(2 * len(dataset_ids) / 3)

    def non_regression(metric: str) -> tuple[bool, list[str]]:
        failures = [
            dataset_id
            for dataset_id in dataset_ids
            if deltas[dataset_id][metric] < 0.0
        ]
        return not failures, failures

    standard_pass, standard_failures = non_regression("standard_overall_auc")
    holdout_pass, holdout_failures = non_regression("holdout_overall_auc")
    zero_improved = [
        dataset_id for dataset_id in dataset_ids if deltas[dataset_id]["zero_auc"] > 0.0
    ]
    zero_threshold = [
        dataset_id
        for dataset_id in dataset_ids
        if deltas[dataset_id]["zero_auc"] >= 0.001
    ]

    gates: dict[str, dict[str, Any]] = {
        "standard_overall_auc_non_regression": {
            "pass": standard_pass,
            "failed_datasets": standard_failures,
        },
        "holdout_overall_auc_non_regression": {
            "pass": holdout_pass,
            "failed_datasets": holdout_failures,
        },
        "zero_auc_improved_two_thirds": {
            "pass": len(zero_improved) >= required_improvements,
            "improved_datasets": zero_improved,
            "required": required_improvements,
        },
        "zero_auc_delta_at_least_0.001": {
            "pass": bool(zero_threshold),
            "qualifying_datasets": zero_threshold,
            "required_delta": 0.001,
        },
        "same_frozen_cohort": {
            "pass": baseline_cohort == candidate_cohort,
        },
    }
    failed_gates = [name for name, result in gates.items() if not result["pass"]]
    zero_deltas = [deltas[dataset_id]["zero_auc"] for dataset_id in dataset_ids]
    weighted_deltas = [
        deltas[dataset_id]["weighted_doa"] for dataset_id in dataset_ids
    ]
    total_parameter_count = sum(
        int(row["parameter_count"]) for row in candidate.values()
    )
    ranking = {
        "mean_zero_auc_delta": math.fsum(zero_deltas) / len(zero_deltas),
        "worst_zero_auc_delta": min(zero_deltas),
        "mean_weighted_doa_delta": math.fsum(weighted_deltas)
        / len(weighted_deltas),
        "negative_parameter_count": -float(total_parameter_count),
    }
    for name, value in ranking.items():
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f"computed ranking value is not finite: {name}")
    return {
        "schema_version": 1,
        "cohort_sha256": baseline_cohort,
        "baseline_architecture_fingerprint": baseline_fingerprint,
        "candidate_architecture_fingerprint": candidate_fingerprint,
        "dataset_ids": dataset_ids,
        "dataset_count": len(dataset_ids),
        "required_improvements": required_improvements,
        "deltas": deltas,
        "ranking": ranking,
        "gates": gates,
        "failed_gates": failed_gates,
        "pass": not failed_gates,
    }


def _load_rows(path: Path) -> Sequence[Mapping[str, object]]:
    _reject_test_references(str(path), location="metrics path")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "rows" in payload:
        payload = payload["rows"]
    if not isinstance(payload, list):
        raise ValueError(f"validation metrics must be a JSON list: {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a unified validation candidate.")
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("candidate-decision.json"),
    )
    args = parser.parse_args(argv)
    decision = evaluate_candidate(
        _load_rows(args.baseline_metrics),
        _load_rows(args.candidate_metrics),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(
            decision,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
