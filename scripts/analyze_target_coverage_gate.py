from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.q_matrix import normalize_concept_sequence


EPSILON = 1.0e-7
ALIGNMENT_COLUMNS = ("stu_id", "exer_id", "label")
OUTCOMES = (
    "external_log_loss",
    "external_brier",
    "full_log_loss",
    "full_log_loss_advantage",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    data_dir: Path
    full_predictions: Path
    external_predictions: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only Gate A for target-local concept coverage. "
            "The implementation deliberately never opens test.csv."
        )
    )
    parser.add_argument(
        "--dataset-spec",
        action="append",
        nargs=4,
        required=True,
        metavar=("NAME", "DATA_DIR", "FULL_VALID_CSV", "EXTERNAL_VALID_CSV"),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_interactions(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    required = {"stu_id", "exer_id", "label"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}.")
    result = frame.copy()
    result["stu_id"] = result["stu_id"].astype(str)
    result["exer_id"] = result["exer_id"].astype(str)
    result["label"] = pd.to_numeric(result["label"], errors="raise").astype(int)
    if not result["label"].isin([0, 1]).all():
        raise ValueError(f"{source}: labels must be binary.")
    return result


def q_map_from_frame(frame: pd.DataFrame, source: Path) -> dict[str, frozenset[str]]:
    if not {"exer_id", "cpt_seq"}.issubset(frame.columns):
        raise ValueError(f"{source}: Q metadata requires exer_id and cpt_seq.")
    mapping: dict[str, set[str]] = defaultdict(set)
    for row in frame.itertuples(index=False):
        mapping[str(row.exer_id)].update(normalize_concept_sequence(row.cpt_seq))
    result = {item: frozenset(concepts) for item, concepts in mapping.items() if concepts}
    if not result:
        raise ValueError(f"{source}: empty Q mapping.")
    return result


def load_train_valid(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, frozenset[str]], dict[str, str]]:
    """Load train/valid and Q metadata only; test.csv is never touched."""
    train_path, valid_path = data_dir / "train.csv", data_dir / "valid.csv"
    train_raw, valid_raw = pd.read_csv(train_path), pd.read_csv(valid_path)
    train = normalize_interactions(train_raw, train_path)
    valid = normalize_interactions(valid_raw, valid_path)
    q_path = data_dir / "Q_matrix.csv"
    hashes = {
        "train.csv": sha256_file(train_path),
        "valid.csv": sha256_file(valid_path),
    }
    if q_path.exists():
        q_map = q_map_from_frame(pd.read_csv(q_path), q_path)
        hashes["Q_matrix.csv"] = sha256_file(q_path)
    else:
        # The admitted Junyi protocol is one-exercise/one-concept and has no
        # separate Q file. Only train/valid metadata, never labels, define Q.
        if "cpt_seq" not in train_raw or "cpt_seq" not in valid_raw:
            raise ValueError(f"{data_dir}: no Q_matrix.csv or cpt_seq metadata.")
        metadata = pd.concat(
            [
                train_raw[["exer_id", "cpt_seq"]],
                valid_raw[["exer_id", "cpt_seq"]],
            ],
            ignore_index=True,
        )
        q_map = q_map_from_frame(metadata, data_dir)
        if any(len(concepts) != 1 for concepts in q_map.values()):
            raise ValueError(f"{data_dir}: inferred Q is not one-to-one.")
        hashes["Q_matrix.csv"] = "inferred_from_train_and_valid_cpt_seq"
    missing = set(train["exer_id"]).union(valid["exer_id"]).difference(q_map)
    if missing:
        raise ValueError(f"{data_dir}: items missing from Q: {sorted(missing)[:5]}.")
    return train, valid, q_map, hashes


def probabilities(frame: pd.DataFrame, source: Path) -> np.ndarray:
    if "prob" not in frame:
        raise ValueError(f"{source}: missing prob.")
    values = pd.to_numeric(frame["prob"], errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"{source}: invalid probabilities.")
    return values


def load_aligned_predictions(
    valid: pd.DataFrame, full_path: Path, external_path: Path
) -> pd.DataFrame:
    full = normalize_interactions(pd.read_csv(full_path), full_path)
    external = normalize_interactions(pd.read_csv(external_path), external_path)
    if len(valid) != len(full) or len(valid) != len(external):
        raise ValueError(
            f"row mismatch: valid={len(valid)}, Full={len(full)}, external={len(external)}"
        )
    for name, frame in (("Full", full), ("external", external)):
        for column in ALIGNMENT_COLUMNS:
            left, right = frame[column].to_numpy(), valid[column].to_numpy()
            if not np.array_equal(left, right):
                mismatch = np.flatnonzero(left != right)
                row = int(mismatch[0]) if len(mismatch) else -1
                raise ValueError(f"{name} {column} mismatch at row {row}.")
    result = valid[list(ALIGNMENT_COLUMNS)].copy()
    result["full_prob"] = probabilities(full, full_path)
    result["external_prob"] = probabilities(external, external_path)
    return result


def build_analysis_frame(
    train: pd.DataFrame,
    aligned: pd.DataFrame,
    q_map: dict[str, frozenset[str]],
) -> pd.DataFrame:
    seen: dict[str, set[str]] = defaultdict(set)
    for row in train.itertuples(index=False):
        seen[str(row.stu_id)].update(q_map[str(row.exer_id)])
    unknown = set(aligned["stu_id"]).difference(train["stu_id"])
    if unknown:
        raise ValueError(f"transductive Gate A found unseen students: {sorted(unknown)[:5]}")
    concepts = set().union(*q_map.values())
    target, target_count, target_seen, global_coverage = [], [], [], []
    for row in aligned.itertuples(index=False):
        required, observed = q_map[str(row.exer_id)], seen[str(row.stu_id)]
        overlap = len(required.intersection(observed))
        target.append(overlap / len(required))
        target_count.append(len(required))
        target_seen.append(overlap)
        global_coverage.append(len(observed) / len(concepts))
    frame = aligned.copy()
    frame["target_concept_count"] = target_count
    frame["target_seen_concept_count"] = target_seen
    frame["target_coverage"] = target
    frame["uncovered_fraction"] = 1.0 - frame["target_coverage"]
    frame["coverage_bucket"] = np.select(
        [
            frame["target_coverage"].eq(0),
            frame["target_coverage"].lt(0.5),
            frame["target_coverage"].lt(1),
        ],
        ["zero", "low", "partial"],
        default="full",
    )
    frame["student_global_coverage"] = global_coverage
    student_stats = train.groupby("stu_id")["label"].agg(
        student_history_count="size", student_history_accuracy="mean"
    )
    item_stats = train.groupby("exer_id")["label"].agg(
        item_train_count="size", item_train_accuracy="mean"
    )
    frame = frame.join(student_stats, on="stu_id").join(item_stats, on="exer_id")
    per_student = frame.groupby("stu_id")["student_global_coverage"].first()
    rank = per_student.rank(method="average", pct=True)
    quantile = np.minimum(np.floor(rank.to_numpy() * 5).astype(int), 4)
    bin_map = {student: f"Q{value + 1}" for student, value in zip(rank.index, quantile)}
    frame["global_coverage_bin"] = frame["stu_id"].map(bin_map)
    labels = frame["label"].to_numpy(float)
    for model in ("external", "full"):
        prob = frame[f"{model}_prob"].to_numpy(float)
        clipped = np.clip(prob, EPSILON, 1 - EPSILON)
        frame[f"{model}_log_loss"] = -(
            labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)
        )
        frame[f"{model}_brier"] = np.square(labels - prob)
    frame["full_log_loss_advantage"] = (
        frame["external_log_loss"] - frame["full_log_loss"]
    )
    return frame


def safe_auc(frame: pd.DataFrame, column: str) -> float | None:
    return (
        float(roc_auc_score(frame["label"], frame[column]))
        if frame["label"].nunique() == 2
        else None
    )


def grouped_metrics(frame: pd.DataFrame, columns: Iterable[str]) -> list[dict[str, Any]]:
    names = list(columns)
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(names, sort=True, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(names, keys))
        row.update(
            {
                "rows": len(group),
                "students": group["stu_id"].nunique(),
                "items": group["exer_id"].nunique(),
                "label_rate": group["label"].mean(),
                "mean_target_coverage": group["target_coverage"].mean(),
                "external_auc": safe_auc(group, "external_prob"),
                "full_auc": safe_auc(group, "full_prob"),
                "external_log_loss": group["external_log_loss"].mean(),
                "full_log_loss": group["full_log_loss"].mean(),
                "external_brier": group["external_brier"].mean(),
                "full_brier": group["full_brier"].mean(),
                "full_log_loss_advantage": group["full_log_loss_advantage"].mean(),
            }
        )
        rows.append(row)
    return rows


def group_means(values: np.ndarray, codes: np.ndarray) -> np.ndarray:
    groups = int(codes.max()) + 1
    counts = np.bincount(codes, minlength=groups).astype(float)
    means = np.column_stack(
        [
            np.bincount(codes, weights=values[:, index], minlength=groups)
            for index in range(values.shape[1])
        ]
    )
    means[counts > 0] /= counts[counts > 0, None]
    return means


def residualize_two_way(
    values: np.ndarray,
    student_codes: np.ndarray,
    item_codes: np.ndarray,
    tolerance: float = 1.0e-10,
    max_iterations: int = 10_000,
) -> tuple[np.ndarray, int]:
    residual = np.atleast_2d(values).astype(float, copy=True)
    if residual.shape[0] != len(student_codes):
        residual = residual.T
    for iteration in range(1, max_iterations + 1):
        student_mean = group_means(residual, student_codes)
        residual -= student_mean[student_codes]
        item_mean = group_means(residual, item_codes)
        residual -= item_mean[item_codes]
        update = max(np.abs(student_mean).max(), np.abs(item_mean).max())
        if update < tolerance:
            return residual, iteration
    raise RuntimeError(
        "two-way fixed-effect residualization did not converge: "
        f"iterations={max_iterations}, final_update={update:.3e}"
    )


def two_way_fixed_effects(
    frame: pd.DataFrame,
    outcomes: Iterable[str],
    bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    outcome_names = list(outcomes)
    student_codes, students = pd.factorize(frame["stu_id"], sort=True)
    item_codes, items = pd.factorize(frame["exer_id"], sort=True)
    values = np.column_stack(
        [frame["uncovered_fraction"].to_numpy(float)]
        + [frame[name].to_numpy(float) for name in outcome_names]
    )
    residual, iterations = residualize_two_way(values, student_codes, item_codes)
    x = residual[:, 0]
    denominator = float(x @ x)
    base = {
        "rows": len(frame),
        "students": len(students),
        "items": len(items),
        "within_x_sum_squares": denominator,
        "residualization_iterations": iterations,
    }
    if denominator < 1.0e-14:
        return [
            {
                **base,
                "outcome": outcome,
                "status": "not_identified",
                "reason": "coverage has no variation after student/item fixed effects",
            }
            for outcome in outcome_names
        ]
    cluster_den = np.bincount(
        student_codes, weights=np.square(x), minlength=len(students)
    )
    rng = np.random.default_rng(seed)
    multiplicity = rng.multinomial(
        len(students),
        np.full(len(students), 1 / len(students)),
        size=bootstrap,
    )
    boot_den = multiplicity @ cluster_den
    results = []
    for offset, outcome in enumerate(outcome_names, start=1):
        y = residual[:, offset]
        beta = float((x @ y) / denominator)
        cluster_num = np.bincount(
            student_codes, weights=x * y, minlength=len(students)
        )
        samples = (multiplicity @ cluster_num) / boot_den
        results.append(
            {
                **base,
                "outcome": outcome,
                "status": "ok",
                "estimand": "change when target coverage moves from 1 to 0",
                "beta_uncovered_fraction": beta,
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "probability_positive": float(np.mean(samples > 0)),
                "bootstrap_replicates": bootstrap,
                "bootstrap_unit": "student",
                "fixed_effects": ["student", "exercise"],
            }
        )
    return results


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            json_safe(payload),
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def analyze_dataset(
    spec: DatasetSpec, bootstrap: int, seed: int, output_dir: Path
) -> dict[str, Any]:
    train, valid, q_map, data_hashes = load_train_valid(spec.data_dir)
    aligned = load_aligned_predictions(
        valid, spec.full_predictions, spec.external_predictions
    )
    frame = build_analysis_frame(train, aligned, q_map)
    frame.to_csv(output_dir / f"{spec.name}_validation_enriched.csv", index=False)
    prevalence = grouped_metrics(frame, ["coverage_bucket"])
    performance = grouped_metrics(
        frame, ["global_coverage_bin", "coverage_bucket"]
    )
    overall_fe = two_way_fixed_effects(frame, OUTCOMES, bootstrap, seed)
    for row in overall_fe:
        row.update(dataset=spec.name, scope="overall")
    stratum_fe = []
    for index, (coverage_bin, group) in enumerate(
        frame.groupby("global_coverage_bin", sort=True)
    ):
        rows = two_way_fixed_effects(
            group, ["external_log_loss"], bootstrap, seed + 101 * (index + 1)
        )
        for row in rows:
            row.update(
                dataset=spec.name,
                scope=coverage_bin,
                global_coverage_min=group["student_global_coverage"].min(),
                global_coverage_max=group["student_global_coverage"].max(),
            )
        stratum_fe.extend(rows)
    effect = next(row for row in overall_fe if row["outcome"] == "external_log_loss")
    support = effect["status"] == "ok" and effect["ci_low"] > 0
    identified_strata = [row for row in stratum_fe if row["status"] == "ok"]
    return {
        "dataset": spec.name,
        "rows": len(frame),
        "students": frame["stu_id"].nunique(),
        "items": frame["exer_id"].nunique(),
        "target_coverage_unique": frame["target_coverage"].nunique(),
        "dataset_support": support,
        "external_log_loss_effect": effect,
        "identified_global_coverage_strata": len(identified_strata),
        "positive_global_coverage_strata": sum(
            row["beta_uncovered_fraction"] > 0 for row in identified_strata
        ),
        "data_hashes": data_hashes,
        "prediction_hashes": {
            "Full": sha256_file(spec.full_predictions),
            "external": sha256_file(spec.external_predictions),
        },
        "prevalence": prevalence,
        "stratified_performance": performance,
        "two_way_fixed_effects": overall_fe,
        "global_coverage_strata_fixed_effects": stratum_fe,
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive.")
    specs = [
        DatasetSpec(name, Path(data), Path(full), Path(external))
        for name, data, full, external in args.dataset_spec
    ]
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("dataset names must be unique.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = [
        analyze_dataset(
            spec,
            args.bootstrap,
            args.seed + 1009 * index,
            args.output_dir,
        )
        for index, spec in enumerate(specs)
    ]
    pd.DataFrame(
        [dict(dataset=report["dataset"], **row) for report in reports for row in report["prevalence"]]
    ).to_csv(args.output_dir / "coverage_prevalence.csv", index=False)
    pd.DataFrame(
        [
            dict(dataset=report["dataset"], **row)
            for report in reports
            for row in report["stratified_performance"]
        ]
    ).to_csv(
        args.output_dir / "global_coverage_stratified_performance.csv", index=False
    )
    pd.DataFrame(
        [row for report in reports for row in report["two_way_fixed_effects"]]
    ).to_csv(args.output_dir / "two_way_fixed_effects.csv", index=False)
    pd.DataFrame(
        [
            row
            for report in reports
            for row in report["global_coverage_strata_fixed_effects"]
        ]
    ).to_csv(
        args.output_dir / "global_coverage_strata_fixed_effects.csv", index=False
    )
    supporting = [report["dataset"] for report in reports if report["dataset_support"]]
    unidentified = [
        report["dataset"]
        for report in reports
        if report["external_log_loss_effect"]["status"] != "ok"
    ]
    summary = {
        "schema_version": 1,
        "analysis": "target-conditioned support gap / Gate A",
        "protocol": "standard validation only; train history defines coverage",
        "test_access": "test.csv is not opened by this script",
        "estimand": (
            "two-way student/exercise fixed-effect change in per-row loss "
            "when target-local coverage moves from 1 to 0"
        ),
        "bootstrap": {
            "replicates": args.bootstrap,
            "seed": args.seed,
            "unit": "student",
            "method": (
                "student multiplicity-weighted bootstrap approximation on "
                "full-sample two-way fixed-effect residualized sufficient statistics"
            ),
            "fixed_effect_projection": (
                "computed once on the full validation sample; "
                "not refitted per replicate"
            ),
        },
        "gate_rule": (
            "PASS iff external log-loss fixed-effect CI lower bound is >0 "
            "on at least three datasets"
        ),
        "supporting_datasets": supporting,
        "unidentified_datasets": unidentified,
        "gate_pass": len(supporting) >= 3,
        "datasets": reports,
    }
    write_json(args.output_dir / "gate_a_summary.json", summary)


if __name__ == "__main__":
    main()

