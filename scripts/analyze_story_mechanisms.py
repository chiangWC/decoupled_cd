from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.q_matrix import normalize_concept_sequence


EPS = 1.0e-7
MIN_ITEM_RESPONSES = 20
BETA_STRENGTH = 10.0


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    data_dir: Path


@dataclass(frozen=True)
class PredictionSpec:
    name: str
    data_dir: Path
    full_path: Path
    control_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit coverage, history calibration, and exact-Q aliasing."
    )
    parser.add_argument(
        "--coverage-dataset", action="append", nargs=2, default=[], metavar=("NAME", "DIR")
    )
    parser.add_argument(
        "--mechanism-dataset", action="append", nargs=2, default=[], metavar=("NAME", "DIR")
    )
    parser.add_argument(
        "--history-link", action="append", nargs=4, default=[],
        metavar=("NAME", "DIR", "FULL", "CONTROL"),
    )
    parser.add_argument(
        "--requirement-link", action="append", nargs=4, default=[],
        metavar=("NAME", "DIR", "FULL", "CONTROL"),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, frozenset[str]]]:
    q_frame = pd.read_csv(data_dir / "Q_matrix.csv")
    q_sets: dict[str, set[str]] = defaultdict(set)
    for row in q_frame.itertuples(index=False):
        q_sets[str(row.exer_id)].update(normalize_concept_sequence(row.cpt_seq))
    q_map = {item: frozenset(concepts) for item, concepts in q_sets.items()}
    if not q_map or any(not concepts for concepts in q_map.values()):
        raise ValueError(f"{data_dir}: invalid Q-matrix")
    frames = []
    for split in ("train", "valid", "test"):
        frame = pd.read_csv(data_dir / f"{split}.csv")
        frame = frame[["stu_id", "exer_id", "label"]].copy()
        frame["stu_id"] = frame["stu_id"].astype(str)
        frame["exer_id"] = frame["exer_id"].astype(str)
        frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
        if not frame["label"].isin([0, 1]).all():
            raise ValueError(f"{data_dir}/{split}: non-binary label")
        missing = set(frame["exer_id"]).difference(q_map)
        if missing:
            raise ValueError(f"{data_dir}/{split}: exercise missing from Q")
        frames.append(frame)
    return frames[0], frames[1], frames[2], q_map


def student_seen(train: pd.DataFrame, q_map: dict[str, frozenset[str]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in train.itertuples(index=False):
        result[str(row.stu_id)].update(q_map[str(row.exer_id)])
    return result


def coverage_bucket(
    train: pd.DataFrame, target: pd.DataFrame, q_map: dict[str, frozenset[str]]
) -> np.ndarray:
    seen = student_seen(train, q_map)
    result = np.empty(len(target), dtype=object)
    for index, row in enumerate(target.itertuples(index=False)):
        required = q_map[str(row.exer_id)]
        overlap = len(required.intersection(seen.get(str(row.stu_id), set())))
        result[index] = "zero" if overlap == 0 else "full" if overlap == len(required) else "partial"
    return result


def coverage_audit(spec: DatasetSpec) -> list[dict[str, Any]]:
    train, valid, test, q_map = read_dataset(spec.data_dir)
    full = pd.concat([train, valid, test], ignore_index=True)
    pair_counts: Counter[tuple[str, str]] = Counter()
    for row in full.itertuples(index=False):
        for concept in q_map[str(row.exer_id)]:
            pair_counts[(str(row.stu_id), concept)] += 1
    train_pairs = set(zip(train["stu_id"], train["exer_id"]))
    train_students = set(train["stu_id"])
    train_items = set(train["exer_id"])
    train_concepts = set().union(*(q_map[item] for item in train_items))
    reports = []
    for split, target in (("valid", valid), ("test", test)):
        buckets = coverage_bucket(train, target, q_map)
        counts = Counter(buckets)
        any_singleton = all_singleton = known_item = known_concepts = known_student = same_item = 0
        for row in target.itertuples(index=False):
            student, item = str(row.stu_id), str(row.exer_id)
            concepts = q_map[item]
            flags = [pair_counts[(student, concept)] == 1 for concept in concepts]
            any_singleton += int(any(flags))
            all_singleton += int(all(flags))
            known_student += int(student in train_students)
            same_item += int((student, item) in train_pairs)
            known_item += int(item in train_items)
            known_concepts += int(concepts.issubset(train_concepts))
        rows = len(target)
        reports.append({
            "dataset": spec.name, "split": split, "rows": rows,
            "zero_rows": int(counts["zero"]), "zero_rate": counts["zero"] / rows,
            "partial_rows": int(counts["partial"]), "partial_rate": counts["partial"] / rows,
            "full_rows": int(counts["full"]), "full_rate": counts["full"] / rows,
            "incomplete_rate": (counts["zero"] + counts["partial"]) / rows,
            "student_seen_rate": known_student / rows,
            "student_item_seen_rate": same_item / rows,
            "global_item_seen_rate": known_item / rows,
            "global_concepts_seen_rate": known_concepts / rows,
            "any_student_concept_singleton_rate": any_singleton / rows,
            "all_student_concepts_singleton_rate": all_singleton / rows,
        })
    return reports


def student_history_features(
    train: pd.DataFrame, q_map: dict[str, frozenset[str]]
) -> pd.DataFrame:
    global_rate = float(train["label"].mean())
    item = train.groupby("exer_id")["label"].agg(item_correct="sum", item_count="count")
    student_item = train.groupby(["stu_id", "exer_id"])["label"].agg(
        student_correct="sum", student_count="count"
    ).reset_index().join(item, on="exer_id")
    other_correct = student_item["item_correct"] - student_item["student_correct"]
    other_count = student_item["item_count"] - student_item["student_count"]
    student_item["loo_ease"] = (
        other_correct + BETA_STRENGTH * global_rate
    ) / (other_count + BETA_STRENGTH)
    student_item["weighted_ease"] = student_item["loo_ease"] * student_item["student_count"]
    result = student_item.groupby("stu_id").agg(
        history_count=("student_count", "sum"), history_correct=("student_correct", "sum"),
        weighted_ease=("weighted_ease", "sum")
    ).reset_index()
    result["history_accuracy"] = result["history_correct"] / result["history_count"]
    result["mean_history_item_ease"] = result["weighted_ease"] / result["history_count"]
    result["calibration_residual"] = result["history_accuracy"] - result["mean_history_item_ease"]
    result["absolute_calibration_mismatch"] = result["calibration_residual"].abs()
    seen = student_seen(train, q_map)
    concepts = set().union(*q_map.values())
    result["concept_coverage"] = result["stu_id"].map(lambda s: len(seen[str(s)]) / len(concepts))
    result["log_history_count"] = np.log1p(result["history_count"])
    return result


def item_ease(train: pd.DataFrame) -> tuple[dict[str, float], float]:
    global_rate = float(train["label"].mean())
    stats = train.groupby("exer_id")["label"].agg(correct="sum", count="count")
    stats["ease"] = (stats["correct"] + BETA_STRENGTH * global_rate) / (
        stats["count"] + BETA_STRENGTH
    )
    return stats["ease"].to_dict(), global_rate


def standardized(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].to_numpy(float)
    scale = values.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (values - values.mean(axis=0)) / scale


def ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    design = np.column_stack([np.ones(len(x)), x])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ coef
    denominator = np.square(y - y.mean()).sum()
    r2 = 1.0 - np.square(y - fitted).sum() / denominator if denominator > 0 else 0.0
    return coef, float(r2)


def difficulty_audit(spec: DatasetSpec, bootstrap: int, seed: int) -> dict[str, Any]:
    train, valid, _, q_map = read_dataset(spec.data_dir)
    history = student_history_features(train, q_map)
    ease, global_rate = item_ease(train)
    target = valid.copy()
    target["target_ease"] = target["exer_id"].map(ease).fillna(global_rate)
    target["normalized_outcome"] = target["label"] - target["target_ease"]
    outcome = target.groupby("stu_id").agg(
        target_count=("label", "size"), future_residual=("normalized_outcome", "mean")
    ).reset_index()
    frame = history.merge(outcome, on="stu_id")
    frame = frame.loc[(frame["history_count"] >= 10) & (frame["target_count"] >= 3)].reset_index(drop=True)
    if len(frame) < 30:
        raise ValueError(f"{spec.name}: insufficient eligible students")
    base_columns = ["history_accuracy", "log_history_count", "concept_coverage"]
    full_columns = ["history_accuracy", "mean_history_item_ease", "log_history_count", "concept_coverage"]
    y = frame["future_residual"].to_numpy(float)
    base_x, full_x = standardized(frame, base_columns), standardized(frame, full_columns)
    coef, full_r2 = ols(y, full_x)
    _, base_r2 = ols(y, base_x)
    rng = np.random.default_rng(seed)
    coef_samples, r2_samples = np.empty(bootstrap), np.empty(bootstrap)
    for index in range(bootstrap):
        sample = rng.integers(0, len(frame), size=len(frame))
        sample_coef, sample_full_r2 = ols(y[sample], full_x[sample])
        _, sample_base_r2 = ols(y[sample], base_x[sample])
        coef_samples[index] = sample_coef[2]
        r2_samples[index] = sample_full_r2 - sample_base_r2
    return {
        "dataset": spec.name, "eligible_students": len(frame),
        "history_ease_coefficient": float(coef[2]),
        "history_ease_coefficient_ci_low": float(np.quantile(coef_samples, .025)),
        "history_ease_coefficient_ci_high": float(np.quantile(coef_samples, .975)),
        "probability_coefficient_negative": float(np.mean(coef_samples < 0)),
        "base_r2": base_r2, "difficulty_augmented_r2": full_r2,
        "incremental_r2": full_r2 - base_r2,
        "incremental_r2_ci_low": float(np.quantile(r2_samples, .025)),
        "incremental_r2_ci_high": float(np.quantile(r2_samples, .975)),
    }


def q_group_statistics(
    train: pd.DataFrame, q_map: dict[str, frozenset[str]]
) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
    signature = {item: "|".join(sorted(concepts)) for item, concepts in q_map.items()}
    items = train.groupby("exer_id")["label"].agg(correct="sum", responses="count").reset_index()
    items["signature"] = items["exer_id"].map(signature)
    items = items.loc[items["responses"] >= MIN_ITEM_RESPONSES].copy()
    eligible = items.groupby("signature")["exer_id"].nunique()
    items = items.loc[items["signature"].isin(eligible.index[eligible >= 2])].copy()
    if items.empty:
        return pd.DataFrame(), signature, items
    items["group_correct"] = items.groupby("signature")["correct"].transform("sum")
    items["group_responses"] = items.groupby("signature")["responses"].transform("sum")
    items["q_prob"] = items["group_correct"] / items["group_responses"]
    items["item_prob"] = (items["correct"] + BETA_STRENGTH * items["q_prob"]) / (
        items["responses"] + BETA_STRENGTH
    )
    items["sq_dev"] = np.square(items["item_prob"] - items["q_prob"])
    items["noise"] = items["q_prob"] * (1 - items["q_prob"]) / (items["responses"] + BETA_STRENGTH)
    groups = items.groupby("signature").agg(
        eligible_items=("exer_id", "nunique"), train_responses=("responses", "sum"),
        min_rate=("item_prob", "min"), max_rate=("item_prob", "max"),
        observed_variance=("sq_dev", "mean"), expected_noise=("noise", "mean")
    ).reset_index()
    groups["item_rate_range"] = groups["max_rate"] - groups["min_rate"]
    groups["excess_variance"] = (groups["observed_variance"] - groups["expected_noise"]).clip(lower=0)
    return groups, signature, items


def same_q_audit(spec: DatasetSpec, bootstrap: int, seed: int, group_output: Path) -> dict[str, Any]:
    train, valid, _, q_map = read_dataset(spec.data_dir)
    groups, signature, items = q_group_statistics(train, q_map)
    if groups.empty:
        return {"dataset": spec.name, "status": "not_applicable", "eligible_q_groups": 0}
    groups.to_csv(group_output, index=False)
    evaluation = valid.copy()
    evaluation["signature"] = evaluation["exer_id"].map(signature)
    evaluation = evaluation.merge(items[["exer_id", "signature", "q_prob", "item_prob"]],
                                  on=["exer_id", "signature"], how="inner")
    labels = evaluation["label"].to_numpy(int)
    q_prob = np.clip(evaluation["q_prob"].to_numpy(float), EPS, 1 - EPS)
    item_prob = np.clip(evaluation["item_prob"].to_numpy(float), EPS, 1 - EPS)
    evaluation["brier_advantage"] = np.square(labels - q_prob) - np.square(labels - item_prob)
    evaluation["log_advantage"] = (
        -(labels * np.log(q_prob) + (1 - labels) * np.log(1 - q_prob))
        + labels * np.log(item_prob) + (1 - labels) * np.log(1 - item_prob)
    )
    unit = evaluation.groupby("signature")[["brier_advantage", "log_advantage"]].mean().to_numpy()
    rng = np.random.default_rng(seed)
    brier_samples, log_samples = np.empty(bootstrap), np.empty(bootstrap)
    for index in range(bootstrap):
        sample = unit[rng.integers(0, len(unit), size=len(unit))]
        brier_samples[index], log_samples[index] = sample.mean(axis=0)
    return {
        "dataset": spec.name, "status": "ok", "eligible_q_groups": len(groups),
        "eligible_items": int(groups["eligible_items"].sum()), "validation_rows": len(evaluation),
        "median_item_rate_range": float(groups["item_rate_range"].median()),
        "groups_range_at_least_0_1_rate": float(np.mean(groups["item_rate_range"] >= .1)),
        "groups_positive_excess_variance_rate": float(np.mean(groups["excess_variance"] > 0)),
        "q_only_auc": float(roc_auc_score(labels, q_prob)),
        "item_aware_auc": float(roc_auc_score(labels, item_prob)),
        "item_aware_auc_delta": float(roc_auc_score(labels, item_prob) - roc_auc_score(labels, q_prob)),
        "brier_advantage": float(unit[:, 0].mean()),
        "row_weighted_brier_advantage": float(evaluation["brier_advantage"].mean()),
        "brier_advantage_ci_low": float(np.quantile(brier_samples, .025)),
        "brier_advantage_ci_high": float(np.quantile(brier_samples, .975)),
        "log_loss_advantage": float(unit[:, 1].mean()),
        "row_weighted_log_loss_advantage": float(evaluation["log_advantage"].mean()),
        "log_loss_advantage_ci_low": float(np.quantile(log_samples, .025)),
        "log_loss_advantage_ci_high": float(np.quantile(log_samples, .975)),
        "q_only_log_loss": float(log_loss(labels, q_prob)),
        "item_aware_log_loss": float(log_loss(labels, item_prob)),
    }


def predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)[["stu_id", "exer_id", "label", "prob"]].copy()
    frame["stu_id"], frame["exer_id"] = frame["stu_id"].astype(str), frame["exer_id"].astype(str)
    return frame


def aligned_predictions(full_path: Path, control_path: Path) -> pd.DataFrame:
    full, control = predictions(full_path), predictions(control_path)
    if len(full) != len(control):
        raise ValueError("prediction row count mismatch")
    for column in ("stu_id", "exer_id", "label"):
        if not np.array_equal(full[column], control[column]):
            raise ValueError(f"prediction mismatch on {column}")
    return full.rename(columns={"prob": "full_prob"}).assign(control_prob=control["prob"].to_numpy())


def auc(frame: pd.DataFrame, column: str) -> float:
    return float(roc_auc_score(frame["label"], frame[column])) if frame["label"].nunique() == 2 else float("nan")


def bootstrap_difference(low: np.ndarray, high: np.ndarray, bootstrap: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(bootstrap)
    for index in range(bootstrap):
        samples[index] = rng.choice(high, len(high), replace=True).mean() - rng.choice(low, len(low), replace=True).mean()
    return float(high.mean() - low.mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def history_link(spec: PredictionSpec, bootstrap: int, seed: int) -> dict[str, Any]:
    train, valid, _, q_map = read_dataset(spec.data_dir)
    frame = aligned_predictions(spec.full_path, spec.control_path)
    if not all(np.array_equal(frame[column], valid[column]) for column in ("stu_id", "exer_id", "label")):
        raise ValueError(f"{spec.name}: predictions do not match valid")
    frame["bucket"] = coverage_bucket(train, valid, q_map)
    frame = frame.loc[frame["bucket"].eq("zero")].copy()
    frame["brier_advantage"] = np.square(frame["label"] - frame["control_prob"]) - np.square(frame["label"] - frame["full_prob"])
    frame["prediction_shift"] = frame["full_prob"] - frame["control_prob"]
    unit = frame.groupby("stu_id").agg(
        target_rows=("label", "size"),
        brier_advantage=("brier_advantage", "mean"),
        prediction_shift=("prediction_shift", "mean"),
    ).reset_index()
    unit = unit.merge(student_history_features(train, q_map), on="stu_id")
    unit = unit.loc[unit["target_rows"] >= 2].copy()
    unit["quartile"] = pd.qcut(unit["absolute_calibration_mismatch"].rank(method="first"), 4, labels=False) + 1
    low, high = unit.loc[unit["quartile"].eq(1)], unit.loc[unit["quartile"].eq(4)]
    difference, ci_low, ci_high = bootstrap_difference(low["brier_advantage"].to_numpy(), high["brier_advantage"].to_numpy(), bootstrap, seed)
    unit["signed_quartile"] = pd.qcut(unit["calibration_residual"].rank(method="first"), 4, labels=False) + 1
    signed_low = unit.loc[unit["signed_quartile"].eq(1), "prediction_shift"].to_numpy()
    signed_high = unit.loc[unit["signed_quartile"].eq(4), "prediction_shift"].to_numpy()
    shift_difference, shift_ci_low, shift_ci_high = bootstrap_difference(
        signed_low, signed_high, bootstrap, seed + 1000
    )
    low_rows, high_rows = frame.loc[frame["stu_id"].isin(low["stu_id"])], frame.loc[frame["stu_id"].isin(high["stu_id"])]
    return {
        "dataset": spec.name, "target_rows": len(frame), "eligible_students": len(unit),
        "spearman_mismatch_vs_brier_gain": float(unit[["absolute_calibration_mismatch", "brier_advantage"]].corr(method="spearman").iloc[0, 1]),
        "spearman_calibration_residual_vs_prediction_shift": float(
            unit[["calibration_residual", "prediction_shift"]].corr(method="spearman").iloc[0, 1]
        ),
        "low_residual_prediction_shift": float(signed_low.mean()),
        "high_residual_prediction_shift": float(signed_high.mean()),
        "high_minus_low_prediction_shift": shift_difference,
        "high_minus_low_prediction_shift_ci_low": shift_ci_low,
        "high_minus_low_prediction_shift_ci_high": shift_ci_high,
        "low_mismatch_brier_advantage": float(low["brier_advantage"].mean()),
        "high_mismatch_brier_advantage": float(high["brier_advantage"].mean()),
        "high_minus_low_brier_advantage": difference, "high_minus_low_ci_low": ci_low, "high_minus_low_ci_high": ci_high,
        "low_mismatch_auc_delta": auc(low_rows, "full_prob") - auc(low_rows, "control_prob"),
        "high_mismatch_auc_delta": auc(high_rows, "full_prob") - auc(high_rows, "control_prob"),
    }


def requirement_link(spec: PredictionSpec, bootstrap: int, seed: int) -> dict[str, Any]:
    train, valid, _, q_map = read_dataset(spec.data_dir)
    frame = aligned_predictions(spec.full_path, spec.control_path)
    if not all(np.array_equal(frame[column], valid[column]) for column in ("stu_id", "exer_id", "label")):
        raise ValueError(f"{spec.name}: predictions do not match valid")
    groups, signature, _ = q_group_statistics(train, q_map)
    if groups.empty:
        return {"dataset": spec.name, "status": "not_applicable"}
    frame["bucket"], frame["signature"] = coverage_bucket(train, valid, q_map), frame["exer_id"].map(signature)
    frame = frame.loc[frame["bucket"].eq("zero")].merge(groups[["signature", "excess_variance"]], on="signature")
    group_rank = groups[["signature", "excess_variance"]].copy()
    group_rank["quartile"] = pd.qcut(group_rank["excess_variance"].rank(method="first"), 4, labels=False) + 1
    frame = frame.merge(group_rank[["signature", "quartile"]], on="signature")
    frame["brier_advantage"] = np.square(frame["label"] - frame["control_prob"]) - np.square(frame["label"] - frame["full_prob"])
    units = frame.groupby(["signature", "quartile"])["brier_advantage"].mean().reset_index()
    low, high = units.loc[units["quartile"].eq(1)], units.loc[units["quartile"].eq(4)]
    difference, ci_low, ci_high = bootstrap_difference(low["brier_advantage"].to_numpy(), high["brier_advantage"].to_numpy(), bootstrap, seed)
    low_rows, high_rows = frame.loc[frame["quartile"].eq(1)], frame.loc[frame["quartile"].eq(4)]
    return {
        "dataset": spec.name, "status": "ok", "eligible_target_rows": len(frame), "eligible_q_groups": frame["signature"].nunique(),
        "low_heterogeneity_brier_advantage": float(low["brier_advantage"].mean()),
        "high_heterogeneity_brier_advantage": float(high["brier_advantage"].mean()),
        "high_minus_low_brier_advantage": difference, "high_minus_low_ci_low": ci_low, "high_minus_low_ci_high": ci_high,
        "low_heterogeneity_auc_delta": auc(low_rows, "full_prob") - auc(low_rows, "control_prob"),
        "high_heterogeneity_auc_delta": auc(high_rows, "full_prob") - auc(high_rows, "control_prob"),
    }


def write_outputs(output_dir: Path, stem: str, reports: list[dict[str, Any]]) -> None:
    pd.DataFrame(reports).to_csv(output_dir / f"{stem}.csv", index=False)
    with (output_dir / f"{stem}.json").open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2, ensure_ascii=False, allow_nan=False)


def plot_forest(frame: pd.DataFrame, value: str, low: str, high: str, label: str, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    frame = frame.dropna(subset=[value, low, high]).reset_index(drop=True)
    if frame.empty:
        return
    values = frame[value].to_numpy(float)
    errors = np.vstack([values - frame[low].to_numpy(float), frame[high].to_numpy(float) - values])
    figure, axis = plt.subplots(figsize=(7.5, 1.8 + .6 * len(frame)))
    axis.errorbar(values, np.arange(len(frame)), xerr=errors, fmt="o", capsize=3)
    axis.axvline(0, color="gray", linestyle="--", linewidth=1)
    axis.set_yticks(np.arange(len(frame)), frame["dataset"])
    axis.set_xlabel(label)
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage_specs = [DatasetSpec(name, Path(path)) for name, path in args.coverage_dataset]
    mechanism_specs = [DatasetSpec(name, Path(path)) for name, path in args.mechanism_dataset]
    history_specs = [PredictionSpec(name, Path(path), Path(full), Path(control)) for name, path, full, control in args.history_link]
    requirement_specs = [PredictionSpec(name, Path(path), Path(full), Path(control)) for name, path, full, control in args.requirement_link]
    coverage = [report for spec in coverage_specs for report in coverage_audit(spec)]
    difficulty = [difficulty_audit(spec, args.bootstrap, args.seed + index) for index, spec in enumerate(mechanism_specs)]
    same_q = [same_q_audit(spec, args.bootstrap, args.seed + index, args.output_dir / f"same_q_groups_{spec.name}.csv") for index, spec in enumerate(mechanism_specs)]
    history_gain = [history_link(spec, args.bootstrap, args.seed + index) for index, spec in enumerate(history_specs)]
    requirement_gain = [requirement_link(spec, args.bootstrap, args.seed + index) for index, spec in enumerate(requirement_specs)]
    write_outputs(args.output_dir, "coverage_prevalence", coverage)
    write_outputs(args.output_dir, "history_difficulty", difficulty)
    write_outputs(args.output_dir, "same_q_aliasing", same_q)
    write_outputs(args.output_dir, "history_module_gain", history_gain)
    write_outputs(args.output_dir, "requirement_module_gain", requirement_gain)
    plot_forest(pd.DataFrame(difficulty), "history_ease_coefficient", "history_ease_coefficient_ci_low", "history_ease_coefficient_ci_high", "Partial effect of easier history on item-normalized future outcome", args.output_dir / "history_difficulty_forest.svg")
    plot_forest(pd.DataFrame(same_q), "brier_advantage", "brier_advantage_ci_low", "brier_advantage_ci_high", "Brier advantage: item-aware prior over exact-Q-only prior", args.output_dir / "same_q_brier_forest.svg")
    manifest = {
        "schema_version": 1, "seed": args.seed, "bootstrap": args.bootstrap,
        "min_item_responses": MIN_ITEM_RESPONSES, "beta_strength": BETA_STRENGTH,
        "notes": ["All statistics use train rows only.", "Item ease is leave-one-student-out in the history audit.", "Bootstrap units are students or exact-Q groups; this is not multi-seed training."],
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
