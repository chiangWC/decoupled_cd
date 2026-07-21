from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EVALUATION_SEED = 2024
BOOTSTRAP_REPLICATES = 2000
VARIANT_NAMES = ("full", "pooled", "capacity", "standard_ncd")
CONTROL_NAMES = VARIANT_NAMES[1:]
REQUIRED_DATASETS = ("ASSIST17", "MOOCRadar")
METRIC_NAMES = ("auc", "acc", "rmse", "brier", "ece")


@dataclass(frozen=True)
class EvaluationColumns:
    row_id: str = "row_id"
    student_id: str = "student_id"
    item_id: str = "item_id"
    q_pair_id: str = "q_pair_id"
    q_count: str = "q_count"
    eligible: str = "eligible"
    probability: str = "prob"
    label: str = "label"

    @property
    def identities(self) -> tuple[str, ...]:
        return (self.student_id, self.item_id, self.q_pair_id, self.q_count, self.eligible)


DEFAULT_COLUMNS = EvaluationColumns()
_FORBIDDEN_OUTCOMES = {"label", "outcome", "response", "correct", "is_correct"}


def _hash_values(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode())
        digest.update(b"\x1f")
    return digest.hexdigest()


def row_order_sha256(row_ids: Sequence[object]) -> str:
    return _hash_values(str(value) for value in row_ids)


def _unique_ids(frame: pd.DataFrame, name: str, column: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"{name} is missing outcome-free {column!r}.")
    values = frame[column].astype(str)
    if (values.str.len() == 0).any():
        raise ValueError(f"{name} contains an empty outcome-free row ID.")
    duplicate = values[values.duplicated(keep=False)]
    if not duplicate.empty:
        raise ValueError(f"{name} contains duplicate outcome-free row IDs: {sorted(set(duplicate))[:5]}")
    return values


def _normalize_metadata(frame: pd.DataFrame, columns: EvaluationColumns) -> pd.DataFrame:
    output = frame.copy()
    for column in (columns.student_id, columns.item_id, columns.q_pair_id):
        output[column] = output[column].astype(str)
        if (output[column].str.len() == 0).any():
            raise ValueError(f"{column} values must be non-empty.")
    q_count = pd.to_numeric(output[columns.q_count], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(q_count).all() or (q_count < 0).any() or not np.equal(q_count, np.floor(q_count)).all():
        raise ValueError("q_count must contain finite non-negative integers.")
    output[columns.q_count] = q_count.astype(np.int64)
    if pd.api.types.is_bool_dtype(output[columns.eligible].dtype):
        output[columns.eligible] = output[columns.eligible].astype(bool)
    else:
        eligible = pd.to_numeric(output[columns.eligible], errors="raise").to_numpy()
        if not np.isin(eligible, [0, 1]).all():
            raise ValueError("eligible must contain booleans or 0/1.")
        output[columns.eligible] = eligible.astype(bool)
    return output


def _probabilities(values: pd.Series, name: str) -> np.ndarray:
    result = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
        raise ValueError(f"{name} must be finite probabilities in [0, 1].")
    return result


def align_prediction_manifests(
    *, predictions: Mapping[str, pd.DataFrame], labels: pd.DataFrame,
    columns: EvaluationColumns = DEFAULT_COLUMNS,
) -> pd.DataFrame:
    """One-to-one join of four outcome-free manifests and separately opened labels."""
    if set(predictions) != set(VARIANT_NAMES):
        raise ValueError(f"Prediction variants must be exactly {list(VARIANT_NAMES)}.")
    full = predictions["full"]
    for variant, frame in predictions.items():
        forbidden = {str(name).lower() for name in frame.columns} & _FORBIDDEN_OUTCOMES
        if forbidden:
            raise ValueError(f"{variant} prediction manifest is not outcome-free: {sorted(forbidden)}")
        missing = set(columns.identities) - set(frame.columns)
        if missing:
            raise ValueError(f"{variant} prediction manifest is missing {sorted(missing)}")
    full_ids = _unique_ids(full, "full prediction manifest", columns.row_id)
    full_set = set(full_ids)
    output = full.loc[:, [columns.row_id, *columns.identities]].copy().reset_index(drop=True)
    output[columns.row_id] = full_ids.to_numpy()
    output = _normalize_metadata(output, columns)
    for variant in VARIANT_NAMES:
        frame = predictions[variant]
        ids = _unique_ids(frame, f"{variant} prediction manifest", columns.row_id)
        if set(ids) != full_set:
            raise ValueError(f"{variant} outcome-free row-ID set differs from Full.")
        aligned = frame.assign(_rid=ids.to_numpy()).set_index("_rid", drop=True).reindex(full_ids)
        metadata = _normalize_metadata(aligned.loc[:, list(columns.identities)], columns)
        for identity in columns.identities:
            if not np.array_equal(output[identity].to_numpy(), metadata[identity].to_numpy()):
                raise ValueError(f"{variant} disagrees on {identity!r}.")
        probability_column = columns.probability if columns.probability in frame else f"prob_{variant}"
        if probability_column not in frame:
            raise ValueError(f"{variant} predictions are missing probability.")
        output[f"prob_{variant}"] = _probabilities(aligned[probability_column], variant)
    label_ids = _unique_ids(labels, "label table", columns.row_id)
    if set(label_ids) != full_set:
        raise ValueError("Label outcome-free row-ID set differs from Full.")
    if columns.label not in labels:
        raise ValueError(f"Label table is missing {columns.label!r}.")
    aligned_labels = labels.assign(_rid=label_ids.to_numpy()).set_index("_rid", drop=True).reindex(full_ids)
    outcomes = pd.to_numeric(aligned_labels[columns.label], errors="raise").to_numpy()
    if not np.isin(outcomes, [0, 1]).all():
        raise ValueError("Labels must be binary 0/1.")
    output[columns.label] = outcomes.astype(np.int64)
    output.attrs["row_order_sha256"] = row_order_sha256(full_ids)
    return output


def binary_metrics(labels: Sequence[int], probabilities: Sequence[float], *, ece_bins: int = 10) -> dict[str, float]:
    labels_array = np.asarray(labels)
    probability_array = np.asarray(probabilities, dtype=float)
    if labels_array.ndim != 1 or probability_array.ndim != 1 or len(labels_array) != len(probability_array) or not len(labels_array):
        raise ValueError("Labels and probabilities must be equal non-empty vectors.")
    if not np.isin(labels_array, [0, 1]).all():
        raise ValueError("Labels must be binary 0/1.")
    labels_array = labels_array.astype(np.int64)
    if np.unique(labels_array).size != 2:
        raise ValueError("AUC requires both positive and negative labels.")
    probability_array = _probabilities(pd.Series(probability_array), "probabilities")
    if ece_bins < 1:
        raise ValueError("ece_bins must be positive.")
    brier = float(np.mean(np.square(probability_array - labels_array)))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, ece_bins + 1)
    for index in range(ece_bins):
        mask = ((probability_array >= edges[index]) &
                (probability_array <= edges[index + 1] if index == ece_bins - 1 else probability_array < edges[index + 1]))
        if mask.any():
            ece += float(mask.mean()) * abs(float(labels_array[mask].mean()) - float(probability_array[mask].mean()))
    return {
        "auc": float(roc_auc_score(labels_array, probability_array)),
        "acc": float(np.mean((probability_array >= 0.5) == labels_array)),
        "rmse": float(math.sqrt(brier)), "brier": brier, "ece": float(ece),
    }


def _metrics(frame: pd.DataFrame, columns: EvaluationColumns) -> dict[str, dict[str, float]]:
    labels = frame[columns.label].to_numpy()
    return {name: binary_metrics(labels, frame[f"prob_{name}"].to_numpy()) for name in VARIANT_NAMES}


def _envelope(metrics: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    return {
        "delta_auc": float(metrics["full"]["auc"] - max(metrics[name]["auc"] for name in CONTROL_NAMES)),
        "brier_regression": float(metrics["full"]["brier"] - min(metrics[name]["brier"] for name in CONTROL_NAMES)),
        "delta_auc_vs_control": {name: float(metrics["full"]["auc"] - metrics[name]["auc"]) for name in CONTROL_NAMES},
        "brier_delta_vs_control": {name: float(metrics["full"]["brier"] - metrics[name]["brier"]) for name in CONTROL_NAMES},
    }


def _slice(frame: pd.DataFrame, name: str, gating: bool, columns: EvaluationColumns) -> dict[str, Any]:
    labels = frame[columns.label].to_numpy()
    base = {"name": name, "gating": gating, "row_count": int(len(frame)),
            "student_count": int(frame[columns.student_id].nunique()),
            "item_count": int(frame[columns.item_id].nunique()),
            "q_pair_count": int(frame[columns.q_pair_id].nunique()),
            "positive_count": int((labels == 1).sum()), "negative_count": int((labels == 0).sum())}
    if not len(frame) or np.unique(labels).size != 2:
        reason = "empty_slice" if not len(frame) else "auc_requires_both_positive_and_negative_labels"
        if gating:
            raise ValueError(f"{name} cannot be evaluated: {reason}.")
        return {**base, "identified": False, "reason": reason}
    metrics = _metrics(frame, columns)
    return {**base, "identified": True, "metrics": metrics, "envelope": _envelope(metrics)}


def _top5(frame: pd.DataFrame, column: str) -> list[str]:
    counts = frame[column].astype(str).value_counts(sort=False).rename_axis("id").reset_index(name="count")
    return counts.sort_values(["count", "id"], ascending=[False, True], kind="mergesort")["id"].head(5).tolist()


def conservative_envelope_cluster_bootstrap(
    frame: pd.DataFrame, *, cluster_column: str, gating: bool,
    columns: EvaluationColumns = DEFAULT_COLUMNS,
    replicates: int = BOOTSTRAP_REPLICATES, seed: int = EVALUATION_SEED,
) -> dict[str, Any]:
    if replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("Requirement-surface evaluation fixes 2000 bootstraps.")
    if seed != EVALUATION_SEED:
        raise ValueError("Requirement-surface evaluation fixes bootstrap seed=2024.")
    clusters, inverse = np.unique(frame[cluster_column].astype(str), return_inverse=True)
    if len(clusters) < 2:
        raise ValueError(f"{cluster_column} bootstrap requires at least two clusters.")
    labels = frame[columns.label].to_numpy(dtype=np.int64)
    if np.unique(labels).size != 2:
        raise ValueError("Bootstrap AUC requires both positive and negative labels.")
    probabilities = {name: frame[f"prob_{name}"].to_numpy(dtype=float) for name in VARIANT_NAMES}
    observed = _envelope(_metrics(frame, columns))
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    controls = {name: [] for name in CONTROL_NAMES}
    envelope: list[float] = []
    invalid: list[int] = []
    for replicate in range(replicates):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        digest.update(np.asarray([replicate], dtype="<i8").tobytes())
        digest.update(np.asarray(sampled, dtype="<i8").tobytes())
        weights = np.bincount(sampled, minlength=len(clusters))[inverse].astype(float)
        if weights[labels == 0].sum() == 0 or weights[labels == 1].sum() == 0:
            invalid.append(replicate)
            continue
        full_auc = float(roc_auc_score(labels, probabilities["full"], sample_weight=weights))
        current = []
        for control in CONTROL_NAMES:
            delta = full_auc - float(roc_auc_score(labels, probabilities[control], sample_weight=weights))
            controls[control].append(delta)
            current.append(delta)
        envelope.append(min(current))
    result = {"identified": False, "gating": gating, "unit": cluster_column,
              "cluster_count": int(len(clusters)), "requested": replicates, "used": len(envelope),
              "invalid_single_class": len(invalid), "invalid_single_class_indices": invalid,
              "sample_index_sha256": digest.hexdigest(), "seed": seed}
    if len(envelope) < max(100, math.ceil(0.9 * replicates)):
        return {**result, "reason": "too_many_single_class_bootstrap_samples"}
    values = np.asarray(envelope)
    return {**result, "identified": True,
            "conservative_envelope": {"observed_delta_auc": observed["delta_auc"],
                "ci_low": float(np.quantile(values, .025)), "ci_high": float(np.quantile(values, .975)),
                "probability_positive": float(np.mean(values > 0))},
            "replicate_delta_auc_vs_control": controls,
            "replicate_envelope_delta_auc": envelope,
            "replicate_delta_sha256": _hash_values(f"{i}:{v:.17g}" for i, v in enumerate(envelope))}


def evaluate_requirement_surface(
    aligned: pd.DataFrame, *, dataset: str, columns: EvaluationColumns = DEFAULT_COLUMNS,
    expected_row_order_sha256: str | None = None,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES, seed: int = EVALUATION_SEED,
) -> dict[str, Any]:
    required = {columns.row_id, *columns.identities, columns.label, *(f"prob_{name}" for name in VARIANT_NAMES)}
    if required - set(aligned.columns):
        raise ValueError(f"Aligned table is missing columns: {sorted(required - set(aligned.columns))}")
    frame = _normalize_metadata(aligned.copy().reset_index(drop=True), columns)
    ids = _unique_ids(frame, "aligned evaluation table", columns.row_id)
    order_hash = row_order_sha256(ids)
    if expected_row_order_sha256 is not None and expected_row_order_sha256 != order_hash:
        raise ValueError("Outcome-free row-order SHA-256 is unexpected.")
    if bootstrap_replicates != BOOTSTRAP_REPLICATES or seed != EVALUATION_SEED:
        raise ValueError("Requirement-surface evaluation fixes 2000 bootstraps and seed=2024.")
    eligible = frame[columns.eligible].to_numpy(dtype=bool)
    q_count = frame[columns.q_count].to_numpy(dtype=int)
    q2 = frame.loc[eligible & (q_count == 2)].reset_index(drop=True)
    pooled = frame.loc[eligible].reset_index(drop=True)
    top_items, top_pairs = _top5(q2, columns.item_id), _top5(q2, columns.q_pair_id)
    no_items = q2.loc[~q2[columns.item_id].astype(str).isin(top_items)].reset_index(drop=True)
    no_pairs = q2.loc[~q2[columns.q_pair_id].astype(str).isin(top_pairs)].reset_index(drop=True)
    slices = {
        "q2": _slice(q2, "q2", True, columns),
        "eligible_pooled": _slice(pooled, "eligible_pooled", True, columns),
        "q3": _slice(frame.loc[eligible & (q_count == 3)], "q3", False, columns),
        "q4": _slice(frame.loc[eligible & (q_count == 4)], "q4", False, columns),
        "q2_without_top5_items": _slice(no_items, "q2_without_top5_items", True, columns),
        "q2_without_top5_q_pairs": _slice(no_pairs, "q2_without_top5_q_pairs", True, columns),
    }
    return {"schema_version": 1, "dataset": dataset, "row_count": int(len(frame)),
            "eligible_row_count": int(eligible.sum()), "q2_row_count": int(len(q2)),
            "row_order_sha256": order_hash, "variant_order": list(VARIANT_NAMES),
            "metric_order": list(METRIC_NAMES), "slices": slices,
            "top5_item_removal": {"removed_ids": top_items, "removed_row_count": int(len(q2) - len(no_items))},
            "top5_q_pair_removal": {"removed_ids": top_pairs, "removed_row_count": int(len(q2) - len(no_pairs))},
            "student_cluster_bootstrap": conservative_envelope_cluster_bootstrap(
                q2, cluster_column=columns.student_id, gating=True, columns=columns),
            "q_pair_cluster_bootstrap_sensitivity": conservative_envelope_cluster_bootstrap(
                q2, cluster_column=columns.q_pair_id, gating=False, columns=columns),
            "bootstrap_is_multi_seed": False}


def evaluate_prediction_manifests(*, predictions: Mapping[str, pd.DataFrame], labels: pd.DataFrame,
                                  dataset: str, columns: EvaluationColumns = DEFAULT_COLUMNS,
                                  expected_row_order_sha256: str | None = None) -> dict[str, Any]:
    return evaluate_requirement_surface(
        align_prediction_manifests(predictions=predictions, labels=labels, columns=columns),
        dataset=dataset, columns=columns, expected_row_order_sha256=expected_row_order_sha256)


def aggregate_stage1_gate(dataset_summaries: Mapping[str, Mapping[str, Any]], *,
                          surface_noncollapse: Mapping[str, bool] | bool) -> dict[str, Any]:
    missing = set(REQUIRED_DATASETS) - set(dataset_summaries)
    if missing:
        raise ValueError(f"Stage-1 gate requires dataset summaries for {sorted(missing)}.")
    selected = {name: dataset_summaries[name] for name in REQUIRED_DATASETS}
    if isinstance(surface_noncollapse, Mapping):
        if set(REQUIRED_DATASETS) - set(surface_noncollapse):
            raise ValueError("Surface noncollapse decisions are incomplete.")
        noncollapse = {name: bool(surface_noncollapse[name]) for name in REQUIRED_DATASETS}
    else:
        noncollapse = {name: bool(surface_noncollapse) for name in REQUIRED_DATASETS}
    get = lambda summary, key, field: float(summary["slices"][key]["envelope"][field])
    q2 = {name: get(summary, "q2", "delta_auc") for name, summary in selected.items()}
    brier = {name: get(summary, "q2", "brier_regression") for name, summary in selected.items()}
    pooled = {name: get(summary, "eligible_pooled", "delta_auc") for name, summary in selected.items()}
    top_item = {name: get(summary, "q2_without_top5_items", "delta_auc") for name, summary in selected.items()}
    top_pair = {name: get(summary, "q2_without_top5_q_pairs", "delta_auc") for name, summary in selected.items()}
    ci = {name: (float(summary["student_cluster_bootstrap"]["conservative_envelope"]["ci_low"])
                 if summary["student_cluster_bootstrap"].get("identified") else float("-inf"))
          for name, summary in selected.items()}
    checks = {
        "both_q2_delta_auc_at_least_0.005": all(v >= .005 for v in q2.values()),
        "one_q2_delta_auc_at_least_0.010": any(v >= .010 for v in q2.values()),
        "one_student_cluster_ci_low_above_zero": any(v > 0 for v in ci.values()),
        "both_q2_brier_regression_at_most_0.001": all(v <= .001 for v in brier.values()),
        "both_eligible_pooled_delta_nonnegative": all(v >= 0 for v in pooled.values()),
        "both_top5_item_removed_q2_delta_positive": all(v > 0 for v in top_item.values()),
        "both_top5_q_pair_removed_q2_delta_positive": all(v > 0 for v in top_pair.values()),
        "both_full_surfaces_noncollapsed": all(noncollapse.values()),
    }
    return {"schema_version": 1, "required_datasets": list(REQUIRED_DATASETS), "checks": checks,
            "passed": all(checks.values()), "q2_delta_auc": q2, "q2_brier_regression": brier,
            "eligible_pooled_delta_auc": pooled, "top5_item_removed_q2_delta_auc": top_item,
            "top5_q_pair_removed_q2_delta_auc": top_pair, "student_cluster_ci_low": ci,
            "surface_noncollapse": noncollapse, "bootstrap_is_multi_seed": False}
