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
DONOR_REPLICATES = 200
CONTROL_NAMES: tuple[str, ...] = (
    "strong_opms",
    "perm_pair",
    "late_fusion",
)
MODEL_NAMES: tuple[str, ...] = ("real_pair", *CONTROL_NAMES)
PROBABILITY_COLUMNS: Mapping[str, str] = {
    name: f"prob_{name}" for name in MODEL_NAMES
}
REQUIRED_COLUMNS: tuple[str, ...] = (
    "source_row_id",
    "stu_id",
    "exer_id",
    "optimizer_item_frequency",
    "coverage_bucket",
    "label",
    *PROBABILITY_COLUMNS.values(),
)


@dataclass(frozen=True)
class EvaluationResult:
    summary: dict[str, Any]
    leave_one_item_out: pd.DataFrame


def _sha256_values(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def row_order_sha256(source_row_ids: Sequence[object]) -> str:
    return _sha256_values(str(value) for value in source_row_ids)


def _unique_source_ids(frame: pd.DataFrame, *, name: str) -> pd.Series:
    if "source_row_id" not in frame:
        raise ValueError(f"{name} is missing source_row_id.")
    identifiers = frame["source_row_id"].astype(str)
    if (identifiers.str.len() == 0).any():
        raise ValueError(f"{name} contains an empty source_row_id.")
    duplicates = identifiers[identifiers.duplicated(keep=False)]
    if not duplicates.empty:
        examples = sorted(set(duplicates))[:5]
        raise ValueError(
            f"{name} contains duplicate source_row_id values: {examples}"
        )
    return identifiers


def _probability_column(frame: pd.DataFrame, model_name: str) -> str:
    preferred = PROBABILITY_COLUMNS[model_name]
    if preferred in frame:
        return preferred
    if "prob" in frame:
        return "prob"
    raise ValueError(
        f"{model_name} predictions require {preferred!r} or 'prob'."
    )


def align_prediction_frames(
    *,
    real_pair: pd.DataFrame,
    strong_opms: pd.DataFrame,
    perm_pair: pd.DataFrame,
    late_fusion: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Align all predictions and labels to unique RealPair source-row IDs."""
    frames = {
        "real_pair": real_pair,
        "strong_opms": strong_opms,
        "perm_pair": perm_pair,
        "late_fusion": late_fusion,
    }
    real_ids = _unique_source_ids(real_pair, name="real_pair")
    real_id_set = set(real_ids)
    output = real_pair.copy().reset_index(drop=True)
    output["source_row_id"] = real_ids.to_numpy()
    output = output.drop(
        columns=[
            column
            for column in (*PROBABILITY_COLUMNS.values(), "label")
            if column in output
        ]
    )

    for model_name, frame in frames.items():
        ids = _unique_source_ids(frame, name=model_name)
        if set(ids) != real_id_set:
            missing = sorted(real_id_set - set(ids))[:5]
            unexpected = sorted(set(ids) - real_id_set)[:5]
            raise ValueError(
                f"{model_name} row-ID set differs from RealPair: "
                f"missing={missing}, unexpected={unexpected}."
            )
        aligned = frame.assign(_source_row_id=ids.to_numpy()).set_index(
            "_source_row_id", drop=True
        ).reindex(real_ids.to_numpy())
        for identity in ("stu_id", "exer_id"):
            if identity in output and identity in aligned:
                left = output[identity].astype(str).to_numpy()
                right = aligned[identity].astype(str).to_numpy()
                if not np.array_equal(left, right):
                    mismatch = int(np.flatnonzero(left != right)[0])
                    raise ValueError(
                        f"{model_name} disagrees on {identity} at "
                        f"RealPair row {mismatch}."
                    )
        probability_column = _probability_column(frame, model_name)
        output[PROBABILITY_COLUMNS[model_name]] = aligned[
            probability_column
        ].to_numpy()

    label_ids = _unique_source_ids(labels, name="labels")
    if set(label_ids) != real_id_set:
        raise ValueError("Label row-ID set differs from RealPair predictions.")
    if "label" not in labels:
        raise ValueError("labels is missing label.")
    aligned_labels = labels.assign(
        _source_row_id=label_ids.to_numpy()
    ).set_index("_source_row_id", drop=True).reindex(real_ids.to_numpy())
    for identity in ("stu_id", "exer_id"):
        if identity in output and identity in aligned_labels:
            if not np.array_equal(
                output[identity].astype(str).to_numpy(),
                aligned_labels[identity].astype(str).to_numpy(),
            ):
                raise ValueError(f"Labels disagree with RealPair on {identity}.")
    output["label"] = aligned_labels["label"].to_numpy()
    output.attrs["real_row_order_sha256"] = row_order_sha256(real_ids)
    return output


def validate_prediction_table(
    frame: pd.DataFrame,
    *,
    expected_real_order_sha256: str | None = None,
) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    output = frame.copy().reset_index(drop=True)
    identifiers = _unique_source_ids(output, name="prediction table")
    output["source_row_id"] = identifiers.to_numpy()
    output["stu_id"] = output["stu_id"].astype(str)
    output["exer_id"] = output["exer_id"].astype(str)
    order_hash = row_order_sha256(identifiers)
    if (
        expected_real_order_sha256 is not None
        and order_hash != expected_real_order_sha256
    ):
        raise ValueError("RealPair prediction row-order SHA-256 is unexpected.")

    labels = pd.to_numeric(output["label"], errors="raise").to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary.")
    if np.unique(labels).size != 2:
        raise ValueError("The aligned evaluation table must contain both labels.")
    output["label"] = labels.astype(np.int64)
    for column in PROBABILITY_COLUMNS.values():
        probabilities = pd.to_numeric(
            output[column], errors="raise"
        ).to_numpy(dtype=np.float64)
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            raise ValueError(
                f"{column} must contain finite probabilities in [0, 1]."
            )
        output[column] = probabilities
    output.attrs["real_row_order_sha256"] = order_hash
    return output


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    brier = float(np.mean(np.square(probabilities - labels)))
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "brier": brier,
        "acc": float(np.mean((probabilities >= 0.5) == labels)),
        "rmse": float(math.sqrt(brier)),
    }


def _all_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    return {
        name: _metrics(labels, frame[column].to_numpy(dtype=np.float64))
        for name, column in PROBABILITY_COLUMNS.items()
    }


def _deltas(
    metrics: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        control: {
            "auc": float(
                metrics["real_pair"]["auc"] - metrics[control]["auc"]
            ),
            # Positive is a RealPair calibration regression.
            "brier": float(
                metrics["real_pair"]["brier"] - metrics[control]["brier"]
            ),
        }
        for control in CONTROL_NAMES
    }


def _student_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    students, inverse = np.unique(
        frame["stu_id"].astype(str).to_numpy(), return_inverse=True
    )
    if len(students) < 2:
        return {
            "identified": False,
            "reason": "fewer_than_two_student_clusters",
            "gating": True,
            "requested": replicates,
            "used": 0,
            "invalid_single_class": replicates,
        }
    labels = frame["label"].to_numpy(dtype=np.int64)
    probabilities = {
        name: frame[column].to_numpy(dtype=np.float64)
        for name, column in PROBABILITY_COLUMNS.items()
    }
    observed_deltas = _deltas(_all_metrics(frame))
    rng = np.random.default_rng(seed)
    sample_digest = hashlib.sha256()
    used_indices: list[int] = []
    invalid_indices: list[int] = []
    replicate_deltas: dict[str, list[float]] = {
        control: [] for control in CONTROL_NAMES
    }
    replicate_deltas["joint_min"] = []

    for replicate in range(replicates):
        sampled = rng.integers(0, len(students), size=len(students))
        sample_digest.update(np.asarray([replicate], dtype="<i8").tobytes())
        sample_digest.update(np.asarray(sampled, dtype="<i8").tobytes())
        cluster_counts = np.bincount(sampled, minlength=len(students))
        weights = cluster_counts[inverse]
        if weights[labels == 1].sum() == 0 or weights[labels == 0].sum() == 0:
            invalid_indices.append(replicate)
            continue
        used_indices.append(replicate)
        real_auc = float(
            roc_auc_score(
                labels, probabilities["real_pair"], sample_weight=weights
            )
        )
        current: list[float] = []
        for control in CONTROL_NAMES:
            control_auc = float(
                roc_auc_score(
                    labels, probabilities[control], sample_weight=weights
                )
            )
            delta = real_auc - control_auc
            replicate_deltas[control].append(delta)
            current.append(delta)
        replicate_deltas["joint_min"].append(min(current))

    minimum_used = max(100, int(math.ceil(0.90 * replicates)))
    result: dict[str, Any] = {
        "identified": False,
        "gating": True,
        "unit": "student",
        "student_count": int(len(students)),
        "student_order_sha256": _sha256_values(students),
        "requested": int(replicates),
        "used": len(used_indices),
        "invalid_single_class": len(invalid_indices),
        "invalid_single_class_indices": invalid_indices,
        "invalid_single_class_indices_sha256": _sha256_values(invalid_indices),
        "used_replicate_indices_sha256": _sha256_values(used_indices),
        "sample_index_sha256": sample_digest.hexdigest(),
        "seed": int(seed),
        "controls_order": list(CONTROL_NAMES),
    }
    if len(used_indices) < minimum_used:
        result["reason"] = "too_many_single_class_bootstrap_samples"
        return result

    controls: dict[str, Any] = {}
    for control in CONTROL_NAMES:
        values = np.asarray(replicate_deltas[control], dtype=np.float64)
        controls[control] = {
            "observed_delta_auc": observed_deltas[control]["auc"],
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "probability_positive": float(np.mean(values > 0.0)),
        }
    joint_values = np.asarray(
        replicate_deltas["joint_min"], dtype=np.float64
    )
    joint = {
        "observed_delta_auc": min(
            observed_deltas[control]["auc"] for control in CONTROL_NAMES
        ),
        "ci_low": float(np.quantile(joint_values, 0.025)),
        "ci_high": float(np.quantile(joint_values, 0.975)),
        "probability_positive": float(np.mean(joint_values > 0.0)),
    }
    result.update(
        {
            "identified": True,
            "controls": controls,
            "joint_min_delta": joint,
            "joint_min_delta_ci": [joint["ci_low"], joint["ci_high"]],
            "replicate_deltas": replicate_deltas,
            "replicate_deltas_sha256": _sha256_values(
                f"{control}:{index}:{value:.17g}"
                for control in (*CONTROL_NAMES, "joint_min")
                for index, value in enumerate(replicate_deltas[control])
            ),
        }
    )
    return result


def evaluate_target_local_pairing(
    predictions_with_label: pd.DataFrame,
    *,
    expected_real_order_sha256: str | None = None,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = EVALUATION_SEED,
) -> EvaluationResult:
    if seed != EVALUATION_SEED:
        raise ValueError("Target-local pairing evaluation fixes seed=2024.")
    if bootstrap_replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("Target-local pairing evaluation fixes 2000 bootstraps.")
    frame = validate_prediction_table(
        predictions_with_label,
        expected_real_order_sha256=expected_real_order_sha256,
    )
    metrics = _all_metrics(frame)
    deltas = _deltas(metrics)
    _validate_item_frequency(frame)
    bootstrap = _student_cluster_bootstrap(
        frame, replicates=bootstrap_replicates, seed=seed
    )
    top_frequency = _top_frequency_item_removal(frame)
    item_bootstrap = _target_item_cluster_bootstrap(
        frame, replicates=bootstrap_replicates, seed=seed
    )
    leave_summary, leave_table = _leave_one_target_item_out(
        frame, full_deltas=deltas
    )
    coverage_slices = _coverage_descriptive_slices(frame)
    summary = {
        "schema_version": 1,
        "row_count": int(len(frame)),
        "student_count": int(frame["stu_id"].nunique()),
        "item_count": int(frame["exer_id"].nunique()),
        "real_row_order_sha256": frame.attrs["real_row_order_sha256"],
        "controls_order": list(CONTROL_NAMES),
        "metrics": metrics,
        "deltas_vs_control": deltas,
        "dataset_effect": min(
            deltas[control]["auc"] for control in CONTROL_NAMES
        ),
        "joint_min_delta_ci": (
            bootstrap["joint_min_delta_ci"]
            if bootstrap["identified"]
            else None
        ),
        "max_brier_regression": max(
            0.0,
            max(deltas[control]["brier"] for control in CONTROL_NAMES),
        ),
        "student_cluster_bootstrap": bootstrap,
        "coverage_descriptive_slices": coverage_slices,
        "top1pct_train_frequency_item_removal": top_frequency,
        "item_cluster_bootstrap": item_bootstrap,
        "leave_one_item_out": leave_summary,
    }
    return EvaluationResult(
        summary=summary,
        leave_one_item_out=leave_table,
    )


_COVERAGE_ALIASES: Mapping[str, str] = {
    "exact_zero": "exact_zero",
    "zero": "exact_zero",
    "none_seen": "exact_zero",
    "partial": "partial",
    "low_coverage": "partial",
    "partial_seen": "partial",
    "full": "full",
    "observed": "full",
    "all_seen": "full",
}


def _validate_item_frequency(frame: pd.DataFrame) -> None:
    frequency = pd.to_numeric(
        frame["optimizer_item_frequency"], errors="raise"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(frequency).all() or (frequency < 0).any():
        raise ValueError(
            "optimizer_item_frequency must be finite and nonnegative."
        )
    frame["optimizer_item_frequency"] = frequency
    spread = frame.groupby("exer_id")["optimizer_item_frequency"].agg(
        lambda values: float(values.max() - values.min())
    )
    if (spread > 1e-12).any():
        raise ValueError(
            "optimizer_item_frequency must be constant within a target item."
        )


def _coverage_descriptive_slices(frame: pd.DataFrame) -> dict[str, Any]:
    raw = frame["coverage_bucket"].astype(str)
    normalized = raw.map(_COVERAGE_ALIASES)
    if normalized.isna().any():
        unknown = sorted(set(raw[normalized.isna()]))
        raise ValueError(f"Unknown coverage_bucket values: {unknown}")
    output: dict[str, Any] = {
        "gating": False,
        "selection_use": "descriptive_only_not_used_for_model_selection",
        "bucket_order": ["exact_zero", "partial", "full"],
        "buckets": {},
    }
    for bucket in output["bucket_order"]:
        subset = frame.loc[normalized == bucket]
        labels = subset["label"].to_numpy(dtype=np.int64)
        entry: dict[str, Any] = {
            "identified": bool(
                len(subset) > 0 and np.unique(labels).size == 2
            ),
            "rows": int(len(subset)),
            "students": int(subset["stu_id"].nunique()),
            "label_0": int(np.sum(labels == 0)),
            "label_1": int(np.sum(labels == 1)),
        }
        if entry["identified"]:
            full_metrics = _all_metrics(subset)
            entry["metrics"] = {
                model: {
                    "auc": values["auc"],
                    "brier": values["brier"],
                }
                for model, values in full_metrics.items()
            }
            entry["deltas_vs_control"] = _deltas(full_metrics)
        else:
            entry["metrics"] = None
            entry["deltas_vs_control"] = None
            entry["reason"] = (
                "empty_or_single_class_descriptive_slice"
            )
        output["buckets"][bucket] = entry
    return output


def _top_frequency_item_removal(frame: pd.DataFrame) -> dict[str, Any]:
    items = (
        frame.loc[:, ["exer_id", "optimizer_item_frequency"]]
        .drop_duplicates("exer_id")
        .sort_values(
            ["optimizer_item_frequency", "exer_id"],
            ascending=[False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    top_count = max(1, int(math.ceil(0.01 * len(items))))
    removed_items = items.iloc[:top_count]["exer_id"].astype(str).tolist()
    retained = frame.loc[~frame["exer_id"].isin(removed_items)]
    output: dict[str, Any] = {
        "identified": False,
        "gating": False,
        "definition": (
            "remove ceil(1% of unique validation target items ranked by "
            "optimizer-train-only item frequency; item ID breaks ties"
        ),
        "target_item_count": int(len(items)),
        "removed_item_count": int(top_count),
        "removed_items": removed_items,
        "removed_items_sha256": _sha256_values(removed_items),
        "removed_row_count": int(len(frame) - len(retained)),
        "remaining_row_count": int(len(retained)),
    }
    if retained.empty or retained["label"].nunique() < 2:
        output["reason"] = "remaining_rows_do_not_contain_both_labels"
        return output
    metrics = _all_metrics(retained)
    deltas = _deltas(metrics)
    output.update(
        {
            "identified": True,
            "metrics": metrics,
            "deltas_vs_control": deltas,
            "dataset_effect": min(
                deltas[control]["auc"] for control in CONTROL_NAMES
            ),
            "max_brier_regression": max(
                0.0,
                max(deltas[control]["brier"] for control in CONTROL_NAMES),
            ),
        }
    )
    return output


def _target_item_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    item_count = int(frame["exer_id"].nunique())
    if item_count < 20:
        return {
            "identified": False,
            "gating": False,
            "unit": "validation_target_item",
            "item_count": item_count,
            "minimum_items": 20,
            "requested": int(replicates),
            "used": 0,
            "reason": "fewer_than_20_validation_target_items",
        }
    proxy = frame.copy()
    proxy["stu_id"] = frame["exer_id"].astype(str)
    result = _student_cluster_bootstrap(
        proxy, replicates=replicates, seed=seed
    )
    result["gating"] = False
    result["unit"] = "validation_target_item"
    result["item_count"] = result.pop("student_count")
    result["item_order_sha256"] = result.pop("student_order_sha256")
    return result


def _float_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"identified": False, "reason": "no_values"}
    return {
        "identified": True,
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "q025": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "q975": float(np.quantile(values, 0.975)),
        "max": float(values.max()),
    }


def _leave_one_target_item_out(
    frame: pd.DataFrame,
    *,
    full_deltas: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for item in sorted(frame["exer_id"].unique()):
        removed = frame["exer_id"] == item
        retained = frame.loc[~removed]
        row: dict[str, Any] = {
            "exer_id": str(item),
            "optimizer_item_frequency": float(
                frame.loc[removed, "optimizer_item_frequency"].iloc[0]
            ),
            "rows_removed": int(removed.sum()),
            "remaining_rows": int(len(retained)),
            "remaining_positive": int(retained["label"].sum()),
            "remaining_negative": int(
                len(retained) - retained["label"].sum()
            ),
            "identified": bool(
                len(retained) > 0 and retained["label"].nunique() == 2
            ),
        }
        if row["identified"]:
            metrics = _all_metrics(retained)
            deltas = _deltas(metrics)
            row["real_auc_without_item"] = metrics["real_pair"]["auc"]
            for control in CONTROL_NAMES:
                row[f"delta_{control}_without_item"] = deltas[control]["auc"]
                row[f"influence_{control}"] = (
                    deltas[control]["auc"]
                    - full_deltas[control]["auc"]
                )
        rows.append(row)
    table = pd.DataFrame(rows)
    identified = table.loc[table["identified"]]
    summary: dict[str, Any] = {
        "identified": bool(len(identified) > 0),
        "gating": False,
        "definition": (
            "remove one validation target item; evaluate only if remaining "
            "labels contain both classes"
        ),
        "target_item_count": int(len(table)),
        "identified_item_count": int(len(identified)),
        "not_identified_item_count": int(len(table) - len(identified)),
        "per_control": {},
    }
    for control in CONTROL_NAMES:
        column = f"influence_{control}"
        if identified.empty or column not in identified:
            summary["per_control"][control] = {
                "identified": False,
                "reason": "no_identified_leave-one-item-out_sample",
            }
            continue
        values = identified[column].to_numpy(dtype=np.float64)
        position = int(np.argmax(np.abs(values)))
        summary["per_control"][control] = {
            **_float_summary(values),
            "max_absolute_influence": float(np.max(np.abs(values))),
            "q95_absolute_influence": float(
                np.quantile(np.abs(values), 0.95)
            ),
            "most_influential_item": str(
                identified.iloc[position]["exer_id"]
            ),
            "most_influential_signed_change": float(values[position]),
        }
    summary["row_table_sha256"] = _sha256_values(
        "\x1e".join(map(str, row))
        for row in table.fillna("NA").itertuples(index=False, name=None)
    )
    return summary, table


def summarize_donor_sensitivity(
    *,
    labels: Sequence[int],
    real_probabilities: Sequence[float],
    donor_probabilities: np.ndarray,
    mapping_hashes: Sequence[str],
    source_row_ids: Sequence[object],
    expected_row_order_sha256: str | None = None,
    seed: int = EVALUATION_SEED,
) -> dict[str, Any]:
    """Describe 200 fixed-model donor mappings; never compute a p-value."""
    if seed != EVALUATION_SEED:
        raise ValueError("Donor sensitivity fixes seed=2024.")
    label_array = np.asarray(labels, dtype=np.int64)
    row_ids = np.asarray(source_row_ids).astype(str)
    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("Donor sensitivity source_row_id values must be unique.")
    if len(label_array) != len(row_ids):
        raise ValueError("Labels and source_row_ids have different lengths.")
    real_probability_array = np.asarray(
        real_probabilities, dtype=np.float64
    )
    if real_probability_array.shape != (len(row_ids),):
        raise ValueError(
            "real_probabilities must have one probability per aligned row."
        )
    if (
        not np.isfinite(real_probability_array).all()
        or (real_probability_array < 0).any()
        or (real_probability_array > 1).any()
    ):
        raise ValueError(
            "Real probabilities must be finite and in [0, 1]."
        )
    if (
        not np.isin(label_array, [0, 1]).all()
        or np.unique(label_array).size != 2
    ):
        raise ValueError(
            "Donor sensitivity labels must contain both binary classes."
        )
    probabilities = np.asarray(donor_probabilities, dtype=np.float64)
    expected_shape = (DONOR_REPLICATES, len(row_ids))
    if probabilities.shape != expected_shape:
        raise ValueError(
            f"donor_probabilities must have shape {expected_shape}, "
            f"got {probabilities.shape}."
        )
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError(
            "Donor probabilities must be finite and in [0, 1]."
        )
    if len(mapping_hashes) != DONOR_REPLICATES or any(
        not str(value) for value in mapping_hashes
    ):
        raise ValueError("Exactly 200 nonempty mapping hashes are required.")
    order_hash = row_order_sha256(row_ids)
    if (
        expected_row_order_sha256 is not None
        and order_hash != expected_row_order_sha256
    ):
        raise ValueError("Donor sensitivity row-order SHA-256 is unexpected.")

    reference_real_auc = float(
        roc_auc_score(label_array, real_probability_array)
    )
    reference_real_brier = float(
        np.mean(np.square(real_probability_array - label_array))
    )
    auc_values = np.asarray(
        [roc_auc_score(label_array, row) for row in probabilities],
        dtype=np.float64,
    )
    brier_values = np.mean(
        np.square(probabilities - label_array[None, :]), axis=1
    )
    auc_change_values = auc_values - reference_real_auc
    brier_change_values = brier_values - reference_real_brier
    probability_hashes = [
        hashlib.sha256(np.asarray(row, dtype="<f8").tobytes()).hexdigest()
        for row in probabilities
    ]
    unique_mapping_hashes = sorted(set(map(str, mapping_hashes)))
    return {
        "schema_version": 1,
        "analysis": "fixed_model_donor_sensitivity",
        "inferential_test": "not_computed",
        "p_value_computed": False,
        "seed": seed,
        "row_count": int(len(row_ids)),
        "replicates": DONOR_REPLICATES,
        "row_order_sha256": order_hash,
        "reference_real_auc": reference_real_auc,
        "reference_real_brier": reference_real_brier,
        "reference_real_probability_sha256": hashlib.sha256(
            np.asarray(real_probability_array, dtype="<f8").tobytes()
        ).hexdigest(),
        "change_direction": "donor_minus_real",
        "mapping_hashes": list(map(str, mapping_hashes)),
        "mapping_hashes_sha256": _sha256_values(mapping_hashes),
        "unique_mapping_hash_count": len(unique_mapping_hashes),
        "unique_mapping_hashes": unique_mapping_hashes,
        "probability_matrix_sha256": hashlib.sha256(
            np.asarray(probabilities, dtype="<f8").tobytes()
        ).hexdigest(),
        "probability_sha256": probability_hashes,
        "probability_hashes_sha256": _sha256_values(probability_hashes),
        "unique_probability_hash_count": len(set(probability_hashes)),
        "auc": {
            **_float_summary(auc_values),
            "values": auc_values.tolist(),
        },
        "brier": {
            **_float_summary(brier_values),
            "values": brier_values.tolist(),
        },
        "donor_minus_real_auc": {
            **_float_summary(auc_change_values),
            "values": auc_change_values.tolist(),
            "values_sha256": _sha256_values(
                f"{value:.17g}" for value in auc_change_values
            ),
        },
        "donor_minus_real_brier": {
            **_float_summary(brier_change_values),
            "values": brier_change_values.tolist(),
            "values_sha256": _sha256_values(
                f"{value:.17g}" for value in brier_change_values
            ),
        },
    }
